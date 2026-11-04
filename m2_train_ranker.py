#!/usr/bin/env python
"""MODULE 2 - Subgraph Ranker (chạy độc lập).

Đầu vào : <data-dir>/<task>/train_full.jsonl  (metapath + rel_score của Module 1; mặc định data/authors/datasets)
          <data-dir>/<task>/test_truth.jsonl  (để đánh giá NDCG/Recall)
Đầu ra  : outputs/module2/<task>/<tag>/
            test_ranked_top5.jsonl, test_ranked_all.jsonl, metrics.json      <- bản MỚI NHẤT (Module 3 đọc các file này)
            *_<timestamp>.{json,jsonl}                                       <- bản lưu theo thời điểm chạy (không bị ghi đè)
            model_<timestamp>/                                               <- TRỌNG SỐ đã huấn luyện (xgboost luôn lưu; ce_* khi --save-model)
            latest.json                                                      <- trỏ tới lần chạy gần nhất (timestamp, thư mục trọng số)
          + log: logs/m2_train_ranker_<timestamp>.log

Ranker:  xgboost | ce_rmse | ce_ranknet | ce_listnet      (ce_* = Cross-Encoder roberta-base, loss pointwise/pairwise/listwise)

--cv-folds K : cross-validation K-fold trên TẤT CẢ cặp (train+test) -> xếp hạng out-of-fold cv_ranked_{top5,all}.jsonl; dùng cho
               `m3_zero_shot.py --eval-split full` (đánh giá trên bộ câu hỏi lớn hơn nhiều mà ranker không nhìn thấy cặp đang xếp hạng).

Ví dụ:
  python m2_train_ranker.py --dry-run                                  # kiểm tra nhanh, model nhỏ, không tải trọng số
  python m2_train_ranker.py --tasks comagc --rankers xgboost           # ~1-2 phút CPU
  python m2_train_ranker.py --rankers ce_rmse ce_ranknet ce_listnet --epochs 10 --save-model   # cần GPU (xem README)
  python m2_train_ranker.py --rankers xgboost --cv-folds 5              # xếp hạng out-of-fold cho --eval-split full
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from lib import data as D
from lib import metrics as M
from lib import paths, runlog
from lib.rankers import CrossEncoderRanker, XGBSubgraphRanker, evaluate_scores, split_dev

RANKERS = ["xgboost", "ce_rmse", "ce_ranknet", "ce_listnet"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=paths.ALL_TASKS, choices=paths.ALL_TASKS)
    ap.add_argument("--rankers", nargs="+", default=["xgboost"], choices=RANKERS)
    ap.add_argument("--tag-suffix", default="", help="thêm hậu tố vào tên thư mục kết quả (vd '_nl')")
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--include-node-labels", dest="inc", action="store_true", default=True,
                    help="ghép node-type vào văn bản path (bài báo dùng; mặc định BẬT)")
    ap.add_argument("--no-node-labels", dest="inc", action="store_false")
    ap.add_argument("--neg-num", type=int, default=5, help="số path/cặp dùng để huấn luyện (tác giả: 5)")
    # cross-encoder
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-groups", type=int, default=8, help="số cặp/bước (tác giả: accumulate 8 cặp)")
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--target-transform", choices=["none", "signed_log"], default="none",
                    help="none = rel_score thô như tác giả; signed_log = nén thang đo (cải tiến tuỳ chọn)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (GPU)")
    ap.add_argument("--dev-frac", type=float, default=0.1, help="tỉ lệ train dùng làm dev để chọn epoch")
    ap.add_argument("--select-on-test", action="store_true",
                    help="(KHÔNG khuyến nghị) chọn epoch theo test như code tác giả - chỉ để so sánh")
    ap.add_argument("--save-model", action="store_true", help="lưu trọng số cross-encoder vào <tag>/model_<timestamp>/ (~500MB mỗi model); XGBoost luôn được lưu")
    # xgboost
    ap.add_argument("--xgb-trees", type=int, default=100)
    ap.add_argument("--ngram-epochs", type=int, default=10)
    ap.add_argument("--ngram-batch", type=int, default=256)
    ap.add_argument("--ngram-lr", type=float, default=0.01)
    # chung
    ap.add_argument("--cv-folds", type=int, default=0, help="K>1: cross-validation trên tất cả cặp, ghi cv_ranked_*.jsonl (thay cho đánh giá train/test)")
    ap.add_argument("--data-dir", default=None, help="dùng dữ liệu Module 1 tự chấm (m1_score_metapaths.py) thay cho dữ liệu tác giả")
    ap.add_argument("--out", default=None, help="thư mục gốc kết quả (mặc định outputs/, hoặc outputs/data_<tên> khi có --data-dir)")
    ap.add_argument("--dry-run", action="store_true", help="tập con nhỏ + model nhỏ ngẫu nhiên; ghi vào outputs/_dryrun")
    ap.add_argument("--benchmark-steps", type=int, default=0,
                    help="đo tốc độ 1 bước train của roberta-base thật (khởi tạo ngẫu nhiên, không tải trọng số) rồi thoát")
    return ap.parse_args()


def benchmark(args):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_labels = 1
    model = AutoModelForSequenceClassification.from_config(cfg).to(dev)
    tok = AutoTokenizer.from_pretrained(args.model)
    items = D.load_train("comagc")[:args.batch_groups * 3]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    times = []
    for s in range(args.benchmark_steps):
        chunk = items[(s % 3) * args.batch_groups:(s % 3 + 1) * args.batch_groups]
        qs, ps = [], []
        for it in chunk:
            for m in it["metapaths"][:5]:
                qs.append(D.query_text(it))
                ps.append(D.path_text(m, True))
        enc = tok(qs, ps, padding=True, truncation=True, max_length=500, return_tensors="pt").to(dev)
        t0 = time.time()
        loss = model(**enc).logits.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"step {s}: {times[-1]:.2f}s  seq_len={enc['input_ids'].shape} device={dev}")
    print(f"median step (8 cặp x <=5 path) = {sorted(times)[len(times)//2]:.2f}s trên {dev}")


def build_and_fit(ranker, train, corpus_items, eval_items, args, log):
    """Huấn luyện 1 ranker; eval_items chỉ dùng khi --select-on-test. Trả (model, history)."""
    if ranker == "xgboost":
        model = XGBSubgraphRanker(args.inc, args.neg_num, args.seed, args.xgb_trees,
                                  1 if args.dry_run else args.ngram_epochs, args.ngram_batch, args.ngram_lr, log=log)
        model.fit(train, corpus_items=corpus_items)        # corpus n-gram: văn bản (không nhãn) như tác giả
        return model, []
    loss = ranker.split("_", 1)[1]
    model = CrossEncoderRanker(loss, args.model, args.inc, args.max_len, args.lr, 2 if args.dry_run else args.epochs,
                               args.batch_groups, args.neg_num, args.seed, args.target_transform, args.amp,
                               tiny=args.dry_run, log=log)
    if args.select_on_test and eval_items is not None:
        tr, dev = train, eval_items                          # chọn epoch bằng NDCG@5 (corrected) trên test - không khuyến nghị
        log("  !! --select-on-test: tái hiện hành vi của tác giả (rò rỉ test vào chọn epoch)")
    else:
        tr, dev = split_dev(train, args.dev_frac, args.seed)
    return model, model.fit(tr, dev)


def save_weights(model, ranker, out_dir, ts, extra, args):
    """Lưu trọng số vào <tag>/model_<ts>/ (xgboost luôn; cross-encoder khi --save-model). Trả đường dẫn hoặc None."""
    if ranker != "xgboost" and not args.save_model:
        return None
    wdir = out_dir / f"model_{ts}"
    model.save(wdir)
    (wdir / "train_meta.json").write_text(json.dumps({"timestamp": runlog.iso_now(), "ranker": ranker, **extra}, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    return wdir


def run_task_ranker(task: str, ranker: str, args, log=print):
    train = D.load_train(task)
    truth = D.load_test_truth(task)
    if args.dry_run:
        train, truth = train[:24], truth[:10]
    tag = ranker + args.tag_suffix
    out_dir = paths.module2_dir(task, tag, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = runlog.now_ts()
    log(f"\n=== [{task}] ranker={ranker} | train={len(train)} test={len(truth)} -> {out_dir}  (run {ts})")
    t0 = time.time()
    model, history = build_and_fit(ranker, train, train + truth, truth, args, log)
    train_sec = time.time() - t0
    wdir = save_weights(model, ranker, out_dir, ts, {"task": task, "train_pairs": len(train), "data_dir": args.data_dir}, args)
    if wdir:
        log(f"  trọng số -> {wdir}")

    scores = model.score_items(truth)
    results, top5, allr = evaluate_scores(truth, scores, seed=args.seed)
    refs = M.reference_orderings(truth, pool_size=5, n_random=200, seed=args.seed)
    meta = {"task": task, "ranker": tag, "timestamp": runlog.iso_now(), "run_id": ts, "log_file": str(runlog.log_path() or ""),
            "weights_dir": str(wdir) if wdir else None, "data_dir": args.data_dir,
            "seed": args.seed, "train_pairs": len(train), "test_pairs": len(truth),
            "train_seconds": round(train_sec, 1), "include_node_labels": args.inc, "dry_run": args.dry_run,
            "config": {k: v for k, v in vars(args).items() if k not in ("tasks", "rankers")},
            "results": results, "reference": refs, "history": history}
    mtxt = json.dumps(meta, indent=1, ensure_ascii=False)
    # bản có timestamp (không bao giờ bị ghi đè) + bản mới nhất (Module 3 / summarize đọc)
    D.write_jsonl(out_dir / f"test_ranked_top5_{ts}.jsonl", top5)
    D.write_jsonl(out_dir / f"test_ranked_all_{ts}.jsonl", allr)
    (out_dir / f"metrics_{ts}.json").write_text(mtxt, encoding="utf-8")
    for name in ("test_ranked_top5", "test_ranked_all"):
        shutil.copyfile(out_dir / f"{name}_{ts}.jsonl", out_dir / f"{name}.jsonl")
    (out_dir / "metrics.json").write_text(mtxt, encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps({"timestamp": ts, "weights_dir": str(wdir) if wdir else None,
                                                     "metrics": f"metrics_{ts}.json"}, indent=1), encoding="utf-8")
    p, c5, ca = results["paper_style_top5"], results["corrected_top5"], results["corrected_all"]
    log(f"  paper-style (tác giả) : NDCG@1/3/5={p['NDCG@1']}/{p['NDCG@3']}/{p['NDCG@5']}  Recall@5={p['Recall@5']}")
    log(f"  corrected (pool top5) : NDCG@5={c5['NDCG@5']}  Overlap@5={c5['Overlap@5']}  Top1Hit={c5['Top1Hit']}")
    log(f"  corrected (pool all)  : NDCG@5={ca['NDCG@5']}  Overlap@5={ca['Overlap@5']}  Top1Hit={ca['Top1Hit']}")
    log(f"  random reference      : paper NDCG@5={refs['random_expected']['paper']['NDCG@5']}  corrected(top5) NDCG@5={refs['random_expected']['corrected']['NDCG@5']}")
    return meta


def run_task_ranker_cv(task: str, ranker: str, args, log=print):
    """K-fold CV trên tất cả cặp (train+test). Mỗi cặp được xếp hạng bởi model KHÔNG huấn luyện trên chính nó."""
    items = D.load_full(task)
    if args.dry_run:
        items = items[:30]
    K = args.cv_folds
    tag = ranker + args.tag_suffix
    out_dir = paths.module2_dir(task, tag, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = runlog.now_ts()
    log(f"\n=== [{task}] ranker={ranker} | CV {K}-fold trên {len(items)} cặp -> {out_dir}  (run {ts})")
    idx = list(range(len(items)))
    random.Random(args.seed).shuffle(idx)
    folds = [idx[k::K] for k in range(K)]
    scores = [None] * len(items)
    t0 = time.time()
    for k, te_idx in enumerate(folds):
        te_set = set(te_idx)
        tr = [items[i] for i in range(len(items)) if i not in te_set]
        te = [items[i] for i in te_idx]
        log(f"  -- fold {k + 1}/{K}: train={len(tr)} held-out={len(te)}")
        model, _ = build_and_fit(ranker, tr, items, None, args, log)
        for i, s in zip(te_idx, model.score_items(te)):
            scores[i] = s
    results, top5, allr = evaluate_scores(items, scores, seed=args.seed)
    refs = M.reference_orderings(items, pool_size=5, n_random=50, seed=args.seed)
    meta = {"task": task, "ranker": tag, "mode": f"cv{K}", "timestamp": runlog.iso_now(), "run_id": ts, "log_file": str(runlog.log_path() or ""),
            "data_dir": args.data_dir, "seed": args.seed, "pairs": len(items), "seconds": round(time.time() - t0, 1),
            "config": {k_: v for k_, v in vars(args).items() if k_ not in ("tasks", "rankers")}, "results": results, "reference": refs}
    D.write_jsonl(out_dir / f"cv_ranked_top5_{ts}.jsonl", top5)
    D.write_jsonl(out_dir / f"cv_ranked_all_{ts}.jsonl", allr)
    (out_dir / f"cv_metrics_{ts}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ("cv_ranked_top5", "cv_ranked_all"):
        shutil.copyfile(out_dir / f"{name}_{ts}.jsonl", out_dir / f"{name}.jsonl")
    shutil.copyfile(out_dir / f"cv_metrics_{ts}.json", out_dir / "cv_metrics.json")
    c5, ca = results["corrected_top5"], results["corrected_all"]
    log(f"  [CV {K}-fold, {len(items)} cặp] corrected top5: NDCG@5={c5['NDCG@5']} Top1Hit={c5['Top1Hit']} | all: NDCG@5={ca['NDCG@5']} Top1Hit={ca['Top1Hit']} "
        f"| random: top5 NDCG@5={refs['random_expected']['corrected']['NDCG@5']}")
    return meta


def main():
    args = parse()
    runlog.start("m2_train_ranker")
    paths.apply_data_args(args, dry_run=args.dry_run)
    if args.benchmark_steps:
        benchmark(args)
        return
    print(f"## Dữ liệu: {args.data_dir}\n## Kết quả: {args.out}")
    for task in args.tasks:
        for r in args.rankers:
            if args.cv_folds and args.cv_folds > 1:
                run_task_ranker_cv(task, r, args)
            else:
                run_task_ranker(task, r, args)


if __name__ == "__main__":
    main()
