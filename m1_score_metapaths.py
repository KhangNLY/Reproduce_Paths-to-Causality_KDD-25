#!/usr/bin/env python
"""MODULE 1 (chạy ĐỘC LẬP, không dùng điểm của tác giả) - Subgraph Relevancy Estimation: tự chấm điểm metapath bằng LLM.

Với mỗi (cặp biến, MỘT metapath) -> hỏi LLM `causal / non-causal` (prompt tcma, k=1) -> metapath 'relevant' nếu LLM đoán đúng
nhãn người gán -> prob_score = log-prob của token quyết định nhãn -> rel_score (xem lib/relevancy.py) -> xếp hạng trong từng cặp.

ĐẦU VÀO là các ỨNG VIÊN (cặp + metapath đã trích từ KG). Mặc định lấy từ data/authors/datasets/<task>/{train_full,test_truth}.jsonl
nhưng MỌI điểm của tác giả (rel_score, prob_score, relevant) bị BỎ trước khi dùng. Bước trích metapath từ Hetionet/Wikidata
(Neo4j / CLOCQ) của tác giả không được công bố nên không tái lập ở đây; muốn dùng ứng viên của riêng bạn, đưa --input-dir với
cấu trúc <task>/{train_full,test_truth}.jsonl, mỗi dòng:
  {"qid", "sentence", "e1", "e2", "ground_truth": "0|1", "metapaths": [{"pathid", "stops", "reltypes", "nodelabels"}, ...]}

ĐẦU RA: outputs/module1/rescored/<tag>_<timestamp>/<task>/{train_full,test_truth}.jsonl (cùng schema dữ liệu tác giả) + meta.json + report.md.
Dùng tiếp cho Module 2/3:   python m2_train_ranker.py --data-dir outputs/module1/rescored/<run> ...   (kết quả vào outputs/data_<run>/)

Ví dụ:
  python m1_score_metapaths.py --dry-run                                         # LLM giả, vài cặp: kiểm tra pipeline
  python m1_score_metapaths.py --backend openai --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \\
        --model mistralai/mistral-nemo --style mistral --tasks comagc --limit-pairs 20      # thử nhỏ (API có logprobs)
  python m1_score_metapaths.py --backend hf --model mistralai/Mistral-7B-Instruct-v0.1 --quant 4bit     # đúng cách bài báo (local)
  python m1_score_metapaths.py ... --resume outputs/module1/rescored/<run>               # chạy tiếp sau khi bị ngắt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from lib import data as D
from lib import paths, runlog
from lib.llm import add_llm_args, make_llm
from lib.prompts import CHAT_ANSWER_SUFFIX, CHAT_SYSTEM_PROMPT, N_VARIANTS, build_prompt, verbalize_robust
from lib.relevancy import FORMULAS, LOGP_MODES, label_logprob, rel_score, spearman, strip_scores

LOGP_EMPTY = -20.0       # log-prob gán khi model không sinh được token nào (câu trả lời rỗng)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=paths.ALL_TASKS, choices=paths.ALL_TASKS)
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
                    help="train = train_full.jsonl (dữ liệu huấn luyện Module 2) | test = test_truth.jsonl")
    ap.add_argument("--input-dir", default=None, help="thư mục ứng viên <task>/{train_full,test_truth}.jsonl (mặc định: dữ liệu tác giả, đã bỏ điểm)")
    ap.add_argument("--limit-pairs", type=int, default=0, help="chỉ chấm N cặp đầu mỗi (task, split); 0 = tất cả")
    ap.add_argument("--style", choices=["mistral", "llama", "gemma"], default="mistral")
    ap.add_argument("--path-format", choices=["stops", "typed"], default="stops")
    ap.add_argument("--prompt-variant", type=int, default=0, choices=range(N_VARIANTS), help="biến thể câu chỉ dẫn (0 = gốc của tác giả)")
    ap.add_argument("--prompt-suffix", default=None, help="nối vào cuối prompt (mặc định: chat API -> câu ép 1 từ; hf/completion -> không)")
    ap.add_argument("--logp-mode", choices=LOGP_MODES, default="authors",
                    help="authors = token đầu tiên thuộc NEG/POS (y như code tác giả) | first_token | label_tokens (tổng log-prob chuỗi nhãn)")
    ap.add_argument("--score-formula", choices=FORMULAS, default="authors",
                    help="authors = ±1/|logp| (công thức thật trong dữ liệu đã công bố) | eq1_prob = 1±p | eq1_logprob = 1±logp (Eq.1 theo chữ)")
    ap.add_argument("--logp-floor", type=float, default=1e-3, help="chặn dưới của |logp| để 1/|logp| không vô hạn")
    ap.add_argument("--chunk", type=int, default=32, help="số prompt mỗi lượt gọi (lưu tiến độ sau mỗi lượt)")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--resume", default=None, help="thư mục run cũ để chạy tiếp (bỏ qua các path đã có kết quả thô)")
    ap.add_argument("--out", default=None, help="mặc định outputs/module1/rescored")
    ap.add_argument("--no-compare", action="store_true", help="không so sánh với điểm của tác giả")
    ap.add_argument("--dry-run", action="store_true", help="LLM giả, 4 cặp/(task, split), ghi vào outputs/_dryrun")
    add_llm_args(ap)
    ap.set_defaults(max_new_tokens=8)
    return ap.parse_args()


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def load_candidates(task, split, input_dir):
    name = "train_full.jsonl" if split == "train" else "test_truth.jsonl"
    f = (Path(input_dir) / task / name) if input_dir else (paths.AUTHORS_DATASETS / task / name)
    if not f.exists():
        raise FileNotFoundError(f"không thấy {f}. Chạy `python tools/fetch_authors_data.py` hoặc dùng --input-dir")
    raw = D.load_jsonl(f)
    return raw, [strip_scores(it) for it in raw]


def score_split(task, split, args, llm, run_dir, suffix):
    raw_items, items = load_candidates(task, split, args.input_dir)
    if args.limit_pairs:
        raw_items, items = raw_items[:args.limit_pairs], items[:args.limit_pairs]
    rawf = run_dir / "raw" / f"{task}_{split}.jsonl"
    rawf.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if rawf.exists():
        for line in open(rawf, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[r["sha"]] = r
            except Exception:
                pass
    jobs = []        # (pair_idx, path_idx, prompt, sha)
    for i, it in enumerate(items):
        for j, m in enumerate(it["metapaths"]):
            p = build_prompt(it, [m], "tcma", task, args.style, args.path_format, args.prompt_variant) + suffix
            sha = hashlib.sha1(f"{getattr(args, 'system_prompt', '')}\x00{p}".encode("utf-8")).hexdigest()
            jobs.append((i, j, p, sha))
    todo = [x for x in jobs if x[3] not in done]
    print(f"\n=== [{task}/{split}] {len(items)} cặp, {len(jobs)} metapath; đã có {len(jobs) - len(todo)}, cần gọi LLM {len(todo)} lần", flush=True)
    t0 = time.time()
    for b in range(0, len(todo), args.chunk):
        part = todo[b:b + args.chunk]
        outs = llm.generate_scored([p for _, _, p, _ in part])
        with open(rawf, "a", encoding="utf-8") as f:
            for (i, j, p, sha), o in zip(part, outs):
                rec = {"sha": sha, "pair": i, "path": j, "ts": runlog.iso_now(), "text": o["text"], "tokens": o["tokens"]}
                done[sha] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                runlog.file_only(f"[{task}/{split}] pair={i} path={j} out={o['text']!r} tok0={o['tokens'][:1]}")
        k = min(b + args.chunk, len(todo))
        el = time.time() - t0
        eta = el / k * (len(todo) - k) if k else 0
        print(f"  [{task}/{split}] {k}/{len(todo)} ({100 * k / len(todo):.0f}%)  {el:.0f}s  ETA {eta:.0f}s", flush=True)

    # ---- dựng lại file có điểm -------------------------------------------------------------------------------------------
    out_items, n_unparsed, n_empty = [], 0, 0
    by_pos = {(i, j): sha for i, j, _, sha in jobs}
    for i, it in enumerate(items):
        gt = int(it["ground_truth"])
        new_mp = []
        for j, m in enumerate(it["metapaths"]):
            r = done[by_pos[(i, j)]]
            pred, ok = verbalize_robust(r["text"])
            logp, _tok = label_logprob(r["tokens"], args.logp_mode)
            if logp is None:
                logp, n_empty = LOGP_EMPTY, n_empty + 1
            n_unparsed += (not ok)
            correct = (pred == gt)
            nm = dict(m)
            nm.update(rel_score=round(rel_score(correct, logp, args.score_formula, args.logp_floor), 5), prob_score=logp,
                      relevant="1" if correct else "0", llm_output=r["text"], parsed_ok=ok)
            new_mp.append(nm)
        order = sorted(range(len(new_mp)), key=lambda j: -new_mp[j]["rel_score"])          # sort ổn định, giảm dần
        new = dict(it)
        new["metapaths"] = [new_mp[j] for j in order]
        out_items.append(new)
    st = {"pairs": len(items), "paths": len(jobs), "unparsed": n_unparsed, "empty_logprob": n_empty,
          "llm_correct_rate": round(statistics.fmean(m["relevant"] == "1" for it in out_items for m in it["metapaths"]), 4)}
    if n_unparsed > 0.2 * max(len(jobs), 1):
        print(f"  !! CẢNH BÁO [{task}/{split}]: {n_unparsed}/{len(jobs)} câu trả lời không có nhãn causal/non-causal (bị tính là non-causal) -> "
              "điểm Module 1 không đáng tin. Xem raw/*.jsonl; thử --system-prompt/--prompt-suffix hoặc model khác.", flush=True)
    st.update(compare_with_authors(task, split, raw_items, out_items, args))
    return out_items, st


def compare_with_authors(task, split, raw_items, mine, args):
    """So khớp với điểm của tác giả (khác LLM nên KHÔNG kỳ vọng giống hệt) - chỉ để kiểm tra độ hợp lý."""
    if args.no_compare or args.input_dir or "rel_score" not in raw_items[0]["metapaths"][0]:
        return {}
    agree = tot = 0
    rhos = []
    for a, b in zip(raw_items, mine):
        fa = {m["pathid"]: str(m["relevant"]) for m in a["metapaths"]}
        sa = {m["pathid"]: m["rel_score"] for m in a["metapaths"]}
        pids = [m["pathid"] for m in b["metapaths"]]
        for m in b["metapaths"]:
            agree += (fa[m["pathid"]] == m["relevant"])
            tot += 1
        rho = spearman([sa[p] for p in pids], [m["rel_score"] for m in b["metapaths"]])
        if rho is not None:
            rhos.append(rho)
    return {"vs_authors_relevant_agreement": round(agree / tot, 4),
            "vs_authors_mean_spearman_rel_score": round(statistics.fmean(rhos), 4) if rhos else None,
            "authors_llm_correct_rate": round(statistics.fmean(str(m["relevant"]) == "1" for it in raw_items for m in it["metapaths"]), 4)}


def main():
    args = parse()
    runlog.start("m1_score_metapaths")
    if args.dry_run:
        args.backend, args.limit_pairs = "mock", args.limit_pairs or 4
    paths.apply_data_args(args, dry_run=args.dry_run)
    base = Path(args.out) / "module1" / "rescored"
    if args.backend == "openai" and args.api_mode == "chat":       # model chat cần bị ép trả đúng 1 từ (xem README, mục Deviations)
        args.system_prompt = args.system_prompt or CHAT_SYSTEM_PROMPT
    suffix = args.prompt_suffix if args.prompt_suffix is not None else (CHAT_ANSWER_SUFFIX if (args.backend == "openai" and args.api_mode == "chat") else "")
    suffix = suffix.replace("\\n", "\n")
    ts = runlog.now_ts()
    tag = args.run_tag or f"{args.backend}_{slug(args.model.split('/')[-1]) if args.backend != 'mock' else 'mock'}_{args.logp_mode}"
    run_dir = Path(args.resume) if args.resume else base / f"{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp": runlog.iso_now(), "run_dir": str(run_dir), "log_file": str(runlog.log_path() or ""),
            "args": {k: v for k, v in vars(args).items()}, "prompt_suffix": suffix}
    (run_dir / f"run_args_{ts}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"## Module 1 (tự chấm điểm) -> {run_dir}\n## model={args.model if args.backend != 'mock' else 'mock'} backend={args.backend} "
          f"logp_mode={args.logp_mode} formula={args.score_formula} variant={args.prompt_variant}", flush=True)

    llm = make_llm(args)
    stats = {}
    for t in args.tasks:
        for s in args.splits:
            items, st = score_split(t, s, args, llm, run_dir, suffix)
            D.write_jsonl(run_dir / t / ("train_full.jsonl" if s == "train" else "test_truth.jsonl"), items)
            stats[f"{t}/{s}"] = st
            print(f"  -> {t}/{s}: {st}", flush=True)
    (run_dir / "meta.json").write_text(json.dumps({**meta, "stats": stats, "finished": runlog.iso_now()}, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    L = ["# Module 1 - tự chấm điểm metapath", "", f"* run: `{run_dir.name}`  ·  model: `{args.model}` ({args.backend})  ·  logp-mode: `{args.logp_mode}`  ·  formula: `{args.score_formula}`",
         f"* log file: `{runlog.log_path()}`", "",
         "| task/split | pairs | paths | LLM đúng (relevant=1) | unparsed | khớp cờ relevant với tác giả | Spearman(rel_score) với tác giả |", "|---|---|---|---|---|---|---|"]
    for k, st in stats.items():
        L.append(f"| {k} | {st['pairs']} | {st['paths']} | {st['llm_correct_rate']} | {st['unparsed']} | "
                 f"{st.get('vs_authors_relevant_agreement', '-')} | {st.get('vs_authors_mean_spearman_rel_score', '-')} |")
    L += ["", "Khác LLM với tác giả (mistral-7b-instruct-v0.1 local) nên cờ/điểm KHÔNG kỳ vọng giống hệt; hai cột cuối chỉ để kiểm tra độ hợp lý.",
          "", f"Dùng tiếp: `python m2_train_ranker.py --data-dir {run_dir} --rankers xgboost`"]
    (run_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\n## Xong. Dữ liệu đã chấm: {run_dir}")


if __name__ == "__main__":
    main()
