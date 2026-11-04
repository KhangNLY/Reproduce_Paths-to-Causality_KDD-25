#!/usr/bin/env python
"""MODULE 3 - Zero-shot Knowledge-Based Causal Discovery (chạy độc lập, với BẤT KỲ cách chọn metapath nào).

Các option chọn subgraph (xem lib/selection.py):
  no_subgraph  no_subgraph_pair  random  similarity  similarity_recompute  gpt  gpt_rerank
  ranker:<tag>  authors_ranker:<name>  oracle

Mỗi CẶP BIẾN = một câu hỏi (một prompt) -> LLM trả `causal` / `non-causal` (đúng giao thức của bài báo). Bài báo chỉ đánh giá trên
tập test 24-38 cặp/dataset; để có bộ câu hỏi rộng hơn dùng:
  --eval-split full          đánh giá trên TẤT CẢ cặp có metapath (train+test: 347-838 cặp/dataset, gấp ~10-20 lần). LLM zero-shot không
                             được huấn luyện nên dùng cặp train để đánh giá là hợp lệ; riêng ranker:<tag> cần xếp hạng out-of-fold:
                             chạy trước `m2_train_ranker.py --cv-folds 5`. File similarity/gpt/authors_ranker của tác giả chỉ có cho test.
  --prompt-variants 0 1 2 3 4   thử 5 cách diễn đạt câu chỉ dẫn (0 = câu gốc) -> báo cáo trung bình ± độ lệch chuẩn (độ nhạy với prompt)
  (luôn có) khoảng tin cậy bootstrap 95% của F1/Acc trong metrics.json và summary.

Ví dụ:
  python m3_zero_shot.py --dry-run                                         # kiểm tra toàn bộ pipeline với LLM giả
  python m3_zero_shot.py --backend openai --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \\
        --model meta-llama/llama-3.1-8b-instruct --tasks comagc --options no_subgraph random ranker:xgboost
  python m3_zero_shot.py --backend hf --model mistralai/Mistral-7B-Instruct-v0.1 --quant 4bit --pool all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from lib import data as D
from lib import paths, runlog
from lib.llm import add_llm_args, make_llm
from lib.metrics import binary_prf, bootstrap_ci
from lib.prompts import N_VARIANTS, build_prompt, verbalize_authors, verbalize_robust
from lib.selection import OptionUnavailable, select_paths

DEFAULT_OPTIONS = ["no_subgraph", "no_subgraph_pair", "random", "similarity", "gpt",
                   "ranker:xgboost", "ranker:ce_rmse", "ranker:ce_ranknet", "ranker:ce_listnet"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=paths.ALL_TASKS, choices=paths.ALL_TASKS)
    ap.add_argument("--options", nargs="+", default=DEFAULT_OPTIONS)
    ap.add_argument("--pool", choices=["oracle_top5", "all"], default="oracle_top5",
                    help="oracle_top5 = protocol tác giả (đã lọc bằng nhãn) | all = không rò rỉ nhãn")
    ap.add_argument("--eval-split", choices=["test", "full"], default="test",
                    help="test = tập test của bài báo (24-38 cặp) | full = tất cả cặp có metapath (train+test); ranker:<tag> dùng file out-of-fold của m2 --cv-folds")
    ap.add_argument("--prompt-variants", type=int, nargs="+", default=[0], choices=range(N_VARIANTS),
                    help=f"biến thể câu chỉ dẫn 0..{N_VARIANTS - 1} (0 = gốc); nhiều giá trị -> chạy từng biến thể + báo cáo trung bình")
    ap.add_argument("--n-subgraph", type=int, default=1, help="top-k metapath đưa vào prompt (bài báo: k=1)")
    ap.add_argument("--random-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--style", choices=["mistral", "llama", "gemma"], default="mistral",
                    help="câu kết thúc prompt (mistral: '[Relation]:', llama/gemma: 'The relation between a and b is')")
    ap.add_argument("--path-format", choices=["stops", "typed"], default="stops",
                    help="stops = chỉ tên node (code tác giả) | typed = có node-type & relation (Eq.7 trong bài)")
    ap.add_argument("--sim-embedder", choices=["minilm", "hash"], default="minilm")
    add_llm_args(ap)
    ap.add_argument("--prompt-suffix", default="", help="chuỗi nối vào CUỐI mọi prompt (vd chat API: \"\\nAnswer with only one word: 'causal' or 'non-causal'.\"); dùng đồng nhất cho mọi option")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N cặp đầu mỗi task (0 = tất cả)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="số lần bootstrap cho khoảng tin cậy 95%% (0 = tắt)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--data-dir", default=None, help="dùng dữ liệu Module 1 tự chấm (m1_score_metapaths.py) thay cho dữ liệu tác giả")
    ap.add_argument("--out", default=None, help="thư mục gốc kết quả (mặc định outputs/, hoặc outputs/data_<tên> khi có --data-dir)")
    ap.add_argument("--force", action="store_true", help="chạy lại cả phần đã có metrics.json")
    ap.add_argument("--show-prompts", type=int, default=0, help="in N prompt đầu của mỗi option")
    ap.add_argument("--dry-run", action="store_true", help="LLM giả, 3 cặp/task, ghi vào outputs/_dryrun, in prompt mẫu")
    return ap.parse_args()


def expand_options(opts, seeds):
    out = []
    for o in opts:
        if o == "random":
            out += [f"random:{s}" for s in seeds]
        else:
            out.append(o)
    return out


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def run_option(task, option, truth, llm, args, run_dir, variant=0):
    odir = run_dir / task / (slug(option) + ("" if variant == 0 else f"__pv{variant}"))
    mfile = odir / "metrics.json"
    label = f"{task}/{option}" + (f"/pv{variant}" if variant else "")
    if mfile.exists() and not args.force:
        print(f"  [skip] {label} đã có kết quả")
        return json.loads(mfile.read_text(encoding="utf-8"))
    template, selected = select_paths(task, option, truth, args.n_subgraph, args.pool, args.seed, args.out, args.sim_embedder,
                                      args.eval_split)
    prompts = [build_prompt(it, sel, template, task, args.style, args.path_format, variant) for it, sel in zip(truth, selected)]
    if args.prompt_suffix:
        prompts = [p + args.prompt_suffix.replace("\\n", "\n") for p in prompts]
    if args.show_prompts:
        for i in range(min(args.show_prompts, len(prompts))):
            print(f"\n----- PROMPT [{label}] #{i} -----\n{prompts[i]}\n----- (gt={truth[i]['ground_truth']}) -----")
    odir.mkdir(parents=True, exist_ok=True)
    pfile = odir / "predictions.jsonl"
    done = {}
    if pfile.exists() and not args.force:                     # resume theo hash prompt
        for line in open(pfile, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[r["prompt_sha"]] = r
            except Exception:
                pass
    elif pfile.exists():
        pfile.unlink()
    t0 = time.time()
    rows = [None] * len(truth)
    todo = []
    for i, p in enumerate(prompts):
        sha = hashlib.sha1((getattr(args, "system_prompt", "") + "\x00" + p).encode("utf-8")).hexdigest()   # resume theo (system prompt + prompt)
        if sha in done:
            rows[i] = done[sha]
        else:
            todo.append((i, p, sha))
    for b in range(0, len(todo), args.chunk):
        part = todo[b:b + args.chunk]
        outs = llm.generate([p for _, p, _ in part])
        with open(pfile, "a", encoding="utf-8") as f:
            for (i, p, sha), o in zip(part, outs):
                it = truth[i]
                robust, ok = verbalize_robust(o)
                r = {"idx": i, "qid": it.get("qid"), "e1": it["e1"], "e2": it["e2"], "ground_truth": int(it["ground_truth"]),
                     "selected": [s["stops"] for s in selected[i]], "prompt_sha": sha, "prompt": p, "output": o,
                     "pred_authors": verbalize_authors(o), "pred_robust": robust, "parsed_ok": ok,
                     "variant": variant, "ts": runlog.iso_now()}
                rows[i] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                runlog.file_only(f"[{label}] #{i} gt={r['ground_truth']} pred={robust} ok={ok} out={o!r}")
        k = min(b + args.chunk, len(todo))
        if len(todo) >= 4 * args.chunk and (k == len(todo) or (b // args.chunk) % 10 == 9):
            el = time.time() - t0
            print(f"    ... {label}: {k}/{len(todo)} ({100 * k / len(todo):.0f}%)  {el:.0f}s  ETA {el / k * (len(todo) - k):.0f}s", flush=True)
    y = [r["ground_truth"] for r in rows]
    m_auth = binary_prf(y, [r["pred_authors"] for r in rows])
    m_rob = binary_prf(y, [r["pred_robust"] for r in rows])
    ci = bootstrap_ci(y, [r["pred_robust"] for r in rows], args.bootstrap, args.seed) if args.bootstrap else {}
    res = {"task": task, "option": option, "template": template, "pool": args.pool, "n_subgraph": args.n_subgraph,
           "eval_split": args.eval_split, "prompt_variant": variant,
           "model": args.model if args.backend != "mock" else "mock", "backend": args.backend, "n": len(rows),
           "unparsed": sum(1 for r in rows if not r["parsed_ok"]),
           "authors_verbalizer": m_auth, "robust_verbalizer": m_rob, **ci,
           "avg_prompt_chars": round(sum(len(r["prompt"]) for r in rows) / len(rows), 1),
           "seconds_this_run": round(time.time() - t0, 1), "new_calls": len(todo),
           "finished_at": runlog.iso_now(), "log_file": str(runlog.log_path() or "")}
    res["pred_causal"] = m_rob["TP"] + m_rob["FP"]
    mfile.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if res["unparsed"] > 0.2 * len(rows):
        print(f"  !! CẢNH BÁO: {res['unparsed']}/{len(rows)} câu trả lời không có nhãn -> kết quả không đáng tin. Xem predictions.jsonl; "
              f"thử --max-new-tokens lớn hơn hoặc --prompt-suffix.")
    ci_txt = f"  F1 95%CI=[{ci['F1_ci95'][0]:.0f},{ci['F1_ci95'][1]:.0f}]" if ci else ""
    print(f"  {task:8s} {option + (f' pv{variant}' if variant else ''):24s} n={len(rows):3d}  F1(authors)={m_auth['F1']:6.2f}  F1(robust)={m_rob['F1']:6.2f}  "
          f"P/R={m_rob['P']:.1f}/{m_rob['R']:.1f}{ci_txt}  unparsed={res['unparsed']}  ({res['seconds_this_run']}s, {len(todo)} calls)")
    return res


def _mean_row(task, label, rs, variant="mean"):
    d = {"task": task, "option": label, "variant": variant, "n": rs[0]["n"], "unparsed": sum(x["unparsed"] for x in rs),
         "model": rs[0]["model"], "pool": rs[0]["pool"], "n_subgraph": rs[0]["n_subgraph"], "eval_split": rs[0]["eval_split"]}
    for k in ["P", "R", "F1", "Acc", "F1_authors_verbalizer"]:
        d[k] = round(statistics.fmean(x[k] for x in rs), 2)
    d["F1_std"] = round(statistics.pstdev(x["F1"] for x in rs), 2)
    d["pred_causal"] = ""
    d["F1_ci95"] = ""
    return d


def write_summary(run_dir, results, ts):
    cols = ["task", "option", "variant", "eval_split", "n", "P", "R", "F1", "F1_ci95", "Acc", "F1_authors_verbalizer", "unparsed",
            "pred_causal", "model", "pool", "n_subgraph", "F1_std"]
    rows = []
    for r in results:
        rb = r["robust_verbalizer"]
        ci = r.get("F1_ci95")
        rows.append({"task": r["task"], "option": r["option"], "variant": r.get("prompt_variant", 0), "eval_split": r.get("eval_split", "test"),
                     "n": r["n"], "P": rb["P"], "R": rb["R"], "F1": rb["F1"], "F1_ci95": f"[{ci[0]:.0f}, {ci[1]:.0f}]" if ci else "",
                     "Acc": rb["Acc"], "F1_authors_verbalizer": r["authors_verbalizer"]["F1"], "unparsed": r["unparsed"],
                     "pred_causal": r.get("pred_causal", ""), "model": r["model"], "pool": r["pool"], "n_subgraph": r["n_subgraph"], "F1_std": ""})
    flat = list(rows)
    seed_rows = []
    # (1) trung bình các seed random (theo từng biến thể prompt)
    for task in sorted({r["task"] for r in flat}):
        for v in sorted({r["variant"] for r in flat if r["task"] == task}):
            rs = [r for r in flat if r["task"] == task and r["variant"] == v and r["option"].startswith("random:")]
            if len(rs) > 1:
                seed_rows.append(_mean_row(task, f"random(mean of {len(rs)} seeds)", rs, v))
    rows += seed_rows
    # (2) trung bình các biến thể prompt (theo từng option; random đã gộp seed ở bước 1)
    cand = [r for r in flat if not r["option"].startswith("random:")] + seed_rows
    for task in sorted({r["task"] for r in cand}):
        for o in sorted({r["option"] for r in cand if r["task"] == task}):
            rs = [r for r in cand if r["task"] == task and r["option"] == o]
            if len({r["variant"] for r in rs}) > 1:
                rows.append(_mean_row(task, f"{o} [mean of {len(rs)} prompt variants]", rs))
    for suffix in ("", f"_{ts}"):                              # bản mới nhất + bản có timestamp
        with open(run_dir / f"summary{suffix}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        head = ["task", "option", "pv", "split", "n", "P", "R", "F1", "F1 95% CI", "Acc", "F1 (authors verbalizer)", "unparsed", "#pred causal", "F1 std"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in sorted(rows, key=lambda x: (x["task"], x["option"], str(x["variant"]))):
            lines.append(f"| {r['task']} | {r['option']} | {r['variant']} | {r['eval_split']} | {r['n']} | {r['P']} | {r['R']} | {r['F1']} | {r['F1_ci95']} | "
                         f"{r['Acc']} | {r['F1_authors_verbalizer']} | {r['unparsed']} | {r.get('pred_causal', '')} | {r['F1_std']} |")
        (run_dir / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    args = parse()
    runlog.start("m3_zero_shot")
    if args.dry_run:
        args.backend = "mock"
        args.limit = args.limit or 3
        args.show_prompts = args.show_prompts or 1
        if args.sim_embedder == "minilm":
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                print("## (dry-run) chưa cài sentence-transformers -> dùng embedder 'hash' giả cho similarity_recompute")
                args.sim_embedder = "hash"
    paths.apply_data_args(args, dry_run=args.dry_run)
    options = expand_options(args.options, args.random_seeds)
    tag = args.run_tag or f"{args.backend}_{slug(args.model.split('/')[-1]) if args.backend != 'mock' else 'mock'}_{args.pool}_k{args.n_subgraph}"
    if args.eval_split == "full" and not args.run_tag:
        tag += "_full"
    ts = runlog.now_ts()
    run_dir = Path(args.out) / "module3" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {**vars(args), "timestamp": runlog.iso_now(), "log_file": str(runlog.log_path() or "")}
    for name in ("run_args.json", f"run_args_{ts}.json"):                   # bản mới nhất + bản có timestamp
        (run_dir / name).write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    truths = {}
    n_calls = 0
    for t in args.tasks:
        tr = D.load_test_truth(t) if args.eval_split == "test" else D.load_full(t)
        truths[t] = tr[:args.limit] if args.limit else tr
        n_calls += len(truths[t]) * len(options) * len(args.prompt_variants)
    print(f"## Run: {tag}  (eval-split={args.eval_split}, prompt-variants={args.prompt_variants})\n## tasks={args.tasks} options={options}\n"
          f"## số cặp: { {t: len(v) for t, v in truths.items()} }  -> tổng số lần gọi LLM tối đa ≈ {n_calls} (chưa trừ phần đã cache/resume)")

    llm = make_llm(args)
    results, skipped = [], []
    for t in args.tasks:
        print(f"\n=== {t}: {paths.TASKS[t]['desc']}  ({len(truths[t])} cặp)")
        for o in options:
            for v in args.prompt_variants:
                try:
                    results.append(run_option(t, o, truths[t], llm, args, run_dir, v))
                except OptionUnavailable as e:
                    print(f"  [skip] {t}/{o}: {e}")
                    skipped.append({"task": t, "option": o, "reason": str(e)})
                    break
    if results:
        write_summary(run_dir, results, ts)
    (run_dir / "skipped.json").write_text(json.dumps(skipped, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n## Xong. Bảng tổng hợp: {run_dir / 'summary.md'} (bản có timestamp: summary_{ts}.md; bỏ qua {len(skipped)} option không khả dụng)")


if __name__ == "__main__":
    main()
