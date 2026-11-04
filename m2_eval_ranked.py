#!/usr/bin/env python
"""MODULE 2 (chỉ đánh giá) - tính NDCG@k / Recall@k cho file xếp hạng có sẵn, KHÔNG cần huấn luyện.

  python m2_eval_ranked.py --authors                 # đánh giá ranker/GPT mà tác giả đã công bố (tái lập Table D2)
  python m2_eval_ranked.py --task comagc --files outputs/module2/comagc/xgboost/test_ranked_top5.jsonl

Cột 'paper_*' = đúng cách tính của tác giả (qrels gán ngược, Recall@5 hằng số);
cột 'corr_*'  = cách tính đã sửa (path tốt nhất có gain cao nhất; Overlap@k = Recall@k so với tập top-k của oracle).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

from lib import data as D
from lib import metrics as M
from lib import paths, runlog

HEAD = ["task", "ranker", "paper_NDCG@1", "paper_NDCG@3", "paper_NDCG@5", "paper_Recall@5",
        "corr_NDCG@1", "corr_NDCG@5", "corr_Overlap@5", "Top1Hit", "MeanOracleRank_Top1"]


def eval_file(task, name, pred, truth):
    p = M.paper_style_ranking_eval(truth, pred)
    c = M.corrected_ranking_eval(truth, pred, pool_size=5)
    return {"task": task, "ranker": name, "paper_NDCG@1": p["NDCG@1"], "paper_NDCG@3": p["NDCG@3"], "paper_NDCG@5": p["NDCG@5"],
            "paper_Recall@5": p["Recall@5"], "corr_NDCG@1": c["NDCG@1"], "corr_NDCG@5": c["NDCG@5"],
            "corr_Overlap@5": c["Overlap@5"], "Top1Hit": c["Top1Hit"], "MeanOracleRank_Top1": c["MeanOracleRank_Top1"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authors", action="store_true", help="đánh giá file dự đoán đã công bố của tác giả (4 task)")
    ap.add_argument("--task", choices=paths.ALL_TASKS)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--out", default=str(paths.OUTPUTS / "module2" / "eval_ranked.csv"))
    args = ap.parse_args()
    runlog.start("m2_eval_ranked")
    rows = []
    if args.authors:
        for t in paths.ALL_TASKS:
            truth = D.load_test_truth(t)
            ref = M.reference_orderings(truth, pool_size=5, n_random=300)
            for nm in ("oracle", "reverse_oracle", "random_expected"):
                rows.append({"task": t, "ranker": f"[ref] {nm}", "paper_NDCG@1": ref[nm]["paper"]["NDCG@1"],
                             "paper_NDCG@3": ref[nm]["paper"]["NDCG@3"], "paper_NDCG@5": ref[nm]["paper"]["NDCG@5"],
                             "paper_Recall@5": ref[nm]["paper"]["Recall@5"], "corr_NDCG@1": ref[nm]["corrected"]["NDCG@1"],
                             "corr_NDCG@5": ref[nm]["corrected"]["NDCG@5"], "corr_Overlap@5": ref[nm]["corrected"]["Overlap@5"],
                             "Top1Hit": ref[nm]["corrected"]["Top1Hit"], "MeanOracleRank_Top1": ref[nm]["corrected"]["MeanOracleRank_Top1"]})
            for d in sorted(glob.glob(str(paths.AUTHOR_CKPT / t / "*"))):
                f = glob.glob(d + "/test_pred_*.jsonl")
                if f:
                    rows.append(eval_file(t, "authors:" + os.path.basename(d), D.load_jsonl(f[0]), truth))
            rows.append(eval_file(t, "authors:RankGPT(gpt-3.5)", D.load_authors_gpt(t), truth))
    if args.files:
        if not args.task:
            ap.error("--files cần --task")
        truth = D.load_test_truth(args.task)
        for f in args.files:
            rows.append(eval_file(args.task, Path(f).parent.name, D.load_jsonl(f), truth))
    if not rows:
        ap.error("chọn --authors và/hoặc --files")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stamped_out = runlog.stamped(args.out)
    for dest in (Path(args.out), stamped_out):               # bản mới nhất + bản có timestamp
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEAD)
            w.writeheader()
            w.writerows(rows)
    print("| " + " | ".join(HEAD) + " |\n|" + "---|" * len(HEAD))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in HEAD) + " |")
    print(f"\nĐã ghi {args.out} (và bản có timestamp: {stamped_out.name})")


if __name__ == "__main__":
    main()

