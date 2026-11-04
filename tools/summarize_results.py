#!/usr/bin/env python
"""Tổng hợp kết quả Module 2 và Module 3 thành bảng markdown/CSV.

  python tools/summarize_results.py                      # quét outputs/
  python tools/summarize_results.py --root outputs/_dryrun
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import paths, runlog  # noqa: E402


def module2_table(root: Path):
    rows = []
    for f in sorted((root / "module2").glob("*/*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        r = m["results"]
        rows.append({"task": m["task"], "ranker": m["ranker"], "run_id": m.get("run_id", ""), "train_s": m["train_seconds"],
                     "paper_NDCG@5": r["paper_style_top5"]["NDCG@5"], "paper_Recall@5": r["paper_style_top5"]["Recall@5"],
                     "corr_NDCG@5_top5": r["corrected_top5"]["NDCG@5"], "Top1Hit_top5": r["corrected_top5"]["Top1Hit"],
                     "corr_NDCG@5_all": r["corrected_all"]["NDCG@5"], "Top1Hit_all": r["corrected_all"]["Top1Hit"],
                     "rand_paper_NDCG@5": m["reference"]["random_expected"]["paper"]["NDCG@5"],
                     "rand_corr_NDCG@5": m["reference"]["random_expected"]["corrected"]["NDCG@5"]})
    return rows


def module3_table(root: Path):
    rows = []
    for f in sorted((root / "module3").glob("*/summary.csv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["run"] = f.parent.name
                rows.append(r)
    return rows


def md(rows, cols):
    if not rows:
        return "_(chưa có kết quả)_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.OUTPUTS))
    args = ap.parse_args()
    runlog.start("summarize_results")
    root = Path(args.root)
    m2, m3 = module2_table(root), module3_table(root)
    c2 = ["task", "ranker", "run_id", "train_s", "paper_NDCG@5", "paper_Recall@5", "corr_NDCG@5_top5", "Top1Hit_top5",
          "corr_NDCG@5_all", "Top1Hit_all", "rand_paper_NDCG@5", "rand_corr_NDCG@5"]
    c3 = ["run", "task", "option", "variant", "eval_split", "n", "P", "R", "F1", "F1_ci95", "Acc", "F1_authors_verbalizer", "unparsed"]
    text = f"<!-- generated {runlog.iso_now()} -->\n# Module 2\n\n" + md(m2, c2) + "\n# Module 3\n\n" + md(m3, c3)
    ts = runlog.now_ts()
    (root / "SUMMARY.md").write_text(text, encoding="utf-8")                   # bản mới nhất
    (root / f"SUMMARY_{ts}.md").write_text(text, encoding="utf-8")             # bản có timestamp
    print(text)
    print(f"-> {root / 'SUMMARY.md'}  (và SUMMARY_{ts}.md)")


if __name__ == "__main__":
    main()

