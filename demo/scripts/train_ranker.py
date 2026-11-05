from __future__ import annotations
import argparse, csv, os
from collections import Counter
import sys
from pathlib import Path
import numpy as np
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_NAMES

def main():
    ap = argparse.ArgumentParser(description="Train an XGBoost pairwise ranker from path relevance labels")
    ap.add_argument("--input", default="data/benchmark/ranking_train.csv")
    ap.add_argument("--output", default="models/xgb_ranker.json")
    ap.add_argument("--estimators", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    src = ROOT / args.input
    dst = ROOT / args.output
    if not src.exists():
        raise SystemExit(f"Missing {src}; first run scripts/prepare_ranker_data.py")
    with src.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"group_id", "label", *FEATURE_NAMES}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Input requires columns {sorted(required)}")
    # Preserve group-contiguous row order for XGBRanker.
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    valid_groups = {g: rs for g, rs in groups.items() if len(rs) >= 2 and len({int(r['label']) for r in rs}) > 1}
    if len(valid_groups) < 2:
        raise SystemExit(f"Need at least 2 query groups with >=2 paths and varying relevance labels; got {len(valid_groups)}. Generate more ranking data.")
    ordered = [r for group in valid_groups.values() for r in group]
    X = np.asarray([[float(r[name]) for name in FEATURE_NAMES] for r in ordered], dtype=float)
    y = np.asarray([int(r["label"]) for r in ordered], dtype=int)
    group_sizes = [len(rs) for rs in valid_groups.values()]
    model = XGBRanker(
        objective="rank:pairwise", n_estimators=args.estimators, max_depth=3,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=args.seed, tree_method="hist", eval_metric="ndcg@5",
    )
    model.fit(X, y, group=group_sizes, verbose=False)
    dst = ROOT / args.output
    dst.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(dst)
    print(f"Saved {dst}; groups={len(group_sizes)}, paths={len(y)}, relevance_distribution={dict(Counter(y))}")
    print("For a scientific evaluation, split pairs/groups before pseudo-label generation; these training pseudo-labels are not gold evaluation labels.")

if __name__ == "__main__":
    main()
