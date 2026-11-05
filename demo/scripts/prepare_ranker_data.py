from __future__ import annotations
import argparse, csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.neo4j_store import KGStore
from src.features import FEATURE_NAMES, path_features, path_text
from src.llm import score_paths_relevance

def main():
    p = argparse.ArgumentParser(description="Create weakly supervised path-relevance labels with local Ollama")
    p.add_argument("--input", default="data/benchmark/causal_pairs.csv")
    p.add_argument("--output", default="data/benchmark/ranking_train.csv")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-paths", type=int, default=15)
    args = p.parse_args()
    input_path = ROOT / args.input
    output_path = ROOT / args.output
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}. See data/benchmark/README.md for expected CSV schema.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    store = KGStore()
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"entity_1", "entity_2", "context"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise SystemExit(f"CSV must include columns {sorted(required)}; got {reader.fieldnames}")
            for i, row in enumerate(reader):
                if i >= args.limit:
                    break
                e1 = store.resolve_entity(row["entity_1"])
                e2 = store.resolve_entity(row["entity_2"])
                if not e1 or not e2 or e1["id"] == e2["id"]:
                    print(f"[skip] row {i+1}: unable to resolve distinct entities")
                    continue
                paths = store.shortest_paths(e1["id"], e2["id"], settings.max_hops, args.max_paths)
                if not paths:
                    print(f"[skip] row {i+1}: no KG path")
                    continue
                for path in paths:
                    path["path_text"] = path_text(path)
                scores = score_paths_relevance(e1["name"], e2["name"], row["context"], paths)
                group = f"pair_{i:06d}"
                for path, label in zip(paths, scores):
                    feats = path_features(path, row["context"], e1["name"], e2["name"])
                    item = {"group_id": group, "label": label, "entity_1": e1["name"], "entity_2": e2["name"], "context": row["context"], "path_text": path["path_text"]}
                    item.update(dict(zip(FEATURE_NAMES, feats)))
                    rows_out.append(item)
                print(f"[ok] pair {i+1}: {len(paths)} paths rated")
    finally:
        store.close()
    if not rows_out:
        raise SystemExit("No ranking rows created. Verify entity names, KG load and Ollama model.")
    cols = ["group_id", "label", "entity_1", "entity_2", "context", "path_text", *FEATURE_NAMES]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} path examples to {output_path}")
    print("Important: labels are local-LLM pseudo-labels, not human gold labels. Keep benchmark test pairs out of ranker training.")

if __name__ == "__main__":
    main()
