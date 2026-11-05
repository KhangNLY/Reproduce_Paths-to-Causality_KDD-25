from __future__ import annotations
import argparse, csv, json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from src.config import settings
from src.llm import LocalLLMError
from src.neo4j_store import KGStore
from src.pipeline import analyze_pair

def main():
    ap = argparse.ArgumentParser(description="Evaluate pairwise predictions on a labeled CSV")
    ap.add_argument("--input", default="data/benchmark/causal_pairs.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--output", default="results/evaluation_predictions.csv")
    ap.add_argument("--no-llm", action="store_true", help="Not implemented: kept as a guard against accidental misuse")
    ap.add_argument("--metapath-jsonl", default=None,
                    help="Optional benchmark jsonl with metapaths; when set, entities are resolved "
                         "from metapath endpoints (the paper's own entity linking) instead of raw "
                         "benchmark names, which mostly do not match Hetionet node names.")
    args = ap.parse_args()
    if args.no_llm:
        raise SystemExit("Evaluation uses the local LLM. Omit --no-llm.")
    src = ROOT / args.input
    if not src.exists():
        raise SystemExit(f"Missing {src}; see data/benchmark/README.md")
    with src.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    if not rows or not {"entity_1", "entity_2", "context", "label"}.issubset(rows[0]):
        raise SystemExit("Input CSV must include entity_1, entity_2, context, label")
    records = []
    metapath_by_qid = {}
    if args.metapath_jsonl:
        mp_path = ROOT / args.metapath_jsonl
        if not mp_path.exists():
            raise SystemExit(f"Missing {mp_path}")
        for line in mp_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            mps = row.get("metapaths") or []
            if not mps:
                continue
            metapath_by_qid[(row["e1"], row["e2"])] = (mps[0]["stops"].split(" - ")[0].strip(),
                                                       mps[0]["stops"].split(" - ")[-1].strip())
    for idx, row in enumerate(rows, 1):
        store = KGStore()
        try:
            names = metapath_by_qid.get((row["entity_1"], row["entity_2"]))
            e1 = store.resolve_entity(names[0] if names else row["entity_1"])
            e2 = store.resolve_entity(names[1] if names else row["entity_2"])
        finally:
            store.close()
        if not e1 or not e2 or e1["id"] == e2["id"]:
            print(f"[skip] row {idx}: unable to resolve entities", flush=True)
            continue
        # Ollama qua host.docker.internal có thể read-timeout khi máy tải nặng;
        # retry từng pair thay vì để một lần fail giết cả batch.
        pred = None
        for attempt in range(2):
            try:
                pred = analyze_pair(e1, e2, row["context"])["prediction"]["label"]
                break
            except LocalLLMError as error:
                print(f"[retry {attempt + 1}/2] row {idx}: {error}", flush=True)
                if attempt == 1:
                    pred = "error"
        records.append({"entity_1": row["entity_1"], "entity_2": row["entity_2"], "context": row["context"], "gold": row["label"].lower().strip(), "prediction": pred})
        # Ghi incremental để một crash giữa chừng không mất tiến độ đã chạy.
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["entity_1", "entity_2", "context", "gold", "prediction"])
            w.writeheader()
            w.writerows(records)
        print(f"[{idx}/{len(rows)}] gold={row['label']} prediction={pred}", flush=True)
    labels = [r["gold"] for r in records]
    preds = [r["prediction"] for r in records]
    scored = [(y,p) for y,p in zip(labels,preds) if p in {"causal", "non-causal"} and y in {"causal", "non-causal"}]
    print(f"Processed={len(records)}, scored (excluding uncertain)={len(scored)}")
    if scored:
        y, p = zip(*scored)
        print(classification_report(y, p, labels=["causal", "non-causal"], zero_division=0))
        print("Confusion matrix rows=gold, cols=prediction:", confusion_matrix(y, p, labels=["causal", "non-causal"]).tolist())
        print("Macro P/R/F1:", precision_recall_fscore_support(y, p, labels=["causal", "non-causal"], average="macro", zero_division=0))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entity_1", "entity_2", "context", "gold", "prediction"])
        w.writeheader(); w.writerows(records)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
