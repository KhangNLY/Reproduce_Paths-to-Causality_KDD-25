from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import translate_to_vietnamese


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate and cache COMAGC samples for the Streamlit UI")
    parser.add_argument("--input", default="data/benchmark/comagc/test_truth_sample.jsonl")
    parser.add_argument("--output", default="data/benchmark/comagc/processed_samples.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--refresh", action="store_true", help="Retranslate records already present in the cache")
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}")

    source_records = load_records(input_path)
    if args.limit:
        source_records = source_records[: args.limit]

    cached: dict[str, dict] = {}
    if output_path.exists() and not args.refresh:
        cached = {str(row["qid"]): row for row in load_records(output_path) if row.get("context_vi")}

    processed = []
    for record in source_records:
        qid = str(record["qid"])
        cached_record = cached.get(qid)
        if cached_record:
            processed.append(cached_record)
            print(f"[cached] qid={qid}", flush=True)
            continue

        context_en = str(record["sentence"]).strip()
        context_vi = translate_to_vietnamese(context_en)
        processed.append(
            {
                "qid": qid,
                "entity_1": str(record["e1"]).strip(),
                "entity_2": str(record["e2"]).strip(),
                "context_en": context_en,
                "context_vi": context_vi,
                "label": "causal" if str(record.get("ground_truth", "")) == "1" else "non-causal",
                "ground_truth": str(record.get("ground_truth", "")),
                "metapaths": record.get("metapaths", []),
            }
        )
        print(f"[translated] qid={qid}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in processed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(processed)} cached COMAGC samples to {output_path}")


if __name__ == "__main__":
    main()
