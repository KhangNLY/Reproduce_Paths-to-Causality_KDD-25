from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import LocalLLMError, direct_predict_causality, translate_to_vietnamese


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entity_name(row: dict, side: int) -> str:
    return str(row.get(f"entity_{side}") or row.get(f"e{side}") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached direct Z.ai GLM inference for dataset samples")
    parser.add_argument("--datasets", default="ade,comagc,gene,semeval")
    parser.add_argument("--train-limit", type=int, default=20)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    from src.neo4j_store import KGStore

    store = KGStore()
    try:
        for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
            dataset_dir = ROOT / "data" / "benchmark" / dataset
            for source_path in sorted(dataset_dir.glob("processed_*.jsonl")):
                if source_path.name == "processed_samples.jsonl" or "sample" in source_path.name or source_path.name.startswith("zai_inference_"):
                    continue
                records = load_jsonl(source_path)
                if source_path.stem == "processed_train_full" and args.train_limit:
                    records = records[: args.train_limit]
                output_path = source_path.with_name(source_path.name.replace("processed_", "zai_inference_", 1))
                temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                cached = {} if args.refresh or not output_path.exists() else {
                    f"{row.get('qid')}::{row.get('source_file', '')}": row
                    for row in load_jsonl(output_path)
                    if row.get("status") in {"ok", "unresolved_entity"}
                }
                results = list(cached.values())
                for index, row in enumerate(records, start=1):
                    key = f"{row.get('qid')}::{row.get('source_file', source_path.name)}"
                    if key in cached:
                        print(f"[{dataset}/{source_path.name}] {index}/{len(records)} cached", flush=True)
                        continue
                    name_1, name_2 = entity_name(row, 1), entity_name(row, 2)
                    base = {
                        "dataset": dataset,
                        "source_file": row.get("source_file", source_path.name),
                        "qid": row.get("qid"),
                        "entity_1": name_1,
                        "entity_2": name_2,
                        "ground_truth": row.get("ground_truth"),
                        "context_en": row.get("context_en", row.get("sentence", "")),
                    }
                    preferred_kind_1 = "Gene" if dataset == "comagc" else None
                    preferred_kind_2 = "Disease" if dataset == "comagc" else None
                    entity_1 = store.resolve_entity(name_1, preferred_kind=preferred_kind_1)
                    entity_2 = store.resolve_entity(name_2, preferred_kind=preferred_kind_2)
                    if not entity_1 or not entity_2:
                        base.update({"status": "unresolved_entity", "prediction": None})
                    else:
                        try:
                            relations = store.direct_relations(entity_1["id"], entity_2["id"])
                            prediction = direct_predict_causality(entity_1, entity_2, base["context_en"], relations)
                            base.update({
                                "status": "ok",
                                "entity_1_resolved": entity_1,
                                "entity_2_resolved": entity_2,
                                "direct_relations": relations,
                                "prediction": prediction,
                                "context_vi": row.get("context_vi") or translate_to_vietnamese(base["context_en"]),
                                "explanation_vi": translate_to_vietnamese(prediction["explanation"]),
                            })
                        except LocalLLMError as error:
                            base.update({"status": "llm_error", "error": str(error), "prediction": None})
                    results.append(base)
                    with temp_path.open("w", encoding="utf-8") as handle:
                        for result in results:
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(f"[{dataset}/{source_path.name}] {index}/{len(records)} processed ({base['status']})", flush=True)
                with output_path.open("w", encoding="utf-8") as handle:
                    for result in results:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                temp_path.unlink(missing_ok=True)
                print(f"Wrote {len(results)} cached Z.ai results to {output_path}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
