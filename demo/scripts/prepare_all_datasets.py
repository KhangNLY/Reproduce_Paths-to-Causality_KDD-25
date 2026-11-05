from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import translate_to_vietnamese

DATASETS = ("ade", "comagc", "gene", "semeval")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contains_chinese(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def translate_without_chinese(text: str, retries: int = 3) -> tuple[str, int]:
    translated = ""
    for attempt in range(1, retries + 1):
        try:
            translated = translate_to_vietnamese(text, strict=attempt > 1)
        except Exception as exc:
            print(f"[translation retry {attempt}/{retries}] {exc}", flush=True)
            continue
        if not contains_chinese(translated):
            return translated, attempt
    return translated, retries


def process_file(source_path: Path, refresh: bool, limit: int) -> None:
    output_path = source_path.with_name(f"processed_{source_path.stem}.jsonl")
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    source_records = load_records(source_path)
    if limit:
        source_records = source_records[:limit]

    cached: dict[str, dict] = {}
    cache_path = temp_path if temp_path.exists() and not refresh else output_path
    if cache_path.exists() and not refresh:
        cached = {
            f"{row.get('qid')}::{row.get('sentence', '')}": row
            for row in load_records(cache_path)
            if row.get("context_vi") and not contains_chinese(str(row.get("context_vi")))
        }

    processed = list(cached.values())
    processed_keys = {
        f"{row.get('qid')}::{row.get('sentence', '')}" for row in processed
    }
    for index, record in enumerate(source_records, start=1):
        sentence = str(record.get("sentence", "")).strip()
        cache_key = f"{record.get('qid')}::{sentence}"
        cached_record = cached.get(cache_key)
        if cached_record:
            print(f"[{source_path.parent.name}/{source_path.name}] {index}/{len(source_records)} cached", flush=True)
            continue

        context_vi, attempts = translate_without_chinese(sentence)
        translated = dict(record)
        translated["dataset"] = source_path.parent.name
        translated["source_file"] = source_path.name
        translated["context_en"] = sentence
        translated["context_vi"] = context_vi
        translated["translation_attempts"] = attempts
        translated["translation_status"] = "ok" if not contains_chinese(context_vi) else "contains_cjk_after_retries"
        processed.append(translated)
        processed_keys.add(cache_key)
        with temp_path.open("w", encoding="utf-8") as handle:
            for cached_row in processed:
                handle.write(json.dumps(cached_row, ensure_ascii=False) + "\n")
        suffix = "" if not contains_chinese(context_vi) else " WARNING: Chinese remains"
        print(f"[{source_path.parent.name}/{source_path.name}] {index}/{len(source_records)} translated (attempts={attempts}){suffix}", flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in processed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.unlink(missing_ok=True)
    print(f"Wrote {len(processed)} records to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate and cache all author datasets for the Streamlit UI")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset names")
    parser.add_argument("--limit", type=int, default=0, help="Limit records per source file; 0 means all")
    parser.add_argument("--train-limit", type=int, default=20, help="Limit train_full.jsonl records; 0 means all")
    parser.add_argument("--refresh", action="store_true", help="Retranslate all records instead of using cache")
    args = parser.parse_args()

    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        dataset_dir = ROOT / "data" / "benchmark" / dataset
        if not dataset_dir.exists():
            raise SystemExit(f"Missing dataset directory: {dataset_dir}")
        for source_path in sorted(dataset_dir.glob("*.jsonl")):
            if source_path.name.startswith("processed_") or source_path.name == "processed_samples.jsonl":
                continue
            limit = args.train_limit if source_path.stem == "train_full" else args.limit
            process_file(source_path, refresh=args.refresh, limit=limit)


if __name__ == "__main__":
    main()
