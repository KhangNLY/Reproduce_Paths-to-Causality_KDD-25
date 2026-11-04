"""Đọc/ghi dữ liệu Module 1 (metapath đã được tác giả chấm điểm) và tiện ích về metapath."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from . import paths


def load_jsonl(p) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p, rows: Iterable[dict]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_train(task: str) -> List[dict]:
    return load_jsonl(paths.task_dir(task) / "train_full.jsonl")


def load_test_truth(task: str) -> List[dict]:
    """Test + metapath đã chấm điểm (Module 1). Metapath đã được SẮP XẾP giảm dần theo rel_score."""
    return load_jsonl(paths.task_dir(task) / "test_truth.jsonl")


def load_authors_sim(task: str) -> List[dict]:
    return load_jsonl(paths.authors_task_dir(task) / "test_sim.jsonl")


def load_authors_gpt(task: str) -> List[dict]:
    return load_jsonl(paths.authors_task_dir(task) / paths.TASKS[task]["gpt_file"])


def load_full(task: str) -> List[dict]:
    """TẤT CẢ cặp có metapath = train_full + test_truth (dùng cho --eval-split full / cross-validation)."""
    return load_train(task) + load_test_truth(task)


# ---------------------------------------------------------------------------
# Văn bản hoá metapath
# ---------------------------------------------------------------------------
def path_text(m: dict, include_node_labels: bool = False) -> str:
    """Chuỗi đưa vào ranker. Giống tác giả: 'stops' (tên node), tuỳ chọn ghép node-type vào từng node.

    Ví dụ: 'BAX - THAP11 - Fulvestrant'  ->  'Gene BAX - Gene THAP11 - Compound Fulvestrant'
    """
    stops = str(m["stops"])
    if not include_node_labels:
        return stops
    nl = str(m.get("nodelabels", "")).split(" - ")
    st = stops.split(" - ")
    if len(nl) != len(st):          # tên node có chứa ' - ' -> không ghép được, dùng stops thuần
        return stops
    return " - ".join(f"{a} {b}" for a, b in zip(nl, st))


def typed_path_text(m: dict) -> str:
    """Dạng 'có kiểu' của Eq.(7) trong bài: (type1 v1) -rel-> (type2 v2) ...  (dùng cho ablation --path-format typed)."""
    stops = str(m["stops"]).split(" - ")
    nl = str(m.get("nodelabels", "")).split(" - ")
    rel = [r for r in str(m.get("reltypes", "")).split(" - ")]
    if len(nl) != len(stops) or len(rel) != len(stops) - 1:
        return str(m["stops"])
    parts = []
    for i in range(len(stops) - 1):
        r = rel[i].split("_")[0].lower()
        parts.append(f"{nl[i]} {stops[i]} --{r}--> {nl[i+1]} {stops[i+1]}")
    return "; ".join(parts)


def query_text(item: dict) -> str:
    return f"{item['e1']} {item['e2']}"


def check_aligned(a: List[dict], b: List[dict], name: str = "") -> None:
    """Hai danh sách item phải cùng thứ tự (so theo cặp e1,e2)."""
    if len(a) != len(b):
        raise ValueError(f"[{name}] số item khác nhau: {len(a)} vs {len(b)}")
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if (x["e1"], x["e2"]) != (y["e1"], y["e2"])]
    if bad:
        raise ValueError(f"[{name}] {len(bad)} item lệch cặp (e1,e2), ví dụ index {bad[:5]}")

