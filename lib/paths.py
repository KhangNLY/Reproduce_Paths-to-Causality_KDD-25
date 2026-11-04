"""Đường dẫn chuẩn của dự án. Mọi thứ nằm trong My_experiments/ (không phụ thuộc ptc_experiments).

Nguồn dữ liệu Module 1 (metapath đã chấm điểm) có thể đổi bằng `use_data_dir(<thư mục>)` (hoặc biến môi trường PTC_DATA_DIR,
hoặc cờ --data-dir của các script): thư mục đó phải có cấu trúc <task>/{train_full,test_truth}.jsonl giống data/authors/datasets.
Khi dùng dữ liệu tự chấm điểm (m1_score_metapaths.py) thì kết quả Module 2/3 mặc định ghi vào outputs/data_<tên thư mục>/
để không đè lên kết quả chạy bằng dữ liệu của tác giả.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../My_experiments
AUTHORS = ROOT / "data" / "authors"                     # bản sao read-only của repo tác giả (tools/fetch_authors_data.py)
AUTHORS_DATASETS = AUTHORS / "datasets"
DATASETS = AUTHORS_DATASETS                             # thư mục dữ liệu ĐANG DÙNG (đổi bằng use_data_dir)
DATA_NAME = None                                        # None = dữ liệu tác giả; khác None = tên thư mục dữ liệu tự chấm
AUTHOR_CKPT = AUTHORS / "checkpoints"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / "cache"
LOGS = ROOT / "logs"

# task -> thông tin dataset (tên file RankGPT của tác giả khác nhau theo task)
TASKS = {
    "comagc":  {"gpt_file": "test_gpt_c.jsonl", "ckpt_suffix": "-nl", "desc": "gene-disease (CoMAGC, Hetionet)"},
    "ade":     {"gpt_file": "test_gpt_a.jsonl", "ckpt_suffix": "-nl", "desc": "drug-side effect (ADE, Hetionet)"},
    "gene":    {"gpt_file": "test_gpt_g.jsonl", "ckpt_suffix": "-nl", "desc": "gene-gene (GENEC, Hetionet)"},
    "semeval": {"gpt_file": "test_gpt_s.jsonl", "ckpt_suffix": "",    "desc": "open-domain (SemEval-2010 T8, Wikidata)"},
}
ALL_TASKS = list(TASKS)


def use_data_dir(p) -> Path:
    """Chuyển nguồn dữ liệu Module 1 sang thư mục p (cấu trúc <task>/train_full.jsonl, test_truth.jsonl)."""
    global DATASETS, DATA_NAME
    p = Path(p).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"--data-dir không tồn tại: {p}")
    DATASETS = p
    DATA_NAME = None if p == AUTHORS_DATASETS else p.name
    return p


def apply_data_args(args, dry_run: bool = False) -> Path:
    """Gọi sau parse_args: áp dụng --data-dir / PTC_DATA_DIR và chuẩn hoá args.out (None -> thư mục mặc định)."""
    d = getattr(args, "data_dir", None) or os.environ.get("PTC_DATA_DIR")
    if d:
        use_data_dir(d)
    out = getattr(args, "out", None)
    if out is None:
        out = OUTPUTS if DATA_NAME is None else OUTPUTS / f"data_{DATA_NAME}"
        if dry_run:
            out = out / "_dryrun"
    args.out = str(out)
    args.data_dir = str(DATASETS)
    return Path(args.out)


def task_dir(task: str) -> Path:
    return DATASETS / task


def authors_task_dir(task: str) -> Path:
    """Luôn trỏ vào dữ liệu GỐC của tác giả (các file test_sim / test_gpt chỉ có ở đó)."""
    return AUTHORS_DATASETS / task


def module2_dir(task: str, tag: str, out_root: Path = OUTPUTS) -> Path:
    return Path(out_root) / "module2" / task / tag
