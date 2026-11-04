"""Các 'option chọn subgraph' của Module 3.

Mỗi option -> (template_id, danh sách [paths đã chọn] cho từng cặp trong test_truth).

  no_subgraph        template tca (cặp biến + ngữ cảnh), không metapath
  no_subgraph_pair   template tnb (chỉ cặp biến - đúng bảng bài báo/run_kbcd.sh)
  random[:SEED]      chọn ngẫu nhiên từ pool
  similarity         top-n theo file test_sim.jsonl của tác giả (all-MiniLM-L6-v2, pool riêng của file đó)
  similarity_recompute  tự tính lại cosine all-MiniLM-L6-v2 trên CÙNG pool với các option còn lại
  gpt                top-n theo file RankGPT (gpt-3.5) của tác giả   [chỉ pool oracle_top5]
  gpt_rerank         top-n theo file do tools/rankgpt_rerank.py tạo ra (gọi API của bạn)
  ranker:<tag>       top-n theo ranker của Module 2  (tag: xgboost | ce_rmse | ce_ranknet | ce_listnet | ...)
  authors_ranker:<n> top-n theo file dự đoán tác giả đã công bố (n: xgboost | rmse | ranknet | listnet) [chỉ oracle_top5]
  oracle             sắp theo rel_score (nhìn nhãn thật qua Module 1) - CHỈ là cận trên, không phải phương pháp

pool:  oracle_top5 = 5 path đầu của test_truth (protocol của tác giả; THỰC RA đã được lọc bằng nhãn thật)
       all         = toàn bộ ứng viên (không rò rỉ nhãn)
"""
from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple

from . import data as D
from . import paths

NO_PATH_OPTIONS = {"no_subgraph": "tca", "no_subgraph_pair": "tnb"}


class OptionUnavailable(Exception):
    """Option không chạy được với cấu hình hiện tại (vd pool 'all' với file tác giả)."""


def pool_paths(item: dict, pool: str) -> List[dict]:
    return item["metapaths"][:5] if pool == "oracle_top5" else list(item["metapaths"])


def _align(truth: List[dict], rows: List[dict], name: str) -> List[dict]:
    """rows có thể dài hơn truth khi dùng --limit (chỉ lấy phần đầu); (e1,e2) phải khớp từng dòng."""
    if len(rows) < len(truth):
        raise ValueError(f"[{name}] file có {len(rows)} dòng < {len(truth)} cặp test")
    rows = rows[:len(truth)]
    D.check_aligned(truth, rows, name)
    return rows


def _from_ranked_file(truth: List[dict], rows: List[dict], n: int, name: str) -> List[List[dict]]:
    rows = _align(truth, rows, name)
    return [r["metapaths"][:n] for r in rows]


def _ranked_path_module2(task: str, tag: str, pool: str, out_root, split: str = "test") -> str:
    """split=test: xếp hạng tập test (ranker huấn luyện trên train) | split=full: xếp hạng out-of-fold (--cv-folds) của TẤT CẢ cặp."""
    pre = "test" if split == "test" else "cv"
    fn = f"{pre}_ranked_top5.jsonl" if pool == "oracle_top5" else f"{pre}_ranked_all.jsonl"
    return str(paths.module2_dir(task, tag, out_root) / fn)


def _authors_ckpt_file(task: str, name: str) -> str:
    suf = paths.TASKS[task]["ckpt_suffix"]
    sub = {"xgboost": "xgboost", "rmse": f"roberta-rmse{suf}", "ranknet": f"roberta-rnet{suf}",
           "listnet": f"roberta-lnet{suf}"}.get(name)
    if sub is None:
        raise OptionUnavailable(f"authors_ranker:{name} không hợp lệ (xgboost|rmse|ranknet|listnet)")
    files = glob.glob(str(paths.AUTHOR_CKPT / task / sub / "test_pred_*.jsonl"))
    if not files:
        raise OptionUnavailable(f"không có file dự đoán tác giả cho {task}/{sub}")
    return files[0]


def select_paths(task: str, option: str, truth: List[dict], n: int, pool: str, seed: int,
                 out_root=paths.OUTPUTS, sim_embedder: str = "minilm", split: str = "test") -> Tuple[str, List[List[dict]]]:
    """Trả về (template_id, selected[i] = list metapath cho item i).  split: test | full (tất cả cặp = train + test)."""
    N = len(truth)
    if option in NO_PATH_OPTIONS:
        return NO_PATH_OPTIONS[option], [[] for _ in range(N)]

    name, _, arg = option.partition(":")

    if name == "random":
        s = int(arg) if arg else seed
        sel = []
        for i, it in enumerate(truth):
            cand = pool_paths(it, pool)
            rng = random.Random(f"{s}-{task}-{i}")
            sel.append(rng.sample(cand, min(n, len(cand))))
        return "tcma", sel

    if name == "oracle":
        return "tcma", [pool_paths(it, pool)[:n] for it in truth]

    if name in ("similarity", "gpt", "authors_ranker") and split != "test":
        raise OptionUnavailable(f"file '{name}' của tác giả chỉ có cho tập test (--eval-split test); với --eval-split full dùng "
                                "similarity_recompute / gpt_rerank / ranker:<tag>")

    if name == "similarity":
        rows = _align(truth, D.load_authors_sim(task), "similarity")
        sel = [sorted(r["metapaths"], key=lambda m: -m["sim_score"])[:n] for r in rows]
        return "tcma", sel

    if name == "similarity_recompute":
        from .similarity import rank_by_similarity
        sel = []
        for it in truth:
            ranked = rank_by_similarity(it, pool_paths(it, pool), sim_embedder)
            sel.append(ranked[:n])
        return "tcma", sel

    if name == "gpt":
        if pool != "oracle_top5":
            raise OptionUnavailable("file RankGPT của tác giả chỉ xếp 5 path đầu của test_truth -> chỉ dùng với --pool oracle_top5 "
                                    "(muốn pool 'all' hãy dùng gpt_rerank)")
        return "tcma", _from_ranked_file(truth, D.load_authors_gpt(task), n, "gpt")

    if name == "gpt_rerank":
        sfx = "" if split == "test" else "_full"
        fn = Path(out_root) / "rankgpt" / task / (("ranked_top5" if pool == "oracle_top5" else "ranked_all") + sfx + ".jsonl")
        if not fn.exists():
            raise OptionUnavailable(f"chưa có {fn}; chạy tools/rankgpt_rerank.py trước")
        return "tcma", _from_ranked_file(truth, D.load_jsonl(fn), n, "gpt_rerank")

    if name == "ranker":
        if not arg:
            raise OptionUnavailable("dùng ranker:<tag>, ví dụ ranker:xgboost")
        fn = _ranked_path_module2(task, arg, pool, out_root, split)
        import os
        if not os.path.exists(fn):
            hint = "m2_train_ranker.py --cv-folds 5" if split != "test" else "m2_train_ranker.py"
            raise OptionUnavailable(f"chưa có {fn}; chạy {hint} (tag={arg}) trước")
        return "tcma", _from_ranked_file(truth, D.load_jsonl(fn), n, option)

    if name == "authors_ranker":
        if pool != "oracle_top5":
            raise OptionUnavailable("file tác giả chỉ có protocol oracle_top5")
        return "tcma", _from_ranked_file(truth, D.load_jsonl(_authors_ckpt_file(task, arg)), n, option)

    raise OptionUnavailable(f"option không biết: {option}")


