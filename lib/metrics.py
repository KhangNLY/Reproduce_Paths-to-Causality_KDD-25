"""Metric cho Module 2 (xếp hạng) và Module 3 (phân loại causal / non-causal).

Có HAI cách tính NDCG/Recall cho Module 2:

1) `paper_style_ranking_eval`  - mô phỏng CHÍNH XÁC code đánh giá của tác giả (ltr_eval.py + dt_utils.write_file):
     qrels = { pathid : vị_trí_trong_test_truth (1 = path tốt nhất) }  -> độ liên quan GÁN NGƯỢC (path tốt nhất có gain thấp nhất)
     Recall@k = min(k, n)/n  (mọi path trong qrels đều 'relevant') -> KHÔNG phụ thuộc vào ranker
   Dùng để tái lập con số trong Table D2 của bài báo (NDCG@5 ~ 56, Recall@5 ~ 71.43 trên COMAGC).

2) `corrected_ranking_eval`  - phiên bản đã sửa: path tốt nhất theo Module 1 có gain cao nhất,
     Overlap@k (tức Recall@k so với tập top-k của oracle), top-1 hit, vị trí oracle trung bình của top-1.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence

KS_DEFAULT = (1, 3, 5)


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _pathids(item: dict) -> List:
    return [m["pathid"] for m in item["metapaths"]]


def _dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def _align_by_index(truth_items: List[dict], pred_items: List[dict]):
    if len(truth_items) != len(pred_items):
        raise ValueError(f"truth {len(truth_items)} item != pred {len(pred_items)} item")
    for t, p in zip(truth_items, pred_items):
        if (t["e1"], t["e2"]) != (p["e1"], p["e2"]):
            raise ValueError(f"Lệch item: {(t['e1'], t['e2'])} vs {(p['e1'], p['e2'])}")
        if set(_pathids(t)) != set(_pathids(p)):
            raise ValueError(f"Tập pathid khác nhau ở cặp {(t['e1'], t['e2'])}: dùng ranked file cùng pool với test_truth")
        yield t, p


# ---------------------------------------------------------------------------
# (1) Mô phỏng đánh giá của tác giả
# ---------------------------------------------------------------------------
def paper_style_ranking_eval(truth_items: List[dict], pred_items: List[dict], ks: Iterable[int] = KS_DEFAULT) -> Dict[str, float]:
    ks = tuple(ks)
    acc = {f"NDCG@{k}": 0.0 for k in ks}
    acc.update({f"MAP@{k}": 0.0 for k in ks})
    acc.update({f"Recall@{k}": 0.0 for k in ks})
    nq = 0
    for t, p in _align_by_index(truth_items, pred_items):
        tids = _pathids(t)
        n = len(tids)
        qrel = {pid: r + 1 for r, pid in enumerate(tids)}          # <-- gán ngược của tác giả: best = 1
        run = _pathids(p)
        ideal = sorted(qrel.values(), reverse=True)
        gains = [qrel[pid] for pid in run]
        for k in ks:
            idcg = _dcg(ideal, k)
            acc[f"NDCG@{k}"] += _dcg(gains, k) / idcg if idcg > 0 else 0.0
            kk = min(k, n)
            acc[f"MAP@{k}"] += kk / n                                # mọi doc đều relevant
            acc[f"Recall@{k}"] += kk / n
        nq += 1
    return {k: round(v / max(nq, 1), 5) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# (2) Đánh giá đã sửa
# ---------------------------------------------------------------------------
def restrict_pool(item: dict, pool_size: int | None) -> dict:
    if pool_size is None:
        return item
    out = dict(item)
    out["metapaths"] = item["metapaths"][:pool_size]
    return out


def corrected_ranking_eval(truth_items: List[dict], pred_items: List[dict], ks: Iterable[int] = KS_DEFAULT,
                           pool_size: int | None = None) -> Dict[str, float]:
    """truth_items: thứ tự oracle (rel_score giảm dần). pred_items: thứ tự do ranker dự đoán (cùng tập pathid).

    pool_size=5 => chỉ xét 5 path đầu của truth (protocol của tác giả); None => toàn bộ ứng viên.
    """
    ks = tuple(ks)
    acc: Dict[str, float] = {}
    for k in ks:
        acc[f"NDCG@{k}"] = 0.0
        acc[f"Overlap@{k}"] = 0.0
    acc["Top1Hit"] = 0.0
    acc["Top1_LLM_correct"] = 0.0
    acc["MeanOracleRank_Top1"] = 0.0
    nq = 0
    for t, p in _align_by_index(truth_items, pred_items):
        t = restrict_pool(t, pool_size)
        tids = _pathids(t)
        m = len(tids)
        keep = set(tids)
        run = [pid for pid in _pathids(p) if pid in keep]
        gain = {pid: m - r for r, pid in enumerate(tids)}            # best = m, worst = 1
        ideal = sorted(gain.values(), reverse=True)
        gains = [gain[pid] for pid in run]
        for k in ks:
            idcg = _dcg(ideal, k)
            acc[f"NDCG@{k}"] += _dcg(gains, k) / idcg if idcg > 0 else 0.0
            kk = min(k, m)
            acc[f"Overlap@{k}"] += len(set(run[:kk]) & set(tids[:kk])) / kk
        acc["Top1Hit"] += float(run[0] == tids[0])
        flag = {mm["pathid"]: str(mm.get("relevant", "")) for mm in t["metapaths"]}
        acc["Top1_LLM_correct"] += float(flag[run[0]] == "1")
        acc["MeanOracleRank_Top1"] += tids.index(run[0]) + 1
        nq += 1
    return {k: round(v / max(nq, 1), 5) for k, v in acc.items()}


def reference_orderings(truth_items: List[dict], ks: Iterable[int] = KS_DEFAULT, pool_size: int | None = None,
                        n_random: int = 500, seed: int = 0) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Điểm tham chiếu: oracle / reverse-oracle / random (kỳ vọng) dưới cả hai cách tính."""
    rng = random.Random(seed)

    def make(order_fn):
        out = []
        for t in truth_items:
            tt = restrict_pool(t, pool_size)
            q = dict(t)
            q["metapaths"] = order_fn(list(tt["metapaths"])) + list(t["metapaths"][len(tt["metapaths"]):])
            out.append(q)
        return out

    def both(preds):
        # paper-style luôn trên pool 5 đầu nếu pool_size=5 -> dùng truth đầy đủ (giống tác giả)
        return {"paper": paper_style_ranking_eval(truth_items, preds, ks),
                "corrected": corrected_ranking_eval(truth_items, preds, ks, pool_size)}

    res = {"oracle": both(make(lambda x: x)), "reverse_oracle": both(make(lambda x: x[::-1]))}
    sums: Dict[str, Dict[str, float]] = {"paper": {}, "corrected": {}}
    for _ in range(n_random):
        r = both(make(lambda x: rng.sample(x, len(x))))
        for kind in sums:
            for k, v in r[kind].items():
                sums[kind][k] = sums[kind].get(k, 0.0) + v
    res["random_expected"] = {kind: {k: round(v / n_random, 5) for k, v in d.items()} for kind, d in sums.items()}
    return res


# ---------------------------------------------------------------------------
# Phân loại causal / non-causal (Module 3)
# ---------------------------------------------------------------------------
def binary_prf(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {"P": round(100 * prec, 2), "R": round(100 * rec, 2), "F1": round(100 * f1, 2), "Acc": round(100 * acc, 2),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": len(y_true)}



def bootstrap_ci(y_true: Sequence[int], y_pred: Sequence[int], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, list]:
    """Khoảng tin cậy bootstrap (percentile) 95% cho F1 và Accuracy - tập test chỉ 24-38 cặp nên khoảng này RẤT rộng."""
    n = len(y_true)
    if n == 0:
        return {"F1_ci95": [0.0, 0.0], "Acc_ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    yt, yp = list(y_true), list(y_pred)
    f1s, accs = [], []
    for _ in range(n_boot):
        tp = fp = fn = tn = 0
        for _i in range(n):
            j = rng.randrange(n)
            t, p = yt[j], yp[j]
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1
            else:
                tn += 1
        d = 2 * tp + fp + fn
        f1s.append(100.0 * 2 * tp / d if d else 0.0)
        accs.append(100.0 * (tp + tn) / n)
    f1s.sort()
    accs.sort()
    lo, hi = int(alpha / 2 * n_boot), int((1 - alpha / 2) * n_boot) - 1
    return {"F1_ci95": [round(f1s[lo], 2), round(f1s[hi], 2)], "Acc_ci95": [round(accs[lo], 2), round(accs[hi], 2)]}
