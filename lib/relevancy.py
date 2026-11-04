"""MODULE 1 - Subgraph Relevancy Estimation: tiện ích chấm điểm metapath bằng LLM (log-prob của nhãn -> rel_score).

Quy trình của bài báo (§4.2 + Appendix C), phần chúng tôi tái lập được:
  1. Với mỗi (cặp biến, MỘT metapath): hỏi LLM `causal` / `non-causal` (prompt tcma với k=1).
  2. metapath 'relevant' (cờ "1") nếu nhãn LLM == nhãn người gán; ngược lại cờ "0".
  3. prob_score = log-prob của token quyết định nhãn  (code tác giả: llm.inference_with_score).
  4. rel_score:
       authors     : +1/|logp| nếu đúng, -1/|logp| nếu sai   <- công thức thực sự có trong dữ liệu đã công bố (12.623/12.623 path)
       eq1_prob    : 1 + p nếu đúng, 1 - p nếu sai, p = exp(logp)       <- Eq.(1) trong bài (p là xác suất)
       eq1_logprob : 1 + logp nếu đúng, 1 - logp nếu sai                  <- Eq.(1) đọc theo chữ ("p là log-prob")
  5. Sắp xếp các metapath của mỗi cặp theo rel_score giảm dần  ->  file train_full.jsonl / test_truth.jsonl.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from .prompts import verbalize_robust

# danh sách token quyết định nhãn - y nguyên code tác giả (src/llm.py)
NEG = ["non", "false", "negative", "no", "neg", "fal"]
POS = ["caus", "aus", "true", "positive", "yes", "pos", "tru"]

LOGP_MODES = ("authors", "first_token", "label_tokens")
FORMULAS = ("authors", "eq1_prob", "eq1_logprob")

_SPECIAL = re.compile(r"<\|[^|>]*\|>|</?s>|\[/?INST\]|<unk>|<pad>")


def _clean(tok: str) -> str:
    return tok.replace("▁", " ").replace("Ġ", " ").strip().lower()


def label_logprob(tokens: Sequence[Sequence], mode: str = "authors") -> Tuple[Optional[float], Optional[str]]:
    """tokens = [(token, logprob), ...] theo thứ tự sinh. Trả (logprob của nhãn, token được dùng).

    authors      : token đầu tiên thuộc NEG/POS (kể cả mảnh subword); không có thì token đầu tiên (như inference_with_score)
    first_token  : log-prob của token đầu tiên không rỗng
    label_tokens : tổng log-prob các token cho tới khi nhãn hoàn chỉnh (= xác suất sinh ra chuỗi nhãn)
    """
    toks = [(t, float(lp)) for t, lp in tokens if _clean(t) != "" and not _SPECIAL.fullmatch(t.strip())]
    if not toks:
        return None, None
    if mode == "first_token":
        return toks[0][1], toks[0][0]
    if mode == "label_tokens":
        text, s = "", 0.0
        for t, lp in toks:
            text += t.replace("▁", " ").replace("Ġ", " ")
            s += lp
            if verbalize_robust(text)[1]:
                return s, text.strip()
        return toks[0][1], toks[0][0]
    if mode != "authors":
        raise ValueError(f"logp mode không hợp lệ: {mode}")
    for t, lp in toks:
        c = _clean(t)
        if c in NEG or c in POS:
            return lp, t
    return toks[0][1], toks[0][0]


def check_logprob_integrity(text: str, tokens: Sequence[Sequence]) -> Optional[str]:
    """None nếu chuỗi token khớp với text; ngược lại trả mô tả vấn đề (một số provider chỉ trả logprob của token dừng)."""
    norm = lambda s: re.sub(r"\s+", "", _SPECIAL.sub("", s.replace("▁", " ").replace("Ġ", " "))).lower()  # noqa: E731
    joined = norm("".join(t for t, _ in tokens))
    want = norm(text)
    if joined == want:
        return None
    if want and (joined.startswith(want) or want.startswith(joined)) and min(len(joined), len(want)) >= 0.8 * max(len(joined), len(want)):
        return None
    return f"ghép token = {joined!r} ≠ text = {want!r}"


def rel_score(correct: bool, logp: float, formula: str = "authors", floor: float = 1e-3) -> float:
    if formula == "authors":
        return (1.0 if correct else -1.0) / max(abs(logp), floor)
    if formula == "eq1_prob":
        p = math.exp(logp)
        return 1.0 + p if correct else 1.0 - p
    if formula == "eq1_logprob":
        return 1.0 + logp if correct else 1.0 - logp
    raise ValueError(f"công thức không hợp lệ: {formula}")


def strip_scores(item: dict) -> dict:
    """Giữ lại ứng viên (pair + metapath) nhưng BỎ MỌI điểm do tác giả chấm (rel_score, prob_score, relevant)."""
    out = {k: v for k, v in item.items() if k not in ("metapaths", "llm_pred")}      # llm_pred = dự đoán cấp cặp của tác giả
    out["metapaths"] = [{k: v for k, v in m.items() if k not in ("rel_score", "prob_score", "relevant", "ranker_score",
                                                                 "sim_score", "rankgpt_raw", "llm_output", "llm_pred", "parsed_ok")}
                        for m in item["metapaths"]]
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Hệ số tương quan hạng (không cần scipy); None nếu < 2 phần tử hoặc hằng."""
    n = len(a)
    if n < 2:
        return None

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return None if da == 0 or db == 0 else num / (da * db)
