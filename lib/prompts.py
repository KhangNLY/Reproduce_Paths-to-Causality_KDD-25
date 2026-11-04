"""Prompt zero-shot - sao chép đúng template của tác giả (src/zero_templatizer.py) + verbalizer.

Template:
  tnb  = chỉ cặp biến                       ('no-subgraph' trong bài: "only the variable pair")
  tca  = cặp biến + câu ngữ cảnh            (đúng mô tả 'no-subgraph' của đề bài: cặp biến + ngữ cảnh văn bản)
  tcma = cặp biến + ngữ cảnh + top-k metapath (phương pháp đề xuất; k=1 cho kết quả tốt nhất theo bài báo)
"""
from __future__ import annotations

import re
from typing import List

from . import data as D

_REMOVE = {"<e1>", "</e1>", "<e2>", "</e2>"}

PREFIX = {
    "comagc": "Given the following information, classify the relation between the gene and disease pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n",
    "ade": "Given the following information, classify the relation between the drug and side effect pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n",
    "gene": "Given the following information, classify the relation between the gene pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'.\n",
    "semeval": "Given the following information, classify the relation between the entity pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n",
}

# --- Biến thể câu chỉ dẫn (kiểm tra độ nhạy với prompt; biến thể 0 = câu gốc của tác giả) -------------------------------
PAIR_DESC = {"comagc": "gene and disease pair", "ade": "drug and side effect pair", "gene": "gene pair", "semeval": "entity pair"}
INSTRUCTION_VARIANTS = {
    1: "Based on the information below, decide whether the {pair_desc} is in a cause-effect relationship. Answer 'causal' if there is a cause-effect relationship, otherwise answer 'non-causal'.\n",
    2: "Read the following information about a {pair_desc} and determine whether a causal relationship exists between its two elements. Respond with 'causal' or 'non-causal'.\n",
    3: "Is there a cause-effect relationship between the two elements of the {pair_desc}? Use the information below and reply with 'causal' or 'non-causal'.\n",
    4: "Task: causal relation classification for the {pair_desc}. Use the information below. Output 'causal' if there is a cause-effect relationship; output 'non-causal' otherwise.\n",
}
N_VARIANTS = 1 + len(INSTRUCTION_VARIANTS)

# Khi gọi API chat (OpenRouter...) thay vì prompt thô: ép model trả đúng 1 từ (xem README, mục Deviations)
CHAT_SYSTEM_PROMPT = "You are a strict classifier. Reply with exactly one word: causal or non-causal. Do not explain."
CHAT_ANSWER_SUFFIX = "\nAnswer with only one word: 'causal' or 'non-causal'."


def instruction_prefix(task: str, variant: int = 0) -> str:
    if variant == 0:
        return PREFIX[task]
    if variant not in INSTRUCTION_VARIANTS:
        raise ValueError(f"prompt variant {variant} không có (0..{N_VARIANTS - 1})")
    return INSTRUCTION_VARIANTS[variant].format(pair_desc=PAIR_DESC[task])


def clean_sentence(s: str) -> str:
    return " ".join(x for x in s.split() if x not in _REMOVE)


def build_prompt(item: dict, paths: List[dict], template: str, task: str, style: str = "mistral",
                 path_format: str = "stops", variant: int = 0) -> str:
    """style: mistral | llama | gemma  (chỉ khác câu kết thúc prompt, như trong tác giả). variant: biến thể câu chỉ dẫn (0 = gốc)."""
    prefix = instruction_prefix(task, variant)
    pair = f"[Pair]: {item['e1']} and {item['e2']}\n"
    if template == "tnb":
        return f"{prefix}{pair}The relation between the pair is"
    sentence = f"[Context sentence]: {clean_sentence(item['sentence'])}\n"
    if template == "tca":
        return f"{prefix}{pair}{sentence}[Relation]:"
    if template == "tcma":
        fmt = D.typed_path_text if path_format == "typed" else (lambda m: str(m["stops"]))
        mpath_context = "[Relation paths]: " + "".join(fmt(p) + ". " for p in paths) + "\n"
        if style == "gemma":
            tail = f"The relation between {item['e1']} and {item['e2']} is"
        elif style == "llama":
            tail = f"The relation between {item['e1']} and {item['e2']} is:"
        else:
            tail = "[Relation]:"
        return f"{prefix}{pair}{sentence}{mpath_context}{tail}"
    raise ValueError(f"template không hợp lệ: {template}")


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------
def verbalize_authors(text: str) -> int:
    """Y NGUYÊN verbalizer của tác giả (utils.verbalizer): kiểm tra chuỗi con 'no' TRƯỚC.

    Hạn chế: 'causal. Knowledge' chứa 'no' (trong 'knowledge') -> bị gán non-causal.
    """
    t = text.lower()
    if any(v in t for v in ["non-cau", "false", "negative", "no"]):
        return 0
    if any(v in t for v in ["causal", "true", "positive", "yes"]):
        return 1
    return 0


_NEG = re.compile(r"non[\s\-_]?causal|not[\s\-]+causal|no[\s\-]+causal|\bno\b|\bfalse\b|\bnegative\b|\bnot\b")
_POS = re.compile(r"causal|\btrue\b|\bpositive\b|\byes\b")


def verbalize_robust(text: str):
    """Lấy nhãn xuất hiện SỚM NHẤT trong câu trả lời. Trả (label, parsed_ok)."""
    t = text.lower()
    n = _NEG.search(t)
    p = _POS.search(t)
    if n and (not p or n.start() <= p.start()):
        return 0, True
    if p:
        return 1, True
    return 0, False

