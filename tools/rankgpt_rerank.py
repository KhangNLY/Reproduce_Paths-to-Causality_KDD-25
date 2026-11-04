#!/usr/bin/env python
"""GPT-based ranker (RankGPT, Sun et al. 2023) - tự chạy lại bằng API của BẠN (tuỳ chọn; cần API key).

Tác giả đã công bố sẵn kết quả (datasets/<task>/test_gpt_*.jsonl) -> option `gpt` của Module 3 dùng thẳng file đó, KHÔNG cần API.
Script này dùng khi muốn: (a) tái tạo bằng LLM khác, (b) xếp hạng pool 'all' (không lọc nhãn).

Lưu ý quan trọng: code tác giả đưa path cho GPT theo thứ tự oracle (path tốt nhất đứng đầu) => GPT có thể 'ăn' thứ tự đó.
Mặc định ở đây XÁO NGẪU NHIÊN thứ tự path trước khi đưa cho GPT (--no-shuffle để giống tác giả).

  python tools/rankgpt_rerank.py --dry-run
  $env:OPENAI_API_KEY='sk-...'; python tools/rankgpt_rerank.py --tasks comagc --pool all --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import data as D  # noqa: E402
from lib import paths, runlog  # noqa: E402
from lib.llm import add_llm_args, make_llm  # noqa: E402

SYSTEM = "You are RankGPT, an intelligent assistant that can rank paths based on their relevancy to entity pair."


def conversation(query: str, contents):
    num = len(contents)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"I will provide you with {num} paths, each indicated by number identifier []. \n"
                                        f"Rank the paths based on their usefulness in inferring causal relationships between entity pair: {query}."},
            {"role": "assistant", "content": "Okay, please provide the paths."}]
    for i, c in enumerate(contents, 1):
        msgs.append({"role": "user", "content": f"[{i}] {c.strip()}"})
        msgs.append({"role": "assistant", "content": f"Received passage [{i}]."})
    msgs.append({"role": "user", "content": f"Entity pair: {query}. \nRank the {num} paths above based on their usefulness in inferring "
                                            f"causal relationships between entity pair. The paths should be listed in descending order using "
                                            f"identifiers. The most relevant entity pair should be listed first. The output format should be "
                                            f"[] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."})
    return msgs


def parse_permutation(text: str, n: int):
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    seen, order = set(), []
    for d in digits:
        k = int(d) - 1
        if 0 <= k < n and k not in seen:
            seen.add(k)
            order.append(k)
    return order + [k for k in range(n) if k not in seen]       # path thiếu -> giữ vị trí cuối (như tác giả)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=paths.ALL_TASKS, choices=paths.ALL_TASKS)
    ap.add_argument("--pool", choices=["oracle_top5", "all"], default="all")
    ap.add_argument("--no-shuffle", action="store_true", help="giữ thứ tự oracle khi đưa cho GPT (như tác giả)")
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-split", choices=["test", "full"], default="test", help="test = tập test | full = tất cả cặp (train+test) cho m3 --eval-split full")
    ap.add_argument("--data-dir", default=None, help="dùng dữ liệu Module 1 tự chấm thay cho dữ liệu tác giả")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    add_llm_args(ap)
    ap.set_defaults(model="gpt-4o-mini")
    args = ap.parse_args()
    runlog.start("rankgpt_rerank")
    if args.dry_run:
        args.backend, args.limit = "mock", args.limit or 3
    paths.apply_data_args(args, dry_run=args.dry_run)
    if args.backend == "hf":
        ap.error("RankGPT cần chat nhiều lượt: dùng --backend openai (hoặc mock để thử)")
    llm = make_llm(args)
    for t in args.tasks:
        truth = D.load_test_truth(t) if args.eval_split == "test" else D.load_full(t)
        truth = truth[:args.limit] if args.limit else truth
        convs, plans = [], []
        for i, it in enumerate(truth):
            k = 5 if args.pool == "oracle_top5" else len(it["metapaths"])
            pool = list(range(min(k, len(it["metapaths"]))))
            shown = pool[:] if args.no_shuffle else random.Random(f"{args.seed}-{t}-{i}").sample(pool, len(pool))
            convs.append(conversation(D.query_text(it), [str(it["metapaths"][j]["stops"]) for j in shown]))
            plans.append((pool, shown))
        print(f"[{t}] gọi LLM {len(convs)} lần (pool={args.pool}, shuffle={not args.no_shuffle}) ...")
        outs = llm.chat_many(convs, max_tokens=100)
        rows_top5, rows_all = [], []
        for it, (pool, shown), o in zip(truth, plans, outs):
            order = parse_permutation(o, len(shown))
            ranked = [shown[j] for j in order]
            rest = list(range(len(pool), len(it["metapaths"])))
            new = copy.deepcopy(it)
            new["metapaths"] = [dict(it["metapaths"][j], rankgpt_raw=o) for j in ranked + rest]
            (rows_top5 if args.pool == "oracle_top5" else rows_all).append(new)
        sfx = "" if args.eval_split == "test" else "_full"
        name = ("ranked_top5" if args.pool == "oracle_top5" else "ranked_all") + sfx + ".jsonl"
        dest = Path(args.out) / "rankgpt" / t / name
        D.write_jsonl(runlog.stamped(dest), rows_top5 or rows_all)               # bản có timestamp (không bị ghi đè)
        D.write_jsonl(dest, rows_top5 or rows_all)                                # bản mới nhất (m3 option gpt_rerank đọc)
        print(f"  -> {dest}  (và {runlog.stamped(dest).name})")


if __name__ == "__main__":
    main()

