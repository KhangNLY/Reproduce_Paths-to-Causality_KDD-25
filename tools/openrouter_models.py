#!/usr/bin/env python
"""Liệt kê model đang được OpenRouter phục vụ (endpoint công khai /models, KHÔNG cần API key).

  python tools/openrouter_models.py                       # lọc sẵn các họ model 7-9B / gần với bài báo
  python tools/openrouter_models.py --grep mistral llama  # lọc theo chuỗi con của id
  python tools/openrouter_models.py --probe meta-llama/llama-3.1-8b-instruct   # (cần OPENROUTER_API_KEY) gọi thử 1 prompt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib  # noqa: E402,F401
from lib import runlog  # noqa: E402  (nạp .env nếu có)

DEFAULT_PATTERN = r"mistral-7b|ministral|llama-3-8b|llama-3\.1-8b|llama-3\.2-(1|3)b|gemma-7b|gemma-2-9b|gemma-3-(4|12)b|qwen.*(7b|8b)|gpt-3\.5|gpt-4o-mini"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grep", nargs="*", default=None, help="các chuỗi con (OR) cần lọc theo id")
    ap.add_argument("--probe", default="", help="id model để gọi thử (tốn vài token)")
    args = ap.parse_args()
    runlog.start("openrouter_models")

    r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    r.raise_for_status()
    models = r.json()["data"]
    pat = re.compile("|".join(map(re.escape, args.grep)) if args.grep else DEFAULT_PATTERN, re.I)
    rows = [m for m in models if pat.search(m["id"])]
    print(f"OpenRouter đang liệt kê {len(models)} model; khớp bộ lọc: {len(rows)}\n")
    print(f"{'id':56s} {'ctx':>8s} {'USD/1M in':>10s} {'USD/1M out':>11s}  logprobs")
    for m in sorted(rows, key=lambda x: x["id"]):
        p = m.get("pricing", {})
        f = lambda v: f"{float(v) * 1e6:.3f}" if v not in (None, "") else "?"  # noqa: E731
        lp = "logprobs" in (m.get("supported_parameters") or [])
        print(f"{m['id']:56s} {str(m.get('context_length', '?')):>8s} {f(p.get('prompt')):>10s} {f(p.get('completion')):>11s}  {'yes' if lp else 'no'}")

    if args.probe:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("Đặt OPENROUTER_API_KEY trước khi dùng --probe")
        rr = requests.post("https://openrouter.ai/api/v1/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": args.probe, "max_tokens": 6, "temperature": 0,
                                 "messages": [{"role": "user", "content": "Is smoking a cause of lung cancer? Answer 'causal' or 'non-causal'.\n[Relation]:"}]},
                           timeout=60)
        print("\nHTTP", rr.status_code)
        print(rr.text[:800])


if __name__ == "__main__":
    main()


