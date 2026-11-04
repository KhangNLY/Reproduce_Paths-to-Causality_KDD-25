#!/usr/bin/env python
"""So sánh kết quả của project này với kết quả công bố trong bài báo (Table 1, 2, D2 + SACHS) -> outputs/COMPARISON.md.

Số liệu của bài nằm ở docs/paper_results.json (chép tự động từ bảng trong PDF). Số liệu của project lấy từ
outputs/module3/<run>/summary.csv và outputs/module2/<task>/<ranker>/metrics.json.

  python tools/compare_with_paper.py
  python tools/compare_with_paper.py --protocol all          # so với protocol không rò rỉ nhãn (bài báo không có số này)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import paths, runlog  # noqa: E402

PAPER = paths.ROOT / "docs" / "paper_results.json"
DS = [("comagc", "comagc", "COMAGC"), ("ade", "ade", "ADE"), ("genec", "gene", "GENEC"), ("semeval", "semeval", "SEMEVAL")]  # (paper, ours, label)
# protocol oracle_top5 (giống bài): no-subgraph = chỉ cặp biến; similarity/gpt = file của tác giả.
# protocol all (không rò rỉ nhãn): không có file tác giả -> no-subgraph = cặp + ngữ cảnh; similarity tự tính; gpt = RankGPT chạy lại.
ALT = {"all": {"no-subgraph": ["no_subgraph"], "similarity": ["similarity_recompute"], "gpt": ["gpt_rerank"]}}
METHODS_BASE = [("no-subgraph", ["no_subgraph_pair"], "no-subgraph (pair only)"),
           ("random", ["random:"], "random"),
           ("similarity", ["similarity"], "similarity (MiniLM)"),
           ("gpt", ["gpt"], "GPT-based (RankGPT)"),
           ("rmse", ["ranker:ce_rmse"], "Ranker: RoBERTa-RMSE"),
           ("ranknet", ["ranker:ce_ranknet"], "Ranker: RoBERTa-RankNet"),
           ("listnet", ["ranker:ce_listnet"], "Ranker: RoBERTa-ListNet"),
           ("xgboost", ["ranker:xgboost"], "Ranker: XGBoost")]
RANKERS = ["rmse", "ranknet", "listnet", "xgboost"]


def load_runs(root: Path, pool: str):
    """-> {llm: {task: {option: F1}}} (random = trung bình các seed). Chỉ lấy run chat/completion đầy đủ 4 task của pool đó."""
    out = {}
    for f in sorted((root / "module3").glob(f"or_*_{pool}_k1/summary.csv")):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if not rows or rows[0].get("eval_split", "test") != "test":
            continue
        llm = rows[0]["model"]
        d = out.setdefault(llm, {})
        for r in rows:
            if "mean" in r["option"] or r.get("variant", "0") not in ("0", ""):
                continue
            d.setdefault(r["task"], {}).setdefault(r["option"], float(r["F1"]))
    return out


def ours_f1(runs, llm, task, keys):
    d = runs[llm].get(task, {})
    for k in keys:
        if k.endswith(":") and k.startswith("random"):
            v = [x for o, x in d.items() if o.startswith("random:")]
            if v:
                return statistics.fmean(v)
        elif k in d:
            return d[k]
    return None


def paper_f1(P, dsname, method):
    tab = P["table2_semeval_wikidata"] if dsname == "semeval" else P["table1_biomedical_hetionet"][dsname]
    return {llm: rows[method][2] for llm, rows in tab.items()}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def claims(cells):
    """cells[(ds, llm)][method] = F1 -> các 'claim' của bài, đo bằng chênh lệch F1 trung bình."""
    def avg(fn):
        v = [fn(c) for c in cells.values()]
        v = [x for x in v if x is not None]
        return statistics.fmean(v) if v else None

    def diff(a, b):
        return lambda c: (a(c) - b(c)) if a(c) is not None and b(c) is not None else None

    best = lambda c: max((c[m] for m in RANKERS if c.get(m) is not None), default=None)  # noqa: E731
    g = lambda m: (lambda c: c.get(m))  # noqa: E731
    rk_mean = lambda c: mean([c.get(m) for m in RANKERS])  # noqa: E731
    ce_mean = lambda c: mean([c.get("rmse"), c.get("ranknet"), c.get("listnet")])  # noqa: E731
    wins = lambda fn: (sum(1 for c in cells.values() if fn(c) is not None and fn(c) > 0), sum(1 for c in cells.values() if fn(c) is not None))  # noqa: E731
    rows = [
        ("C1  random subgraph − no-subgraph", diff(g("random"), g("no-subgraph"))),
        ("C2  best Subgraph Ranker − no-subgraph", diff(best, g("no-subgraph"))),
        ("C3  best Subgraph Ranker − random", diff(best, g("random"))),
        ("C4  best Subgraph Ranker − similarity", diff(best, g("similarity"))),
        ("C3b mean of 4 Subgraph Rankers − random (no max-over-4 bias)", diff(rk_mean, g("random"))),
        ("C4b mean of 4 Subgraph Rankers − similarity (no max-over-4 bias)", diff(rk_mean, g("similarity"))),
        ("C5  best Subgraph Ranker − GPT-based", diff(best, g("gpt"))),
        ("C6  RankNet − RMSE (pairwise vs pointwise)", diff(g("ranknet"), g("rmse"))),
        ("C7  RMSE − ListNet (pointwise vs listwise)", diff(g("rmse"), g("listnet"))),
        ("C8  mean(cross-encoders) − XGBoost", diff(ce_mean, g("xgboost"))),
    ]
    return [(name, avg(fn), wins(fn)) for name, fn in rows]


METHODS = METHODS_BASE


def build(root, protocol):
    """Đọc outputs/ + docs/paper_results.json -> dict {text, summary(list dòng bảng), claims(list dòng), d2(list dòng), llms}."""
    global METHODS
    METHODS = [(pm, ALT.get(protocol, {}).get(pm, ok), ml.replace("pair only", "pair + context") if protocol == "all" else ml)
               for pm, ok, ml in METHODS_BASE]
    root = Path(root)
    P = json.loads(PAPER.read_text(encoding="utf-8"))
    runs = load_runs(root, protocol)
    if not runs:
        raise SystemExit(f"Chưa có kết quả Module 3 (protocol {protocol}) trong {root}/module3")
    llms = sorted(runs)
    short = {m: m.split("/")[-1] for m in llms}
    L = [f"# Kết quả của project vs bài báo (F1 %, protocol `{protocol}`)", "",
         f"_tạo bởi tools/compare_with_paper.py lúc {runlog.iso_now()}_", "",
         f"* Bài báo: mistral-7b-instruct, llama-3-8b-instruct, gemma-7b-it (chạy local, prompt thô, greedy, 6 token). Project: {', '.join(short.values())} (OpenRouter chat + chỉ dẫn 1 từ).",
         ("* **Protocol `all`** (không rò rỉ nhãn) không có số tương ứng trong bài: cột `paper` là số của bài (protocol oracle_top5) chỉ để tham chiếu, Δ không phải so sánh cùng điều kiện." if protocol == "all" else "* Protocol `oracle_top5` = giao thức của bài (ứng viên đã lọc bằng nhãn)."),
         "* `paper(avg)` = trung bình 3 LLM của bài; `ours(avg)` = trung bình các LLM của project. **Không kỳ vọng khớp từng số** (khác LLM, khác cách gọi, test chỉ 24-38 cặp).", ""]
    paper_cells, our_cells = {}, {}
    insert_at = len(L)
    for pds, ods, label in DS:
        L += [f"## {label}", "", "| method | " + " | ".join(f"paper {m.split('-')[0]}" for m in ("mistral-7b", "llama-3-8b", "gemma-7b")) + " | **paper(avg)** | "
              + " | ".join(f"ours {short[m]}" for m in llms) + " | **ours(avg)** | Δ(ours−paper) |", "|" + "---|" * (6 + len(llms) + 1)]
        for pm, okeys, mlabel in METHODS:
            pf = paper_f1(P, pds, pm)
            pv = [pf.get(n) for n in ("mistral-7b-instruct", "llama-3-8b-instruct", "gemma-7b-it")]
            ov = [ours_f1(runs, m, ods, okeys) for m in llms]
            for n, v in zip(("mistral-7b-instruct", "llama-3-8b-instruct", "gemma-7b-it"), pv):
                paper_cells.setdefault((pds, n), {})[pm] = v
            for m, v in zip(llms, ov):
                our_cells.setdefault((pds, m), {})[pm] = v
            pa, oa = mean(pv), mean(ov)
            L.append(f"| {mlabel} | " + " | ".join(fmt(v) for v in pv) + f" | **{fmt(pa)}** | " + " | ".join(fmt(v) for v in ov)
                     + f" | **{fmt(oa)}** | {('%+.1f' % (oa - pa)) if pa is not None and oa is not None else '-'} |")
        L.append("")

    # ---- bảng tóm tắt: trung bình F1 theo (dataset, method) -------------------------------------------------------------
    S = ["## Tóm tắt: F1 trung bình theo LLM (paper = 3 LLM, ours = %d LLM)" % len(llms), "",
         "| method | " + " | ".join(f"{lab} paper / ours" for _, _, lab in DS) + " | **avg paper / ours** |", "|" + "---|" * (len(DS) + 2)]
    for pm, okeys, mlabel in METHODS:
        cells, pa_all, oa_all = [], [], []
        for pds, ods, label in DS:
            pa = mean([v.get(pm) for (d, _), v in paper_cells.items() if d == pds])
            oa = mean([v.get(pm) for (d, _), v in our_cells.items() if d == pds])
            pa_all.append(pa)
            oa_all.append(oa)
            cells.append(f"{fmt(pa)} / {fmt(oa)}")
        S.append(f"| {mlabel} | " + " | ".join(cells) + f" | **{fmt(mean(pa_all))} / {fmt(mean(oa_all))}** |")
    S.append("")
    L[insert_at:insert_at] = S
    L += ["## Các kết luận chính của bài - có tái lập được không?", "",
          "Chênh lệch F1 trung bình trên các ô (dataset × LLM); `thắng` = số ô có chênh lệch > 0 / tổng số ô. "
          "Các dòng 'best Subgraph Ranker' lấy max của 4 ranker nên lạc quan (thiên lệch chọn lọc); dòng 'C3b/C4b' dùng trung bình 4 ranker để đối chứng.", "",
          "| claim | paper: Δ F1 | paper: thắng | ours: Δ F1 | ours: thắng |", "|---|---|---|---|---|"]
    cp, co = claims(paper_cells), claims(our_cells)
    claim_rows = [f"| {name} | {fmt(pa, 2)} | {pw[0]}/{pw[1]} | {fmt(oa, 2)} | {ow[0]}/{ow[1]} |" for (name, pa, pw), (_, oa, ow) in zip(cp, co)]
    L += claim_rows
    d2_start = None
    L += ["", "## Module 2 - Subgraph Ranker (Table D2: NDCG@5 / Recall@5 của RoBERTa-base)", "",
          "`ours (paper-style)` dùng đúng cách tính NDCG của tác giả (độ liên quan gán ngược, xem README); `ours (corrected)` là cách tính đã sửa.", "",
          "| dataset | loss | paper NDCG@5 | ours paper-style | ours corrected (pool 5) | random corrected | paper Recall@5 | ours Recall@5 |", "|---|---|---|---|---|---|---|---|"]
    d2_start = len(L)
    for pds, ods, label in DS:
        for loss in ("rmse", "ranknet", "listnet"):
            f = root / "module2" / ods / f"ce_{loss}" / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            r = m["results"]
            L.append(f"| {label} | {loss} | {P['tableD2_ranker_eval_roberta_base'][pds]['NDCG@5'][loss]:.2f} | {100 * r['paper_style_top5']['NDCG@5']:.2f} | "
                     f"{100 * r['corrected_top5']['NDCG@5']:.2f} | {100 * m['reference']['random_expected']['corrected']['NDCG@5']:.2f} | "
                     f"{P['tableD2_ranker_eval_roberta_base'][pds]['Recall@5']:.2f} | {100 * r['paper_style_top5']['Recall@5']:.2f} |")
    d2_rows = L[d2_start:]
    L += ["", "## Chưa tái lập", "",
          "* **Table 3 (SACHS vs PC / A* / DirectLiNGAM)**: cần dữ liệu quan sát SACHS + metapath Hetionet cho 11 protein; repo tác giả không có dữ liệu/mã cho phần này.",
          f"  Số của bài (không chạy lại): no-subgraph F1={P['table3_sachs']['no-subgraph']['F1']}, RMSE={P['table3_sachs']['rmse']['F1']}, RankNet={P['table3_sachs']['ranknet']['F1']}, "
          f"ListNet={P['table3_sachs']['listnet']['F1']}, XGBoost={P['table3_sachs']['xgboost']['F1']}; PC={P['table3_sachs']['PC']['F1']}, A*={P['table3_sachs']['A*']['F1']}, DirectLiNGAM={P['table3_sachs']['DirectLiNGAM']['F1']}.",
          "* **Trích metapath từ Hetionet (Neo4j) / Wikidata (CLOCQ)** và **chấm điểm Module 1 bằng mistral-7b-instruct-v0.1 local**: mã không công bố; project dùng ứng viên đã trích của tác giả và có `m1_score_metapaths.py` để tự chấm lại bằng LLM khác."]
    text = "\n".join(L) + "\n"
    return {"text": text, "summary": S[2:-1], "claims": claim_rows, "d2": d2_rows, "llms": llms, "protocol": protocol}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(paths.OUTPUTS))
    ap.add_argument("--protocol", choices=["oracle_top5", "all"], default="oracle_top5")
    args = ap.parse_args()
    runlog.start("compare_with_paper")
    root = Path(args.root)
    text = build(root, args.protocol)["text"]
    out = root / ("COMPARISON.md" if args.protocol == "oracle_top5" else "COMPARISON_all.md")
    out.write_text(text, encoding="utf-8")
    runlog.stamped(out).write_text(text, encoding="utf-8")
    snap = paths.ROOT / "docs" / "results"                      # bản chụp được commit cùng README (outputs/ bị git bỏ qua)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / out.name).write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out} (và {runlog.stamped(out).name})")


if __name__ == "__main__":
    main()
