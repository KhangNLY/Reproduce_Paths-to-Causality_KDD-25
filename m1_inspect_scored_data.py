#!/usr/bin/env python
"""MODULE 1 - Subgraph Relevancy Estimation (chế độ KIỂM ĐỊNH): dùng TRỰC TIẾP dữ liệu đã chấm điểm (không gọi LLM).

Script này KHÔNG chấm điểm lại. Nó nạp các file *.jsonl (tác giả gọi là 'json') trong data/authors/datasets (hoặc --data-dir),
kiểm định chúng và in báo cáo: kích thước, công thức rel_score thực tế, thứ tự, và các điểm cần lưu ý
(rò rỉ nhãn qua pool 'oracle_top5', công thức Eq.(1) trong bài khác dữ liệu...).
Muốn TỰ chấm điểm metapath bằng LLM của bạn (không dùng điểm của tác giả): xem m1_score_metapaths.py.

  python m1_inspect_scored_data.py
  python m1_inspect_scored_data.py --data-dir outputs/module1/rescored/<run>   # kiểm định dữ liệu tự chấm
  python m1_inspect_scored_data.py --verify-sim        # cần sentence-transformers: kiểm chứng baseline similarity
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lib import data as D
from lib import metrics as M
from lib import paths, runlog


def audit_task(task: str):
    tr, te = D.load_train(task), D.load_test_truth(task)
    r = {"task": task, "desc": paths.TASKS[task]["desc"]}

    def stats(items, name):
        n = [len(x["metapaths"]) for x in items]
        r[name] = {"pairs": len(items), "paths_total": sum(n), "paths_per_pair_mean": round(statistics.fmean(n), 2),
                   "paths_per_pair_min_max": [min(n), max(n)],
                   "causal_rate": round(statistics.fmean(int(x["ground_truth"]) for x in items), 3)}
    stats(tr, "train"); stats(te, "test_truth")

    # công thức rel_score
    for name, items in (("train", tr), ("test_truth", te)):
        tot = a = b = 0
        for x in items:
            for m in x["metapaths"]:
                lp, rs, rel = m["prob_score"], m["rel_score"], str(m["relevant"])
                f_real = (1 / abs(lp)) if rel == "1" else (-1 / abs(lp))      # công thức thực tế trong dữ liệu
                f_eq1 = (1 + lp) if rel == "1" else (1 - lp)                  # Eq.(1) trong bài (p = log-prob)
                tot += 1
                a += abs(f_real - rs) <= 1e-2 * max(1.0, abs(rs))
                b += abs(f_eq1 - rs) < 1e-3
        r[f"{name}_formula"] = {"paths": tot, "match_pm_1_over_abs_logp": a, "match_eq1_in_paper": b}
    r["sorted_desc_by_rel_score"] = all(all(p["rel_score"] >= q["rel_score"] for p, q in zip(x["metapaths"], x["metapaths"][1:]))
                                        for x in tr + te)
    r["items_with_mixed_relevant_flags"] = {"train": sum(1 for x in tr if len({str(m["relevant"]) for m in x["metapaths"]}) > 1),
                                            "test": sum(1 for x in te if len({str(m["relevant"]) for m in x["metapaths"]}) > 1)}

    # pool oracle_top5 có bị lọc bằng nhãn không?
    kept, dropped = [], []
    for x in te:
        kept += [str(m["relevant"]) == "1" for m in x["metapaths"][:5]]
        dropped += [str(m["relevant"]) == "1" for m in x["metapaths"][5:]]
    r["pool_oracle_top5"] = {"paths_dropped": len(dropped), "P(LLM_correct|kept)": round(statistics.fmean(kept), 3),
                             "P(LLM_correct|dropped)": round(statistics.fmean(dropped), 3) if dropped else None}

    # pool của file similarity khác pool của test_truth
    try:      # chỉ so được khi dữ liệu đang kiểm định là dữ liệu tác giả (file sim/gpt của tác giả cùng pool)
        sim, gpt = D.load_authors_sim(task), D.load_authors_gpt(task)
        ov = [len({m["stops"] for m in a["metapaths"]} & {m["stops"] for m in b["metapaths"]}) / max(1, len(a["metapaths"]))
              for a, b in zip(te, sim)]
        r["sim_file_pool_overlap_with_test_truth"] = round(statistics.fmean(ov), 3)
        r["gpt_file_same_pool_as_test_truth"] = all({m["pathid"] for m in a["metapaths"]} == {m["pathid"] for m in b["metapaths"]}
                                                    for a, b in zip(te, gpt))
    except Exception as e:     # noqa: BLE001
        r["sim_file_pool_overlap_with_test_truth"] = None
        r["gpt_file_same_pool_as_test_truth"] = f"n/a ({type(e).__name__})"

    # tham chiếu NDCG: oracle / reverse / random dưới 2 cách tính
    r["reference_orderings"] = M.reference_orderings(te, pool_size=5, n_random=300)
    return r


def to_markdown(reports) -> str:
    L = ["# Module 1 - báo cáo kiểm định dữ liệu đã chấm điểm\n"]
    L.append("| task | train pairs | test pairs | paths/pair (train/test) | causal rate test | rel_score = ±1/\\|logp\\| | khớp Eq.(1) bài báo | sorted | mixed-flag test pairs |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        f = r["train_formula"]; g = r["test_truth_formula"]
        L.append(f"| {r['task']} | {r['train']['pairs']} | {r['test_truth']['pairs']} | {r['train']['paths_per_pair_mean']}/{r['test_truth']['paths_per_pair_mean']} | "
                 f"{r['test_truth']['causal_rate']} | {f['match_pm_1_over_abs_logp'] + g['match_pm_1_over_abs_logp']}/{f['paths'] + g['paths']} | "
                 f"{f['match_eq1_in_paper'] + g['match_eq1_in_paper']} | {r['sorted_desc_by_rel_score']} | {r['items_with_mixed_relevant_flags']['test']} |")
    L.append("\n## Pool `oracle_top5` có bị lọc bằng nhãn?\n")
    L.append("| task | paths bị loại | P(LLM đúng \\| path được giữ) | P(LLM đúng \\| path bị loại) | overlap pool file similarity vs test_truth |")
    L.append("|---|---|---|---|---|")
    for r in reports:
        p = r["pool_oracle_top5"]
        L.append(f"| {r['task']} | {p['paths_dropped']} | {p['P(LLM_correct|kept)']} | {p['P(LLM_correct|dropped)']} | {r.get('sim_file_pool_overlap_with_test_truth')} |")
    L.append("\n## Điểm tham chiếu NDCG@5 (pool 5 path đầu của test_truth)\n")
    L.append("| task | cách tính | oracle | reverse-oracle | random (kỳ vọng) |")
    L.append("|---|---|---|---|---|")
    for r in reports:
        ref = r["reference_orderings"]
        L.append(f"| {r['task']} | paper-style (tác giả, gán ngược) | {ref['oracle']['paper']['NDCG@5']} | {ref['reverse_oracle']['paper']['NDCG@5']} | {ref['random_expected']['paper']['NDCG@5']} |")
        L.append(f"| {r['task']} | corrected | {ref['oracle']['corrected']['NDCG@5']} | {ref['reverse_oracle']['corrected']['NDCG@5']} | {ref['random_expected']['corrected']['NDCG@5']} |")
    return "\n".join(L) + "\n"


def verify_sim(tasks, k=15):
    from lib.similarity import score_paths
    print("\n== Kiểm chứng baseline similarity (recompute MiniLM vs sim_score tác giả) ==")
    for t in tasks:
        errs = []
        for it in D.load_authors_sim(t)[:k]:
            mine = score_paths(it, it["metapaths"])
            errs.append(max(abs(a - b["sim_score"]) for a, b in zip(mine, it["metapaths"])))
        print(f"  {t:8s} max|sai số| = {max(errs):.4f}  ({'KHỚP' if max(errs) < 1e-3 else 'KHÔNG khớp - công thức tác giả khác'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=paths.ALL_TASKS, choices=paths.ALL_TASKS)
    ap.add_argument("--verify-sim", action="store_true")
    ap.add_argument("--out", default=None, help="thư mục gốc kết quả (mặc định outputs/; ghi vào <out>/module1/)")
    ap.add_argument("--data-dir", default=None, help="kiểm định dữ liệu tự chấm (m1_score_metapaths.py) thay cho dữ liệu tác giả")
    ap.add_argument("--show-example", action="store_true", help="in 1 bản ghi mẫu")
    args = ap.parse_args()
    runlog.start("m1_inspect_scored_data")
    paths.apply_data_args(args)
    print(f"## Dữ liệu kiểm định: {args.data_dir}")
    reports = [audit_task(t) for t in args.tasks]
    out = Path(args.out) / "module1"
    out.mkdir(parents=True, exist_ok=True)
    ts = runlog.now_ts()
    js = json.dumps({"timestamp": runlog.iso_now(), "data_dir": args.data_dir, "reports": reports}, indent=1, ensure_ascii=False)
    md = to_markdown(reports) + f"\n_Dữ liệu: `{args.data_dir}` · thời điểm: {runlog.iso_now()}_\n"
    for suffix in ("", f"_{ts}"):                                       # bản mới nhất + bản có timestamp
        (out / f"report{suffix}.json").write_text(js, encoding="utf-8")
        (out / f"report{suffix}.md").write_text(md, encoding="utf-8")
    print(md)
    if args.show_example:
        ex = D.load_test_truth(args.tasks[0])[0]
        print(json.dumps(ex, indent=1, ensure_ascii=False)[:2500])
    if args.verify_sim:
        verify_sim(args.tasks)
    print(f"\nĐã ghi {out / ('report_' + ts + '.md')} (và report.md = bản mới nhất; file .json tương ứng)")


if __name__ == "__main__":
    main()
