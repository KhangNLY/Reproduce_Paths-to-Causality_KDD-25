"""Xuất kết quả một lần chạy demo thành file HTML tĩnh (mở bằng trình duyệt,
không cần Streamlit/Neo4j/Ollama) để chia sẻ: copy được text, xem được đồ thị.

File gồm: header thông tin input, bảng dự đoán (demo 3 chế độ + dự đoán tác giả),
danh sách path sau ranker kèm score, đồ thị PyVis top-k nhúng inline, thống kê pool.
"""
from __future__ import annotations

import datetime
import html as _html
import json


def _esc(v) -> str:
    return _html.escape(str(v))


def _table(rows: list[list], header: bool = True) -> str:
    out = ['<table style="border-collapse:collapse;margin:8px 0;width:100%">']
    for i, row in enumerate(rows):
        out.append("<tr>")
        for cell in row:
            tag = "th" if (header and i == 0) else "td"
            style = ("background:#eef3f7;" if tag == "th" else "") + \
                    "border:1px solid #8fa6b3;padding:6px 10px;text-align:left;vertical-align:top;"
            out.append(f'<{tag} style="{style}">{cell}</{tag}>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def build_report_html(result: dict, compare_results: dict | None = None,
                      author_preds: dict | None = None, graph_html: str = "",
                      pool_graph_html: str = "") -> str:
    """result: dict analyze_pair (hoặc compare_results['paper']); trả chuỗi HTML hoàn chỉnh."""
    e1, e2 = result.get("entity_1") or {}, result.get("entity_2") or {}
    pred = result.get("prediction") or {}
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<title>Demo Paths to Causality — {_esc(e1.get('name','?'))} → {_esc(e2.get('name','?'))}</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1080px;margin:24px auto;padding:0 16px;color:#111}}
h1{{font-size:1.4rem}} h2{{font-size:1.1rem;border-bottom:2px solid #087f78;padding-bottom:4px;margin-top:28px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-weight:600}}
.causal{{background:#d9f2e3;color:#0a6b3d}} .non{{background:#fbe0e0;color:#a12626}} .unc{{background:#fdf0d9;color:#9a6b00}}
.meta{{color:#555;font-size:.9rem}} .muted{{color:#777;font-size:.85rem}}
pre{{background:#f6f8fa;padding:10px;border-radius:8px;white-space:pre-wrap;word-break:break-word}}</style>
</head><body>
<h1>Paths to Causality — kết quả demo</h1>
<p class="meta">Xuất lúc {now} · tự chứa hoàn toàn (đồ thị inline) — mở bằng trình duyệt bất kỳ, copy thoải mái.</p>""")

    # ---- Input ----
    parts.append("<h2>1. Input</h2>")
    parts.append(_table([
        ["Entity 1", f"{_esc(e1.get('name',''))} ({_esc(e1.get('kind',''))}, {_esc(e1.get('id',''))})"],
        ["Entity 2", f"{_esc(e2.get('name',''))} ({_esc(e2.get('kind',''))}, {_esc(e2.get('id',''))})"],
        ["Context (EN)", _esc(result.get("context", ""))],
        ["Context (VI)", _esc(result.get("context_vi", ""))],
    ], header=False))

    # ---- Dự đoán ----
    parts.append("<h2>2. Dự đoán</h2>")
    label = str(pred.get("label", "?"))
    badge = "causal" if label == "causal" else ("non" if label == "non-causal" else "unc")
    parts.append(f'<p>Dự đoán của demo: <span class="badge {badge}">{_esc(label)}</span> '
                 f'· model <code>{_esc(pred.get("model","?"))}</code> · ranking <code>{_esc(result.get("ranker_mode","?"))}</code></p>')
    if compare_results:
        rows = [["Chế độ", "Dự đoán", "Ghi chú ranker"]]
        names = {"paper": "Pipeline paper (KG → ranker → top-k)",
                 "no_ranker": "No-ranker (KG raw → LLM)",
                 "no_kg": "No-KG (chỉ context)"}
        for k, lbl in names.items():
            r = compare_results.get(k)
            if not r:
                continue
            p = r.get("prediction") or {}
            rows.append([_esc(lbl), _esc(p.get("label", "?")), _esc(r.get("ranker_mode", ""))])
        parts.append(_table(rows))
    if author_preds:
        rows = [["Ranker (dự đoán có sẵn của tác giả)", "llm_pred", "Ground truth"]]
        for name, info in author_preds.items():
            if info.get("pred") in ("0", "1"):
                rows.append([_esc(name), _esc("causal" if info["pred"] == "1" else "non-causal"),
                             _esc(info.get("ground_truth", ""))])
        if len(rows) > 1:
            parts.append(_table(rows))
    expl = str(pred.get("explanation", ""))
    if expl:
        parts.append(f"<p><b>Explanation (EN):</b></p><pre>{_esc(expl)}</pre>")
    if result.get("explanation_vi"):
        parts.append(f"<p><b>Giải thích (VI):</b></p><pre>{_esc(result['explanation_vi'])}</pre>")

    # ---- Path sau ranker ----
    ranked = result.get("ranked_paths") or []
    if ranked:
        parts.append(f"<h2>3. Path sau ranker ({len(ranked)} path, sort theo rank_score)</h2>")
        rows = [["#", "rank_score", "hops", "path"]]
        for i, p in enumerate(ranked[:30], start=1):
            src = ""
            if p.get("source") == "paper":
                src = f" <i>(rel_score={float(p.get('rel_score', 0)):.4f})</i>"
            rows.append([str(i), f"{float(p.get('rank_score', 0)):.4f}", str(p.get("hops", "?")),
                         _esc(p.get("path_text") or "") + src])
        parts.append(_table(rows))

    # ---- Đồ thị ----
    if graph_html:
        parts.append("<h2>4. Đồ thị top-k path vào LLM</h2>"
                     '<p class="muted">Mũi tên liền = chiều cạnh Hetionet; đường đỏ nét đứt (nếu có) = dự đoán pair-level causal, chỉ là overlay.</p>')
        parts.append(graph_html)
    if pool_graph_html:
        parts.append('<h2>5. Đồ thị pool metapath (trước ranker)</h2>')
        parts.append(pool_graph_html)

    # ---- Thống kê ----
    stats = result.get("path_stats") or {}
    if stats:
        parts.append("<h2>6. Thống kê pool metapath</h2>")
        rows = [["Tổng số path", str(stats.get("total", result.get("paths_found", 0)))],
                ["Hops (min–max / trung bình)", f"{stats.get('hops_min',0)}–{stats.get('hops_max',0)} / {stats.get('hops_mean',0)}"],
                ["Số loại relation", str(stats.get("relation_types", 0))],
                ["Nguồn path", _esc(result.get("path_source", ""))],
                ["Phân bố theo hops", _esc(json.dumps(stats.get("by_hops", {}), ensure_ascii=False))]]
        parts.append(_table(rows, header=False))

    parts.append('<p class="muted">Nhãn causal/non-causal là dự đoán theo context + đường dẫn KG, không phải bằng chứng khoa học. '
                 'Generated by the self-hosted demo (Streamlit) — Paths to Causality (KDD\'25) reproduction.</p>')
    parts.append("</body></html>")
    return "".join(parts)
