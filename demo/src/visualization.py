"""Graph payload preparation and PyVis HTML rendering for ranked KG paths.

Kept separate from Streamlit so graph construction can be unit-tested without
requiring a running Neo4j/Ollama stack.
"""
from __future__ import annotations
from typing import Any
from src.features import NODE_KIND_LABELS_VI
from src.relation_labels import relation_description, relation_verb_vi

NODE_COLORS = {
    "Gene": "#7db7ff",
    "Disease": "#ff9999",
    "Compound": "#96d39b",
    "Anatomy": "#e7c66b",
    "Biological Process": "#c4a7e7",
    "Cellular Component": "#c4a7e7",
    "Molecular Function": "#f4b183",
    "Pathway": "#8dd3c7",
    "Side Effect": "#f4b7c3",
    "Symptom": "#f4b7c3",
    "Pharmacologic Class": "#b8c4e8",
}


def build_graph_payload(
    paths: list[dict[str, Any]],
    entity_1: dict[str, Any],
    entity_2: dict[str, Any],
    prediction_label: str,
) -> dict[str, Any]:
    """Deduplicate the selected path subgraph while preserving native KG edge direction.

    `path_edges[].source/target` represent the real stored Hetionet direction, not
    necessarily the order in which an undirected shortest path traversed them.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, str]] = {}

    # Include endpoints even if no path exists, so the UI can show a meaningful
    # empty/disconnected graph rather than hiding the visualization altogether.
    for role, endpoint in (("Entity 1", entity_1), ("Entity 2", entity_2)):
        if endpoint and endpoint.get("id") is not None:
            nid = str(endpoint["id"])
            nodes[nid] = {
                "id": nid,
                "name": str(endpoint.get("name") or nid),
                "kind": str(endpoint.get("kind") or "Unknown"),
                "endpoint_role": role,
            }

    for path in paths:
        for node in path.get("path_nodes", []):
            if node.get("id") is None:
                continue
            nid = str(node["id"])
            nodes.setdefault(nid, {
                "id": nid,
                "name": str(node.get("name") or nid),
                "kind": str(node.get("kind") or "Unknown"),
                "endpoint_role": None,
            })
        for edge in path.get("path_edges", []):
            src, dst = edge.get("source"), edge.get("target")
            if src is None or dst is None:
                continue
            src, dst = str(src), str(dst)
            relation = str(edge.get("type") or "RELATED_TO")
            # Defensive fallback if a partially formed path payload omits a node.
            nodes.setdefault(src, {"id": src, "name": src, "kind": "Unknown", "endpoint_role": None})
            nodes.setdefault(dst, {"id": dst, "name": dst, "kind": "Unknown", "endpoint_role": None})
            edges[(src, dst, relation)] = {
                "source": src,
                "target": dst,
                "type": relation,
            }

    source_id = str(entity_1.get("id", ""))
    target_id = str(entity_2.get("id", ""))
    label = str(prediction_label or "uncertain").lower().strip()
    # A causal pair prediction is undirected in this paper's pairwise setup.
    # Do not draw a connector for non-causal/uncertain labels: that would visually
    # imply a relation that the model did not predict.
    prediction_overlay = None
    if label == "causal" and source_id in nodes and target_id in nodes and source_id != target_id:
        prediction_overlay = {
            "source": source_id,
            "target": target_id,
            "label": "DỰ ĐOÁN: CÓ QUAN HỆ NHÂN QUẢ (không định hướng)",
        }

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "source_id": source_id,
        "target_id": target_id,
        "prediction_label": label,
        "prediction_overlay": prediction_overlay,
    }


def build_pair_graph_payload(
    relations: list[dict[str, Any]],
    entity_1: dict[str, Any],
    entity_2: dict[str, Any],
) -> dict[str, Any]:
    """Build a graph containing only the two selected nodes and their direct edges."""
    return {
        "nodes": [
            {
                "id": str(entity["id"]),
                "name": str(entity.get("name", entity["id"])),
                "kind": str(entity.get("kind", "Unknown")),
                "endpoint_role": role,
            }
            for role, entity in (("Entity 1", entity_1), ("Entity 2", entity_2))
        ],
        "edges": [
            {
                "source": str(relation["source"]),
                "target": str(relation["target"]),
                "type": str(relation.get("type", "RELATED_TO")),
            }
            for relation in relations
        ],
        "source_id": str(entity_1["id"]),
        "target_id": str(entity_2["id"]),
        "prediction_label": "uncertain",
        "prediction_overlay": None,
    }


def render_graph_html(payload: dict[str, Any], height: str = "600px") -> str:
    """Render a self-contained PyVis HTML fragment (PyVis is imported lazily)."""
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError("PyVis is not installed. Install dependencies from requirements.txt.") from exc

    import html

    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    node_ids = {str(node["id"]) for node in payload.get("nodes", [])}
    node_by_id = {str(node["id"]): node for node in payload.get("nodes", [])}
    for node in payload.get("nodes", []):
        nid = str(node["id"])
        name = str(node.get("name") or nid)
        kind = str(node.get("kind") or "Unknown")
        kind_vi = NODE_KIND_LABELS_VI.get(kind, kind)
        role = str(node.get("endpoint_role") or "Intermediate node")
        # Label text is rendered as canvas text; title is HTML, so escape its fields.
        net.add_node(
            nid,
            label=f"{name}\n({kind_vi})",
            title=f"{html.escape(role)}<br>ID: {html.escape(nid)}<br>Loại node: {html.escape(kind_vi)} ({html.escape(kind)})",
            color=NODE_COLORS.get(kind, "#d9d9d9"),
            shape="dot",
            size=29 if node.get("endpoint_role") else 20,
            borderWidth=3 if node.get("endpoint_role") else 1,
        )

    for edge in payload.get("edges", []):
        src, dst = str(edge["source"]), str(edge["target"])
        if src not in node_ids or dst not in node_ids:
            continue
        relation = str(edge.get("type") or "RELATED_TO")
        source_kind = str(node_by_id.get(src, {}).get("kind") or "Entity")
        target_kind = str(node_by_id.get(dst, {}).get("kind") or "Entity")
        en_description, vi_description = relation_description(relation, source_kind, target_kind)
        net.add_edge(
            src,
            dst,
            label=relation_verb_vi(relation),
            title=f"Hetionet (EN): {html.escape(en_description)}<br>Hetionet (VI): {html.escape(vi_description)}",
            arrows="to",
            color="#4a4a4a",
            width=2,
            font={"size": 10, "color": "#000000", "align": "middle", "strokeWidth": 0},
        )

    overlay = payload.get("prediction_overlay")
    if overlay:
        src, dst = str(overlay["source"]), str(overlay["target"])
        if src in node_ids and dst in node_ids:
            net.add_edge(
                src,
                dst,
                label=overlay["label"],
                title="Đây chỉ là dự đoán cho cặp entity; không lưu vào Neo4j và không suy ra chiều nhân quả.",
                dashes=True,
                arrows="",
                color="#d62728",
                width=4,
                font={"size": 13, "color": "#d62728", "align": "middle"},
            )

    net.set_options('''{
      "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -22000, "centralGravity": 0.2, "springLength": 150, "springConstant": 0.04}},
      "interaction": {"hover": true, "navigationButtons": true, "keyboard": true, "multiselect": true},
      "edges": {"smooth": {"type": "dynamic"}},
      "nodes": {"font": {"size": 14, "face": "Arial", "color": "#000000"}}
    }''')
    return net.generate_html()
