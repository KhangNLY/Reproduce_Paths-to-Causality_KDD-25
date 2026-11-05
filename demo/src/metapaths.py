"""Metapath extraction for the demo UI.

Two sources, in line with the paper (KDD'25 "Paths to Causality"):
1. Local Hetionet via Neo4j allShortestPaths (<= max_hops), the same query the
   paper reports in Appendix B.
2. Fallback: the benchmark files ship the paper's own extracted metapaths
   (Module-1 output with rel_score/prob_score). Pairs whose entities the paper
   resolved to nodes absent from the local Hetionet snapshot (and SEMEVAL,
   which the paper links against Wikidata, not Hetionet) use this source.

The two sources are normalized into the same path-dict shape
({path_nodes, path_edges, hops, source}) so ranking, stats and visualization
treat them identically.
"""
from __future__ import annotations
from collections import Counter
from typing import Any


def parse_paper_metapaths(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the benchmark's `metapaths` entries into normal path dicts.

    The paper data only carries node names/labels and relation codes, not the
    stored edge direction, so edges are built left-to-right and path_text will
    mark them with the "?" orientation flag. rel_score/prob_score (the paper's
    Module-1 Subgraph Relevancy Estimation output) are preserved for display.
    """
    out: list[dict[str, Any]] = []
    for m in sample.get("metapaths") or []:
        stops = [s.strip() for s in str(m.get("stops", "")).split(" - ")]
        if len(stops) < 2 or not stops[0]:
            continue
        labels = [s.strip() for s in str(m.get("nodelabels", "")).split(" - ")]
        rels = [s.strip() for s in str(m.get("reltypes", "")).split(" - ")]
        labels += ["Unknown"] * (len(stops) - len(labels))
        nodes = [
            {"id": f"paper::{name}", "name": name, "kind": labels[i]}
            for i, name in enumerate(stops)
        ]
        edges = [
            {"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "type": rels[i]}
            for i in range(min(len(rels), len(stops) - 1))
        ]
        out.append({
            "path_nodes": nodes,
            "path_edges": edges,
            "hops": len(stops) - 1,
            "rel_score": float(m.get("rel_score", 0.0) or 0.0),
            "prob_score": float(m.get("prob_score", 0.0) or 0.0),
            "source": "paper",
        })
    return out


def extract_kg_paths(store, source_id: str, target_id: str, max_hops: int, limit: int) -> list[dict[str, Any]]:
    """Extract metapaths from local Hetionet; each path is tagged source=hetionet."""
    if not source_id or not target_id or source_id == target_id:
        return []
    paths = store.shortest_paths(source_id, target_id, max_hops, limit)
    for p in paths:
        p["source"] = "hetionet"
    return paths


def build_path_pool(resolved_e1, resolved_e2, sample: dict | None, kg_paths_fn) -> tuple[list[dict[str, Any]], str]:
    """Prefer local Hetionet extraction; fall back to the paper's benchmark metapaths.

    kg_paths_fn(source_id, target_id) -> list of paths (a thin wrapper around the
    Neo4j store so callers can cache it).
    """
    if resolved_e1 and resolved_e2:
        try:
            paths = kg_paths_fn(resolved_e1["id"], resolved_e2["id"])
        except Exception:
            paths = []
        if paths:
            return paths, "hetionet"
    paper = parse_paper_metapaths(sample or {})
    if paper:
        return paper, "paper"
    return [], "none"


def path_pool_stats(paths: list[dict[str, Any]]) -> dict[str, Any]:
    hops = [int(p.get("hops", 0)) for p in paths]
    rels = Counter(
        str(e.get("type")) for p in paths for e in p.get("path_edges", [])
    )
    return {
        "total": len(paths),
        "hops_min": min(hops) if hops else 0,
        "hops_max": max(hops) if hops else 0,
        "hops_mean": round(sum(hops) / len(hops), 2) if hops else 0.0,
        "relation_types": len(rels),
        "top_relations": rels.most_common(5),
        "by_hops": dict(sorted(Counter(hops).items())),
    }
