from __future__ import annotations
from src.config import settings
from src.neo4j_store import KGStore
from src.ranker import PathRanker
from src.llm import predict_causality, translate_to_vietnamese
from src.features import path_text_vi
from src.metapaths import build_path_pool, path_pool_stats

MODES = ("paper", "no_ranker", "no_kg")


def analyze_pair(
    entity_1: dict,
    entity_2: dict,
    context: str,
    top_k: int | None = None,
    context_vi: str | None = None,
    progress=None,
    mode: str = "paper",
    sample: dict | None = None,
) -> dict:
    """Run one pairwise prediction.

    Modes (parallel comparison in the UI, mirroring the paper's Table 1/2 rows):
    - "paper":      KG extraction -> Subgraph Ranker -> top-k subgraphs as prompt (§4).
    - "no_ranker":  raw KG extraction goes straight to the LLM, unranked (baseline 2/3).
    - "no_kg":      context + entity pair only, no KG data at all (baseline 1).
    Paths come from local Hetionet when both endpoints resolve there; otherwise
    the benchmark's own metapaths (paper Module-1 output) are used.
    """
    top_k = top_k or settings.top_k_paths
    store = KGStore()
    try:
        raw_paths, path_source = build_path_pool(
            entity_1,
            entity_2,
            sample,
            lambda s, t: store.shortest_paths(s, t, settings.max_hops, settings.max_paths),
        )
    finally:
        store.close()
    if progress:
        progress(f"State 3/5 · KG: {len(raw_paths)} metapath từ {path_source}")

    ranker = PathRanker()
    if mode == "no_kg":
        ranked = []
        ranker_mode = "No-KG baseline (chỉ context + entity, không dùng KG)"
    elif mode == "no_ranker":
        ranked = [{"rank_score": float(i), "path_text": _text(p), **p}
                  for i, p in enumerate(raw_paths)]
        ranker_mode = f"Không ranker — đưa thẳng {len(raw_paths)} path raw vào LLM"
    elif path_source == "paper":
        ranked = ranker.rank_by_paper_score(raw_paths)
        ranker_mode = "Xếp theo rel_score của paper (Module-1) — node không có trong Hetionet local"
    else:
        ranked = ranker.rank(raw_paths, context, entity_1["name"], entity_2["name"])
        ranker_mode = ranker.mode
    for path in ranked:
        path.setdefault("path_text_vi", path_text_vi(path))
    if progress:
        progress(f"State 4/5 · Ranker: {ranker_mode}")

    prediction = predict_causality(
        entity_1["name"], entity_2["name"], context, ranked, top_k=top_k
    )
    if progress:
        progress("State 5/5 · LLM: đã suy luận quan hệ causal/non-causal/uncertain")
    translated = translate_to_vietnamese(prediction["explanation"])
    translated_context = context_vi if context_vi is not None else translate_to_vietnamese(context)
    return {
        "entity_1": entity_1,
        "entity_2": entity_2,
        "context": context,
        "context_vi": translated_context,
        "paths_found": len(raw_paths),
        "path_source": path_source,
        "path_stats": path_pool_stats(raw_paths),
        "ranker_mode": ranker_mode,
        "ranked_paths": ranked,
        "top_paths": ranked[:top_k],
        "prediction": prediction,
        "explanation_vi": translated,
        "is_trained_ranker": "XGB" in ranker_mode,
        "mode": mode,
    }


def _text(path: dict) -> str:
    from src.features import path_text
    return path_text(path)
