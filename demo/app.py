from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from src.neo4j_store import KGStore
from src.pipeline import analyze_pair
from src import config
from src.config import settings
from src.llm import LocalLLMError, describe_input_vi, direct_predict_causality, translate_to_vietnamese
from src.visualization import build_graph_payload, build_pair_graph_payload, render_graph_html
from src.metapaths import build_path_pool
from src.features import path_text
from src.ranker import PathRanker

PREDICTION_LABELS_VI = {
    "causal": "Có quan hệ nhân quả đề xuất",
    "non-causal": "Không có quan hệ nhân quả đề xuất",
    "uncertain": "Chưa đủ bằng chứng",
}
GROUND_TRUTH_LABELS_VI = {
    "1": "Causal (ground truth)",
    "0": "Non-causal (ground truth)",
}

st.set_page_config(page_title="Paths to Causality — Local Demo", layout="wide")
st.title("Paths to Causality · Local KG + LLM demo")
st.caption("Self-hosted demo inspired by the KDD 2025 paper. Pairwise prediction is not scientific proof of causality.")
st.markdown(
    """
    <style>
    :root {
        --ink: #000000;
        --muted: #1f1f1f;
        --line: #8fa6b3;
        --line-strong: #628092;
        --teal: #087f78;
        --teal-dark: #075e59;
        --teal-soft: #edf8f7;
        --zone: #eef3f7;
        --zone-strong: #e2eaf0;
    }
    .stApp, [data-testid="stAppViewContainer"] { background: #ffffff; color: var(--ink); }
    [data-testid="stSidebar"] { background: var(--zone-strong); border-right: 2px solid var(--line-strong); }
    [data-testid="stSidebar"] > div:first-child { background: var(--zone-strong); }
    [data-testid="stSidebar"] .block-container { padding: 1.25rem 1rem 2rem; }
    h1 { color: var(--ink); letter-spacing: -0.02em; font-weight: 750; }
    h2, h3 { color: var(--ink); margin-top: 1.25rem; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] { background: var(--zone); border: 2px solid var(--line); border-radius: 12px; padding: 0.75rem 0.9rem; min-height: 5rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); white-space: normal; overflow-wrap: anywhere; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: clamp(0.9rem, 1.5vw, 1.35rem); line-height: 1.2; white-space: normal; overflow-wrap: anywhere; }
    [data-testid="stVerticalBlockBorderWrapper"] { border: 2px solid var(--line) !important; background: #ffffff; border-radius: 12px; }
    div[data-baseweb="select"] > div { min-height: 2.8rem; height: auto; border-color: var(--line-strong); background: var(--zone); }
    div[data-baseweb="select"] span { white-space: normal; overflow-wrap: anywhere; word-break: break-word; color: var(--ink); }
    div[data-baseweb="select"] > div:focus-within { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal); }
    textarea, input { overflow-wrap: anywhere; word-break: break-word; color: var(--ink); }
    textarea:focus, input:focus { border-color: var(--teal) !important; box-shadow: 0 0 0 1px var(--teal) !important; }
    [data-testid="stTextInput"], [data-testid="stTextArea"] { background: var(--zone); border-radius: 10px; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { overflow-wrap: anywhere; word-break: break-word; }
    [data-testid="stMarkdownContainer"] a { color: var(--teal-dark); }
    [data-testid="stExpander"] { border: 2px solid var(--line); border-radius: 10px; background: #ffffff; overflow-wrap: anywhere; word-break: break-word; }
    [data-testid="stExpander"] summary:hover { background: var(--teal-soft); }
    [data-testid="stDataFrame"] { border: 2px solid var(--line); border-radius: 10px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stAlert"] * { color: var(--ink) !important; }
    [data-testid="stStatusWidget"] { border: 1px solid #a9d8d2; background: var(--teal-soft); border-radius: 10px; }
    .stButton > button { border-radius: 9px; border: 2px solid var(--line-strong); color: var(--ink); background: var(--zone); min-height: 2.7rem; font-weight: 650; }
    .stButton > button:hover { border-color: var(--teal); color: var(--teal-dark); background: var(--teal-soft); }
    .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); color: #ffffff; }
    .stButton > button[kind="primary"]:hover { background: var(--teal-dark); border-color: var(--teal-dark); color: #ffffff; }
    [data-testid="stRadio"] label { color: var(--ink); }
    [data-testid="stRadio"] label p { color: var(--ink); }
    [data-testid="stSlider"] [role="slider"] { background: var(--teal); border-color: var(--teal); }
    [data-testid="stSliderThumbValue"] { color: var(--ink) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATASET_ROOT = Path(__file__).resolve().parent / "data" / "benchmark"

@st.cache_data
def load_dataset_samples() -> list[dict]:
    samples = []
    # Demo hiển thị sample tập TEST để đánh giá; tập train chỉ dùng cho ranker.
    paths = sorted(DATASET_ROOT.glob("*/processed_test_*.jsonl"))
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                sample.setdefault("dataset", path.parent.name)
                sample.setdefault("source_file", path.name.removeprefix("processed_"))
                sample.setdefault("context_en", sample.get("sentence", ""))
                samples.append(sample)
    inference_cache = {}
    # Cache zai_inference của MỌI split (kể cả train) vẫn được nạp:
    # UI chỉ show sample test, nhưng cache tra theo (dataset, source_file, qid)
    # nên sample test chỉ khớp cache test — không có rò rỉ kết quả giữa split.
    for path in sorted(DATASET_ROOT.glob("*/zai_inference_*.jsonl")):
        for result in path.open(encoding="utf-8"):
            if not result.strip():
                continue
            row = json.loads(result)
            key = (path.parent.name, row.get("source_file", ""), str(row.get("qid")))
            inference_cache[key] = row
    for sample in samples:
        key = (sample.get("dataset"), sample.get("source_file", ""), str(sample.get("qid")))
        if key in inference_cache:
            sample["zai_result"] = inference_cache[key]
    return samples


# Dự đoán CÓ SẴN của tác giả theo từng ranker (không cần chạy lại LLM):
# test_truth = Subgraph Ranker, test_gpt_* = RankGPT, test_sim = similarity.
# (dataset, e1, e2, qid) -> {ranker_label: llm_pred}
@st.cache_data
def load_author_predictions() -> dict:
    from collections import defaultdict
    out: dict = defaultdict(dict)
    labels = {"truth": "Subgraph Ranker (tác giả)", "sim": "Similarity (tác giả)"}
    for path in sorted(DATASET_ROOT.glob("*/test_*.jsonl")):
        ds = path.parent.name
        stem = path.stem  # test_truth / test_gpt_x / test_sim
        label = None
        if stem == "test_truth":
            label = labels["truth"]
        elif stem == "test_sim":
            label = labels["sim"]
        elif stem.startswith("test_gpt_"):
            label = "RankGPT (tác giả)"
        if not label:
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            out[(ds, str(r.get("e1")), str(r.get("e2")), str(r.get("qid")))][label] = {
                "pred": str(r.get("llm_pred", "")),
                "ground_truth": str(r.get("ground_truth", "")),
            }
    return dict(out)

samples = load_dataset_samples()


def sample_entity(sample: dict | None, key: str) -> str:
    if not sample:
        return ""
    return str(sample.get(key) or sample.get("e" + key[-1], "")).strip()


def sample_split(sample: dict) -> str:
    source_file = str(sample.get("source_file", ""))
    return "train" if source_file.startswith("train") else "test"


@st.cache_data(show_spinner=False)
def cached_input_description(entity_1: str, entity_2: str, context: str) -> str:
    return describe_input_vi(entity_1, entity_2, context)

@st.cache_resource
def get_store():
    return KGStore()


@st.cache_resource
def get_ranker(choice: str = "auto"):
    return PathRanker(choice=choice)


@st.cache_data(show_spinner=False)
def cached_pool(e1: dict, e2: dict, sample_index: int | None) -> tuple[list, str]:
    """Extract the metapath pool for a pair (KG first, benchmark fallback).

    Cached so slider changes and reruns do not re-query Neo4j.
    """
    sample = samples[sample_index] if sample_index is not None else None
    s = KGStore()
    try:
        return build_path_pool(
            e1, e2, sample,
            lambda a, b: s.shortest_paths(a, b, settings.max_hops, settings.max_paths),
        )
    finally:
        s.close()

try:
    store = get_store()
    stats = store.stats()
except Exception as e:
    st.error(f"Không kết nối được Neo4j: {e}")
    st.info("Khởi động dịch vụ bằng `docker compose up -d --build`, sau đó nạp Hetionet theo README.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("KG nodes", f"{stats['nodes']:,}")
m2.metric("KG edges", f"{stats['edges']:,}")
m3.metric("Max hops", settings.max_hops)
m4.metric("Pipeline LLM", settings.llm_model or settings.ollama_model)

with st.sidebar:
    st.header("Configuration")
    dataset_options = sorted({str(sample.get("dataset", "")).lower() for sample in samples})
    dataset_choice = st.selectbox("1. Dataset", ["Manual input"] + dataset_options)
    dataset_samples = [sample for sample in samples if str(sample.get("dataset", "")).lower() == dataset_choice]
    split_options = sorted({sample_split(sample) for sample in dataset_samples})
    split_choice = st.selectbox("2. Tập dữ liệu", ["Tất cả"] + split_options, disabled=not dataset_samples)
    split_samples = dataset_samples if split_choice == "Tất cả" else [sample for sample in dataset_samples if sample_split(sample) == split_choice]
    source_options = sorted({str(sample.get("source_file", "")) for sample in split_samples})
    source_choice = st.selectbox("3. File nguồn", ["Tất cả"] + source_options, disabled=not split_samples)
    source_samples = split_samples if source_choice == "Tất cả" else [sample for sample in split_samples if sample.get("source_file") == source_choice]
    ground_truth_choice = st.selectbox("4. Ground truth", ["Tất cả", "Causal", "Non-causal"], disabled=not source_samples)
    if ground_truth_choice != "Tất cả":
        expected_ground_truth = "1" if ground_truth_choice == "Causal" else "0"
        source_samples = [sample for sample in source_samples if str(sample.get("ground_truth", "")) == expected_ground_truth]
    sample_options = ["Không chọn sample"] + [f"qid={sample['qid']} · {sample.get('e1', sample.get('entity_1', ''))} → {sample.get('e2', sample.get('entity_2', ''))}" for sample in source_samples]
    sample_choice = st.selectbox("5. Sample ID", sample_options, disabled=not source_samples)
    inference_mode = st.radio(
        "6. Inference mode",
        [
            "So sánh 3 chế độ (paper · no-ranker · no-KG)",
            "Pipeline (KG → ranker → Ollama)",
            "Direct LLM (Z.ai GLM-5.3)",
        ],
        help="Direct LLM gửi node/edge/context thẳng tới Z.ai và không chạy ranker.",
    )
    top_k = st.slider("Top-k paths to LLM", min_value=1, max_value=5, value=min(settings.top_k_paths, 5))

    st.markdown("**LLM provider**")
    provider_choice = st.selectbox(
        "Pipeline LLM",
        ["ollama", "openai"],
        index=0 if settings.llm_provider == "ollama" else 1,
        format_func=lambda v: "Ollama local (raw API)" if v == "ollama" else "OpenAI-compatible (OpenRouter, vLLM, LM Studio, Ollama /v1…)",
        help="Chuyển toàn bộ pipeline (classify/translate/relevance) sang endpoint tương thích OpenAI. Key đọc từ biến môi trường.",
    )
    llm_model_input = ""
    llm_base_input = ""
    if provider_choice == "openai":
        llm_base_input = st.text_input(
            "Base URL", value=settings.llm_base_url,
            help="vd https://openrouter.ai/api/v1 · http://host.docker.internal:11434/v1 (Ollama) · http://host.docker.internal:8000/v1 (vLLM)",
        )
        llm_model_input = st.text_input(
            "Model ID", value=settings.llm_model or "",
            placeholder="vd mistralai/mistral-nemo · qwen2.5:7b",
        )
        st.text_input(
            "API key (tùy chọn)", type="password", key="llm_api_key_input",
            help="Điền thẳng vào UI cho tiện; để trống sẽ dùng biến môi trường "
                 f"`{settings.llm_api_key_env}` (server local không cần key).",
        )
        import os as _os
        _ui_key = st.session_state.get("llm_api_key_input", "").strip()
        if _ui_key:
            _os.environ[settings.llm_api_key_env] = _ui_key
        st.caption(f"Key ưu tiên từ ô nhập trên; nếu trống dùng biến môi trường `{settings.llm_api_key_env}`.")
    prompt_style = st.selectbox(
        "Prompt classify",
        ["demo", "paper"],
        index=0 if settings.causal_prompt_style == "demo" else 1,
        format_func=lambda v: "Demo (JSON, có explanation)" if v == "demo" else "Paper (tcma template, trả 1 từ)",
    )
    config.reload(
        llm_provider=provider_choice,
        llm_base_url=llm_base_input or None,
        llm_model=llm_model_input or None,
        causal_prompt_style=prompt_style,
    )
    st.caption(
        f"Provider: **{settings.llm_provider}** · model `{settings.llm_model or settings.ollama_model}`"
        + (f" · endpoint `{settings.llm_base_url}`" if settings.llm_provider == "openai" else "")
        + f" · prompt `{settings.causal_prompt_style}`"
    )

    st.markdown("**Ranker / checkpoint**")
    ranker_choices = PathRanker.choices()
    ranker_values = [v for v, _ in ranker_choices]
    ranker_labels = {v: lbl for v, lbl in ranker_choices}
    default_ranker = st.session_state.get("ranker_choice", "auto")
    ranker_choice = st.selectbox(
        "Ranking model",
        ranker_values,
        index=ranker_values.index(default_ranker) if default_ranker in ranker_values else 0,
        format_func=lambda v: ranker_labels.get(v, v),
        help="Checkpoint cross-encoder (.safetensors) nằm tại models/ce; XGBoost artifact tại models/repo_ranker. Chọn 'Tự động' để dùng tốt nhất khả dụng.",
    )
    st.session_state["ranker_choice"] = ranker_choice
    st.caption("Inference dùng ranker chọn ở đây; caption 'Ranking method' trong kết quả xác nhận model thực sự chạy.")
    st.markdown("**Services**")
    st.write("Streamlit · Neo4j · Ollama")

selected_sample = None
if sample_choice != "Không chọn sample" and source_samples:
    selected_sample = source_samples[sample_options.index(sample_choice) - 1]

sample_qid = (selected_sample.get("dataset"), selected_sample.get("source_file"), selected_sample.get("qid")) if selected_sample else None
if st.session_state.get("loaded_sample_qid") != sample_qid:
    st.session_state["q1"] = sample_entity(selected_sample, "entity_1") if selected_sample else "FGF6"
    st.session_state["q2"] = sample_entity(selected_sample, "entity_2") if selected_sample else "prostate cancer"
    st.session_state["context"] = selected_sample["context_en"] if selected_sample else "FGF6 contributes to the growth of prostate cancer."
    st.session_state["loaded_sample_qid"] = sample_qid

if selected_sample:
    metadata = st.container(border=True)
    with metadata:
        st.markdown("**Selected sample metadata**")
        meta_columns = st.columns([1, 1, 1.5, 1, 1.8])
        meta_columns[0].metric("Dataset", str(selected_sample.get("dataset", "")).upper())
        meta_columns[1].metric("Tập", sample_split(selected_sample))
        meta_columns[2].metric("File", selected_sample.get("source_file", ""))
        meta_columns[3].metric("Sample ID", selected_sample.get("qid", ""))
        ground_truth = str(selected_sample.get("ground_truth", ""))
        meta_columns[4].metric("Ground truth", GROUND_TRUTH_LABELS_VI.get(ground_truth, "Không có"))
        st.caption("Ground truth chỉ dùng để đánh giá sau infer, không được gửi vào prompt model.")

# Khi chọn sample test, entity linking ưu tiên đúng node mà paper đã dùng: endpoint
# của metapath đầu tiên trong benchmark (tên benchmark thường ≠ tên node Hetionet).
linked_note = None
if selected_sample and (selected_sample.get("metapaths") or []):
    _stops = selected_sample["metapaths"][0].get("stops", "").split(" - ")
    if len(_stops) >= 2:
        _first, _last = store.search_entities(_stops[0].strip(), limit=1), store.search_entities(_stops[-1].strip(), limit=1)
        if _first and _last:
            paper_e1, paper_e2 = _first[0], _last[0]
            linked_note = (paper_e1, paper_e2)

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Input (English)")
    if linked_note:
        st.info(
            f"Entity linking theo metapath của paper: **{linked_note[0]['name']}** ({linked_note[0]['kind']}) "
            f"và **{linked_note[1]['name']}** ({linked_note[1]['kind']}). Tên gốc benchmark: "
            f"“{sample_entity(selected_sample, 'entity_1')}” / “{sample_entity(selected_sample, 'entity_2')}”."
        )
        e1, e2 = linked_note
    else:
        q1 = st.text_input("Entity 1 search", key="q1")
        candidates1 = store.search_entities(q1, limit=15)
        if candidates1:
            preferred_entity_1_kind = selected_sample and selected_sample.get("dataset") == "comagc" and "Gene"
            e1_index = next((i for i, candidate in enumerate(candidates1) if preferred_entity_1_kind and candidate["kind"] == preferred_entity_1_kind), 0)
            e1 = st.selectbox("Choose entity 1", candidates1, index=e1_index, format_func=lambda x: f"{x['name']} · {x['kind']} · {x['id']}", key="entity1")
        else:
            e1 = None
            st.warning("No matching entity for Entity 1 yet. Try another name fragment.")
        q2 = st.text_input("Entity 2 search", key="q2")
        candidates2 = store.search_entities(q2, limit=15)
        if candidates2:
            preferred_entity_2_kind = selected_sample and selected_sample.get("dataset") == "comagc" and "Disease"
            e2_index = next((i for i, candidate in enumerate(candidates2) if preferred_entity_2_kind and candidate["kind"] == preferred_entity_2_kind), 0)
            e2 = st.selectbox("Choose entity 2", candidates2, index=e2_index, format_func=lambda x: f"{x['name']} · {x['kind']} · {x['id']}", key="entity2")
        else:
            e2 = None
            st.warning("No matching entity for Entity 2 yet. Try another name fragment.")
with right:
    st.subheader("2. Context")
    context = st.text_area(
        "English textual context (from benchmark or user input)",
        height=150,
        key="context",
        help="This is an illustrative example. Replace it with a context row from the benchmark for evaluation.",
    )
    if selected_sample:
        st.markdown("**Context tiếng Việt (đã dịch và cache):**")
        st.write(selected_sample["context_vi"])
        try:
            st.markdown("**Mô tả input ngắn bằng LLM:**")
            st.write(cached_input_description(sample_entity(selected_sample, "entity_1"), sample_entity(selected_sample, "entity_2"), context))
        except LocalLLMError as error:
            st.warning(f"Chưa tạo được mô tả input bằng LLM: {error}")
    st.caption("Benchmark label is deliberately not requested here, so it cannot leak into prediction.")

pair_relations = []
if e1 and e2:
    try:
        pair_relations = store.direct_relations(e1["id"], e2["id"])
    except Exception as error:
        st.warning(f"Không lấy được direct KG relations: {error}")

st.subheader("3. Direct KG relations giữa hai entity")
if e1 and e2:
    if pair_relations:
        st.caption(f"Tìm thấy {len(pair_relations)} edge trực tiếp trong Hetionet. Đây là quan hệ đã lưu trong KG, chưa phải causal prediction.")
        relation_rows = [
            {
                "Source": relation["source_name"],
                "Relation code": relation["type"],
                "Target": relation["target_name"],
                "Direction": f"{relation['source_kind']} → {relation['target_kind']}",
            }
            for relation in pair_relations
        ]
        st.dataframe(relation_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Không có edge trực tiếp giữa hai entity; các path trung gian vẫn được truy vấn trong Pipeline mode.")
    pair_graph = render_graph_html(build_pair_graph_payload(pair_relations, e1, e2), height="360px")
    components.html(pair_graph, height=380, scrolling=True)
else:
    st.info("Chọn được hai entity trong Neo4j để xem direct relations.")

st.subheader("3b. Metapath trước & sau ranker")
if e1 and e2:
    try:
        pool, pool_source = cached_pool(e1, e2, samples.index(selected_sample) if selected_sample else None)
    except Exception as error:
        pool, pool_source = [], "none"
        st.warning(f"Không extract được metapath pool: {error}")
    if pool:
        # ---- TRƯỚC ranker: 1 subgraph gộp toàn bộ pool (giới hạn hiển thị 100 path) ----
        POOL_SHOW_CAP = 100
        pool_show = pool[:POOL_SHOW_CAP]
        pool_ranker = get_ranker(st.session_state.get("ranker_choice", "auto"))
        if pool_source == "paper":
            ranked_preview = pool_ranker.rank_by_paper_score(pool)
        else:
            ranked_preview = pool_ranker.rank(pool, context, e1["name"], e2["name"])

        st.markdown(f"**Trước ranker · subgraph của {len(pool_show)}/{len(pool)} metapath** (thứ tự extraction, hiển thị tối đa {POOL_SHOW_CAP})")
        st.caption(
            "Pool TRƯỚC ranker giữ đúng thứ tự extraction (chưa xếp hạng liên quan). "
            + ("Nguồn: Hetionet local." if pool_source == "hetionet" else "Nguồn: metapath benchmark (node không có trong Hetionet local).")
        )
        g_pool = render_graph_html(build_graph_payload(pool_show, e1, e2, "uncertain"), height="420px")
        components.html(g_pool, height=440, scrolling=True)

        # ---- SAU ranker: kéo thả chọn 1..30 path đã sort theo score ----
        n_ranked = min(30, len(ranked_preview))
        n_sel = st.slider(
            "Số metapath sau ranker hiển thị trên subgraph", min_value=1, max_value=n_ranked,
            value=min(3, n_ranked), key="n_ranked_show",
            help="Path đã được ranker sort theo rank_score giảm dần; subgraph vẽ đúng số path bạn chọn.",
        )
        sel = ranked_preview[:n_sel]
        st.markdown(f"**Sau ranker · top {n_sel}/{len(ranked_preview)} theo rank_score** (`{pool_ranker.mode}`)")
        for idx, p in enumerate(sel, start=1):
            score = p.get("rank_score", 0.0)
            extra = ""
            if pool_source == "paper":
                extra = f" · rel_score={p.get('rel_score', 0):.4f}"
            st.markdown(f"`#{idx}` · score `{float(score):.4f}` · {p.get('hops', '?')} hops{extra}")
            st.caption((p.get("path_text") or path_text(p))[:180] + ("…" if len(p.get("path_text") or "") > 180 else ""))
        g_after = render_graph_html(build_graph_payload(sel, e1, e2, "uncertain"), height="420px")
        components.html(g_after, height=440, scrolling=True)
        st.caption("Subgraph trên là các path SAU xếp hạng (score cao nhất đứng đầu) — kéo slider để tăng/giảm số path vẽ; đồ thị dưới pool là toàn bộ ứng viên chưa xếp hạng.")
    else:
        st.info("Chưa có metapath pool cho cặp entity này (KG không có đường đi trong giới hạn hop và benchmark không kèm metapath).")
else:
    st.info("Chọn hai entity để xem metapath trước/sau ranker.")

# Do not display a previous run's graph/results after its input has changed.
input_signature = (
    e1.get("id") if e1 else None,
    e2.get("id") if e2 else None,
    context.strip(),
    int(top_k),
)
if st.session_state.get("analysis_signature") != input_signature:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("compare_results", None)

compare_mode = inference_mode.startswith("So sánh")
run_label = (
    "Run Direct LLM → Z.ai GLM-5.3" if inference_mode.startswith("Direct")
    else "So sánh song song: paper · no-ranker · no-KG" if compare_mode
    else "Run KG paths → ranker → local Ollama"
)
run = st.button(run_label, type="primary", disabled=not (e1 and e2 and context.strip()), use_container_width=True)
if run:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("compare_results", None)
    if e1["id"] == e2["id"]:
        st.error("Hãy chọn hai thực thể khác nhau.")
    elif compare_mode:
        compare = {}
        progress_box = st.status("So sánh 3 chế độ · đang chạy…", expanded=True)
        try:
            for mode_key, mode_label in (
                ("paper", "Pipeline paper (KG → ranker → top-k → LLM)"),
                ("no_ranker", "No-ranker (toàn bộ path KG raw → thẳng LLM)"),
                ("no_kg", "No-KG baseline (chỉ context + entity)"),
            ):
                progress_box.write(f"▶ {mode_label}")
                compare[mode_key] = analyze_pair(
                    e1, e2, context.strip(), top_k=top_k,
                    context_vi=selected_sample.get("context_vi") if selected_sample else None,
                    mode=mode_key, sample=selected_sample,
                    ranker_choice=st.session_state.get("ranker_choice", "auto"),
                )
            st.session_state["compare_results"] = compare
            st.session_state["analysis_signature"] = input_signature
            progress_box.update(label="Đã chạy xong cả 3 chế độ", state="complete")
        except LocalLLMError as e:
            progress_box.update(label="Thất bại · Ollama chưa trả kết quả", state="error")
            st.error(str(e))
        except Exception as e:
            progress_box.update(label="Thất bại · xem chi tiết", state="error")
            st.exception(e)
    else:
        with st.spinner("Retrieving paths, ranking and querying the local LLM…"):
            progress_box = st.status("State 1/5 · Entity linking", expanded=True)

            def report_progress(message: str) -> None:
                progress_box.write(message)
                progress_box.update(label=message, state="running")

            try:
                report_progress("State 2/5 · Entity linking: Gene/Disease đã được chọn")
                if inference_mode.startswith("Direct"):
                    cached_direct = selected_sample.get("zai_result") if selected_sample else None
                    if cached_direct and cached_direct.get("status") == "ok":
                        report_progress("State 3/5 · Direct LLM: dùng kết quả Z.ai đã cache")
                        direct_prediction = cached_direct["prediction"]
                        explanation_vi = cached_direct.get("explanation_vi", "")
                        context_vi = cached_direct.get("context_vi", "")
                        pair_relations = cached_direct.get("direct_relations", pair_relations)
                    else:
                        report_progress("State 3/5 · Direct LLM: gửi node, edge và context tới Z.ai")
                        direct_prediction = direct_predict_causality(e1, e2, context.strip(), pair_relations)
                        report_progress("State 4/5 · Direct LLM: nhận được causal/non-causal")
                        explanation_vi = translate_to_vietnamese(direct_prediction["explanation"])
                        context_vi = translate_to_vietnamese(context.strip())
                    result = {
                        "entity_1": e1,
                        "entity_2": e2,
                        "context": context.strip(),
                        "context_vi": context_vi,
                        "paths_found": 0,
                        "ranker_mode": "Không dùng ranker (Direct Z.ai)",
                        "ranked_paths": [],
                        "top_paths": [],
                        "prediction": direct_prediction,
                        "explanation_vi": explanation_vi,
                        "direct_relations": pair_relations,
                        "mode": "direct",
                    }
                    report_progress("State 5/5 · Direct LLM: đã dịch explanation sang tiếng Việt")
                else:
                    result = analyze_pair(
                        e1,
                        e2,
                        context.strip(),
                        top_k=top_k,
                        context_vi=selected_sample.get("context_vi") if selected_sample else None,
                        progress=report_progress,
                        sample=selected_sample,
                        ranker_choice=st.session_state.get("ranker_choice", "auto"),
                    )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_signature"] = input_signature
                progress_box.update(label="State 5/5 · Hoàn tất pipeline", state="complete")
            except LocalLLMError as e:
                progress_box.update(label="Pipeline failed · Ollama chưa trả kết quả", state="error")
                st.error(str(e))
            except Exception as e:
                progress_box.update(label="Pipeline failed · kiểm tra chi tiết lỗi bên dưới", state="error")
                st.exception(e)

compare_results = st.session_state.get("compare_results")
if compare_results:
    st.divider()
    st.subheader("4. So sánh 3 chế độ trên cùng một cặp entity")
    st.caption("Cùng input, ba đường suy luận khác nhau — tương ứng các dòng trong bảng so sánh của paper: pipeline đầy đủ, baseline không ranker, và baseline không dùng KG. Các dòng 'tác giả' là dự đoán có sẵn trong benchmark (không chạy lại LLM).")
    truth = GROUND_TRUTH_LABELS_VI.get(str((selected_sample or {}).get("ground_truth", "")), "—")
    header = st.columns([2.2, 1.2, 1.2, 2.6])
    for col, label in zip(header, ["Chế độ", "Dự đoán", "Ground truth", "Đường suy luận"]):
        col.markdown(f"**{label}**")
    for mode_key, mode_label in (
        ("paper", "Pipeline paper (KG → ranker → top-k)"),
        ("no_ranker", "No-ranker (KG raw → LLM)"),
        ("no_kg", "No-KG (chỉ context)"),
    ):
        r = compare_results.get(mode_key)
        if not r:
            continue
        row = st.columns([2.2, 1.2, 1.2, 2.6])
        pred_label = r["prediction"]["label"]
        row[0].write(mode_label)
        row[1].markdown(
            f":green[{PREDICTION_LABELS_VI.get(pred_label, pred_label)}]"
            if pred_label == "causal"
            else f":red[{PREDICTION_LABELS_VI.get(pred_label, pred_label)}]"
            if pred_label == "non-causal"
            else f":orange[{PREDICTION_LABELS_VI.get(pred_label, pred_label)}]"
        )
        row[2].write(truth)
        row[3].caption(r["ranker_mode"])
        with st.expander(f"Explanation ({mode_label})", expanded=False):
            st.write(r["prediction"]["explanation"])
            st.write(r["explanation_vi"])

    # ---- Dự đoán CÓ SẴN của tác giả theo từng ranker (đọc từ benchmark, không gọi LLM) ----
    author_preds = {}
    if selected_sample:
        author_all = load_author_predictions()
        author_preds = author_all.get((
            str(selected_sample.get("dataset", "")).lower(),
            str(selected_sample.get("e1", selected_sample.get("entity_1", ""))),
            str(selected_sample.get("e2", selected_sample.get("entity_2", ""))),
            str(selected_sample.get("qid")),
        ), {})
    if author_preds:
        st.markdown("**Dự đoán của tác giả (đã có sẵn trong benchmark — mỗi ranker một tập dự đoán):**")
        for label, info in author_preds.items():
            pred = info.get("pred")
            if pred not in ("0", "1"):
                continue  # file không kèm llm_pred cho ranker này (vd test_sim chỉ có ranking)
            row = st.columns([2.2, 1.2, 1.2, 2.6])
            gt = GROUND_TRUTH_LABELS_VI.get(info.get("ground_truth", ""), "—")
            shown = PREDICTION_LABELS_VI.get("causal" if pred == "1" else "non-causal", pred)
            color = ":green[" if pred == "1" else ":red["
            row[0].write(label)
            row[1].markdown(f"{color}{shown}]")
            row[2].write(gt)
            row[3].caption("llm_pred có sẵn trong file test tương ứng ranker")
    elif selected_sample:
        st.caption("(Cặp này không có dự đoán của tác giả trong benchmark.)")
    st.session_state["analysis_result"] = compare_results.get("paper")

result = st.session_state.get("analysis_result")
if result:
    st.divider()
    st.subheader("4. KG interpretation · Giải thích tiếng Việt")
    st.caption("Mỗi path được đọc theo chiều cạnh thật trong Hetionet. Tên node giữ nguyên để entity linking chính xác; loại node và relation được giải thích bằng tiếng Việt.")
    for idx, path in enumerate(result["ranked_paths"][:top_k], start=1):
        with st.expander(f"Path {idx} · {path['hops']} hops", expanded=idx == 1):
            st.write(path["path_text_vi"])

    st.subheader("5. Prediction")
    pred = result["prediction"]
    c1, c2, c3 = st.columns([1.3, 0.8, 1.5])
    c1.markdown("**Quan hệ sau infer**")
    c1.write(PREDICTION_LABELS_VI.get(pred["label"], pred["label"]))
    c2.markdown("**Candidate paths**")
    c2.write(result["paths_found"])
    c3.markdown("**Ranking method**")
    c3.write(result["ranker_mode"])
    st.markdown("**Explanation (English)**")
    st.write(pred["explanation"])
    st.markdown("**Giải thích quan hệ bằng tiếng Việt**")
    st.write(result["explanation_vi"])
    with st.expander("View context translation"):
        st.write("**English:**", result["context"])
        st.write("**Tiếng Việt:**", result["context_vi"])
    st.warning("Nhãn causal/non-causal là dự đoán cho cặp thực thể dựa trên context và đường dẫn KG; không khẳng định quan hệ nhân quả đã được chứng minh. Nếu có đường nét đứt, đó chỉ là overlay dự đoán, không được ghi vào Hetionet.")

    st.subheader("6. Ranked KG paths")
    if not result["ranked_paths"]:
        st.info("Không tìm được đường đi ngắn nhất trong Hetionet với giới hạn hop hiện tại. Điều này không chứng minh hai thực thể không có quan hệ nhân quả.")
    else:
        for idx, p in enumerate(result["ranked_paths"][:10], start=1):
            with st.expander(f"Rank {idx} · score {p['rank_score']:.4f} · {p['hops']} hops" + (" · vào prompt LLM" if idx <= top_k else ""), expanded=idx <= top_k):
                st.write(p["path_text"])
                if p.get("source") == "paper":
                    st.caption(f"Nguồn: metapath benchmark (rel_score={p.get('rel_score', 0):.4f}, prob_score={p.get('prob_score', 0):.4f})")
                if p.get("features"):
                    st.json(p["features"])
                elif p.get("ranker_input_text"):
                    st.caption(f"Ranker input (repo n-gram): `{p['ranker_input_text']}`")
    st.subheader("7. Metapath pool · Thống kê & đồ thị")
    stats = result.get("path_stats") or {}
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Số metapath extract", stats.get("total", result["paths_found"]))
    s2.metric("Số hops (min–max)", f"{stats.get('hops_min', 0)}–{stats.get('hops_max', 0)}")
    s3.metric("Hops trung bình", stats.get("hops_mean", 0))
    s4.metric("Số loại relation", stats.get("relation_types", 0))
    st.caption(
        "Nguồn path: "
        + ("extract trực tiếp từ Hetionet local (allShortestPaths ≤ 4 hops)." if result.get("path_source") == "hetionet"
           else "metapath có sẵn trong benchmark (node không có trong Hetionet local); nhãn rel_score/prob_score là output Module-1 của paper."
           if result.get("path_source") == "paper" else "không có (chế độ no-KG).")
    )
    if stats.get("by_hops"):
        st.bar_chart({"số path theo hops": stats["by_hops"]})
    if stats.get("top_relations"):
        st.write("Relation phổ biến nhất trong pool: " + ", ".join(f"`{rel}` ×{n}" for rel, n in stats["top_relations"]))

    st.subheader("8. Selected paths in Hetionet · top-{} vào LLM".format(top_k))
    graph_payload = build_graph_payload(
        result["top_paths"], result["entity_1"], result["entity_2"], pred["label"]
    )
    graph_html = render_graph_html(graph_payload, height="600px")
    components.html(graph_html, height=620, scrolling=True)
    st.caption("Mũi tên liền = chiều cạnh đã lưu trong Hetionet; đồ thị chỉ hiển thị các cạnh nằm trên top-k đường dẫn được chọn, không phải toàn bộ KG.")
    if pred["label"] == "causal":
        st.caption("Đường đỏ nét đứt không có mũi tên = dự đoán pair-level causal của model. Nó không xác định chiều nhân quả, không phải cạnh đã xác nhận và không được ghi vào Neo4j.")
    else:
        st.caption("Model không dự đoán nhãn causal nên không vẽ connector dự đoán. Nhãn non-causal/uncertain không làm thay đổi dữ liệu Hetionet.")
    if not result["ranked_paths"]:
        st.info("Đồ thị đang hiển thị hai thực thể đầu vào nhưng không có đường dẫn Hetionet trong giới hạn hop hiện tại. Không tìm thấy path không đồng nghĩa với non-causal.")

st.divider()
