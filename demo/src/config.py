import os
from dataclasses import dataclass, fields

@dataclass
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "local-demo-change-me")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
    ollama_translation_num_predict: int = int(os.getenv("OLLAMA_TRANSLATION_NUM_PREDICT", "96"))
    zai_api_key: str = os.getenv("ZAI_API_KEY", "")
    zai_base_url: str = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4").rstrip("/")
    zai_model: str = os.getenv("ZAI_MODEL", "glm-5.3")
    zai_num_predict: int = int(os.getenv("ZAI_NUM_PREDICT", "512"))
    max_hops: int = int(os.getenv("MAX_HOPS", "4"))
    max_paths: int = int(os.getenv("MAX_PATHS", "30"))
    top_k_paths: int = int(os.getenv("TOP_K_PATHS", "3"))
    ranker_path: str = os.getenv("RANKER_PATH", "models/xgb_ranker.json")
    # LLM provider cho pipeline (predict / relevance / translate / describe).
    # ollama = raw /api/generate (mặc định, như trước) | openai = mọi endpoint
    # tương thích /chat/completions (Ollama /v1, vLLM, LM Studio, OpenRouter...).
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_api_key_env: str = os.getenv("LLM_API_KEY_ENV", "LLM_API_KEY")
    llm_num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "512"))
    # demo = prompt JSON (giải thích kèm theo) | paper = template zero-shot tcma
    # của repo chính (cặp + context + top-k relation paths, trả 1 từ).
    causal_prompt_style: str = os.getenv("CAUSAL_PROMPT_STYLE", "demo")
    paper_task: str = os.getenv("PAPER_TASK", "comagc")
    paper_style: str = os.getenv("PAPER_STYLE", "mistral")
    # Thư mục artifact ranker do repo chính train (m2_train_ranker.py --rankers
    # xgboost): xgb_ranker.json + ngram_embeddings.npz + config.json.
    repo_ranker_dir: str = os.getenv("REPO_RANKER_DIR", "models/repo_ranker")
    # Thư mục chứa các cross-encoder ce_* (m2_train_ranker.py --save-model),
    # mỗi model_best_<task>_<ranker>_<ts>/ gồm model.safetensors + tokenizer.
    # Tự chọn model có NDCG@5 test cao nhất. Cần torch+transformers (requirements.txt);
    # nếu thiếu thư viện hoặc thư mục -> bỏ qua và dùng ranker kế tiếp.
    repo_ce_dir: str = os.getenv("REPO_CE_DIR", "models/ce")

settings = Settings()


def reload(**overrides) -> None:
    """Cập nhật settings lúc runtime từ UI (Streamlit widget).

    Nhận keyword tương ứng tên field; giá trị None/'' bỏ qua (giữ nguyên).
    """
    mapping = {f.name: f for f in fields(Settings)}
    for key, value in overrides.items():
        if key not in mapping or value is None or value == "":
            continue
        current = getattr(settings, key)
        if isinstance(current, int) and not isinstance(value, bool):
            value = int(value)
        elif isinstance(current, bool):
            value = bool(value)
        else:
            value = str(value)
        setattr(settings, key, value)
