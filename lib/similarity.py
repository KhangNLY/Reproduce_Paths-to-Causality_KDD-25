cd"""Similarity-based ranker (baseline F.1): cosine giữa embedding(cặp biến) và embedding(path) bằng all-MiniLM-L6-v2.

Định dạng đã được kiểm chứng KHỚP TUYỆT ĐỐI (sai số 0.0000) với cột sim_score trong test_sim.jsonl của tác giả cho COMAGC:
    query = "e1 e2"            path = stops với ' - ' thay bằng khoảng trắng
(Với SemEval/Wikidata công thức của tác giả không tái lập được chính xác -> xem docs.)
"""
from __future__ import annotations

import hashlib
from typing import Dict, List
.
_MODEL = {}


def _minilm():
    if "m" not in _MODEL:
        try:
            import transformers
            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            import sys
            raise SystemExit(
                "Thiếu thư viện 'sentence-transformers' trong Python đang chạy:\n"
                f"  {sys.executable}\n"
                "Nguyên nhân thường gặp: bạn chạy bằng Python hệ thống thay vì môi trường .venv của dự án.\n"
                "Cách sửa (chọn 1):\n"
                "  - dùng .venv:  .venv\\Scripts\\python <script>   (Windows)  |  .venv/bin/python <script>  (Linux/macOS)\n"
                "  - hoặc cài vào Python này:  python -m pip install sentence-transformers\n"
                "  - hoặc dùng: python setup_env.py") from e
        try:
            _MODEL["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
        except Exception:
            _MODEL["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _MODEL["m"]


def _hash_embed(texts: List[str], dim: int = 64):
    """Embedder giả (bag-of-words băm) - CHỈ để dry-run khi chưa cài sentence-transformers."""
    import numpy as np
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def embed(texts: List[str], embedder: str = "minilm"):
    import numpy as np
    if embedder == "hash":
        return _hash_embed(texts)
    e = _minilm().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(e)


def sim_text(item: dict, path: dict):
    return f"{item['e1']} {item['e2']}", str(path["stops"]).replace(" - ", " ")


def score_paths(item: dict, paths: List[dict], embedder: str = "minilm") -> List[float]:
    q = f"{item['e1']} {item['e2']}"
    P = [sim_text(item, p)[1] for p in paths]
    E = embed([q] + P, embedder)
    return [float(E[0] @ E[i + 1]) for i in range(len(P))]


def rank_by_similarity(item: dict, paths: List[dict], embedder: str = "minilm") -> List[dict]:
    sc = score_paths(item, paths, embedder)
    order = sorted(range(len(paths)), key=lambda i: -sc[i])
    return [dict(paths[i], sim_score=sc[i]) for i in order]

