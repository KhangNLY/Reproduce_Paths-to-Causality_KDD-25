"""Adapter nạp ranker do repo chính train (lib/rankers.XGBSubgraphRanker).

Repo chính lưu một thư mục artifact gồm 3 file (xem m2_train_ranker.py):
  xgb_ranker.json       cây XGBoost (objective=rank:ndcg)
  ngram_embeddings.npz  W (vocab x 128) + vocab của n-gram LM
  config.json           include_node_labels, neg_num, seed, n_trees, dim

Demo không cần torch/transformers để chạy ranker này: embedding tại inference
chỉ là tra bảng W và lấy trung bình theo token (NgramEmbedder.embed), nên
adapter tái hiện đúng bằng numpy + xgboost.

Quan trọng nhất là tái lập ĐÚNG chuỗi text mà model đã học:
  query = "e1 e2"
  path  = stops, ghép node-label theo config.include_node_labels
          ("Gene BAX - Gene THAP11 - Compound Fulvestrant")
Sai format (vd thêm mũi tên quan hệ như path_text của UI) sẽ lệch phân bố token
so với lúc train và làm hỏng ranking.
"""
from __future__ import annotations

import json
import os

import numpy as np


def repo_path_text(path: dict, include_node_labels: bool) -> str:
    """Path-dict của demo -> chuỗi 'stops' như lib/data.path_text của repo."""
    nodes = path.get("path_nodes", [])
    names = [str(n.get("name", "")) for n in nodes]
    if not include_node_labels:
        return " - ".join(names)
    labels = [str(n.get("kind", "")) for n in nodes]
    return " - ".join(f"{lab} {nm}" for lab, nm in zip(labels, names))


class RepoRanker:
    """Load artifact repo chính; trả None từ try_load() nếu thiếu/lỗi file.

    Import xgboost lười (trong try_load) để các module chỉ dùng repo_path_text
    không bị vỡ khi thiếu thư viện.
    """

    def __init__(self, model, w: np.ndarray, w2i: dict, dim: int, include_node_labels: bool, source_dir: str):
        self.model = model
        self.w = w
        self.w2i = w2i
        self.dim = dim
        self.include_node_labels = include_node_labels
        self.source_dir = source_dir

    @classmethod
    def try_load(cls, dir_path: str) -> "RepoRanker | None":
        if not dir_path or not os.path.isdir(dir_path):
            return None
        try:
            from xgboost import XGBRanker
            cfg = json.loads(open(os.path.join(dir_path, "config.json"), encoding="utf-8").read())
            z = np.load(os.path.join(dir_path, "ngram_embeddings.npz"), allow_pickle=True)
            model = XGBRanker()
            model.load_model(os.path.join(dir_path, "xgb_ranker.json"))
            w2i = {str(word): i for i, word in enumerate(z["vocab"].tolist())}
            return cls(model, z["W"], w2i, int(cfg.get("dim", z["W"].shape[1])),
                       bool(cfg.get("include_node_labels", True)), dir_path)
        except Exception:
            return None

    def embed(self, text: str) -> np.ndarray:
        ids = [self.w2i[t] for t in text.lower().split() if t in self.w2i]
        if not ids:
            return np.zeros(self.dim, dtype="float32")
        return self.w[ids].mean(axis=0)

    def score(self, entity_1: str, entity_2: str, paths: list[dict]) -> list[float]:
        query = f"{entity_1} {entity_2}"
        x = np.vstack([
            self.embed(f"{query} {repo_path_text(p, self.include_node_labels)}")
            for p in paths
        ])
        return [float(v) for v in self.model.predict(x)]
