from __future__ import annotations
import os
import numpy as np
from src.features import FEATURE_NAMES, path_features, path_text
from src.config import settings
from src.repo_ranker import RepoRanker, repo_path_text


class PathRanker:
    """Ranks paths with, in order of preference:

    1. RepoCE:    cross-encoder ce_rmse/ce_ranknet/ce_listnet do repo chính
                  train (m2_train_ranker.py --save-model; model.safetensors +
                  tokenizer trong model_best_<task>_<ranker>_<ts>/). Chọn model
                  có NDCG@5 test cao nhất. Cần torch+transformers trong Python
                  chạy demo; nếu thiếu -> tự bỏ qua.
    2. RepoXGB:   artifact xgboost+n-gram của repo chính (xgb_ranker.json +
                  ngram_embeddings.npz + config.json tại REPO_RANKER_DIR).
    3. Local XGB: model XGBRanker 6-feature tự train trong demo
                  (scripts/train_ranker.py) tại RANKER_PATH.
    4. Lexical heuristic khi không có gì.

    Paper-sourced metapaths (rel_score của Module-1) vẫn xếp theo
    rank_by_paper_score(), không đi qua các model này.
    """
    def __init__(self):
        from src.repo_ce_ranker import RepoCrossEncoderRanker, discover_best_ce
        self.ce_model_dir = discover_best_ce(settings.repo_ce_dir)
        self.ce_model = (RepoCrossEncoderRanker.try_load(self.ce_model_dir)
                         if self.ce_model_dir else None)
        self.model = None
        self.repo_model = RepoRanker.try_load(settings.repo_ranker_dir)
        if os.path.exists(settings.ranker_path):
            try:
                from xgboost import XGBRanker
                model = XGBRanker()
                model.load_model(settings.ranker_path)
                self.model = model
            except Exception:
                self.model = None

    @property
    def mode(self) -> str:
        if self.ce_model is not None:
            m = self.ce_model.meta
            return (f"Repo cross-encoder {self.ce_model.name} · {m.get('backbone', 'roberta-base')} "
                    f"(NDCG@5 test={m.get('test_corrected_top5_NDCG@5', '?')})")
        if self.repo_model is not None:
            return (f"Repo XGB rank:ndcg + n-gram 128d (models/repo_ranker, "
                    f"labels={self.repo_model.include_node_labels and 'có' or 'không'} node-label)")
        if self.model is not None:
            return "XGBoost pairwise ranker (objective=rank:pairwise)"
        return "Lexical heuristic (no trained XGBoost model found)"

    def rank(self, paths: list[dict], context: str, entity_1: str, entity_2: str) -> list[dict]:
        if not paths:
            return []
        include_labels = True
        if self.ce_model is not None:
            scores = self.ce_model.score(entity_1, entity_2, paths)
            include_labels = self.ce_model.include_node_labels
        elif self.repo_model is not None:
            scores = self.repo_model.score(entity_1, entity_2, paths)
            include_labels = self.repo_model.include_node_labels
        else:
            x = np.asarray([path_features(p, context, entity_1, entity_2) for p in paths], dtype=float)
            if self.model is not None:
                scores = self.model.predict(x)
            else:
                # Transparent fallback. It is a baseline, not a trained ML result.
                scores = 0.65 * x[:, 3] + 0.25 * x[:, 4] + 0.10 * x[:, 5]
        out = []
        for p, score in zip(paths, scores):
            item = dict(p)
            item["rank_score"] = float(score)
            if self.ce_model is not None or self.repo_model is not None:
                # Giữ lại chuỗi 'stops' đúng distribution lúc train để hiển thị/so sánh.
                item["ranker_input_text"] = repo_path_text(p, include_labels)
            else:
                feats = path_features(p, context, entity_1, entity_2)
                item["features"] = dict(zip(FEATURE_NAMES, feats))
            item["path_text"] = __import__("src.features", fromlist=["path_text"]).path_text(p)
            out.append(item)
        return sorted(out, key=lambda z: z["rank_score"], reverse=True)

    def rank_by_paper_score(self, paths: list[dict]) -> list[dict]:
        out = []
        for p in paths:
            item = dict(p)
            item["rank_score"] = float(p.get("rel_score", 0.0) or 0.0)
            item["path_text"] = path_text(p)
            out.append(item)
        return sorted(out, key=lambda z: z["rank_score"], reverse=True)
