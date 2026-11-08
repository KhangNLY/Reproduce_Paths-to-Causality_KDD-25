from __future__ import annotations
import json
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

    choice (từ UI): None = tự động theo thứ tự trên; "ce:<tên>" = buộc dùng
    cross-encoder đó (nếu load được, không thì báo rõ); "repo_xgb", "local_xgb",
    "lexical" = buộc dùng cấp tương ứng.
    """
    def __init__(self, choice: str | None = None):
        from src.repo_ce_ranker import RepoCrossEncoderRanker, discover_best_ce
        self.choice = choice
        self.ce_model_dir = None
        self.ce_model = None
        self.repo_model = None
        self.model = None
        if choice in (None, "", "auto") or choice.startswith("ce:"):
            wanted = choice[3:] if choice and choice.startswith("ce:") else None
            candidates = []
            if wanted:
                for name in sorted(os.listdir(settings.repo_ce_dir)) if os.path.isdir(settings.repo_ce_dir) else []:
                    if wanted in name:
                        candidates.append(os.path.join(settings.repo_ce_dir, name))
            else:
                best = discover_best_ce(settings.repo_ce_dir)
                candidates = [best] if best else []
            for d in candidates:
                loaded = RepoCrossEncoderRanker.try_load(d)
                if loaded is not None:
                    self.ce_model, self.ce_model_dir = loaded, d
                    break
        if self.ce_model is None and choice in (None, "", "auto", "repo_xgb"):
            self.repo_model = RepoRanker.try_load(settings.repo_ranker_dir)
        if self.ce_model is None and self.repo_model is None and choice in (None, "", "auto", "local_xgb"):
            if os.path.exists(settings.ranker_path):
                try:
                    from xgboost import XGBRanker
                    model = XGBRanker()
                    model.load_model(settings.ranker_path)
                    self.model = model
                except Exception:
                    self.model = None

    @classmethod
    def choices(cls) -> list[tuple[str, str]]:
        """Liệt kê (value, nhãn) cho UI: tự động + từng checkpoint khả dụng."""
        from src.repo_ce_ranker import discover_best_ce
        out = [("auto", "Tự động (CE repo → XGB repo → XGB local → lexical)")]
        root = settings.repo_ce_dir
        if os.path.isdir(root):
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                can_ce = True
            except ImportError:
                can_ce = False
            best = discover_best_ce(root)
            for name in sorted(os.listdir(root)):
                d = os.path.join(root, name)
                if not name.startswith("model_") or not os.path.isdir(d):
                    continue
                if not os.path.exists(os.path.join(d, "model.safetensors")):
                    continue
                try:
                    meta = json.loads(open(os.path.join(d, "train_meta.json"), encoding="utf-8").read())
                    ndcg = meta.get("test_corrected_top5_NDCG@5")
                    label = f"Cross-encoder {meta.get('ranker', 'ce')} ({meta.get('task', '')}, NDCG@5={ndcg})"
                except Exception:
                    label = name
                if not can_ce:
                    label += " — cần torch (chưa cài trong env này)"
                if best and os.path.samefile(best, d):
                    label += " · tốt nhất"
                out.append((f"ce:{name}", label))
        if RepoRanker.try_load(settings.repo_ranker_dir) is not None:
            out.append(("repo_xgb", "XGBoost + n-gram của repo (models/repo_ranker)"))
        if os.path.exists(settings.ranker_path):
            out.append(("local_xgb", "XGBoost local 6-feature (models/xgb_ranker.json)"))
        out.append(("lexical", "Lexical heuristic baseline"))
        return out

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
        if self.choice == "lexical":
            return "Lexical heuristic (đã chọn trong UI)"
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
            item["path_text"] = path_text(p)
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
