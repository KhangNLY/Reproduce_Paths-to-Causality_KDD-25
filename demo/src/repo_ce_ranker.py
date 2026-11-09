"""Adapter nạp cross-encoder ranker do repo chính train (m2_train_ranker.py --save-model).

Mỗi thư mục model (vd model_best_comagc_ce_ranknet_<ts>/) là output chuẩn của
transformers save_pretrained:
  model.safetensors   trọng số RoBERTa + đầu regression num_labels=1
  config.json         kiến trúc RobertaForSequenceClassification
  tokenizer.json, tokenizer_config.json
  train_meta.json     task, ranker (ce_rmse|ce_ranknet|ce_listnet), metrics
  metrics.json        các chỉ số đánh giá

Inference tái lập ĐÚNG input lúc train (lib/rankers.CrossEncoderRanker):
  query   = "<e1> <e2>"                                (lib/data.query_text)
  passage = "Gene BAX - Gene THAP11 - Compound ..."    (lib/data.path_text, có node-label)
  encode  = tokenizer(query, passage, truncation, max_length=500); score = logit duy nhất.

Cần torch + transformers (đã có trong requirements.txt của demo). Import lười
để phần còn lại của demo không bị vỡ khi thiếu thư viện; khi thiếu, try_load()
trả None và PathRanker rơi xuống ranker kế tiếp trong chuỗi ưu tiên.
"""
from __future__ import annotations

import json
import os


def ce_passage_text(path: dict, include_node_labels: bool = True) -> str:
    """Dùng chung định dạng 'stops' với XGB adapter để hai loại ranker thấy cùng text."""
    from src.repo_ranker import repo_path_text
    return repo_path_text(path, include_node_labels)


class RepoCrossEncoderRanker:
    def __init__(self, model, tok, torch, device, max_len: int, include_node_labels: bool,
                 meta: dict, source_dir: str):
        self.model = model
        self.tok = tok
        self.torch = torch
        self.device = device
        self.max_len = max_len
        self.include_node_labels = include_node_labels
        self.meta = meta
        self.source_dir = source_dir

    @property
    def name(self) -> str:
        return str(self.meta.get("ranker", "ce"))

    @classmethod
    def try_load(cls, dir_path: str, device: str | None = None, max_len: int = 500,
                 include_node_labels: bool = True) -> "RepoCrossEncoderRanker | None":
        if not dir_path or not os.path.isdir(dir_path):
            return None
        if not os.path.exists(os.path.join(dir_path, "model.safetensors")):
            return None
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            print(f"!! Có trọng số cross-encoder tại {dir_path} nhưng thiếu torch/transformers "
                  "trong Python đang chạy -> bỏ qua (xem requirements.txt của demo).")
            return None
        try:
            # Ưu tiên GPU khi có (Colab T4 / CUDA local); tự rơi về CPU.
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            tok = AutoTokenizer.from_pretrained(dir_path)
            model = AutoModelForSequenceClassification.from_pretrained(dir_path)
            model.to(dev)
            model.eval()
            meta = {}
            meta_path = os.path.join(dir_path, "train_meta.json")
            if os.path.exists(meta_path):
                meta = json.loads(open(meta_path, encoding="utf-8").read())
            return cls(model, tok, torch, dev, max_len, include_node_labels, meta, dir_path)
        except Exception as exc:  # trọng số hỏng/thiếu file -> coi như không có
            print(f"!! Không load được cross-encoder từ {dir_path}: {exc}")
            return None

    def score(self, entity_1: str, entity_2: str, paths: list[dict],
              batch_size: int = 8) -> list[float]:
        query = f"{entity_1} {entity_2}"
        scores: list[float] = []
        with self.torch.no_grad():
            for b in range(0, len(paths), batch_size):
                chunk = paths[b:b + batch_size]
                enc = self.tok(
                    [query] * len(chunk),
                    [ce_passage_text(p, self.include_node_labels) for p in chunk],
                    padding=True, truncation=True, max_length=self.max_len, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(float(v) for v in logits.cpu().tolist())
        return scores


def discover_best_ce(models_root: str) -> str | None:
    """Quét <models_root>/model_best_*/ (đặt cả 3 model cạnh nhau), chọn model
    có test_corrected_top5_NDCG@5 cao nhất theo metrics.json."""
    if not models_root or not os.path.isdir(models_root):
        return None
    best_key, best_dir = None, None
    for name in sorted(os.listdir(models_root)):
        d = os.path.join(models_root, name)
        if not name.startswith("model_") or not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, "model.safetensors")):
            continue
        try:
            m = json.loads(open(os.path.join(d, "metrics.json"), encoding="utf-8").read())
            key = float(m.get("test_corrected_top5_NDCG@5", float("-inf")))
        except Exception:
            key = float("-inf")
        if best_key is None or key > best_key:
            best_key, best_dir = key, d
    return best_dir
