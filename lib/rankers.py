"""Subgraph Ranker (Module 2): Cross-Encoder (RMSE / RankNet / ListNet) và XGBoost + n-gram embedding.

Khác biệt có chủ đích so với code tác giả (đều ghi trong docs/01_paper_summary_vi.md):
  * Chọn epoch tốt nhất trên tập DEV tách từ train (tác giả chọn trên chính tập test).
  * Loss tự cài, bỏ qua phần padding (tác giả chèn '<padding_passage>' và tính vào loss).
  * XGBoost: nhãn = (m - vị_trí) để path tốt nhất có nhãn LỚN nhất; code gốc dùng vị trí (0 = tốt nhất) nên ngược chiều
    và còn có lỗi index theo qid. n-gram LM huấn luyện theo mini-batch (code gốc: batch=1, chậm hàng chục lần).
  * Khi có điểm bằng nhau, thứ tự được xáo ngẫu nhiên có seed (tránh 'ăn' thứ tự oracle sẵn có trong file).
"""
from __future__ import annotations

import copy
import math
import random
import time
from typing import Callable, Dict, List, Optional

from . import data as D
from . import metrics as M


# ---------------------------------------------------------------------------
# Tiện ích xếp hạng
# ---------------------------------------------------------------------------
def rank_items(items: List[dict], scores: List[List[float]], seed: int = 0, top_only: Optional[int] = None) -> List[dict]:
    """Sắp xếp lại metapath theo điểm giảm dần. top_only=5: chỉ sắp 5 path đầu (giống tác giả), phần còn lại giữ nguyên."""
    out = []
    for i, (it, sc) in enumerate(zip(items, scores)):
        mp = it["metapaths"]
        k = len(mp) if top_only is None else min(top_only, len(mp))
        rng = random.Random(f"{seed}-{i}")
        idx = list(range(k))
        rng.shuffle(idx)                                   # tie-break ngẫu nhiên có seed
        idx.sort(key=lambda j: -sc[j])                     # sort ổn định
        new = copy.deepcopy(it)
        new["metapaths"] = [dict(mp[j], ranker_score=float(sc[j])) for j in idx] + copy.deepcopy(mp[k:])
        out.append(new)
    return out


def evaluate_scores(truth: List[dict], scores: List[List[float]], seed: int = 0) -> Dict[str, dict]:
    top5 = rank_items(truth, scores, seed, top_only=5)
    allr = rank_items(truth, scores, seed, top_only=None)
    return {
        "paper_style_top5": M.paper_style_ranking_eval(truth, top5),
        "corrected_top5": M.corrected_ranking_eval(truth, top5, pool_size=5),
        "corrected_all": M.corrected_ranking_eval(truth, allr, pool_size=None),
    }, top5, allr


def split_dev(items: List[dict], frac: float, seed: int):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_dev = max(1, int(round(len(items) * frac))) if frac > 0 else 0
    dev = [items[i] for i in idx[:n_dev]]
    tr = [items[i] for i in idx[n_dev:]]
    return tr, dev


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------
def _loss_fns():
    import torch
    import torch.nn.functional as F

    def rmse(s, y):
        return torch.sqrt(torch.mean((s - y) ** 2) + 1e-12)

    def ranknet(s, y):
        d_y = y[:, None] - y[None, :]
        d_s = s[:, None] - s[None, :]
        mask = d_y > 0
        if mask.sum() == 0:
            return s.sum() * 0.0
        return F.softplus(-d_s[mask]).mean()               # = BCEWithLogits(d_s, 1)

    def listnet(s, y):
        return -(F.softmax(y, dim=0) * F.log_softmax(s, dim=0)).sum()

    return {"rmse": rmse, "ranknet": ranknet, "listnet": listnet}


def _target(y: float, transform: str) -> float:
    if transform == "signed_log":
        return math.copysign(math.log1p(abs(y)), y)
    return y


class CrossEncoderRanker:
    def __init__(self, loss: str, model_name: str = "roberta-base", include_node_labels: bool = True, max_len: int = 500,
                 lr: float = 5e-5, epochs: int = 10, batch_groups: int = 8, neg_num: int = 5, seed: int = 1111,
                 target_transform: str = "none", amp: bool = False, tiny: bool = False, device: Optional[str] = None,
                 log: Callable[[str], None] = print):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        assert loss in ("rmse", "ranknet", "listnet")
        self.torch = torch
        self.loss_name, self.loss_fn = loss, _loss_fns()[loss]
        self.inc, self.max_len, self.lr, self.epochs = include_node_labels, max_len, lr, epochs
        self.batch_groups, self.neg_num, self.seed = batch_groups, neg_num, seed
        self.tt, self.amp, self.log = target_transform, amp, log
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        random.seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tiny:                                            # dry-run: model nhỏ khởi tạo ngẫu nhiên, không tải trọng số
            cfg = AutoConfig.from_pretrained(model_name)
            cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size = 64, 2, 2, 128
            cfg.num_labels = 1
            self.model = AutoModelForSequenceClassification.from_config(cfg)
        else:
            cfg = AutoConfig.from_pretrained(model_name)
            cfg.num_labels = 1
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        self.model.to(self.device)
        self.best_state = None

    # -- mã hoá -------------------------------------------------------------
    def _encode(self, queries, passages):
        enc = self.tok(queries, passages, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def _group_logits(self, items: List[dict], paths_per_item: List[List[dict]]):
        qs, ps, sizes = [], [], []
        for it, mp in zip(items, paths_per_item):
            q = D.query_text(it)
            for m in mp:
                qs.append(q)
                ps.append(D.path_text(m, self.inc))
            sizes.append(len(mp))
        enc = self._encode(qs, ps)
        with self.torch.autocast(device_type=self.device.type, dtype=self.torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.squeeze(-1)
        return list(self.torch.split(logits.float(), sizes))

    # -- huấn luyện ---------------------------------------------------------
    def fit(self, train_items: List[dict], dev_items: List[dict]):
        torch = self.torch
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        rng = random.Random(self.seed)
        train_items = [it for it in train_items if len(it["metapaths"]) >= 1]
        best, history = -1.0, []
        for ep in range(self.epochs):
            self.model.train()
            order = list(range(len(train_items)))
            rng.shuffle(order)
            t0, tot, nb = time.time(), 0.0, 0
            for b in range(0, len(order), self.batch_groups):
                chunk = [train_items[i] for i in order[b:b + self.batch_groups]]
                mps = [it["metapaths"][:self.neg_num] for it in chunk]
                logits = self._group_logits(chunk, mps)
                losses = []
                for lg, mp in zip(logits, mps):
                    y = torch.tensor([_target(float(m["rel_score"]), self.tt) for m in mp], device=self.device)
                    losses.append(self.loss_fn(lg, y))
                loss = torch.stack(losses).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            dev_ndcg = self.eval_dev(dev_items) if dev_items else float("nan")
            history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "dev_NDCG@5": dev_ndcg, "sec": round(time.time() - t0, 1)})
            self.log(f"  [{self.loss_name}] epoch {ep}: loss={tot / max(nb, 1):.4f} dev_NDCG@5={dev_ndcg:.4f} ({time.time() - t0:.1f}s)")
            score = dev_ndcg if dev_items else float(ep)           # không có dev -> lấy epoch cuối
            if score > best:
                best = score
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                history[-1]["best"] = True
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return history

    def eval_dev(self, dev_items: List[dict]) -> float:
        usable = [it for it in dev_items if len(it["metapaths"]) >= 2]
        if not usable:
            return 1.0
        sc = self.score_items(usable, max_paths=self.neg_num)
        trunc = [dict(it, metapaths=it["metapaths"][:self.neg_num]) for it in usable]
        ranked = rank_items(trunc, sc, seed=self.seed)
        # truth = thứ tự train (đã giảm dần theo rel_score); ranked chứa cùng tập pathid
        return M.corrected_ranking_eval(trunc, ranked, ks=(5,))["NDCG@5"]

    def score_items(self, items: List[dict], max_paths: Optional[int] = None, bs_groups: int = 8) -> List[List[float]]:
        self.model.eval()
        out: List[List[float]] = []
        with self.torch.no_grad():
            for b in range(0, len(items), bs_groups):
                chunk = items[b:b + bs_groups]
                mps = [it["metapaths"][:max_paths] if max_paths else it["metapaths"] for it in chunk]
                logits = self._group_logits(chunk, mps)
                out.extend([lg.cpu().tolist() for lg in logits])
        return out

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# ---------------------------------------------------------------------------
# XGBoost + n-gram embedding
# ---------------------------------------------------------------------------
class NgramEmbedder:
    """Mô hình ngôn ngữ n-gram nhỏ (context=2, dim=128) - lấy embedding làm đặc trưng cho XGBoost (Appendix D.2.1)."""

    def __init__(self, context: int = 2, dim: int = 128, epochs: int = 10, batch: int = 256, lr: float = 0.01,
                 seed: int = 1111, device: Optional[str] = None, log: Callable[[str], None] = print):
        import torch
        self.torch, self.context, self.dim, self.epochs, self.batch, self.lr = torch, context, dim, epochs, batch, lr
        self.seed, self.log = seed, log
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.w2i: Dict[str, int] = {}

    def fit(self, texts: List[str]):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        tokens = " ".join(texts).lower().split()
        vocab = sorted(set(tokens))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        ids = torch.tensor([self.w2i[w] for w in tokens], dtype=torch.long)
        n = len(ids)
        if n <= self.context:
            raise ValueError("corpus quá nhỏ để huấn luyện n-gram")
        # ngữ cảnh = [tok[i-1], tok[i-2]] như tác giả
        X = torch.stack([ids[self.context - j - 1: n - j - 1] for j in range(self.context)], dim=1)
        y = ids[self.context:]
        V = len(vocab)
        self.emb = nn.Embedding(V, self.dim)
        net = nn.Sequential(nn.Linear(self.context * self.dim, 128), nn.ReLU(), nn.Linear(128, V))
        self.emb.to(self.device)
        net.to(self.device)
        params = list(self.emb.parameters()) + list(net.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        lossf = nn.CrossEntropyLoss()
        X, y = X.to(self.device), y.to(self.device)
        g = torch.Generator().manual_seed(self.seed)
        self.log(f"  n-gram LM: vocab={V}, tokens={n}, samples={len(y)}")
        for ep in range(self.epochs):
            perm = torch.randperm(len(y), generator=g).to(self.device)
            tot = 0.0
            for b in range(0, len(y), self.batch):
                idx = perm[b:b + self.batch]
                e = self.emb(X[idx]).flatten(1)
                loss = lossf(net(e), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            self.log(f"    ngram epoch {ep}: nll={tot / len(y):.4f}")
        self.W = self.emb.weight.detach().cpu().numpy()
        return self

    def embed(self, text: str):
        import numpy as np
        ids = [self.w2i[t] for t in text.lower().split() if t in self.w2i]
        if not ids:
            return np.zeros(self.dim, dtype="float32")
        return self.W[ids].mean(axis=0)


class XGBSubgraphRanker:
    def __init__(self, include_node_labels: bool = True, neg_num: int = 5, seed: int = 1111, n_trees: int = 100,
                 ngram_epochs: int = 10, ngram_batch: int = 256, ngram_lr: float = 0.01, log: Callable[[str], None] = print):
        self.inc, self.neg_num, self.seed, self.n_trees, self.log = include_node_labels, neg_num, seed, n_trees, log
        self.ng = NgramEmbedder(epochs=ngram_epochs, batch=ngram_batch, lr=ngram_lr, seed=seed, log=log)

    def _sent(self, it, m):
        return f"{D.query_text(it)} {D.path_text(m, self.inc)}"

    def fit(self, train_items: List[dict], corpus_items: List[dict]):
        import numpy as np
        import xgboost as xgb
        corpus = [self._sent(it, m) for it in corpus_items for m in it["metapaths"]]   # văn bản thuần, không dùng nhãn
        self.ng.fit(corpus)
        X, y, qid = [], [], []
        for gi, it in enumerate(train_items):
            mp = it["metapaths"][:self.neg_num]
            for r, m in enumerate(mp):
                X.append(self.ng.embed(self._sent(it, m)))
                y.append(len(mp) - r)                     # path tốt nhất -> nhãn lớn nhất
                qid.append(gi)
        X, y, qid = np.vstack(X), np.array(y), np.array(qid)
        self.model = xgb.XGBRanker(tree_method="hist", objective="rank:ndcg", lambdarank_num_pair_per_sample=8,
                                   lambdarank_pair_method="topk", n_estimators=self.n_trees, random_state=self.seed)
        self.model.fit(X, y, qid=qid)                     # qid đã sắp liên tục theo item
        self.log(f"  XGBRanker fitted on {X.shape[0]} paths / {len(train_items)} pairs")
        return self

    def score_items(self, items: List[dict], max_paths: Optional[int] = None) -> List[List[float]]:
        import numpy as np
        out = []
        for it in items:
            mp = it["metapaths"][:max_paths] if max_paths else it["metapaths"]
            X = np.vstack([self.ng.embed(self._sent(it, m)) for m in mp])
            out.append([float(v) for v in self.model.predict(X)])
        return out

    # -- lưu / nạp trọng số -----------------------------------------------
    def save(self, path):
        """Lưu: xgb_ranker.json (cây) + ngram_embeddings.npz (embedding n-gram + từ điển) + config.json."""
        import json as _json
        from pathlib import Path as _P
        import numpy as np
        p = _P(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p / "xgb_ranker.json"))
        vocab = sorted(self.ng.w2i, key=self.ng.w2i.get)
        np.savez_compressed(p / "ngram_embeddings.npz", W=self.ng.W, vocab=np.array(vocab, dtype=object))
        (p / "config.json").write_text(_json.dumps({"include_node_labels": self.inc, "neg_num": self.neg_num, "seed": self.seed,
                                                    "n_trees": self.n_trees, "dim": self.ng.dim}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        import json as _json
        from pathlib import Path as _P
        import numpy as np
        import xgboost as xgb
        p = _P(path)
        cfg = _json.loads((p / "config.json").read_text(encoding="utf-8"))
        obj = cls(cfg["include_node_labels"], cfg["neg_num"], cfg["seed"], cfg["n_trees"])
        z = np.load(p / "ngram_embeddings.npz", allow_pickle=True)
        obj.ng.W = z["W"]
        obj.ng.dim = cfg["dim"]
        obj.ng.w2i = {w: i for i, w in enumerate(z["vocab"].tolist())}
        obj.model = xgb.XGBRanker()
        obj.model.load_model(str(p / "xgb_ranker.json"))
        return obj

