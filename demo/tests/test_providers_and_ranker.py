"""Smoke tests cho provider layer, paper prompt và repo-ranker adapter.

Không cần Neo4j/Ollama/Docker: các test mock requests.post hoặc dùng artifact
giả nhỏ. Phần cần xgboost tự skip khi thư viện chưa cài (chạy đầy đủ trong
Docker Compose của demo).
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src import llm


class _Reply:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _set(key: str, value):
    object.__setattr__(settings, key, value)


class ProviderRoutingTests(unittest.TestCase):
    """chat() phải điều phối đúng endpoint theo LLM_PROVIDER."""

    def setUp(self):
        import requests
        self._orig_post = requests.post
        self.calls = []
        self._orig = {k: getattr(settings, k) for k in
                      ("llm_provider", "llm_base_url", "llm_model", "ollama_base_url", "ollama_model")}

    def tearDown(self):
        import requests
        requests.post = self._orig_post
        for k, v in self._orig.items():
            _set(k, v)

    def test_openai_provider_hits_chat_completions(self):
        import requests

        def fake_post(url, **kwargs):
            self.calls.append((url, kwargs))
            return _Reply({"choices": [{"message": {"content": " causal"}}]})

        requests.post = fake_post
        _set("llm_provider", "openai")
        _set("llm_base_url", "http://vllm:8000/v1")
        _set("llm_model", "mistralai/mistral-nemo")
        out = llm.chat([{"role": "user", "content": "hi"}], temperature=0.0)
        self.assertEqual(out, "causal")
        url, kwargs = self.calls[0]
        self.assertTrue(url.endswith("/chat/completions"), url)
        self.assertEqual(kwargs["json"]["model"], "mistralai/mistral-nemo")
        self.assertNotIn("Authorization", kwargs["headers"])  # không key cho local server

    def test_openai_provider_sends_bearer_key_when_env_set(self):
        import os
        import requests

        def fake_post(url, **kwargs):
            self.calls.append((url, kwargs))
            return _Reply({"choices": [{"message": {"content": "ok"}}]})

        requests.post = fake_post
        _set("llm_provider", "openai")
        _set("llm_base_url", "https://openrouter.ai/api/v1")
        old = os.environ.get("LLM_API_KEY")
        os.environ["LLM_API_KEY"] = "sk-test"
        try:
            llm.chat([{"role": "user", "content": "hi"}])
        finally:
            if old is None:
                os.environ.pop("LLM_API_KEY", None)
            else:
                os.environ["LLM_API_KEY"] = old
        self.assertEqual(self.calls[0][1]["headers"]["Authorization"], "Bearer sk-test")

    def test_ollama_provider_keeps_raw_generate_api(self):
        import requests

        def fake_post(url, **kwargs):
            self.calls.append((url, kwargs))
            return _Reply({"response": " non-causal"})

        requests.post = fake_post
        _set("llm_provider", "ollama")
        out = llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out, " non-causal")
        url, kwargs = self.calls[0]
        self.assertTrue(url.endswith("/api/generate"), url)
        self.assertEqual(kwargs["json"]["model"], settings.ollama_model)
        self.assertIn("prompt", kwargs["json"])  # raw API dùng prompt, không dùng messages


class PaperPromptTests(unittest.TestCase):
    def setUp(self):
        self._orig = {k: getattr(settings, k) for k in ("causal_prompt_style", "paper_task", "paper_style")}

    def tearDown(self):
        for k, v in self._orig.items():
            _set(k, v)

    def _paths(self):
        return [
            {"path_nodes": [{"name": "TRPV6", "kind": "Gene"}, {"name": "TRAF1", "kind": "Gene"},
                            {"name": "prostate cancer", "kind": "Disease"}],
             "path_edges": [], "hops": 2},
            {"path_nodes": [{"name": "BAX", "kind": "Gene"}, {"name": "THAP11", "kind": "Gene"}],
             "path_edges": [], "hops": 1},
        ]

    def test_paper_prompt_matches_repo_tcma_shape(self):
        _set("paper_style", "mistral")
        _set("paper_task", "comagc")
        p = llm._paper_prompt("TRPV6", "prostate cancer", "TRPV6 contributes to prostate cancer.",
                              self._paths(), top_k=2)
        self.assertIn("[Pair]: TRPV6 and prostate cancer", p)
        self.assertIn("[Context sentence]: TRPV6 contributes to prostate cancer.", p)
        self.assertIn("[Relation paths]: TRPV6 - TRAF1 - prostate cancer. BAX - THAP11.", p)
        self.assertTrue(p.rstrip().endswith("[Relation]:"))
        self.assertIn("gene and disease pair", p)

    def test_paper_prompt_llama_tail(self):
        _set("paper_style", "llama")
        p = llm._paper_prompt("A", "B", "ctx", self._paths(), top_k=1)
        self.assertTrue(p.rstrip().endswith("The relation between A and B is:"))

    def test_verbalize_matches_repo_robust_behavior(self):
        self.assertEqual(llm._verbalize(" causal")[0], "causal")
        self.assertEqual(llm._verbalize(" non-causal")[0], "non-causal")
        # 'causal. Knowledge' chứa 'no' trong 'knowledge' -> verbalizer tác giả
        # trả non-causal; bản robust lấy nhãn sớm nhất -> vẫn non-causal vì
        # _NEG có \bno\b... thực tế 'causal' đứng trước: phải là causal.
        self.assertEqual(llm._verbalize("causal. Knowledge")[0], "causal")
        self.assertFalse(llm._verbalize("unrelated answer")[1])  # không parse được


class RepoRankerTests(unittest.TestCase):
    def test_repo_path_text_with_and_without_labels(self):
        p = {"path_nodes": [{"name": "BAX", "kind": "Gene"}, {"name": "Fulvestrant", "kind": "Compound"}]}
        self.assertEqual(__import__("src.repo_ranker", fromlist=["repo_path_text"]).repo_path_text(p, True),
                         "Gene BAX - Compound Fulvestrant")
        self.assertEqual(__import__("src.repo_ranker", fromlist=["repo_path_text"]).repo_path_text(p, False),
                         "BAX - Fulvestrant")

    def test_try_load_returns_none_when_dir_missing(self):
        from src.repo_ranker import RepoRanker
        self.assertIsNone(RepoRanker.try_load(""))

    def test_end_to_end_with_fake_artifact(self):
        try:
            import xgboost  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("xgboost/numpy chưa cài trong Python này (có trong Docker demo)")
        import tempfile
        import numpy as np
        from xgboost import XGBRanker
        from src.repo_ranker import RepoRanker

        with tempfile.TemporaryDirectory() as tmp:
            # Artifact giả: n-gram 8-d, 1 cây XGBRanker fit trên 2 group.
            vocab = np.array(["gene", "bax", "disease", "cancer", "drug"], dtype=object)
            w = np.random.RandomState(0).rand(len(vocab), 8).astype("float32")
            np.savez_compressed(Path(tmp) / "ngram_embeddings.npz", W=w, vocab=vocab)
            (Path(tmp) / "config.json").write_text(json.dumps(
                {"include_node_labels": True, "neg_num": 5, "seed": 1111, "n_trees": 4, "dim": 8}), encoding="utf-8")
            m = XGBRanker(n_estimators=2, tree_method="hist", objective="rank:ndcg")
            X = np.vstack([w.mean(0), w[0], w.mean(0), w[1]])
            m.fit(X, np.array([2, 1, 2, 1]), qid=np.array([0, 0, 1, 1]))
            m.save_model(str(Path(tmp) / "xgb_ranker.json"))

            r = RepoRanker.try_load(tmp)
            self.assertIsNotNone(r)
            self.assertTrue(r.include_node_labels)
            paths = [
                {"path_nodes": [{"name": "BAX", "kind": "Gene"}, {"name": "cancer", "kind": "Disease"}]},
                {"path_nodes": [{"name": "gene", "kind": "Gene"}, {"name": "drug", "kind": "Compound"}]},
            ]
            scores = r.score("BAX", "cancer", paths)
            self.assertEqual(len(scores), 2)
            self.assertTrue(all(isinstance(s, float) for s in scores))
            # OOV hoàn toàn -> vector 0, vẫn chảy qua model không crash.
            self.assertEqual(len(r.score("zzz", "yyy", paths[:1])), 1)


class RepoCERankerTests(unittest.TestCase):
    """discover_best_ce chọn đúng model theo metrics; try_load an toàn khi thiếu torch."""

    def _write_model_dir(self, root: Path, name: str, ndcg: float, with_weights: bool = True):
        d = root / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"test_corrected_top5_NDCG@5": ndcg}), encoding="utf-8")
        (d / "train_meta.json").write_text(json.dumps({"ranker": name.split("_")[3], "backbone": "roberta-base",
                                                       "test_corrected_top5_NDCG@5": ndcg}), encoding="utf-8")
        if with_weights:
            (d / "model.safetensors").write_bytes(b"fake")
            (d / "config.json").write_text("{}", encoding="utf-8")
        return d

    def test_discover_best_ce_picks_highest_ndcg(self):
        import tempfile
        from src.repo_ce_ranker import discover_best_ce
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_dir(root, "model_best_comagc_ce_rmse_20260101-000000", 0.82)
            best = self._write_model_dir(root, "model_best_comagc_ce_listnet_20260101-000001", 0.87)
            self._write_model_dir(root, "model_best_comagc_ce_ranknet_20260101-000002", 0.84)
            not_a_model = root / "not_model_dir"
            not_a_model.mkdir()
            (not_a_model / "metrics.json").write_text(json.dumps({"test_corrected_top5_NDCG@5": 0.99}), encoding="utf-8")
            self.assertEqual(discover_best_ce(str(root)), str(best))

    def test_discover_ignores_dirs_without_safetensors(self):
        import tempfile
        from src.repo_ce_ranker import discover_best_ce
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_model_dir(root, "model_best_comagc_ce_rmse_1", 0.8, with_weights=False)  # chỉ metrics
            self.assertIsNone(discover_best_ce(str(root)))
            self.assertIsNone(discover_best_ce(""))
            self.assertIsNone(discover_best_ce("/khong/ton/tai"))

    def test_try_load_returns_none_for_dir_without_weights(self):
        from src.repo_ce_ranker import RepoCrossEncoderRanker
        self.assertIsNone(RepoCrossEncoderRanker.try_load(""))
        self.assertIsNone(RepoCrossEncoderRanker.try_load("/khong/ton/tai"))

    def test_ce_passage_text_matches_xgb_format(self):
        from src.repo_ce_ranker import ce_passage_text
        p = {"path_nodes": [{"name": "BAX", "kind": "Gene"}, {"name": "Fulvestrant", "kind": "Compound"}]}
        self.assertEqual(ce_passage_text(p, True), "Gene BAX - Compound Fulvestrant")
        self.assertEqual(ce_passage_text(p, False), "BAX - Fulvestrant")


class ImportRepoRankerZipTests(unittest.TestCase):
    """--src file.zip: giải nén, định vị bộ artifact (gốc hoặc lồng), suy task."""

    def _make_artifact(self, d: Path, labels: bool = True):
        d.mkdir(parents=True, exist_ok=True)
        (d / "xgb_ranker.json").write_text("{}", encoding="utf-8")
        (d / "ngram_embeddings.npz").write_bytes(b"fake")
        (d / "config.json").write_text(json.dumps(
            {"include_node_labels": labels, "neg_num": 5, "seed": 1111, "n_trees": 100, "dim": 128}),
            encoding="utf-8")

    def _zip_dir(self, src_root: Path, zip_path: Path, prefix: Path | None = None):
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in sorted(src_root.rglob("*")):
                if f.is_file():
                    arcname = f.relative_to(prefix or src_root)
                    zf.write(f, arcname)

    def setUp(self):
        import tempfile
        from scripts.import_repo_ranker import REQUIRED, _guess_task_from_path, resolve_zip_source
        self.tmp = Path(tempfile.mkdtemp(prefix="test_zip_import_"))
        self.resolve_zip_source = resolve_zip_source
        self._guess_task_from_path = _guess_task_from_path
        self.REQUIRED = REQUIRED
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_nested_repo_layout_zip(self):
        # zip mô phỏng GitHub: <repo>-main/outputs/module2/<task>/<tag>/model_<ts>/...
        d = self.tmp / "build" / "outputs" / "module2" / "comagc" / "xgb" / "model_20261101_120000"
        self._make_artifact(d)
        z = self.tmp / "nested.zip"
        self._zip_dir(self.tmp / "build", z)
        src = self.resolve_zip_source(z)
        self.assertTrue(all((src / f).exists() for f in self.REQUIRED))
        self.assertEqual(self._guess_task_from_path(src), "comagc")

    def test_flat_zip_at_root(self):
        d = self.tmp / "flat"
        self._make_artifact(d)
        z = self.tmp / "flat.zip"
        self._zip_dir(d, z)
        src = self.resolve_zip_source(z)
        self.assertTrue(all((src / f).exists() for f in self.REQUIRED))
        self.assertEqual(self._guess_task_from_path(src), "default")

    def test_multiple_artifacts_picks_newest_timestamp(self):
        for ts in ("model_20260101_000000", "model_20261231_235959"):
            self._make_artifact(self.tmp / "build" / "module2" / "comagc" / "xgb" / ts)
        z = self.tmp / "multi.zip"
        self._zip_dir(self.tmp / "build", z)
        src = self.resolve_zip_source(z)
        self.assertEqual(src.name, "model_20261231_235959")

    def test_incomplete_zip_raises_system_exit(self):
        z = self.tmp / "broken.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("xgb_ranker.json", "{}")
        with self.assertRaises(SystemExit):
            self.resolve_zip_source(z)


if __name__ == "__main__":
    unittest.main()
