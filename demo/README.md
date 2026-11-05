# Paths to Causality — Self-hosted Demo

A standalone, locally hosted demo inspired by **Paths to Causality: Finding Informative Subgraphs Within Knowledge Graphs for Knowledge-Based Causal Discovery** (KDD 2025). It is not the authors' official implementation and should not be described as an exact reproduction. The official repository notes that some experimental code/data depend on proprietary libraries and are not fully released: https://github.com/susantiyuni/path-to-causality

## What this repo runs locally

- **Neo4j Community**: stores Hetionet v1.0 and retrieves short paths.
- **Ollama + Qwen2.5 7B**: local model for relation classification, weak relevance labeling for paths, and English-to-Vietnamese explanation/context translation.
- **XGBoost Ranker**: optional trained pairwise ranker. If no checkpoint is present, the UI clearly reports and uses a lexical heuristic baseline instead.
- **Streamlit + PyVis**: local UI and interactive visualization of the selected ranked paths.
- **No hosted inference APIs by default**. Internet is needed once to download Hetionet, the Docker images and the Ollama model; after assets are present, inference runs locally. Ollama runs on the host machine, outside Docker Compose. Optional: the pipeline LLM and the `Direct LLM` mode can be pointed at any OpenAI-compatible endpoint or Z.ai (see "Pipeline LLM providers" and "Direct Z.ai inference"), and ranker checkpoints trained by the reproduction repo are loaded from fixed paths under `models/` (see "Model checkpoints: fixed locations").

Hetionet v1.0 contains 47,031 nodes and 2,250,197 relationships across 11 node types and 24 relationship types. Check the upstream source/licensing notes before redistributing data: https://github.com/hetio/hetionet

## Architecture

```text
English entity pair + English context
              |
              v
        Neo4j / Hetionet
              |
       candidate shortest paths (<= 4 hops)
              |
              v
    XGBoost ranker (or lexical baseline)
              |
        top-k paths
              |
              v
    Local Ollama LLM (Qwen2.5 7B)
       |                  |
       v                  v
 causal/non-causal     English explanation
                          |
                          v
                 Local LLM translates to Vietnamese
                          |
                          v
             Streamlit + interactive path graph
```

**Visualization semantics:** Solid arrows show the stored direction of real Hetionet edges on selected top-k paths. Entity 1 and Entity 2 are highlighted. A red dashed connector is shown only for a `causal` model prediction; it is undirected because this is a pair-level classifier, not a causal-direction estimator. `non-causal` and `uncertain` do not create a connector. This connector is a UI overlay, never stored in Neo4j. The view contains selected paths, not the entire 2.25-million-edge KG.

Path text also preserves the direction stored on each Hetionet edge: if a shortest path traverses an edge in reverse, it is displayed with a left-pointing relation (`<-`) instead of incorrectly rendering every traversal as a forward arrow.

## Prerequisites

- Docker Engine + Docker Compose plugin.
- For Qwen2.5 7B, plan for about 5 GB model storage plus working memory; CPU-only inference is possible but slower. GPU acceleration depends on your Ollama/Docker runtime configuration.
- This compose file reserves 2 GB Neo4j heap + 1 GB page cache. Adjust for your host.
- Cross-encoder checkpoints under `models/ce/` add ~500 MB per model on disk (~1.5 GB for the full COMAGC set). They are not committed to git and not baked into the Docker image — copy them to the fixed paths after cloning (see "Model checkpoints: fixed locations").

## Quickstart

1. Copy env file:

   ```bash
   cp .env.example .env
   ```

   Change `NEO4J_PASSWORD` in `.env` before exposing this machine to any other host. This is a local-demo password, not production hardening.

2. Download Hetionet v1.0 files:

   ```bash
   bash scripts/download_hetionet.sh
   ```

   The edges file is stored with Git LFS upstream. The script uses the Git LFS media URL and runs `gzip -t` so a tiny LFS pointer text file cannot be mistaken for a successfully downloaded `.gz` archive.

3. Build and start services:

   ```bash
   docker compose up -d --build
   ```

4. Install and start Ollama on the host machine, then pull the local model:

   ```bash
   ollama pull qwen2.5:7b
   ```

   Confirm it is present:

   ```bash
   ollama list
   ```

5. Load Hetionet into local Neo4j (first import takes time and disk):

   ```bash
   docker compose exec app python scripts/load_hetionet.py
   ```

   To deliberately reload from scratch later, use `python scripts/load_hetionet.py --reset`. It deletes the existing `:Entity` graph.

6. Open:

   - Streamlit UI: http://localhost:8501
   - Neo4j Browser: http://localhost:7474
   - Ollama API on the host: http://127.0.0.1:11434

The initial UI includes an illustrative English context. Search/select entities from the **actual nodes loaded into Hetionet** before running the model. If a node has no KG path within the hop limit, that is not evidence of non-causality.

## COMAGC sample flow

The repository includes a small COMAGC sample under `data/benchmark/comagc/`. Translate and cache its Vietnamese contexts before opening the UI:

```bash
docker compose exec app python scripts/prepare_comagc.py \
   --input data/benchmark/comagc/test_truth_sample.jsonl \
   --output data/benchmark/comagc/processed_samples.jsonl \
   --limit 10
```

After the cache is created, refresh Streamlit and choose a `COMAGC qid` in the sidebar. The English context is sent to Ollama; the cached Vietnamese context is displayed in the UI. The selected entities are linked to Neo4j, paths are retrieved from Hetionet, ranked, and passed to the local LLM. For COMAGC samples with duplicate names, the UI prefers a `Gene` for Entity 1 and a `Disease` for Entity 2.

## Direct Z.ai inference

The UI also provides a `Direct LLM (Z.ai GLM-5.3)` mode. It sends the selected nodes, all direct edges between them, and the English context directly to Z.ai. This mode does not run shortest-path ranking or XGBoost. Configure the key locally in `.env`:

```env
ZAI_API_KEY=your-key
ZAI_BASE_URL=https://api.z.ai/api/paas/v4
ZAI_MODEL=glm-5.3
```

To run one cached direct inference pass across all datasets, using 20 records from each `train_full.jsonl` and all test records:

```bash
docker compose exec -T app python scripts/run_zai_inference.py \
   --datasets ade,comagc,gene,semeval \
   --train-limit 20
```

Results are saved as `zai_inference_*.jsonl` beside each processed dataset file. The UI uses these cached results when available, so selecting a sample does not call Z.ai again.

## Pipeline LLM providers

Pipeline inference (causal classification, relevance weak-labeling, translation, input description) is no longer hard-wired to local Ollama. Set `LLM_PROVIDER` in `.env`:

- `LLM_PROVIDER=ollama` (default): raw Ollama `/api/generate` on the host, exactly as before.
- `LLM_PROVIDER=openai`: any OpenAI-compatible `/chat/completions` endpoint — Ollama's own `/v1`, vLLM, LM Studio, OpenRouter, Together, Groq, etc. Point `LLM_BASE_URL` at the server, set `LLM_MODEL`, and (only for hosted providers) export the API key in the environment variable named by `LLM_API_KEY_ENV` (default `LLM_API_KEY`). Keys are never written to files.

Examples:

```env
# Same local Ollama, but through its OpenAI-compatible endpoint
LLM_PROVIDER=openai
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:7b

# vLLM serving the model the reproduction pipeline itself used
LLM_PROVIDER=openai
LLM_BASE_URL=http://host.docker.internal:8000/v1
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.1

# OpenRouter (same route the reproduction repo uses)
LLM_PROVIDER=openai
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=mistralai/mistral-nemo
```

`CAUSAL_PROMPT_STYLE=paper` switches the classifier from the demo's JSON prompt to the reproduction repo's zero-shot `tcma` template (pair + context sentence + top-k relation paths, one-word answer, robust verbalizer) — the exact prompt shape the paper's numbers were produced with. `PAPER_TASK` selects the per-task instruction wording (comagc/ade/gene/semeval) and `PAPER_STYLE` the closing phrase (mistral/llama/gemma), matching `lib/prompts.py` of the parent repo.

## Model checkpoints: fixed locations

The demo looks for trained checkpoints from the reproduction repo at **fixed paths inside `demo/models/`** (no configuration hunting). Docker Compose mounts `./models:/app/models`, so the same folder works both inside and outside Docker. Paths are set via `.env` (`REPO_CE_DIR`, `REPO_RANKER_DIR`, `RANKER_PATH`) and the defaults already point at these fixed locations:

| Checkpoint | Fixed path | Contents | Loaded by |
|---|---|---|---|
| Cross-encoder (paper's RoBERTa RankNet-style rankers) | `models/ce/model_best_<task>_<ranker>_<ts>/` | `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `train_meta.json`, `metrics.json` | `src/repo_ce_ranker.py` |
| XGBoost + n-gram artifact | `models/repo_ranker/` | `xgb_ranker.json`, `ngram_embeddings.npz`, `config.json` | `src/repo_ranker.py` |
| Demo's own 6-feature XGBoost | `models/xgb_ranker.json` | single XGBRanker model file | `src/ranker.py` |

### Cross-encoder checkpoints (`.safetensors`)

Export from the reproduction repo with `python m2_train_ranker.py --rankers ce_rmse ce_ranknet ce_listnet --save-model`, then place each `model_best_<task>_<ranker>_<timestamp>/` folder under `models/ce/`. If several checkpoints are present, the demo **auto-selects the one with the best `test_corrected_top5_NDCG@5`** (from each folder's `metrics.json`) and the UI reports it as `Repo cross-encoder <loss> · roberta-base (NDCG@5 test=...)`. Current default set: `ce_rmse`, `ce_ranknet`, `ce_listnet` trained on COMAGC.

These checkpoints require `torch` + `transformers`, which are **commented out** in `requirements.txt` by default to keep the Docker image small and the build fast. Uncomment those two lines and rebuild when you want the cross-encoder inside Docker. Without them the demo skips the CE checkpoints (with a console notice) and falls back to the XGBoost artifact or lexical baseline — nothing breaks. Running Streamlit directly on the host (outside Docker) also works: `pip install torch transformers` in that environment.

Inference reproduces the training input exactly: `tokenizer("<e1> <e2>", "Gene BAX - Gene THAP11 - Compound Fulvestrant", max_length=500)` and the single regression logit is the rank score.

### XGBoost + n-gram artifact

Train in the reproduction repo (`python m2_train_ranker.py --rankers xgboost`), then copy the artifact folder into the fixed path:

```bash
# in demo/: auto-discovers ../outputs (newest model_<ts>), or point at a folder / a .zip
python scripts/import_repo_ranker.py
python scripts/import_repo_ranker.py --src ../outputs/module2/comagc/xgb/model_<ts>
python scripts/import_repo_ranker.py --src ~/Downloads/ranker_comagc.zip
```

No torch needed — inference is a numpy embedding lookup plus XGBoost trees. When present, the UI reports `Repo XGB rank:ndcg + n-gram 128d`. The adapter rebuilds the exact training text (`"<e1> <e2> <node-label> <name> - ..."`); the demo's decorative `path_text` (arrows, relation verbs) is intentionally not fed to it.

### Ranking priority & fallbacks

`PathRanker` picks the first available, so the demo always runs even with an empty `models/` folder:

1. `models/ce/` cross-encoder (if `torch`/`transformers` import and the folder has `model.safetensors`),
2. `models/repo_ranker/` XGBoost + n-gram,
3. `models/xgb_ranker.json` demo 6-feature XGBoost,
4. transparent lexical heuristic (visible in the UI as a baseline).

Paper-sourced metapaths (pairs whose entities are not in local Hetionet) keep using the paper's own `rel_score` ordering and never go through these models.

Checkpoints under `models/ce/` and `models/repo_ranker/` are git-ignored (they are ~500 MB per model) and excluded from the Docker build context — mount or copy them in; they are expected at exactly these paths at runtime.


## The ranker: honest MVP behavior

The paper's contribution includes estimating the relevance of candidate paths and training a Learning-to-Rank model before passing the top paths to an LLM. A ranked list by itself is not enough to claim this contribution has been reproduced.

This repo has two modes:

1. **No checkpoint:** a small transparent lexical heuristic is used and marked in the UI. This makes the interactive demo usable immediately, but it is a baseline, not learned XGBoost.
2. **Checkpoint present:** an XGBoost pairwise ranker is loaded from `models/xgb_ranker.json`.

To train the XGBoost ranker, first prepare a benchmark CSV according to `data/benchmark/README.md`. Use only training/development pairs (not held-out test pairs) to generate pseudo relevance labels:

```bash
docker compose exec app python scripts/prepare_ranker_data.py --input data/benchmark/causal_pairs.csv --limit 100

docker compose exec app python scripts/train_ranker.py
```

`prepare_ranker_data.py` asks the local LLM to assign 0/1/2 path-relevance scores. These are **LLM pseudo-labels**, not human gold relevance judgments. For a sound report, split by query/entity pair first, document the split, and never use test pairs to create labels, train the ranker, or tune prompts. You may replace pseudo-labels with human relevance judgments if available.

After saving the checkpoint, refresh the UI. It should report `XGBoost pairwise ranker (objective=rank:pairwise)` as its ranking method. This is an XGBoost implementation of a pairwise ranking objective, not the paper's separate RoBERTa-based RankNet model.

## Benchmark evaluation

Place the paper's English benchmark in `data/benchmark/causal_pairs.csv` using the schema documented in `data/benchmark/README.md`. Keep only rows whose entity names can be mapped unambiguously to Hetionet and report how many rows are excluded. Do not assume every COMAGC row maps to Hetionet.

Run a small smoke evaluation first:

```bash
docker compose exec app python scripts/evaluate.py --input data/benchmark/causal_pairs.csv --limit 10
```

Then run the full evaluation on the held-out split. Output is written to `results/evaluation_predictions.csv`; console output includes Precision, Recall, F1 and a confusion matrix. The script excludes `uncertain` predictions from the scored subset, so also report coverage (`scored / processed`) alongside metrics. Keep the same LLM, prompt, path limit and test split across baselines.

Suggested ablation rows for a report:

- Local LLM without KG paths.
- Local LLM with random paths.
- Local LLM with lexical ranking.
- Local LLM with trained XGBoost ranking.

The app implements the ranked-path setting. For a publication-quality ablation, add explicit switches for each baseline and compute all metrics on the same fixed held-out cases.

## What is and is not implemented

Implemented: local graph store, entity search, shortest-path retrieval, candidate path features, lexical fallback, XGBoost training utility, local LLM prediction, local LLM relevance weak labeling, English→Vietnamese translation, graph visualization and CSV evaluation utility.

Not claimed: exact reproduction of the authors' proprietary/partially released components; clinical validation; causal effect estimation; automatic graph mutation; or proof that a predicted causal label is scientifically true.

## Local smoke tests

The visualization data preparation and path-direction semantics can be tested without Docker, Neo4j, or Ollama:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py src scripts tests
bash -n scripts/download_hetionet.sh
```

These tests verify that the selected-path graph deduplicates repeated edges, displays each KG relation in its stored direction, adds an undirected dashed connector only for `causal`, and still shows the two selected endpoints when no path exists. Provider tests additionally verify that `LLM_PROVIDER` routes to the right endpoint shape (Ollama raw vs OpenAI-compatible, with/without bearer key) and that the paper-style prompt matches the reproduction repo's `tcma` template. They use a lightweight PyVis mock for the renderer; actual browser rendering still requires installing dependencies and starting Docker Compose.

## Useful upstream references

- Paper: https://doi.org/10.1145/3711896.3737076
- Authors' repo and limitations: https://github.com/susantiyuni/path-to-causality
- Hetionet v1.0 source and formats: https://github.com/hetio/hetionet
- Ollama model library (Qwen2.5): https://registry.ollama.com/library/qwen2.5
