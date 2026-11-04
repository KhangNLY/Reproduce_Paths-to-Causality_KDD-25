#!/usr/bin/env python
"""Chạy FULL project (Module 1 -> 2 -> 3) bằng một lệnh, qua OpenRouter. Mặc định chạy 2 model: Llama-3.1-8B và Ministral-8B.

  python run_full.py --print-only            # chỉ in các lệnh sẽ chạy (không tốn gì)
  python run_full.py --smoke                 # thử nhanh: 1 task, 5 cặp, 1 model (vài chục lần gọi)
  python run_full.py                         # chạy đủ (Module 1 dùng dữ liệu tác giả đã chấm sẵn)
  python run_full.py --rescore-m1            # Module 1 ĐỘC LẬP: tự chấm lại metapath bằng LLM (~12.6k lần gọi), rồi chạy Module 2/3 trên dữ liệu đó
  python run_full.py --with-full-eval        # thêm protocol C: bộ câu hỏi lớn (--eval-split full, ranker out-of-fold) - tốn ~10x số lần gọi
  python run_full.py --only m3               # chỉ chạy các bước Module 3 (resume được: prompt đã có kết quả sẽ bỏ qua)

Các bước: m1 | m1score | m2eval | m2xgb | m2ce | m2cv | rankgpt | m3 | m3full | summary
Cần: OPENROUTER_API_KEY trong môi trường hoặc file .env. Cross-encoder tự bỏ qua nếu không có GPU CUDA (dùng --force-ce để ép chạy CPU).
Log: logs/run_<timestamp>/ (log của run_full và của MỖI bước con).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def reexec_in_venv():
    """Nếu có .venv của dự án mà đang chạy bằng Python khác (vd Python hệ thống) -> chạy lại chính script này bằng .venv.
    Tránh lỗi 'No module named ...' do quên kích hoạt venv. Tắt bằng PTC_NO_REEXEC=1."""
    venv = ROOT / ".venv"
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if os.environ.get("PTC_NO_REEXEC") or not py.exists():
        return
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"## Đang chạy bằng {sys.executable}\n## -> chuyển sang môi trường của dự án: {py}", flush=True)
    sys.exit(subprocess.call([str(py), *sys.argv], cwd=ROOT))


if __name__ == "__main__":
    reexec_in_venv()
import lib  # noqa: E402,F401  (nạp .env)
from lib import runlog  # noqa: E402
from lib.prompts import CHAT_ANSWER_SUFFIX, CHAT_SYSTEM_PROMPT  # noqa: E402

STEPS = ["m1", "m1score", "m2eval", "m2xgb", "m2ce", "m2cv", "rankgpt", "m3", "m3full", "summary"]
DEFAULT_STEPS = ["m1", "m2eval", "m2xgb", "m2ce", "rankgpt", "m3", "summary"]
DEFAULT_MODELS = ["meta-llama/llama-3.1-8b-instruct@llama", "mistralai/ministral-8b-2512@mistral"]
BASE_URL = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def cuda_ok():
    r = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                       capture_output=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="dạng <id_model>@<style>; style: llama|mistral|gemma")
    ap.add_argument("--only", nargs="+", choices=STEPS, help="chỉ chạy các bước này")
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="bỏ qua các bước này")
    ap.add_argument("--smoke", action="store_true", help="thử nhanh: comagc, 5 cặp, model đầu tiên, không CE/RankGPT")
    ap.add_argument("--rescore-m1", action="store_true", help="Module 1 độc lập: tự chấm điểm metapath (m1_score_metapaths.py), các bước sau dùng dữ liệu đó")
    ap.add_argument("--m1-model", default="mistralai/mistral-nemo", help="model chấm điểm Module 1 (cần provider có logprobs; xem README)")
    ap.add_argument("--m1-style", default="mistral")
    ap.add_argument("--with-full-eval", action="store_true", help="thêm protocol C: m2 --cv-folds + m3 --eval-split full (bộ câu hỏi lớn)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--with-rankgpt", action="store_true", help="chạy lại RankGPT (gpt-4o-mini) để có option gpt_rerank")
    ap.add_argument("--rankgpt-model", default="openai/gpt-4o-mini")
    ap.add_argument("--with-authors-rankers", action="store_true", help="thêm option authors_ranker:* (ranker tác giả công bố) vào protocol A")
    ap.add_argument("--save-ce-weights", action="store_true", help="lưu trọng số cross-encoder (12 model x ~500MB = ~6GB)")
    ap.add_argument("--force-ce", action="store_true", help="ép huấn luyện cross-encoder dù không có CUDA (rất chậm)")
    ap.add_argument("--api-mode", choices=["chat", "completion"], default="chat",
                    help="chat (mặc định, kèm câu chỉ dẫn trả lời 1 từ) | completion (prompt thô như bài báo; mỗi model/provider xử lý khác nhau, hay bị lan man)")
    ap.add_argument("--answer-suffix", default=CHAT_ANSWER_SUFFIX,
                    help="nối vào cuối mọi prompt khi --api-mode chat; đặt \"\" để tắt")
    ap.add_argument("--system-prompt", default=CHAT_SYSTEM_PROMPT,
                    help="system prompt khi --api-mode chat (mặc định ép trả lời 1 từ; thử nghiệm cho thấy thiếu nó Llama-3.1 hay mở đầu \"Based on...\"/\"## Step 1\" -> không có nhãn)")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = tự chọn (chat: 8, completion: 32)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    ts = runlog.now_ts()
    os.environ["PTC_PARENT_RUN"] = ts                           # log của các bước con nằm chung logs/run_<ts>/
    if not args.print_only:
        runlog.start("run_full")

    steps = args.only or list(DEFAULT_STEPS)
    if args.rescore_m1 and "m1score" not in steps:
        steps.insert(steps.index("m1") + 1 if "m1" in steps else 0, "m1score")
    if args.with_full_eval:
        for extra in ("m2cv", "m3full"):
            if extra not in steps:
                steps.insert(steps.index("summary") if "summary" in steps else len(steps), extra)
    run_steps = [s for s in STEPS if s in steps and s not in args.skip]
    models = args.models[:1] if args.smoke else args.models
    tasks = ["comagc"] if args.smoke else ["comagc", "ade", "gene", "semeval"]
    limit = ["--limit", "5"] if args.smoke else []
    if args.smoke:
        run_steps = [s for s in run_steps if s in ("m1", "m2xgb", "m3", "summary")]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    import importlib.util
    need = [m for m in ("sentence_transformers",) if any(x in run_steps for x in ("m3", "m3full")) and importlib.util.find_spec(m) is None]
    if need and not args.print_only:
        sys.exit(f"Thiếu thư viện {need} trong {sys.executable}. Cài môi trường: python setup_env.py (hoặc python -m pip install -r requirements.txt)")
    uses_api = any(s in run_steps for s in ("m1score", "m3", "m3full", "rankgpt"))
    if uses_api and not args.print_only and not os.environ.get(KEY_ENV):
        sys.exit(f"Thiếu {KEY_ENV}: đặt trong PowerShell ($env:{KEY_ENV}='...') hoặc ghi vào file .env (xem .env.example)")

    def run(*a):
        cmd = [sys.executable, *a]
        print("\n$ " + " ".join(cmd), flush=True)
        if not args.print_only:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    data = []            # --data-dir cho các bước sau Module 1 khi dùng dữ liệu tự chấm
    api = ["--backend", "openai", "--base-url", BASE_URL, "--api-key-env", KEY_ENV, "--concurrency", str(args.concurrency)]
    t0 = time.time()
    if "m1" in run_steps:
        run("m1_inspect_scored_data.py")
    if "m1score" in run_steps:
        tag = f"run_full_{slug(args.m1_model.split('/')[-1])}"
        run("m1_score_metapaths.py", *api, "--model", args.m1_model, "--style", args.m1_style, "--tasks", *tasks,
            "--run-tag", tag, "--max-new-tokens", "8", *(["--limit-pairs", "5"] if args.smoke else []))
        found = sorted((ROOT / "outputs/module1/rescored").glob(f"{tag}_*"), key=lambda p: p.stat().st_mtime) if not args.print_only else []
        d = found[-1] if found else Path("outputs/module1/rescored") / f"{tag}_<timestamp>"
        data = ["--data-dir", str(d)]
        print(f"## Các bước sau dùng dữ liệu tự chấm: {d}")
        run("m1_inspect_scored_data.py", *data, "--tasks", *tasks)
    out_root = ROOT / "outputs"
    if data:
        out_root = ROOT / "outputs" / f"data_{Path(data[1]).name}"
    if "m2eval" in run_steps and not data:
        run("m2_eval_ranked.py", "--authors")
    ce_rankers = ["ce_rmse", "ce_ranknet", "ce_listnet"]
    have_ce = False
    if "m2xgb" in run_steps:
        run("m2_train_ranker.py", *data, "--tasks", *tasks, "--rankers", "xgboost")
    if "m2ce" in run_steps:
        if args.print_only or args.force_ce or cuda_ok():
            run("m2_train_ranker.py", *data, "--tasks", *tasks, "--rankers", *ce_rankers, "--epochs", str(args.epochs),
                *(["--save-model"] if args.save_ce_weights else []))
            have_ce = True
        else:
            print("\n(!) Bỏ qua cross-encoder: không có CUDA. Cài torch CUDA (python setup_env.py --cuda cu126) hoặc dùng --force-ce.")
    have_ce = have_ce or all((out_root / "module2" / t / r / "test_ranked_all.jsonl").exists() for t in tasks for r in ce_rankers)
    if "m2cv" in run_steps:
        run("m2_train_ranker.py", *data, "--tasks", *tasks, "--rankers", "xgboost", "--cv-folds", str(args.cv_folds))

    have_gpt = False
    if "rankgpt" in run_steps and args.with_rankgpt and not args.smoke:
        run("tools/rankgpt_rerank.py", *data, *api, "--model", args.rankgpt_model, "--tasks", *tasks, "--pool", "all")
        have_gpt = True
    have_gpt = have_gpt or all((out_root / "rankgpt" / t / "ranked_all.jsonl").exists() for t in tasks)

    def chat_extra():
        mt = args.max_new_tokens or (8 if args.api_mode == "chat" else 32)
        extra = ["--api-mode", args.api_mode, "--max-new-tokens", str(mt)]
        if args.api_mode == "chat" and args.system_prompt:
            extra += ["--system-prompt", args.system_prompt]
        if args.api_mode == "chat" and args.answer_suffix:
            extra += ["--prompt-suffix", args.answer_suffix.replace("\n", "\\n")]
        return extra

    if "m3" in run_steps or "m3full" in run_steps:
        if not args.no_probe and not args.print_only:
            for m in models:
                run("tools/openrouter_models.py", "--probe", m.rsplit("@", 1)[0])
        rankers = ["ranker:xgboost"] + ([f"ranker:{r}" for r in ce_rankers] if have_ce else [])
        optA = ["no_subgraph", "no_subgraph_pair", "random", "similarity", "gpt", *rankers]
        if args.with_authors_rankers:
            optA += [f"authors_ranker:{n}" for n in ("xgboost", "rmse", "ranknet", "listnet")]
        optB = ["no_subgraph", "random", "similarity_recompute", *rankers] + (["gpt_rerank"] if have_gpt else [])
        optC = ["no_subgraph", "random", "similarity_recompute", "ranker:xgboost"]        # protocol C: chỉ ranker có file out-of-fold
        for m in models:
            mid, style = (m.rsplit("@", 1) + ["mistral"])[:2] if "@" in m else (m, "mistral")
            plan = []
            if "m3" in run_steps:
                plan += [("oracle_top5", optA, "test"), ("all", optB, "test")]
            if "m3full" in run_steps:
                plan += [("all", optC, "full")]
            for pool, opts, split in plan:
                tag = f"{'smoke_' if args.smoke else ''}or_{slug(mid.split('/')[-1])}_{args.api_mode[:4]}_{pool}_k1" + ("_full" if split == "full" else "")
                run("m3_zero_shot.py", *data, *api, "--model", mid, "--style", style, "--pool", pool, "--tasks", *tasks,
                    "--eval-split", split, "--options", *opts, "--run-tag", tag, *chat_extra(), *limit)
    if "summary" in run_steps:
        run("tools/summarize_results.py", *(["--root", str(out_root)] if data else []))
    print(f"\nXong ({'chỉ in lệnh' if args.print_only else f'{(time.time() - t0) / 60:.1f} phút'}). Bảng: {out_root.name}/SUMMARY.md và {out_root.name}/module3/<run>/summary.md")
    if not args.print_only:
        print(f"Log: logs/run_{ts}/")


if __name__ == "__main__":
    main()
