#!/usr/bin/env python
"""Kiểm tra TOÀN BỘ pipeline ở chế độ dry-run (LLM giả, model nhỏ ngẫu nhiên, tập con nhỏ).
Không tải model lớn, không gọi API. Kết quả ghi vào outputs/_dryrun/ (xoá được); log: logs/run_<timestamp>/.

  python dry_run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
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
from lib import runlog  # noqa: E402

TS = runlog.now_ts()
os.environ["PTC_PARENT_RUN"] = TS
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def run(*args):
    cmd = [sys.executable, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def has(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0


def main():
    runlog.start("dry_run_all")
    rankers = ["xgboost"]
    if has("transformers"):
        rankers += ["ce_rmse", "ce_ranknet", "ce_listnet"]
    else:
        print("(bỏ qua dry-run cross-encoder: chưa cài transformers)")
    run("m1_inspect_scored_data.py")
    run("m1_score_metapaths.py", "--dry-run", "--tasks", "comagc", "semeval", "--run-tag", "dry_m1")
    run("m2_eval_ranked.py", "--authors")
    run("m2_train_ranker.py", "--dry-run", "--tasks", "comagc", "semeval", "--rankers", *rankers)
    run("m2_train_ranker.py", "--dry-run", "--tasks", "comagc", "--rankers", "xgboost", "--cv-folds", "3")
    run("tools/rankgpt_rerank.py", "--dry-run", "--tasks", "comagc", "semeval", "--pool", "all")
    run("tools/rankgpt_rerank.py", "--dry-run", "--tasks", "comagc", "semeval", "--pool", "oracle_top5")
    run("m3_zero_shot.py", "--dry-run", "--tasks", "comagc", "semeval", "--pool", "oracle_top5", "--run-tag", "dry_top5", "--force")
    run("m3_zero_shot.py", "--dry-run", "--tasks", "comagc", "semeval", "--pool", "all", "--run-tag", "dry_all", "--force",
        "--options", "no_subgraph", "random", "similarity", "similarity_recompute", "gpt_rerank", "ranker:xgboost", "oracle")
    run("m3_zero_shot.py", "--dry-run", "--tasks", "comagc", "--pool", "all", "--eval-split", "full", "--run-tag", "dry_full", "--force",
        "--prompt-variants", "0", "1", "--options", "no_subgraph", "random", "similarity_recompute", "ranker:xgboost")
    run("tools/summarize_results.py", "--root", "outputs/_dryrun")
    print("\nDry-run xong: không lỗi.")


if __name__ == "__main__":
    main()
