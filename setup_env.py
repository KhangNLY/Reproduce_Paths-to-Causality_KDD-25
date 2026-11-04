#!/usr/bin/env python
"""Cài môi trường cho My_experiments bằng Python (thay cho script .ps1).

  python setup_env.py                       # tạo .venv, cài torch (mặc định pip) + requirements.txt   (profile full)
  python setup_env.py --cuda cu126          # cài torch bản CUDA 12.6 (cần cho GPU; xem https://pytorch.org/get-started/locally/)
  python setup_env.py --profile api         # chỉ phần tối thiểu: Module 1, XGBoost, Module 3 qua API/OpenRouter (không transformers)
  python setup_env.py --no-venv             # cài thẳng vào python đang chạy (không tạo .venv)
  python setup_env.py --dry-run             # chỉ in các lệnh sẽ chạy

Cách thủ công tương đương (PowerShell/cmd):
  python -m venv .venv
  .venv\\Scripts\\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126    # bước CUDA (tuỳ chọn)
  .venv\\Scripts\\python -m pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_ONLY = ["numpy>=1.24", "requests>=2.31", "tqdm>=4.66", "xgboost>=2.0", "torch>=2.1"]


def run(cmd, dry):
    print("$", " ".join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["full", "api"], default="full",
                    help="full = + transformers, sentence-transformers, accelerate, bitsandbytes | api = tối thiểu")
    ap.add_argument("--cuda", default="none", help="none | cu118 | cu121 | cu126 | cu128 | cpu  (bản torch)")
    ap.add_argument("--venv-dir", default=".venv")
    ap.add_argument("--no-venv", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from lib import runlog
    runlog.start("setup_env")

    if args.no_venv:
        py = Path(sys.executable)
    else:
        venv = ROOT / args.venv_dir
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not py.exists():
            run([sys.executable, "-m", "venv", venv], args.dry_run)
    run([py, "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)

    # 1) torch: bản mặc định trên Windows là CPU-only -> chọn --cuda để dùng GPU
    if args.cuda not in ("none",):
        # torch CPU đã cài sẽ được pip coi là "đã thỏa mãn" -> phải gỡ trước khi cài bản CUDA
        run([py, "-m", "pip", "uninstall", "-y", "torch"], args.dry_run)
        run([py, "-m", "pip", "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{args.cuda}"], args.dry_run)

    # 2) phần còn lại
    if args.profile == "full":
        run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"], args.dry_run)
    else:
        run([py, "-m", "pip", "install", *API_ONLY], args.dry_run)

    if args.dry_run:
        return
    check = (
        "import importlib,sys\n"
        "for m in ['numpy','requests','tqdm','xgboost','torch','transformers','sentence_transformers','accelerate','bitsandbytes']:\n"
        "    try:\n"
        "        mod=importlib.import_module(m); print(f'  OK   {m:22s}', getattr(mod,'__version__',''))\n"
        "    except Exception as e:\n"
        "        print(f'  --   {m:22s} (không có: {type(e).__name__})')\n"
        "import torch\n"
        "print('torch CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
    )
    print("\n== Kiểm tra môi trường ==")
    subprocess.run([str(py), "-c", check], check=False)
    print(f"\nXong. Chạy bằng:  {py} m1_inspect_scored_data.py   (hoặc kích hoạt venv rồi dùng `python`)")
    print("Nếu 'torch CUDA: False' mà bạn có GPU NVIDIA: chạy lại với --cuda cu126 (hoặc bản phù hợp).")


if __name__ == "__main__":
    main()




