"""Copy bộ artifact ranker do repo chính train vào demo/models/repo_ranker.

Nguồn nhận được ở 3 dạng:
  1. Thư mục model_<timestamp> của repo chính (sau `m2_train_ranker.py
     --rankers xgboost`): outputs/module2/<task>/<tag>/model_<ts>/
  2. File .zip chứa bộ 3 artifact (xgb_ranker.json + ngram_embeddings.npz +
     config.json), zip ở gốc HOẶC lồng trong 1 thư mục con (case GitHub zip:
     <tên-repo>-main/outputs/.../model_<ts>/...)
  3. Bỏ --src: tự dò ../outputs của repo cha, lấy model mới nhất.

Ví dụ:
  python scripts/import_repo_ranker.py                       # tự dò outputs/ của repo cha
  python scripts/import_repo_ranker.py --src ../outputs/module2/comagc/xgb/model_20260101_120000
  python scripts/import_repo_ranker.py --src ~/Downloads/ranker_comagc.zip
  python scripts/import_repo_ranker.py --src <dir|zip> --task comagc   # đặt tên task trong PROVENANCE
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = ("xgb_ranker.json", "ngram_embeddings.npz", "config.json")


def _check_required(dir_path: Path) -> bool:
    return all((dir_path / f).exists() for f in REQUIRED)


def resolve_zip_source(zip_path: Path) -> Path:
    """Giải nén zip vào thư mục tạm và định vị thư mục chứa đủ 3 artifact.

    Chấp nhận zip có artifact ở gốc hoặc lồng bất kỳ độ sâu nào; nếu nhiều nơi
    chứa đủ bộ (nhiều task trong 1 zip), chọn nơi có timestamp mới nhất trong
    tên thư mục model_*.
    """
    tmp = Path(tempfile.mkdtemp(prefix="repo_ranker_zip_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    candidates = [d for d in tmp.rglob("config.json")
                  if _check_required(d.parent)]
    if not candidates:
        # cũng chấp nhận zip mà người đóng gói dồn 3 file vào gốc nhưng thiếu config
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(
            f"Trong {zip_path.name} không tìm thấy bộ đủ {', '.join(REQUIRED)} "
            "(tìm cả ở gốc lẫn các thư mục con)."
        )
    if len(candidates) == 1:
        return candidates[0].parent
    newest = max(candidates, key=lambda c: c.parent.name)  # model_<ts>: sort tên = thời gian
    print(f"Zip chứa {len(candidates)} bộ artifact; chọn {newest.parent} (timestamp mới nhất).")
    return newest.parent


def find_latest_model_dir(outputs_root: Path) -> tuple[Path, str] | None:
    """Dò outputs/module2/*/*/model_* ; trả (dir, task) mới nhất theo tên timestamp."""
    best: tuple[Path, str, str] | None = None
    module2 = outputs_root / "module2"
    roots = [module2] if module2.exists() else []
    # outputs/data_<tên>/module2/... khi chạy m2 với --data-dir
    roots += sorted(p / "module2" for p in outputs_root.glob("data_*") if (p / "module2").exists())
    for m2 in roots:
        for model_dir in m2.glob("*/*/model_*"):
            if not _check_required(model_dir):
                continue
            task = model_dir.parts[model_dir.parts.index("module2") + 1]
            key = model_dir.name  # model_<timestamp>: sort theo tên = theo thời gian
            if best is None or key > best[2]:
                best = (model_dir, task, key)
    return (best[0], best[1]) if best else None


def _guess_task_from_path(p: Path) -> str:
    """Suy tên task từ đường dẫn chuẩn .../module2/<task>/<tag>/model_<ts>... ; không khớp thì 'default'."""
    parts = [x.name if isinstance(x, Path) else str(x) for x in p.parts]
    if "module2" in parts:
        i = parts.index("module2")
        if i + 1 < len(parts) and parts[i + 1].startswith("model_") is False:
            return parts[i + 1]
    return "default"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None,
                    help="thư mục model_<timestamp> HOẶC file .zip chứa bộ artifact")
    ap.add_argument("--outputs", default=str((ROOT / ".." / "outputs").resolve()),
                    help="thư mục outputs/ của repo chính (mặc định: ../outputs so với demo/)")
    ap.add_argument("--task", default=None, help="chỉ dùng cho PROVENANCE khi dò tự động (mặc định: lấy từ đường dẫn nguồn)")
    ap.add_argument("--dest-dir", default="models/repo_ranker",
                    help="thư mục đích trong demo (mặc định: models/repo_ranker, khớp REPO_RANKER_DIR)")
    args = ap.parse_args()

    if args.src:
        src_path = Path(args.src).expanduser()
        if not src_path.exists():
            raise SystemExit(f"Không tìm thấy {src_path}")
        if src_path.suffix.lower() == ".zip":
            src = resolve_zip_source(src_path)
            task = args.task or _guess_task_from_path(src)
            atexit_register_cleanup(src)
        else:
            src = src_path.resolve()
            task = args.task or "default"
            if not _check_required(src):
                raise SystemExit(f"Thiếu artifact trong {src}; cần đủ: {', '.join(REQUIRED)}")
    else:
        found = find_latest_model_dir(Path(args.outputs).resolve())
        if not found:
            raise SystemExit(
                f"Không tìm thấy bộ artifact nào trong {args.outputs}. Hãy chạy repo chính: "
                "python m2_train_ranker.py --rankers xgboost --tasks <task> rồi chạy lại script này, "
                "hoặc chỉ định --src <thư mục model_<ts> | file .zip>."
            )
        src, task = found
        if args.task:
            task = args.task
        print(f"Tìm thấy artifact mới nhất: {src} (task={task})")

    dst = ROOT / args.dest_dir
    dst.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED:
        shutil.copy2(src / f, dst / f)
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    (dst / "PROVENANCE.txt").write_text(
        f"source: {src}\ntask: {task}\nconfig: {json.dumps(cfg, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    print(f"Đã copy {len(REQUIRED)} file + PROVENANCE.txt -> {dst}")
    print("Cập nhật .env của demo:  REPO_RANKER_DIR=" + str(dst.relative_to(ROOT)))
    print("Rồi khởi động lại Streamlit; UI sẽ báo 'Repo XGB rank:ndcg + n-gram 128d'.")


def atexit_register_cleanup(tmp_dir: Path) -> None:
    import atexit
    atexit.register(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))


if __name__ == "__main__":
    main()

