#!/usr/bin/env python
"""Đổi tên/sao chép kết quả CŨ (chạy trước khi có timestamp) sang quy ước mới, dựa vào thời điểm sửa file:

  outputs/module2/<task>/<tag>/model/            -> model_<ts>/           (đổi tên; trọng số không bị sao chép)
  outputs/module2/<task>/<tag>/metrics.json ...  -> thêm bản metrics_<ts>.json, test_ranked_*_<ts>.jsonl, latest.json (nếu chưa có)

An toàn khi chạy lại (bỏ qua thứ đã có timestamp). Chỉ dùng được cho thư mục outputs của project.

  python tools/stamp_legacy_outputs.py --dry-run     # xem trước
  python tools/stamp_legacy_outputs.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import paths, runlog  # noqa: E402


def ts_of(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d-%H%M%S")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(paths.OUTPUTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    runlog.start("stamp_legacy_outputs")
    n = 0
    for tag_dir in sorted(Path(args.root).glob("module2/*/*")):
        if not tag_dir.is_dir():
            continue
        mdir = tag_dir / "model"
        metrics = tag_dir / "metrics.json"
        ts_m = ts_of(metrics) if metrics.exists() else None
        wdir_new = None
        if mdir.is_dir():
            wdir_new = tag_dir / f"model_{ts_of(mdir)}"
            print(f"rename  {mdir}  ->  {wdir_new.name}")
            if not args.dry_run:
                mdir.rename(wdir_new)
            n += 1
        if ts_m and not list(tag_dir.glob("metrics_*.json")):
            for name, ext in (("metrics", ".json"), ("test_ranked_top5", ".jsonl"), ("test_ranked_all", ".jsonl")):
                src = tag_dir / f"{name}{ext}"
                if src.exists():
                    print(f"copy    {src.name}  ->  {name}_{ts_m}{ext}")
                    if not args.dry_run:
                        shutil.copyfile(src, tag_dir / f"{name}_{ts_m}{ext}")
            if not args.dry_run and not (tag_dir / "latest.json").exists():
                existing = sorted(tag_dir.glob("model_*"))
                (tag_dir / "latest.json").write_text(json.dumps({"timestamp": ts_m, "weights_dir": str(existing[-1]) if existing else None,
                                                                 "metrics": f"metrics_{ts_m}.json", "note": "migrated from legacy layout"}, indent=1),
                                                     encoding="utf-8")
            n += 1
    print(f"xong ({n} thay đổi{', dry-run: chưa ghi gì' if args.dry_run else ''})")


if __name__ == "__main__":
    main()

