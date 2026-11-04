#!/usr/bin/env python
"""Tải dữ liệu của tác giả (repo susantiyuni/path-to-causality, commit ghim cố định) vào data/authors/ và kiểm tra checksum.

Repo gốc không khai báo license nên thư mục data/authors/ KHÔNG nằm trong git của project này - ai clone project đều chạy lệnh này một lần:

  python tools/fetch_authors_data.py                 # tải + giải nén + kiểm tra SHA-256
  python tools/fetch_authors_data.py --verify-only   # chỉ kiểm tra file đang có
  python tools/fetch_authors_data.py --make-manifest # (người bảo trì) ghi lại tools/authors_data_manifest.json từ data/authors hiện có

Lấy đúng các file Module 1/2/3 cần: datasets/<task>/{train_full,test_truth,test_sim,test_gpt_*}.jsonl và
checkpoints/<task>/*/{params.txt,test_pred_*.jsonl}.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import paths, runlog  # noqa: E402

REPO = "susantiyuni/path-to-causality"
COMMIT = "6f9f701eb24bc1173581d4d6ab93e5533cf48b5e"
URL = f"https://github.com/{REPO}/archive/{COMMIT}.zip"
MANIFEST = Path(__file__).resolve().parent / "authors_data_manifest.json"
WANT = ("datasets/", "checkpoints/")
KEEP_SUFFIX = (".jsonl", "params.txt")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wanted(rel: str) -> bool:
    if not rel.startswith(WANT) or not rel.endswith(KEEP_SUFFIX):
        return False
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith("datasets/"):
        return name in ("train_full.jsonl", "test_truth.jsonl", "test_sim.jsonl") or name.startswith("test_gpt_")
    return name == "params.txt" or name.startswith("test_pred_")


def verify() -> bool:
    if not MANIFEST.exists():
        print("(!) chưa có tools/authors_data_manifest.json -> bỏ qua kiểm tra checksum")
        return True
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = [rel for rel, h in man["files"].items() if not (paths.AUTHORS / rel).exists() or sha256(paths.AUTHORS / rel) != h]
    if bad:
        print(f"!! {len(bad)}/{len(man['files'])} file thiếu hoặc sai checksum, ví dụ: {bad[:3]}")
        return False
    print(f"OK: {len(man['files'])} file khớp checksum (commit {man['commit'][:7]})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--make-manifest", action="store_true")
    ap.add_argument("--force", action="store_true", help="tải lại cả khi dữ liệu đã đủ và đúng checksum")
    args = ap.parse_args()
    runlog.start("fetch_authors_data")

    if args.make_manifest:
        files = {p.relative_to(paths.AUTHORS).as_posix(): sha256(p) for p in sorted(paths.AUTHORS.rglob("*"))
                 if p.is_file() and wanted(p.relative_to(paths.AUTHORS).as_posix())}
        MANIFEST.write_text(json.dumps({"repo": REPO, "commit": COMMIT, "files": files}, indent=1), encoding="utf-8")
        print(f"đã ghi {MANIFEST} ({len(files)} file)")
        return
    if args.verify_only:
        sys.exit(0 if verify() else 1)
    if not args.force and MANIFEST.exists() and verify():
        print("Dữ liệu đã đủ, không cần tải. (dùng --force để tải lại)")
        return

    print(f"Tải {URL} ...", flush=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    print(f"  {len(blob) / 1e6:.1f} MB; giải nén các file cần thiết vào {paths.AUTHORS}", flush=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        root = z.namelist()[0].split("/")[0]
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = info.filename[len(root) + 1:]
            if wanted(rel):
                dest = paths.AUTHORS / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(info))
                n += 1
    (paths.AUTHORS / "SOURCE.txt").write_text(f"https://github.com/{REPO}\ncommit {COMMIT}\nfetched by tools/fetch_authors_data.py\n", encoding="utf-8")
    print(f"  đã ghi {n} file")
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()

