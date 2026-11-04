"""Đọc file .env tối giản (không cần thư viện ngoài). Dòng dạng KEY=VALUE, '#' là chú thích."""
from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = ENV_PATH) -> list[str]:
    """Trả về danh sách TÊN biến đã nạp (không bao giờ in giá trị). Biến đã có trong môi trường được ưu tiên."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded

