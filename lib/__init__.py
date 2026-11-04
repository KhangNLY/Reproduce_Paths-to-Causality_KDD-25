"""Thư viện dùng chung cho My_experiments (tái lập 'Paths to Causality', KDD'25)."""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):     # console Windows (cp1252) không in được tiếng Việt -> ép UTF-8, ký tự lạ thì thay thế
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .envfile import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()          # nạp My_experiments/.env (nếu có) vào biến môi trường; không ghi đè biến đã đặt sẵn
