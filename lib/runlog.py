"""Ghi log ra FILE trong lúc script đang chạy (tee stdout/stderr) + tiện ích timestamp cho mọi artifact.

Cách dùng (đầu hàm main của mỗi script):

    from lib import runlog
    runlog.start("m3_zero_shot")          # -> logs/m3_zero_shot_<YYYYmmdd-HHMMSS>.log   (ghi + flush từng dòng)

* Mọi thứ in ra console (print, warning, traceback, thanh tiến trình tqdm/HF) đều được ghi kèm timestamp.
* Thanh tiến trình dùng '\\r' chỉ ghi trạng thái cuối cùng của mỗi dòng (log không bị phình).
* `runlog.file_only(msg)` : chỉ ghi vào file (vd chi tiết từng câu hỏi/câu trả lời), không in ra console.
* Chuỗi giống API key (sk-..., hf_..., "Bearer ...") bị che thành *** trước khi ghi vào file.
* Tắt log: biến môi trường PTC_NO_LOG=1. Đổi thư mục log: PTC_LOG_DIR=... (mặc định My_experiments/logs/).
* run_full.py / dry_run_all.py đặt PTC_PARENT_RUN=<timestamp> -> log của các bước con nằm chung thư mục logs/run_<timestamp>/.
"""
from __future__ import annotations

import atexit
import os
import platform
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import paths

LOG_DIR_DEFAULT = paths.ROOT / "logs"

_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{10,}|(?i:bearer)\s+[A-Za-z0-9._\-]{12,})")

_state: dict = {"path": None, "fh": None, "lock": threading.Lock(), "t0": None, "script": None}


def now_ts() -> str:
    """Timestamp dùng trong tên file/thư mục: 20261104-193012 (giờ máy)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def iso_now() -> str:
    """Timestamp ISO-8601 có múi giờ, dùng trong nội dung JSON."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def mask(s: str) -> str:
    return _SECRET.sub("***", s)


def stamped(path, ts: Optional[str] = None) -> Path:
    """a/b/metrics.json -> a/b/metrics_<ts>.json   (thư mục: a/b/model -> a/b/model_<ts>)."""
    p = Path(path)
    ts = ts or now_ts()
    return p.with_name(f"{p.stem}_{ts}{p.suffix}")


def log_path() -> Optional[Path]:
    return _state["path"]


def _write_line(prefix: str, line: str) -> None:
    fh = _state["fh"]
    if fh is None:
        return
    with _state["lock"]:
        try:
            fh.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {prefix} {mask(line)}\n")
            fh.flush()
        except Exception:        # log không bao giờ được làm hỏng thí nghiệm
            pass


class _Tee:
    def __init__(self, stream, prefix: str):
        self._stream, self._prefix, self._buf = stream, prefix, ""

    def write(self, s):
        try:
            n = self._stream.write(s)
        except Exception:
            n = len(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rsplit("\r", 1)[-1]              # thanh tiến trình: giữ trạng thái cuối
            if line.strip():
                _write_line(self._prefix, line)
        if len(self._buf) > 20000:                       # dòng '\r' vô tận: tránh phình bộ nhớ
            self._buf = self._buf[-2000:]
        return n

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):                         # isatty, encoding, reconfigure...
        return getattr(self._stream, name)


def file_only(msg: str) -> None:
    """Ghi 1 dòng chỉ vào file log (không in console)."""
    _write_line("|f", msg)


def start(script: str, log_dir: Optional[str] = None, argv: Optional[list] = None) -> Optional[Path]:
    """Bắt đầu ghi log. Trả về đường dẫn file log (hoặc None nếu bị tắt / đã bắt đầu)."""
    if _state["fh"] is not None or os.environ.get("PTC_NO_LOG") == "1":
        return _state["path"]
    parent = os.environ.get("PTC_PARENT_RUN", "")
    d = Path(log_dir or os.environ.get("PTC_LOG_DIR") or LOG_DIR_DEFAULT)
    if parent:
        d = d / f"run_{parent}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{script}_{now_ts()}.log"
    if p.exists():
        p = d / f"{script}_{now_ts()}_{os.getpid()}.log"
    fh = open(p, "a", encoding="utf-8", errors="replace")
    _state.update(path=p, fh=fh, t0=time.time(), script=script)
    sys.stdout = _Tee(sys.stdout, "|")                   # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, "|E")                  # type: ignore[assignment]
    argv = sys.argv if argv is None else argv
    _write_line("#", f"script      : {script}")
    _write_line("#", f"started     : {iso_now()}")
    _write_line("#", f"command     : {' '.join(argv)}")
    _write_line("#", f"cwd         : {os.getcwd()}")
    _write_line("#", f"python      : {sys.version.split()[0]} | {platform.platform()}")
    if parent:
        _write_line("#", f"parent run  : {parent}")
    prev_hook = sys.excepthook

    def _hook(tp, val, tb):                              # đánh dấu lần chạy bị lỗi (traceback vẫn đi qua stderr -> log)
        _state["failed"] = not issubclass(tp, (KeyboardInterrupt, SystemExit)) or tp is KeyboardInterrupt
        prev_hook(tp, val, tb)

    sys.excepthook = _hook
    atexit.register(_finish)
    print(f"## Log file: {p}", flush=True)
    return p


def _finish() -> None:
    fh = _state["fh"]
    if fh is None:
        return
    dt = time.time() - (_state["t0"] or time.time())
    _write_line("#", f"finished    : {iso_now()}  (elapsed {dt:.1f}s){'  [ABORTED/EXCEPTION - xem traceback phía trên]' if _state.get('failed') else ''}")
    try:
        fh.close()
    except Exception:
        pass
    _state["fh"] = None


def log_exception(prefix: str = "uncaught exception") -> None:
    _write_line("|E", prefix + ":\n" + traceback.format_exc())
