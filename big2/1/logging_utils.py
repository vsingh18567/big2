from __future__ import annotations

import atexit
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_ACTIVE_LOG_PATH: Path | None = None
_ACTIVE_LOG_HANDLE: TextIO | None = None


class TeeTextIO:
    def __init__(self, *streams: TextIO):
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
        self.errors = getattr(streams[0], "errors", "strict")

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self) -> int:
        return self._streams[0].fileno()


def _close_active_log_handle() -> None:
    global _ACTIVE_LOG_HANDLE
    if _ACTIVE_LOG_HANDLE is None:
        return
    _ACTIVE_LOG_HANDLE.flush()
    _ACTIVE_LOG_HANDLE.close()
    _ACTIVE_LOG_HANDLE = None


atexit.register(_close_active_log_handle)


def log_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def configure_process_logging(log_path: str | Path) -> Path:
    global _ACTIVE_LOG_HANDLE, _ACTIVE_LOG_PATH

    resolved_log_path = Path(log_path).resolve()
    if _ACTIVE_LOG_PATH == resolved_log_path and _ACTIVE_LOG_HANDLE is not None:
        return resolved_log_path

    _close_active_log_handle()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_LOG_HANDLE = resolved_log_path.open("a", buffering=1)
    sys.stderr = TeeTextIO(_ORIGINAL_STDERR, _ACTIVE_LOG_HANDLE)
    _ACTIVE_LOG_PATH = resolved_log_path
    return resolved_log_path
