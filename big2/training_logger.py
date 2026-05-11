from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


class TrainingHistoryLogger:
    """Writes per-batch training history to JSON so interrupted runs still leave usable data."""

    def __init__(self, path: str | Path | None, *, metadata: dict[str, Any], write_every: int = 1):
        self.path = Path(path) if path is not None else None
        self.write_every = max(1, write_every)
        self.started_at = time.time()
        self.data: dict[str, Any] = {
            "metadata": _json_safe(metadata),
            "status": "running",
            "started_at": self.started_at,
            "elapsed_seconds": 0.0,
            "batches": [],
            "evaluations": [],
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.flush()

    def log_batch(self, batch: int, **metrics: Any) -> None:
        record = {"batch": batch, **metrics}
        self.data["batches"].append(_json_safe(record))
        self.data["elapsed_seconds"] = time.time() - self.started_at
        if self.path is not None and batch % self.write_every == 0:
            self.flush()

    def log_evaluation(self, batch: int, **metrics: Any) -> None:
        record = {"batch": batch, **metrics}
        self.data["evaluations"].append(_json_safe(record))
        self.data["elapsed_seconds"] = time.time() - self.started_at
        self.flush()

    def finish(self, *, status: str = "completed", **metadata_updates: Any) -> None:
        self.data["status"] = status
        self.data["elapsed_seconds"] = time.time() - self.started_at
        if metadata_updates:
            self.data["metadata"].update(_json_safe(metadata_updates))
        self.flush()

    def flush(self) -> None:
        if self.path is None:
            return
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2) + "\n")
        tmp_path.replace(self.path)
