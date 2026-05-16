from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from big2.training.rust_ppo.config import RustPPOConfig
from big2.training.rust_ppo.model import RustCandidateActorCritic


def checkpoint_path(checkpoint_dir: str | Path, batch: int) -> Path:
    return Path(checkpoint_dir) / f"batch_{batch:06d}.pt"


def save_checkpoint(
    *,
    checkpoint_dir: str | Path,
    batch: int,
    policy: RustCandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    config: RustPPOConfig,
    metrics: dict[str, Any] | None = None,
) -> Path:
    path = checkpoint_path(checkpoint_dir, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batch": batch,
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics or {},
        },
        path,
    )
    return path


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    paths = sorted(Path(checkpoint_dir).glob("batch_*.pt"))
    return paths[-1] if paths else None


def load_checkpoint(
    *,
    path: str | Path,
    policy: RustCandidateActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    policy.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
