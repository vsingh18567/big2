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


def load_checkpoint_partial(
    *,
    path: str | Path,
    policy: RustCandidateActorCritic,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    current_state = policy.state_dict()
    compatible_state = {
        key: value
        for key, value in payload["model_state"].items()
        if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
    }
    partial_loaded_keys: list[str] = []
    _copy_action_projection_prefix(
        checkpoint_state=payload["model_state"],
        current_state=current_state,
        compatible_state=compatible_state,
        partial_loaded_keys=partial_loaded_keys,
    )
    missing, unexpected = policy.load_state_dict(compatible_state, strict=False)
    payload["partial_load"] = {
        "loaded_keys": sorted(compatible_state),
        "partial_loaded_keys": sorted(partial_loaded_keys),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "skipped_checkpoint_keys": sorted(set(payload["model_state"]) - set(compatible_state)),
    }
    return payload


def _copy_action_projection_prefix(
    *,
    checkpoint_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    compatible_state: dict[str, torch.Tensor],
    partial_loaded_keys: list[str],
) -> None:
    key = "action_projection.0.weight"
    if key not in checkpoint_state or key not in current_state or key in compatible_state:
        return

    source = checkpoint_state[key]
    target = current_state[key].clone()
    if source.ndim != 2 or target.ndim != 2:
        return
    if source.shape[0] != target.shape[0] or source.shape[1] > target.shape[1]:
        return

    target[:, : source.shape[1]] = source
    target[:, source.shape[1] :] = 0.0
    compatible_state[key] = target
    partial_loaded_keys.append(key)
