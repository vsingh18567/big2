from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MOVE_KIND_COUNT = 9
MOVE_FEATURE_DIM = 52 + MOVE_KIND_COUNT + 1 + 13 + 13 + 4 + 5


@dataclass(frozen=True)
class MoveMetadata:
    """Python view of one Rust `MoveMeta` row."""

    move_id: int
    mask: int
    kind: int
    num_cards: int
    primary_rank: int
    secondary_rank: int
    high_suit: int
    ranks_desc: tuple[int, ...]

    @property
    def is_pass(self) -> bool:
        return self.kind == 0


class MoveMetadataTable:
    """Feature table and structured metadata for all Rust global moves."""

    def __init__(
        self,
        rows: list[tuple[int, int, int, int, int, int, int, list[int]]],
        *,
        feature_dim: int,
        features_flat: list[float],
    ):
        self.rows = [MoveMetadata(*row[:7], tuple(row[7])) for row in rows]
        if not self.rows:
            raise ValueError("move metadata cannot be empty")
        for expected_id, row in enumerate(self.rows):
            if row.move_id != expected_id:
                raise ValueError(f"move metadata must be sorted by id; expected {expected_id}, got {row.move_id}")
        if feature_dim != MOVE_FEATURE_DIM:
            raise ValueError(f"Expected Rust move feature dim {MOVE_FEATURE_DIM}, got {feature_dim}")
        expected_values = len(self.rows) * feature_dim
        if len(features_flat) != expected_values:
            raise ValueError(f"Expected {expected_values} flattened move feature values, got {len(features_flat)}")
        self.features_np = np.asarray(features_flat, dtype=np.float32).reshape(len(self.rows), feature_dim)

    @property
    def num_actions(self) -> int:
        return len(self.rows)

    @property
    def feature_dim(self) -> int:
        return self.features_np.shape[1]

    def as_tensor(self, device: str | torch.device) -> torch.Tensor:
        return torch.as_tensor(self.features_np, dtype=torch.float32, device=device)

    def get(self, move_id: int) -> MoveMetadata:
        return self.rows[move_id]
