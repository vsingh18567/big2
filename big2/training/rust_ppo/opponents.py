from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from big2.training.rust_ppo.metadata import MoveMetadataTable

OBS_LAST_MOVE_KIND_START = 104
OBS_CARDS_REMAINING_START = 120


def valid_slots(candidate_mask_row: torch.Tensor) -> list[int]:
    return [int(idx) for idx in torch.nonzero(candidate_mask_row, as_tuple=False).flatten().tolist()]


def random_slot(candidate_mask_row: torch.Tensor, rng: random.Random) -> int:
    slots = valid_slots(candidate_mask_row)
    if not slots:
        raise ValueError("Cannot choose from an empty candidate row")
    return rng.choice(slots)


def greedy_slot(candidate_ids_row: torch.Tensor, candidate_mask_row: torch.Tensor, metadata: MoveMetadataTable) -> int:
    slots = valid_slots(candidate_mask_row)
    if not slots:
        raise ValueError("Cannot choose from an empty candidate row")
    non_pass = [slot for slot in slots if not metadata.get(int(candidate_ids_row[slot])).is_pass]
    if not non_pass:
        return slots[0]
    return min(non_pass, key=lambda slot: greedy_key(int(candidate_ids_row[slot]), metadata))


def smart_slot(
    obs_row: torch.Tensor,
    candidate_ids_row: torch.Tensor,
    candidate_mask_row: torch.Tensor,
    metadata: MoveMetadataTable,
) -> int:
    slots = valid_slots(candidate_mask_row)
    if not slots:
        raise ValueError("Cannot choose from an empty candidate row")

    non_pass = [slot for slot in slots if not metadata.get(int(candidate_ids_row[slot])).is_pass]
    if not non_pass:
        return slots[0]

    cards_remaining = int(round(float(obs_row[OBS_CARDS_REMAINING_START].item()) * 13))
    for slot in non_pass:
        move = metadata.get(int(candidate_ids_row[slot]))
        if move.num_cards >= cards_remaining:
            return slot

    active_kind = int(torch.argmax(obs_row[OBS_LAST_MOVE_KIND_START : OBS_LAST_MOVE_KIND_START + 9]).item())
    can_pass = any(metadata.get(int(candidate_ids_row[slot])).is_pass for slot in slots)
    if can_pass and active_kind >= 7:
        pass_slots = [slot for slot in slots if metadata.get(int(candidate_ids_row[slot])).is_pass]
        if pass_slots:
            return pass_slots[0]

    return min(non_pass, key=lambda slot: smart_key(int(candidate_ids_row[slot]), metadata, cards_remaining))


def slots_to_move_ids(candidate_ids: torch.Tensor, slots: Sequence[int]) -> torch.Tensor:
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=candidate_ids.device)
    return candidate_ids.gather(1, slot_tensor.unsqueeze(1)).squeeze(1)


def greedy_key(move_id: int, metadata: MoveMetadataTable) -> tuple[int, int, int, int]:
    move = metadata.get(move_id)
    return (move.kind, move.primary_rank, move.high_suit, move.num_cards)


def smart_key(move_id: int, metadata: MoveMetadataTable, cards_remaining: int) -> tuple[float, int, int]:
    move = metadata.get(move_id)
    phase_penalty = 0.0
    if cards_remaining > 8 and move.primary_rank >= 10:
        phase_penalty += 20.0
    if cards_remaining <= 5:
        phase_penalty -= move.num_cards * 5.0
    else:
        phase_penalty -= move.num_cards * 2.0
    return (phase_penalty + move.primary_rank, move.kind, move.high_suit)
