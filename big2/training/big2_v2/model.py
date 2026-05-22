from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

OWN_HAND_START = 0
OWN_HAND_END = 52
MOVE_CARD_MASK_START = 0
MOVE_CARD_MASK_END = 52
MOVE_KIND_START = 52
MOVE_KIND_END = 61
MOVE_PRIMARY_RANK_START = 62
MOVE_PRIMARY_RANK_END = 75
MOVE_HIGH_SUIT_START = 88
MOVE_HIGH_SUIT_END = 92
OBS_LAST_MOVE_KIND_START = 104
OBS_LAST_MOVE_KIND_END = 113
OBS_LAST_MOVE_PRIMARY_RANK = 114
OBS_LAST_MOVE_HIGH_SUIT_START = 116
OBS_LAST_MOVE_HIGH_SUIT_END = 120
OBS_FREE_LEAD = 128
DYNAMIC_ACTION_FEATURE_DIM = 110


@dataclass(frozen=True)
class ActionSelection:
    move_ids: torch.Tensor
    slots: torch.Tensor
    logprobs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class Big2V2ActorCritic(nn.Module):
    """Candidate-scoring actor-critic for Big2 v2 observations and global move IDs."""

    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        move_features: torch.Tensor,
        obs_hidden: int = 512,
        action_emb_dim: int = 128,
        action_feature_hidden: int = 128,
        action_hidden: int = 256,
        candidate_set_context: bool = False,
        dynamic_action_features: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.candidate_set_context = candidate_set_context
        self.dynamic_action_features = dynamic_action_features

        self.register_buffer("move_features", move_features.float())
        move_feature_dim = int(move_features.shape[1])

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_hidden),
            nn.ReLU(),
            nn.LayerNorm(obs_hidden),
            nn.Linear(obs_hidden, obs_hidden),
            nn.ReLU(),
            nn.LayerNorm(obs_hidden),
        )
        self.move_id_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.move_feature_encoder = nn.Sequential(
            nn.Linear(move_feature_dim, action_feature_hidden),
            nn.ReLU(),
            nn.LayerNorm(action_feature_hidden),
        )
        action_input_dim = action_emb_dim + action_feature_hidden
        if dynamic_action_features:
            self.candidate_outcome_encoder = nn.Sequential(
                nn.Linear(DYNAMIC_ACTION_FEATURE_DIM, action_feature_hidden),
                nn.ReLU(),
                nn.LayerNorm(action_feature_hidden),
            )
            action_input_dim += action_feature_hidden
            rank_projection = torch.zeros(52, 13)
            suit_projection = torch.zeros(52, 4)
            for card in range(52):
                rank_projection[card, card // 4] = 1.0
                suit_projection[card, card % 4] = 1.0
            self.register_buffer("card_rank_projection", rank_projection)
            self.register_buffer("card_suit_projection", suit_projection)
            self.register_buffer("rank_values", torch.arange(13, dtype=torch.float32))
            self.register_buffer("kind_values", torch.arange(9, dtype=torch.float32))
            self.register_buffer("suit_values", torch.arange(4, dtype=torch.float32))
        self.action_projection = nn.Sequential(
            nn.Linear(action_input_dim, action_hidden),
            nn.ReLU(),
            nn.LayerNorm(action_hidden),
        )
        self.state_to_action = nn.Linear(obs_hidden, action_hidden)
        if candidate_set_context:
            self.candidate_context_projection = nn.Sequential(
                nn.Linear(action_hidden * 2, action_hidden),
                nn.ReLU(),
                nn.LayerNorm(action_hidden),
                nn.Linear(action_hidden, action_hidden),
            )
            self.candidate_value_projection = nn.Sequential(
                nn.Linear(action_hidden * 2, obs_hidden),
                nn.ReLU(),
                nn.LayerNorm(obs_hidden),
                nn.Linear(obs_hidden, obs_hidden),
            )
        self.value_head = nn.Sequential(
            nn.Linear(obs_hidden, obs_hidden),
            nn.ReLU(),
            nn.Linear(obs_hidden, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_h = self.state_encoder(obs.float())
        values = self.value_head(state_h).squeeze(-1)

        safe_ids = candidate_ids.clamp_min(0)
        selected_move_features = self.move_features[safe_ids]
        id_emb = self.move_id_embedding(safe_ids)
        feature_emb = self.move_feature_encoder(selected_move_features)
        action_inputs = [id_emb, feature_emb]
        if self.dynamic_action_features:
            candidate_outcome_features = self._candidate_outcome_features(
                obs.float(),
                selected_move_features,
                candidate_mask,
            )
            action_inputs.append(self.candidate_outcome_encoder(candidate_outcome_features.float()))
        action_h = self.action_projection(torch.cat(action_inputs, dim=-1))

        if self.candidate_set_context:
            candidate_summary = self._candidate_set_summary(action_h, candidate_mask)
            state_action_base = self.state_to_action(state_h)
            state_action_h = state_action_base + self.candidate_context_projection(candidate_summary)
            values = self.value_head(state_h + self.candidate_value_projection(candidate_summary)).squeeze(-1)
        else:
            state_action_h = self.state_to_action(state_h)
        logits = (state_action_h.unsqueeze(1) * action_h).sum(dim=-1) / (action_h.shape[-1] ** 0.5)
        logits = logits.masked_fill(~candidate_mask, -1.0e9)
        return logits, values

    def _candidate_outcome_features(
        self,
        obs: torch.Tensor,
        selected_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        own_hand = obs[:, OWN_HAND_START:OWN_HAND_END].unsqueeze(1)
        move_mask = selected_move_features[..., MOVE_CARD_MASK_START:MOVE_CARD_MASK_END].clamp(0.0, 1.0)
        remaining = (own_hand * (1.0 - move_mask)).clamp(0.0, 1.0)

        rank_counts = remaining @ self.card_rank_projection
        suit_counts = remaining @ self.card_suit_projection
        remaining_count = remaining.sum(dim=-1, keepdim=True)
        played_count = move_mask.sum(dim=-1, keepdim=True)
        current_count = own_hand.sum(dim=-1, keepdim=True).clamp_min(1.0)

        move_kind = selected_move_features[..., MOVE_KIND_START:MOVE_KIND_END]
        is_pass = move_kind[..., 0:1]
        is_non_pass = 1.0 - is_pass
        rank_present = rank_counts > 0.0
        rank_values = self.rank_values.view(1, 1, 13)
        high_rank = torch.where(rank_present, rank_values, torch.full_like(rank_counts, -1.0)).max(dim=-1).values
        low_rank = torch.where(rank_present, rank_values, torch.full_like(rank_counts, 13.0)).min(dim=-1).values

        free_lead = obs[:, OBS_FREE_LEAD : OBS_FREE_LEAD + 1].unsqueeze(1)
        free_lead_feature = free_lead.expand(-1, selected_move_features.shape[1], -1)
        response_turn = 1.0 - free_lead
        is_response = response_turn * is_non_pass
        last_kind = obs[:, OBS_LAST_MOVE_KIND_START:OBS_LAST_MOVE_KIND_END].unsqueeze(1)
        same_kind = (move_kind * last_kind).sum(dim=-1, keepdim=True) * is_response
        move_kind_idx = (move_kind * self.kind_values.view(1, 1, 9)).sum(dim=-1, keepdim=True)
        last_kind_idx = (last_kind * self.kind_values.view(1, 1, 9)).sum(dim=-1, keepdim=True)
        kind_delta = ((move_kind_idx - last_kind_idx) / 8.0) * is_response

        move_primary_rank = (
            selected_move_features[..., MOVE_PRIMARY_RANK_START:MOVE_PRIMARY_RANK_END]
            * self.rank_values.view(1, 1, 13)
        ).sum(dim=-1, keepdim=True) / 12.0
        last_primary_rank = obs[:, OBS_LAST_MOVE_PRIMARY_RANK : OBS_LAST_MOVE_PRIMARY_RANK + 1].unsqueeze(1)
        primary_rank_delta = (move_primary_rank - last_primary_rank) * same_kind
        move_high_suit = (
            selected_move_features[..., MOVE_HIGH_SUIT_START:MOVE_HIGH_SUIT_END]
            * self.suit_values.view(1, 1, 4)
        ).sum(dim=-1, keepdim=True) / 3.0
        last_high_suit = (
            obs[:, OBS_LAST_MOVE_HIGH_SUIT_START:OBS_LAST_MOVE_HIGH_SUIT_END].unsqueeze(1)
            * self.suit_values.view(1, 1, 4)
        ).sum(dim=-1, keepdim=True) / 3.0
        high_suit_delta = (move_high_suit - last_high_suit) * same_kind
        candidate_relative_features = self._candidate_relative_features(
            candidate_mask=candidate_mask,
            is_pass=is_pass,
            is_non_pass=is_non_pass,
            response_turn=response_turn,
            move_kind=move_kind,
            last_kind=last_kind,
            move_kind_idx=move_kind_idx,
            move_primary_rank=move_primary_rank,
            move_high_suit=move_high_suit,
        )

        scalar_features = torch.cat(
            [
                remaining_count / 13.0,
                played_count / 5.0,
                played_count / current_count,
                is_pass,
                (remaining_count <= 0.0).float(),
                (remaining_count == 1.0).float(),
                (remaining_count == 2.0).float(),
                (remaining_count <= 3.0).float(),
                (rank_counts == 1.0).float().sum(dim=-1, keepdim=True) / 13.0,
                (rank_counts == 2.0).float().sum(dim=-1, keepdim=True) / 13.0,
                (rank_counts == 3.0).float().sum(dim=-1, keepdim=True) / 13.0,
                (rank_counts == 4.0).float().sum(dim=-1, keepdim=True) / 13.0,
                torch.clamp(high_rank + 1.0, min=0.0).unsqueeze(-1) / 13.0,
                torch.where(low_rank < 13.0, (low_rank + 1.0) / 13.0, torch.zeros_like(low_rank)).unsqueeze(-1),
                rank_counts[..., 12:13] / 4.0,
                rank_counts[..., 9:13].sum(dim=-1, keepdim=True) / 16.0,
                rank_counts[..., 0:4].sum(dim=-1, keepdim=True) / 16.0,
                free_lead_feature,
                is_response,
                same_kind,
                kind_delta,
                primary_rank_delta,
                high_suit_delta,
                candidate_relative_features,
            ],
            dim=-1,
        )
        return torch.cat([remaining, rank_counts / 4.0, suit_counts / 13.0, scalar_features], dim=-1)

    def _candidate_relative_features(
        self,
        *,
        candidate_mask: torch.Tensor,
        is_pass: torch.Tensor,
        is_non_pass: torch.Tensor,
        response_turn: torch.Tensor,
        move_kind: torch.Tensor,
        last_kind: torch.Tensor,
        move_kind_idx: torch.Tensor,
        move_primary_rank: torch.Tensor,
        move_high_suit: torch.Tensor,
    ) -> torch.Tensor:
        valid = candidate_mask.unsqueeze(-1).float()
        valid_non_pass = valid * is_non_pass
        non_pass_count = valid_non_pass.sum(dim=1, keepdim=True).clamp_min(0.0)
        candidate_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        has_non_pass = (non_pass_count > 0.0).float()

        optional_pass = is_pass * response_turn * has_non_pass
        forced_pass = is_pass * (1.0 - has_non_pass)
        sole_non_pass = is_non_pass * (non_pass_count == 1.0).float()

        selected_strength = (move_kind_idx / 8.0) + move_primary_rank + (move_high_suit * 0.125)
        strength_masked_min = selected_strength.masked_fill(valid_non_pass <= 0.0, torch.finfo(selected_strength.dtype).max)
        strength_masked_max = selected_strength.masked_fill(valid_non_pass <= 0.0, torch.finfo(selected_strength.dtype).min)
        min_strength = strength_masked_min.min(dim=1, keepdim=True).values
        max_strength = strength_masked_max.max(dim=1, keepdim=True).values
        strength_sum = (selected_strength * valid_non_pass).sum(dim=1, keepdim=True)
        mean_strength = strength_sum / non_pass_count.clamp_min(1.0)
        min_strength = torch.where(has_non_pass > 0.0, min_strength, torch.zeros_like(min_strength))
        max_strength = torch.where(has_non_pass > 0.0, max_strength, torch.zeros_like(max_strength))
        mean_strength = torch.where(has_non_pass > 0.0, mean_strength, torch.zeros_like(mean_strength))
        is_weakest_non_pass = is_non_pass * (selected_strength <= min_strength + 1.0e-6).float()
        is_strongest_non_pass = is_non_pass * (selected_strength >= max_strength - 1.0e-6).float()

        same_kind_mask = (move_kind * last_kind).sum(dim=-1, keepdim=True) * valid_non_pass * response_turn
        same_kind_count = same_kind_mask.sum(dim=1, keepdim=True)
        has_same_kind = (same_kind_count > 0.0).float()
        same_kind_min = selected_strength.masked_fill(same_kind_mask <= 0.0, torch.finfo(selected_strength.dtype).max)
        min_same_kind_strength = same_kind_min.min(dim=1, keepdim=True).values
        min_same_kind_strength = torch.where(
            has_same_kind > 0.0,
            min_same_kind_strength,
            torch.zeros_like(min_same_kind_strength),
        )
        selected_same_kind_delta = (selected_strength - min_same_kind_strength) * same_kind_mask
        is_weakest_same_kind = same_kind_mask * (selected_strength <= min_same_kind_strength + 1.0e-6).float()

        return torch.cat(
            [
                candidate_count.expand_as(is_pass).clamp_max(32.0) / 32.0,
                non_pass_count.expand_as(is_pass).clamp_max(32.0) / 32.0,
                has_non_pass.expand_as(is_pass),
                optional_pass,
                forced_pass,
                sole_non_pass,
                min_strength.expand_as(is_pass) / 3.0,
                mean_strength.expand_as(is_pass) / 3.0,
                max_strength.expand_as(is_pass) / 3.0,
                selected_strength / 3.0,
                ((selected_strength - min_strength) * is_non_pass) / 3.0,
                ((max_strength - selected_strength) * is_non_pass) / 3.0,
                is_weakest_non_pass,
                is_strongest_non_pass,
                same_kind_count.expand_as(is_pass).clamp_max(16.0) / 16.0,
                has_same_kind.expand_as(is_pass),
                selected_same_kind_delta / 3.0,
                is_weakest_same_kind,
            ],
            dim=-1,
        )

    @staticmethod
    def _candidate_set_summary(action_h: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        mask = candidate_mask.unsqueeze(-1)
        counts = mask.sum(dim=1).clamp_min(1)
        mean = action_h.masked_fill(~mask, 0.0).sum(dim=1) / counts
        masked_for_max = action_h.masked_fill(~mask, torch.finfo(action_h.dtype).min)
        max_values = masked_for_max.max(dim=1).values
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
        return torch.cat([mean, max_values], dim=-1)

    def distribution(
        self,
        obs: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
        logits, values = self(obs, candidate_ids, candidate_mask)
        if (~candidate_mask).all(dim=1).any():
            raise ValueError("Cannot build policy distribution for a row with no valid candidates")
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()
        return dist, values, entropy

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        sample: bool = True,
    ) -> ActionSelection:
        dist, values, entropy = self.distribution(obs, candidate_ids, candidate_mask)
        slots = dist.sample() if sample else dist.probs.argmax(dim=1)
        logprobs = dist.log_prob(slots)
        move_ids = candidate_ids.gather(1, slots.unsqueeze(1)).squeeze(1)
        return ActionSelection(move_ids=move_ids, slots=slots, logprobs=logprobs, entropy=entropy, values=values)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values, entropy = self.distribution(obs, candidate_ids, candidate_mask)
        return dist.log_prob(slots), values, entropy


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Utility for tests and diagnostics."""

    return F.log_softmax(logits.masked_fill(~mask, -1.0e9), dim=-1)
