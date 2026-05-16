from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ActionSelection:
    move_ids: torch.Tensor
    slots: torch.Tensor
    logprobs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class RustCandidateActorCritic(nn.Module):
    """Candidate-scoring actor-critic for Rust observations and global move IDs."""

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
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

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
        self.action_projection = nn.Sequential(
            nn.Linear(action_emb_dim + action_feature_hidden, action_hidden),
            nn.ReLU(),
            nn.LayerNorm(action_hidden),
        )
        self.state_to_action = nn.Linear(obs_hidden, action_hidden)
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
        id_emb = self.move_id_embedding(safe_ids)
        feature_emb = self.move_feature_encoder(self.move_features[safe_ids])
        action_h = self.action_projection(torch.cat([id_emb, feature_emb], dim=-1))

        state_action_h = self.state_to_action(state_h).unsqueeze(1)
        logits = (state_action_h * action_h).sum(dim=-1) / (action_h.shape[-1] ** 0.5)
        logits = logits.masked_fill(~candidate_mask, -1.0e9)
        return logits, values

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
