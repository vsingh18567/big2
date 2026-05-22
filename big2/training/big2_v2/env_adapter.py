from __future__ import annotations

from dataclasses import dataclass

import torch

from big2.training.big2_v2.metadata import MoveMetadataTable


@dataclass(frozen=True)
class RustBatch:
    """Torch tensor view of one `big2_rust.Big2VecEnv` batch."""

    num_envs: int
    obs_dim: int
    max_candidates: int
    obs: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    current_player: torch.Tensor
    done: torch.Tensor
    final_rewards: torch.Tensor
    truncated_candidate_lists: int


class RustVecEnvAdapter:
    """Thin tensor adapter around the PyO3 Rust vectorized environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        max_candidates: int,
        device: str | torch.device = "cpu",
        require_three_diamond_open: bool = True,
        allow_wheel_straight: bool = False,
        allow_two_in_straight: bool = False,
        game_ends_on_first_out: bool = True,
        passed_player_may_reenter: bool = False,
    ):
        import big2_rust

        self.device = torch.device(device)
        self.raw = big2_rust.Big2VecEnv(
            num_envs,
            seed,
            max_candidates,
            require_three_diamond_open,
            allow_wheel_straight,
            allow_two_in_straight,
            game_ends_on_first_out,
            passed_player_may_reenter,
        )
        feature_dim, features_flat = self.raw.move_features()
        self.metadata = MoveMetadataTable(
            self.raw.move_metadata(),
            feature_dim=feature_dim,
            features_flat=features_flat,
        )

    @property
    def num_actions(self) -> int:
        return self.metadata.num_actions

    def reset(self) -> RustBatch:
        return self._convert_batch(self.raw.reset())

    def reset_done(self, env_indices: list[int]) -> RustBatch:
        return self._convert_batch(self.raw.reset_done(env_indices))

    def step(self, action_ids: torch.Tensor | list[int]) -> RustBatch:
        if isinstance(action_ids, torch.Tensor):
            action_list = [int(x) for x in action_ids.detach().cpu().tolist()]
        else:
            action_list = [int(x) for x in action_ids]
        return self._convert_batch(self.raw.step(action_list))

    def _convert_batch(self, batch: tuple) -> RustBatch:
        if len(batch) != 10:
            raise ValueError(f"Expected 10-field Rust batch tuple, got {len(batch)} fields")

        (
            num_envs,
            obs_dim,
            max_candidates,
            observations,
            candidate_ids,
            candidate_mask,
            current_player,
            done,
            final_rewards,
            truncated_candidate_lists,
        ) = batch

        obs = torch.tensor(observations, dtype=torch.float32, device=self.device).view(num_envs, obs_dim)
        candidates = torch.tensor(candidate_ids, dtype=torch.long, device=self.device).view(num_envs, max_candidates)
        mask = torch.tensor(candidate_mask, dtype=torch.bool, device=self.device).view(num_envs, max_candidates)
        players = torch.tensor(current_player, dtype=torch.long, device=self.device)
        done_t = torch.tensor(done, dtype=torch.bool, device=self.device)
        rewards = torch.tensor(final_rewards, dtype=torch.float32, device=self.device).view(num_envs, 4)

        return RustBatch(
            num_envs=num_envs,
            obs_dim=obs_dim,
            max_candidates=max_candidates,
            obs=obs,
            candidate_ids=candidates,
            candidate_mask=mask,
            current_player=players,
            done=done_t,
            final_rewards=rewards,
            truncated_candidate_lists=int(truncated_candidate_lists),
        )
