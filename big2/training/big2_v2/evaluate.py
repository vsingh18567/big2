from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from big2.training.big2_v2.env_adapter import RustVecEnvAdapter
from big2.training.big2_v2.model import Big2V2ActorCritic
from big2.training.big2_v2.opponents import greedy_slot, smart_slot


@dataclass(frozen=True)
class EvalResult:
    opponent: str
    policy_seat: int
    games: int
    wins: int
    win_rate: float
    win_rate_ci95_low: float
    win_rate_ci95_high: float
    average_reward: float
    reward_std: float
    episode_length_mean: float
    truncated_candidate_lists: int


@torch.no_grad()
def evaluate_policy(
    *,
    policy: Big2V2ActorCritic,
    opponent: str,
    games: int,
    num_envs: int,
    seed: int,
    max_candidates: int,
    policy_seat: int,
    device: str | torch.device,
) -> EvalResult:
    """Evaluate one policy-controlled seat against one fixed metadata opponent."""

    if opponent not in {"random", "greedy", "smart"}:
        raise ValueError(f"Unsupported eval opponent: {opponent}")

    env = RustVecEnvAdapter(num_envs=num_envs, seed=seed, max_candidates=max_candidates, device=device)
    batch = env.reset()
    torch.manual_seed(seed)
    policy.eval()

    completed = 0
    wins = 0
    reward_total = 0.0
    rewards: list[float] = []
    episode_step_counts = [0 for _ in range(num_envs)]
    episode_lengths: list[int] = []
    truncated_candidate_lists = batch.truncated_candidate_lists
    while completed < games:
        action_ids = torch.zeros(batch.num_envs, dtype=torch.long, device=batch.obs.device)
        policy_indices: list[int] = []
        valid_rows = batch.candidate_mask.any(dim=1)

        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                action_ids[env_idx] = 0
                continue
            if int(batch.current_player[env_idx].item()) == policy_seat:
                policy_indices.append(env_idx)
                continue
            if opponent == "random":
                valid = torch.nonzero(batch.candidate_mask[env_idx], as_tuple=False).flatten()
                slot = int(valid[torch.randint(len(valid), (1,), device=valid.device)].item())
            elif opponent == "greedy":
                slot = greedy_slot(batch.candidate_ids[env_idx], batch.candidate_mask[env_idx], env.metadata)
            else:
                slot = smart_slot(
                    batch.obs[env_idx],
                    batch.candidate_ids[env_idx],
                    batch.candidate_mask[env_idx],
                    env.metadata,
                )
            action_ids[env_idx] = batch.candidate_ids[env_idx, slot]

        if policy_indices:
            idx = torch.tensor(policy_indices, dtype=torch.long, device=batch.obs.device)
            selection = policy.act(
                batch.obs[idx],
                batch.candidate_ids[idx],
                batch.candidate_mask[idx],
                sample=False,
            )
            action_ids[idx] = selection.move_ids

        next_batch = env.step(action_ids)
        truncated_candidate_lists += next_batch.truncated_candidate_lists
        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                continue
            episode_step_counts[env_idx] += 1
        done_indices: list[int] = []
        for env_idx in range(next_batch.num_envs):
            if not bool(next_batch.done[env_idx].item()):
                continue
            completed += 1
            seat_reward = float(next_batch.final_rewards[env_idx, policy_seat].item())
            reward_total += seat_reward
            rewards.append(seat_reward)
            episode_lengths.append(episode_step_counts[env_idx])
            episode_step_counts[env_idx] = 0
            if seat_reward > 0:
                wins += 1
            done_indices.append(env_idx)
            if completed >= games:
                break

        batch = env.reset_done(done_indices) if done_indices else next_batch

    win_rate = wins / games if games else 0.0
    ci_low, ci_high = _wilson_interval(wins, games)
    reward_mean = reward_total / games if games else 0.0
    reward_std = _std(rewards)
    return EvalResult(
        opponent=opponent,
        policy_seat=policy_seat,
        games=games,
        wins=wins,
        win_rate=win_rate,
        win_rate_ci95_low=ci_low,
        win_rate_ci95_high=ci_high,
        average_reward=reward_mean,
        reward_std=reward_std,
        episode_length_mean=sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
        truncated_candidate_lists=truncated_candidate_lists,
    )


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
