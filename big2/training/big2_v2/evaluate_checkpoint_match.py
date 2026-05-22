from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from big2.training.big2_v2.diagnose_policy import load_policy_from_checkpoint
from big2.training.big2_v2.env_adapter import RustVecEnvAdapter


@torch.no_grad()
def evaluate_checkpoint_match(
    *,
    target_checkpoint: Path,
    opponent_checkpoint: Path,
    games: int,
    num_envs: int,
    seed: int,
    max_candidates: int,
    target_seat: int,
    device: str,
    sample_target: bool = False,
    sample_opponent: bool = False,
) -> dict[str, Any]:
    target, _target_config = load_policy_from_checkpoint(
        checkpoint=target_checkpoint,
        device=device,
        max_candidates=max_candidates,
        num_envs=num_envs,
        seed=seed,
    )
    opponent, _opponent_config = load_policy_from_checkpoint(
        checkpoint=opponent_checkpoint,
        device=device,
        max_candidates=max_candidates,
        num_envs=num_envs,
        seed=seed + 97,
    )
    target.eval()
    opponent.eval()
    torch.manual_seed(seed)

    env = RustVecEnvAdapter(num_envs=num_envs, seed=seed, max_candidates=max_candidates, device=device)
    batch = env.reset()
    completed = 0
    wins = 0
    reward_total = 0.0
    rewards: list[float] = []
    episode_step_counts = [0 for _ in range(num_envs)]
    episode_lengths: list[int] = []
    truncated_candidate_lists = batch.truncated_candidate_lists

    while completed < games:
        action_ids = torch.zeros(batch.num_envs, dtype=torch.long, device=batch.obs.device)
        target_indices: list[int] = []
        opponent_indices: list[int] = []
        valid_rows = batch.candidate_mask.any(dim=1)

        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                action_ids[env_idx] = 0
                continue
            if int(batch.current_player[env_idx].item()) == target_seat:
                target_indices.append(env_idx)
            else:
                opponent_indices.append(env_idx)

        if target_indices:
            idx = torch.tensor(target_indices, dtype=torch.long, device=batch.obs.device)
            selection = target.act(
                batch.obs[idx],
                batch.candidate_ids[idx],
                batch.candidate_mask[idx],
                sample=sample_target,
            )
            action_ids[idx] = selection.move_ids

        if opponent_indices:
            idx = torch.tensor(opponent_indices, dtype=torch.long, device=batch.obs.device)
            selection = opponent.act(
                batch.obs[idx],
                batch.candidate_ids[idx],
                batch.candidate_mask[idx],
                sample=sample_opponent,
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
            seat_reward = float(next_batch.final_rewards[env_idx, target_seat].item())
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
    return {
        "target_checkpoint": str(target_checkpoint),
        "opponent_checkpoint": str(opponent_checkpoint),
        "target_seat": target_seat,
        "games": games,
        "wins": wins,
        "win_rate": win_rate,
        "win_rate_ci95_low": _wilson_interval(wins, games)[0],
        "win_rate_ci95_high": _wilson_interval(wins, games)[1],
        "average_reward": reward_total / games if games else 0.0,
        "reward_std": _std(rewards),
        "episode_length_mean": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
        "truncated_candidate_lists": truncated_candidate_lists,
        "sample_target": sample_target,
        "sample_opponent": sample_opponent,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one PPO checkpoint seat against another PPO checkpoint.")
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-seat", type=int, default=0)
    parser.add_argument("--all-seats", action="store_true")
    parser.add_argument("--sample-target", action="store_true")
    parser.add_argument("--sample-opponent", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    seats = range(4) if args.all_seats else (args.target_seat,)
    results = [
        evaluate_checkpoint_match(
            target_checkpoint=Path(args.target_checkpoint),
            opponent_checkpoint=Path(args.opponent_checkpoint),
            games=args.games,
            num_envs=args.num_envs,
            seed=args.seed + seat * 1000,
            max_candidates=args.max_candidates,
            target_seat=seat,
            device=args.device,
            sample_target=args.sample_target,
            sample_opponent=args.sample_opponent,
        )
        for seat in seats
    ]
    payload = {"results": results}
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    for result in results:
        print(
            f"seat={result['target_seat']} win_rate={result['win_rate']:.3f} "
            f"reward={result['average_reward']:.3f} games={result['games']} "
            f"ci=[{result['win_rate_ci95_low']:.3f}, {result['win_rate_ci95_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
