from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from big2.training.rust_ppo.checkpoints import find_latest_checkpoint, load_checkpoint, save_checkpoint
from big2.training.rust_ppo.config import LoggingMode, OpponentMixConfig, RustPPOConfig
from big2.training.rust_ppo.env_adapter import RustVecEnvAdapter
from big2.training.rust_ppo.evaluate import evaluate_policy
from big2.training.rust_ppo.model import RustCandidateActorCritic
from big2.training.rust_ppo.rollout import collect_rollout
from big2.training.rust_ppo.update import ppo_update


def build_policy(env: RustVecEnvAdapter, config: RustPPOConfig) -> RustCandidateActorCritic:
    return RustCandidateActorCritic(
        obs_dim=env.reset().obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor(config.device),
        obs_hidden=config.obs_hidden,
        action_emb_dim=config.action_emb_dim,
        action_feature_hidden=config.action_feature_hidden,
        action_hidden=config.action_hidden,
    ).to(config.device)


def run_smoke(config: RustPPOConfig) -> None:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    env = RustVecEnvAdapter(
        num_envs=config.num_envs,
        seed=config.seed,
        max_candidates=config.max_candidates,
        device=config.device,
    )
    policy = build_policy(env, config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    batch = env.reset()
    buffer, _batch = collect_rollout(
        env=env,
        policy=policy,
        steps=config.rollout_steps,
        opponent_mix=config.opponent_mix,
        rng=rng,
        initial_batch=batch,
        step_penalty=config.step_penalty,
    )
    stats = ppo_update(
        policy=policy,
        buffer=buffer,
        optimizer=optimizer,
        ppo_epochs=config.ppo_epochs,
        mini_batch_size=config.mini_batch_size,
        clip_epsilon=config.clip_epsilon,
        gamma=config.gamma,
        lam=config.lam,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
    )
    print(f"rust_ppo smoke: samples={stats.samples} loss={stats.total_loss:.4f} entropy={stats.entropy:.4f}")


def run_training(config: RustPPOConfig, *, resume: bool = False) -> None:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    env = RustVecEnvAdapter(
        num_envs=config.num_envs,
        seed=config.seed,
        max_candidates=config.max_candidates,
        device=config.device,
    )
    policy = build_policy(env, config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    start_batch = 1

    if resume:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            payload = load_checkpoint(path=latest, policy=policy, optimizer=optimizer, map_location=config.device)
            start_batch = int(payload["batch"]) + 1

    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True) if metrics_path.parent != Path("") else None
    batch = env.reset()
    episode_step_counts = [0 for _ in range(config.num_envs)]
    config_row = {
        "event": "config",
        "config": asdict(config),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
    }
    _write_metrics_row(metrics_path, config_row)
    _print_row(config_row, config.logging_mode)

    for batch_idx in range(start_batch, config.batches + 1):
        batch_started_at = time.perf_counter()
        rollout_started_at = time.perf_counter()
        buffer, batch = collect_rollout(
            env=env,
            policy=policy,
            steps=config.rollout_steps,
            opponent_mix=config.opponent_mix,
            rng=rng,
            initial_batch=batch,
            step_penalty=config.step_penalty,
            episode_step_counts=episode_step_counts,
        )
        rollout_seconds = time.perf_counter() - rollout_started_at
        update_started_at = time.perf_counter()
        stats = ppo_update(
            policy=policy,
            buffer=buffer,
            optimizer=optimizer,
            ppo_epochs=config.ppo_epochs,
            mini_batch_size=config.mini_batch_size,
            clip_epsilon=config.clip_epsilon,
            gamma=config.gamma,
            lam=config.lam,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            device=config.device,
        )
        update_seconds = time.perf_counter() - update_started_at

        row = _build_training_row(
            batch_idx=batch_idx,
            stats=stats,
            buffer=buffer,
            config=config,
            rollout_seconds=rollout_seconds,
            update_seconds=update_seconds,
        )

        eval_seconds = 0.0
        if config.eval_interval > 0 and batch_idx % config.eval_interval == 0:
            eval_started_at = time.perf_counter()
            evals = _run_evals(policy=policy, config=config, batch_idx=batch_idx)
            eval_seconds = time.perf_counter() - eval_started_at
            row["eval"] = evals

        checkpoint_seconds = 0.0
        if config.checkpoint_interval > 0 and batch_idx % config.checkpoint_interval == 0:
            checkpoint_started_at = time.perf_counter()
            path = save_checkpoint(
                checkpoint_dir=config.checkpoint_dir,
                batch=batch_idx,
                policy=policy,
                optimizer=optimizer,
                config=config,
                metrics=row,
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_started_at
            row["checkpoint_path"] = str(path)

        row["timing"]["eval_seconds"] = eval_seconds
        row["timing"]["checkpoint_seconds"] = checkpoint_seconds
        row["timing"]["batch_seconds"] = time.perf_counter() - batch_started_at

        _write_metrics_row(metrics_path, row)
        _print_row(row, config.logging_mode)


def _build_training_row(
    *,
    batch_idx: int,
    stats,
    buffer,
    config: RustPPOConfig,
    rollout_seconds: float,
    update_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event": "batch",
        "batch": batch_idx,
        "samples": stats.samples,
        "policy_loss": stats.policy_loss,
        "value_loss": stats.value_loss,
        "entropy": stats.entropy,
        "total_loss": stats.total_loss,
        "candidate_count_mean": buffer.candidate_count_mean,
        "candidate_count_max": buffer.candidate_count_max,
        "candidate_count_rows": buffer.candidate_count_rows,
        "truncated_candidate_lists": buffer.truncated_candidate_lists,
        "max_candidates": config.max_candidates,
    }
    if config.logging_mode in {"medium", "max"}:
        row.update(
            {
                "ppo": {
                    "approx_kl": stats.approx_kl,
                    "clip_fraction": stats.clip_fraction,
                    "ratio_mean": stats.ratio_mean,
                    "ratio_std": stats.ratio_std,
                    "ratio_max": stats.ratio_max,
                    "advantage_mean": stats.advantage_mean,
                    "advantage_std": stats.advantage_std,
                    "return_mean": stats.return_mean,
                    "return_std": stats.return_std,
                    "value_explained_variance": stats.value_explained_variance,
                    "grad_norm": stats.grad_norm,
                    "action_probability_mean": stats.action_probability_mean,
                    "action_probability_p10": stats.action_probability_p10,
                    "action_probability_p90": stats.action_probability_p90,
                },
                "rollout": {
                    "controller_counts": dict(buffer.controller_counts),
                    "learner_turns": len(buffer),
                    "episodes_completed": buffer.episodes_completed,
                    "episode_length_mean": _mean(buffer.episode_lengths),
                    "episode_length_p50": _percentile(buffer.episode_lengths, 0.50),
                    "episode_length_p95": _percentile(buffer.episode_lengths, 0.95),
                    "terminal_reward_mean": (
                        sum(buffer.terminal_reward_by_seat_total) / (4 * buffer.episodes_completed)
                        if buffer.episodes_completed
                        else 0.0
                    ),
                    "terminal_reward_by_seat_mean": [
                        reward / buffer.episodes_completed if buffer.episodes_completed else 0.0
                        for reward in buffer.terminal_reward_by_seat_total
                    ],
                    "wins_by_seat": buffer.wins_by_seat,
                    "pass_rate": (
                        buffer.pass_actions / sum(buffer.controller_counts.values())
                        if sum(buffer.controller_counts.values())
                        else 0.0
                    ),
                },
                "candidates": {
                    "count_p50": _percentile(buffer.candidate_counts, 0.50),
                    "count_p90": _percentile(buffer.candidate_counts, 0.90),
                    "count_p95": _percentile(buffer.candidate_counts, 0.95),
                    "count_p99": _percentile(buffer.candidate_counts, 0.99),
                },
                "timing": {
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    "eval_seconds": 0.0,
                    "checkpoint_seconds": 0.0,
                    "batch_seconds": 0.0,
                    "samples_per_second": stats.samples / update_seconds if update_seconds > 0 else 0.0,
                    "env_steps_per_second": (
                        (config.num_envs * config.rollout_steps) / rollout_seconds if rollout_seconds > 0 else 0.0
                    ),
                },
            }
        )
    else:
        row["timing"] = {"eval_seconds": 0.0, "checkpoint_seconds": 0.0, "batch_seconds": 0.0}

    if config.logging_mode == "max":
        row["candidates"]["count_histogram"] = _histogram(buffer.candidate_counts)
        row["rollout"]["selected_move_kind_counts"] = dict(buffer.selected_move_kind_counts)
        row["rollout"]["learner_entropy_by_candidate_bucket"] = {
            bucket: {
                "count": len(values),
                "mean": _mean(values),
                "p10": _percentile(values, 0.10),
                "p90": _percentile(values, 0.90),
            }
            for bucket, values in sorted(buffer.learner_entropy_by_candidate_bucket.items())
        }
    return row


def _run_evals(*, policy: RustCandidateActorCritic, config: RustPPOConfig, batch_idx: int) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    seats = range(4) if config.eval_all_seats else (config.eval_policy_seat,)
    for opponent in ("random", "greedy", "smart"):
        seat_results = []
        for seat in seats:
            result = evaluate_policy(
                policy=policy,
                opponent=opponent,
                games=config.eval_games,
                num_envs=config.eval_num_envs,
                seed=config.seed + 10_000 + batch_idx + seat * 1_000,
                max_candidates=config.max_candidates,
                policy_seat=seat,
                device=config.device,
            )
            seat_results.append(asdict(result))
        if config.eval_all_seats:
            wins = sum(result["wins"] for result in seat_results)
            games = sum(result["games"] for result in seat_results)
            rewards = [result["average_reward"] for result in seat_results]
            evals[opponent] = {
                "aggregate": {
                    "opponent": opponent,
                    "games": games,
                    "wins": wins,
                    "win_rate": wins / games if games else 0.0,
                    "average_reward": _mean(rewards),
                    "win_rate_ci95_low": _wilson_low(wins, games),
                    "win_rate_ci95_high": _wilson_high(wins, games),
                },
                "seats": {str(result["policy_seat"]): result for result in seat_results},
            }
        else:
            evals[opponent] = seat_results[0]
    return evals


def _write_metrics_row(metrics_path: Path, row: dict[str, Any]) -> None:
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _print_row(row: dict[str, Any], logging_mode: LoggingMode) -> None:
    if logging_mode == "minimal" and "eval" not in row:
        return
    print(json.dumps(row))


def _mean(values: list[float] | list[int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: list[float] | list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _histogram(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        bucket_start = (value // 5) * 5
        label = f"{bucket_start:03d}_{bucket_start + 4:03d}"
        counts[label] = counts.get(label, 0) + 1
    return counts


def _wilson_low(wins: int, games: int) -> float:
    low, _high = _wilson_interval(wins, games)
    return low


def _wilson_high(wins: int, games: int) -> float:
    _low, high = _wilson_interval(wins, games)
    return high


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    margin = z * ((p * (1.0 - p) + z * z / (4.0 * games)) / games) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-run Rust-backed Big 2 PPO.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy-weight", type=float, default=0.0)
    parser.add_argument("--smart-weight", type=float, default=0.0)
    parser.add_argument("--random-weight", type=float, default=0.0)
    parser.add_argument("--learner-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default="rust_ppo_checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--metrics-path", default="rust_ppo_metrics.jsonl")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--eval-num-envs", type=int, default=16)
    parser.add_argument("--eval-policy-seat", type=int, default=0)
    parser.add_argument("--eval-all-seats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-mode", choices=("minimal", "medium", "max"), default="medium")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RustPPOConfig(
        num_envs=args.num_envs,
        batches=args.batches,
        rollout_steps=args.rollout_steps,
        max_candidates=args.max_candidates,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        metrics_path=args.metrics_path,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_num_envs=args.eval_num_envs,
        eval_policy_seat=args.eval_policy_seat,
        eval_all_seats=args.eval_all_seats,
        logging_mode=args.logging_mode,
        opponent_mix=OpponentMixConfig(
            learner_weight=args.learner_weight,
            random_weight=args.random_weight,
            greedy_weight=args.greedy_weight,
            smart_weight=args.smart_weight,
        ),
    )
    if args.train:
        run_training(config, resume=args.resume)
    else:
        run_smoke(config)


if __name__ == "__main__":
    main()
