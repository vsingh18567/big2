from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from big2.training.big2_v2.checkpoints import (
    find_latest_checkpoint,
    load_checkpoint,
    load_checkpoint_partial,
    save_checkpoint,
)
from big2.training.big2_v2.config import LoggingMode, OpponentMixConfig, Big2V2Config
from big2.training.big2_v2.curriculum import (
    TrainingControls,
    smart_greedy_score_from_evals,
    training_controls_for_batch,
)
from big2.training.big2_v2.env_adapter import RustVecEnvAdapter
from big2.training.big2_v2.evaluate import evaluate_policy
from big2.training.big2_v2.model import Big2V2ActorCritic
from big2.training.big2_v2.rollout import Big2V2RolloutState, collect_rollout
from big2.training.big2_v2.update import ppo_update

SMART_GREEDY_DEFAULT_MIX = OpponentMixConfig(
    learner_weight=0.75,
    random_weight=0.10,
    greedy_weight=0.10,
    smart_weight=0.05,
)
OPPONENT_WEIGHT_ARGS = {
    "--learner-weight",
    "--random-weight",
    "--greedy-weight",
    "--smart-weight",
    "--checkpoint-opponent-weight",
}


@dataclass(frozen=True)
class CheckpointOpponentPool:
    policies: list[Big2V2ActorCritic]
    paths: list[Path]


def build_policy(
    env: RustVecEnvAdapter,
    config: Big2V2Config,
    *,
    obs_dim: int | None = None,
    candidate_set_context: bool | None = None,
    dynamic_action_features: bool | None = None,
) -> Big2V2ActorCritic:
    if obs_dim is None:
        obs_dim = env.reset().obs_dim
    if candidate_set_context is None:
        candidate_set_context = config.candidate_set_context
    if dynamic_action_features is None:
        dynamic_action_features = config.dynamic_action_features
    return Big2V2ActorCritic(
        obs_dim=obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor(config.device),
        obs_hidden=config.obs_hidden,
        action_emb_dim=config.action_emb_dim,
        action_feature_hidden=config.action_feature_hidden,
        action_hidden=config.action_hidden,
        candidate_set_context=candidate_set_context,
        dynamic_action_features=dynamic_action_features,
    ).to(config.device)


def run_smoke(config: Big2V2Config) -> None:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    env = RustVecEnvAdapter(
        num_envs=config.num_envs,
        seed=config.seed,
        max_candidates=config.max_candidates,
        device=config.device,
    )
    batch = env.reset()
    policy = build_policy(env, config, obs_dim=batch.obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    controls = training_controls_for_batch(config, batch_idx=1, latest_smart_greedy_score=None)
    buffer, _batch = collect_rollout(
        env=env,
        policy=policy,
        steps=config.rollout_steps,
        opponent_mix=controls.opponent_mix,
        rng=rng,
        initial_batch=batch,
        terminal_reward_mode=config.terminal_reward_mode,
        controller_assignment=config.controller_assignment,
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
        entropy_coef=controls.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
    )
    print(f"big2_v2 smoke: samples={stats.samples} loss={stats.total_loss:.4f} entropy={stats.entropy:.4f}")


def run_training(config: Big2V2Config, *, resume: bool = False) -> None:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    env = RustVecEnvAdapter(
        num_envs=config.num_envs,
        seed=config.seed,
        max_candidates=config.max_candidates,
        device=config.device,
    )
    batch = env.reset()
    policy = build_policy(env, config, obs_dim=batch.obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    metrics_path = Path(config.metrics_path)
    latest_smart_greedy_score = _latest_smart_greedy_score_from_metrics(metrics_path) if resume else None
    start_batch = 1

    if resume:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            payload = load_checkpoint(path=latest, policy=policy, optimizer=optimizer, map_location=config.device)
            start_batch = int(payload["batch"]) + 1
            if latest_smart_greedy_score is None:
                latest_smart_greedy_score = smart_greedy_score_from_evals(payload.get("metrics", {}).get("eval"))
    elif config.init_checkpoint is not None:
        payload = load_checkpoint_partial(path=config.init_checkpoint, policy=policy, map_location=config.device)
        latest_smart_greedy_score = smart_greedy_score_from_evals(payload.get("metrics", {}).get("eval"))

    checkpoint_opponent_pool = _load_checkpoint_opponent_pool(env, config, obs_dim=batch.obs_dim)
    metrics_path.parent.mkdir(parents=True, exist_ok=True) if metrics_path.parent != Path("") else None
    rollout_state = Big2V2RolloutState.create(config.num_envs)
    config_row = {
        "event": "config",
        "config": asdict(config),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
    }
    _write_metrics_row(metrics_path, config_row)
    _print_row(config_row, config.logging_mode)

    for batch_idx in range(start_batch, config.batches + 1):
        controls = training_controls_for_batch(
            config,
            batch_idx=batch_idx,
            latest_smart_greedy_score=latest_smart_greedy_score,
        )
        batch_started_at = time.perf_counter()
        rollout_started_at = time.perf_counter()
        buffer, batch = collect_rollout(
            env=env,
            policy=policy,
            steps=config.rollout_steps,
            opponent_mix=controls.opponent_mix,
            rng=rng,
            initial_batch=batch,
            terminal_reward_mode=config.terminal_reward_mode,
            controller_assignment=config.controller_assignment,
            rollout_state=rollout_state,
            checkpoint_policies=checkpoint_opponent_pool.policies,
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
            entropy_coef=controls.entropy_coef,
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
            controls=controls,
            checkpoint_opponent_pool=checkpoint_opponent_pool,
        )

        eval_seconds = 0.0
        if config.eval_interval > 0 and batch_idx % config.eval_interval == 0:
            eval_started_at = time.perf_counter()
            evals = _run_evals(policy=policy, config=config, batch_idx=batch_idx)
            eval_seconds = time.perf_counter() - eval_started_at
            row["eval"] = evals
            score = smart_greedy_score_from_evals(evals)
            if score is not None:
                latest_smart_greedy_score = score
                row["training_controls"]["eval_smart_greedy_score"] = score
                next_controls = training_controls_for_batch(
                    config,
                    batch_idx=batch_idx + 1,
                    latest_smart_greedy_score=latest_smart_greedy_score,
                )
                row["training_controls"]["next_curriculum_phase"] = next_controls.curriculum_phase
                row["training_controls"]["next_entropy_coef"] = next_controls.entropy_coef
                row["training_controls"]["next_opponent_mix"] = asdict(next_controls.opponent_mix)

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
            if config.checkpoint_opponent_refresh:
                checkpoint_opponent_pool = _load_checkpoint_opponent_pool(env, config, obs_dim=batch.obs_dim)

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
    config: Big2V2Config,
    rollout_seconds: float,
    update_seconds: float,
    controls: TrainingControls | None = None,
    checkpoint_opponent_pool: CheckpointOpponentPool | None = None,
) -> dict[str, Any]:
    if controls is None:
        controls = training_controls_for_batch(config, batch_idx=batch_idx, latest_smart_greedy_score=None)
    checkpoint_opponent_paths = checkpoint_opponent_pool.paths if checkpoint_opponent_pool is not None else []
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
        "training_controls": controls.as_metrics(),
        "controller_assignment": config.controller_assignment,
    }
    row["training_controls"]["checkpoint_opponent_pool_size"] = len(checkpoint_opponent_paths)
    row["training_controls"]["checkpoint_opponent_paths"] = [str(path) for path in checkpoint_opponent_paths]
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


def _run_evals(*, policy: Big2V2ActorCritic, config: Big2V2Config, batch_idx: int) -> dict[str, Any]:
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


def _load_checkpoint_opponent_pool(
    env: RustVecEnvAdapter,
    config: Big2V2Config,
    *,
    obs_dim: int,
) -> CheckpointOpponentPool:
    paths = _checkpoint_opponent_paths(config)
    policies: list[Big2V2ActorCritic] = []
    for path in paths:
        opponent = build_policy(
            env,
            config,
            obs_dim=obs_dim,
            candidate_set_context=_checkpoint_uses_candidate_set_context(path, device=config.device),
            dynamic_action_features=_checkpoint_uses_dynamic_action_features(path, device=config.device),
        )
        load_checkpoint_partial(path=path, policy=opponent, map_location=config.device)
        opponent.eval()
        for param in opponent.parameters():
            param.requires_grad_(False)
        policies.append(opponent)
    return CheckpointOpponentPool(policies=policies, paths=paths)


def _checkpoint_uses_candidate_set_context(path: Path, *, device: str) -> bool:
    payload = torch.load(path, map_location=device)
    return any(key.startswith("candidate_context_projection.") for key in payload.get("model_state", {}))


def _checkpoint_uses_dynamic_action_features(path: Path, *, device: str) -> bool:
    payload = torch.load(path, map_location=device)
    return any(key.startswith("candidate_outcome_encoder.") for key in payload.get("model_state", {}))


def _checkpoint_opponent_paths(config: Big2V2Config) -> list[Path]:
    source_dir = Path(config.checkpoint_opponent_dir or config.checkpoint_dir)
    if not source_dir.exists():
        return []
    paths = []
    for path in source_dir.glob("batch_*.pt"):
        batch = _checkpoint_batch(path)
        if batch is None:
            continue
        if config.checkpoint_opponent_stride > 1 and batch % config.checkpoint_opponent_stride != 0:
            continue
        paths.append((batch, path))
    paths.sort(key=lambda item: item[0])
    selected = [path for _batch, path in paths]
    if config.checkpoint_opponent_limit > 0:
        selected = selected[-config.checkpoint_opponent_limit :]
    return selected


def _checkpoint_batch(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("batch_"):
        return None
    try:
        return int(stem.removeprefix("batch_"))
    except ValueError:
        return None


def _write_metrics_row(metrics_path: Path, row: dict[str, Any]) -> None:
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _latest_smart_greedy_score_from_metrics(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    latest_score: float | None = None
    with metrics_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = smart_greedy_score_from_evals(row.get("eval"))
            if score is not None:
                latest_score = score
    return latest_score


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
    parser = argparse.ArgumentParser(description="Smoke-run Big2 v2.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--candidate-set-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-action-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--entropy-schedule", choices=("constant", "linear"), default="constant")
    parser.add_argument("--entropy-start-coef", type=float, default=None)
    parser.add_argument("--entropy-end-coef", type=float, default=None)
    parser.add_argument("--entropy-schedule-batches", type=int, default=0)
    parser.add_argument("--terminal-reward-mode", choices=("card-fraction", "win-loss"), default="win-loss")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy-weight", type=float, default=0.20)
    parser.add_argument("--smart-weight", type=float, default=0.20)
    parser.add_argument("--random-weight", type=float, default=0.0)
    parser.add_argument("--learner-weight", type=float, default=0.55)
    parser.add_argument("--checkpoint-opponent-weight", type=float, default=0.05)
    parser.add_argument(
        "--controller-assignment",
        choices=("turn", "episode-seat", "single-learner", "single-learner-uniform", "table-profile"),
        default="table-profile",
    )
    parser.add_argument("--curriculum", choices=("off", "smart-greedy"), default="off")
    parser.add_argument("--checkpoint-dir", default="big2_v2_checkpoints")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--checkpoint-opponent-dir", default=None)
    parser.add_argument("--checkpoint-opponent-limit", type=int, default=4)
    parser.add_argument("--checkpoint-opponent-stride", type=int, default=25)
    parser.add_argument("--checkpoint-opponent-refresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-path", default="big2_v2_metrics.jsonl")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--eval-num-envs", type=int, default=16)
    parser.add_argument("--eval-policy-seat", type=int, default=0)
    parser.add_argument("--eval-all-seats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-mode", choices=("minimal", "medium", "max"), default="medium")
    args = parser.parse_args()
    args.opponent_weights_explicit = _opponent_weights_explicit()
    return args


def _opponent_weights_explicit() -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:] for option in OPPONENT_WEIGHT_ARGS)


def _opponent_mix_from_args(args: argparse.Namespace) -> OpponentMixConfig:
    if args.curriculum == "smart-greedy" and not args.opponent_weights_explicit:
        return SMART_GREEDY_DEFAULT_MIX
    return OpponentMixConfig(
        learner_weight=args.learner_weight,
        random_weight=args.random_weight,
        greedy_weight=args.greedy_weight,
        smart_weight=args.smart_weight,
        checkpoint_weight=args.checkpoint_opponent_weight,
    )


def main() -> None:
    args = parse_args()
    opponent_mix = _opponent_mix_from_args(args)
    config = Big2V2Config(
        num_envs=args.num_envs,
        batches=args.batches,
        rollout_steps=args.rollout_steps,
        max_candidates=args.max_candidates,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        candidate_set_context=args.candidate_set_context,
        dynamic_action_features=args.dynamic_action_features,
        entropy_coef=args.entropy_coef,
        entropy_schedule=args.entropy_schedule,
        entropy_start_coef=args.entropy_start_coef,
        entropy_end_coef=args.entropy_end_coef,
        entropy_schedule_batches=args.entropy_schedule_batches,
        terminal_reward_mode=args.terminal_reward_mode.replace("-", "_"),
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        init_checkpoint=args.init_checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_opponent_dir=args.checkpoint_opponent_dir,
        checkpoint_opponent_limit=args.checkpoint_opponent_limit,
        checkpoint_opponent_stride=args.checkpoint_opponent_stride,
        checkpoint_opponent_refresh=args.checkpoint_opponent_refresh,
        metrics_path=args.metrics_path,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_num_envs=args.eval_num_envs,
        eval_policy_seat=args.eval_policy_seat,
        eval_all_seats=args.eval_all_seats,
        logging_mode=args.logging_mode,
        curriculum=args.curriculum.replace("-", "_"),
        controller_assignment=args.controller_assignment.replace("-", "_"),
        opponent_mix=opponent_mix,
    )
    if args.train:
        run_training(config, resume=args.resume)
    else:
        run_smoke(config)


if __name__ == "__main__":
    main()
