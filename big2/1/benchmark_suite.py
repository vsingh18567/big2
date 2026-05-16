from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
import random

from big2.play.config_loader import (
    LLM_AGENT_TYPES,
    AgentConfig,
    AgentType,
    ArenaConfig,
    ArenaEntrantConfig,
    load,
    load_arena_config,
)
from big2.play.agents import LLMPlayAgent
from big2.play.elo_arena import compute_final_score
from big2.play.logging_utils import configure_process_logging, log_timestamp
from big2.play.runner import build_turn
from big2.simulator.cards import PASS
from big2.simulator.env import Big2Env

DEFAULT_CHECKPOINT_DIR = "training_checkpoints"
MATCHUP_THREE_GREEDY = "vs_3_greedy"
MATCHUP_THREE_PPO = "vs_3_latest_ppo"
MATCHUP_ORDER = (
    MATCHUP_THREE_GREEDY,
    MATCHUP_THREE_PPO,
)


@dataclass(frozen=True)
class BenchmarkSeatSpec:
    entrant_id: str
    role: str
    config: AgentConfig


@dataclass(frozen=True)
class BenchmarkTask:
    global_game_index: int
    candidate_id: str
    candidate_agent_type: str
    matchup: str
    matchup_game_index: int
    seat_seed: int
    deal_seed: int
    seats: tuple[BenchmarkSeatSpec, ...]


@dataclass(frozen=True)
class BenchmarkPlayerResult:
    global_game_index: int
    candidate_id: str
    candidate_agent_type: str
    matchup: str
    matchup_game_index: int
    entrant_id: str
    entrant_role: str
    agent_type: str
    checkpoint_path: str
    seat: int
    winner_seat: int
    won: bool
    cards_remaining: int
    score: int
    pass_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_elapsed_ms: int


class WorkerExecutionError(RuntimeError):
    pass


def chunked(values: list[BenchmarkTask], chunk_size: int) -> list[list[BenchmarkTask]]:
    iterator = iter(values)
    chunks: list[list[BenchmarkTask]] = []
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return chunks
        chunks.append(chunk)


def find_latest_checkpoint_path(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> str:
    checkpoint_path = Path(checkpoint_dir)
    candidates = sorted(checkpoint_path.glob("batch_*.pt"))
    if not candidates:
        raise ValueError(f"No PPO checkpoints found in {checkpoint_path}")
    return str(candidates[-1])


def create_output_paths(output_dir: str | Path) -> tuple[Path, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"benchmark_{timestamp}.csv", output_path / f"benchmark_{timestamp}.json"


def create_resume_output_paths(output_dir: str | Path, run_id: str | None) -> tuple[Path, Path]:
    if run_id is None:
        return create_output_paths(output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_run_id = run_id.replace("/", "_")
    return output_path / f"benchmark_{safe_run_id}.csv", output_path / f"benchmark_{safe_run_id}.json"


def create_log_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".log")


def configure_worker_logging(log_path: str) -> None:
    configure_process_logging(log_path)


def build_matchup_opponents(latest_checkpoint_path: str) -> dict[str, tuple[BenchmarkSeatSpec, ...]]:
    greedy_config = AgentConfig(agent_type=AgentType.GREEDY)
    ppo_config = AgentConfig(agent_type=AgentType.TRAINED_PPO, checkpoint_path=latest_checkpoint_path)
    return {
        MATCHUP_THREE_GREEDY: (
            BenchmarkSeatSpec("greedy_1", "opponent", greedy_config),
            BenchmarkSeatSpec("greedy_2", "opponent", greedy_config),
            BenchmarkSeatSpec("greedy_3", "opponent", greedy_config),
        ),
        MATCHUP_THREE_PPO: (
            BenchmarkSeatSpec("latest_ppo_1", "opponent", ppo_config),
            BenchmarkSeatSpec("latest_ppo_2", "opponent", ppo_config),
            BenchmarkSeatSpec("latest_ppo_3", "opponent", ppo_config),
        ),
    }


def validate_candidate_config(candidate_config: ArenaConfig) -> None:
    if not candidate_config.entrants:
        raise ValueError("Benchmark config must contain at least one entrant")
    invalid_ids = [entrant.id for entrant in candidate_config.entrants if entrant.agent_type not in LLM_AGENT_TYPES]
    if invalid_ids:
        raise ValueError(
            "Benchmark config must contain only LLM entrants. "
            f"Invalid entrants: {', '.join(invalid_ids)}"
        )


def build_benchmark_tasks(
    candidate_config: ArenaConfig,
    *,
    games_per_matchup: int,
    latest_checkpoint_path: str,
    seed: int,
) -> list[BenchmarkTask]:
    if games_per_matchup <= 0:
        raise ValueError("games_per_matchup must be positive")
    validate_candidate_config(candidate_config)
    matchup_opponents = build_matchup_opponents(latest_checkpoint_path)
    tasks: list[BenchmarkTask] = []
    global_game_index = 0
    for candidate in candidate_config.entrants:
        candidate_seat = BenchmarkSeatSpec(candidate.id, "candidate", candidate)
        for matchup in MATCHUP_ORDER:
            opponents = list(matchup_opponents[matchup])
            for matchup_game_index in range(games_per_matchup):
                seats = [candidate_seat, *opponents]
                setup_seed_base = seed + matchup_game_index
                seat_seed = setup_seed_base
                deal_seed = setup_seed_base + 1_000_000
                rng = random.Random(seat_seed)
                rng.shuffle(seats)
                tasks.append(
                    BenchmarkTask(
                        global_game_index=global_game_index,
                        candidate_id=candidate.id,
                        candidate_agent_type=candidate.agent_type.value,
                        matchup=matchup,
                        matchup_game_index=matchup_game_index,
                        seat_seed=seat_seed,
                        deal_seed=deal_seed,
                        seats=tuple(seats),
                    )
                )
                global_game_index += 1
    return tasks


def build_game_results(
    task: BenchmarkTask,
    env: Big2Env,
    pass_counts: list[int],
    token_usage: list[dict[str, int]],
    elapsed_ms: list[int],
) -> list[BenchmarkPlayerResult]:
    if env.winner is None:
        raise ValueError("Expected a winner for completed benchmark game")
    results: list[BenchmarkPlayerResult] = []
    for seat, seat_spec in enumerate(task.seats):
        usage = token_usage[seat]
        results.append(
            BenchmarkPlayerResult(
                global_game_index=task.global_game_index,
                candidate_id=task.candidate_id,
                candidate_agent_type=task.candidate_agent_type,
                matchup=task.matchup,
                matchup_game_index=task.matchup_game_index,
                entrant_id=seat_spec.entrant_id,
                entrant_role=seat_spec.role,
                agent_type=seat_spec.config.agent_type.value,
                checkpoint_path=seat_spec.config.checkpoint_path or "",
                seat=seat,
                winner_seat=env.winner,
                won=seat == env.winner,
                cards_remaining=len(env.hands[seat]),
                score=compute_final_score(env, seat),
                pass_count=pass_counts[seat],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                llm_elapsed_ms=elapsed_ms[seat],
            )
        )
    return results


def assign_game_session_ids(task: BenchmarkTask, agents: list[object]) -> None:
    for seat, agent in enumerate(agents):
        if isinstance(agent, LLMPlayAgent):
            agent.set_game_session_id(
                f"big2-benchmark-game-{task.global_game_index}-seat-{seat}-{task.seats[seat].entrant_id}"
            )


def play_benchmark_game(task: BenchmarkTask) -> list[BenchmarkPlayerResult]:
    env = Big2Env(n_players=4)
    random.seed(task.deal_seed)
    env.reset()
    agents = [load(seat_spec.config) for seat_spec in task.seats]
    assign_game_session_ids(task, agents)
    pass_counts = [0 for _ in task.seats]
    token_usage = [
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for _ in task.seats
    ]
    elapsed_ms = [0 for _ in task.seats]

    while not env.done:
        actor_id = env.current_player
        turn = build_turn(env, actor_id)
        choice = agents[actor_id].choose_action(turn)
        if choice.action.type == PASS:
            pass_counts[actor_id] += 1
        llm_usage = choice.metadata.get("llm_usage")
        if isinstance(llm_usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_usage[actor_id][key] += int(llm_usage.get(key, 0) or 0)
        elapsed_ms[actor_id] += int(choice.metadata.get("llm_elapsed_ms", 0) or 0)
        env.step(choice.action)
        for agent in agents:
            agent.observe_action(actor_id, choice.action)

    for agent in agents:
        agent.on_game_end(env.winner)

    results = build_game_results(task, env, pass_counts, token_usage, elapsed_ms)
    candidate_result = next(result for result in results if result.entrant_role == "candidate")
    print(
        f"{log_timestamp()} [BENCHMARK] game={task.global_game_index} candidate={task.candidate_id} "
        f"matchup={task.matchup} seat={candidate_result.seat} won={candidate_result.won} "
        f"score={candidate_result.score} passes={candidate_result.pass_count} "
        f"tokens={candidate_result.total_tokens} elapsed_ms={candidate_result.llm_elapsed_ms}",
        file=sys.stderr,
        flush=True,
    )
    return results


def run_tasks_chunk(tasks: list[BenchmarkTask]) -> list[BenchmarkPlayerResult]:
    results: list[BenchmarkPlayerResult] = []
    try:
        for task in tasks:
            results.extend(play_benchmark_game(task))
    except Exception as exc:
        task_label = "unknown"
        if "task" in locals():
            task_label = (
                f"game={task.global_game_index} candidate={task.candidate_id} "
                f"matchup={task.matchup} matchup_game_index={task.matchup_game_index}"
            )
        raise WorkerExecutionError(
            f"Benchmark worker failed for {task_label}: {exc.__class__.__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        ) from None
    return results


def run_task(task: BenchmarkTask) -> list[BenchmarkPlayerResult]:
    try:
        return play_benchmark_game(task)
    except Exception as exc:
        raise WorkerExecutionError(
            f"Benchmark worker failed for game={task.global_game_index} candidate={task.candidate_id} "
            f"matchup={task.matchup} matchup_game_index={task.matchup_game_index}: "
            f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"
        ) from None


def aggregate_candidate_results(results: list[BenchmarkPlayerResult]) -> dict[str, object]:
    candidate_rows = [result for result in results if result.entrant_role == "candidate"]
    if not candidate_rows:
        return {}
    seat_counter = Counter(result.seat for result in candidate_rows)
    return {
        "games_played": len(candidate_rows),
        "wins": sum(1 for result in candidate_rows if result.won),
        "win_rate": sum(1 for result in candidate_rows if result.won) / len(candidate_rows),
        "average_score": sum(result.score for result in candidate_rows) / len(candidate_rows),
        "average_cards_remaining": sum(result.cards_remaining for result in candidate_rows) / len(candidate_rows),
        "average_pass_count": sum(result.pass_count for result in candidate_rows) / len(candidate_rows),
        "average_prompt_tokens": sum(result.prompt_tokens for result in candidate_rows) / len(candidate_rows),
        "average_completion_tokens": sum(result.completion_tokens for result in candidate_rows) / len(candidate_rows),
        "average_total_tokens": sum(result.total_tokens for result in candidate_rows) / len(candidate_rows),
        "average_llm_elapsed_ms": sum(result.llm_elapsed_ms for result in candidate_rows) / len(candidate_rows),
        "seat_counts": dict(sorted(seat_counter.items())),
    }


def build_summary(
    results: list[BenchmarkPlayerResult],
    *,
    candidates: tuple[ArenaEntrantConfig, ...],
    latest_checkpoint_path: str,
    games_per_matchup: int,
    seed: int,
    workers: int,
    parallel_backend: str,
    csv_path: Path,
    run_id: str | None = None,
    resume: bool = False,
    total_games: int | None = None,
) -> dict[str, object]:
    overall: dict[str, dict[str, object]] = {}
    by_matchup: dict[str, dict[str, dict[str, object]]] = {}
    for candidate in candidates:
        candidate_rows = [result for result in results if result.candidate_id == candidate.id]
        overall[candidate.id] = aggregate_candidate_results(candidate_rows)
        by_matchup[candidate.id] = {}
        for matchup in MATCHUP_ORDER:
            matchup_rows = [
                result
                for result in candidate_rows
                if result.matchup == matchup
            ]
            by_matchup[candidate.id][matchup] = aggregate_candidate_results(matchup_rows)
    return {
        "games_per_matchup": games_per_matchup,
        "matchups": list(MATCHUP_ORDER),
        "seed": seed,
        "shared_setup_across_candidates": True,
        "workers": workers,
        "parallel_backend": parallel_backend,
        "latest_checkpoint_path": latest_checkpoint_path,
        "candidates": [
            {
                "id": candidate.id,
                "agent_type": candidate.agent_type.value,
            }
            for candidate in candidates
        ],
        "overall": overall,
        "by_matchup": by_matchup,
        "csv_path": str(csv_path),
        "run_id": run_id,
        "resume": resume,
        "completed_games": len(completed_game_indices(results)),
        "total_games": len(results) // 4 if total_games is None else total_games,
    }


def write_csv_results(path: str | Path, results: list[BenchmarkPlayerResult]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def append_csv_game_results(path: str | Path, results: list[BenchmarkPlayerResult]) -> None:
    csv_path = Path(path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        if write_header:
            writer.writeheader()
        for result in sorted(results, key=lambda result: result.seat):
            writer.writerow(asdict(result))
        handle.flush()


def load_csv_results(path: str | Path) -> list[BenchmarkPlayerResult]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    return [
        BenchmarkPlayerResult(
            global_game_index=int(row["global_game_index"]),
            candidate_id=row["candidate_id"],
            candidate_agent_type=row["candidate_agent_type"],
            matchup=row["matchup"],
            matchup_game_index=int(row["matchup_game_index"]),
            entrant_id=row["entrant_id"],
            entrant_role=row["entrant_role"],
            agent_type=row["agent_type"],
            checkpoint_path=row["checkpoint_path"],
            seat=int(row["seat"]),
            winner_seat=int(row["winner_seat"]),
            won=row["won"] == "True",
            cards_remaining=int(row["cards_remaining"]),
            score=int(row["score"]),
            pass_count=int(row["pass_count"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            total_tokens=int(row["total_tokens"]),
            llm_elapsed_ms=int(row["llm_elapsed_ms"]),
        )
        for row in rows
    ]


def completed_game_indices(results: list[BenchmarkPlayerResult]) -> set[int]:
    grouped: dict[int, list[BenchmarkPlayerResult]] = defaultdict(list)
    for result in results:
        grouped[result.global_game_index].append(result)
    return {game_index for game_index, game_results in grouped.items() if len(game_results) == 4}


def run_benchmark(
    candidate_config: ArenaConfig,
    *,
    games_per_matchup: int,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    output_dir: str | Path = "play_evals",
    seed: int = 0,
    workers: int = 1,
    run_id: str | None = None,
    resume: bool = False,
    tee_stderr_to_file: bool = False,
) -> tuple[list[BenchmarkPlayerResult], dict[str, object], Path, Path]:
    validate_candidate_config(candidate_config)
    latest_checkpoint_path = find_latest_checkpoint_path(checkpoint_dir)
    tasks = build_benchmark_tasks(
        candidate_config,
        games_per_matchup=games_per_matchup,
        latest_checkpoint_path=latest_checkpoint_path,
        seed=seed,
    )

    for candidate in candidate_config.entrants:
        agent = load(candidate)
        if getattr(agent, "client", None) is None:
            raise ValueError(
                "LLM benchmark entrants require OPENROUTER_API_KEY before running benchmark. "
                f"Missing credentials for: {candidate.id}"
            )

    csv_path, json_path = create_resume_output_paths(output_dir, run_id)
    log_path = create_log_path(csv_path)
    if not resume:
        for output_path in (csv_path, json_path, log_path):
            if output_path.exists():
                output_path.unlink()
    if tee_stderr_to_file:
        configure_process_logging(log_path)
    existing_results = load_csv_results(csv_path) if resume else []
    completed_indices = completed_game_indices(existing_results)
    pending_tasks = [task for task in tasks if task.global_game_index not in completed_indices]
    if resume:
        print(
            f"{log_timestamp()} [BENCHMARK] resume completed_games={len(completed_indices)} "
            f"pending_games={len(pending_tasks)} csv={csv_path}",
            file=sys.stderr,
            flush=True,
        )

    backend = "serial"
    new_results: list[BenchmarkPlayerResult] = []
    if not pending_tasks:
        backend = "resume_noop" if resume else "serial"
    elif workers <= 1:
        for task in pending_tasks:
            game_results = run_task(task)
            append_csv_game_results(csv_path, game_results)
            new_results.extend(game_results)
    else:
        bounded_workers = min(workers, len(pending_tasks))
        executor_cls = ProcessPoolExecutor
        try:
            with executor_cls(
                max_workers=bounded_workers,
                initializer=configure_worker_logging,
                initargs=(str(log_path),),
            ) as executor:
                backend = "process"
                futures = [executor.submit(run_task, task) for task in pending_tasks]
                try:
                    for future in as_completed(futures):
                        game_results = future.result()
                        append_csv_game_results(csv_path, game_results)
                        new_results.extend(game_results)
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
        except (NotImplementedError, PermissionError, OSError):
            with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
                backend = "thread"
                futures = [executor.submit(run_task, task) for task in pending_tasks]
                try:
                    for future in as_completed(futures):
                        game_results = future.result()
                        append_csv_game_results(csv_path, game_results)
                        new_results.extend(game_results)
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise

    all_results = load_csv_results(csv_path) if csv_path.exists() else new_results
    all_results = sorted(all_results, key=lambda result: (result.global_game_index, result.seat))
    summary = build_summary(
        all_results,
        candidates=candidate_config.entrants,
        latest_checkpoint_path=latest_checkpoint_path,
        games_per_matchup=games_per_matchup,
        seed=seed,
        workers=workers,
        parallel_backend=backend,
        csv_path=csv_path,
        run_id=run_id,
        resume=resume,
        total_games=len(tasks),
    )
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return all_results, summary, csv_path, json_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LLM Big 2 agents against fixed deterministic opponent sets.")
    parser.add_argument("--config", required=True, help="JSON file describing the LLM candidates to benchmark")
    parser.add_argument("--games", type=int, required=True, help="Games per matchup for each candidate")
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing PPO checkpoints; latest batch_*.pt will be used",
    )
    parser.add_argument("--output-dir", default="play_evals", help="Directory for CSV and JSON outputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for benchmark deals and seat assignment")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for benchmark games")
    parser.add_argument("--run-id", default=None, help="Stable output id for resumable benchmark runs")
    parser.add_argument("--resume", action="store_true", help="Resume a prior run by skipping completed game ids in CSV")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    candidate_config = load_arena_config(args.config)
    _results, summary, csv_path, json_path = run_benchmark(
        candidate_config,
        games_per_matchup=args.games,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        workers=args.workers,
        run_id=args.run_id,
        resume=args.resume,
        tee_stderr_to_file=True,
    )
    print(f"Ran {args.games} games per matchup for {len(candidate_config.entrants)} candidates")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Log: {create_log_path(csv_path)}")
    for candidate_id, candidate_summary in summary["overall"].items():
        print(
            f"{candidate_id}: wins={candidate_summary['wins']}/{candidate_summary['games_played']} "
            f"win_rate={candidate_summary['win_rate']:.3f} "
            f"avg_score={candidate_summary['average_score']:.3f} "
            f"avg_tokens={candidate_summary['average_total_tokens']:.1f} "
            f"avg_elapsed_ms={candidate_summary['average_llm_elapsed_ms']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
