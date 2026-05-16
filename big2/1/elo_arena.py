from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from big2.play.logging_utils import log_timestamp
from big2.play.agents import LLMPlayAgent
from big2.play.config_loader import ArenaConfig, ArenaEntrantConfig, load, load_arena_config
from big2.play.runner import build_turn
from big2.play.types import PlayAgent
from big2.simulator.cards import PASS
from big2.simulator.env import Big2Env

DEFAULT_ELO = 1500.0
DEFAULT_K_FACTOR = 24.0


class WorkerExecutionError(RuntimeError):
    pass


def _assign_game_session_ids(agents: list[PlayAgent], entrants_by_seat: list[ArenaEntrantConfig], *, game_index: int) -> None:
    for seat, agent in enumerate(agents):
        if isinstance(agent, LLMPlayAgent):
            agent.set_game_session_id(f"big2-arena-game-{game_index}-seat-{seat}-{entrants_by_seat[seat].id}")


@dataclass(frozen=True)
class GamePlayerResult:
    game_index: int
    entrant_id: str
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


def expected_score(player_rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_rating - player_rating) / 400.0))


def update_elo_pair(ratings: dict[str, float], entrant_a: str, entrant_b: str, result_a: float, k_factor: float) -> None:
    rating_a = ratings[entrant_a]
    rating_b = ratings[entrant_b]
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    result_b = 1.0 - result_a
    ratings[entrant_a] = rating_a + k_factor * (result_a - expected_a)
    ratings[entrant_b] = rating_b + k_factor * (result_b - expected_b)


def compute_final_score(env: Big2Env, seat: int) -> int:
    cards_remaining = len(env.hands[seat])
    if seat == env.winner:
        return -sum(len(env.hands[player]) for player in range(env.n_players) if player != seat)
    return cards_remaining


def build_game_results(
    *,
    game_index: int,
    entrants_by_seat: list[ArenaEntrantConfig],
    env: Big2Env,
    pass_counts: list[int],
    token_usage: list[dict[str, int]],
) -> list[GamePlayerResult]:
    if env.winner is None:
        raise ValueError("Expected a winner for completed Big 2 game")
    results: list[GamePlayerResult] = []
    for seat, entrant in enumerate(entrants_by_seat):
        usage = token_usage[seat]
        results.append(
            GamePlayerResult(
                game_index=game_index,
                entrant_id=entrant.id,
                agent_type=entrant.agent_type.value,
                checkpoint_path=entrant.checkpoint_path or "",
                seat=seat,
                winner_seat=env.winner,
                won=seat == env.winner,
                cards_remaining=len(env.hands[seat]),
                score=compute_final_score(env, seat),
                pass_count=pass_counts[seat],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )
        )
    return results


def update_win_elo(ratings: dict[str, float], results: list[GamePlayerResult], k_factor: float) -> None:
    winner = next(result for result in results if result.won)
    for loser in results:
        if loser.entrant_id == winner.entrant_id:
            continue
        update_elo_pair(ratings, winner.entrant_id, loser.entrant_id, 1.0, k_factor)


def update_score_elo(ratings: dict[str, float], results: list[GamePlayerResult], k_factor: float) -> None:
    for left_index, left in enumerate(results):
        for right in results[left_index + 1 :]:
            if left.score < right.score:
                result_left = 1.0
            elif left.score > right.score:
                result_left = 0.0
            else:
                result_left = 0.5
            update_elo_pair(ratings, left.entrant_id, right.entrant_id, result_left, k_factor)


def play_game_with_metrics(
    env: Big2Env,
    entrants_by_seat: list[ArenaEntrantConfig],
    agents_by_id: dict[str, PlayAgent],
    *,
    game_index: int,
    game_seed: int,
) -> list[GamePlayerResult]:
    random.seed(game_seed)
    env.reset()
    pass_counts = [0 for _ in entrants_by_seat]
    token_usage = [
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for _ in entrants_by_seat
    ]
    agents = [agents_by_id[entrant.id] for entrant in entrants_by_seat]
    _assign_game_session_ids(agents, entrants_by_seat, game_index=game_index)
    while not env.done:
        actor_id = env.current_player
        print(
            f"{log_timestamp()} [ARENA] game={game_index} turn_actor={actor_id} entrant={entrants_by_seat[actor_id].id}",
            file=sys.stderr,
            flush=True,
        )
        turn = build_turn(env, actor_id)
        choice = agents[actor_id].choose_action(turn)
        if choice.action.type == PASS:
            pass_counts[actor_id] += 1
        llm_usage = choice.metadata.get("llm_usage")
        if isinstance(llm_usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_usage[actor_id][key] += int(llm_usage.get(key, 0) or 0)
        env.step(choice.action)
        for agent in agents:
            agent.observe_action(actor_id, choice.action)

    for agent in agents:
        agent.on_game_end(env.winner)

    results = build_game_results(
        game_index=game_index,
        entrants_by_seat=entrants_by_seat,
        env=env,
        pass_counts=pass_counts,
        token_usage=token_usage,
    )
    print(
        f"{log_timestamp()} [ARENA] game={game_index} winner_seat={env.winner} winner_id={entrants_by_seat[env.winner].id}",
        file=sys.stderr,
        flush=True,
    )
    for result in results:
        print(
            f"{log_timestamp()} [ARENA] "
            f"game={result.game_index} seat={result.seat} entrant={result.entrant_id} "
            f"won={result.won} score={result.score} cards_left={result.cards_remaining} "
            f"passes={result.pass_count} tokens={result.total_tokens}",
            file=sys.stderr,
            flush=True,
        )
    return results


def sample_entrants_for_game(
    entrants: tuple[ArenaEntrantConfig, ...],
    *,
    game_index: int,
    seed: int,
) -> list[ArenaEntrantConfig]:
    rng = random.Random(seed + game_index)
    selected = rng.sample(list(entrants), 4)
    rng.shuffle(selected)
    return selected


def chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    iterator = iter(values)
    chunks: list[list[int]] = []
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return chunks
        chunks.append(chunk)


def run_games_chunk(
    arena_config: ArenaConfig,
    game_indices: list[int],
    *,
    seed: int,
) -> list[GamePlayerResult]:
    env = Big2Env(n_players=4)
    agents_by_id = {entrant.id: load(entrant) for entrant in arena_config.entrants}
    results: list[GamePlayerResult] = []
    try:
        for game_index in game_indices:
            entrants_by_seat = sample_entrants_for_game(
                arena_config.entrants,
                game_index=game_index,
                seed=seed,
            )
            results.extend(
                play_game_with_metrics(
                    env,
                    entrants_by_seat,
                    agents_by_id,
                    game_index=game_index,
                    game_seed=seed + 1_000_000 + game_index,
                )
            )
    except Exception as exc:
        game_label = f"game={game_index}" if "game_index" in locals() else "game=unknown"
        raise WorkerExecutionError(
            f"Arena worker failed for {game_label}: {exc.__class__.__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        ) from None
    return results


def apply_elo_updates(
    arena_config: ArenaConfig,
    results: list[GamePlayerResult],
    *,
    k_factor: float,
) -> tuple[dict[str, float], dict[str, float]]:
    win_elo = {entrant.id: DEFAULT_ELO for entrant in arena_config.entrants}
    score_elo = {entrant.id: DEFAULT_ELO for entrant in arena_config.entrants}
    grouped_results: dict[int, list[GamePlayerResult]] = defaultdict(list)
    for result in results:
        grouped_results[result.game_index].append(result)
    for game_index in sorted(grouped_results):
        game_results = sorted(grouped_results[game_index], key=lambda result: result.seat)
        update_win_elo(win_elo, game_results, k_factor)
        update_score_elo(score_elo, game_results, k_factor)
    return win_elo, score_elo


def aggregate_results(results: list[GamePlayerResult]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[GamePlayerResult]] = defaultdict(list)
    for result in results:
        grouped[result.entrant_id].append(result)

    summary: dict[str, dict[str, object]] = {}
    for entrant_id, entrant_results in grouped.items():
        seat_counter = Counter(result.seat for result in entrant_results)
        summary[entrant_id] = {
            "games_played": len(entrant_results),
            "wins": sum(1 for result in entrant_results if result.won),
            "win_rate": sum(1 for result in entrant_results if result.won) / len(entrant_results),
            "average_score": sum(result.score for result in entrant_results) / len(entrant_results),
            "average_cards_remaining": sum(result.cards_remaining for result in entrant_results) / len(entrant_results),
            "average_pass_count": sum(result.pass_count for result in entrant_results) / len(entrant_results),
            "average_prompt_tokens": sum(result.prompt_tokens for result in entrant_results) / len(entrant_results),
            "average_completion_tokens": sum(result.completion_tokens for result in entrant_results) / len(entrant_results),
            "average_total_tokens": sum(result.total_tokens for result in entrant_results) / len(entrant_results),
            "seat_counts": dict(sorted(seat_counter.items())),
        }
    return summary


def create_output_paths(output_dir: str | Path) -> tuple[Path, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"arena_{timestamp}.csv", output_path / f"arena_{timestamp}.json"


def run_arena(
    arena_config: ArenaConfig,
    *,
    games: int,
    output_dir: str | Path = "play_evals",
    seed: int = 0,
    k_factor: float = DEFAULT_K_FACTOR,
    workers: int = 1,
) -> tuple[list[GamePlayerResult], dict[str, object], Path, Path]:
    if games <= 0:
        raise ValueError("games must be positive")
    if len(arena_config.entrants) < 4:
        raise ValueError("Arena requires at least 4 entrants")
    agents_by_id = {entrant.id: load(entrant) for entrant in arena_config.entrants}
    missing_llm_ids = [
        entrant.id
        for entrant in arena_config.entrants
        if isinstance(agents_by_id[entrant.id], LLMPlayAgent) and agents_by_id[entrant.id].client is None
    ]
    if missing_llm_ids:
        raise ValueError(
            "LLM entrants require OPENROUTER_API_KEY before running arena. "
            f"Missing credentials for: {', '.join(missing_llm_ids)}"
        )
    game_indices = list(range(games))
    backend = "serial"
    if workers <= 1:
        all_results = run_games_chunk(arena_config, game_indices, seed=seed)
    else:
        bounded_workers = min(workers, games)
        chunk_size = max(1, (games + bounded_workers - 1) // bounded_workers)
        all_results = []
        executor_cls = ProcessPoolExecutor
        try:
            with executor_cls(max_workers=bounded_workers) as executor:
                backend = "process"
                futures = [
                    executor.submit(run_games_chunk, arena_config, game_chunk, seed=seed)
                    for game_chunk in chunked(game_indices, chunk_size)
                ]
                try:
                    for future in as_completed(futures):
                        all_results.extend(future.result())
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
        except (NotImplementedError, PermissionError, OSError):
            with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
                backend = "thread"
                futures = [
                    executor.submit(run_games_chunk, arena_config, game_chunk, seed=seed)
                    for game_chunk in chunked(game_indices, chunk_size)
                ]
                try:
                    for future in as_completed(futures):
                        all_results.extend(future.result())
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise

    all_results = sorted(all_results, key=lambda result: (result.game_index, result.seat))
    win_elo, score_elo = apply_elo_updates(arena_config, all_results, k_factor=k_factor)

    csv_path, json_path = create_output_paths(output_dir)
    write_csv_results(csv_path, all_results)

    summary = {
        "games": games,
        "seed": seed,
        "k_factor": k_factor,
        "workers": workers,
        "parallel_backend": backend,
        "entrants": [
            {
                "id": entrant.id,
                "agent_type": entrant.agent_type.value,
                "checkpoint_path": entrant.checkpoint_path,
                "device": entrant.device,
            }
            for entrant in arena_config.entrants
        ],
        "win_elo": dict(sorted(win_elo.items(), key=lambda item: item[1], reverse=True)),
        "score_elo": dict(sorted(score_elo.items(), key=lambda item: item[1], reverse=True)),
        "aggregates": aggregate_results(all_results),
        "csv_path": str(csv_path),
    }
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    return all_results, summary, csv_path, json_path


def write_csv_results(path: str | Path, results: list[GamePlayerResult]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Big 2 Elo arena over a pool of entrants.")
    parser.add_argument("--config", required=True, help="JSON file describing the entrant pool")
    parser.add_argument("--games", type=int, required=True, help="Number of games to run")
    parser.add_argument("--output-dir", default="play_evals", help="Directory for CSV and JSON arena outputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for entrant sampling and seat assignment")
    parser.add_argument("--k-factor", type=float, default=DEFAULT_K_FACTOR, help="Elo K-factor for both ratings")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for game execution")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    arena_config = load_arena_config(args.config)
    _results, summary, csv_path, json_path = run_arena(
        arena_config,
        games=args.games,
        output_dir=args.output_dir,
        seed=args.seed,
        k_factor=args.k_factor,
        workers=args.workers,
    )
    print(f"Ran {args.games} games across {len(arena_config.entrants)} entrants")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print("Top win Elo:")
    for entrant_id, rating in list(summary["win_elo"].items())[:5]:
        print(f"{entrant_id}: {rating:.2f}")
    print("Top score Elo:")
    for entrant_id, rating in list(summary["score_elo"].items())[:5]:
        print(f"{entrant_id}: {rating:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
