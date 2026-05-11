from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

from big2.simulator.cards import (
    FLUSH,
    FOUR_KIND,
    FULLHOUSE,
    PAIR,
    PASS,
    SINGLE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPLE,
    Combo,
)
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy

COMBO_TYPE_NAMES = {
    PASS: "pass",
    SINGLE: "single",
    PAIR: "pair",
    TRIPLE: "triple",
    STRAIGHT: "straight",
    FLUSH: "flush",
    FULLHOUSE: "full_house",
    FOUR_KIND: "four_kind",
    STRAIGHT_FLUSH: "straight_flush",
}


@dataclass(frozen=True)
class DecisionRecord:
    episode: int
    turn: int
    player: int
    hand_size: int
    phase: str
    has_control: bool
    active_trick_type: str
    active_trick_size: int
    passes_in_row: int
    num_legal: int
    num_non_pass: int
    num_singles: int
    num_pairs: int
    num_triples: int
    num_five_card: int


def hand_phase(hand_size: int) -> str:
    if hand_size > 10:
        return "early"
    if hand_size > 5:
        return "mid"
    return "late"


def choose_action(strategy: str, candidates: list[Combo], hand: list[int], trick_pile: Combo | None) -> Combo:
    if strategy == "random":
        return random.choice(candidates)
    if strategy == "greedy":
        return greedy_strategy(candidates)
    if strategy == "smart":
        return smart_strategy(candidates, hand, trick_pile)
    raise ValueError(f"Unknown rollout strategy: {strategy}")


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return float(ordered[idx])


def summarize_counts(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def record_decision(episode: int, turn: int, env: Big2Env, candidates: list[Combo]) -> DecisionRecord:
    player = env.current_player
    counts = Counter(c.type for c in candidates)
    active_type = "none"
    active_size = 0
    if env.trick_pile is not None:
        active_type = COMBO_TYPE_NAMES.get(env.trick_pile.type, f"type_{env.trick_pile.type}")
        active_size = env.trick_pile.size()

    num_five_card = sum(
        counts[t]
        for t in (
            STRAIGHT,
            FLUSH,
            FULLHOUSE,
            FOUR_KIND,
            STRAIGHT_FLUSH,
        )
    )
    return DecisionRecord(
        episode=episode,
        turn=turn,
        player=player,
        hand_size=len(env.hands[player]),
        phase=hand_phase(len(env.hands[player])),
        has_control=env.trick_pile is None or env.trick_pile.type == PASS,
        active_trick_type=active_type,
        active_trick_size=active_size,
        passes_in_row=env.passes_in_row,
        num_legal=len(candidates),
        num_non_pass=sum(1 for c in candidates if c.type != PASS),
        num_singles=counts[SINGLE],
        num_pairs=counts[PAIR],
        num_triples=counts[TRIPLE],
        num_five_card=num_five_card,
    )


def collect_records(episodes: int, n_players: int, strategy: str, seed: int) -> list[DecisionRecord]:
    random.seed(seed)
    records: list[DecisionRecord] = []

    for episode in range(episodes):
        env = Big2Env(n_players)
        env.reset()
        turn = 0
        while not env.done:
            player = env.current_player
            candidates = env.legal_candidates(player)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            records.append(record_decision(episode, turn, env, candidates))
            action = choose_action(strategy, candidates, env.hands[player], env.trick_pile)
            env.step(action)
            turn += 1

    return records


def grouped_summary(records: list[DecisionRecord], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, field))].append(record.num_legal)
    return {key: summarize_counts(values) for key, values in sorted(groups.items())}


def build_summary(records: list[DecisionRecord], episodes: int, n_players: int, strategy: str, seed: int) -> dict:
    legal_counts = [record.num_legal for record in records]
    non_pass_counts = [record.num_non_pass for record in records]
    turns_by_episode = Counter(record.episode for record in records)
    combo_totals = {
        "single": sum(record.num_singles for record in records),
        "pair": sum(record.num_pairs for record in records),
        "triple": sum(record.num_triples for record in records),
        "five_card": sum(record.num_five_card for record in records),
        "pass": sum(record.num_legal - record.num_non_pass for record in records),
    }
    max_record = max(records, key=lambda record: record.num_legal) if records else None

    return {
        "metadata": {
            "episodes": episodes,
            "n_players": n_players,
            "rollout_strategy": strategy,
            "seed": seed,
            "num_decisions": len(records),
        },
        "turns_per_episode": summarize_counts(list(turns_by_episode.values())),
        "legal_actions": summarize_counts(legal_counts),
        "non_pass_legal_actions": summarize_counts(non_pass_counts),
        "legal_actions_by_phase": grouped_summary(records, "phase"),
        "legal_actions_by_hand_size": grouped_summary(records, "hand_size"),
        "legal_actions_by_active_trick_type": grouped_summary(records, "active_trick_type"),
        "candidate_type_totals": combo_totals,
        "max_legal_decision": asdict(max_record) if max_record is not None else None,
    }


def write_csv(records: list[DecisionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DecisionRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Big 2 legal action-space sizes from simulator rollouts.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of games to simulate.")
    parser.add_argument("--n-players", type=int, default=4, help="Number of players in Big2Env.")
    parser.add_argument(
        "--strategy",
        choices=["random", "greedy", "smart"],
        default="random",
        help="Rollout policy used to generate visited states.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for JSON summary.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional path for per-decision CSV rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records(
        episodes=args.episodes,
        n_players=args.n_players,
        strategy=args.strategy,
        seed=args.seed,
    )
    summary = build_summary(
        records=records,
        episodes=args.episodes,
        n_players=args.n_players,
        strategy=args.strategy,
        seed=args.seed,
    )

    summary_json = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary_json + "\n")
    else:
        print(summary_json)

    if args.csv_output is not None:
        write_csv(records, args.csv_output)


if __name__ == "__main__":
    main()
