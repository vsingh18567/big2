#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

DEFAULT_CSV_PATH = Path("play_evals/benchmark_20260410T032308Z.csv") if len(sys.argv) == 1 else sys.argv[1]
EXPECTED_PLAYERS_PER_GAME = 4
NUMERIC_FIELDS = (
    "global_game_index",
    "matchup_game_index",
    "seat",
    "winner_seat",
    "cards_remaining",
    "score",
    "pass_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "llm_elapsed_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a benchmark CSV produced by big2.play.benchmark_suite.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to the benchmark CSV (default: {DEFAULT_CSV_PATH})",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row: dict[str, Any] = dict(raw_row)
        for field in NUMERIC_FIELDS:
            row[field] = int(raw_row[field])
        row["won"] = raw_row["won"] == "True"
        rows.append(row)
    return rows


def safe_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.2f}"


def seat_counts(rows: list[dict[str, Any]]) -> str:
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        counts[row["seat"]] += 1
    return ", ".join(f"{seat}:{counts[seat]}" for seat in sorted(counts))


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    wins = sum(1 for row in rows if row["won"])
    return {
        "games": str(len(rows)),
        "wins": str(wins),
        "win_rate": format_rate(wins, len(rows)),
        "avg_score": format_float(safe_mean([row["score"] for row in rows])),
        "avg_cards_left": format_float(safe_mean([row["cards_remaining"] for row in rows])),
        "avg_passes": format_float(safe_mean([row["pass_count"] for row in rows])),
        "avg_tokens": format_float(safe_mean([row["total_tokens"] for row in rows])),
        "avg_elapsed_ms": format_float(safe_mean([row["llm_elapsed_ms"] for row in rows])),
        "seats": seat_counts(rows),
    }


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    rendered = [render_row(headers), render_row(["-" * width for width in widths])]
    rendered.extend(render_row(row) for row in rows)
    return "\n".join(rendered)


def analyze(csv_path: Path) -> str:
    rows = load_rows(csv_path)
    if not rows:
        return f"No rows found in {csv_path}"

    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        games[row["global_game_index"]].append(row)

    complete_games = sum(1 for game_rows in games.values() if len(game_rows) == EXPECTED_PLAYERS_PER_GAME)
    candidate_rows = [row for row in rows if row["entrant_role"] == "candidate"]
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates[str(row["candidate_id"])].append(row)

    sections = [
        f"Benchmark analysis: {csv_path}",
        (
            "Rows: "
            f"{len(rows)} total, "
            f"{len(candidate_rows)} candidate rows, "
            f"{len(games)} discovered games, "
            f"{complete_games} complete games"
        ),
    ]

    overall_rows: list[list[str]] = []
    for candidate_id in sorted(candidates):
        summary = summarize_candidate_rows(candidates[candidate_id])
        overall_rows.append(
            [
                candidate_id,
                summary["games"],
                summary["wins"],
                summary["win_rate"],
                summary["avg_score"],
                summary["avg_cards_left"],
                summary["avg_passes"],
                summary["avg_tokens"],
                summary["avg_elapsed_ms"],
                summary["seats"],
            ]
        )

    sections.append("")
    sections.append("Overall candidate results")
    sections.append(
        render_table(
            [
                "candidate",
                "games",
                "wins",
                "win_rate",
                "avg_score",
                "avg_cards_left",
                "avg_passes",
                "avg_tokens",
                "avg_elapsed_ms",
                "seats",
            ],
            overall_rows,
        )
    )

    sections.append("")
    sections.append("Per-matchup candidate results")
    for candidate_id in sorted(candidates):
        matchups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates[candidate_id]:
            matchups[str(row["matchup"])].append(row)

        sections.append(f"candidate={candidate_id}")
        matchup_rows: list[list[str]] = []
        for matchup in sorted(matchups):
            summary = summarize_candidate_rows(matchups[matchup])
            matchup_rows.append(
                [
                    matchup,
                    summary["games"],
                    summary["wins"],
                    summary["win_rate"],
                    summary["avg_score"],
                    summary["avg_cards_left"],
                    summary["avg_passes"],
                    summary["avg_tokens"],
                    summary["avg_elapsed_ms"],
                    summary["seats"],
                ]
            )
        sections.append(
            render_table(
                [
                    "matchup",
                    "games",
                    "wins",
                    "win_rate",
                    "avg_score",
                    "avg_cards_left",
                    "avg_passes",
                    "avg_tokens",
                    "avg_elapsed_ms",
                    "seats",
                ],
                matchup_rows,
            )
        )
        sections.append("")

    return "\n".join(sections).rstrip()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    print(analyze(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
