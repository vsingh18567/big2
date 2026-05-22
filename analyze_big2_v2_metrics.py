#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a big2_v2 metrics JSONL and write a small set of matplotlib charts."
    )
    parser.add_argument("jsonl_path", help="Path to a *_metrics.jsonl file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNG charts. Defaults to <jsonl stem>_charts next to the JSONL.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for generated charts. Defaults to the JSONL stem.",
    )
    parser.add_argument("--show", action="store_true", help="Show charts interactively after saving them.")
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                print(f"Skipping non-object JSON value at line {line_number}")
                bad_lines += 1
    return rows, bad_lines


def nested_get(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def format_number(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def format_int(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return str(int(number))


def safe_mean(values: Iterable[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def row_batch(row: dict[str, Any]) -> int | None:
    value = nested_get(row, "batch")
    if isinstance(value, int):
        return value
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def metric_series(rows: list[dict[str, Any]], metric_path: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        batch = row_batch(row)
        value = as_float(nested_get(row, metric_path))
        if batch is None or value is None:
            continue
        xs.append(batch)
        ys.append(value)
    return xs, ys


def eval_win_rate(row: dict[str, Any], opponent: str) -> float | None:
    return (
        as_float(nested_get(row, f"eval.{opponent}.aggregate.win_rate"))
        or as_float(nested_get(row, f"eval.{opponent}.win_rate"))
        or as_float(nested_get(row, f"eval.{opponent}"))
    )


def eval_series(rows: list[dict[str, Any]], opponent: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        batch = row_batch(row)
        value = eval_win_rate(row, opponent)
        if batch is None or value is None:
            continue
        xs.append(batch)
        ys.append(value)
    return xs, ys


def eval_rows(batch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in batch_rows if isinstance(row.get("eval"), dict)]


def eval_score(row: dict[str, Any]) -> float | None:
    greedy = eval_win_rate(row, "greedy")
    smart = eval_win_rate(row, "smart")
    if greedy is None and smart is None:
        return None
    return (greedy or 0.0) + (smart or 0.0)


def best_eval(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [(eval_score(row), row) for row in rows]
    scored = [(score, row) for score, row in scored if score is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


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


def setup_matplotlib(show: bool) -> None:
    if not show:
        matplotlib.use("Agg")


def save_line_chart(
    *,
    title: str,
    ylabel: str,
    rows: list[dict[str, Any]],
    series: list[tuple[str, str]],
    out_path: Path,
    horizontal_at: float | None = None,
) -> bool:
    import matplotlib.pyplot as plt

    plotted = False
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, metric_path in series:
        xs, ys = metric_series(rows, metric_path)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o" if len(xs) < 40 else None, linewidth=1.8, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    if horizontal_at is not None:
        ax.axhline(horizontal_at, color="0.55", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Batch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_eval_chart(rows: list[dict[str, Any]], out_path: Path) -> bool:
    import matplotlib.pyplot as plt

    if not rows:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for opponent in ("random", "greedy", "smart"):
        xs, ys = eval_series(rows, opponent)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=opponent)
        plotted = True

    combined_xs: list[int] = []
    combined_ys: list[float] = []
    for row in rows:
        batch = row_batch(row)
        score = eval_score(row)
        if batch is None or score is None:
            continue
        combined_xs.append(batch)
        combined_ys.append(score)
    if combined_xs:
        ax.plot(combined_xs, combined_ys, marker="o", linewidth=2.2, label="greedy+smart")
        plotted = True

    best = best_eval(rows)
    if best is not None:
        batch = row_batch(best)
        score = eval_score(best)
        if batch is not None and score is not None:
            ax.scatter([batch], [score], s=80, color="black", zorder=5, label="best greedy+smart")

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title("Evaluation win rates")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Win rate / combined score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def save_throughput_chart(rows: list[dict[str, Any]], out_path: Path) -> bool:
    import matplotlib.pyplot as plt

    xs: list[int] = []
    samples_per_second: list[float] = []
    batch_seconds: list[float] = []
    for row in rows:
        batch = row_batch(row)
        samples = as_float(row.get("samples"))
        seconds = as_float(nested_get(row, "timing.batch_seconds"))
        if batch is None or samples is None or seconds is None or seconds <= 0:
            continue
        xs.append(batch)
        samples_per_second.append(samples / seconds)
        batch_seconds.append(seconds)

    if not xs:
        return False

    fig, left = plt.subplots(figsize=(11, 6))
    right = left.twinx()
    line_a = left.plot(xs, samples_per_second, color="tab:blue", linewidth=1.8, label="samples/sec")
    line_b = right.plot(xs, batch_seconds, color="tab:orange", linewidth=1.5, label="batch seconds")

    left.set_title("Throughput and batch time")
    left.set_xlabel("Batch")
    left.set_ylabel("Samples/sec")
    right.set_ylabel("Batch seconds")
    left.grid(True, alpha=0.25)
    lines = line_a + line_b
    left.legend(lines, [line.get_label() for line in lines])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def write_charts(batch_rows: list[dict[str, Any]], out_dir: Path, prefix: str, show: bool) -> list[Path]:
    setup_matplotlib(show)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    eval_only = eval_rows(batch_rows)

    chart_specs = [
        (
            out_dir / f"{prefix}_eval.png",
            lambda path: save_eval_chart(eval_only, path),
        ),
        (
            out_dir / f"{prefix}_ppo_health.png",
            lambda path: save_line_chart(
                title="PPO health metrics",
                ylabel="Metric value",
                rows=batch_rows,
                series=[
                    ("entropy", "entropy"),
                    ("approx_kl", "ppo.approx_kl"),
                    ("clip_fraction", "ppo.clip_fraction"),
                    ("value_explained_variance", "ppo.value_explained_variance"),
                ],
                out_path=path,
                horizontal_at=0.0,
            ),
        ),
        (
            out_dir / f"{prefix}_action_probability.png",
            lambda path: save_line_chart(
                title="Sampled action probabilities",
                ylabel="Probability",
                rows=batch_rows,
                series=[
                    ("p10", "ppo.action_probability_p10"),
                    ("mean", "ppo.action_probability_mean"),
                    ("p90", "ppo.action_probability_p90"),
                ],
                out_path=path,
            ),
        ),
        (
            out_dir / f"{prefix}_candidates.png",
            lambda path: save_line_chart(
                title="Candidate generation",
                ylabel="Candidates / truncated lists",
                rows=batch_rows,
                series=[
                    ("candidate_count_mean", "candidate_count_mean"),
                    ("candidate_count_max", "candidate_count_max"),
                    ("truncated_candidate_lists", "truncated_candidate_lists"),
                ],
                out_path=path,
            ),
        ),
        (
            out_dir / f"{prefix}_throughput.png",
            lambda path: save_throughput_chart(batch_rows, path),
        ),
    ]

    for path, writer in chart_specs:
        if writer(path):
            written.append(path)

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    return written


def summarize(path: Path, rows: list[dict[str, Any]], bad_lines: int, chart_paths: list[Path]) -> str:
    config_rows = [row for row in rows if row.get("event") == "config"]
    batch_rows = [row for row in rows if row.get("event") == "batch"]
    eval_only = eval_rows(batch_rows)

    sections: list[str] = [f"Metrics analysis: {path}"]
    sections.append(f"Rows: {len(rows)} parsed, {len(batch_rows)} batch rows, {len(eval_only)} eval rows")
    if bad_lines:
        sections.append(f"Skipped malformed/non-object lines: {bad_lines}")

    if config_rows:
        config = config_rows[0].get("config", {})
        if isinstance(config, dict):
            sections.append(
                "Config: "
                f"run_name={config.get('run_name', 'n/a')}, "
                f"seed={config.get('seed', 'n/a')}, "
                f"total_batches={config.get('total_batches', 'n/a')}, "
                f"lr={config.get('learning_rate', config.get('lr', 'n/a'))}, "
                f"checkpoint_opponent_weight={config.get('checkpoint_opponent_weight', 'n/a')}"
            )

    if not batch_rows:
        sections.append("")
        sections.append("No batch rows found.")
        return "\n".join(sections)

    batches = [batch for row in batch_rows if (batch := row_batch(row)) is not None]
    latest = max(batch_rows, key=lambda row: row_batch(row) or -1)
    sections.append(
        "Batch range: "
        f"{min(batches) if batches else 'n/a'} to {max(batches) if batches else 'n/a'}; "
        f"latest samples={format_int(latest.get('samples'))}"
    )

    if eval_only:
        latest_eval = max(eval_only, key=lambda row: row_batch(row) or -1)
        best = best_eval(eval_only)
        eval_table: list[list[str]] = []
        for label, row in (("latest", latest_eval), ("best greedy+smart", best)):
            if row is None:
                continue
            eval_table.append(
                [
                    label,
                    format_int(row_batch(row)),
                    format_number(eval_win_rate(row, "random")),
                    format_number(eval_win_rate(row, "greedy")),
                    format_number(eval_win_rate(row, "smart")),
                    format_number(eval_score(row)),
                    str(row.get("checkpoint_path", "n/a")),
                ]
            )
        sections.append("")
        sections.append("Evaluation")
        sections.append(
            render_table(
                ["row", "batch", "random", "greedy", "smart", "greedy+smart", "checkpoint"],
                eval_table,
            )
        )
    else:
        sections.append("")
        sections.append("Evaluation: no eval rows found.")

    latest_ppo_rows = [
        [
            "latest",
            format_int(row_batch(latest)),
            format_number(latest.get("policy_loss")),
            format_number(latest.get("value_loss")),
            format_number(latest.get("entropy")),
            format_number(nested_get(latest, "ppo.approx_kl"), digits=5),
            format_number(nested_get(latest, "ppo.clip_fraction")),
            format_number(nested_get(latest, "ppo.value_explained_variance")),
            format_number(nested_get(latest, "ppo.action_probability_mean")),
        ]
    ]
    sections.append("")
    sections.append("Latest training health")
    sections.append(
        render_table(
            ["row", "batch", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "value_ev", "act_prob"],
            latest_ppo_rows,
        )
    )

    avg_batch_seconds = safe_mean(as_float(nested_get(row, "timing.batch_seconds")) for row in batch_rows)
    avg_rollout_seconds = safe_mean(as_float(nested_get(row, "timing.rollout_seconds")) for row in batch_rows)
    total_truncated = sum(int(value) for row in batch_rows if (value := as_float(row.get("truncated_candidate_lists"))) is not None)
    max_candidate_count = max(
        [value for row in batch_rows if (value := as_float(row.get("candidate_count_max"))) is not None],
        default=None,
    )
    sections.append("")
    sections.append("Run mechanics")
    sections.append(
        render_table(
            ["avg_batch_sec", "avg_rollout_sec", "total_truncated_lists", "max_candidate_count"],
            [
                [
                    format_number(avg_batch_seconds),
                    format_number(avg_rollout_seconds),
                    str(total_truncated),
                    format_int(max_candidate_count),
                ]
            ],
        )
    )

    if chart_paths:
        sections.append("")
        sections.append("Charts written")
        sections.extend(str(path) for path in chart_paths)
    else:
        sections.append("")
        sections.append("Charts written: none; no plottable series were found.")

    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")

    rows, bad_lines = load_jsonl(jsonl_path)
    batch_rows = [row for row in rows if row.get("event") == "batch"]
    out_dir = args.out_dir or jsonl_path.with_name(f"{jsonl_path.stem}_charts")
    prefix = args.prefix or jsonl_path.stem
    chart_paths = write_charts(batch_rows, out_dir, prefix, args.show)
    print(summarize(jsonl_path, rows, bad_lines, chart_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
