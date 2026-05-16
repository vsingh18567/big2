import csv
import json
import random

import pytest

from big2.play.agents import GreedyAgent, SmartAgent
from big2.play.config_loader import AgentType, LLM_AGENT_TYPES, load_arena_config
from big2.play.benchmark_suite import (
    MATCHUP_ORDER,
    MATCHUP_THREE_GREEDY,
    MATCHUP_THREE_PPO,
    WorkerExecutionError,
    build_benchmark_tasks,
    find_latest_checkpoint_path,
    load_csv_results,
    run_tasks_chunk,
    run_benchmark,
)


def test_find_latest_checkpoint_path_uses_latest_batch_file(tmp_path) -> None:
    (tmp_path / "batch_00100.pt").write_text("")
    (tmp_path / "batch_00300.pt").write_text("")
    (tmp_path / "batch_00200.pt").write_text("")

    latest_path = find_latest_checkpoint_path(tmp_path)

    assert latest_path.endswith("batch_00300.pt")


def test_build_benchmark_tasks_creates_all_matchups(tmp_path) -> None:
    candidate_config = load_arena_config(
        _write_config(
            {
                "entrants": [
                    {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
                    {"id": "kimi", "agent_type": "moonshotai/kimi-k2.5"},
                ]
            },
            tmp_path=tmp_path,
        )
    )

    tasks = build_benchmark_tasks(
        candidate_config,
        games_per_matchup=2,
        latest_checkpoint_path="training_checkpoints/batch_03200.pt",
        seed=5,
    )

    assert len(tasks) == 2 * len(MATCHUP_ORDER) * 2
    assert {task.matchup for task in tasks} == {
        MATCHUP_THREE_GREEDY,
        MATCHUP_THREE_PPO,
    }
    assert all(len(task.seats) == 4 for task in tasks)
    assert all(sum(1 for seat in task.seats if seat.role == "candidate") == 1 for task in tasks)
    first_slot_tasks = [
        task for task in tasks
        if task.matchup == MATCHUP_THREE_GREEDY and task.matchup_game_index == 0
    ]
    first_ppo_slot_tasks = [
        task for task in tasks
        if task.matchup == MATCHUP_THREE_PPO and task.matchup_game_index == 0
    ]
    assert len(first_slot_tasks) == 2
    assert len(first_ppo_slot_tasks) == 2
    assert {task.seat_seed for task in first_slot_tasks} == {5}
    assert {task.deal_seed for task in first_slot_tasks} == {1_000_005}
    assert {task.seat_seed for task in first_ppo_slot_tasks} == {5}
    assert {task.deal_seed for task in first_ppo_slot_tasks} == {1_000_005}


def test_build_benchmark_tasks_shares_setup_across_candidates_and_varies_by_game_index(tmp_path) -> None:
    candidate_config = load_arena_config(
        _write_config(
            {
                "entrants": [
                    {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
                    {"id": "kimi", "agent_type": "moonshotai/kimi-k2.5"},
                ]
            },
            tmp_path=tmp_path,
        )
    )

    tasks = build_benchmark_tasks(
        candidate_config,
        games_per_matchup=2,
        latest_checkpoint_path="training_checkpoints/batch_03200.pt",
        seed=5,
    )

    first_slot_tasks = [
        task for task in tasks
        if task.matchup == MATCHUP_THREE_GREEDY and task.matchup_game_index == 0
    ]
    second_slot_tasks = [
        task for task in tasks
        if task.matchup == MATCHUP_THREE_GREEDY and task.matchup_game_index == 1
    ]
    first_ppo_slot_tasks = [
        task for task in tasks
        if task.matchup == MATCHUP_THREE_PPO and task.matchup_game_index == 0
    ]

    assert len(first_slot_tasks) == 2
    assert len(second_slot_tasks) == 2
    assert len(first_ppo_slot_tasks) == 2
    assert {task.seat_seed for task in first_slot_tasks} == {5}
    assert {task.deal_seed for task in first_slot_tasks} == {1_000_005}
    assert {task.seat_seed for task in second_slot_tasks} == {6}
    assert {task.deal_seed for task in second_slot_tasks} == {1_000_006}
    assert {task.seat_seed for task in first_ppo_slot_tasks} == {5}
    assert {task.deal_seed for task in first_ppo_slot_tasks} == {1_000_005}

    candidate_seats = {next(index for index, seat in enumerate(task.seats) if seat.role == "candidate") for task in first_slot_tasks}
    assert len(candidate_seats) == 1

    ppo_candidate_seats = {next(index for index, seat in enumerate(task.seats) if seat.role == "candidate") for task in first_ppo_slot_tasks}
    assert candidate_seats == ppo_candidate_seats

    opponent_orders = {
        tuple(seat.entrant_id for seat in task.seats if seat.role == "opponent")
        for task in first_slot_tasks
    }
    assert len(opponent_orders) == 1


def test_run_benchmark_rejects_non_llm_candidates(tmp_path) -> None:
    config_path = _write_config(
        {
            "entrants": [
                {"id": "smart", "agent_type": "smart"},
            ]
        },
        tmp_path=tmp_path,
    )
    candidate_config = load_arena_config(config_path)

    with pytest.raises(ValueError, match="only LLM entrants"):
        run_benchmark(candidate_config, games_per_matchup=1, checkpoint_dir=tmp_path, output_dir=tmp_path)


def test_run_benchmark_writes_outputs_and_summary(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "batch_03200.pt").write_text("")
    config_path = _write_config(
        {
            "entrants": [
                {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
            ]
        },
        tmp_path=tmp_path,
    )
    candidate_config = load_arena_config(config_path)

    class FakeLLMAgent(GreedyAgent):
        def __init__(self) -> None:
            self.client = object()

    def fake_load(config):
        if config.agent_type in LLM_AGENT_TYPES:
            return FakeLLMAgent()
        if config.agent_type == AgentType.SMART:
            return SmartAgent()
        return GreedyAgent()

    monkeypatch.setattr("big2.play.benchmark_suite.load", fake_load)

    results, summary, csv_path, json_path = run_benchmark(
        candidate_config,
        games_per_matchup=1,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path,
        seed=11,
        workers=1,
    )

    assert len(results) == 8
    assert csv_path.exists()
    assert json_path.exists()
    assert summary["latest_checkpoint_path"].endswith("batch_03200.pt")
    assert set(summary["overall"]) == {"deepseek"}
    assert set(summary["by_matchup"]["deepseek"]) == set(MATCHUP_ORDER)
    assert summary["overall"]["deepseek"]["games_played"] == 2

    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {
        "global_game_index",
        "candidate_id",
        "matchup",
        "entrant_role",
        "seat",
        "won",
        "score",
        "pass_count",
        "total_tokens",
        "llm_elapsed_ms",
    } <= set(rows[0])

    summary_payload = json.loads(json_path.read_text())
    assert summary_payload["games_per_matchup"] == 1
    assert summary_payload["matchups"] == list(MATCHUP_ORDER)
    assert summary_payload["shared_setup_across_candidates"] is True
    assert summary_payload["completed_games"] == 2
    assert summary_payload["total_games"] == 2


def test_benchmark_slot_uses_same_deal_for_each_candidate(tmp_path) -> None:
    candidate_config = load_arena_config(
        _write_config(
            {
                "entrants": [
                    {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
                    {"id": "kimi", "agent_type": "moonshotai/kimi-k2.5"},
                ]
            },
            tmp_path=tmp_path,
        )
    )

    tasks = build_benchmark_tasks(
        candidate_config,
        games_per_matchup=1,
        latest_checkpoint_path="training_checkpoints/batch_03200.pt",
        seed=11,
    )
    matchup_tasks = [
        task for task in tasks
        if task.matchup == MATCHUP_THREE_GREEDY and task.matchup_game_index == 0
    ]

    assert len(matchup_tasks) == 2
    task_a, task_b = matchup_tasks
    assert task_a.candidate_id != task_b.candidate_id
    assert task_a.global_game_index != task_b.global_game_index
    assert task_a.seat_seed == task_b.seat_seed
    assert task_a.deal_seed == task_b.deal_seed

    random.seed(task_a.deal_seed)
    deck_a = list(range(52))
    random.shuffle(deck_a)
    random.seed(task_b.deal_seed)
    deck_b = list(range(52))
    random.shuffle(deck_b)
    assert deck_a == deck_b


def test_run_benchmark_parallel_matches_serial(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "batch_03200.pt").write_text("")
    config_path = _write_config(
        {
            "entrants": [
                {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
                {"id": "kimi", "agent_type": "moonshotai/kimi-k2.5"},
            ]
        },
        tmp_path=tmp_path,
    )
    candidate_config = load_arena_config(config_path)

    class FakeLLMAgent(GreedyAgent):
        def __init__(self) -> None:
            self.client = object()

    def fake_load(config):
        if config.agent_type in LLM_AGENT_TYPES:
            return FakeLLMAgent()
        if config.agent_type == AgentType.SMART:
            return SmartAgent()
        return GreedyAgent()

    monkeypatch.setattr("big2.play.benchmark_suite.load", fake_load)

    serial_results, serial_summary, _serial_csv, _serial_json = run_benchmark(
        candidate_config,
        games_per_matchup=1,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "serial",
        seed=19,
        workers=1,
    )
    parallel_results, parallel_summary, _parallel_csv, _parallel_json = run_benchmark(
        candidate_config,
        games_per_matchup=1,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "parallel",
        seed=19,
        workers=2,
    )

    assert serial_results == parallel_results
    assert serial_summary["overall"] == parallel_summary["overall"]
    assert serial_summary["by_matchup"] == parallel_summary["by_matchup"]


def test_run_benchmark_resumes_from_existing_csv(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "batch_03200.pt").write_text("")
    config_path = _write_config(
        {
            "entrants": [
                {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
            ]
        },
        tmp_path=tmp_path,
    )
    candidate_config = load_arena_config(config_path)

    class FakeLLMAgent(GreedyAgent):
        def __init__(self) -> None:
            self.client = object()

    def fake_load(config):
        if config.agent_type in LLM_AGENT_TYPES:
            return FakeLLMAgent()
        if config.agent_type == AgentType.SMART:
            return SmartAgent()
        return GreedyAgent()

    monkeypatch.setattr("big2.play.benchmark_suite.load", fake_load)

    partial_results, _partial_summary, csv_path, _json_path = run_benchmark(
        candidate_config,
        games_per_matchup=1,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path,
        seed=29,
        workers=1,
        run_id="resume-test",
    )
    first_game_rows = [result for result in partial_results if result.global_game_index == 0]
    csv_path.unlink()
    from big2.play.benchmark_suite import append_csv_game_results

    append_csv_game_results(csv_path, first_game_rows)

    resumed_results, resumed_summary, resumed_csv_path, _resumed_json_path = run_benchmark(
        candidate_config,
        games_per_matchup=1,
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path,
        seed=29,
        workers=1,
        run_id="resume-test",
        resume=True,
    )

    assert resumed_csv_path == csv_path
    assert len(resumed_results) == 8
    assert resumed_summary["completed_games"] == 2
    assert resumed_summary["total_games"] == 2
    assert {result.global_game_index for result in load_csv_results(csv_path)} == {0, 1}


def test_run_tasks_chunk_wraps_unserializable_worker_errors(monkeypatch, tmp_path) -> None:
    candidate_config = load_arena_config(
        _write_config(
            {
                "entrants": [
                    {"id": "deepseek", "agent_type": "deepseek/deepseek-v3.2"},
                ]
            },
            tmp_path=tmp_path,
        )
    )
    tasks = build_benchmark_tasks(
        candidate_config,
        games_per_matchup=1,
        latest_checkpoint_path="training_checkpoints/batch_03200.pt",
        seed=5,
    )

    class BadWorkerError(Exception):
        pass

    monkeypatch.setattr(
        "big2.play.benchmark_suite.play_benchmark_game",
        lambda _task: (_ for _ in ()).throw(BadWorkerError("boom")),
    )

    with pytest.raises(WorkerExecutionError, match="BadWorkerError: boom"):
        run_tasks_chunk(tasks[:1])

def _write_config(payload: dict, *, tmp_path):
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps(payload))
    return config_path
