import csv
import json

import pytest

from big2.play.agents import LLMPlayAgent
from big2.play.config_loader import AgentType, ArenaEntrantConfig, load_arena_config
from big2.play.elo_arena import (
    DEFAULT_ELO,
    GamePlayerResult,
    WorkerExecutionError,
    compute_final_score,
    run_arena,
    run_games_chunk,
    update_score_elo,
    update_win_elo,
)


def test_load_arena_config_requires_checkpoint_for_trained_ppo_when_loading() -> None:
    entrant = ArenaEntrantConfig(id="ppo", agent_type=AgentType.TRAINED_PPO, checkpoint_path=None)

    with pytest.raises(ValueError, match="checkpoint_path"):
        from big2.play.config_loader import load

        load(entrant)


def test_load_arena_config_reads_json_file(tmp_path) -> None:
    config_path = tmp_path / "arena.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "g1", "agent_type": "greedy"},
                    {"id": "g2", "agent_type": "greedy"},
                    {"id": "g3", "agent_type": "smart"},
                    {"id": "g4", "agent_type": "openai/gpt-5.4-mini"},
                ]
            }
        )
    )

    config = load_arena_config(config_path)

    assert [entrant.id for entrant in config.entrants] == ["g1", "g2", "g3", "g4"]
    assert config.entrants[2].agent_type == AgentType.SMART
    assert config.entrants[3].agent_type == AgentType.OPENAI_GPT_5_4_MINI


def test_load_arena_config_reads_reasoning_enabled_flag(tmp_path) -> None:
    config_path = tmp_path / "arena.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "g1", "agent_type": "openai/gpt-5.4-mini", "reasoning_enabled": False},
                ]
            }
        )
    )

    config = load_arena_config(config_path)
    entrant = config.entrants[0]

    assert entrant.reasoning_enabled is False

    from big2.play.config_loader import load

    agent = load(entrant)
    assert isinstance(agent, LLMPlayAgent)
    assert agent.reasoning_enabled is False


def test_run_arena_rejects_llm_entrants_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = tmp_path / "arena.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "g1", "agent_type": "greedy"},
                    {"id": "g2", "agent_type": "greedy"},
                    {"id": "g3", "agent_type": "smart"},
                    {"id": "g4", "agent_type": "openai/gpt-5.4-mini"},
                ]
            }
        )
    )
    arena_config = load_arena_config(config_path)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        run_arena(arena_config, games=1, output_dir=tmp_path, seed=7)


def test_compute_final_score_uses_negative_score_for_winner() -> None:
    class DummyEnv:
        n_players = 4
        winner = 1
        hands = [[0, 1], [], [2, 3, 4], [5]]

    assert compute_final_score(DummyEnv(), 1) == -(2 + 3 + 1)
    assert compute_final_score(DummyEnv(), 0) == 2


def test_win_elo_only_updates_winner_vs_losers() -> None:
    ratings = {f"p{i}": DEFAULT_ELO for i in range(4)}
    results = [
        GamePlayerResult(0, "p0", "greedy", "", 0, 0, True, 0, -9, 0, 0, 0, 0),
        GamePlayerResult(0, "p1", "greedy", "", 1, 0, False, 2, 2, 0, 0, 0, 0),
        GamePlayerResult(0, "p2", "greedy", "", 2, 0, False, 3, 3, 0, 0, 0, 0),
        GamePlayerResult(0, "p3", "greedy", "", 3, 0, False, 4, 4, 0, 0, 0, 0),
    ]

    update_win_elo(ratings, results, 24.0)

    assert ratings["p0"] > DEFAULT_ELO
    assert ratings["p1"] < DEFAULT_ELO
    assert ratings["p2"] < DEFAULT_ELO
    assert ratings["p3"] < DEFAULT_ELO


def test_score_elo_prefers_lower_score() -> None:
    ratings = {"a": DEFAULT_ELO, "b": DEFAULT_ELO}
    results = [
        GamePlayerResult(0, "a", "greedy", "", 0, 0, True, 0, -7, 0, 0, 0, 0),
        GamePlayerResult(0, "b", "greedy", "", 1, 0, False, 7, 7, 0, 0, 0, 0),
    ]

    update_score_elo(ratings, results, 24.0)

    assert ratings["a"] > DEFAULT_ELO
    assert ratings["b"] < DEFAULT_ELO


def test_run_arena_writes_csv_and_json(tmp_path) -> None:
    config_path = tmp_path / "arena.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "g1", "agent_type": "greedy"},
                    {"id": "g2", "agent_type": "greedy"},
                    {"id": "g3", "agent_type": "greedy"},
                    {"id": "g4", "agent_type": "greedy"},
                ]
            }
        )
    )
    arena_config = load_arena_config(config_path)

    results, summary, csv_path, json_path = run_arena(arena_config, games=2, output_dir=tmp_path, seed=7)

    assert len(results) == 8
    assert csv_path.exists()
    assert json_path.exists()
    assert set(summary["win_elo"]) == {"g1", "g2", "g3", "g4"}
    assert set(summary["score_elo"]) == {"g1", "g2", "g3", "g4"}

    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {"game_index", "entrant_id", "seat", "won", "score", "pass_count", "total_tokens"} <= set(rows[0])

    summary_payload = json.loads(json_path.read_text())
    assert summary_payload["games"] == 2
    assert len(summary_payload["entrants"]) == 4


def test_run_arena_parallel_matches_serial_on_seeded_games(tmp_path) -> None:
    config_path = tmp_path / "arena.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "g1", "agent_type": "greedy"},
                    {"id": "g2", "agent_type": "greedy"},
                    {"id": "g3", "agent_type": "smart"},
                    {"id": "g4", "agent_type": "smart"},
                ]
            }
        )
    )
    arena_config = load_arena_config(config_path)

    serial_results, serial_summary, _serial_csv, _serial_json = run_arena(
        arena_config,
        games=4,
        output_dir=tmp_path / "serial",
        seed=19,
        workers=1,
    )
    parallel_results, parallel_summary, _parallel_csv, _parallel_json = run_arena(
        arena_config,
        games=4,
        output_dir=tmp_path / "parallel",
        seed=19,
        workers=2,
    )

    assert serial_results == parallel_results
    assert serial_summary["win_elo"] == parallel_summary["win_elo"]
    assert serial_summary["score_elo"] == parallel_summary["score_elo"]


def test_run_games_chunk_wraps_worker_errors(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "arena.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": [
                    {"id": "g1", "agent_type": "greedy"},
                    {"id": "g2", "agent_type": "greedy"},
                    {"id": "g3", "agent_type": "greedy"},
                    {"id": "g4", "agent_type": "greedy"},
                ]
            }
        )
    )
    arena_config = load_arena_config(config_path)

    class BadWorkerError(Exception):
        pass

    monkeypatch.setattr(
        "big2.play.elo_arena.play_game_with_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(BadWorkerError("boom")),
    )

    with pytest.raises(WorkerExecutionError, match="BadWorkerError: boom"):
        run_games_chunk(arena_config, [0], seed=7)
