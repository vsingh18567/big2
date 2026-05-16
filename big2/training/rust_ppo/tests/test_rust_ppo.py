from __future__ import annotations

import math
import random

import torch

from big2.training.rust_ppo.checkpoints import load_checkpoint, save_checkpoint
from big2.training.rust_ppo.config import OpponentMixConfig, RustPPOConfig
from big2.training.rust_ppo.env_adapter import RustVecEnvAdapter
from big2.training.rust_ppo.evaluate import evaluate_policy
from big2.training.rust_ppo.model import RustCandidateActorCritic
from big2.training.rust_ppo.opponents import greedy_slot, smart_slot
from big2.training.rust_ppo.rollout import collect_rollout
from big2.training.rust_ppo.run import _build_training_row, run_training
from big2.training.rust_ppo.update import ppo_update


def make_env(num_envs: int = 2, max_candidates: int = 128) -> RustVecEnvAdapter:
    return RustVecEnvAdapter(num_envs=num_envs, seed=123, max_candidates=max_candidates, device="cpu")


def make_policy(env: RustVecEnvAdapter, obs_dim: int) -> RustCandidateActorCritic:
    return RustCandidateActorCritic(
        obs_dim=obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor("cpu"),
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
    )


def test_env_adapter_shapes_and_metadata() -> None:
    env = make_env(num_envs=3, max_candidates=64)
    batch = env.reset()

    assert batch.obs.shape == (3, batch.obs_dim)
    assert batch.candidate_ids.shape == (3, 64)
    assert batch.candidate_mask.shape == (3, 64)
    assert batch.current_player.shape == (3,)
    assert batch.done.tolist() == [False, False, False]
    assert env.metadata.num_actions == env.num_actions
    assert env.metadata.get(0).is_pass
    assert env.metadata.features_np.shape[0] == env.num_actions


def test_model_masks_invalid_candidate_slots() -> None:
    env = make_env(num_envs=2, max_candidates=32)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    logits, values = policy(batch.obs, batch.candidate_ids, batch.candidate_mask)

    assert logits.shape == (2, 32)
    assert values.shape == (2,)
    assert torch.isfinite(logits[batch.candidate_mask]).all()
    assert (logits[~batch.candidate_mask] < -1.0e8).all()


def test_policy_action_selection_returns_legal_move_ids() -> None:
    env = make_env(num_envs=2, max_candidates=64)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    selection = policy.act(batch.obs, batch.candidate_ids, batch.candidate_mask)

    for env_idx, slot in enumerate(selection.slots.tolist()):
        assert batch.candidate_mask[env_idx, slot]
        assert selection.move_ids[env_idx].item() == batch.candidate_ids[env_idx, slot].item()


def test_metadata_heuristics_choose_valid_slots() -> None:
    env = make_env(num_envs=1, max_candidates=128)
    batch = env.reset()

    greedy = greedy_slot(batch.candidate_ids[0], batch.candidate_mask[0], env.metadata)
    smart = smart_slot(batch.obs[0], batch.candidate_ids[0], batch.candidate_mask[0], env.metadata)

    assert batch.candidate_mask[0, greedy]
    assert batch.candidate_mask[0, smart]
    assert not env.metadata.get(int(batch.candidate_ids[0, greedy])).is_pass


def test_collect_rollout_and_ppo_update_smoke() -> None:
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    buffer, _ = collect_rollout(
        env=env,
        policy=policy,
        steps=4,
        opponent_mix=OpponentMixConfig(learner_weight=1.0),
        rng=random.Random(123),
        initial_batch=batch,
    )

    assert len(buffer) == 8
    assert buffer.candidate_count_rows == 8
    assert buffer.candidate_count_max > 0
    assert buffer.candidate_count_mean > 0.0
    assert buffer.controller_counts["learner"] == 8
    assert sum(buffer.selected_move_kind_counts.values()) == 8
    assert buffer.truncated_candidate_lists == 0
    before = [param.detach().clone() for param in policy.parameters()]
    stats = ppo_update(
        policy=policy,
        buffer=buffer,
        optimizer=optimizer,
        ppo_epochs=1,
        mini_batch_size=8,
        clip_epsilon=0.2,
        gamma=0.99,
        lam=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    )
    after = list(policy.parameters())

    assert stats.samples == 8
    assert math.isfinite(stats.approx_kl)
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.ratio_max > 0.0
    assert stats.grad_norm >= 0.0
    assert stats.action_probability_mean > 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))

    row = _build_training_row(
        batch_idx=1,
        stats=stats,
        buffer=buffer,
        config=RustPPOConfig(logging_mode="max"),
        rollout_seconds=0.1,
        update_seconds=0.2,
    )
    assert "ppo" in row
    assert "rollout" in row
    assert "candidates" in row
    assert "count_histogram" in row["candidates"]


def test_collect_rollout_batches_learner_policy_calls() -> None:
    env = make_env(num_envs=4, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    calls: list[int] = []
    original_act = policy.act

    def counting_act(obs, candidate_ids, candidate_mask, *, sample=True):
        calls.append(obs.shape[0])
        return original_act(obs, candidate_ids, candidate_mask, sample=sample)

    policy.act = counting_act  # type: ignore[method-assign]

    buffer, _ = collect_rollout(
        env=env,
        policy=policy,
        steps=3,
        opponent_mix=OpponentMixConfig(learner_weight=1.0),
        rng=random.Random(123),
        initial_batch=batch,
    )

    assert len(buffer) == 12
    assert calls == [4, 4, 4]


def test_collect_rollout_supports_heuristic_only_mix() -> None:
    env = make_env(num_envs=4, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    buffer, next_batch = collect_rollout(
        env=env,
        policy=policy,
        steps=120,
        opponent_mix=OpponentMixConfig(learner_weight=0.0, greedy_weight=1.0),
        rng=random.Random(123),
        initial_batch=batch,
    )

    assert len(buffer) == 0
    assert buffer.candidate_count_rows > 4
    assert not next_batch.done.any()
    assert next_batch.candidate_mask.any(dim=1).all()


def test_evaluate_policy_tracks_fixed_opponent_results() -> None:
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    result = evaluate_policy(
        policy=policy,
        opponent="greedy",
        games=2,
        num_envs=2,
        seed=456,
        max_candidates=128,
        policy_seat=0,
        device="cpu",
    )

    assert result.opponent == "greedy"
    assert result.policy_seat == 0
    assert result.games == 2
    assert 0 <= result.wins <= 2
    assert 0.0 <= result.win_rate <= 1.0
    assert result.win_rate_ci95_low <= result.win_rate <= result.win_rate_ci95_high
    assert result.episode_length_mean > 0.0


def test_minimal_logging_only_prints_eval_rows(tmp_path, capsys) -> None:
    config = RustPPOConfig(
        num_envs=1,
        batches=1,
        rollout_steps=1,
        max_candidates=128,
        ppo_epochs=1,
        mini_batch_size=1,
        eval_interval=1,
        eval_games=1,
        eval_num_envs=1,
        eval_all_seats=False,
        checkpoint_interval=0,
        metrics_path=str(tmp_path / "metrics.jsonl"),
        logging_mode="minimal",
    )

    run_training(config)

    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(printed) == 1
    assert '"eval"' in printed[0]
    rows = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 2


def test_checkpoint_round_trip(tmp_path) -> None:
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    path = save_checkpoint(
        checkpoint_dir=tmp_path,
        batch=3,
        policy=policy,
        optimizer=optimizer,
        config=RustPPOConfig(),
        metrics={"win_rate": 0.5},
    )
    payload = load_checkpoint(path=path, policy=policy, optimizer=optimizer)

    assert path.name == "batch_000003.pt"
    assert payload["batch"] == 3
    assert payload["metrics"]["win_rate"] == 0.5
