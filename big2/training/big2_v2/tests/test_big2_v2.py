from __future__ import annotations

import math
import random

import torch

from big2.training.big2_v2.checkpoints import load_checkpoint, load_checkpoint_partial, save_checkpoint
from big2.training.big2_v2.config import OpponentMixConfig, Big2V2Config
from big2.training.big2_v2.curriculum import entropy_coef_for_batch, training_controls_for_batch
from big2.training.big2_v2.env_adapter import RustVecEnvAdapter
from big2.training.big2_v2.evaluate import evaluate_policy
from big2.training.big2_v2.model import DYNAMIC_ACTION_FEATURE_DIM, OBS_FREE_LEAD, Big2V2ActorCritic
from big2.training.big2_v2.opponents import greedy_slot, smart_slot
from big2.training.big2_v2.rollout import Big2V2RolloutState, collect_rollout
from big2.training.big2_v2.run import (
    _build_training_row,
    _checkpoint_opponent_paths,
    _load_checkpoint_opponent_pool,
    _opponent_mix_from_args,
    build_policy,
    run_training,
)
from big2.training.big2_v2.update import compute_gae, ppo_update


def make_env(num_envs: int = 2, max_candidates: int = 128) -> RustVecEnvAdapter:
    return RustVecEnvAdapter(num_envs=num_envs, seed=123, max_candidates=max_candidates, device="cpu")


def make_policy(env: RustVecEnvAdapter, obs_dim: int) -> Big2V2ActorCritic:
    return Big2V2ActorCritic(
        obs_dim=obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor("cpu"),
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
    )


def test_linear_entropy_schedule_interpolates_by_batch() -> None:
    config = Big2V2Config(
        batches=101,
        entropy_coef=0.01,
        entropy_schedule="linear",
        entropy_start_coef=0.03,
        entropy_end_coef=0.01,
        entropy_schedule_batches=101,
    )

    assert entropy_coef_for_batch(config, batch_idx=1) == 0.03
    assert math.isclose(entropy_coef_for_batch(config, batch_idx=51), 0.02)
    assert entropy_coef_for_batch(config, batch_idx=101) == 0.01
    assert entropy_coef_for_batch(config, batch_idx=150) == 0.01


def test_smart_greedy_curriculum_selects_phase_from_latest_eval_score() -> None:
    config = Big2V2Config(
        entropy_coef=0.01,
        curriculum="smart_greedy",
        opponent_mix=OpponentMixConfig(
            learner_weight=0.75,
            random_weight=0.10,
            greedy_weight=0.10,
            smart_weight=0.05,
        ),
    )

    base = training_controls_for_batch(config, batch_idx=1, latest_smart_greedy_score=None)
    challenge = training_controls_for_batch(config, batch_idx=90, latest_smart_greedy_score=0.82)
    plateau_breaker = training_controls_for_batch(config, batch_idx=260, latest_smart_greedy_score=0.86)

    assert base.curriculum_phase == "base"
    assert base.opponent_mix.learner_weight == 0.75
    assert base.entropy_coef == 0.01
    assert challenge.curriculum_phase == "challenge"
    assert challenge.opponent_mix.smart_weight == 0.10
    assert challenge.opponent_mix.checkpoint_weight == 0.15
    assert challenge.entropy_coef == 0.015
    assert plateau_breaker.curriculum_phase == "plateau_breaker"
    assert plateau_breaker.opponent_mix.learner_weight == 0.50
    assert plateau_breaker.opponent_mix.checkpoint_weight == 0.25
    assert plateau_breaker.entropy_coef == 0.02


def test_smart_greedy_cli_uses_default_base_mix_when_weights_are_not_explicit() -> None:
    class Args:
        curriculum = "smart-greedy"
        opponent_weights_explicit = False
        learner_weight = 1.0
        random_weight = 0.0
        greedy_weight = 0.0
        smart_weight = 0.0
        checkpoint_opponent_weight = 0.0

    mix = _opponent_mix_from_args(Args())

    assert mix.learner_weight == 0.75
    assert mix.random_weight == 0.10
    assert mix.greedy_weight == 0.10
    assert mix.smart_weight == 0.05
    assert mix.checkpoint_weight == 0.0


def test_smart_greedy_cli_preserves_explicit_base_mix() -> None:
    class Args:
        curriculum = "smart-greedy"
        opponent_weights_explicit = True
        learner_weight = 0.8
        random_weight = 0.05
        greedy_weight = 0.05
        smart_weight = 0.05
        checkpoint_opponent_weight = 0.05

    mix = _opponent_mix_from_args(Args())

    assert mix.learner_weight == 0.8
    assert mix.random_weight == 0.05
    assert mix.greedy_weight == 0.05
    assert mix.smart_weight == 0.05
    assert mix.checkpoint_weight == 0.05


def test_compute_gae_uses_nonterminal_bootstrap_value() -> None:
    rewards = torch.tensor([0.0])
    values = torch.tensor([0.5])
    dones = torch.tensor([0.0])

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.99,
        lam=0.95,
        bootstrap_value=0.75,
    )

    assert torch.allclose(returns, torch.tensor([0.7425]))
    assert torch.allclose(advantages, torch.tensor([0.2425]))


def test_checkpoint_opponent_paths_apply_stride_and_limit(tmp_path) -> None:
    for batch in (25, 50, 60, 75, 100):
        (tmp_path / f"batch_{batch:06d}.pt").write_text("")

    paths = _checkpoint_opponent_paths(
        Big2V2Config(
            checkpoint_opponent_dir=str(tmp_path),
            checkpoint_opponent_stride=25,
            checkpoint_opponent_limit=2,
        )
    )

    assert [path.name for path in paths] == ["batch_000075.pt", "batch_000100.pt"]


def test_checkpoint_opponent_pool_loads_frozen_policies(tmp_path) -> None:
    config = Big2V2Config(
        checkpoint_opponent_dir=str(tmp_path),
        checkpoint_opponent_limit=2,
        checkpoint_opponent_stride=1,
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
    )
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    optimizer_policy = build_policy(env, config, obs_dim=batch.obs_dim)
    optimizer = torch.optim.Adam(optimizer_policy.parameters(), lr=1e-3)
    for checkpoint_batch in (1, 2, 3):
        save_checkpoint(
            checkpoint_dir=tmp_path,
            batch=checkpoint_batch,
            policy=optimizer_policy,
            optimizer=optimizer,
            config=config,
        )

    pool = _load_checkpoint_opponent_pool(env, config, obs_dim=batch.obs_dim)

    assert [path.name for path in pool.paths] == ["batch_000002.pt", "batch_000003.pt"]
    assert len(pool.policies) == 2
    assert all(not param.requires_grad for policy in pool.policies for param in policy.parameters())


def test_partial_checkpoint_load_warm_starts_candidate_context_model(tmp_path) -> None:
    config = Big2V2Config(
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
        candidate_set_context=False,
        dynamic_action_features=False,
    )
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    base_policy = build_policy(env, config, obs_dim=batch.obs_dim)
    optimizer = torch.optim.Adam(base_policy.parameters(), lr=1e-3)
    checkpoint = save_checkpoint(
        checkpoint_dir=tmp_path,
        batch=1,
        policy=base_policy,
        optimizer=optimizer,
        config=config,
    )
    context_policy = build_policy(
        env,
        Big2V2Config(
            obs_hidden=64,
            action_emb_dim=16,
            action_feature_hidden=16,
            action_hidden=32,
            candidate_set_context=True,
        ),
        obs_dim=batch.obs_dim,
    )

    payload = load_checkpoint_partial(path=checkpoint, policy=context_policy)

    assert "state_encoder.0.weight" in payload["partial_load"]["loaded_keys"]
    assert any(key.startswith("candidate_context_projection.") for key in payload["partial_load"]["missing_keys"])


def test_partial_checkpoint_load_warm_starts_dynamic_action_model(tmp_path) -> None:
    base_config = Big2V2Config(
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
        candidate_set_context=True,
        dynamic_action_features=False,
    )
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    base_policy = build_policy(env, base_config, obs_dim=batch.obs_dim)
    optimizer = torch.optim.Adam(base_policy.parameters(), lr=1e-3)
    checkpoint = save_checkpoint(
        checkpoint_dir=tmp_path,
        batch=1,
        policy=base_policy,
        optimizer=optimizer,
        config=base_config,
    )
    dynamic_policy = build_policy(
        env,
        Big2V2Config(
            obs_hidden=64,
            action_emb_dim=16,
            action_feature_hidden=16,
            action_hidden=32,
            candidate_set_context=True,
            dynamic_action_features=True,
        ),
        obs_dim=batch.obs_dim,
    )

    payload = load_checkpoint_partial(path=checkpoint, policy=dynamic_policy)

    source_projection = base_policy.state_dict()["action_projection.0.weight"]
    target_projection = dynamic_policy.state_dict()["action_projection.0.weight"]

    assert "state_encoder.0.weight" in payload["partial_load"]["loaded_keys"]
    assert "candidate_context_projection.0.weight" in payload["partial_load"]["loaded_keys"]
    assert any(key.startswith("candidate_outcome_encoder.") for key in payload["partial_load"]["missing_keys"])
    assert "action_projection.0.weight" in payload["partial_load"]["partial_loaded_keys"]
    assert "action_projection.0.weight" not in payload["partial_load"]["skipped_checkpoint_keys"]
    assert torch.allclose(target_projection[:, : source_projection.shape[1]], source_projection)
    assert torch.count_nonzero(target_projection[:, source_projection.shape[1] :]) == 0


def test_partial_checkpoint_load_warm_starts_extended_dynamic_features(tmp_path) -> None:
    config = Big2V2Config(
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
        candidate_set_context=True,
        dynamic_action_features=True,
    )
    env = make_env(num_envs=2, max_candidates=128)
    batch = env.reset()
    source_policy = build_policy(env, config, obs_dim=batch.obs_dim)
    optimizer = torch.optim.Adam(source_policy.parameters(), lr=1e-3)
    checkpoint = save_checkpoint(
        checkpoint_dir=tmp_path,
        batch=1,
        policy=source_policy,
        optimizer=optimizer,
        config=config,
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    old_width = DYNAMIC_ACTION_FEATURE_DIM - 18
    source_outcome_weight = checkpoint_payload["model_state"]["candidate_outcome_encoder.0.weight"][:, :old_width].clone()
    checkpoint_payload["model_state"]["candidate_outcome_encoder.0.weight"] = source_outcome_weight
    torch.save(checkpoint_payload, checkpoint)

    target_policy = build_policy(env, config, obs_dim=batch.obs_dim)
    payload = load_checkpoint_partial(path=checkpoint, policy=target_policy)
    target_outcome_weight = target_policy.state_dict()["candidate_outcome_encoder.0.weight"]

    assert "candidate_outcome_encoder.0.weight" in payload["partial_load"]["partial_loaded_keys"]
    assert torch.allclose(target_outcome_weight[:, :old_width], source_outcome_weight)
    assert torch.count_nonzero(target_outcome_weight[:, old_width:]) == 0


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


def test_candidate_set_context_ignores_masked_candidate_ids() -> None:
    env = make_env(num_envs=2, max_candidates=64)
    batch = env.reset()
    policy = Big2V2ActorCritic(
        obs_dim=batch.obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor("cpu"),
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
        candidate_set_context=True,
    )

    logits, values = policy(batch.obs, batch.candidate_ids, batch.candidate_mask)
    changed_candidate_ids = batch.candidate_ids.clone()
    changed_candidate_ids[~batch.candidate_mask] = (changed_candidate_ids[~batch.candidate_mask] + 17) % env.num_actions
    changed_logits, changed_values = policy(batch.obs, changed_candidate_ids, batch.candidate_mask)

    assert torch.allclose(logits[batch.candidate_mask], changed_logits[batch.candidate_mask], atol=1e-6)
    assert torch.allclose(values, changed_values, atol=1e-6)
    assert (changed_logits[~batch.candidate_mask] < -1.0e8).all()


def test_policy_action_selection_returns_legal_move_ids() -> None:
    env = make_env(num_envs=2, max_candidates=64)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    selection = policy.act(batch.obs, batch.candidate_ids, batch.candidate_mask)

    for env_idx, slot in enumerate(selection.slots.tolist()):
        assert batch.candidate_mask[env_idx, slot]
        assert selection.move_ids[env_idx].item() == batch.candidate_ids[env_idx, slot].item()


def test_dynamic_action_features_feed_action_scoring() -> None:
    env = make_env(num_envs=2, max_candidates=64)
    batch = env.reset()
    policy = Big2V2ActorCritic(
        obs_dim=batch.obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor("cpu"),
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
        dynamic_action_features=True,
    )

    logits, values = policy(batch.obs, batch.candidate_ids, batch.candidate_mask)

    assert logits.shape == batch.candidate_ids.shape
    assert values.shape == (batch.num_envs,)
    assert torch.isfinite(logits[batch.candidate_mask]).all()
    assert (logits[~batch.candidate_mask] < -1.0e8).all()
    with torch.no_grad():
        altered_obs = batch.obs.clone()
        altered_obs[:, :52] = 0.0
        altered_logits, _ = policy(altered_obs, batch.candidate_ids, batch.candidate_mask)
    assert not torch.allclose(logits[batch.candidate_mask], altered_logits[batch.candidate_mask])


def test_dynamic_action_features_mark_optional_and_forced_passes() -> None:
    env = make_env(num_envs=1, max_candidates=128)
    batch = env.reset()
    policy = Big2V2ActorCritic(
        obs_dim=batch.obs_dim,
        num_actions=env.num_actions,
        move_features=env.metadata.as_tensor("cpu"),
        obs_hidden=64,
        action_emb_dim=16,
        action_feature_hidden=16,
        action_hidden=32,
        dynamic_action_features=True,
    )
    non_pass_id = next(row.move_id for row in env.metadata.rows if not row.is_pass)
    obs = torch.zeros((1, batch.obs_dim))
    obs[:, OBS_FREE_LEAD] = 0.0
    selected_move_features = policy.move_features[torch.tensor([[0, non_pass_id]])]

    optional_features = policy._candidate_outcome_features(
        obs,
        selected_move_features,
        torch.tensor([[True, True]]),
    )
    forced_features = policy._candidate_outcome_features(
        obs,
        selected_move_features,
        torch.tensor([[True, False]]),
    )

    optional_pass_idx = DYNAMIC_ACTION_FEATURE_DIM - 15
    forced_pass_idx = DYNAMIC_ACTION_FEATURE_DIM - 14
    assert optional_features.shape[-1] == DYNAMIC_ACTION_FEATURE_DIM
    assert optional_features[0, 0, optional_pass_idx] == 1.0
    assert optional_features[0, 0, forced_pass_idx] == 0.0
    assert forced_features[0, 0, optional_pass_idx] == 0.0
    assert forced_features[0, 0, forced_pass_idx] == 1.0


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
        config=Big2V2Config(logging_mode="max"),
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
        return original_act(
            obs,
            candidate_ids,
            candidate_mask,
            sample=sample,
        )

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


def test_collect_rollout_single_learner_assignment_is_episode_seat_stable() -> None:
    env = make_env(num_envs=8, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    state = Big2V2RolloutState.create(batch.num_envs)

    buffer, _ = collect_rollout(
        env=env,
        policy=policy,
        steps=8,
        opponent_mix=OpponentMixConfig(learner_weight=0.75, greedy_weight=0.25),
        rng=random.Random(123),
        initial_batch=batch,
        controller_assignment="single_learner",
        rollout_state=state,
    )

    assert all(controllers.count("learner") == 1 for controllers in state.seat_controllers)
    assert all(
        controller in {"learner", "greedy"} for controllers in state.seat_controllers for controller in controllers
    )
    assert set(buffer.controller_counts) <= {"learner", "greedy"}


def test_collect_rollout_single_learner_uniform_uses_episode_opponent_profiles() -> None:
    env = make_env(num_envs=12, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    state = Big2V2RolloutState.create(batch.num_envs)

    collect_rollout(
        env=env,
        policy=policy,
        steps=1,
        opponent_mix=OpponentMixConfig(learner_weight=0.0, greedy_weight=0.5, smart_weight=0.5),
        rng=random.Random(123),
        initial_batch=batch,
        controller_assignment="single_learner_uniform",
        rollout_state=state,
    )

    for controllers in state.seat_controllers:
        assert controllers.count("learner") == 1
        opponents = {controller for controller in controllers if controller != "learner"}
        assert len(opponents) == 1
        assert opponents <= {"greedy", "smart"}


def test_collect_rollout_table_profile_uses_learner_weight_for_self_play_tables() -> None:
    env = make_env(num_envs=16, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    state = Big2V2RolloutState.create(batch.num_envs)

    collect_rollout(
        env=env,
        policy=policy,
        steps=1,
        opponent_mix=OpponentMixConfig(learner_weight=1.0, greedy_weight=0.0, smart_weight=0.0),
        rng=random.Random(123),
        initial_batch=batch,
        controller_assignment="table_profile",
        rollout_state=state,
    )

    assert all(controllers == ["learner", "learner", "learner", "learner"] for controllers in state.seat_controllers)


def test_collect_rollout_table_profile_uses_uniform_target_opponent_tables() -> None:
    env = make_env(num_envs=16, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    state = Big2V2RolloutState.create(batch.num_envs)

    collect_rollout(
        env=env,
        policy=policy,
        steps=1,
        opponent_mix=OpponentMixConfig(learner_weight=0.0, greedy_weight=0.0, smart_weight=1.0),
        rng=random.Random(123),
        initial_batch=batch,
        controller_assignment="table_profile",
        rollout_state=state,
    )

    for controllers in state.seat_controllers:
        assert controllers.count("learner") == 1
        assert controllers.count("smart") == 3


def test_collect_rollout_records_bootstrap_values_for_unfinished_learner_tails() -> None:
    env = make_env(num_envs=8, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)
    state = Big2V2RolloutState.create(batch.num_envs)

    buffer, next_batch = collect_rollout(
        env=env,
        policy=policy,
        steps=8,
        opponent_mix=OpponentMixConfig(learner_weight=1.0),
        rng=random.Random(123),
        initial_batch=batch,
        controller_assignment="episode_seat",
        rollout_state=state,
    )

    current_keys = {
        (env_idx, int(next_batch.current_player[env_idx].item()))
        for env_idx in range(next_batch.num_envs)
        if not bool(next_batch.done[env_idx].item()) and bool(next_batch.candidate_mask[env_idx].any().item())
    }
    assert buffer.bootstrap_values
    assert set(buffer.bootstrap_values) <= current_keys
    assert all(math.isfinite(value) for value in buffer.bootstrap_values.values())


def test_collect_rollout_credits_terminal_rewards_to_latest_learner_records() -> None:
    env = make_env(num_envs=4, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    buffer, _ = collect_rollout(
        env=env,
        policy=policy,
        steps=120,
        opponent_mix=OpponentMixConfig(learner_weight=1.0),
        rng=random.Random(321),
        initial_batch=batch,
    )

    assert buffer.episodes_completed > 0
    terminal_records = [record for record in buffer.records if record.done]
    assert len(terminal_records) == buffer.episodes_completed * 4
    assert all(abs(record.reward) > 0.0 for record in terminal_records)


def test_collect_rollout_supports_binary_terminal_rewards() -> None:
    env = make_env(num_envs=4, max_candidates=128)
    batch = env.reset()
    policy = make_policy(env, batch.obs_dim)

    buffer, _ = collect_rollout(
        env=env,
        policy=policy,
        steps=120,
        opponent_mix=OpponentMixConfig(learner_weight=1.0),
        rng=random.Random(321),
        initial_batch=batch,
        terminal_reward_mode="win_loss",
    )

    terminal_rewards = [record.reward for record in buffer.records if record.done]
    assert terminal_rewards
    assert set(terminal_rewards) <= {-1.0, 1.0}


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
    config = Big2V2Config(
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
        config=Big2V2Config(),
        metrics={"win_rate": 0.5},
    )
    payload = load_checkpoint(path=path, policy=policy, optimizer=optimizer)

    assert path.name == "batch_000003.pt"
    assert payload["batch"] == 3
    assert payload["metrics"]["win_rate"] == 0.5
