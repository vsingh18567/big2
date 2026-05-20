from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch

from big2.training.rust_ppo.config import ControllerAssignmentMode, OpponentMixConfig, TerminalRewardMode
from big2.training.rust_ppo.env_adapter import RustBatch, RustVecEnvAdapter
from big2.training.rust_ppo.model import RustCandidateActorCritic
from big2.training.rust_ppo.opponents import greedy_slot, random_slot, smart_slot


@dataclass
class RustRolloutRecord:
    env_index: int
    player: int
    obs: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    slot: torch.Tensor
    move_id: torch.Tensor
    old_logprob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


@dataclass
class RustRolloutBuffer:
    records: list[RustRolloutRecord] = field(default_factory=list)
    candidate_count_total: int = 0
    candidate_count_rows: int = 0
    candidate_count_max: int = 0
    truncated_candidate_lists: int = 0
    candidate_counts: list[int] = field(default_factory=list)
    controller_counts: Counter[str] = field(default_factory=Counter)
    selected_move_kind_counts: Counter[str] = field(default_factory=Counter)
    pass_actions: int = 0
    episodes_completed: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    terminal_reward_by_seat_total: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    wins_by_seat: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    learner_entropy_by_candidate_bucket: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    bootstrap_values: dict[tuple[int, int], float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def append(self, record: RustRolloutRecord) -> None:
        self.records.append(record)

    @property
    def candidate_count_mean(self) -> float:
        if self.candidate_count_rows == 0:
            return 0.0
        return self.candidate_count_total / self.candidate_count_rows

    def observe_candidate_counts(self, batch: RustBatch) -> None:
        counts = batch.candidate_mask.sum(dim=1)
        active = (~batch.done) & (counts > 0)
        active_counts = counts[active]
        if active_counts.numel() > 0:
            self.candidate_count_total += int(active_counts.sum().item())
            self.candidate_count_rows += int(active_counts.numel())
            self.candidate_count_max = max(self.candidate_count_max, int(active_counts.max().item()))
            self.candidate_counts.extend(int(count.item()) for count in active_counts.detach().cpu())
        self.truncated_candidate_lists += batch.truncated_candidate_lists

    def by_trajectory(self) -> dict[tuple[int, int], list[RustRolloutRecord]]:
        grouped: dict[tuple[int, int], list[RustRolloutRecord]] = defaultdict(list)
        for record in self.records:
            grouped[(record.env_index, record.player)].append(record)
        return grouped

    def observe_action(self, controller: str, move_id: int, metadata) -> None:
        self.controller_counts[controller] += 1
        move = metadata.get(move_id)
        kind = str(move.kind)
        self.selected_move_kind_counts[kind] += 1
        if move.is_pass:
            self.pass_actions += 1

    def observe_learner_entropy(self, candidate_count: int, entropy: float) -> None:
        self.learner_entropy_by_candidate_bucket[_candidate_count_bucket(candidate_count)].append(entropy)

    def observe_terminal(self, final_rewards: torch.Tensor, episode_length: int) -> None:
        self.episodes_completed += 1
        self.episode_lengths.append(episode_length)
        rewards = final_rewards.detach().cpu().tolist()
        for seat, reward in enumerate(rewards):
            self.terminal_reward_by_seat_total[seat] += float(reward)
        winner = max(range(len(rewards)), key=lambda seat: rewards[seat])
        if rewards[winner] > 0:
            self.wins_by_seat[winner] += 1


@dataclass
class RustRolloutState:
    """State that must survive across fixed-length rollout batches."""

    episode_step_counts: list[int]
    seat_controllers: list[list[str | None]]

    @classmethod
    def create(cls, num_envs: int) -> RustRolloutState:
        return cls(
            episode_step_counts=[0 for _ in range(num_envs)],
            seat_controllers=[[None for _ in range(4)] for _ in range(num_envs)],
        )


def _candidate_count_bucket(count: int) -> str:
    if count <= 2:
        return "01_02"
    if count <= 5:
        return "03_05"
    if count <= 10:
        return "06_10"
    if count <= 25:
        return "11_25"
    if count <= 50:
        return "26_50"
    return "51_plus"


def _sample_controller(
    *,
    opponent_mix: OpponentMixConfig,
    rng: random.Random,
    checkpoint_available: bool,
    include_learner: bool,
) -> str:
    names, weights = opponent_mix.normalized()
    filtered = [
        (name, weight)
        for name, weight in zip(names, weights, strict=True)
        if (include_learner or name != "learner") and (checkpoint_available or name != "checkpoint")
    ]
    if not filtered:
        return "learner"
    total = sum(weight for _name, weight in filtered)
    if total <= 0:
        return "learner"
    controller_names = [name for name, _weight in filtered]
    controller_weights = [weight / total for _name, weight in filtered]
    return rng.choices(controller_names, weights=controller_weights, k=1)[0]


def _assign_episode_controllers(
    *,
    state: RustRolloutState,
    env_idx: int,
    opponent_mix: OpponentMixConfig,
    controller_assignment: ControllerAssignmentMode,
    rng: random.Random,
    checkpoint_available: bool,
) -> None:
    if controller_assignment == "turn":
        return

    if controller_assignment == "table_profile":
        profile = _sample_controller(
            opponent_mix=opponent_mix,
            rng=rng,
            checkpoint_available=checkpoint_available,
            include_learner=True,
        )
        if profile == "learner":
            state.seat_controllers[env_idx] = ["learner" for _ in range(4)]
            return
        learner_seat = rng.randrange(4)
        state.seat_controllers[env_idx] = [
            "learner" if player == learner_seat else profile for player in range(4)
        ]
        return

    if controller_assignment in {"single_learner", "single_learner_uniform"}:
        opponent_total = (
            opponent_mix.random_weight
            + opponent_mix.greedy_weight
            + opponent_mix.smart_weight
            + (opponent_mix.checkpoint_weight if checkpoint_available else 0.0)
        )
        if opponent_total <= 0:
            state.seat_controllers[env_idx] = ["learner" for _ in range(4)]
            return

        learner_seat = rng.randrange(4)
        uniform_opponent = (
            _sample_controller(
                opponent_mix=opponent_mix,
                rng=rng,
                checkpoint_available=checkpoint_available,
                include_learner=False,
            )
            if controller_assignment == "single_learner_uniform"
            else None
        )
        controllers = []
        for player in range(4):
            if player == learner_seat:
                controllers.append("learner")
            else:
                controllers.append(
                    uniform_opponent
                    if uniform_opponent is not None
                    else
                    _sample_controller(
                        opponent_mix=opponent_mix,
                        rng=rng,
                        checkpoint_available=checkpoint_available,
                        include_learner=False,
                    )
                )
        state.seat_controllers[env_idx] = controllers
        return

    state.seat_controllers[env_idx] = [
        _sample_controller(
            opponent_mix=opponent_mix,
            rng=rng,
            checkpoint_available=checkpoint_available,
            include_learner=True,
        )
        for _player in range(4)
    ]


def _controller_for_turn(
    *,
    state: RustRolloutState,
    env_idx: int,
    player: int,
    opponent_mix: OpponentMixConfig,
    controller_assignment: ControllerAssignmentMode,
    rng: random.Random,
    checkpoint_available: bool,
) -> str:
    if controller_assignment == "turn":
        return _sample_controller(
            opponent_mix=opponent_mix,
            rng=rng,
            checkpoint_available=checkpoint_available,
            include_learner=True,
        )

    controller = state.seat_controllers[env_idx][player]
    if controller is None:
        _assign_episode_controllers(
            state=state,
            env_idx=env_idx,
            opponent_mix=opponent_mix,
            controller_assignment=controller_assignment,
            rng=rng,
            checkpoint_available=checkpoint_available,
        )
        controller = state.seat_controllers[env_idx][player]
    return controller or "learner"


def collect_rollout(
    *,
    env: RustVecEnvAdapter,
    policy: RustCandidateActorCritic,
    steps: int,
    opponent_mix: OpponentMixConfig,
    checkpoint_policies: list[RustCandidateActorCritic] | None = None,
    rng: random.Random | None = None,
    initial_batch: RustBatch | None = None,
    step_penalty: float = 0.0,
    terminal_reward_mode: TerminalRewardMode = "card_fraction",
    controller_assignment: ControllerAssignmentMode = "turn",
    rollout_state: RustRolloutState | None = None,
    episode_step_counts: list[int] | None = None,
) -> tuple[RustRolloutBuffer, RustBatch]:
    """Collect learner-controlled PPO records from a Rust vectorized env."""

    if rng is None:
        rng = random.Random()
    checkpoint_policies = checkpoint_policies or []
    batch = initial_batch if initial_batch is not None else env.reset()
    buffer = RustRolloutBuffer()
    if rollout_state is None:
        rollout_state = RustRolloutState.create(batch.num_envs)
        if episode_step_counts is not None:
            rollout_state.episode_step_counts = episode_step_counts
    if episode_step_counts is not None and len(episode_step_counts) != batch.num_envs:
        raise ValueError("episode_step_counts length must match env count")
    if len(rollout_state.episode_step_counts) != batch.num_envs:
        raise ValueError("rollout_state episode_step_counts length must match env count")
    if len(rollout_state.seat_controllers) != batch.num_envs:
        raise ValueError("rollout_state seat_controllers length must match env count")

    checkpoint_available = bool(checkpoint_policies)
    for env_idx in range(batch.num_envs):
        if controller_assignment != "turn" and all(
            controller is None for controller in rollout_state.seat_controllers[env_idx]
        ):
            _assign_episode_controllers(
                state=rollout_state,
                env_idx=env_idx,
                opponent_mix=opponent_mix,
                controller_assignment=controller_assignment,
                rng=rng,
                checkpoint_available=checkpoint_available,
            )

    policy.eval()
    latest_learner_records: dict[tuple[int, int], RustRolloutRecord] = {}
    for _ in range(steps):
        buffer.observe_candidate_counts(batch)
        action_ids = torch.zeros(batch.num_envs, dtype=torch.long, device=batch.obs.device)
        pending_records: dict[int, RustRolloutRecord] = {}
        learner_indices: list[int] = []
        checkpoint_indices: dict[int, list[int]] = defaultdict(list)

        valid_rows = batch.candidate_mask.any(dim=1)
        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                action_ids[env_idx] = 0
                continue

            player = int(batch.current_player[env_idx].item())
            controller = _controller_for_turn(
                state=rollout_state,
                env_idx=env_idx,
                player=player,
                opponent_mix=opponent_mix,
                controller_assignment=controller_assignment,
                rng=rng,
                checkpoint_available=checkpoint_available,
            )

            if controller == "learner":
                learner_indices.append(env_idx)
            elif controller == "checkpoint":
                checkpoint_idx = rng.randrange(len(checkpoint_policies))
                checkpoint_indices[checkpoint_idx].append(env_idx)
            elif controller == "random":
                slot = random_slot(batch.candidate_mask[env_idx], rng)
                action_ids[env_idx] = batch.candidate_ids[env_idx, slot]
                buffer.observe_action("random", int(action_ids[env_idx].item()), env.metadata)
            elif controller == "greedy":
                slot = greedy_slot(batch.candidate_ids[env_idx], batch.candidate_mask[env_idx], env.metadata)
                action_ids[env_idx] = batch.candidate_ids[env_idx, slot]
                buffer.observe_action("greedy", int(action_ids[env_idx].item()), env.metadata)
            elif controller == "smart":
                slot = smart_slot(
                    batch.obs[env_idx],
                    batch.candidate_ids[env_idx],
                    batch.candidate_mask[env_idx],
                    env.metadata,
                )
                action_ids[env_idx] = batch.candidate_ids[env_idx, slot]
                buffer.observe_action("smart", int(action_ids[env_idx].item()), env.metadata)
            else:
                raise ValueError(f"Unknown controller: {controller}")

        if learner_indices:
            idx = torch.tensor(learner_indices, dtype=torch.long, device=batch.obs.device)
            selection = policy.act(
                batch.obs[idx],
                batch.candidate_ids[idx],
                batch.candidate_mask[idx],
                sample=True,
            )
            action_ids[idx] = selection.move_ids
            for local_idx, env_idx in enumerate(learner_indices):
                pending_records[env_idx] = RustRolloutRecord(
                    env_index=env_idx,
                    player=int(batch.current_player[env_idx].item()),
                    obs=batch.obs[env_idx].detach().cpu(),
                    candidate_ids=batch.candidate_ids[env_idx].detach().cpu(),
                    candidate_mask=batch.candidate_mask[env_idx].detach().cpu(),
                    slot=selection.slots[local_idx].detach().cpu(),
                    move_id=selection.move_ids[local_idx].detach().cpu(),
                    old_logprob=selection.logprobs[local_idx].detach().cpu(),
                    value=selection.values[local_idx].detach().cpu(),
                    reward=step_penalty,
                    done=False,
                )
                buffer.observe_action("learner", int(selection.move_ids[local_idx].item()), env.metadata)
                candidate_count = int(batch.candidate_mask[env_idx].sum().item())
                buffer.observe_learner_entropy(candidate_count, float(selection.entropy[local_idx].item()))

        for checkpoint_idx, env_indices in checkpoint_indices.items():
            idx = torch.tensor(env_indices, dtype=torch.long, device=batch.obs.device)
            selection = checkpoint_policies[checkpoint_idx].act(
                batch.obs[idx],
                batch.candidate_ids[idx],
                batch.candidate_mask[idx],
                sample=True,
            )
            action_ids[idx] = selection.move_ids
            for local_idx in range(len(env_indices)):
                buffer.observe_action("checkpoint", int(selection.move_ids[local_idx].item()), env.metadata)

        next_batch = env.step(action_ids)
        done_indices = torch.nonzero(next_batch.done, as_tuple=False).flatten().detach().cpu().tolist()
        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                continue
            rollout_state.episode_step_counts[env_idx] += 1
        for record in pending_records.values():
            player = record.player
            reward = record.reward
            buffer.append(
                RustRolloutRecord(
                    env_index=record.env_index,
                    player=record.player,
                    obs=record.obs,
                    candidate_ids=record.candidate_ids,
                    candidate_mask=record.candidate_mask,
                    slot=record.slot,
                    move_id=record.move_id,
                    old_logprob=record.old_logprob,
                    value=record.value,
                    reward=reward,
                    done=False,
                )
            )
            latest_learner_records[(record.env_index, record.player)] = buffer.records[-1]

        if done_indices:
            for env_idx in done_indices:
                final_rewards = _terminal_rewards(next_batch.final_rewards[env_idx], terminal_reward_mode)
                buffer.observe_terminal(final_rewards, rollout_state.episode_step_counts[env_idx])
                for player in range(4):
                    latest_record = latest_learner_records.pop((env_idx, player), None)
                    if latest_record is not None:
                        latest_record.reward += float(final_rewards[player].item())
                        latest_record.done = True
                rollout_state.episode_step_counts[env_idx] = 0
                rollout_state.seat_controllers[env_idx] = [None for _ in range(4)]
            next_batch = env.reset_done(done_indices)
            for env_idx in done_indices:
                _assign_episode_controllers(
                    state=rollout_state,
                    env_idx=env_idx,
                    opponent_mix=opponent_mix,
                    controller_assignment=controller_assignment,
                    rng=rng,
                    checkpoint_available=checkpoint_available,
                )
        batch = next_batch

    _observe_bootstrap_values(
        buffer=buffer,
        batch=batch,
        policy=policy,
        state=rollout_state,
        opponent_mix=opponent_mix,
        controller_assignment=controller_assignment,
        rng=rng,
        checkpoint_available=checkpoint_available,
    )

    return buffer, batch


def _terminal_rewards(final_rewards: torch.Tensor, mode: TerminalRewardMode) -> torch.Tensor:
    if mode == "card_fraction":
        return final_rewards
    if mode != "win_loss":
        raise ValueError(f"Unsupported terminal_reward_mode: {mode}")
    winner = int(final_rewards.argmax().item())
    rewards = torch.full_like(final_rewards, -1.0)
    if float(final_rewards[winner].item()) > 0.0:
        rewards[winner] = 1.0
    return rewards


@torch.no_grad()
def _observe_bootstrap_values(
    *,
    buffer: RustRolloutBuffer,
    batch: RustBatch,
    policy: RustCandidateActorCritic,
    state: RustRolloutState,
    opponent_mix: OpponentMixConfig,
    controller_assignment: ControllerAssignmentMode,
    rng: random.Random,
    checkpoint_available: bool,
) -> None:
    if not buffer.records or controller_assignment == "turn":
        return
    trajectories = buffer.by_trajectory()
    active_rows: list[int] = []
    keys: list[tuple[int, int]] = []
    valid_rows = batch.candidate_mask.any(dim=1)
    for env_idx in range(batch.num_envs):
        if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
            continue
        player = int(batch.current_player[env_idx].item())
        key = (env_idx, player)
        if key not in trajectories:
            continue
        controller = _controller_for_turn(
            state=state,
            env_idx=env_idx,
            player=player,
            opponent_mix=opponent_mix,
            controller_assignment=controller_assignment,
            rng=rng,
            checkpoint_available=checkpoint_available,
        )
        if controller != "learner":
            continue
        active_rows.append(env_idx)
        keys.append(key)

    if not active_rows:
        return
    idx = torch.tensor(active_rows, dtype=torch.long, device=batch.obs.device)
    _logits, values = policy(
        batch.obs[idx],
        batch.candidate_ids[idx],
        batch.candidate_mask[idx],
    )
    for key, value in zip(keys, values.detach().cpu().tolist(), strict=True):
        buffer.bootstrap_values[key] = float(value)
