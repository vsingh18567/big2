from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LoggingMode = Literal["minimal", "medium", "max"]
EntropyScheduleMode = Literal["constant", "linear"]
CurriculumMode = Literal["off", "smart_greedy"]
ControllerAssignmentMode = Literal["turn", "episode_seat", "single_learner", "single_learner_uniform", "table_profile"]
TerminalRewardMode = Literal["card_fraction", "win_loss"]


@dataclass
class OpponentMixConfig:
    """Sampling weights for who controls each non-terminal Rust env turn."""

    learner_weight: float = 1.0
    random_weight: float = 0.0
    greedy_weight: float = 0.0
    smart_weight: float = 0.0
    checkpoint_weight: float = 0.0

    def normalized(self) -> tuple[list[str], list[float]]:
        names = ["learner", "random", "greedy", "smart", "checkpoint"]
        weights = [
            self.learner_weight,
            self.random_weight,
            self.greedy_weight,
            self.smart_weight,
            self.checkpoint_weight,
        ]
        total = sum(weights)
        if total <= 0:
            raise ValueError("Opponent mix weights must sum to a positive value")
        return names, [weight / total for weight in weights]


@dataclass
class RustPPOConfig:
    """Configuration for PPO over the Rust vectorized environment."""

    num_envs: int = 64
    seed: int = 42
    max_candidates: int = 2048
    rollout_steps: int = 128
    batches: int = 1

    obs_hidden: int = 512
    action_emb_dim: int = 128
    action_feature_hidden: int = 128
    action_hidden: int = 256
    candidate_set_context: bool = False
    dynamic_action_features: bool = False

    ppo_epochs: int = 2
    mini_batch_size: int = 512
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_schedule: EntropyScheduleMode = "constant"
    entropy_start_coef: float | None = None
    entropy_end_coef: float | None = None
    entropy_schedule_batches: int = 0
    max_grad_norm: float = 0.5

    step_penalty: float = 0.0
    progress_reward_coef: float = 0.0
    pass_penalty: float = 0.0
    terminal_reward_mode: TerminalRewardMode = "card_fraction"

    opponent_mix: OpponentMixConfig = field(default_factory=OpponentMixConfig)
    controller_assignment: ControllerAssignmentMode = "turn"
    curriculum: CurriculumMode = "off"
    device: str = "cpu"

    checkpoint_dir: str = "rust_ppo_checkpoints"
    init_checkpoint: str | None = None
    metrics_path: str = "rust_ppo_metrics.jsonl"
    checkpoint_interval: int = 10
    checkpoint_opponent_dir: str | None = None
    checkpoint_opponent_limit: int = 4
    checkpoint_opponent_stride: int = 25
    checkpoint_opponent_refresh: bool = True
    eval_interval: int = 10
    eval_games: int = 64
    eval_num_envs: int = 16
    eval_policy_seat: int = 0
    eval_all_seats: bool = True
    logging_mode: LoggingMode = "medium"
