from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LoggingMode = Literal["minimal", "medium", "max"]


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

    ppo_epochs: int = 2
    mini_batch_size: int = 512
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    step_penalty: float = 0.0
    progress_reward_coef: float = 0.0
    pass_penalty: float = 0.0

    opponent_mix: OpponentMixConfig = field(default_factory=OpponentMixConfig)
    device: str = "cpu"

    checkpoint_dir: str = "rust_ppo_checkpoints"
    metrics_path: str = "rust_ppo_metrics.jsonl"
    checkpoint_interval: int = 10
    eval_interval: int = 10
    eval_games: int = 64
    eval_num_envs: int = 16
    eval_policy_seat: int = 0
    eval_all_seats: bool = True
    logging_mode: LoggingMode = "medium"
