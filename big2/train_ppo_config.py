import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import nn

from big2.nn import MLPPolicyConfig, SetPoolPolicyConfig
from big2.train_helpers import CheckpointManager, CurriculumConfig


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Game settings
    n_players: int = 4

    # Training loop
    batches: int = 5000
    episodes_per_batch: int = 64
    mini_batch_size: int | None = 256
    use_batched_collection: bool = True

    # PPO hyperparameters
    ppo_epochs: int = 2
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95

    # Optimizer
    lr: float = 1e-3

    # Loss coefficients
    value_coef: float = 0.5
    entropy_beta: float = 0.05

    # Learning rate schedule
    warmup_fraction: float = 0.1  # 10% warmup
    min_lr_ratio: float = 0.1  # LR won't drop below 10% of initial

    # Entropy schedule
    target_entropy_start: float = 0.8
    target_entropy_end: float = 0.45
    entropy_beta_min: float = 0.05
    entropy_beta_max: float = 0.2

    # Evaluation
    eval_interval: int = 50
    eval_games: int = 500

    # Model architecture
    policy_arch: str = "setpool"

    # Reproducibility
    seed: int = 42
    device: str = "cpu"

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PPOConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


def dump_training_run(
    config: PPOConfig,
    policy: nn.Module,
    loss_history: list[float],
    policy_loss_history: list[float],
    value_loss_history: list[float],
    entropy_history: list[float],
    eval_episodes: list[int],
    win_rates: list[float],
    win_rates_smart: list[float],
    checkpoint_manager: CheckpointManager | None = None,
    output_dir: str = "training_run_dump",
    model_save_path: str | None = None,
    training_curves_path: str | None = None,
) -> str:
    """
    Dump all configs and training results to a directory.

    Args:
        config: PPOConfig used for training
        policy: Trained policy model
        loss_history: List of total loss values per batch
        policy_loss_history: List of policy loss values per batch
        value_loss_history: List of value loss values per batch
        entropy_history: List of entropy values per batch
        eval_episodes: List of batch numbers where evaluation occurred
        win_rates: List of win rates vs greedy at each evaluation
        win_rates_smart: List of win rates vs smart at each evaluation
        checkpoint_manager: Optional CheckpointManager instance with curriculum config
        output_dir: Directory to save dump files
        model_save_path: Path where final model was saved (if any)
        training_curves_path: Path where training curves plot was saved (if any)

    Returns:
        Path to the main dump JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract policy architecture config from the policy instance
    policy_class_name = type(policy).__name__
    if policy_class_name == "MLPPolicy":
        policy_config = asdict(
            MLPPolicyConfig(
                n_players=policy.n_players,  # type: ignore[attr-defined]
                card_vocab=53,  # Default, can't easily extract
                card_emb_dim=policy.card_emb.embedding_dim,  # type: ignore[attr-defined]
                hidden=policy.card_embedding_enc.out_features,  # type: ignore[attr-defined]
                action_hidden=policy.action_enc[0].out_features,  # type: ignore[attr-defined]
                device=str(policy.device),  # type: ignore[attr-defined]
            )
        )
    elif policy_class_name == "SetPoolPolicy":
        policy_config = asdict(
            SetPoolPolicyConfig(
                n_players=policy.n_players,  # type: ignore[attr-defined]
                card_vocab=53,  # Default, can't easily extract
                card_emb_dim=policy.card_emb.embedding_dim,  # type: ignore[attr-defined]
                hidden=policy.hand_enc[0].out_features,  # type: ignore[attr-defined]
                action_hidden=policy.action_enc[0].out_features,  # type: ignore[attr-defined]
                device=str(policy.device),  # type: ignore[attr-defined]
            )
        )
    else:
        # Fallback: use defaults based on policy_arch
        if config.policy_arch.lower() in {"mlp", "mlppolicy"}:
            policy_config = asdict(MLPPolicyConfig(n_players=config.n_players, device=config.device))
        else:
            policy_config = asdict(SetPoolPolicyConfig(n_players=config.n_players, device=config.device))

    # Extract curriculum config from checkpoint manager (or use defaults)
    if checkpoint_manager is not None:
        curriculum_config = asdict(checkpoint_manager.config)
        checkpoint_state = {
            "num_checkpoints": len(checkpoint_manager.checkpoints),
            "checkpoint_steps": [step for step, _ in checkpoint_manager.checkpoints],
            "ema_win_rate_greedy": (
                checkpoint_manager.ema_win_rate_greedy if checkpoint_manager.ema_win_rate_greedy is not None else None
            ),
            "ema_win_rate_smart": (
                checkpoint_manager.ema_win_rate_smart if checkpoint_manager.ema_win_rate_smart is not None else None
            ),
            "mastery_greedy": checkpoint_manager.mastery_greedy,
            "mastery_smart": checkpoint_manager.mastery_smart,
            "current_opponent_mix": asdict(checkpoint_manager.compute_dynamic_opponent_mix()),
        }
    else:
        curriculum_config = asdict(CurriculumConfig())
        checkpoint_state = {
            "num_checkpoints": 0,
            "checkpoint_steps": [],
            "ema_win_rate_greedy": None,
            "ema_win_rate_smart": None,
            "mastery_greedy": 0.0,
            "mastery_smart": 0.0,
            "current_opponent_mix": None,
        }

    # Prepare training results
    training_results = {
        "loss_history": loss_history,
        "policy_loss_history": policy_loss_history,
        "value_loss_history": value_loss_history,
        "entropy_history": entropy_history,
        "eval_episodes": eval_episodes,
        "win_rates_vs_greedy": win_rates,
        "win_rates_vs_smart": win_rates_smart,
        "final_loss": loss_history[-1] if loss_history else None,
        "final_policy_loss": policy_loss_history[-1] if policy_loss_history else None,
        "final_value_loss": value_loss_history[-1] if value_loss_history else None,
        "final_entropy": entropy_history[-1] if entropy_history else None,
        "final_win_rate_vs_greedy": win_rates[-1] if win_rates else None,
        "final_win_rate_vs_smart": win_rates_smart[-1] if win_rates_smart else None,
    }

    # Compile everything into a single dump
    dump_data = {
        "ppo_config": asdict(config),
        "policy_config": policy_config,
        "policy_arch": config.policy_arch,
        "curriculum_config": curriculum_config,
        "checkpoint_manager_state": checkpoint_state,
        "training_results": training_results,
        "model_save_path": model_save_path,
        "training_curves_path": training_curves_path,
    }

    # Save main dump file
    dump_file = output_path / "training_run_dump.json"
    with open(dump_file, "w") as f:
        json.dump(dump_data, f, indent=2)

    # Also save individual config files for easy loading
    config.save(str(output_path / "ppo_config.json"))
    curriculum_config_file = output_path / "curriculum_config.json"
    with open(curriculum_config_file, "w") as f:
        json.dump(curriculum_config, f, indent=2)

    print(f"Training run dump saved to {dump_file}")
    print(f"  - PPO config: {output_path / 'ppo_config.json'}")
    print(f"  - Curriculum config: {curriculum_config_file}")
    print(f"  - Full dump: {dump_file}")

    return str(dump_file)
