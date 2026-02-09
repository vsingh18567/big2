"""Real-time training monitor for PPO training.

Writes training stats to a JSON file that can be read by a web dashboard.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingStats:
    """Current training statistics."""

    # Progress
    current_batch: int = 0
    total_batches: int = 0
    start_time: float = 0.0
    elapsed_seconds: float = 0.0

    # Loss metrics (per batch)
    loss_history: list[float] = field(default_factory=list)
    policy_loss_history: list[float] = field(default_factory=list)
    value_loss_history: list[float] = field(default_factory=list)
    entropy_history: list[float] = field(default_factory=list)

    # Evaluation metrics (at eval intervals)
    eval_episodes: list[int] = field(default_factory=list)
    win_rates_greedy: list[float] = field(default_factory=list)
    win_rates_smart: list[float] = field(default_factory=list)
    win_rates_random: list[float] = field(default_factory=list)
    avg_cards_remaining: list[float] = field(default_factory=list)
    avg_scores_greedy: list[float] = field(default_factory=list)
    avg_scores_random: list[float] = field(default_factory=list)
    avg_scores_smart: list[float] = field(default_factory=list)

    # Current training state
    current_lr: float = 0.0
    current_entropy_beta: float = 0.0
    steps_per_episode: float = 0.0

    # Curriculum/opponent info
    num_checkpoints: int = 0
    ema_win_rate_greedy: float | None = None
    ema_win_rate_smart: float | None = None
    mastery_greedy: float = 0.0
    mastery_smart: float = 0.0
    opponent_mix: dict[str, float] = field(default_factory=dict)

    # Config summary
    config_summary: dict = field(default_factory=dict)

    # Status
    status: str = "idle"  # idle, running, paused, completed, error
    status_message: str = ""


class TrainingMonitor:
    """Monitor that writes training stats to a JSON file for real-time visualization."""

    def __init__(
        self,
        stats_file: str = "training_stats.json",
        write_interval: float = 1.0,
    ):
        """
        Initialize the training monitor.

        Args:
            stats_file: Path to the JSON file to write stats to
            write_interval: Minimum seconds between file writes
        """
        self.stats_file = Path(stats_file)
        self.write_interval = write_interval
        self.stats = TrainingStats()
        self._lock = threading.Lock()
        self._last_write_time = 0.0
        self._dirty = False

    def start_training(self, total_batches: int, config_summary: dict | None = None):
        """Called when training starts."""
        with self._lock:
            self.stats = TrainingStats(
                current_batch=0,
                total_batches=total_batches,
                start_time=time.time(),
                status="running",
                status_message="Training started",
                config_summary=config_summary or {},
            )
            self._dirty = True
        self._write_stats(force=True)

    def update_batch(
        self,
        batch: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        total_loss: float,
        lr: float | None = None,
        entropy_beta: float | None = None,
        steps_per_episode: float | None = None,
    ):
        """Called after each training batch."""
        with self._lock:
            self.stats.current_batch = batch
            self.stats.elapsed_seconds = time.time() - self.stats.start_time
            self.stats.loss_history.append(total_loss)
            self.stats.policy_loss_history.append(policy_loss)
            self.stats.value_loss_history.append(value_loss)
            self.stats.entropy_history.append(entropy)

            if lr is not None:
                self.stats.current_lr = lr
            if entropy_beta is not None:
                self.stats.current_entropy_beta = entropy_beta
            if steps_per_episode is not None:
                self.stats.steps_per_episode = steps_per_episode

            self._dirty = True

        self._write_stats()

    def update_evaluation(
        self,
        batch: int,
        win_rate_greedy: float,
        win_rate_smart: float,
        win_rate_random: float | None = None,
        avg_cards_remaining: float | None = None,
        avg_score_greedy: float | None = None,
        avg_score_random: float | None = None,
        avg_score_smart: float | None = None,
    ):
        """Called after each evaluation."""
        with self._lock:
            self.stats.eval_episodes.append(batch)
            self.stats.win_rates_greedy.append(win_rate_greedy)
            self.stats.win_rates_smart.append(win_rate_smart)
            if win_rate_random is not None:
                self.stats.win_rates_random.append(win_rate_random)
            if avg_cards_remaining is not None:
                self.stats.avg_cards_remaining.append(avg_cards_remaining)
            if avg_score_greedy is not None:
                self.stats.avg_scores_greedy.append(avg_score_greedy)
            if avg_score_random is not None:
                self.stats.avg_scores_random.append(avg_score_random)
            if avg_score_smart is not None:
                self.stats.avg_scores_smart.append(avg_score_smart)
            self._dirty = True

        self._write_stats(force=True)

    def update_curriculum(
        self,
        num_checkpoints: int,
        ema_win_rate_greedy: float | None,
        ema_win_rate_smart: float | None,
        mastery_greedy: float,
        mastery_smart: float,
        opponent_mix: dict[str, float],
    ):
        """Called when curriculum state changes."""
        with self._lock:
            self.stats.num_checkpoints = num_checkpoints
            self.stats.ema_win_rate_greedy = ema_win_rate_greedy
            self.stats.ema_win_rate_smart = ema_win_rate_smart
            self.stats.mastery_greedy = mastery_greedy
            self.stats.mastery_smart = mastery_smart
            self.stats.opponent_mix = opponent_mix
            self._dirty = True

        self._write_stats()

    def set_status(self, status: str, message: str = ""):
        """Update training status."""
        with self._lock:
            self.stats.status = status
            self.stats.status_message = message
            self._dirty = True

        self._write_stats(force=True)

    def finish_training(self, success: bool = True, message: str = ""):
        """Called when training completes."""
        with self._lock:
            self.stats.elapsed_seconds = time.time() - self.stats.start_time
            self.stats.status = "completed" if success else "error"
            self.stats.status_message = message or ("Training completed" if success else "Training failed")
            self._dirty = True

        self._write_stats(force=True)

    def _write_stats(self, force: bool = False):
        """Write stats to file if dirty and enough time has passed."""
        current_time = time.time()

        with self._lock:
            if not self._dirty:
                return

            if not force and (current_time - self._last_write_time) < self.write_interval:
                return

            stats_dict = asdict(self.stats)
            self._dirty = False
            self._last_write_time = current_time

        # Write outside of lock to minimize lock time
        try:
            with open(self.stats_file, "w") as f:
                json.dump(stats_dict, f)
        except Exception as e:
            print(f"Warning: Failed to write training stats: {e}")

    def get_stats(self) -> TrainingStats:
        """Get current stats (thread-safe copy)."""
        with self._lock:
            return TrainingStats(**asdict(self.stats))


# Global monitor instance for easy access
_global_monitor: TrainingMonitor | None = None
