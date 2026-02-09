import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from big2.nn import SetPoolQNetwork, combo_to_action_vector
from big2.simulator.cards import PASS, Combo, card_name
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy


@dataclass
class StepRecord:
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    reward: float
    max_logprob: torch.Tensor | None = None
    action: str | None = None


def _card_count_for_model(model: nn.Module) -> int:
    return int(getattr(model, "card_universe_size", 52))


def safe_categorical_from_logits(logits: torch.Tensor, candidates: list[Combo]) -> torch.distributions.Categorical:
    """
    Create a Categorical distribution from logits with stability checks.

    Args:
        logits: Raw logits tensor
        candidates: List of candidate actions (for error reporting)

    Returns:
        Categorical distribution

    Raises:
        ValueError: If logits contain NaN/Inf or candidates are empty
        RuntimeError: If softmax probabilities don't sum to 1 (within tolerance)
    """
    if len(candidates) == 0:
        raise ValueError("Cannot create distribution from empty candidates list")

    if logits.numel() != len(candidates):
        raise ValueError(f"Logits length ({logits.numel()}) doesn't match candidates length ({len(candidates)})")

    # Check for NaN/Inf
    if torch.isnan(logits).any():
        raise ValueError(f"NaN detected in logits. Candidates: {len(candidates)}")
    if torch.isinf(logits).any():
        raise ValueError(f"Inf detected in logits. Candidates: {len(candidates)}")

    # Stabilize logits: subtract max to prevent overflow
    logits_stable = logits - logits.max()

    # Compute probabilities
    probs = F.softmax(logits_stable, dim=0)

    # Validate probability sum
    prob_sum = probs.sum().item()
    if abs(prob_sum - 1.0) > 1e-5:
        raise RuntimeError(
            f"Probabilities sum to {prob_sum:.8f}, expected 1.0. "
            f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}], "
            f"Candidates: {len(candidates)}"
        )

    return torch.distributions.Categorical(probs=probs)


def _policy_logits_and_value(
    policy: nn.Module, state: np.ndarray, candidates: list[Combo]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single-state policy forward pass and return logits/value."""
    st = torch.from_numpy(state[np.newaxis, :]).long().to(policy.device)  # type: ignore[attr-defined]
    card_count = _card_count_for_model(policy)
    action_feats = [[combo_to_action_vector(c, card_count=card_count) for c in candidates]]
    logits_list, values = policy(st, action_feats)
    return logits_list[0], values[0]


def sample_action_from_policy(
    policy: nn.Module, state: np.ndarray, candidates: list[Combo]
) -> tuple[int, Combo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an action from the policy distribution and return details."""
    logits, value = _policy_logits_and_value(policy, state, candidates)
    dist = safe_categorical_from_logits(logits, candidates)
    idx = dist.sample()
    logprob = dist.log_prob(idx)
    entropy = dist.entropy()
    max_prob = dist.probs.max()
    action_idx = int(idx.item())
    return action_idx, candidates[action_idx], logprob, entropy, value, max_prob


def select_action_sampled(policy: nn.Module, state: np.ndarray, candidates: list[Combo]) -> Combo:
    """Sample an action without tracking gradients (for opponents)."""
    with torch.no_grad():
        action_idx, action, _logprob, _entropy, _value, _max_prob = sample_action_from_policy(policy, state, candidates)
    return action


def select_action(
    policy: nn.Module, state: np.ndarray, candidates: list[Combo]
) -> tuple[Combo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an action and return tensors used for policy-gradient losses."""
    _action_idx, chosen, logprob, entropy, value, max_prob = sample_action_from_policy(policy, state, candidates)
    return chosen, logprob, entropy, value, max_prob


def episode(
    env: Big2Env,
    policy: nn.Module,
    opponent_strategies: dict[int, nn.Module | Callable[[list[Combo]], Combo]] | None = None,
    model_players: set[int] | None = None,
):
    """
    Run one episode with opponent sampling support.

    Args:
        env: Big2Env instance
        policy: Current policy for model players
        opponent_strategies: Dict mapping player_id -> strategy (MLPPolicy, greedy_strategy, or random_strategy)
        model_players: Set of player IDs that use the current policy (for trajectory recording)

    Returns:
        Trajectory dict for model players only
    """
    if opponent_strategies is None:
        opponent_strategies = {}
    if model_players is None:
        model_players = set(range(env.n_players))

    # Records per player (only for model players)
    traj: dict[int, list[StepRecord]] = {p: [] for p in range(env.n_players)}
    state = env.reset()
    while True:
        p = env.current_player
        candidates = env.legal_candidates(p)
        # Safety: if no candidates somehow, force PASS
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        if p in opponent_strategies:
            # Opponent player: use provided strategy
            strategy = opponent_strategies[p]
            if isinstance(strategy, nn.Module):
                # Policy-based opponent (current policy or past checkpoint)
                action = select_action_sampled(strategy, state, candidates)
            elif strategy == smart_strategy:
                # Smart strategy needs hand and trick_pile
                action = strategy(candidates, env.hands[p], env.trick_pile)
            elif strategy == greedy_strategy or strategy == random_strategy:
                # Simple function strategy
                action = strategy(candidates)
            else:
                # Fallback to greedy
                action = greedy_strategy(candidates)
            next_state, done = env.step(action)
        else:
            # Model player: sample action and record trajectory
            action, logprob, entropy, value, max_logprob = select_action(policy, state, candidates)

            if action.type == PASS:
                action_str = "PASS"
            else:
                action_str = " ".join([card_name(c) for c in action.cards])

            next_state, done = env.step(action)
            if not done:
                reward = 0.0
            # Store step
            traj[p].append(
                StepRecord(
                    logprob=logprob,
                    entropy=entropy,
                    value=value,
                    reward=reward,
                    max_logprob=max_logprob,
                    action=action_str,
                )
            )

        # Assign terminal rewards at the end
        if done:
            winner = env.winner
            for q in model_players:
                # Only update rewards for model players
                # If winner: reward is the number of cards in the other players' hands
                # If loser: reward is the negative number of cards in our hand
                final_r = sum(len(env.hands[i]) for i in range(env.n_players) if i != q) if q == winner else 0.0
                # set last step reward for each player (Monte Carlo return will propagate)
                if len(traj[q]) > 0:
                    traj[q][-1] = StepRecord(
                        logprob=traj[q][-1].logprob,
                        entropy=traj[q][-1].entropy,
                        value=traj[q][-1].value,
                        reward=final_r,
                        max_logprob=traj[q][-1].max_logprob,
                        action=traj[q][-1].action,
                    )
            break
        state = next_state
    return traj


def select_action_greedy(policy: nn.Module, state: np.ndarray, candidates: list[Combo]) -> Combo:
    """Select action greedily (no sampling) for evaluation."""
    with torch.no_grad():
        logits, _value = _policy_logits_and_value(policy, state, candidates)
        idx = logits.argmax()
        return candidates[idx]


def random_strategy(candidates: list[Combo]) -> Combo:
    """Select a random legal action."""
    return random.choice(candidates)


def select_action_q_greedy(qnet: SetPoolQNetwork, state: np.ndarray, candidates: list[Combo]) -> Combo:
    """Select action greedily (argmax Q) for evaluation/opponents."""
    with torch.no_grad():
        st = torch.from_numpy(state[np.newaxis, :]).long().to(qnet.device)
        action_feats = [[combo_to_action_vector(c, card_count=qnet.card_universe_size) for c in candidates]]
        q_list = qnet(st, action_feats)
        qvals = q_list[0]
        idx = int(qvals.argmax().item())
        return candidates[idx]


def select_action_q_epsilon_greedy(
    qnet: SetPoolQNetwork, state: np.ndarray, candidates: list[Combo], epsilon: float
) -> tuple[Combo, int]:
    """Epsilon-greedy w.r.t. argmax Q."""
    if len(candidates) == 0:
        candidates = [Combo(PASS, [], ())]
    if random.random() < epsilon:
        idx = random.randrange(len(candidates))
        return candidates[idx], idx
    with torch.no_grad():
        st = torch.from_numpy(state[np.newaxis, :]).long().to(qnet.device)
        action_feats = [[combo_to_action_vector(c, card_count=qnet.card_universe_size) for c in candidates]]
        q_list = qnet(st, action_feats)
        qvals = q_list[0]
        idx = int(qvals.argmax().item())
        return candidates[idx], idx


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    win_rate_vs_greedy: float
    win_rate_vs_random: float
    win_rate_vs_smart: float
    avg_cards_remaining_when_losing: float
    avg_score_vs_greedy: float
    avg_score_vs_random: float
    avg_score_vs_smart: float
    win_rate_by_starting_position: dict[int, float]
    total_games: int


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning mastery thresholds."""

    greedy_mastery_target: float = 0.40  # 40% win rate = mastered greedy
    smart_mastery_target: float = 0.35  # 35% win rate = mastered smart
    greedy_retention_fraction: float = 0.15  # 15% weight for retention of mastered greedy
    smart_retention_fraction: float = 0.25  # 25% weight for retention of mastered smart
    ema_alpha: float = 0.33  # Smoothing factor for EMA
    max_checkpoints: int = 50  # Keep last N checkpoints


@dataclass
class OpponentMix:
    """Defines the distribution of opponent strategies for a training stage."""

    greedy_weight: float
    smart_weight: float
    current_weight: float
    checkpoint_weight: float
    random_weight: float

    def get_weights_and_strategies(self) -> tuple[list[float], list[str]]:
        """Return normalized weights and strategy names in a consistent order.

        Normalizing here prevents sampling failures if the configured weights
        drift from summing to 1. A tiny drift only triggers the normalization
        (not an error); a zero or negative total raises to surface bad configs.
        """
        raw_weights = [
            self.greedy_weight,
            self.smart_weight,
            self.current_weight,
            self.checkpoint_weight,
            self.random_weight,
        ]
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("Opponent mix weights must sum to a positive value")
        weights = [w / total for w in raw_weights]
        return (weights, ["greedy", "smart", "current", "checkpoint", "random"])


def play_evaluation_game(
    current_policy: nn.Module,
    n_players: int,
    cards_per_player: int | None = None,
    opponent_strategy: Callable[[list[Combo]], Combo] = greedy_strategy,
    device="cpu",
) -> tuple[int | None, int, int, int]:
    """
    Play one evaluation game where player 0 uses current_policy and others use opponent_strategy.

    Returns:
        (winner, starting_position, cards_remaining_for_player_0, score_for_player_0)
    """
    env = Big2Env(n_players, cards_per_player=cards_per_player)
    state = env.reset()
    starting_position = env.current_player

    while not env.done:
        p = env.current_player
        candidates = env.legal_candidates(p)
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        # Player 0 uses current policy, others use opponent strategy
        if p == 0:
            action = select_action_greedy(current_policy, state, candidates)
        else:
            # Handle smart_strategy which needs hand and trick_pile
            if opponent_strategy == smart_strategy:
                action = opponent_strategy(candidates, env.hands[p], env.trick_pile)
            else:
                action = opponent_strategy(candidates)

        state, _ = env.step(action)

    if env.winner == 0:
        score = sum(len(env.hands[p]) for p in range(n_players) if p != 0)
        cards_remaining = 0
    else:
        cards_remaining = len(env.hands[0])
        score = -cards_remaining
    return env.winner, starting_position, cards_remaining, score


def evaluate_policy(
    current_policy: nn.Module,
    n_players: int,
    cards_per_player: int | None = None,
    num_games: int = 500,
    device="cpu",
) -> EvaluationMetrics:
    """
    Comprehensive evaluation of current policy.

    Returns:
        EvaluationMetrics with various performance metrics
    """
    wins_vs_greedy = 0
    wins_vs_random = 0
    wins_vs_smart = 0
    cards_remaining_sum = 0
    losses_count = 0
    score_sum_greedy = 0
    score_sum_random = 0
    score_sum_smart = 0
    wins_by_position: dict[int, int] = {}
    games_by_position: dict[int, int] = {}

    # Evaluate against greedy
    for _ in range(num_games):
        winner, starting_pos, cards_remaining, score = play_evaluation_game(
            current_policy,
            n_players,
            cards_per_player=cards_per_player,
            opponent_strategy=greedy_strategy,
            device=device,
        )
        if winner == 0:
            wins_vs_greedy += 1
            wins_by_position[starting_pos] = wins_by_position.get(starting_pos, 0) + 1
        else:
            cards_remaining_sum += cards_remaining
            losses_count += 1
        score_sum_greedy += score
        games_by_position[starting_pos] = games_by_position.get(starting_pos, 0) + 1

    # Evaluate against random
    for _ in range(num_games):
        winner, _, _, score = play_evaluation_game(
            current_policy,
            n_players,
            cards_per_player=cards_per_player,
            opponent_strategy=random_strategy,
            device=device,
        )
        if winner == 0:
            wins_vs_random += 1
        score_sum_random += score

    # Evaluate against smart strategy
    for _ in range(num_games):
        winner, _, _, score = play_evaluation_game(
            current_policy,
            n_players,
            cards_per_player=cards_per_player,
            opponent_strategy=smart_strategy,
            device=device,
        )
        if winner == 0:
            wins_vs_smart += 1
        score_sum_smart += score

    win_rate_vs_greedy = wins_vs_greedy / num_games
    win_rate_vs_random = wins_vs_random / num_games
    win_rate_vs_smart = wins_vs_smart / num_games
    avg_cards_remaining = cards_remaining_sum / losses_count if losses_count > 0 else 0.0
    win_rate_by_position = {pos: wins_by_position.get(pos, 0) / games_by_position[pos] for pos in games_by_position}
    avg_score_vs_greedy = score_sum_greedy / num_games
    avg_score_vs_random = score_sum_random / num_games
    avg_score_vs_smart = score_sum_smart / num_games

    return EvaluationMetrics(
        win_rate_vs_greedy=win_rate_vs_greedy,
        win_rate_vs_random=win_rate_vs_random,
        win_rate_vs_smart=win_rate_vs_smart,
        avg_cards_remaining_when_losing=avg_cards_remaining,
        avg_score_vs_greedy=avg_score_vs_greedy,
        avg_score_vs_random=avg_score_vs_random,
        avg_score_vs_smart=avg_score_vs_smart,
        win_rate_by_starting_position=win_rate_by_position,
        total_games=num_games,
    )


class CheckpointManager:
    """Manages past checkpoints for opponent sampling with mastery-based curriculum learning."""

    def __init__(
        self,
        checkpoint_dir: str = "big2",
        device: str = "cpu",
        n_players: int = 4,
        cards_per_player: int | None = None,
        curriculum_config: CurriculumConfig | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.n_players = n_players
        self.cards_per_player = cards_per_player
        self.checkpoints: list[tuple[int, nn.Module]] = []  # (step, policy)

        # Use provided config or defaults
        self.config = curriculum_config or CurriculumConfig()
        self.max_checkpoints = self.config.max_checkpoints

        # Mastery tracking with EMA (Exponential Moving Average) for stability
        self.ema_alpha = self.config.ema_alpha
        self.ema_win_rate_greedy: float | None = None
        self.ema_win_rate_smart: float | None = None
        self.mastery_greedy = 0.0  # Mastery level for greedy (0.0-1.0)
        self.mastery_smart = 0.0  # Mastery level for smart (0.0-1.0)

    def add_checkpoint(self, step: int, policy: nn.Module):
        """Add a checkpoint to the pool."""
        # Create a copy of the policy (same architecture as the current policy)
        checkpoint_policy = type(policy)(
            n_players=self.n_players,
            cards_per_player=self.cards_per_player,
            device=self.device,
        ).to(self.device)
        checkpoint_policy.load_state_dict(policy.state_dict())
        checkpoint_policy.eval()

        self.checkpoints.append((step, checkpoint_policy))

        # Keep only the most recent checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def update_mastery(self, win_rate_greedy: float, win_rate_smart: float):
        """
        Update mastery levels using EMA (Exponential Moving Average) of win rates.
        This provides stability against noisy evaluations.

        Args:
            win_rate_greedy: Current win rate against greedy opponents
            win_rate_smart: Current win rate against smart opponents
        """
        # Update EMA win rates
        if self.ema_win_rate_greedy is None:
            self.ema_win_rate_greedy = win_rate_greedy
        else:
            self.ema_win_rate_greedy = (
                self.ema_alpha * win_rate_greedy + (1 - self.ema_alpha) * self.ema_win_rate_greedy
            )

        if self.ema_win_rate_smart is None:
            self.ema_win_rate_smart = win_rate_smart
        else:
            self.ema_win_rate_smart = self.ema_alpha * win_rate_smart + (1 - self.ema_alpha) * self.ema_win_rate_smart

        # Compute mastery scores (clipped to [0, 1])
        self.mastery_greedy = min(1.0, max(0.0, self.ema_win_rate_greedy / self.config.greedy_mastery_target))
        self.mastery_smart = min(1.0, max(0.0, self.ema_win_rate_smart / self.config.smart_mastery_target))

    def compute_dynamic_opponent_mix(self) -> OpponentMix:
        """
        Compute opponent mix dynamically based on mastery levels.
        Implements a three-phase curriculum:
        1. Phase 1: Learning to beat greedy (mastery_greedy < 1.0)
        2. Phase 2: Learning to beat smart (mastery_greedy >= 1.0, mastery_smart < 1.0)
        3. Phase 3: Self-play focus (both mastered)

        Retention weights ensure we don't forget previously mastered opponents.
        """
        # Base weights for different phases
        if self.mastery_greedy < 1.0:
            # Phase 1: Focus on learning greedy
            greedy_weight = 0.50 + 0.10 * (1.0 - self.mastery_greedy)  # 50-60% when learning
            smart_weight = 0.15 + 0.05 * self.mastery_greedy  # 15-20% gradually increasing
            current_weight = 0.1 + 0.05 * self.mastery_greedy  # 10-15% gradually increasing
            checkpoint_weight = 0.05 + 0.05 * self.mastery_greedy  # 5-10% gradually increasing
            random_weight = 0.10 - 0.05 * self.mastery_greedy  # 10-5% gradually decreasing
        elif self.mastery_smart < 1.0:
            # Phase 2: Focus on learning smart, retain greedy
            greedy_weight = self.config.greedy_retention_fraction
            smart_weight = 0.4 + 0.05 * (1.0 - self.mastery_smart)  # 40-45% when learning
            current_weight = 0.20 + 0.05 * self.mastery_smart  # 20-25% gradually increasing
            checkpoint_weight = 0.20 + 0.05 * self.mastery_smart  # 15-20% gradually increasing
            random_weight = 0.05
        else:
            # Phase 3: Both mastered, focus on self-play
            greedy_weight = self.config.greedy_retention_fraction
            smart_weight = self.config.smart_retention_fraction
            random_weight = 0.05

            # Remaining weight for self-play (current + checkpoint)
            remaining_weight = 1.0 - greedy_weight - smart_weight - random_weight
            current_weight = remaining_weight * 0.4  # 50% of remaining to current policy
            checkpoint_weight = remaining_weight * 0.6  # 50% of remaining to checkpoints

        return OpponentMix(
            greedy_weight=greedy_weight,
            smart_weight=smart_weight,
            current_weight=current_weight,
            checkpoint_weight=checkpoint_weight,
            random_weight=random_weight,
        )

    def sample_opponent_policy(self, current_policy: nn.Module) -> nn.Module | Callable:
        """
        Sample an opponent policy/strategy based on dynamic mastery-weighted curriculum.
        Returns either a policy or a callable strategy function.

        The opponent mix is computed dynamically based on mastery levels:
        - Phase 1: Learning greedy (50-60% greedy, 15-20% smart/self-play)
        - Phase 2: Learning smart while retaining greedy (15-20% greedy retention, 45-50% smart)
        - Phase 3: Self-play focus with retention (10-15% greedy, 15-20% smart, 55-65% self-play)
        """
        # Compute dynamic opponent mix based on current mastery levels
        opponent_mix = self.compute_dynamic_opponent_mix()
        weights, strategies = opponent_mix.get_weights_and_strategies()

        # Sample strategy type using weighted random choice
        chosen_strategy = np.random.choice(strategies, p=weights)

        if chosen_strategy == "greedy":
            return greedy_strategy
        elif chosen_strategy == "smart":
            return smart_strategy
        elif chosen_strategy == "current":
            return current_policy
        elif chosen_strategy == "checkpoint":
            if len(self.checkpoints) > 0:
                # Sample from recent checkpoints (last 10)
                _, checkpoint_policy = random.choice(self.checkpoints[-20:])
                return checkpoint_policy
            else:
                # Fallback to random if no checkpoints available
                return random_strategy
        else:  # random
            return random_strategy


def compute_gae_from_values(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE) with explicit episode boundaries.

    Args:
        rewards: [T] rewards for each step
        values: [T] value predictions for each step
        dones: [T] tensor with 1.0 at terminal steps, 0.0 otherwise
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: [T]
        returns: [T] = advantages + values_detached
    """
    T = values.size(0)
    values_detached = values.detach()
    adv = torch.zeros_like(values_detached)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        next_nonterminal = 1.0 - dones[t]
        v_next = values_detached[t + 1] if t + 1 < T else torch.zeros((), device=values.device, dtype=values.dtype)
        delta = rewards[t] + gamma * v_next * next_nonterminal - values_detached[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
    returns = adv + values_detached
    return adv, returns


def plot_training_curves(
    loss_history,
    policy_loss_history,
    value_loss_history,
    entropy_history,
    max_logprob_history=None,
    eval_episodes=None,
    win_rates=None,
    win_rates_smart=None,
    opponent_mixes=None,
    save_path="training_curves.png",
):
    """Plot training curves and save to file."""
    # Create figure with subplots - adjust layout based on available data
    has_eval = eval_episodes and win_rates
    has_opponent_mix = opponent_mixes and len(opponent_mixes) > 0

    if has_eval and has_opponent_mix:
        # 7 subplots: 5 loss/metrics + win rate + opponent mix
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[:2, 2])  # Win rate takes top 2 rows of right column
        ax7 = fig.add_subplot(gs[2, 1:])  # Opponent mix takes bottom row spanning 2 columns
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    elif has_eval:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[:, 2])  # Win rate takes full right column
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[0, 2]),
        ]

    fig.suptitle("Big 2 Training Curves", fontsize=16, fontweight="bold")

    episodes = range(1, len(loss_history) + 1)

    # Total Loss
    axes[0].plot(episodes, loss_history, linewidth=1.5, color="#2E86AB")
    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Total Loss", fontsize=11)
    axes[0].set_title("Total Loss", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, len(loss_history))

    # Policy Loss
    axes[1].plot(episodes, policy_loss_history, linewidth=1.5, color="#A23B72")
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].set_ylabel("Policy Loss", fontsize=11)
    axes[1].set_title("Policy Loss", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, len(loss_history))

    # Value Loss
    axes[2].plot(episodes, value_loss_history, linewidth=1.5, color="#F18F01")
    axes[2].set_xlabel("Episode", fontsize=11)
    axes[2].set_ylabel("Value Loss", fontsize=11)
    axes[2].set_title("Value Loss", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(1, len(loss_history))

    # Entropy
    axes[3].plot(episodes, entropy_history, linewidth=1.5, color="#6A994E")
    axes[3].set_xlabel("Episode", fontsize=11)
    axes[3].set_ylabel("Entropy", fontsize=11)
    axes[3].set_title("Policy Entropy", fontsize=12, fontweight="bold")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(1, len(loss_history))

    # Max Logprob (if available)
    if max_logprob_history and len(axes) > 4:
        axes[4].plot(episodes, max_logprob_history, linewidth=1.5, color="#9B59B6")
        axes[4].set_xlabel("Episode", fontsize=11)
        axes[4].set_ylabel("Max Logprob", fontsize=11)
        axes[4].set_title("Maximum Logprob Across Actions", fontsize=12, fontweight="bold")
        axes[4].grid(True, alpha=0.3)
        axes[4].set_xlim(1, len(loss_history))

    # Win Rate (if available)
    if eval_episodes and win_rates and len(axes) > 5:
        axes[5].plot(
            eval_episodes, win_rates, linewidth=2.5, color="#C9184A", marker="o", markersize=6, label="vs Greedy"
        )
        if win_rates_smart:
            axes[5].plot(
                eval_episodes,
                win_rates_smart,
                linewidth=2.5,
                color="#4A90E2",
                marker="s",
                markersize=6,
                label="vs Smart",
            )
        axes[5].axhline(y=0.25, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Expected Random (25%)")
        axes[5].set_xlabel("Episode", fontsize=11)
        axes[5].set_ylabel("Win Rate", fontsize=11)
        axes[5].set_title(
            "Win Rate vs Opponents\n(Current policy as Player 0)",
            fontsize=12,
            fontweight="bold",
        )
        axes[5].grid(True, alpha=0.3)
        axes[5].set_xlim(min(eval_episodes) if eval_episodes else 0, len(loss_history))
        axes[5].set_ylim(0, 1)
        axes[5].legend(loc="lower right")
        # Add percentage labels
        axes[5].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Opponent Mix (if available)
    if opponent_mixes and len(opponent_mixes) > 0 and len(axes) > 6:
        # Extract weights for each opponent type
        greedy_weights = [mix.greedy_weight for mix in opponent_mixes]
        smart_weights = [mix.smart_weight for mix in opponent_mixes]
        current_weights = [mix.current_weight for mix in opponent_mixes]
        checkpoint_weights = [mix.checkpoint_weight for mix in opponent_mixes]
        random_weights = [mix.random_weight for mix in opponent_mixes]

        # Use stacked area plot to show composition
        axes[6].fill_between(eval_episodes, 0, greedy_weights, label="Greedy", color="#FF6B6B", alpha=0.7)
        bottom = greedy_weights
        axes[6].fill_between(
            eval_episodes,
            bottom,
            [b + s for b, s in zip(bottom, smart_weights, strict=False)],
            label="Smart",
            color="#4ECDC4",
            alpha=0.7,
        )
        bottom = [b + s for b, s in zip(bottom, smart_weights, strict=False)]
        axes[6].fill_between(
            eval_episodes,
            bottom,
            [b + c for b, c in zip(bottom, current_weights, strict=False)],
            label="Current",
            color="#95E1D3",
            alpha=0.7,
        )
        bottom = [b + c for b, c in zip(bottom, current_weights, strict=False)]
        axes[6].fill_between(
            eval_episodes,
            bottom,
            [b + c for b, c in zip(bottom, checkpoint_weights, strict=False)],
            label="Checkpoint",
            color="#F38181",
            alpha=0.7,
        )
        bottom = [b + c for b, c in zip(bottom, checkpoint_weights, strict=False)]
        axes[6].fill_between(
            eval_episodes,
            bottom,
            [b + r for b, r in zip(bottom, random_weights, strict=False)],
            label="Random",
            color="#AA96DA",
            alpha=0.7,
        )

        axes[6].set_xlabel("Episode", fontsize=11)
        axes[6].set_ylabel("Opponent Mix Weight", fontsize=11)
        axes[6].set_title("Opponent Strategy Distribution", fontsize=12, fontweight="bold")
        axes[6].grid(True, alpha=0.3)
        axes[6].set_xlim(
            min(eval_episodes) if eval_episodes else 0, max(eval_episodes) if eval_episodes else len(loss_history)
        )
        axes[6].set_ylim(0, 1)
        axes[6].legend(loc="upper left", fontsize=9)
        # Add percentage labels
        axes[6].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def value_of_starting_hand(
    policy: nn.Module,
    hand: list[int],
    n_players: int = 4,
    cards_per_player: int | None = None,
    sims: int = 512,
    device="cpu",
) -> float:
    # Monte Carlo rollouts with frozen policy; returns expected terminal reward for seat 0 with given starting hand
    wins = 0.0
    if cards_per_player is None:
        cards_per_player = 52 // n_players
    total_cards_in_play = n_players * cards_per_player
    if len(hand) != cards_per_player:
        raise ValueError(f"hand size ({len(hand)}) must equal cards_per_player ({cards_per_player})")
    if any(card < 0 or card >= total_cards_in_play for card in hand):
        raise ValueError(f"hand cards must be within 0..{total_cards_in_play - 1}")
    for _ in range(sims):
        env = Big2Env(n_players, cards_per_player=cards_per_player)
        # Force seat 0 hand
        deck = set(range(total_cards_in_play))
        for p in range(n_players):
            env.hands[p] = []
        env.hands[0] = sorted(hand)
        remain = list(deck - set(hand))
        random.shuffle(remain)
        for p in range(1, n_players):
            start_idx = (p - 1) * cards_per_player
            end_idx = p * cards_per_player
            env.hands[p] = sorted(remain[start_idx:end_idx])
        # who holds 3♦ starts
        start_player = 0
        for p in range(n_players):
            if 0 in env.hands[p]:
                start_player = p
                break
        env.current_player = start_player
        env.trick_pile = None
        env.passes_in_row = 0
        env.seen = [0] * env.card_universe_size
        env.done = False
        env.winner = None

        # Rollout
        state = env._obs(env.current_player)
        while True:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            action, logprob, entropy, value, max_logprob = select_action(policy, state, candidates)
            next_state, done = env.step(action)
            if done:
                wins += 1.0 if env.winner == 0 else 0.0
                break
            state = next_state
    return wins / sims
