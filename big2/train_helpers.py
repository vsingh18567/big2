import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from big2.nn import MLPPolicy, combo_to_action_vector
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


def select_action(
    policy: MLPPolicy, state: np.ndarray, candidates: list[Combo]
) -> tuple[Combo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Prepare batch with one element
    st = torch.from_numpy(state[np.newaxis, :]).long().to(policy.device)
    action_feats = [[combo_to_action_vector(c) for c in candidates]]
    logits_list, values = policy(st, action_feats)
    logits = logits_list[0]
    # Softmax over candidate set
    probs = F.softmax(logits, dim=0)
    dist = torch.distributions.Categorical(probs=probs)
    idx = dist.sample()
    logprob = dist.log_prob(idx)
    entropy = dist.entropy()
    # Compute max logprob across all actions
    max_prob = probs.max()
    chosen = candidates[int(idx.item())]
    value = values[0]
    return chosen, logprob, entropy, value, max_prob


def episode(
    env: Big2Env,
    policy: MLPPolicy,
    opponent_strategies: dict[int, MLPPolicy | Callable[[list[Combo]], Combo]] | None = None,
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
            if isinstance(strategy, MLPPolicy):
                # Policy-based opponent (current policy or past checkpoint)
                action = select_action_greedy(strategy, state, candidates)
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


def compute_returns(rewards: list[float], gamma: float = 1.0) -> list[float]:
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def select_action_greedy(policy: MLPPolicy, state: np.ndarray, candidates: list[Combo]) -> Combo:
    """Select action greedily (no sampling) for evaluation."""
    with torch.no_grad():
        st = torch.from_numpy(state[np.newaxis, :]).long().to(policy.device)
        action_feats = [[combo_to_action_vector(c) for c in candidates]]
        logits_list, _ = policy(st, action_feats)
        logits = logits_list[0]
        idx = logits.argmax()
        return candidates[idx]


def random_strategy(candidates: list[Combo]) -> Combo:
    """Select a random legal action."""
    return random.choice(candidates)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    win_rate_vs_greedy: float
    win_rate_vs_random: float
    win_rate_vs_smart: float
    avg_cards_remaining_when_losing: float
    win_rate_by_starting_position: dict[int, float]
    total_games: int


@dataclass
class OpponentMix:
    """Defines the distribution of opponent strategies for a training stage."""

    greedy_weight: float
    smart_weight: float
    current_weight: float
    checkpoint_weight: float
    random_weight: float

    def get_weights_and_strategies(self) -> tuple[list[float], list[str]]:
        """Return weights and strategy names in a consistent order."""
        return (
            [self.greedy_weight, self.smart_weight, self.current_weight, self.checkpoint_weight, self.random_weight],
            ["greedy", "smart", "current", "checkpoint", "random"],
        )


def play_evaluation_game(
    current_policy: MLPPolicy,
    n_players: int,
    opponent_strategy: Callable[[list[Combo]], Combo] = greedy_strategy,
    device="cpu",
) -> tuple[int | None, int, int]:
    """
    Play one evaluation game where player 0 uses current_policy and others use opponent_strategy.

    Returns:
        (winner, starting_position, cards_remaining_for_player_0)
    """
    env = Big2Env(n_players)
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

    cards_remaining = len(env.hands[0]) if env.winner != 0 else 0
    return env.winner, starting_position, cards_remaining


def evaluate_against_greedy(
    current_policy: MLPPolicy, n_players: int, num_games: int = 500, device="cpu"
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
    wins_by_position: dict[int, int] = {}
    games_by_position: dict[int, int] = {}

    # Evaluate against greedy
    for _ in range(num_games):
        winner, starting_pos, cards_remaining = play_evaluation_game(
            current_policy, n_players, opponent_strategy=greedy_strategy, device=device
        )
        if winner == 0:
            wins_vs_greedy += 1
            wins_by_position[starting_pos] = wins_by_position.get(starting_pos, 0) + 1
        else:
            cards_remaining_sum += cards_remaining
            losses_count += 1
        games_by_position[starting_pos] = games_by_position.get(starting_pos, 0) + 1

    # Evaluate against random
    for _ in range(num_games):
        winner, _, _ = play_evaluation_game(current_policy, n_players, opponent_strategy=random_strategy, device=device)
        if winner == 0:
            wins_vs_random += 1

    # Evaluate against smart strategy
    for _ in range(num_games):
        winner, _, _ = play_evaluation_game(current_policy, n_players, opponent_strategy=smart_strategy, device=device)
        if winner == 0:
            wins_vs_smart += 1

    win_rate_vs_greedy = wins_vs_greedy / num_games
    win_rate_vs_random = wins_vs_random / num_games
    win_rate_vs_smart = wins_vs_smart / num_games
    avg_cards_remaining = cards_remaining_sum / losses_count if losses_count > 0 else 0.0
    win_rate_by_position = {pos: wins_by_position.get(pos, 0) / games_by_position[pos] for pos in games_by_position}

    return EvaluationMetrics(
        win_rate_vs_greedy=win_rate_vs_greedy,
        win_rate_vs_random=win_rate_vs_random,
        win_rate_vs_smart=win_rate_vs_smart,
        avg_cards_remaining_when_losing=avg_cards_remaining,
        win_rate_by_starting_position=win_rate_by_position,
        total_games=num_games,
    )


class CheckpointManager:
    """Manages past checkpoints for opponent sampling."""

    # Define opponent mix for each training stage
    STAGE_0 = OpponentMix(
        greedy_weight=0.40, smart_weight=0.0, current_weight=0.30, checkpoint_weight=0.0, random_weight=0.30
    )
    STAGE_1 = OpponentMix(
        greedy_weight=0.30, smart_weight=0.10, current_weight=0.35, checkpoint_weight=0.05, random_weight=0.10
    )
    STAGE_2 = OpponentMix(
        greedy_weight=0.20, smart_weight=0.20, current_weight=0.45, checkpoint_weight=0.10, random_weight=0.05
    )

    def __init__(self, checkpoint_dir: str = "big2", device: str = "cpu", n_players: int = 4):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.n_players = n_players
        self.checkpoints: list[tuple[int, MLPPolicy]] = []  # (step, policy)
        self.max_checkpoints = 20  # Keep last 20 checkpoints
        self.greedy_schedule_stage = 0  # 0: <20%, 1: 20-30%, 2: >=30%

    def add_checkpoint(self, step: int, policy: MLPPolicy):
        """Add a checkpoint to the pool."""
        # Create a copy of the policy
        checkpoint_policy = MLPPolicy(n_players=self.n_players, device=self.device).to(self.device)
        checkpoint_policy.load_state_dict(policy.state_dict())
        checkpoint_policy.eval()

        self.checkpoints.append((step, checkpoint_policy))

        # Keep only the most recent checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def update_greedy_schedule(self, win_rate: float):
        """
        Update the greedy schedule stage based on win rate against greedy.
        Schedule only advances forward (never regresses).

        Stage 0: win_rate < 0.20 → heavy random/greedy mix (easy)
        Stage 1: 0.20 ≤ win_rate < 0.30 → add a bit of smart/self-play
        Stage 2: win_rate ≥ 0.30 → more self-play and smart, less random
        """
        if win_rate >= 0.30 and self.greedy_schedule_stage < 2:
            self.greedy_schedule_stage = 2
        elif win_rate >= 0.20 and self.greedy_schedule_stage < 1:
            self.greedy_schedule_stage = 1

    def sample_opponent_policy(self, current_policy: MLPPolicy) -> MLPPolicy | Callable:
        """
        Sample an opponent policy/strategy based on the current greedy schedule stage.
        Returns either a policy or a callable strategy function.

        The stage determines the opponent mix:
        - Stage 0 (<20% win rate): More greedy/smart to learn basics
        - Stage 1 (20-30% win rate): Balanced mix with more self-play
        - Stage 2 (≥30% win rate): Heavy self-play with occasional baselines
        """
        # Select the appropriate opponent mix based on stage
        opponent_mix = [self.STAGE_0, self.STAGE_1, self.STAGE_2][self.greedy_schedule_stage]
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
                # Sample from last 5 checkpoints
                _, checkpoint_policy = random.choice(self.checkpoints[-5:])
                return checkpoint_policy
            else:
                # Fallback to random if no checkpoints available
                return random_strategy
        else:  # random
            return random_strategy


def compute_gae_from_values(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    rewards: [T]  (on that player's decision steps; typically 0,...,0, ±1 at the end)
    values:  [T]  (V(s_t) at each of those steps)
    returns: [T]  = advantages + values (used as targets for value function)
    """
    T = values.size(0)
    # Detach values to avoid backprop through advantage computation
    values_detached = values.detach()
    # v_next is values shifted left, with 0 bootstrap at the last step (episode end)
    v_next = torch.cat([values_detached[1:], torch.zeros(1, device=values.device, dtype=values.dtype)])
    deltas = rewards + gamma * v_next - values_detached

    adv = torch.zeros_like(values_detached)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * lam * gae
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
    save_path="training_curves.png",
):
    """Plot training curves and save to file."""
    # Create figure with 6 subplots if we have evaluation data, otherwise 5
    if eval_episodes and win_rates:
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
        axes[5].plot(eval_episodes, win_rates, linewidth=2.5, color="#C9184A", marker="o", markersize=6)
        axes[5].axhline(y=0.25, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Expected Random (25%)")
        axes[5].set_xlabel("Episode", fontsize=11)
        axes[5].set_ylabel("Win Rate", fontsize=11)
        axes[5].set_title(
            "Win Rate vs Greedy Policy\n(Current policy as Player 0 vs greedy opponents)",
            fontsize=12,
            fontweight="bold",
        )
        axes[5].grid(True, alpha=0.3)
        axes[5].set_xlim(min(eval_episodes) if eval_episodes else 0, len(loss_history))
        axes[5].set_ylim(0, 1)
        axes[5].legend(loc="lower right")
        # Add percentage labels
        axes[5].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def value_of_starting_hand(
    policy: MLPPolicy, hand: list[int], n_players: int = 4, sims: int = 512, device="cpu"
) -> float:
    # Monte Carlo rollouts with frozen policy; returns expected terminal reward for seat 0 with given starting hand
    wins = 0.0
    cards_per_player = 52 // n_players
    for _ in range(sims):
        env = Big2Env(n_players)
        # Force seat 0 hand
        deck = set(range(52))
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
        env.seen = [0] * 52
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
