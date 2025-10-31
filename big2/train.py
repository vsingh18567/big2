import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from .nn import MLPPolicy, combo_to_action_vector
from .simulator.cards import PASS, Combo, card_name
from .simulator.env import Big2Env


@dataclass
class StepRecord:
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    reward: float


def select_action(
    policy: MLPPolicy, state: np.ndarray, candidates: list[Combo]
) -> tuple[Combo, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    chosen = candidates[int(idx.item())]
    value = values[0]
    return chosen, logprob, entropy, value


def episode(env: Big2Env, policy: MLPPolicy, device="cpu"):
    # Records per player
    print("NEW EPISODE")
    traj: dict[int, list[StepRecord]] = {p: [] for p in range(4)}
    state = env.reset()
    while True:
        p = env.current_player
        candidates = env.legal_candidates(p)
        # Safety: if no candidates somehow, force PASS
        if not candidates:
            candidates = [Combo(PASS, [], ())]
        action, logprob, entropy, value = select_action(policy, state, candidates)

        # Debug printing disabled
        # hand_cards = env.hands[p]
        # hand_str = " ".join([card_name(c) for c in sorted(hand_cards)])
        # if action.type == PASS:
        #     action_str = "PASS"
        # else:
        #     action_str = " ".join([card_name(c) for c in action.cards])
        # print(f"Player {p} | Hand: [{hand_str}] | Action: {action.type} {action_str}")

        next_state, reward, done, _ = env.step(action)
        # Store step
        traj[p].append(StepRecord(logprob=logprob, entropy=entropy, value=value, reward=0.0))
        # Assign terminal rewards at the end
        if done:
            winner = env.winner
            for q in range(4):
                final_r = 1.0 if q == winner else -1.0
                # set last step reward for each player (Monte Carlo return will propagate)
                if len(traj[q]) > 0:
                    traj[q][-1] = StepRecord(
                        logprob=traj[q][-1].logprob,
                        entropy=traj[q][-1].entropy,
                        value=traj[q][-1].value,
                        reward=final_r,
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
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()
        return candidates[int(idx.item())]


def play_evaluation_game(current_policy: MLPPolicy, baseline_policy: MLPPolicy, device="cpu") -> int:
    """
    Play one game where player 0 uses current_policy and players 1-3 use baseline_policy.
    Returns the winner (0 if current policy wins, 1-3 if baseline wins).
    """
    env = Big2Env(4)
    state = env.reset()

    while not env.done:
        p = env.current_player
        candidates = env.legal_candidates(p)
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        # Player 0 uses current policy, others use baseline
        if p == 0:
            action = select_action_greedy(current_policy, state, candidates)
        else:
            action = select_action_greedy(baseline_policy, state, candidates)

        state, _, _, _ = env.step(action)

    assert env.winner is not None, "Game should have a winner"
    return env.winner


def evaluate_against_baseline(
    current_policy: MLPPolicy, baseline_policy: MLPPolicy, num_games: int = 100, device="cpu"
) -> float:
    """
    Evaluate current policy against baseline by playing num_games.
    Returns win rate of current policy (player 0).
    """
    wins = 0
    for _ in range(num_games):
        winner = play_evaluation_game(current_policy, baseline_policy, device)
        if winner == 0:
            wins += 1

    win_rate = wins / num_games
    return win_rate


def train_selfplay(
    episodes=2000,
    lr=3e-4,
    entropy_beta=0.01,
    value_coef=0.5,
    gamma=1.0,
    seed=42,
    device="cpu",
    eval_interval=50,
    eval_games=100,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Big2Env(4)
    policy = MLPPolicy(device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Track losses for plotting
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []

    # Track evaluation metrics
    eval_episodes: list[int] = []
    win_rates: list[float] = []
    baseline_policy: MLPPolicy | None = None

    for ep in range(1, episodes + 1):
        traj = episode(env, policy, device=device)

        # Build losses
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_term = torch.tensor(0.0, device=device)
        count_steps = 0
        for p in range(4):
            if len(traj[p]) == 0:
                continue
            rewards = [rec.reward for rec in traj[p]]
            returns = compute_returns(rewards, gamma)
            # Tensorize
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            values_t = torch.stack([rec.value for rec in traj[p]])
            logprobs_t = torch.stack([rec.logprob for rec in traj[p]])
            entropies_t = torch.stack([rec.entropy for rec in traj[p]])
            advantages = returns_t - values_t.detach()
            policy_loss = policy_loss - (logprobs_t * advantages).sum()
            value_loss = value_loss + F.mse_loss(values_t, returns_t)
            entropy_term = entropy_term + entropies_t.sum()
            count_steps += len(traj[p])

        loss = policy_loss + value_coef * value_loss - entropy_beta * entropy_term
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Record losses
        loss_history.append(loss.item())
        policy_loss_history.append(policy_loss.item())
        value_loss_history.append(value_loss.item())
        entropy_history.append(entropy_term.item())

        # Evaluation every eval_interval episodes
        if ep % eval_interval == 0:
            avg_len = sum(len(traj[p]) for p in range(4)) / 4.0

            if baseline_policy is not None:
                # Evaluate against baseline from eval_interval episodes ago
                print(f"\n[Episode {ep}] Evaluating against baseline from episode {ep - eval_interval}...")
                policy.eval()
                baseline_policy.eval()
                win_rate = evaluate_against_baseline(policy, baseline_policy, num_games=eval_games, device=device)
                policy.train()
                baseline_policy.train()

                eval_episodes.append(ep)
                win_rates.append(win_rate)

                wins_count = int(win_rate * eval_games)
                print(f"[Episode {ep}] Win rate vs baseline: {win_rate:.2%} ({wins_count}/{eval_games} wins)")
                print(
                    f"[Episode {ep}] loss={loss.item():.3f} pol={policy_loss.item():.3f} "
                    f"val={value_loss.item():.3f} ent={entropy_term.item():.3f} steps/player~{avg_len:.1f}\n"
                )
                # Update baseline checkpoint to current policy
            baseline_policy = MLPPolicy(device=device).to(device)
            baseline_policy.load_state_dict(policy.state_dict())
            baseline_policy.eval()

    return policy, loss_history, policy_loss_history, value_loss_history, entropy_history, eval_episodes, win_rates


def plot_training_curves(
    loss_history,
    policy_loss_history,
    value_loss_history,
    entropy_history,
    eval_episodes=None,
    win_rates=None,
    save_path="training_curves.png",
):
    """Plot training curves and save to file."""
    # Create figure with 5 subplots if we have evaluation data, otherwise 4
    if eval_episodes and win_rates:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[:, 2])  # Win rate takes full right column
        axes = [ax1, ax2, ax3, ax4, ax5]
    else:
        fig, axes_2d = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes_2d.flatten()

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

    # Win Rate (if available)
    if eval_episodes and win_rates and len(axes) > 4:
        axes[4].plot(eval_episodes, win_rates, linewidth=2.5, color="#C9184A", marker="o", markersize=6)
        axes[4].axhline(y=0.25, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random (25%)")
        axes[4].set_xlabel("Episode", fontsize=11)
        axes[4].set_ylabel("Win Rate", fontsize=11)
        axes[4].set_title(
            "Win Rate vs Previous Checkpoint\n(Current policy as Player 0 vs 3 baseline copies)",
            fontsize=12,
            fontweight="bold",
        )
        axes[4].grid(True, alpha=0.3)
        axes[4].set_xlim(min(eval_episodes) if eval_episodes else 0, len(loss_history))
        axes[4].set_ylim(0, 1)
        axes[4].legend(loc="lower right")
        # Add percentage labels
        axes[4].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def value_of_starting_hand(policy: MLPPolicy, hand: list[int], sims: int = 512, device="cpu") -> float:
    # Monte Carlo rollouts with frozen policy; returns expected terminal reward for seat 0 with given starting hand
    wins = 0.0
    for _ in range(sims):
        env = Big2Env(4)
        # Force seat 0 hand
        deck = set(range(52))
        for p in range(4):
            env.hands[p] = []
        env.hands[0] = sorted(hand)
        remain = list(deck - set(hand))
        random.shuffle(remain)
        env.hands[1] = sorted(remain[:13])
        env.hands[2] = sorted(remain[13:26])
        env.hands[3] = sorted(remain[26:39])
        # who holds 3♦ starts
        start_player = 0
        for p in range(4):
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
            action, logprob, entropy, value = select_action(policy, state, candidates)
            next_state, reward, done, _ = env.step(action)
            if done:
                wins += 1.0 if env.winner == 0 else -1.0
                break
            state = next_state
    return wins / sims


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, loss_history, policy_loss_history, value_loss_history, entropy_history, eval_episodes, win_rates = (
        train_selfplay(
            episodes=1000,
            lr=0.001,
            entropy_beta=0.01,
            value_coef=0.5,
            gamma=0.9,
            seed=42,
            device=device,
            eval_interval=50,
            eval_games=100,
        )
    )

    hand = sorted([48, 49, 50, 51, 47, 43, 39, 35, 31, 27, 26, 1, 0])
    val = value_of_starting_hand(policy, hand, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Estimated value:", val)

    hand = sorted(random.sample(range(52), 13))
    val = value_of_starting_hand(policy, hand, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Estimated value:", val)

    # Save the trained model
    save_path = "big2_model.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"\nModel weights saved to {save_path}")

    # Plot and save training curves
    plot_training_curves(
        loss_history, policy_loss_history, value_loss_history, entropy_history, eval_episodes, win_rates
    )
