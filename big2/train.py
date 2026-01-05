import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from big2.nn import MLPPolicy, combo_to_action_vector
from big2.simulator.cards import PASS, Combo, card_name
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy


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


def episode(env: Big2Env, policy: MLPPolicy, greedy_players: set[int] | None = None):
    """
    Run one episode. Players in greedy_players use greedy strategy, others use the model.
    Only model players' trajectories are recorded for training.
    """
    if greedy_players is None:
        greedy_players = set()

    # Records per player (only for model players)
    traj: dict[int, list[StepRecord]] = {p: [] for p in range(env.n_players)}
    state = env.reset()
    while True:
        p = env.current_player
        candidates = env.legal_candidates(p)
        # Safety: if no candidates somehow, force PASS
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        if p in greedy_players:
            # Greedy player: no gradient tracking needed
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
            for q in range(env.n_players):
                # Only update rewards for model players
                if q in greedy_players:
                    continue
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


def play_evaluation_game(current_policy: MLPPolicy, n_players: int, device="cpu") -> int | None:
    """
    Play one game where player 0 uses current_policy and players 1-(n_players-1) use greedy strategy.
    Returns the winner (0 if current policy wins, 1-(n_players-1) if greedy wins).
    """
    env = Big2Env(n_players)
    state = env.reset()

    while not env.done:
        p = env.current_player
        candidates = env.legal_candidates(p)
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        # Player 0 uses current policy, others use greedy strategy
        if p == 0:
            action = select_action_greedy(current_policy, state, candidates)
        else:
            action = greedy_strategy(candidates)

        state, _ = env.step(action)

    return env.winner


def evaluate_against_greedy(current_policy: MLPPolicy, n_players: int, num_games: int = 100, device="cpu") -> float:
    """
    Evaluate current policy against greedy strategy by playing num_games.
    Returns win rate of current policy (player 0).
    """
    wins = 0
    for _ in range(num_games):
        winner = play_evaluation_game(current_policy, n_players, device)
        if winner == 0:
            wins += 1

    win_rate = wins / num_games
    return win_rate


def compute_gae_from_values(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    rewards: [T]  (on that player's decision steps; typically 0,...,0, ±1 at the end)
    values:  [T]  (V(s_t) at each of those steps)
    returns: [T]  = advantages + values
    """
    T = values.size(0)
    values = torch.zeros_like(values)
    # v_next is values shifted left, with 0 bootstrap at the last step (episode end)
    v_next = torch.cat([values[1:], torch.zeros(1, device=values.device, dtype=values.dtype)])
    deltas = rewards + gamma * v_next - values

    adv = torch.zeros_like(values)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train_selfplay(
    n_players=4,
    batches=2000,
    episodes_per_batch=10,
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

    # Adaptive greedy opponent schedule based on win rate
    # Start with all greedy opponents, reduce as model improves
    # Schedule: n_players - 1 -> n_players // 2 (at 25% win rate) -> 0 (at 30% win rate)
    greedy_schedule = [n_players - 1, n_players // 2, 0]
    greedy_thresholds = [0.0, 0.25, 0.30]  # Win rate thresholds to advance
    current_schedule_idx = 0
    n_greedy_opponents = greedy_schedule[current_schedule_idx]

    # Define which players use greedy strategy (last n_greedy_opponents players)
    # Model players are 0 to (n_players - n_greedy_opponents - 1)
    greedy_players = set(range(n_players - n_greedy_opponents, n_players))
    model_players = set(range(n_players)) - greedy_players
    print(f"Greedy opponent schedule: {greedy_schedule} at win rate thresholds {greedy_thresholds}")
    print(f"Starting with {len(model_players)} model players and {len(greedy_players)} greedy opponents")
    print(f"Model players: {sorted(model_players)}, Greedy players: {sorted(greedy_players)}")

    env = Big2Env(n_players)
    policy = MLPPolicy(n_players=n_players, device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Learning rate annealing
    initial_lr = lr
    lr_decay = initial_lr / batches

    # Track losses for plotting
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    max_logprob_history = []

    # Track evaluation metrics
    eval_episodes = []
    win_rates = []

    for batch in tqdm(range(1, batches + 1)):
        batch_logp, batch_val, batch_ret, batch_adv, batch_ent, batch_max_logp = [], [], [], [], [], []
        for _ in range(1, episodes_per_batch + 1):
            traj = episode(env, policy, greedy_players=greedy_players)
            for p in model_players:
                if len(traj[p]) == 0:
                    continue

                # tensors
                rewards_t = torch.tensor([rec.reward for rec in traj[p]], dtype=torch.float32, device=device)  # [T]
                values_t = torch.stack([rec.value for rec in traj[p]]).squeeze(-1)  # [T] (ensure 1D)
                logprobs_t = torch.stack([rec.logprob for rec in traj[p]])  # [T]
                entropies_t = torch.stack([rec.entropy for rec in traj[p]])  # [T]
                max_logprobs_t = torch.stack([rec.max_logprob for rec in traj[p]])  # [T]

                # GAE(lambda) returns
                adv_t, ret_t = compute_gae_from_values(rewards_t, values_t, gamma=gamma, lam=0.95)

                batch_logp.append(logprobs_t)
                batch_val.append(values_t)
                batch_ret.append(ret_t.detach())
                batch_adv.append(adv_t)
                batch_ent.append(entropies_t)
                batch_max_logp.append(max_logprobs_t)

        # concat
        logp = torch.cat(batch_logp)
        vpred = torch.cat(batch_val)
        ret = torch.cat(batch_ret)
        adv = torch.cat(batch_adv)
        ent = torch.cat(batch_ent)
        max_logp = torch.cat(batch_max_logp)

        # normalize advantages across the batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(vpred, ret)
        entropy = ent.mean()
        avg_max_logprob = max_logp.mean()

        loss = policy_loss - entropy_beta * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Anneal learning rate linearly
        new_lr = initial_lr - (batch * lr_decay)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        # Record losses
        loss_history.append(loss.item())
        policy_loss_history.append(policy_loss.item())
        value_loss_history.append(value_loss.item())
        entropy_history.append(entropy.item())
        max_logprob_history.append(avg_max_logprob.item())

        # Evaluation every eval_interval episodes
        if batch % eval_interval == 0:
            avg_len = sum(len(traj[p]) for p in model_players) / len(model_players)

            # Evaluate against greedy policy
            print(f"\n[Step {batch}] Evaluating against greedy policy...")
            policy.eval()
            win_rate = evaluate_against_greedy(policy, n_players, num_games=eval_games, device=device)
            policy.train()

            eval_episodes.append(batch)
            win_rates.append(win_rate)

            print(f"[Step {batch}] Win rate vs greedy: {win_rate:.2%} ({int(win_rate * eval_games)}/{eval_games} wins)")
            print(
                f"[Step {batch}] loss={loss.item():.3f} pol={policy_loss.item():.3f} "
                f"val={value_loss.item():.3f} ent={entropy.item():.3f} steps/player~{avg_len:.1f}"
            )
            print(f"[Step {batch}] Current greedy opponents: {n_greedy_opponents}\n")

            # Check if we should advance to the next stage of greedy opponent reduction
            if current_schedule_idx < len(greedy_schedule) - 1:
                next_threshold = greedy_thresholds[current_schedule_idx + 1]
                if win_rate >= next_threshold:
                    current_schedule_idx += 1
                    n_greedy_opponents = greedy_schedule[current_schedule_idx]
                    greedy_players = set(range(n_players - n_greedy_opponents, n_players))
                    model_players = set(range(n_players)) - greedy_players
                    print(f"*** Win rate {win_rate:.2%} >= {next_threshold:.0%} threshold ***")
                    print(f"*** Advancing to {n_greedy_opponents} greedy opponents ***")
                    print(f"*** Model players: {sorted(model_players)}, Greedy players: {sorted(greedy_players)} ***\n")

            save_path = f"big2_model_step_{batch}.pt"
            torch.save(policy.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    return (
        policy,
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        max_logprob_history,
        eval_episodes,
        win_rates,
    )


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_players = 4  # Greedy opponents adapt based on win rate: 3 -> 2 -> 0
    cards_per_player = 52 // n_players

    (
        policy,
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        max_logprob_history,
        eval_episodes,
        win_rates,
    ) = train_selfplay(
        n_players=n_players,
        batches=500,
        episodes_per_batch=16,
        lr=0.0003,
        entropy_beta=0.02,
        value_coef=0.5,
        gamma=0.99,
        seed=42,
        device=device,
        eval_interval=25,
        eval_games=100,
    )

    hand = sorted([48, 49, 50, 51, 47, 43, 39, 35, 31, 27, 26, 1, 0][:cards_per_player])
    val = value_of_starting_hand(policy, hand, n_players=n_players, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Win rate:", val)

    hand = sorted(random.sample(range(52), cards_per_player))
    val = value_of_starting_hand(policy, hand, n_players=n_players, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Win rate:", val)

    # Save the trained model
    save_path = "big2_model.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"\nModel weights saved to {save_path}")

    # Plot and save training curves
    plot_training_curves(
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        max_logprob_history,
        eval_episodes,
        win_rates,
    )
