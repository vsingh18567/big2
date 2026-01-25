import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from big2.nn import SetPoolQNetwork, combo_to_action_vector
from big2.simulator.cards import PASS, Combo, card_name
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy
from big2.train_helpers import (
    plot_training_curves,
    random_strategy,
    select_action_q_epsilon_greedy,
    select_action_q_greedy,
)


@dataclass
class QStepRecord:
    state: np.ndarray
    action_idx: int
    candidates: list[Combo]
    reward: float
    done: bool = False


def play_evaluation_game_q(
    current_qnet: SetPoolQNetwork,
    n_players: int,
    opponent_strategy: Callable[[list[Combo]], Combo] = greedy_strategy,
    epsilon: float = 0.0,
) -> tuple[int | None, int, int, int]:
    """
    Play one evaluation game where player 0 uses greedy-Q, others use opponent_strategy.

    Returns:
        (winner, starting_position, cards_remaining_for_player_0, score_for_player_0)
    """
    env = Big2Env(n_players)
    _ = env.reset()
    starting_position = env.current_player

    while not env.done:
        p = env.current_player
        candidates = env.legal_candidates(p)
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        state = env._obs(p)
        if p == 0:
            # Evaluate with epsilon=0 by default (pure greedy).
            action, _ = select_action_q_epsilon_greedy(current_qnet, state, candidates, epsilon=epsilon)
        else:
            if opponent_strategy == smart_strategy:
                action = opponent_strategy(candidates, env.hands[p], env.trick_pile)
            else:
                action = opponent_strategy(candidates)
        _, _done = env.step(action)

    if env.winner == 0:
        score = sum(len(env.hands[p]) for p in range(n_players) if p != 0)
        cards_remaining = 0
    else:
        cards_remaining = len(env.hands[0])
        score = -cards_remaining
    return env.winner, starting_position, cards_remaining, score


def evaluate_against_greedy_q(current_qnet: SetPoolQNetwork, n_players: int, num_games: int = 500, device: str = "cpu"):
    """Lightweight evaluation (matches logging needs for checkpoint schedule)."""
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

    for _ in range(num_games):
        winner, starting_pos, cards_remaining, score = play_evaluation_game_q(
            current_qnet, n_players, opponent_strategy=greedy_strategy, epsilon=0.0
        )
        if winner == 0:
            wins_vs_greedy += 1
            wins_by_position[starting_pos] = wins_by_position.get(starting_pos, 0) + 1
        else:
            cards_remaining_sum += cards_remaining
            losses_count += 1
        score_sum_greedy += score
        games_by_position[starting_pos] = games_by_position.get(starting_pos, 0) + 1

    for _ in range(num_games):
        winner, _, _, score = play_evaluation_game_q(
            current_qnet, n_players, opponent_strategy=random_strategy, epsilon=0.0
        )
        if winner == 0:
            wins_vs_random += 1
        score_sum_random += score

    for _ in range(num_games):
        winner, _, _, score = play_evaluation_game_q(
            current_qnet, n_players, opponent_strategy=smart_strategy, epsilon=0.0
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

    return {
        "win_rate_vs_greedy": win_rate_vs_greedy,
        "win_rate_vs_random": win_rate_vs_random,
        "win_rate_vs_smart": win_rate_vs_smart,
        "avg_cards_remaining_when_losing": avg_cards_remaining,
        "avg_score_vs_greedy": avg_score_vs_greedy,
        "avg_score_vs_random": avg_score_vs_random,
        "avg_score_vs_smart": avg_score_vs_smart,
        "win_rate_by_starting_position": win_rate_by_position,
        "total_games": num_games,
    }


@dataclass
class OpponentMix:
    greedy_weight: float
    smart_weight: float
    current_weight: float
    checkpoint_weight: float
    random_weight: float

    def get_weights_and_strategies(self) -> tuple[list[float], list[str]]:
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


class QCheckpointManager:
    STAGE_0 = OpponentMix(
        greedy_weight=0.40, smart_weight=0.10, current_weight=0.25, checkpoint_weight=0.0, random_weight=0.25
    )
    STAGE_1 = OpponentMix(
        greedy_weight=0.30, smart_weight=0.15, current_weight=0.35, checkpoint_weight=0.05, random_weight=0.15
    )
    STAGE_2 = OpponentMix(
        greedy_weight=0.15, smart_weight=0.30, current_weight=0.35, checkpoint_weight=0.10, random_weight=0.10
    )
    STAGE_3 = OpponentMix(
        greedy_weight=0.10, smart_weight=0.30, current_weight=0.40, checkpoint_weight=0.15, random_weight=0.05
    )

    def __init__(self, device: str = "cpu", n_players: int = 4):
        self.device = device
        self.n_players = n_players
        self.checkpoints: list[tuple[int, SetPoolQNetwork]] = []
        self.max_checkpoints = 20
        self.greedy_schedule_stage = 0

    def add_checkpoint(self, step: int, qnet: SetPoolQNetwork):
        checkpoint_qnet = SetPoolQNetwork(n_players=self.n_players, device=self.device).to(self.device)
        checkpoint_qnet.load_state_dict(qnet.state_dict())
        checkpoint_qnet.eval()
        self.checkpoints.append((step, checkpoint_qnet))
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def update_greedy_schedule(self, win_rate: float):
        if win_rate >= 0.40 and self.greedy_schedule_stage < 3:
            self.greedy_schedule_stage = 3
        if win_rate >= 0.35 and self.greedy_schedule_stage < 2:
            self.greedy_schedule_stage = 2
        elif win_rate >= 0.2 and self.greedy_schedule_stage < 1:
            self.greedy_schedule_stage = 1

    def sample_opponent_strategy(self, current_qnet: SetPoolQNetwork) -> SetPoolQNetwork | Callable:
        opponent_mix = [self.STAGE_0, self.STAGE_1, self.STAGE_2, self.STAGE_3][self.greedy_schedule_stage]
        weights, strategies = opponent_mix.get_weights_and_strategies()
        chosen = np.random.choice(strategies, p=weights)
        if chosen == "greedy":
            return greedy_strategy
        if chosen == "smart":
            return smart_strategy
        if chosen == "current":
            return current_qnet
        if chosen == "checkpoint":
            if len(self.checkpoints) > 0:
                _, ckpt = random.choice(self.checkpoints[-10:])
                return ckpt
            return random_strategy
        return random_strategy


def collect_q_trajectories(
    n_players: int,
    qnet: SetPoolQNetwork,
    episodes_per_batch: int,
    opponent_strategies_by_episode: list[dict[int, SetPoolQNetwork | Callable]],
    model_seats: list[int],
    epsilon: float,
    step_penalty: float,
) -> dict[int, list[QStepRecord]]:
    trajectories: dict[int, list[QStepRecord]] = {p: [] for p in range(n_players)}
    env = Big2Env(n_players)

    for ep_idx in range(episodes_per_batch):
        model_seat = model_seats[ep_idx]
        opponent_strategies = opponent_strategies_by_episode[ep_idx]
        state = env.reset()
        episode_traj: list[QStepRecord] = []

        while True:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]

            if p != model_seat:
                strategy = opponent_strategies.get(p, greedy_strategy)
                if isinstance(strategy, SetPoolQNetwork):
                    action = select_action_q_greedy(strategy, state, candidates)
                elif strategy == smart_strategy:
                    action = strategy(candidates, env.hands[p], env.trick_pile)
                elif strategy == greedy_strategy or strategy == random_strategy:
                    action = strategy(candidates)
                else:
                    action = greedy_strategy(candidates)
                next_state, done = env.step(action)
            else:
                action, action_idx = select_action_q_epsilon_greedy(qnet, state, candidates, epsilon=epsilon)
                next_state, done = env.step(action)
                episode_traj.append(
                    QStepRecord(
                        state=state.copy(),
                        action_idx=action_idx,
                        candidates=candidates,
                        reward=step_penalty,
                        done=False,
                    )
                )

            if done:
                if episode_traj:
                    final_r = 1.0 if env.winner == model_seat else -1.0
                    episode_traj[-1].reward += final_r
                    episode_traj[-1].done = True
                    trajectories[model_seat].extend(episode_traj)
                break

            state = next_state

    return trajectories


def sarsa_1step_update(
    qnet: SetPoolQNetwork,
    qnet_target: SetPoolQNetwork,
    trajectories: dict[int, list[QStepRecord]],
    optimizer: optim.Optimizer,
    gamma: float,
    device: str,
) -> float:
    # Split into episodes using done flags
    episodes: list[list[QStepRecord]] = []
    for _, steps in trajectories.items():
        if not steps:
            continue
        start = 0
        for i, rec in enumerate(steps):
            if rec.done:
                episodes.append(steps[start : i + 1])
                start = i + 1

    if not episodes:
        return 0.0

    # Flatten training inputs in (episode order, step order)
    states: list[np.ndarray] = []
    actions_batch: list[list[np.ndarray]] = []
    chosen_indices: list[int] = []
    for ep in episodes:
        for r in ep:
            states.append(r.state)
            actions_batch.append([combo_to_action_vector(c) for c in r.candidates])
            chosen_indices.append(r.action_idx)

    # Compute bootstrap values for all non-terminal steps (1-step SARSA).
    # Since our "time steps" are model-decision points, s_{t+1} is the next time
    # the model acts (after opponents have taken their turns), and a_{t+1} is the
    # action actually taken there by the behavior policy (epsilon-greedy).
    next_states: list[np.ndarray] = []
    next_actions_batch: list[list[np.ndarray]] = []
    next_chosen_indices: list[int] = []
    for ep in episodes:
        for t in range(len(ep) - 1):
            nxt = ep[t + 1]
            next_states.append(nxt.state)
            next_actions_batch.append([combo_to_action_vector(c) for c in nxt.candidates])
            next_chosen_indices.append(nxt.action_idx)

    q_bootstrap_vals: list[float] = []
    if next_states:
        with torch.no_grad():
            next_state_tensor = torch.from_numpy(np.array(next_states)).long().to(device)
            q_targ_list = qnet_target(next_state_tensor, next_actions_batch)
            q_bootstrap_vals = [
                float(q_targ_list[i][next_chosen_indices[i]].item()) for i in range(len(next_chosen_indices))
            ]

    # Build 1-step SARSA targets per episode:
    # y_T = r_T
    # y_t = r_t + gamma * Q_target(s_{t+1}, a_{t+1})
    targets: list[float] = []
    bs_ptr = 0
    for ep in episodes:
        T = len(ep)
        ep_targets = [0.0] * T
        ep_targets[-1] = float(ep[-1].reward)
        for t in range(T - 2, -1, -1):
            bootstrap = q_bootstrap_vals[bs_ptr + t] if next_states else 0.0
            ep_targets[t] = float(ep[t].reward) + gamma * bootstrap
        bs_ptr += max(T - 1, 0)
        targets.extend(ep_targets)

    state_tensor = torch.from_numpy(np.array(states)).long().to(device)
    q_list = qnet(state_tensor, actions_batch)
    chosen_q = torch.stack([q_list[i][chosen_indices[i]] for i in range(len(chosen_indices))], dim=0)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)

    loss = F.mse_loss(chosen_q, target_t)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def train_q_network(
    n_players: int = 4,
    batches: int = 500,
    episodes_per_batch: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 42,
    device: str = "cpu",
    eval_interval: int = 25,
    eval_games: int = 400,
    epsilon_start: float = 0.25,
    epsilon_end: float = 0.05,
    step_penalty: float = -0.01,
    target_update_interval: int = 10,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    qnet = SetPoolQNetwork(n_players=n_players, device=device).to(device)
    qnet_target = SetPoolQNetwork(n_players=n_players, device=device).to(device)
    qnet_target.load_state_dict(qnet.state_dict())
    qnet_target.eval()
    optimizer = optim.Adam(qnet.parameters(), lr=lr)

    # Learning rate scheduler: warmup + cosine annealing (with a floor), matching other training scripts.
    warmup_batches = max(1, batches // 10)  # 10% warmup

    def lr_lambda(current_batch: int) -> float:
        # current_batch is 0-indexed internal counter in LambdaLR
        min_lr_ratio = 0.2  # LR won't drop below 20% of initial
        if current_batch < warmup_batches:
            return current_batch / warmup_batches
        progress = (current_batch - warmup_batches) / max(1, (batches - warmup_batches))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_manager = QCheckpointManager(device=device, n_players=n_players)

    loss_history: list[float] = []
    q_loss_history: list[float] = []
    exploration_history: list[float] = []
    eval_episodes: list[int] = []
    win_rates: list[float] = []

    all_players = set(range(n_players))

    for batch in tqdm(range(1, batches + 1)):
        frac = batch / batches
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        base_seats = np.arange(episodes_per_batch) % n_players
        perm = np.random.permutation(episodes_per_batch)
        model_seats = [int(base_seats[i]) for i in perm]

        opponent_strategies_by_episode: list[dict[int, SetPoolQNetwork | Callable]] = []
        for env_idx in range(episodes_per_batch):
            model_seat = model_seats[env_idx]
            opp_map: dict[int, SetPoolQNetwork | Callable] = {}
            for p in all_players:
                if p == model_seat:
                    continue
                opp_map[p] = checkpoint_manager.sample_opponent_strategy(qnet)
            opponent_strategies_by_episode.append(opp_map)

        trajectories = collect_q_trajectories(
            n_players=n_players,
            qnet=qnet,
            episodes_per_batch=episodes_per_batch,
            opponent_strategies_by_episode=opponent_strategies_by_episode,
            model_seats=model_seats,
            epsilon=epsilon,
            step_penalty=step_penalty,
        )

        q_loss = sarsa_1step_update(
            qnet,
            qnet_target,
            trajectories,
            optimizer,
            gamma=gamma,
            device=device,
        )

        # Step the learning rate scheduler once per batch.
        scheduler.step()

        if target_update_interval > 0 and batch % target_update_interval == 0:
            qnet_target.load_state_dict(qnet.state_dict())
            qnet_target.eval()

        loss_history.append(q_loss)
        q_loss_history.append(q_loss)
        exploration_history.append(epsilon)

        if batch % eval_interval == 0:
            print(f"\n[Step {batch}] Evaluating greedy-Q policy...")
            qnet.eval()
            metrics = evaluate_against_greedy_q(qnet, n_players, num_games=eval_games, device=device)
            qnet.train()

            eval_episodes.append(batch)
            win_rates.append(metrics["win_rate_vs_greedy"])

            checkpoint_manager.update_greedy_schedule(metrics["win_rate_vs_greedy"])

            print(f"[Step {batch}] Evaluation Results ({metrics['total_games']} games):")
            wins_vs_greedy = int(metrics["win_rate_vs_greedy"] * metrics["total_games"])
            wr_str = f"  Win rate vs greedy: {metrics['win_rate_vs_greedy']:.2%}"
            wr_str += f" ({wins_vs_greedy}/{metrics['total_games']} wins)"
            print(wr_str)
            print(f"  Win rate vs random: {metrics['win_rate_vs_random']:.2%}")
            print(f"  Avg score vs greedy: {metrics['avg_score_vs_greedy']:.2f}")
            print(f"  Avg score vs random: {metrics['avg_score_vs_random']:.2f}")
            wins_vs_smart = int(metrics["win_rate_vs_smart"] * metrics["total_games"])
            wr_smart_str = f"  Win rate vs smart: {metrics['win_rate_vs_smart']:.2%}"
            wr_smart_str += f" ({wins_vs_smart}/{metrics['total_games']} wins)"
            print(wr_smart_str)
            print(f"  Avg score vs smart: {metrics['avg_score_vs_smart']:.2f}")
            print(f"  Avg cards remaining when losing: {metrics['avg_cards_remaining_when_losing']:.2f}")
            print("  Win rate by starting position:")
            for pos, wr in sorted(metrics["win_rate_by_starting_position"].items()):
                print(f"    Position {pos}: {wr:.2%}")
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Step {batch}] q_loss={q_loss:.4f} epsilon={epsilon:.3f} lr={current_lr:.6f}")
            n_checkpoints = len(checkpoint_manager.checkpoints)
            stage = checkpoint_manager.greedy_schedule_stage
            print(f"[Step {batch}] Checkpoints in pool: {n_checkpoints}, Greedy schedule stage: {stage}\n")

            save_path = f"big2_q_model_step_{batch}.pt"
            torch.save(qnet.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")
            checkpoint_manager.add_checkpoint(batch, qnet)

    return qnet, loss_history, q_loss_history, exploration_history, eval_episodes, win_rates


def win_rate_of_starting_hand_q(
    qnet: SetPoolQNetwork, hand: list[int], n_players: int = 4, sims: int = 512, device: str = "cpu"
) -> float:
    """Monte Carlo estimate of win rate for seat 0 given a fixed starting hand, using greedy-Q self-play."""
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

        state = env._obs(env.current_player)
        while True:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            action = select_action_q_greedy(qnet, state, candidates)
            next_state, done = env.step(action)
            if done:
                wins += 1.0 if env.winner == 0 else 0.0
                break
            state = next_state
    return wins / sims


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    n_players = 4
    cards_per_player = 52 // n_players

    qnet, loss_history, q_loss_history, exploration_history, eval_episodes, win_rates = train_q_network(
        n_players=n_players,
        batches=10000,
        episodes_per_batch=64,
        lr=8e-4,
        gamma=0.99,
        seed=42,
        device=device,
        eval_interval=500,
        eval_games=400,
        epsilon_start=0.5,
        epsilon_end=0.01,
        step_penalty=-0.01,
    )

    hand = sorted([48, 49, 50, 51, 47, 43, 39, 35, 31, 27, 26, 1, 0][:cards_per_player])
    val = win_rate_of_starting_hand_q(qnet, hand, n_players=n_players, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Win rate:", val)

    hand = sorted(random.sample(range(52), cards_per_player))
    val = win_rate_of_starting_hand_q(qnet, hand, n_players=n_players, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Win rate:", val)

    save_path = "big2_q_model.pt"
    torch.save(qnet.state_dict(), save_path)
    print(f"\nModel weights saved to {save_path}")

    plot_training_curves(
        loss_history,
        q_loss_history,
        q_loss_history,
        exploration_history,
        None,
        eval_episodes,
        win_rates,
        save_path="training_curves_q.png",
    )
