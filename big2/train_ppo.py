import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from big2.nn import MLPPolicy, combo_to_action_vector
from big2.simulator.cards import PASS, Combo, card_name
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy
from big2.train_helpers import (
    CheckpointManager,
    compute_gae_from_values,
    evaluate_against_greedy,
    plot_training_curves,
    random_strategy,
    select_action_greedy,
    value_of_starting_hand,
)


@dataclass
class PPOStepRecord:
    """PPO-specific step record storing state and action for recomputing logprobs."""

    state: np.ndarray
    action_idx: int
    candidates: list[Combo]
    old_logprob: torch.Tensor
    reward: float
    value: torch.Tensor


def collect_ppo_trajectories(
    env: Big2Env,
    policy: MLPPolicy,
    episodes_per_batch: int,
    opponent_strategies: dict[int, MLPPolicy | Callable],
    model_players: set[int],
    device: str,
) -> dict[int, list[PPOStepRecord]]:
    """
    Collect trajectories for PPO training.
    Stores states, actions, and old logprobs for later recomputation.
    """
    trajectories: dict[int, list[PPOStepRecord]] = {p: [] for p in model_players}

    for _ in range(episodes_per_batch):
        state = env.reset()
        episode_trajs: dict[int, list[PPOStepRecord]] = {p: [] for p in model_players}

        while True:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]

            if p in opponent_strategies:
                # Opponent player: use provided strategy
                strategy = opponent_strategies[p]
                if isinstance(strategy, MLPPolicy):
                    action = select_action_greedy(strategy, state, candidates)
                elif strategy == smart_strategy:
                    # Smart strategy needs hand and trick_pile
                    action = strategy(candidates, env.hands[p], env.trick_pile)
                elif strategy == greedy_strategy or strategy == random_strategy:
                    action = strategy(candidates)
                else:
                    action = greedy_strategy(candidates)
                next_state, done = env.step(action)
            else:
                # Model player: sample action and store trajectory data
                if p in model_players:
                    # Get policy output
                    st = torch.from_numpy(state[np.newaxis, :]).long().to(device)
                    action_feats = [[combo_to_action_vector(c) for c in candidates]]
                    logits_list, values = policy(st, action_feats)
                    logits = logits_list[0]
                    probs = F.softmax(logits, dim=0)
                    dist = torch.distributions.Categorical(probs=probs)
                    idx = dist.sample()
                    old_logprob = dist.log_prob(idx)
                    value = values[0]

                    action = candidates[int(idx.item())]
                    next_state, done = env.step(action)
                    reward = -0.001  # Small step penalty to encourage finishing quickly
                    episode_trajs[p].append(
                        PPOStepRecord(
                            state=state.copy(),
                            action_idx=int(idx.item()),
                            candidates=candidates,
                            old_logprob=old_logprob.detach(),
                            reward=reward,
                            value=value.detach(),
                        )
                    )
                else:
                    # Not a model player, just step
                    action = greedy_strategy(candidates)
                    next_state, done = env.step(action)

            if done:
                # Assign terminal rewards
                winner = env.winner
                for q in model_players:
                    if len(episode_trajs[q]) > 0:
                        if q == winner:
                            final_r = 1.0
                        else:
                            final_r = -len(env.hands[q]) / (52 // env.n_players)
                        # Add terminal reward on top of intermediate step penalty
                        episode_trajs[q][-1].reward += final_r
                        # Append to full trajectories
                        trajectories[q].extend(episode_trajs[q])
                break

            state = next_state

    return trajectories


def collect_ppo_trajectories_batched(
    n_players: int,
    policy: MLPPolicy,
    episodes_per_batch: int,
    opponent_strategies: dict[int, MLPPolicy | Callable],
    model_players: set[int],
    device: str,
) -> dict[int, list[PPOStepRecord]]:
    """
    Collect trajectories for PPO training with batched forward passes.
    Runs multiple environments in parallel for improved efficiency.

    Args:
        n_players: Number of players in the game
        policy: The policy network
        episodes_per_batch: Number of episodes to collect
        opponent_strategies: Mapping of opponent player IDs to their strategies
        model_players: Set of player IDs controlled by the model
        device: Device to run on

    Returns:
        Dictionary mapping model player IDs to lists of step records
    """
    trajectories: dict[int, list[PPOStepRecord]] = {p: [] for p in model_players}

    # Create multiple environments to run in parallel
    envs = [Big2Env(n_players) for _ in range(episodes_per_batch)]
    states = [env.reset() for env in envs]
    episode_trajs: list[dict[int, list[PPOStepRecord]]] = [
        {p: [] for p in model_players} for _ in range(episodes_per_batch)
    ]
    active_envs = list(range(episodes_per_batch))

    while active_envs:
        # Collect all environments and their current state
        env_actions = {}  # Maps env_idx to action to take

        # First, collect info for all active environments
        env_info = []  # List of (env_idx, env, p, candidates)
        for env_idx in active_envs:
            env = envs[env_idx]
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            env_info.append((env_idx, env, p, candidates))

        # Batch process model players
        batch_states = []
        batch_candidates = []
        batch_env_indices = []
        batch_players = []

        for env_idx, _, p, candidates in env_info:
            if p in model_players:
                batch_states.append(states[env_idx])
                batch_candidates.append(candidates)
                batch_env_indices.append(env_idx)
                batch_players.append(p)

        # Batch forward pass for all model players
        if batch_states:
            state_tensor = torch.from_numpy(np.array(batch_states)).long().to(device)
            action_feats_batch = [[combo_to_action_vector(c) for c in cands] for cands in batch_candidates]

            with torch.no_grad():
                logits_list, values = policy(state_tensor, action_feats_batch)

            # Store results for each environment
            for i, env_idx in enumerate(batch_env_indices):
                logits = logits_list[i]
                probs = F.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs=probs)
                idx = dist.sample()
                old_logprob = dist.log_prob(idx)
                value = values[i]

                action_idx = int(idx.item())
                action = batch_candidates[i][action_idx]

                # Store trajectory record
                episode_trajs[env_idx][batch_players[i]].append(
                    PPOStepRecord(
                        state=batch_states[i].copy(),
                        action_idx=action_idx,
                        candidates=batch_candidates[i],
                        old_logprob=old_logprob.detach(),
                        reward=-0.001,  # Small step penalty to encourage finishing quickly
                        value=value.detach(),
                    )
                )

                env_actions[env_idx] = action

        # Process opponent players
        for env_idx, env, p, candidates in env_info:
            if env_idx not in env_actions:  # Not already processed as model player
                if p in opponent_strategies:
                    strategy = opponent_strategies[p]
                    if isinstance(strategy, MLPPolicy):
                        action = select_action_greedy(strategy, states[env_idx], candidates)
                    elif strategy == smart_strategy:
                        action = strategy(candidates, env.hands[p], env.trick_pile)
                    elif strategy == greedy_strategy or strategy == random_strategy:
                        action = strategy(candidates)
                    else:
                        action = greedy_strategy(candidates)
                else:
                    action = greedy_strategy(candidates)
                env_actions[env_idx] = action

        # Step all active environments
        next_active_envs = []
        for env_idx in active_envs:
            env = envs[env_idx]
            action = env_actions[env_idx]
            next_state, done = env.step(action)

            if done:
                # Assign terminal rewards
                winner = env.winner
                for q in model_players:
                    if len(episode_trajs[env_idx][q]) > 0:
                        if q == winner:
                            final_r = 1.0
                        else:
                            final_r = -len(env.hands[q]) / (52 // env.n_players)
                        # Add terminal reward on top of intermediate step penalty
                        episode_trajs[env_idx][q][-1].reward += final_r
                        # Append to full trajectories
                        trajectories[q].extend(episode_trajs[env_idx][q])
            else:
                states[env_idx] = next_state
                next_active_envs.append(env_idx)

        active_envs = next_active_envs

    return trajectories


def ppo_update(
    policy: MLPPolicy,
    trajectories: dict[int, list[PPOStepRecord]],
    optimizer: optim.Optimizer,
    ppo_epochs: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_beta: float,
    mini_batch_size: int | None,
    device: str,
    gamma: float,
    lam: float,
) -> tuple[float, float, float, float]:
    """
    Perform PPO update with clipped surrogate objective.

    Returns:
        (policy_loss, value_loss, entropy, total_loss)
    """
    # Flatten trajectories into lists
    all_states = []
    all_action_indices = []
    all_candidates = []
    all_old_logprobs = []
    all_rewards = []
    all_values = []

    for p in trajectories:
        for record in trajectories[p]:
            all_states.append(record.state)
            all_action_indices.append(record.action_idx)
            all_candidates.append(record.candidates)
            all_old_logprobs.append(record.old_logprob)
            all_rewards.append(record.reward)
            all_values.append(record.value)

    if len(all_states) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Convert to tensors
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    old_logprobs_t = torch.stack(all_old_logprobs).to(device)
    old_values_t = torch.stack(all_values).squeeze(-1).to(device)

    # Compute advantages and returns once
    advantages, returns = compute_gae_from_values(rewards_t, old_values_t, gamma=gamma, lam=lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Prepare data for mini-batching
    n_samples = len(all_states)
    if mini_batch_size is None:
        mini_batch_size = n_samples

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    # Multiple epochs over the same data
    for _epoch in range(ppo_epochs):
        # Shuffle indices for mini-batching
        indices = torch.randperm(n_samples, device=device)

        for start_idx in range(0, n_samples, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Get batch data
            batch_states = [all_states[i] for i in batch_indices.cpu().numpy()]
            batch_action_indices = [all_action_indices[i] for i in batch_indices.cpu().numpy()]
            batch_candidates = [all_candidates[i] for i in batch_indices.cpu().numpy()]
            batch_old_logprobs = old_logprobs_t[batch_indices]
            batch_old_values = old_values_t[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Recompute logprobs and values with current policy
            state_tensor = torch.from_numpy(np.array(batch_states)).long().to(device)
            action_feats = [[combo_to_action_vector(c) for c in candidates] for candidates in batch_candidates]
            logits_list, values = policy(state_tensor, action_feats)

            # Extract logprobs for chosen actions
            new_logprobs_list = []
            entropies_list = []
            values_list = []

            for i, (logits, action_idx) in enumerate(zip(logits_list, batch_action_indices, strict=True)):
                probs = F.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs=probs)
                new_logprob = dist.log_prob(torch.tensor(action_idx, device=device))
                entropy = dist.entropy()
                new_logprobs_list.append(new_logprob)
                entropies_list.append(entropy)
                values_list.append(values[i])

            new_logprobs = torch.stack(new_logprobs_list)
            entropies = torch.stack(entropies_list)
            new_values = torch.stack(values_list)

            # Compute ratio and clipped surrogate objective
            ratio = torch.exp(new_logprobs - batch_old_logprobs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss with clipping
            # Ensure both tensors are 1D (handle edge case where batch_size=1)
            if new_values.dim() == 0:
                new_values = new_values.unsqueeze(0)
            if batch_returns.dim() == 0:
                batch_returns = batch_returns.unsqueeze(0)
            if batch_old_values.dim() == 0:
                batch_old_values = batch_old_values.unsqueeze(0)
            value_pred_clipped = batch_old_values + torch.clamp(
                new_values - batch_old_values, -clip_epsilon, clip_epsilon
            )
            value_losses_clipped = F.mse_loss(value_pred_clipped, batch_returns, reduction="none")
            value_losses_unclipped = F.mse_loss(new_values, batch_returns, reduction="none")
            value_loss = torch.max(value_losses_clipped, value_losses_unclipped).mean()

            # Entropy bonus
            entropy = entropies.mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_beta * entropy

            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return (
        total_policy_loss / n_updates if n_updates > 0 else 0.0,
        total_value_loss / n_updates if n_updates > 0 else 0.0,
        total_entropy / n_updates if n_updates > 0 else 0.0,
        (total_policy_loss + value_coef * total_value_loss - entropy_beta * total_entropy) / n_updates
        if n_updates > 0
        else 0.0,
    )


def train_ppo(
    n_players=4,
    batches=2000,
    episodes_per_batch=10,
    ppo_epochs=4,
    clip_epsilon=0.2,
    lr=3e-4,
    entropy_beta=0.01,
    value_coef=0.5,
    gamma=0.999,
    lam=0.95,
    seed=42,
    device="cpu",
    eval_interval=50,
    eval_games=500,
    mini_batch_size=None,
    use_batched_collection=True,
):
    """
    Train using Proximal Policy Optimization (PPO).

    Args:
        n_players: Number of players
        batches: Number of training batches
        episodes_per_batch: Episodes to collect per batch
        ppo_epochs: Number of PPO update epochs per batch
        clip_epsilon: PPO clipping parameter
        lr: Learning rate
        entropy_beta: Entropy coefficient
        value_coef: Value loss coefficient
        gamma: Discount factor
        lam: GAE lambda parameter
        seed: Random seed
        device: Device to use
        eval_interval: Evaluation interval
        eval_games: Number of games for evaluation
        mini_batch_size: Mini-batch size (None = use full batch)
        use_batched_collection: Whether to use batched trajectory collection (parallel envs)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Model players are always player 0, opponents are players 1 to n_players-1
    model_players = {0}
    opponent_players = set(range(1, n_players))

    print(f"Model players: {sorted(model_players)}, Opponent players: {sorted(opponent_players)}")
    print(f"PPO parameters: epochs={ppo_epochs}, clip_epsilon={clip_epsilon}")
    print(
        "Opponent mixture: Adaptive based on win rate schedule "
        "(Stage 0: 20% greedy/35% smart, Stage 1: 10% greedy/25% smart, Stage 2: 5% greedy/20% smart)"
    )

    env = Big2Env(n_players)
    policy = MLPPolicy(n_players=n_players, device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Checkpoint manager for opponent sampling
    checkpoint_manager = CheckpointManager(device=device, n_players=n_players)

    # Learning rate scheduler: warmup + cosine annealing
    warmup_batches = batches // 10  # 10% warmup

    def lr_lambda(current_batch):
        min_lr_ratio = 0.5  # LR won't drop below 50% of initial
        if current_batch < warmup_batches:
            # Linear warmup
            return current_batch / warmup_batches
        else:
            # Cosine annealing after warmup (with floor)
            progress = (current_batch - warmup_batches) / (batches - warmup_batches)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Adaptive entropy coefficient
    current_entropy_beta = entropy_beta
    target_entropy = 0.7  # Target entropy level
    entropy_beta_min = 0.05
    entropy_beta_max = 0.2

    # Track losses for plotting
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []

    # Track evaluation metrics
    eval_episodes = []
    win_rates = []

    for batch in tqdm(range(1, batches + 1)):
        # Collect trajectories
        opponent_strategies = dict.fromkeys(opponent_players)
        for opp_id in opponent_players:
            opponent_strategies[opp_id] = checkpoint_manager.sample_opponent_policy(policy)

        if use_batched_collection:
            trajectories = collect_ppo_trajectories_batched(
                n_players, policy, episodes_per_batch, opponent_strategies, model_players, device
            )
        else:
            trajectories = collect_ppo_trajectories(
                env, policy, episodes_per_batch, opponent_strategies, model_players, device
            )

        # PPO update
        policy_loss, value_loss, entropy, total_loss = ppo_update(
            policy,
            trajectories,
            optimizer,
            ppo_epochs,
            clip_epsilon,
            value_coef,
            current_entropy_beta,
            mini_batch_size,
            device,
            gamma=gamma,
            lam=lam,
        )

        # Step the learning rate scheduler
        scheduler.step()

        # Adaptive entropy coefficient: adjust based on current entropy
        if entropy < target_entropy - 0.15:
            # Entropy too low, encourage more exploration
            current_entropy_beta = min(entropy_beta_max, current_entropy_beta * 1.01)
        elif entropy > target_entropy + 0.15:
            # Entropy too high, allow more exploitation
            current_entropy_beta = max(entropy_beta_min, current_entropy_beta * 0.99)

        # Record losses
        loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(value_loss)
        entropy_history.append(entropy)

        # Evaluation every eval_interval batches
        if batch % eval_interval == 0:
            avg_len = sum(len(trajectories[p]) for p in model_players) / len(model_players) if model_players else 0
            avg_steps_per_episode = avg_len / episodes_per_batch if episodes_per_batch > 0 else 0

            # Comprehensive evaluation
            print(f"\n[Step {batch}] Evaluating policy...")
            policy.eval()
            metrics = evaluate_against_greedy(policy, n_players, num_games=eval_games, device=device)
            policy.train()

            eval_episodes.append(batch)
            win_rates.append(metrics.win_rate_vs_greedy)

            # Update greedy schedule based on win rate
            checkpoint_manager.update_greedy_schedule(metrics.win_rate_vs_greedy)

            print(f"[Step {batch}] Evaluation Results ({metrics.total_games} games):")
            wins_vs_greedy = int(metrics.win_rate_vs_greedy * metrics.total_games)
            wr_str = f"  Win rate vs greedy: {metrics.win_rate_vs_greedy:.2%}"
            wr_str += f" ({wins_vs_greedy}/{metrics.total_games} wins)"
            print(wr_str)
            print(f"  Win rate vs random: {metrics.win_rate_vs_random:.2%}")
            wins_vs_smart = int(metrics.win_rate_vs_smart * metrics.total_games)
            wr_smart_str = f"  Win rate vs smart: {metrics.win_rate_vs_smart:.2%}"
            wr_smart_str += f" ({wins_vs_smart}/{metrics.total_games} wins)"
            print(wr_smart_str)
            print(f"  Avg cards remaining when losing: {metrics.avg_cards_remaining_when_losing:.2f}")
            print("  Win rate by starting position:")
            for pos, wr in sorted(metrics.win_rate_by_starting_position.items()):
                print(f"    Position {pos}: {wr:.2%}")
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {batch}] loss={total_loss:.3f} pol={policy_loss:.3f} "
                f"val={value_loss:.3f} ent={entropy:.3f} steps/player={avg_len:.1f} "
                f"({episodes_per_batch} episodes, ~{avg_steps_per_episode:.1f} steps/episode)"
            )
            print(f"[Step {batch}] lr={current_lr:.6f} ent_beta={current_entropy_beta:.4f}")
            n_checkpoints = len(checkpoint_manager.checkpoints)
            stage = checkpoint_manager.greedy_schedule_stage
            greedy_pct = {0: "20%", 1: "10%", 2: "5%"}[stage]
            print(
                f"[Step {batch}] Checkpoints in pool: {n_checkpoints}, "
                f"Greedy schedule stage: {stage} ({greedy_pct} greedy)\n"
            )

            # Save checkpoint and add to manager
            save_path = f"big2_model_step_{batch}.pt"
            torch.save(policy.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

            checkpoint_manager.add_checkpoint(batch, policy)

    return (
        policy,
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        eval_episodes,
        win_rates,
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    n_players = 4
    cards_per_player = 52 // n_players

    (
        policy,
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        eval_episodes,
        win_rates,
    ) = train_ppo(
        n_players=n_players,
        batches=200,
        episodes_per_batch=64,
        ppo_epochs=3,
        clip_epsilon=0.2,
        lr=3e-4,
        entropy_beta=0.05,
        value_coef=0.5,
        gamma=0.999,
        lam=0.95,
        seed=42,
        device=device,
        eval_interval=25,
        eval_games=250,
        mini_batch_size=256,
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
        None,  # max_logprob_history not tracked in PPO
        eval_episodes,
        win_rates,
    )
