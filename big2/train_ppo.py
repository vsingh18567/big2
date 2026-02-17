import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from big2.nn import combo_to_action_vector, make_policy
from big2.simulator.cards import PASS, Combo, card_name
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy
from big2.train_helpers import (
    CheckpointManager,
    compute_gae_from_values,
    evaluate_policy,
    plot_training_curves,
    random_strategy,
    safe_categorical_from_logits,
    sample_action_from_policy,
    select_action_sampled,
    value_of_starting_hand,
)
from big2.train_ppo_config import PPOConfig, dump_training_run
from big2.training_monitor import TrainingMonitor


@dataclass
class PPOStepRecord:
    """PPO-specific step record storing state and action for recomputing logprobs."""

    state: np.ndarray
    action_idx: int
    candidates: list[Combo]
    old_logprob: torch.Tensor
    reward: float
    value: torch.Tensor
    n_cards_before: int | None = None
    done: bool = False


def _shaped_step_reward(
    *,
    action: Combo,
    candidates: list[Combo],
    n_cards_before: int,
    n_cards_after: int,
    cards_per_player: int,
    step_penalty: float,
    progress_reward_coef: float,
    pass_penalty: float,
) -> float:
    """Compute dense reward shaping for PPO.

    - Encourages shedding cards (normalized by cards_per_player)
    - Discourages passing when a non-pass play exists
    - Keeps a small step penalty to prefer shorter games
    """
    # Progress: cards removed from hand by the action (>= 0)
    cards_played = max(0, n_cards_before - n_cards_after)
    progress = (cards_played / max(1, cards_per_player)) * progress_reward_coef

    # Passing when you could have played something (mild penalty)
    has_non_pass = any(c.type != PASS for c in candidates)
    pass_cost = pass_penalty if (action.type == PASS and has_non_pass) else 0.0

    return step_penalty + progress + pass_cost


def collect_ppo_trajectories(
    env: Big2Env,
    policy: nn.Module,
    episodes_per_batch: int,
    opponent_strategies_by_episode: list[dict[int, nn.Module | Callable]],
    model_seats_by_episode: list[list[int]],
    device: str,
    *,
    step_penalty: float = -0.0001,
    progress_reward_coef: float = 0.001,
    pass_penalty: float = 0,
) -> dict[int, list[PPOStepRecord]]:
    """
    Collect trajectories for PPO training.
    Stores states, actions, and old logprobs for later recomputation.
    """
    trajectories: dict[int, list[PPOStepRecord]] = {p: [] for p in range(env.n_players)}

    if len(model_seats_by_episode) != episodes_per_batch:
        raise ValueError("model_seats_by_episode must have length episodes_per_batch")
    if len(opponent_strategies_by_episode) != episodes_per_batch:
        raise ValueError("opponent_strategies_by_episode must have length episodes_per_batch")

    for ep_idx in range(episodes_per_batch):
        model_seats = set(model_seats_by_episode[ep_idx])
        if not model_seats:
            raise ValueError("At least one model seat is required per episode")
        state = env.reset()
        episode_trajs: dict[int, list[PPOStepRecord]] = {seat: [] for seat in model_seats}
        opponent_strategies = opponent_strategies_by_episode[ep_idx]

        while True:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]

            if p not in model_seats:
                # Opponent player: use provided strategy
                strategy = opponent_strategies.get(p, greedy_strategy)
                if isinstance(strategy, nn.Module):
                    action = select_action_sampled(strategy, state, candidates)
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
                # Get policy output
                idx, action, old_logprob, _entropy, value, _max_prob = sample_action_from_policy(
                    policy, state, candidates
                )
                n_before = len(env.hands[p])
                next_state, done = env.step(action)
                n_after = len(env.hands[p])
                reward = _shaped_step_reward(
                    action=action,
                    candidates=candidates,
                    n_cards_before=n_before,
                    n_cards_after=n_after,
                    cards_per_player=env.cards_per_player,
                    step_penalty=step_penalty,
                    progress_reward_coef=progress_reward_coef,
                    pass_penalty=pass_penalty,
                )
                episode_trajs[p].append(
                    PPOStepRecord(
                        state=state.copy(),
                        action_idx=idx,
                        candidates=candidates,
                        old_logprob=old_logprob.detach(),
                        reward=reward,
                        value=value.detach(),
                        n_cards_before=n_before,
                        done=False,
                    )
                )

            if done:
                # Assign terminal rewards
                winner = env.winner
                for model_seat in model_seats:
                    if len(episode_trajs[model_seat]) > 0:
                        if model_seat == winner:
                            max_cards = env.cards_per_player
                            opp_frac = sum(len(env.hands[i]) for i in range(env.n_players) if i != model_seat) / (
                                max_cards * (env.n_players - 1)
                            )
                            final_r = 1.0 + opp_frac
                        else:
                            # Scale terminal loss by remaining cards, but clamp to -1 once you're
                            # above half a starting hand. (In 4p: cards_per_player=13, half=6.5.)
                            cards_left_frac = len(env.hands[model_seat]) / env.cards_per_player
                            final_r = -1.0 - cards_left_frac
                        # Add terminal reward on top of intermediate step penalty
                        episode_trajs[model_seat][-1].reward += final_r
                        episode_trajs[model_seat][-1].done = True
                        # Append to full trajectories
                        trajectories[model_seat].extend(episode_trajs[model_seat])
                break

            state = next_state

    return trajectories


def collect_ppo_trajectories_batched(
    n_players: int,
    cards_per_player: int | None,
    policy: nn.Module,
    episodes_per_batch: int,
    opponent_strategies_by_env: list[dict[int, nn.Module | Callable]],
    model_seats_by_env: list[list[int]],
    device: str,
    *,
    step_penalty: float = -0.0001,
    progress_reward_coef: float = 0,
    pass_penalty: float = 0,
) -> dict[int, list[PPOStepRecord]]:
    """
    Collect trajectories for PPO training with batched forward passes.
    Runs multiple environments in parallel for improved efficiency.

    Args:
        n_players: Number of players in the game
        policy: The policy network
        episodes_per_batch: Number of episodes to collect
        opponent_strategies_by_env: Per-environment mapping of opponent player IDs to their strategies
        model_seats_by_env: Per-environment list of player IDs controlled by the model
        device: Device to run on

    Returns:
        Dictionary mapping model player IDs to lists of step records
    """
    trajectories: dict[int, list[PPOStepRecord]] = {p: [] for p in range(n_players)}

    # Create multiple environments to run in parallel
    envs = [Big2Env(n_players, cards_per_player=cards_per_player) for _ in range(episodes_per_batch)]
    states = [env.reset() for env in envs]
    if len(model_seats_by_env) != episodes_per_batch:
        raise ValueError("model_seats_by_env must have length episodes_per_batch")
    if len(opponent_strategies_by_env) != episodes_per_batch:
        raise ValueError("opponent_strategies_by_env must have length episodes_per_batch")

    episode_trajs: list[dict[int, list[PPOStepRecord]]] = [
        {seat: [] for seat in model_seats_by_env[env_idx]} for env_idx in range(episodes_per_batch)
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
            if not model_seats_by_env[env_idx]:
                raise ValueError("At least one model seat is required per environment")
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            env_info.append((env_idx, env, p, candidates))
        candidates_by_env_idx = {env_idx: candidates for env_idx, _env, _p, candidates in env_info}

        # Batch process model players
        batch_states = []
        batch_candidates = []
        batch_env_indices = []
        batch_players = []
        batch_n_cards_before: list[int] = []

        for env_idx, _, p, candidates in env_info:
            if p in model_seats_by_env[env_idx]:
                batch_states.append(states[env_idx])
                batch_candidates.append(candidates)
                batch_env_indices.append(env_idx)
                batch_players.append(p)
                batch_n_cards_before.append(len(envs[env_idx].hands[p]))

        # Batch forward pass for all model players
        if batch_states:
            state_tensor = torch.from_numpy(np.array(batch_states)).long().to(device)
            action_feats_batch = [
                [combo_to_action_vector(c, card_count=policy.card_universe_size) for c in cands]
                for cands in batch_candidates
            ]

            with torch.no_grad():
                logits_list, values = policy(state_tensor, action_feats_batch)

            # Store results for each environment
            for i, env_idx in enumerate(batch_env_indices):
                logits = logits_list[i]
                dist = safe_categorical_from_logits(logits, batch_candidates[i])
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
                        reward=0.0,  # filled after env.step once we know card delta
                        value=value.detach(),
                        n_cards_before=batch_n_cards_before[i],
                        done=False,
                    )
                )

                env_actions[env_idx] = action

        # Process opponent players
        for env_idx, env, p, candidates in env_info:
            if env_idx not in env_actions:  # Not already processed as model player
                strategy = opponent_strategies_by_env[env_idx].get(p, greedy_strategy)
                if isinstance(strategy, nn.Module):
                    action = select_action_sampled(strategy, states[env_idx], candidates)
                elif strategy == smart_strategy:
                    action = strategy(candidates, env.hands[p], env.trick_pile)
                elif strategy == greedy_strategy or strategy == random_strategy:
                    action = strategy(candidates)
                else:
                    action = greedy_strategy(candidates)
                env_actions[env_idx] = action

        # Step all active environments
        next_active_envs = []
        for env_idx in active_envs:
            env = envs[env_idx]
            action = env_actions[env_idx]
            actor = env.current_player
            # Only compute shaping for model-controlled players (we store trajectories only for them)
            if actor in model_seats_by_env[env_idx]:
                n_before = len(env.hands[actor])
            next_state, done = env.step(action)
            if actor in model_seats_by_env[env_idx]:
                n_after = len(env.hands[actor])
                candidates = candidates_by_env_idx[env_idx]
                shaped = _shaped_step_reward(
                    action=action,
                    candidates=candidates,
                    n_cards_before=n_before,
                    n_cards_after=n_after,
                    cards_per_player=env.cards_per_player,
                    step_penalty=step_penalty,
                    progress_reward_coef=progress_reward_coef,
                    pass_penalty=pass_penalty,
                )
                # Update the last stored record reward for this actor
                if episode_trajs[env_idx][actor]:
                    episode_trajs[env_idx][actor][-1].reward = shaped

            if done:
                winner = env.winner
                model_seats = model_seats_by_env[env_idx]

                # 1. Calculate the 'pot' of penalties from losers
                # Using a standard Big 2 linear penalty (1 point per card)
                total_penalty_pot = 0.0
                cards_per_player_local = env.cards_per_player

                for q in range(env.n_players):
                    if q != winner:
                        cards_left = len(env.hands[q])
                        # Normalize penalty to keep the scale near [-1, 1]
                        # e.g., Losing with all 13 cards = -1.0
                        penalty = cards_left / cards_per_player_local

                        # If the loser is a model-controlled seat, assign its reward
                        if q in model_seats and len(episode_trajs[env_idx][q]) > 0:
                            episode_trajs[env_idx][q][-1].reward = -penalty
                            episode_trajs[env_idx][q][-1].done = True
                            trajectories[q].extend(episode_trajs[env_idx][q])

                        total_penalty_pot += penalty

                # 2. Assign the accumulated pot to the winner
                if winner in model_seats and len(episode_trajs[env_idx][winner]) > 0:
                    episode_trajs[env_idx][winner][-1].reward = total_penalty_pot
                    episode_trajs[env_idx][winner][-1].done = True
                    trajectories[winner].extend(episode_trajs[env_idx][winner])
            else:
                states[env_idx] = next_state
                next_active_envs.append(env_idx)

        active_envs = next_active_envs

    return trajectories


def ppo_update(
    policy: nn.Module,
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
    all_old_values = []
    all_advantages = []
    all_returns = []

    for p in trajectories:
        ep_states: list[np.ndarray] = []
        ep_action_indices: list[int] = []
        ep_candidates: list[list[Combo]] = []
        ep_old_logprobs: list[torch.Tensor] = []
        ep_rewards: list[float] = []
        ep_values: list[torch.Tensor] = []
        ep_dones: list[float] = []

        for record in trajectories[p]:
            ep_states.append(record.state)
            ep_action_indices.append(record.action_idx)
            ep_candidates.append(record.candidates)
            ep_old_logprobs.append(record.old_logprob)
            ep_rewards.append(record.reward)
            ep_values.append(record.value)
            ep_dones.append(1.0 if record.done else 0.0)

            if record.done:
                rewards_t = torch.tensor(ep_rewards, dtype=torch.float32, device=device)
                # Keep values as a 1D time axis even for single-step episodes.
                values_t = torch.stack(ep_values).reshape(-1).to(device)
                dones_t = torch.tensor(ep_dones, dtype=torch.float32, device=device)

                adv_t, ret_t = compute_gae_from_values(rewards_t, values_t, dones_t, gamma=gamma, lam=lam)

                for idx in range(len(ep_states)):
                    all_states.append(ep_states[idx])
                    all_action_indices.append(ep_action_indices[idx])
                    all_candidates.append(ep_candidates[idx])
                    all_old_logprobs.append(ep_old_logprobs[idx])
                    all_old_values.append(values_t[idx])
                    all_advantages.append(adv_t[idx])
                    all_returns.append(ret_t[idx])

                # reset buffers for next episode
                ep_states.clear()
                ep_action_indices.clear()
                ep_candidates.clear()
                ep_old_logprobs.clear()
                ep_rewards.clear()
                ep_values.clear()
                ep_dones.clear()

    if len(all_states) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Convert to tensors
    old_logprobs_t = torch.stack(all_old_logprobs).to(device)
    old_values_t = torch.stack(all_old_values).reshape(-1).to(device)
    advantages = torch.stack(all_advantages).to(device)
    returns = torch.stack(all_returns).to(device)

    # Normalize advantages with stability guard
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std < 1e-8:
        # If advantages are nearly constant, don't normalize (avoid division by near-zero)
        advantages = advantages - adv_mean
    else:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # Prepare data for mini-batching
    n_samples = len(all_states)
    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    if mini_batch_size is None:
        mini_batch_size = n_samples
    else:
        # Ensure mini_batch_size is at least 1 and doesn't exceed n_samples
        mini_batch_size = max(1, min(mini_batch_size, n_samples))

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

            # Skip empty batches
            if len(batch_indices) == 0:
                continue

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
            action_feats = [
                [combo_to_action_vector(c, card_count=policy.card_universe_size) for c in candidates]
                for candidates in batch_candidates
            ]
            logits_list, values = policy(state_tensor, action_feats)

            # Extract logprobs for chosen actions
            new_logprobs_list = []
            entropies_list = []
            values_list = []

            for i, (logits, action_idx) in enumerate(zip(logits_list, batch_action_indices, strict=True)):
                dist = safe_categorical_from_logits(logits, batch_candidates[i])
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
            nn.utils.clip_grad_norm_(policy.parameters(), 10)
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
    config: PPOConfig | None = None,
    n_players=4,
    cards_per_player: int | None = None,
    model_seat_count=1,
    batches=2000,
    episodes_per_batch=10,
    ppo_epochs=4,
    clip_epsilon=0.2,
    lr=3e-4,
    entropy_beta=0.01,
    value_coef=0.5,
    gamma=0.99,
    lam=1.0,
    seed=42,
    device="cpu",
    eval_interval=50,
    eval_games=500,
    mini_batch_size=None,
    use_batched_collection=True,
    policy_arch: str = "mlp",
    monitor: TrainingMonitor | None = None,
    **kwargs,
):
    """
    Train using Proximal Policy Optimization (PPO).

    Args:
        config: PPOConfig instance. If None, creates from kwargs for backward compatibility.
        **kwargs: Individual parameters (deprecated, use config instead)
    """
    if config is None:
        # Backward compatibility: construct config from kwargs
        config = PPOConfig(
            n_players=n_players,
            cards_per_player=cards_per_player,
            model_seat_count=model_seat_count,
            batches=batches,
            episodes_per_batch=episodes_per_batch,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
            lr=lr,
            entropy_beta=entropy_beta,
            value_coef=value_coef,
            gamma=gamma,
            lam=lam,
            seed=seed,
            device=device,
            eval_interval=eval_interval,
            eval_games=eval_games,
            mini_batch_size=mini_batch_size,
            use_batched_collection=use_batched_collection,
            policy_arch=policy_arch,
            **kwargs,
        )

    if not 1 <= config.model_seat_count <= config.n_players:
        raise ValueError("model_seat_count must be between 1 and n_players")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    all_players = set(range(config.n_players))

    print(
        f"Training across seats: {config.model_seat_count} model-controlled seat(s) per environment "
        "(sampled each episode)"
    )
    print("Opponent diversity within batch: enabled (opponents sampled per-environment/per-seat)")
    print(f"PPO parameters: epochs={config.ppo_epochs}, clip_epsilon={config.clip_epsilon}")
    print(f"Policy architecture: {config.policy_arch}")
    print(
        "Opponent mixture: Mastery-based curriculum learning "
        "(Phase 1: Learn greedy 50-60%, Phase 2: Learn smart 45-50% with greedy retention, "
        "Phase 3: Self-play 55-65% with retention of mastered opponents)"
    )

    env = Big2Env(config.n_players, cards_per_player=config.cards_per_player)
    policy = make_policy(
        config.policy_arch,
        n_players=config.n_players,
        cards_per_player=config.cards_per_player,
        device=config.device,
    ).to(config.device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    # Checkpoint manager for opponent sampling
    checkpoint_manager = CheckpointManager(
        device=config.device,
        n_players=config.n_players,
        cards_per_player=config.cards_per_player,
    )

    # Learning rate scheduler: warmup + cosine annealing
    warmup_batches = int(config.batches * config.warmup_fraction)

    def lr_lambda(current_batch):
        if current_batch < warmup_batches:
            # Linear warmup
            return current_batch / warmup_batches if warmup_batches > 0 else 1.0
        else:
            # Cosine annealing after warmup (with floor)
            progress = (current_batch - warmup_batches) / (config.batches - warmup_batches)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Adaptive entropy coefficient
    current_entropy_beta = config.entropy_beta

    # Track losses for plotting
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []

    # Track evaluation metrics
    eval_episodes = []
    win_rates = []
    win_rates_smart = []
    opponent_mixes = []  # Track opponent mix at each evaluation checkpoint

    # Initialize training monitor if provided
    if monitor is not None:
        config_summary = {
            "n_players": config.n_players,
            "cards_per_player": config.cards_per_player,
            "model_seat_count": config.model_seat_count,
            "episodes_per_batch": config.episodes_per_batch,
            "ppo_epochs": config.ppo_epochs,
            "clip_epsilon": config.clip_epsilon,
            "lr": config.lr,
            "entropy_beta": config.entropy_beta,
            "gamma": config.gamma,
            "lam": config.lam,
            "mini_batch_size": config.mini_batch_size,
            "policy_arch": config.policy_arch,
        }
        monitor.start_training(config.batches, config_summary)

    for batch in tqdm(range(1, config.batches + 1)):
        # Collect trajectories
        model_seats_by_env: list[list[int]] = []
        for _ in range(config.episodes_per_batch):
            seat_order = np.random.permutation(config.n_players)
            seats = seat_order[: config.model_seat_count]
            model_seats_by_env.append([int(s) for s in seats])
        frac = batch / config.batches
        target_entropy = config.target_entropy_start + (config.target_entropy_end - config.target_entropy_start) * frac
        opponent_strategies_by_env: list[dict[int, nn.Module | Callable]] = []
        for env_idx in range(config.episodes_per_batch):
            model_seats = set(model_seats_by_env[env_idx])
            opp_map: dict[int, nn.Module | Callable] = {}
            for p in all_players:
                if p in model_seats:
                    continue
                opp_map[p] = checkpoint_manager.sample_opponent_policy(policy)
            opponent_strategies_by_env.append(opp_map)

        if config.use_batched_collection:
            trajectories = collect_ppo_trajectories_batched(
                config.n_players,
                config.cards_per_player,
                policy,
                config.episodes_per_batch,
                opponent_strategies_by_env,
                model_seats_by_env,
                config.device,
                step_penalty=config.step_penalty,
                progress_reward_coef=config.progress_reward_coef,
                pass_penalty=config.pass_penalty,
            )
        else:
            trajectories = collect_ppo_trajectories(
                env,
                policy,
                config.episodes_per_batch,
                opponent_strategies_by_env,
                model_seats_by_env,
                config.device,
                step_penalty=config.step_penalty,
                progress_reward_coef=config.progress_reward_coef,
                pass_penalty=config.pass_penalty,
            )

        # PPO update
        policy_loss, value_loss, entropy, total_loss = ppo_update(
            policy,
            trajectories,
            optimizer,
            config.ppo_epochs,
            config.clip_epsilon,
            config.value_coef,
            current_entropy_beta,
            config.mini_batch_size,
            config.device,
            gamma=config.gamma,
            lam=config.lam,
        )

        # Step the learning rate scheduler
        scheduler.step()
        if batch / config.batches >= 0.3:
            # Adaptive entropy coefficient: adjust based on current entropy
            if entropy < target_entropy - 0.15:
                # Entropy too low, encourage more exploration
                current_entropy_beta = min(config.entropy_beta_max, current_entropy_beta * 1.01)
            elif entropy > target_entropy + 0.15:
                # Entropy too high, allow more exploitation
                current_entropy_beta = max(config.entropy_beta_min, current_entropy_beta * 0.99)

        # Record losses
        loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(value_loss)
        entropy_history.append(entropy)

        # Update training monitor
        if monitor is not None:
            current_lr = scheduler.get_last_lr()[0]
            total_steps = sum(len(v) for v in trajectories.values())
            steps_per_ep = total_steps / config.episodes_per_batch if config.episodes_per_batch > 0 else 0.0
            monitor.update_batch(
                batch=batch,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                total_loss=total_loss,
                lr=current_lr,
                entropy_beta=current_entropy_beta,
                steps_per_episode=steps_per_ep,
            )

        # Evaluation every eval_interval batches
        if batch % config.eval_interval == 0:
            print("LAMBDA: ", config.lam)
            total_steps = sum(len(v) for v in trajectories.values())
            avg_steps_per_episode = total_steps / config.episodes_per_batch if config.episodes_per_batch > 0 else 0.0

            # Comprehensive evaluation
            print(f"\n[Step {batch}] Evaluating policy...")
            policy.eval()
            metrics = evaluate_policy(
                policy,
                config.n_players,
                cards_per_player=config.cards_per_player,
                num_games=config.eval_games,
                device=config.device,
            )
            policy.train()

            eval_episodes.append(batch)
            win_rates.append(metrics.win_rate_vs_greedy)
            win_rates_smart.append(metrics.win_rate_vs_smart)

            # Update mastery levels based on win rates against greedy and smart
            checkpoint_manager.update_mastery(metrics.win_rate_vs_greedy, metrics.win_rate_vs_smart)

            # Track opponent mix at this checkpoint
            opponent_mix = checkpoint_manager.compute_dynamic_opponent_mix()
            opponent_mixes.append(opponent_mix)

            # Update training monitor with evaluation results
            if monitor is not None:
                monitor.update_evaluation(
                    batch=batch,
                    win_rate_greedy=metrics.win_rate_vs_greedy,
                    win_rate_smart=metrics.win_rate_vs_smart,
                    win_rate_random=metrics.win_rate_vs_random,
                    avg_cards_remaining=metrics.avg_cards_remaining_when_losing,
                    avg_score_greedy=metrics.avg_score_vs_greedy,
                    avg_score_random=metrics.avg_score_vs_random,
                    avg_score_smart=metrics.avg_score_vs_smart,
                )
                # Update curriculum state
                ema_greedy = checkpoint_manager.ema_win_rate_greedy or 0.0
                ema_smart = checkpoint_manager.ema_win_rate_smart or 0.0
                from dataclasses import asdict

                mix_dict = asdict(opponent_mix)
                monitor.update_curriculum(
                    num_checkpoints=len(checkpoint_manager.checkpoints),
                    ema_win_rate_greedy=ema_greedy,
                    ema_win_rate_smart=ema_smart,
                    mastery_greedy=checkpoint_manager.mastery_greedy,
                    mastery_smart=checkpoint_manager.mastery_smart,
                    opponent_mix=mix_dict,
                )

            print(f"[Step {batch}] Evaluation Results ({metrics.total_games} games):")
            wins_vs_greedy = int(metrics.win_rate_vs_greedy * metrics.total_games)
            wr_str = f"  Win rate vs greedy: {metrics.win_rate_vs_greedy:.2%}"
            wr_str += f" ({wins_vs_greedy}/{metrics.total_games} wins)"
            print(wr_str)
            print(f"  Win rate vs random: {metrics.win_rate_vs_random:.2%}")
            print(f"  Avg score vs greedy: {metrics.avg_score_vs_greedy:.2f}")
            print(f"  Avg score vs random: {metrics.avg_score_vs_random:.2f}")
            wins_vs_smart = int(metrics.win_rate_vs_smart * metrics.total_games)
            wr_smart_str = f"  Win rate vs smart: {metrics.win_rate_vs_smart:.2%}"
            wr_smart_str += f" ({wins_vs_smart}/{metrics.total_games} wins)"
            print(wr_smart_str)
            print(f"  Avg score vs smart: {metrics.avg_score_vs_smart:.2f}")
            print(f"  Avg cards remaining when losing: {metrics.avg_cards_remaining_when_losing:.2f}")
            print("  Win rate by starting position:")
            for pos, wr in sorted(metrics.win_rate_by_starting_position.items()):
                print(f"    Position {pos}: {wr:.2%}")
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {batch}] loss={total_loss:.3f} pol={policy_loss:.3f} "
                f"val={value_loss:.3f} ent={entropy:.3f} steps/model_ep={avg_steps_per_episode:.1f} "
                f"({config.episodes_per_batch} episodes, ~{avg_steps_per_episode:.1f} steps/episode)"
            )
            print(f"[Step {batch}] lr={current_lr:.6f} ent_beta={current_entropy_beta:.4f}")
            n_checkpoints = len(checkpoint_manager.checkpoints)
            ema_greedy = (
                checkpoint_manager.ema_win_rate_greedy if checkpoint_manager.ema_win_rate_greedy is not None else 0.0
            )
            ema_smart = (
                checkpoint_manager.ema_win_rate_smart if checkpoint_manager.ema_win_rate_smart is not None else 0.0
            )
            print(
                f"[Step {batch}] Checkpoints in pool: {n_checkpoints}, "
                f"Opponent mix: {checkpoint_manager.compute_dynamic_opponent_mix()}, "
                f"EMA win rates (greedy: {ema_greedy:.2%}, smart: {ema_smart:.2%})\n"
            )

            # Save checkpoint and add to manager
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            save_path = models_dir / f"big2_model_step_{batch}.pt"
            torch.save(policy.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

            checkpoint_manager.add_checkpoint(batch, policy)

    # Finish training monitor
    if monitor is not None:
        monitor.finish_training(success=True, message="Training completed successfully")

    return (
        policy,
        checkpoint_manager,
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        eval_episodes,
        win_rates,
        win_rates_smart,
        opponent_mixes,
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    n_players = 2
    cards_per_player = 5

    # Create config with all training parameters
    config = PPOConfig(
        n_players=n_players,
        cards_per_player=cards_per_player,
        model_seat_count=1,
        batches=8000,
        episodes_per_batch=128,
        ppo_epochs=2,
        clip_epsilon=0.2,
        lr=1e-4,
        entropy_beta=0.05,
        value_coef=0.5,
        gamma=0.99,
        lam=1,
        seed=42,
        device=device,
        eval_interval=20,
        eval_games=250,
        mini_batch_size=128,
        policy_arch="setpool",
    )

    # Save config for reproducibility
    config.save("training_config.json")
    print("Training config saved to training_config.json")
    # To load: config = PPOConfig.load("training_config.json")

    # Create training monitor for real-time dashboard
    monitor = TrainingMonitor(stats_file="training_stats.json")
    print("Training monitor initialized - view at http://localhost:8000/monitor")

    (
        policy,
        checkpoint_manager,
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        eval_episodes,
        win_rates,
        win_rates_smart,
        opponent_mixes,
    ) = train_ppo(config=config, monitor=monitor)

    total_cards_in_play = n_players * cards_per_player
    hand = sorted(range(total_cards_in_play - cards_per_player, total_cards_in_play))
    val = value_of_starting_hand(
        policy, hand, n_players=n_players, cards_per_player=cards_per_player, sims=64, device=device
    )
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Win rate:", val)

    hand = sorted(random.sample(range(total_cards_in_play), cards_per_player))
    val = value_of_starting_hand(
        policy, hand, n_players=n_players, cards_per_player=cards_per_player, sims=64, device=device
    )
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Win rate:", val)

    # Save the trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_save_path = str(models_dir / "big2_model.pt")
    torch.save(policy.state_dict(), model_save_path)
    print(f"\nModel weights saved to {model_save_path}")

    # Plot and save training curves
    training_curves_path = "training_curves.png"
    plot_training_curves(
        loss_history,
        policy_loss_history,
        value_loss_history,
        entropy_history,
        None,  # max_logprob_history not tracked in PPO
        eval_episodes,
        win_rates,
        win_rates_smart,
        opponent_mixes=opponent_mixes,
        save_path=training_curves_path,
    )

    # Dump all configs and results
    dump_training_run(
        config=config,
        policy=policy,
        loss_history=loss_history,
        policy_loss_history=policy_loss_history,
        value_loss_history=value_loss_history,
        entropy_history=entropy_history,
        eval_episodes=eval_episodes,
        win_rates=win_rates,
        win_rates_smart=win_rates_smart,
        checkpoint_manager=checkpoint_manager,
        model_save_path=model_save_path,
        training_curves_path=training_curves_path,
    )
