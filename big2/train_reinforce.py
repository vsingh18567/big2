import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from big2.nn import MLPPolicy
from big2.simulator.cards import card_name
from big2.simulator.env import Big2Env
from big2.train_helpers import (
    CheckpointManager,
    compute_gae_from_values,
    episode,
    evaluate_against_greedy,
    plot_training_curves,
    value_of_starting_hand,
)


def train_selfplay(
    n_players=4,
    batches=2000,
    episodes_per_batch=10,
    lr=1e-3,
    entropy_beta=0.01,
    value_coef=0.5,
    gamma=1.0,
    seed=42,
    device="cpu",
    eval_interval=50,
    eval_games=500,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Opponent sampling: 50% current policy, 30% past checkpoints, 20% greedy
    # Model players are always player 0, opponents are players 1 to n_players-1
    model_players = {0}
    opponent_players = set(range(1, n_players))

    print(f"Model players: {sorted(model_players)}, Opponent players: {sorted(opponent_players)}")
    print(
        "Opponent mixture: Adaptive based on win rate schedule "
        "(<20%: 50% greedy, 20-30%: 25% greedy, >=30%: 15% greedy)"
    )

    env = Big2Env(n_players)
    policy = MLPPolicy(n_players=n_players, device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Checkpoint manager for opponent sampling
    checkpoint_manager = CheckpointManager(device=device, n_players=n_players)

    # Learning rate scheduler: warmup + cosine annealing
    warmup_batches = batches // 10  # 10% warmup

    def lr_lambda(current_batch):
        min_lr_ratio = 0.2  # LR won't drop below 20% of initial
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
    target_entropy = 0.5  # Target entropy level
    entropy_beta_min = 0.02
    entropy_beta_max = 0.2

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
            # Sample opponents for this episode
            opponent_strategies = dict.fromkeys(opponent_players)
            for opp_id in opponent_players:
                opponent_strategies[opp_id] = checkpoint_manager.sample_opponent_policy(policy)
            traj = episode(env, policy, opponent_strategies=opponent_strategies, model_players=model_players)

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

        loss = policy_loss + value_coef * value_loss - current_entropy_beta * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Step the learning rate scheduler
        scheduler.step()

        # Adaptive entropy coefficient: adjust based on current entropy
        if entropy.item() < target_entropy - 0.1:
            # Entropy too low, encourage more exploration
            current_entropy_beta = min(entropy_beta_max, current_entropy_beta * 1.02)
        elif entropy.item() > target_entropy + 0.1:
            # Entropy too high, allow more exploitation
            current_entropy_beta = max(entropy_beta_min, current_entropy_beta * 0.98)

        # Record losses
        loss_history.append(loss.item())
        policy_loss_history.append(policy_loss.item())
        value_loss_history.append(value_loss.item())
        entropy_history.append(entropy.item())
        max_logprob_history.append(avg_max_logprob.item())

        # Evaluation every eval_interval episodes
        if batch % eval_interval == 0:
            avg_len = sum(len(traj[p]) for p in model_players) / len(model_players) if model_players else 0

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
            print(f"  Avg cards remaining when losing: {metrics.avg_cards_remaining_when_losing:.2f}")
            print("  Win rate by starting position:")
            for pos, wr in sorted(metrics.win_rate_by_starting_position.items()):
                print(f"    Position {pos}: {wr:.2%}")
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {batch}] loss={loss.item():.3f} pol={policy_loss.item():.3f} "
                f"val={value_loss.item():.3f} ent={entropy.item():.3f} steps/player~{avg_len:.1f}"
            )
            print(f"[Step {batch}] lr={current_lr:.6f} ent_beta={current_entropy_beta:.4f}")
            n_checkpoints = len(checkpoint_manager.checkpoints)
            stage = checkpoint_manager.greedy_schedule_stage
            greedy_pct = {0: "50%", 1: "25%", 2: "15%"}[stage]
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
        max_logprob_history,
        eval_episodes,
        win_rates,
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
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
        batches=200,
        episodes_per_batch=64,  # Increased from 16 for more stable gradients
        lr=0.0005,
        entropy_beta=0.05,
        value_coef=0.5,
        gamma=0.99,
        seed=42,
        device=device,
        eval_interval=25,
        eval_games=250,
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
