from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from big2.training.big2_v2.model import Big2V2ActorCritic
from big2.training.big2_v2.rollout import Big2V2RolloutBuffer


@dataclass(frozen=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    samples: int
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    ratio_mean: float = 0.0
    ratio_std: float = 0.0
    ratio_max: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    return_mean: float = 0.0
    return_std: float = 0.0
    value_explained_variance: float = 0.0
    grad_norm: float = 0.0
    action_probability_mean: float = 0.0
    action_probability_p10: float = 0.0
    action_probability_p90: float = 0.0


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    bootstrap_value: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    next_value = torch.as_tensor(bootstrap_value, dtype=torch.float32, device=rewards.device)
    for idx in range(rewards.numel() - 1, -1, -1):
        not_done = 1.0 - dones[idx]
        delta = rewards[idx] + gamma * next_value * not_done - values[idx]
        gae = delta + gamma * lam * not_done * gae
        advantages[idx] = gae
        next_value = values[idx]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    *,
    policy: Big2V2ActorCritic,
    buffer: Big2V2RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    ppo_epochs: int,
    mini_batch_size: int,
    clip_epsilon: float,
    gamma: float,
    lam: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    device: str | torch.device,
) -> PPOUpdateStats:
    if len(buffer) == 0:
        return PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0)

    device = torch.device(device)
    records = buffer.records
    obs = torch.stack([record.obs for record in records]).to(device)
    candidate_ids = torch.stack([record.candidate_ids for record in records]).to(device)
    candidate_mask = torch.stack([record.candidate_mask for record in records]).to(device)
    slots = torch.stack([record.slot for record in records]).long().to(device)
    old_logprobs = torch.stack([record.old_logprob for record in records]).to(device)

    advantages_by_index: dict[int, torch.Tensor] = {}
    returns_by_index: dict[int, torch.Tensor] = {}
    record_offset = {id(record): idx for idx, record in enumerate(records)}
    for trajectory_key, trajectory in buffer.by_trajectory().items():
        rewards = torch.tensor([record.reward for record in trajectory], dtype=torch.float32, device=device)
        values = torch.stack([record.value for record in trajectory]).float().to(device)
        dones = torch.tensor([record.done for record in trajectory], dtype=torch.float32, device=device)
        bootstrap_value = 0.0 if bool(dones[-1].item()) else buffer.bootstrap_values.get(trajectory_key, 0.0)
        adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=lam, bootstrap_value=bootstrap_value)
        for local_idx, record in enumerate(trajectory):
            global_idx = record_offset[id(record)]
            advantages_by_index[global_idx] = adv[local_idx]
            returns_by_index[global_idx] = ret[local_idx]

    advantages = torch.stack([advantages_by_index[idx] for idx in range(len(records))])
    returns = torch.stack([returns_by_index[idx] for idx in range(len(records))])
    raw_advantages = advantages.detach()
    raw_returns = returns.detach()
    old_values = torch.stack([record.value for record in records]).float().to(device).detach()
    advantage_mean = float(raw_advantages.mean().item())
    advantage_std = float(raw_advantages.std(unbiased=False).item()) if raw_advantages.numel() > 1 else 0.0
    return_mean = float(raw_returns.mean().item())
    return_std = float(raw_returns.std(unbiased=False).item()) if raw_returns.numel() > 1 else 0.0
    return_var = raw_returns.var(unbiased=False)
    if raw_returns.numel() > 1 and float(return_var.item()) > 1e-8:
        value_explained_variance = float((1.0 - (raw_returns - old_values).var(unbiased=False) / return_var).item())
    else:
        value_explained_variance = 0.0
    action_probs = old_logprobs.exp().detach()
    action_probability_mean = float(action_probs.mean().item())
    action_probability_p10 = float(torch.quantile(action_probs, 0.10).item())
    action_probability_p90 = float(torch.quantile(action_probs, 0.90).item())
    if advantages.numel() > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    total_ratio_mean = 0.0
    total_ratio_std = 0.0
    total_ratio_max = 0.0
    total_grad_norm = 0.0
    updates = 0

    policy.train()
    sample_count = len(records)
    mini_batch_size = max(1, min(mini_batch_size, sample_count))
    for _ in range(ppo_epochs):
        order = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, mini_batch_size):
            idx = order[start : start + mini_batch_size]
            new_logprobs, values, entropy = policy.evaluate_actions(
                obs[idx],
                candidate_ids[idx],
                candidate_mask[idx],
                slots[idx],
            )
            log_ratio = new_logprobs - old_logprobs[idx]
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[idx])
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_mean.item())
            total_loss += float(loss.item())
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
            total_approx_kl += float(approx_kl.item())
            total_clip_fraction += float(clip_fraction.item())
            total_ratio_mean += float(ratio.mean().item())
            total_ratio_std += float(ratio.std(unbiased=False).item()) if ratio.numel() > 1 else 0.0
            total_ratio_max += float(ratio.max().item())
            total_grad_norm += float(grad_norm.item())
            updates += 1

    return PPOUpdateStats(
        policy_loss=total_policy_loss / updates,
        value_loss=total_value_loss / updates,
        entropy=total_entropy / updates,
        total_loss=total_loss / updates,
        samples=sample_count,
        approx_kl=total_approx_kl / updates,
        clip_fraction=total_clip_fraction / updates,
        ratio_mean=total_ratio_mean / updates,
        ratio_std=total_ratio_std / updates,
        ratio_max=total_ratio_max / updates,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
        return_mean=return_mean,
        return_std=return_std,
        value_explained_variance=value_explained_variance,
        grad_norm=total_grad_norm / updates,
        action_probability_mean=action_probability_mean,
        action_probability_p10=action_probability_p10,
        action_probability_p90=action_probability_p90,
    )
