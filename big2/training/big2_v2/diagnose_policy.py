from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import torch

from big2.training.big2_v2.checkpoints import load_checkpoint_partial
from big2.training.big2_v2.config import OpponentMixConfig, Big2V2Config
from big2.training.big2_v2.env_adapter import RustVecEnvAdapter
from big2.training.big2_v2.metadata import MoveMetadataTable
from big2.training.big2_v2.model import Big2V2ActorCritic
from big2.training.big2_v2.opponents import greedy_slot, smart_slot
from big2.training.big2_v2.run import build_policy

OBS_OWN_HAND_START = 0
OBS_OWN_HAND_END = 52
OBS_LAST_MOVE_KIND_START = 104
OBS_LAST_MOVE_KIND_END = 113
OBS_CARDS_REMAINING_START = 120
OBS_FREE_LEAD = 128

KIND_NAMES = {
    0: "Pass",
    1: "Single",
    2: "Pair",
    3: "Triple",
    4: "Straight",
    5: "Flush",
    6: "FullHouse",
    7: "FourOfKind",
    8: "StraightFlush",
}


@dataclass
class BucketStats:
    decisions: int = 0
    wins: int = 0
    reward_sum: float = 0.0
    entropy_sum: float = 0.0
    top_prob_sum: float = 0.0
    selected_passes: int = 0
    greedy_agreements: int = 0
    smart_agreements: int = 0

    def update(self, decision: dict[str, Any], *, won: bool, reward: float) -> None:
        self.decisions += 1
        self.wins += int(won)
        self.reward_sum += reward
        self.entropy_sum += float(decision["entropy"])
        self.top_prob_sum += float(decision["top_prob"])
        self.selected_passes += int(decision["selected_kind"] == "Pass")
        self.greedy_agreements += int(decision["agrees_with_greedy"])
        self.smart_agreements += int(decision["agrees_with_smart"])

    def as_row(self, *, baseline_win_rate: float) -> dict[str, Any]:
        decisions = max(self.decisions, 1)
        win_rate = self.wins / decisions
        return {
            "decisions": self.decisions,
            "wins": self.wins,
            "win_rate": win_rate,
            "delta_vs_overall": win_rate - baseline_win_rate,
            "average_reward": self.reward_sum / decisions,
            "average_entropy": self.entropy_sum / decisions,
            "average_top_prob": self.top_prob_sum / decisions,
            "selected_pass_rate": self.selected_passes / decisions,
            "greedy_agreement_rate": self.greedy_agreements / decisions,
            "smart_agreement_rate": self.smart_agreements / decisions,
        }


@dataclass
class EpisodeStats:
    games: int = 0
    wins: int = 0
    reward_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def average_reward(self) -> float:
        return self.reward_sum / self.games if self.games else 0.0


def load_policy_from_checkpoint(
    *,
    checkpoint: Path,
    device: str,
    max_candidates: int,
    num_envs: int,
    seed: int,
) -> tuple[Big2V2ActorCritic, Big2V2Config]:
    payload = torch.load(checkpoint, map_location=device)
    config = _config_from_payload(payload.get("config", {}))
    config.device = device
    config.max_candidates = max_candidates
    config.num_envs = num_envs
    config.seed = seed

    env = RustVecEnvAdapter(num_envs=1, seed=seed, max_candidates=max_candidates, device=device)
    batch = env.reset()
    if "candidate_context_projection.0.weight" in payload.get("model_state", {}):
        config.candidate_set_context = True
    if "candidate_outcome_encoder.0.weight" in payload.get("model_state", {}):
        config.dynamic_action_features = True
    policy = build_policy(env, config, obs_dim=batch.obs_dim)
    load_checkpoint_partial(path=checkpoint, policy=policy, map_location=device)
    policy.eval()
    return policy, config


@torch.no_grad()
def diagnose_policy(
    *,
    policy: Big2V2ActorCritic,
    opponent: str,
    games: int,
    num_envs: int,
    seed: int,
    max_candidates: int,
    policy_seat: int,
    device: str,
    min_decisions: int,
) -> dict[str, Any]:
    if opponent not in {"random", "greedy", "smart"}:
        raise ValueError(f"Unsupported opponent: {opponent}")

    env = RustVecEnvAdapter(num_envs=num_envs, seed=seed, max_candidates=max_candidates, device=device)
    batch = env.reset()
    torch.manual_seed(seed)

    episode_decisions: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    episode_lengths = [0 for _ in range(num_envs)]
    bucket_stats: dict[str, BucketStats] = defaultdict(BucketStats)
    episode_stats = EpisodeStats()
    truncated_candidate_lists = batch.truncated_candidate_lists

    while episode_stats.games < games:
        action_ids = torch.zeros(batch.num_envs, dtype=torch.long, device=batch.obs.device)
        valid_rows = batch.candidate_mask.any(dim=1)
        policy_indices: list[int] = []

        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                action_ids[env_idx] = 0
                continue
            if int(batch.current_player[env_idx].item()) == policy_seat:
                policy_indices.append(env_idx)
                continue
            if opponent == "random":
                valid = torch.nonzero(batch.candidate_mask[env_idx], as_tuple=False).flatten()
                slot = int(valid[torch.randint(len(valid), (1,), device=valid.device)].item())
            elif opponent == "greedy":
                slot = greedy_slot(batch.candidate_ids[env_idx], batch.candidate_mask[env_idx], env.metadata)
            else:
                slot = smart_slot(
                    batch.obs[env_idx],
                    batch.candidate_ids[env_idx],
                    batch.candidate_mask[env_idx],
                    env.metadata,
                )
            action_ids[env_idx] = batch.candidate_ids[env_idx, slot]

        if policy_indices:
            idx = torch.tensor(policy_indices, dtype=torch.long, device=batch.obs.device)
            dist, _values, entropy = policy.distribution(
                batch.obs[idx],
                batch.candidate_ids[idx],
                batch.candidate_mask[idx],
            )
            probs = dist.probs
            slots = probs.argmax(dim=1)
            action_ids[idx] = batch.candidate_ids[idx].gather(1, slots.unsqueeze(1)).squeeze(1)

            top_two = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
            for local_idx, env_idx in enumerate(policy_indices):
                slot = int(slots[local_idx].item())
                top_prob = float(top_two[local_idx, 0].item())
                second_prob = float(top_two[local_idx, 1].item()) if top_two.shape[1] > 1 else 0.0
                episode_decisions[env_idx].append(
                    _decision_snapshot(
                        obs=batch.obs[env_idx],
                        candidate_ids=batch.candidate_ids[env_idx],
                        candidate_mask=batch.candidate_mask[env_idx],
                        selected_slot=slot,
                        entropy=float(entropy[local_idx].item()),
                        top_prob=top_prob,
                        probability_margin=top_prob - second_prob,
                        metadata=env.metadata,
                    )
                )

        next_batch = env.step(action_ids)
        truncated_candidate_lists += next_batch.truncated_candidate_lists
        for env_idx in range(batch.num_envs):
            if bool(batch.done[env_idx].item()) or not bool(valid_rows[env_idx].item()):
                continue
            episode_lengths[env_idx] += 1

        done_indices: list[int] = []
        for env_idx in range(next_batch.num_envs):
            if not bool(next_batch.done[env_idx].item()):
                continue
            reward = float(next_batch.final_rewards[env_idx, policy_seat].item())
            won = reward > 0.0
            episode_stats.games += 1
            episode_stats.wins += int(won)
            episode_stats.reward_sum += reward
            for decision in episode_decisions[env_idx]:
                for bucket in _decision_buckets(decision):
                    bucket_stats[bucket].update(decision, won=won, reward=reward)
            episode_decisions[env_idx].clear()
            episode_lengths[env_idx] = 0
            done_indices.append(env_idx)
            if episode_stats.games >= games:
                break

        batch = env.reset_done(done_indices) if done_indices else next_batch

    baseline = episode_stats.win_rate
    bucket_rows = {
        bucket: stats.as_row(baseline_win_rate=baseline)
        for bucket, stats in bucket_stats.items()
        if stats.decisions >= min_decisions
    }
    weaknesses = sorted(
        (
            {"bucket": bucket, **row}
            for bucket, row in bucket_rows.items()
            if row["delta_vs_overall"] < 0.0
        ),
        key=lambda row: (row["delta_vs_overall"], -row["decisions"]),
    )
    strengths = sorted(
        ({"bucket": bucket, **row} for bucket, row in bucket_rows.items()),
        key=lambda row: (row["delta_vs_overall"], row["decisions"]),
        reverse=True,
    )
    return {
        "opponent": opponent,
        "policy_seat": policy_seat,
        "games": episode_stats.games,
        "wins": episode_stats.wins,
        "win_rate": baseline,
        "average_reward": episode_stats.average_reward,
        "truncated_candidate_lists": truncated_candidate_lists,
        "buckets": bucket_rows,
        "weaknesses": weaknesses[:20],
        "strengths": strengths[:20],
    }


def _decision_snapshot(
    *,
    obs: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_slot: int,
    entropy: float,
    top_prob: float,
    probability_margin: float,
    metadata: MoveMetadataTable,
) -> dict[str, Any]:
    selected_move_id = int(candidate_ids[selected_slot].item())
    selected = metadata.get(selected_move_id)
    candidate_count = int(candidate_mask.sum().item())
    own_cards = int(obs[OBS_OWN_HAND_START:OBS_OWN_HAND_END].sum().item())
    last_kind = int(obs[OBS_LAST_MOVE_KIND_START:OBS_LAST_MOVE_KIND_END].argmax().item())
    cards_remaining = obs[OBS_CARDS_REMAINING_START : OBS_CARDS_REMAINING_START + 4]
    greedy_move_id = int(candidate_ids[greedy_slot(candidate_ids, candidate_mask, metadata)].item())
    smart_move_id = int(candidate_ids[smart_slot(obs, candidate_ids, candidate_mask, metadata)].item())
    return {
        "candidate_count": candidate_count,
        "candidate_count_bucket": _candidate_count_bucket(candidate_count),
        "forced_action": candidate_count == 1,
        "own_cards": own_cards,
        "own_cards_bucket": _cards_bucket(own_cards),
        "leader_cards_bucket": _cards_bucket(int(round(float(cards_remaining[0].item()) * 13))),
        "next_opponent_cards_bucket": _cards_bucket(int(round(float(cards_remaining[1].item()) * 13))),
        "free_lead": bool(float(obs[OBS_FREE_LEAD].item()) > 0.5),
        "last_kind": KIND_NAMES.get(last_kind, f"kind_{last_kind}"),
        "selected_kind": KIND_NAMES.get(selected.kind, f"kind_{selected.kind}"),
        "selected_num_cards": selected.num_cards,
        "selected_rank_bucket": _rank_bucket(selected.primary_rank),
        "entropy": entropy,
        "top_prob": top_prob,
        "probability_margin": probability_margin,
        "confidence_bucket": _confidence_bucket(top_prob),
        "agrees_with_greedy": selected_move_id == greedy_move_id,
        "agrees_with_smart": selected_move_id == smart_move_id,
    }


def _decision_buckets(decision: dict[str, Any]) -> Iterable[str]:
    context = "lead" if decision["free_lead"] else "response"
    pass_bucket = _pass_bucket(decision)
    yield f"context={context}"
    yield f"forced_action={str(decision['forced_action']).lower()}"
    yield f"context={context}|forced_action={str(decision['forced_action']).lower()}"
    yield f"pass={pass_bucket}"
    yield f"context={context}|pass={pass_bucket}"
    yield f"hand={decision['own_cards_bucket']}"
    yield f"context={context}|hand={decision['own_cards_bucket']}"
    yield f"candidate_count={decision['candidate_count_bucket']}"
    yield f"context={context}|candidate_count={decision['candidate_count_bucket']}"
    yield f"last_kind={decision['last_kind']}"
    yield f"context={context}|last_kind={decision['last_kind']}"
    yield f"selected_kind={decision['selected_kind']}"
    yield f"context={context}|selected_kind={decision['selected_kind']}"
    yield f"selected_rank={decision['selected_rank_bucket']}"
    yield f"confidence={decision['confidence_bucket']}"
    yield f"context={context}|confidence={decision['confidence_bucket']}"
    if decision["agrees_with_smart"]:
        yield "agreement=smart"
    else:
        yield "agreement=not_smart"
    if decision["agrees_with_greedy"]:
        yield "agreement=greedy"
    else:
        yield "agreement=not_greedy"


def _pass_bucket(decision: dict[str, Any]) -> str:
    if decision["selected_kind"] != "Pass":
        return "not_pass"
    if decision["forced_action"]:
        return "forced_pass"
    return "optional_pass"


def _candidate_count_bucket(count: int) -> str:
    if count <= 1:
        return "01"
    if count == 2:
        return "02"
    if count <= 5:
        return "03_05"
    if count <= 10:
        return "06_10"
    if count <= 25:
        return "11_25"
    return "26_plus"


def _cards_bucket(count: int) -> str:
    if count <= 3:
        return "01_03"
    if count <= 6:
        return "04_06"
    if count <= 9:
        return "07_09"
    return "10_13"


def _rank_bucket(rank: int) -> str:
    if rank <= 3:
        return "low_3_6"
    if rank <= 8:
        return "mid_7_J"
    if rank <= 11:
        return "high_Q_A"
    return "two"


def _confidence_bucket(top_prob: float) -> str:
    if top_prob >= 0.95:
        return "very_high"
    if top_prob >= 0.75:
        return "high"
    if top_prob >= 0.50:
        return "medium"
    return "low"


def _config_from_payload(config_data: dict[str, Any]) -> Big2V2Config:
    data = dict(config_data)
    opponent_data = data.pop("opponent_mix", None)
    valid_fields = {field.name for field in fields(Big2V2Config)}
    config = Big2V2Config(**{key: value for key, value in data.items() if key in valid_fields})
    if isinstance(opponent_data, dict):
        config.opponent_mix = OpponentMixConfig(**opponent_data)
    return config


def _print_report(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(
            f"{result['opponent']} seat={result['policy_seat']} "
            f"win_rate={result['win_rate']:.3f} reward={result['average_reward']:.3f} "
            f"games={result['games']} trunc={result['truncated_candidate_lists']}"
        )
        for row in result["weaknesses"][:8]:
            print(
                "  weak "
                f"{row['bucket']}: decisions={row['decisions']} "
                f"wr={row['win_rate']:.3f} delta={row['delta_vs_overall']:.3f} "
                f"reward={row['average_reward']:.3f} top_p={row['average_top_prob']:.3f} "
                f"smart_agree={row['smart_agreement_rate']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Big2 v2 policy weaknesses by decision bucket.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-seat", type=int, default=0)
    parser.add_argument("--all-seats", action="store_true")
    parser.add_argument("--opponents", nargs="+", choices=["random", "greedy", "smart"], default=["greedy", "smart"])
    parser.add_argument("--min-decisions", type=int, default=100)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    seats = range(4) if args.all_seats else (args.policy_seat,)
    policy, _config = load_policy_from_checkpoint(
        checkpoint=checkpoint,
        device=args.device,
        max_candidates=args.max_candidates,
        num_envs=args.num_envs,
        seed=args.seed,
    )
    results = []
    for opponent in args.opponents:
        for seat in seats:
            results.append(
                diagnose_policy(
                    policy=policy,
                    opponent=opponent,
                    games=args.games,
                    num_envs=args.num_envs,
                    seed=args.seed + seat * 1000 + len(results) * 17,
                    max_candidates=args.max_candidates,
                    policy_seat=seat,
                    device=args.device,
                    min_decisions=args.min_decisions,
                )
            )
    payload = {"checkpoint": str(checkpoint), "results": results}
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _print_report(results)


if __name__ == "__main__":
    main()
