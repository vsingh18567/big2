from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from big2.training.big2_v2.config import OpponentMixConfig, Big2V2Config


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    min_smart_greedy_score: float
    opponent_mix: OpponentMixConfig
    entropy_coef: float | None = None


@dataclass(frozen=True)
class TrainingControls:
    opponent_mix: OpponentMixConfig
    entropy_coef: float
    entropy_schedule_coef: float
    curriculum_phase: str
    smart_greedy_score: float | None

    def as_metrics(self) -> dict[str, Any]:
        return {
            "opponent_mix": asdict(self.opponent_mix),
            "entropy_coef": self.entropy_coef,
            "entropy_schedule_coef": self.entropy_schedule_coef,
            "curriculum_phase": self.curriculum_phase,
            "smart_greedy_score": self.smart_greedy_score,
        }


def training_controls_for_batch(
    config: Big2V2Config,
    *,
    batch_idx: int,
    latest_smart_greedy_score: float | None,
) -> TrainingControls:
    scheduled_entropy = entropy_coef_for_batch(config, batch_idx=batch_idx)
    if config.curriculum == "off":
        return TrainingControls(
            opponent_mix=config.opponent_mix,
            entropy_coef=scheduled_entropy,
            entropy_schedule_coef=scheduled_entropy,
            curriculum_phase="off",
            smart_greedy_score=latest_smart_greedy_score,
        )
    if config.curriculum != "smart_greedy":
        raise ValueError(f"Unsupported curriculum: {config.curriculum}")

    phase = smart_greedy_phase(config, latest_smart_greedy_score)
    return TrainingControls(
        opponent_mix=phase.opponent_mix,
        entropy_coef=phase.entropy_coef if phase.entropy_coef is not None else scheduled_entropy,
        entropy_schedule_coef=scheduled_entropy,
        curriculum_phase=phase.name,
        smart_greedy_score=latest_smart_greedy_score,
    )


def entropy_coef_for_batch(config: Big2V2Config, *, batch_idx: int) -> float:
    if config.entropy_schedule == "constant":
        return config.entropy_coef
    if config.entropy_schedule != "linear":
        raise ValueError(f"Unsupported entropy schedule: {config.entropy_schedule}")

    start = config.entropy_start_coef if config.entropy_start_coef is not None else config.entropy_coef
    end = config.entropy_end_coef if config.entropy_end_coef is not None else config.entropy_coef
    schedule_batches = config.entropy_schedule_batches if config.entropy_schedule_batches > 0 else config.batches
    if schedule_batches <= 1:
        return end
    progress = min(max((batch_idx - 1) / (schedule_batches - 1), 0.0), 1.0)
    if progress <= 0.0:
        return start
    if progress >= 1.0:
        return end
    return start + (end - start) * progress


def smart_greedy_score_from_evals(evals: dict[str, Any] | None) -> float | None:
    if not evals:
        return None
    try:
        greedy = float(evals["greedy"]["aggregate"]["win_rate"])
        smart = float(evals["smart"]["aggregate"]["win_rate"])
    except (KeyError, TypeError, ValueError):
        return None
    score = greedy + smart
    return score if math.isfinite(score) else None


def smart_greedy_phase(config: Big2V2Config, score: float | None) -> CurriculumPhase:
    phases = smart_greedy_phases(config)
    if score is None:
        return phases[0]

    selected = phases[0]
    for phase in phases[1:]:
        if score >= phase.min_smart_greedy_score:
            selected = phase
    return selected


def smart_greedy_phases(config: Big2V2Config) -> list[CurriculumPhase]:
    return [
        CurriculumPhase(
            name="base",
            min_smart_greedy_score=float("-inf"),
            opponent_mix=config.opponent_mix,
            entropy_coef=None,
        ),
        CurriculumPhase(
            name="targeted",
            min_smart_greedy_score=0.70,
            opponent_mix=OpponentMixConfig(
                learner_weight=0.65,
                random_weight=0.05,
                greedy_weight=0.10,
                smart_weight=0.10,
                checkpoint_weight=0.10,
            ),
            entropy_coef=0.012,
        ),
        CurriculumPhase(
            name="challenge",
            min_smart_greedy_score=0.80,
            opponent_mix=OpponentMixConfig(
                learner_weight=0.60,
                random_weight=0.05,
                greedy_weight=0.10,
                smart_weight=0.10,
                checkpoint_weight=0.15,
            ),
            entropy_coef=0.015,
        ),
        CurriculumPhase(
            name="plateau_breaker",
            min_smart_greedy_score=0.84,
            opponent_mix=OpponentMixConfig(
                learner_weight=0.50,
                random_weight=0.05,
                greedy_weight=0.10,
                smart_weight=0.10,
                checkpoint_weight=0.25,
            ),
            entropy_coef=0.02,
        ),
    ]
