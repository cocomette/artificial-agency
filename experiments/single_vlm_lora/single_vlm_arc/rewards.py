"""Reward and progress accounting for online adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from single_vlm_arc.config import RewardConfig


@dataclass(slots=True)
class RewardBreakdown:
    score_delta: float
    learning_progress: float
    action_cost: float
    time_cost: float
    update_cost: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score_delta_from_info(previous_info: Any, next_info: Any) -> float:
    """Compute a sparse progress delta from environment info."""

    previous_levels = int(getattr(previous_info, "levels_completed", 0) or 0)
    next_levels = int(getattr(next_info, "levels_completed", 0) or 0)
    previous_wins = int(getattr(previous_info, "win_levels", 0) or 0)
    next_wins = int(getattr(next_info, "win_levels", 0) or 0)
    metadata = getattr(next_info, "metadata", {}) or {}
    raw_score_delta = metadata.get("score_delta")
    if raw_score_delta is not None:
        return float(raw_score_delta)
    return float((next_levels - previous_levels) + (next_wins - previous_wins))


def compute_reward(
    *,
    config: RewardConfig,
    score_delta: float,
    learning_progress: float,
    elapsed_seconds: float,
    update_steps: int,
) -> RewardBreakdown:
    """Combine extrinsic, intrinsic, and cost terms into one scalar reward."""

    action_cost = float(config.action_cost)
    time_cost = float(config.time_cost_weight) * max(float(elapsed_seconds), 0.0)
    update_cost = float(config.update_cost) * max(int(update_steps), 0)
    total = (
        float(config.score_weight) * float(score_delta)
        + float(config.learning_progress_weight) * float(learning_progress)
        - action_cost
        - time_cost
        - update_cost
    )
    return RewardBreakdown(
        score_delta=float(score_delta),
        learning_progress=float(learning_progress),
        action_cost=action_cost,
        time_cost=time_cost,
        update_cost=update_cost,
        total=float(total),
    )
