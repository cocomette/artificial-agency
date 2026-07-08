"""Short-horizon planner over the local online world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from face_of_agi.contracts import ActionSpec, PlannerCandidate
from face_of_agi.environment.config import PlannerRuntimeConfig
from face_of_agi.online.backbone import FeatureVector
from face_of_agi.online.learning import OnlineWorldModel, ValueModel


@dataclass(slots=True)
class PlannerResult:
    """Chosen action plus scored candidates."""

    action: ActionSpec
    candidates: tuple[PlannerCandidate, ...]


class ShortHorizonPlanner:
    """Deterministic short-horizon planner for currently valid actions."""

    def __init__(
        self,
        *,
        config: PlannerRuntimeConfig,
        world_model: OnlineWorldModel,
        value_model: ValueModel,
    ) -> None:
        self.config = config
        self.world_model = world_model
        self.value_model = value_model

    def choose(
        self,
        *,
        features: FeatureVector,
        action_space: Sequence[ActionSpec],
        real_turn_index: int,
    ) -> PlannerResult:
        candidates = self._candidates(action_space)
        if not candidates:
            raise RuntimeError("online planner received an empty action space")
        scored = tuple(
            self._score_candidate(features, action, real_turn_index=real_turn_index)
            for action in candidates
        )
        ordered = tuple(
            sorted(
                scored,
                key=lambda item: (
                    item.score,
                    -_action_order(item.action),
                ),
                reverse=True,
            )
        )
        return PlannerResult(action=ordered[0].action, candidates=ordered)

    def _score_candidate(
        self,
        features: FeatureVector,
        action: ActionSpec,
        *,
        real_turn_index: int,
    ) -> PlannerCandidate:
        predicted_value = self.value_model.value(action)
        uncertainty = self.world_model.uncertainty(features, action)
        action_count = self.world_model.action_count(action)
        information_gain = 1.0 / (1.0 + action_count)
        diagnostic_boost = (
            information_gain
            if real_turn_index < self.config.diagnostic_turns
            else 0.25 * information_gain
        )
        score = predicted_value + diagnostic_boost - 0.05 * uncertainty
        return PlannerCandidate(
            action=action,
            score=score,
            predicted_value=predicted_value,
            uncertainty=uncertainty,
            information_gain=information_gain,
            metadata={
                "action_count": action_count,
                "horizon": self.config.horizon,
            },
        )

    def _candidates(self, action_space: Sequence[ActionSpec]) -> tuple[ActionSpec, ...]:
        expanded: list[ActionSpec] = []
        for action in action_space:
            if action.is_complex() and action.name == "ACTION6":
                expanded.extend(_coordinate_candidates(action, self.config))
            else:
                expanded.append(action)
        return tuple(expanded[: self.config.candidate_count])


def _coordinate_candidates(
    action: ActionSpec,
    config: PlannerRuntimeConfig,
) -> tuple[ActionSpec, ...]:
    base = [
        (32, 32),
        (16, 16),
        (48, 16),
        (16, 48),
        (48, 48),
        (32, 16),
        (32, 48),
        (16, 32),
        (48, 32),
    ]
    step = max(1, 64 // max(1, config.coordinate_candidates))
    for y in range(step // 2, 64, step):
        for x in range(step // 2, 64, step):
            base.append((min(63, x), min(63, y)))
    unique: list[tuple[int, int]] = []
    for coordinate in base:
        if coordinate not in unique:
            unique.append(coordinate)
        if len(unique) >= config.coordinate_candidates:
            break
    return tuple(
        ActionSpec(action_id=action.action_id, data={"x": x, "y": y})
        for x, y in unique
    )


def _action_order(action: ActionSpec) -> int:
    name = action.name
    if name.startswith("ACTION"):
        try:
            return int(name.removeprefix("ACTION"))
        except ValueError:
            return 99
    return 99
