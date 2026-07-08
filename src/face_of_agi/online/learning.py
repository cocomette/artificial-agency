"""Small online world-model, replay, and value components."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import random
from time import perf_counter
from typing import Any

from face_of_agi.contracts import ActionSpec, ReplayStats, TransitionRecord
from face_of_agi.environment.config import OnlineRuntimeConfig, ReplayRuntimeConfig
from face_of_agi.online.backbone import FeatureVector


@dataclass(slots=True)
class EncodedTransition:
    """Feature-space transition used for replay and dynamics updates."""

    id: str
    previous: FeatureVector
    action: ActionSpec
    next: FeatureVector
    record: TransitionRecord
    priority: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TransitionBuffer:
    """Bounded prioritized transition buffer."""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("transition buffer size must be at least 1")
        self.max_size = max_size
        self._items: deque[EncodedTransition] = deque(maxlen=max_size)

    def add(self, transition: EncodedTransition) -> None:
        self._items.append(transition)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, count: int) -> tuple[EncodedTransition, ...]:
        if count <= 0 or not self._items:
            return ()
        ordered = sorted(self._items, key=lambda item: item.priority, reverse=True)
        return tuple(ordered[: min(count, len(ordered))])

    def summary(self) -> dict[str, Any]:
        priorities = [item.priority for item in self._items]
        return {
            "size": len(self._items),
            "max_size": self.max_size,
            "max_priority": max(priorities) if priorities else 0.0,
            "mean_priority": (
                sum(priorities) / len(priorities) if priorities else 0.0
            ),
        }


class OnlineWorldModel:
    """Action-conditioned local latent dynamics with a small ensemble."""

    def __init__(self, config: OnlineRuntimeConfig) -> None:
        self.learning_rate = config.learning_rate
        self.ensemble_size = config.ensemble_size
        self._deltas: list[dict[str, list[float]]] = [
            {} for _ in range(config.ensemble_size)
        ]
        self._counts: dict[str, int] = defaultdict(int)
        self._rng = random.Random(0)

    def predict(self, features: FeatureVector, action: ActionSpec) -> FeatureVector:
        key = action_key(action)
        predictions = self._member_predictions(features, key)
        return _mean_vectors(predictions)

    def uncertainty(self, features: FeatureVector, action: ActionSpec) -> float:
        key = action_key(action)
        predictions = self._member_predictions(features, key)
        if len(predictions) <= 1:
            return 1.0
        mean = _mean_vectors(predictions)
        return sum(_mse(prediction, mean) for prediction in predictions) / len(
            predictions
        )

    def prediction_error(
        self,
        previous: FeatureVector,
        action: ActionSpec,
        observed_next: FeatureVector,
    ) -> float:
        return _mse(self.predict(previous, action), observed_next)

    def update(self, transition: EncodedTransition) -> float:
        key = action_key(transition.action)
        observed_delta = _sub_vectors(transition.next, transition.previous)
        member_errors: list[float] = []
        for index, deltas in enumerate(self._deltas):
            current = deltas.get(key)
            if current is None:
                jitter = 1.0 + (index - self.ensemble_size / 2.0) * 0.01
                current = [value * jitter for value in observed_delta]
            predicted_next = _add_vectors(transition.previous, tuple(current))
            member_errors.append(_mse(predicted_next, transition.next))
            deltas[key] = [
                old + self.learning_rate * (target - old)
                for old, target in zip(current, observed_delta, strict=False)
            ]
        self._counts[key] += 1
        return sum(member_errors) / len(member_errors)

    def action_count(self, action: ActionSpec) -> int:
        return int(self._counts.get(action_key(action), 0))

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "online_world_model.v1",
            "action_counts": dict(self._counts),
            "ensemble_size": self.ensemble_size,
            "learning_rate": self.learning_rate,
        }

    def _member_predictions(
        self,
        features: FeatureVector,
        key: str,
    ) -> tuple[FeatureVector, ...]:
        predictions: list[FeatureVector] = []
        for deltas in self._deltas:
            delta = tuple(deltas.get(key, (0.0,) * len(features)))
            predictions.append(_add_vectors(features, delta))
        return tuple(predictions)


class ValueModel:
    """Small action-value head trained from progress and visible change."""

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self._values: dict[str, float] = {}

    def value(self, action: ActionSpec) -> float:
        return float(self._values.get(action_key(action), 0.0))

    def update(self, transition: EncodedTransition) -> None:
        key = action_key(transition.action)
        score_delta = transition.record.score_delta or 0.0
        reward = score_delta + transition.record.changed_pixel_percent / 1000.0
        current = self._values.get(key, 0.0)
        self._values[key] = current + self.learning_rate * (reward - current)

    def snapshot(self) -> dict[str, Any]:
        return {"type": "online_value_model.v1", "values": dict(self._values)}


class ReplayTrainer:
    """Run bounded world/value updates from the transition buffer."""

    def __init__(
        self,
        *,
        config: ReplayRuntimeConfig,
        buffer: TransitionBuffer,
        world_model: OnlineWorldModel,
        value_model: ValueModel,
    ) -> None:
        self.config = config
        self.buffer = buffer
        self.world_model = world_model
        self.value_model = value_model

    def update_after_real_transition(
        self,
        transition: EncodedTransition,
        *,
        completed_level: bool,
    ) -> ReplayStats:
        started = perf_counter()
        errors = [self.world_model.update(transition)]
        self.value_model.update(transition)
        max_updates = self.config.max_updates_per_turn
        if completed_level:
            max_updates += self.config.solved_level_updates
        sampled_ids: list[str] = []
        replay_count = 0
        while replay_count < max_updates:
            if perf_counter() - started >= self.config.max_seconds_per_turn:
                break
            sample = self.buffer.sample(1)
            if not sample:
                break
            item = sample[0]
            sampled_ids.append(item.id)
            errors.append(self.world_model.update(item))
            self.value_model.update(item)
            replay_count += 1
        return ReplayStats(
            real_update_count=1,
            replay_update_count=replay_count,
            elapsed_seconds=perf_counter() - started,
            sampled_transition_ids=tuple(sampled_ids),
            mean_prediction_error=sum(errors) / len(errors) if errors else None,
        )


def action_key(action: ActionSpec) -> str:
    data = action.data or {}
    if action.name == "ACTION6" and {"x", "y"} <= set(data):
        x_bucket = int(data["x"]) // 8
        y_bucket = int(data["y"]) // 8
        return f"{action.name}:{x_bucket}:{y_bucket}"
    return action.name


def transition_priority(record: TransitionRecord) -> float:
    error = record.prediction_error or 0.0
    score = abs(record.score_delta or 0.0)
    changed = record.changed_pixel_percent / 100.0
    controllable = 0.25 if record.controllable else 0.0
    level = 1.0 if score > 0 else 0.0
    return error + score + changed + controllable + level


def _add_vectors(left: FeatureVector, right: FeatureVector) -> FeatureVector:
    return tuple(a + b for a, b in zip(left, right, strict=False))


def _sub_vectors(left: FeatureVector, right: FeatureVector) -> FeatureVector:
    return tuple(a - b for a, b in zip(left, right, strict=False))


def _mean_vectors(vectors: tuple[FeatureVector, ...]) -> FeatureVector:
    if not vectors:
        return ()
    width = len(vectors[0])
    return tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(width))


def _mse(left: FeatureVector, right: FeatureVector) -> float:
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    return sum((left[index] - right[index]) ** 2 for index in range(width)) / width
