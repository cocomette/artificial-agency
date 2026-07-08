"""Reusable provider/model vision-output profile helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_of_agi.contracts import VisualCoordinateSpace

PROFILE_PATH = Path(__file__).with_name("vision_profiles.json")


@dataclass(frozen=True, slots=True)
class ModelVisionProfile:
    """Resolved visual coordinate convention for one provider model."""

    coordinate_space: VisualCoordinateSpace
    source: str = "model_profile"


def resolve_model_vision_profile(
    *,
    backend: str | None,
    model: str | None,
) -> ModelVisionProfile:
    """Return the model-native visual coordinate convention.

    Coordinates are a property of the model/provider pair, not of a specific
    output field. Unknown models must be added to vision_profiles.json instead
    of configured per run.
    """

    backend_name = _normalize(backend)
    model_name = _normalize(model)
    profiles = _load_profiles()
    coordinate_space = profiles.get(backend_name, {}).get(model_name)
    if coordinate_space is None:
        raise ValueError(
            "unknown vision coordinate profile for "
            f"backend={backend!r}, model={model!r}; add it to "
            f"{PROFILE_PATH.name}"
        )
    return ModelVisionProfile(coordinate_space=coordinate_space)


def _load_profiles() -> dict[str, dict[str, VisualCoordinateSpace]]:
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{PROFILE_PATH.name} must contain a JSON object")

    profiles: dict[str, dict[str, VisualCoordinateSpace]] = {}
    for backend, models in raw.items():
        if not isinstance(backend, str) or not isinstance(models, dict):
            raise ValueError(f"{PROFILE_PATH.name} backend entries must be objects")
        normalized_models: dict[str, VisualCoordinateSpace] = {}
        for model, value in models.items():
            if not isinstance(model, str):
                raise ValueError(f"{PROFILE_PATH.name} model keys must be strings")
            normalized_models[_normalize(model)] = _coordinate_space(value)
        profiles[_normalize(backend)] = normalized_models
    return profiles


def _coordinate_space(value: Any) -> VisualCoordinateSpace:
    if value in {"pixel", "normalized_1000"}:
        return value
    raise ValueError(
        f"{PROFILE_PATH.name} coordinate_space must be pixel or normalized_1000"
    )


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()
