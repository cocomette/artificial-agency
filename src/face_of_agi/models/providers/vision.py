"""Reusable provider/model vision-output profile helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_of_agi.contracts import (
    CANONICAL_VISUAL_AXIS_FRAME,
    CANONICAL_VISUAL_BBOX_ORDER,
    VisualAxisFrame,
    VisualBBoxOrder,
    VisualCoordinateSpace,
)

PROFILE_PATH = Path(__file__).with_name("vision_profiles.json")


@dataclass(frozen=True, slots=True)
class ModelVisionProfile:
    """Resolved visual coordinate convention for one provider model."""

    input_image_size: tuple[int, int]
    coordinate_space: VisualCoordinateSpace
    bbox_order: VisualBBoxOrder = CANONICAL_VISUAL_BBOX_ORDER
    axis_frame: VisualAxisFrame = CANONICAL_VISUAL_AXIS_FRAME
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
    profile = profiles.get(backend_name, {}).get(model_name)
    if profile is None:
        raise ValueError(
            "unknown vision coordinate profile for "
            f"backend={backend!r}, model={model!r}; add it to "
            f"{PROFILE_PATH.name}"
        )
    return profile


def _load_profiles() -> dict[str, dict[str, ModelVisionProfile]]:
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{PROFILE_PATH.name} must contain a JSON object")

    profiles: dict[str, dict[str, ModelVisionProfile]] = {}
    for backend, models in raw.items():
        if not isinstance(backend, str) or not isinstance(models, dict):
            raise ValueError(f"{PROFILE_PATH.name} backend entries must be objects")
        normalized_models: dict[str, ModelVisionProfile] = {}
        for model, value in models.items():
            if not isinstance(model, str):
                raise ValueError(f"{PROFILE_PATH.name} model keys must be strings")
            normalized_models[_normalize(model)] = _profile(value)
        profiles[_normalize(backend)] = normalized_models
    return profiles


def _profile(value: Any) -> ModelVisionProfile:
    if not isinstance(value, dict):
        raise ValueError(f"{PROFILE_PATH.name} model profiles must be objects")
    return ModelVisionProfile(
        input_image_size=_image_size(value.get("input_image_size")),
        coordinate_space=_coordinate_space(value.get("coordinate_space")),
        bbox_order=_bbox_order(value.get("bbox_order")),
        axis_frame=_axis_frame(value.get("axis_frame")),
    )


def _image_size(value: Any) -> tuple[int, int]:
    if isinstance(value, str) and "x" in value:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    elif isinstance(value, list) and len(value) == 2:
        width, height = value
    else:
        raise ValueError(
            f"{PROFILE_PATH.name} input_image_size must be a string like "
            "'256x256' or a [width, height] array"
        )
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{PROFILE_PATH.name} input_image_size must be positive")
    return width, height


def _coordinate_space(value: Any) -> VisualCoordinateSpace:
    if value in {"pixel", "normalized_1000"}:
        return value
    raise ValueError(
        f"{PROFILE_PATH.name} coordinate_space must be pixel or normalized_1000"
    )


def _bbox_order(value: Any) -> VisualBBoxOrder:
    if value in {"xyxy", "yxyx"}:
        return value
    raise ValueError(f"{PROFILE_PATH.name} bbox_order must be xyxy or yxyx")


def _axis_frame(value: Any) -> VisualAxisFrame:
    if value == CANONICAL_VISUAL_AXIS_FRAME:
        return value
    raise ValueError(
        f"{PROFILE_PATH.name} axis_frame must be {CANONICAL_VISUAL_AXIS_FRAME}"
    )


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()
