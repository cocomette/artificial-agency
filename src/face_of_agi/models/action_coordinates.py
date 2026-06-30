"""Coordinate validation helpers for model-facing ACTION6 data."""

from __future__ import annotations

from typing import Any

from face_of_agi.models.observation_text import ARC_GRID_SIZE, ObservationTextConfig

ARC_GRID_MIN = 0
ARC_GRID_MAX = ARC_GRID_SIZE - 1


def action6_coordinate_bounds(
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Return model-visible ACTION6 coordinate bounds for an observation crop."""

    config = _observation_text_config(observation_text_config)
    minimum = config.crop_cells
    maximum = ARC_GRID_MAX - config.crop_cells
    if minimum < ARC_GRID_MIN:
        raise ValueError("observation_text.crop_cells must be non-negative")
    if minimum > maximum:
        raise ValueError("observation_text.crop_cells leaves an empty ACTION6 range")
    return minimum, maximum


def action6_coordinate_range_text(
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    """Return compact prompt text for the current model-visible ACTION6 range."""

    minimum, maximum = action6_coordinate_bounds(observation_text_config)
    return f"{minimum}..{maximum}"


def action6_coordinate_range_phrase(
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    """Return prose prompt text for the current model-visible ACTION6 range."""

    minimum, maximum = action6_coordinate_bounds(observation_text_config)
    return f"{minimum} to {maximum}"


def action6_data_from_visible_crop(
    data: dict[str, Any],
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> dict[str, int]:
    """Return ACTION6 data validated against the model-visible crop."""

    minimum, maximum = action6_coordinate_bounds(observation_text_config)
    return {
        "x": _integer_coordinate(data, "x", minimum=minimum, maximum=maximum),
        "y": _integer_coordinate(data, "y", minimum=minimum, maximum=maximum),
    }


def action6_data_from_arc_grid(data: dict[str, Any]) -> dict[str, int]:
    """Return validated ARC-grid ACTION6 data from model output."""

    return {
        "x": _integer_coordinate(
            data,
            "x",
            minimum=ARC_GRID_MIN,
            maximum=ARC_GRID_MAX,
            range_label="ARC grid",
        ),
        "y": _integer_coordinate(
            data,
            "y",
            minimum=ARC_GRID_MIN,
            maximum=ARC_GRID_MAX,
            range_label="ARC grid",
        ),
    }


def arc_grid_coordinate(data: dict[str, Any], key: str) -> int:
    """Return one integer ARC-grid coordinate from action data."""

    return _integer_coordinate(
        data,
        key,
        minimum=ARC_GRID_MIN,
        maximum=ARC_GRID_MAX,
        range_label="ARC grid",
    )


def _integer_coordinate(
    data: dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
    range_label: str = "visible crop",
) -> int:
    if key not in data:
        raise ValueError(f"ACTION6 data missing {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ACTION6 data {key!r} must be numeric")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"ACTION6 data {key!r} must be an ARC grid integer")
    if not minimum <= numeric <= maximum:
        raise ValueError(
            f"ACTION6 data {key!r} must be in {range_label} {minimum}..{maximum}"
        )
    return int(numeric)


def _observation_text_config(
    value: ObservationTextConfig | dict[str, Any] | None,
) -> ObservationTextConfig:
    if value is None:
        return ObservationTextConfig()
    if isinstance(value, ObservationTextConfig):
        return value
    if isinstance(value, dict):
        return ObservationTextConfig(**value)
    raise TypeError("observation_text_config must be an ObservationTextConfig")
