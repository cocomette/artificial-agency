"""ARC-grid-relative image crop and ACTION6 coordinate helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ARC_GRID_SIZE = 64
ArcGridCropEdges = tuple[int, int, int, int]


def normalize_arc_grid_crop_edges(value: Any) -> ArcGridCropEdges:
    """Return left/top/right/bottom crop edges in ARC 64x64 grid cells."""

    if value is None:
        return (0, 0, 0, 0)
    if isinstance(value, bool):
        raise ValueError("input_image_crop_arc_grid_edges must be an int or 4 ints")
    if isinstance(value, int):
        edges = (value, value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise ValueError("input_image_crop_arc_grid_edges must have 4 values")
        edges = tuple(_crop_edge(item) for item in value)
    else:
        raise ValueError("input_image_crop_arc_grid_edges must be an int or 4 ints")

    left, top, right, bottom = edges
    if left + right >= ARC_GRID_SIZE or top + bottom >= ARC_GRID_SIZE:
        raise ValueError("input_image_crop_arc_grid_edges leaves no visible frame")
    return edges


def crop_image_arc_grid_edges(image: Any, crop_edges: Any) -> Any:
    """Crop a PIL image using edges expressed in source ARC grid cells."""

    box = arc_grid_crop_box(image.size, crop_edges)
    if box == (0, 0, image.size[0], image.size[1]):
        return image
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("input_image_crop_arc_grid_edges resolves to an empty crop")
    return image.crop(box)


def arc_grid_crop_box(
    image_size: tuple[int, int],
    crop_edges: Any,
) -> tuple[int, int, int, int]:
    """Return the PIL crop box for ARC-grid crop edges."""

    edges = normalize_arc_grid_crop_edges(crop_edges)
    left, top, right, bottom = edges
    width, height = image_size
    return (
        _scaled_edge(left, width),
        _scaled_edge(top, height),
        width - _scaled_edge(right, width),
        height - _scaled_edge(bottom, height),
    )


def arc_grid_to_normalized_1000(
    data: dict[str, Any],
    key: str,
    *,
    crop_edges: Any = None,
) -> int:
    """Render one full-frame ARC grid coordinate in crop-relative 0..1000 space."""

    if key not in data:
        raise ValueError(f"ACTION6 data missing {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ACTION6 data {key!r} must be numeric")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"ACTION6 data {key!r} must be an ARC grid integer")
    coordinate = int(numeric)
    if not 0 <= coordinate < ARC_GRID_SIZE:
        raise ValueError(f"ACTION6 data {key!r} must be in ARC grid 0..63")

    start, visible = _axis_start_and_visible(key, crop_edges)
    return _clamp(int((coordinate - start) * 1000 / visible + 0.5), 0, 1000)


def normalized_1000_to_arc_grid(
    value: float,
    key: str,
    *,
    crop_edges: Any = None,
) -> int:
    """Map one crop-relative normalized ACTION6 coordinate to ARC grid space."""

    if not 0 <= value <= 1000:
        raise ValueError(f"complex action.data.{key} must be in normalized 0..1000")
    start, visible = _axis_start_and_visible(key, crop_edges)
    end = start + visible - 1
    return _clamp(start + round(value * visible / 1000), start, end)


def _crop_edge(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("input_image_crop_arc_grid_edges values must be ints")
    if value < 0:
        raise ValueError("input_image_crop_arc_grid_edges values must be non-negative")
    return value


def _axis_start_and_visible(key: str, crop_edges: Any) -> tuple[int, int]:
    left, top, right, bottom = normalize_arc_grid_crop_edges(crop_edges)
    if key == "x":
        return left, ARC_GRID_SIZE - left - right
    if key == "y":
        return top, ARC_GRID_SIZE - top - bottom
    raise ValueError(f"unsupported ACTION6 coordinate axis: {key!r}")


def _scaled_edge(edge: int, image_axis_size: int) -> int:
    return int(edge * image_axis_size / ARC_GRID_SIZE + 0.5)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
