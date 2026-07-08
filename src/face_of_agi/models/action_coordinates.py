"""Coordinate conversion helpers for model-facing ACTION6 data."""

from __future__ import annotations

from typing import Any

ARC_GRID_SIZE = 64
ARC_GRID_MAX = ARC_GRID_SIZE - 1
NORMALIZED_MAX = 1000
ARC_GRID_CROP_EDGE_EPSILON = 1e-9


def action6_data_from_normalized_1000(
    data: dict[str, Any],
    *,
    crop_box_normalized: Any | None = None,
) -> dict[str, int]:
    """Return ARC-grid ACTION6 data from model-visible normalized coordinates."""

    return {
        "x": normalized_1000_to_arc_grid_coordinate(
            data,
            "x",
            crop_box_normalized=crop_box_normalized,
        ),
        "y": normalized_1000_to_arc_grid_coordinate(
            data,
            "y",
            crop_box_normalized=crop_box_normalized,
        ),
    }


def action6_data_to_normalized_1000(
    data: dict[str, Any],
    *,
    crop_box_normalized: Any | None = None,
) -> dict[str, int]:
    """Return model-visible normalized ACTION6 data from ARC-grid coordinates."""

    return {
        "x": arc_grid_coordinate_to_normalized_1000(
            data,
            "x",
            crop_box_normalized=crop_box_normalized,
        ),
        "y": arc_grid_coordinate_to_normalized_1000(
            data,
            "y",
            crop_box_normalized=crop_box_normalized,
        ),
    }


def normalized_crop_box_to_arc_grid_edges(
    crop_box_normalized: Any | None,
) -> tuple[int, int, int, int]:
    """Return left/top/right/bottom ARC-grid crop edges for a normalized crop."""

    crop = _validated_crop_box(crop_box_normalized)
    if crop is None:
        return (0, 0, 0, 0)
    left, top, right, bottom = crop
    return (
        _normalized_crop_edge_to_grid_cells(left, "left"),
        _normalized_crop_edge_to_grid_cells(top, "top"),
        _normalized_crop_edge_to_grid_cells(1.0 - right, "right"),
        _normalized_crop_edge_to_grid_cells(1.0 - bottom, "bottom"),
    )


def arc_grid_edges_to_normalized_crop_box(
    crop_edges: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Return a normalized crop box from ARC-grid left/top/right/bottom edges."""

    left, top, right, bottom = crop_edges
    return (
        left / ARC_GRID_SIZE,
        top / ARC_GRID_SIZE,
        (ARC_GRID_SIZE - right) / ARC_GRID_SIZE,
        (ARC_GRID_SIZE - bottom) / ARC_GRID_SIZE,
    )


def cropped_pixel_to_arc_grid_coordinate(
    pixel: int,
    image_axis_size: int,
    key: str,
    *,
    crop_box_normalized: Any | None = None,
) -> int:
    """Map one pixel in the cropped model-visible image to ARC-grid space."""

    if image_axis_size <= 0:
        raise ValueError("image_axis_size must be positive")
    crop_edges = normalized_crop_box_to_arc_grid_edges(crop_box_normalized)
    left, top, right, bottom = crop_edges
    if key == "x":
        start = left
        visible = ARC_GRID_SIZE - left - right
    elif key == "y":
        start = top
        visible = ARC_GRID_SIZE - top - bottom
    else:
        raise ValueError(f"unsupported ACTION6 coordinate key {key!r}")
    offset = int((pixel + 0.5) * visible / image_axis_size)
    return _clamp_arc_coordinate(start + offset)


def normalized_1000_to_arc_grid_coordinate(
    data: dict[str, Any],
    key: str,
    *,
    crop_box_normalized: Any | None = None,
) -> int:
    """Map one model-visible normalized ACTION6 coordinate to ARC grid space."""

    numeric = _normalized_coordinate(data, key)
    crop = _validated_crop_box(crop_box_normalized)
    if crop is None:
        return _clamp_arc_coordinate(round(numeric * ARC_GRID_SIZE / NORMALIZED_MAX))

    start, end = _crop_axis_bounds(crop, key)
    return _clamp_arc_coordinate(
        round(start + numeric * (end - start) / NORMALIZED_MAX)
    )


def arc_grid_coordinate_to_normalized_1000(
    data: dict[str, Any],
    key: str,
    *,
    crop_box_normalized: Any | None = None,
) -> int:
    """Map one ARC-grid ACTION6 coordinate to model-visible normalized space."""

    numeric = _arc_grid_coordinate(data, key)
    crop = _validated_crop_box(crop_box_normalized)
    if crop is None:
        return _clamp_normalized_coordinate(
            int(numeric * NORMALIZED_MAX / ARC_GRID_SIZE + 0.5)
        )

    start, end = _crop_axis_bounds(crop, key)
    return _clamp_normalized_coordinate(
        int((numeric - start) * NORMALIZED_MAX / (end - start) + 0.5)
    )


def _validated_crop_box(
    value: Any | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(
            "input_image_crop_box_normalized must be a four-item list or tuple"
        )
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError("input_image_crop_box_normalized values must be numbers")
        coordinates.append(float(coordinate))
    left, top, right, bottom = coordinates
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(
            "input_image_crop_box_normalized must satisfy 0 <= left < right <= 1 "
            "and 0 <= top < bottom <= 1"
        )
    return left, top, right, bottom


def _normalized_crop_edge_to_grid_cells(value: float, label: str) -> int:
    cells = value * ARC_GRID_SIZE
    rounded = round(cells)
    if abs(cells - rounded) > ARC_GRID_CROP_EDGE_EPSILON:
        raise ValueError(
            "input_image_crop_box_normalized must align exactly with 64x64 ARC "
            f"grid cell edges for {label}; got {value!r}"
        )
    if not 0 <= rounded < ARC_GRID_SIZE:
        raise ValueError(
            "input_image_crop_box_normalized grid crop edge is outside 0..63"
        )
    return int(rounded)


def _crop_axis_bounds(
    crop: tuple[float, float, float, float],
    key: str,
) -> tuple[float, float]:
    left, top, right, bottom = crop
    if key == "x":
        return left * ARC_GRID_SIZE, right * ARC_GRID_SIZE
    if key == "y":
        return top * ARC_GRID_SIZE, bottom * ARC_GRID_SIZE
    raise ValueError(f"unsupported ACTION6 coordinate key {key!r}")


def _normalized_coordinate(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"complex action.data.{key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"complex action.data.{key} must be numeric")
    numeric = float(value)
    if not 0 <= numeric <= NORMALIZED_MAX:
        raise ValueError(
            f"complex action.data.{key} must be in normalized 0..1000"
        )
    return numeric


def _arc_grid_coordinate(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"ACTION6 data missing {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ACTION6 data {key!r} must be numeric")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"ACTION6 data {key!r} must be an ARC grid integer")
    if not 0 <= numeric <= ARC_GRID_MAX:
        raise ValueError(f"ACTION6 data {key!r} must be in ARC grid 0..63")
    return numeric


def _clamp_arc_coordinate(value: int) -> int:
    return max(0, min(value, ARC_GRID_MAX))


def _clamp_normalized_coordinate(value: int) -> int:
    return max(0, min(value, NORMALIZED_MAX))
