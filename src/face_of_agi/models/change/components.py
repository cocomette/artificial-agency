"""Deterministic component facts for model prompts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

from face_of_agi.contracts import Observation
from face_of_agi.frames import frame_to_pil_image
from face_of_agi.models.arc_grid_crop import (
    ARC_GRID_SIZE,
    crop_image_arc_grid_edges,
    normalize_arc_grid_crop_edges,
)

ARC_SYMBOL_MIN = 0
ARC_SYMBOL_MAX = 15
ARC_SYMBOL_COLOR_NAMES = {
    0: "white",
    1: "silver",
    2: "gray",
    3: "dimgray",
    4: "charcoal",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "cyan",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


@dataclass(frozen=True, slots=True)
class ChangeFrameComponent:
    """One connected component keyed by the internal ARC symbol value."""

    symbol: int
    bbox: tuple[int, int, int, int]
    shape: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ChangeFrameComponentGroup:
    """Same-color, same-shape components listed by position."""

    symbol: int
    size: int
    boxes: tuple[tuple[int, int, int, int], ...]


def frame_components_prompt_text(
    observations: Sequence[Observation],
    *,
    crop_edges: Any | None,
    max_nb_components: int = 50,
) -> str:
    """Render compact component facts for every change-summary frame."""

    component_limit = _normalized_change_summary_component_limit(max_nb_components)
    sections: list[str] = ["## Frame components"]
    grids = tuple(
        _observation_symbol_grid(observation, crop_edges=crop_edges)
        for observation in observations
    )
    for frame_index, grid in enumerate(grids):
        groups = _component_groups(
            _components(grid),
            max_nb_components=component_limit,
        )
        sections.append(_frame_components_text(frame_index, groups))
    return "\n\n".join(sections)


def current_frame_components_prompt_text(
    observation: Observation,
    *,
    crop_edges: Any | None,
    max_nb_components: int = 50,
) -> str:
    """Render compact component facts for one current updater frame."""

    component_limit = _normalized_max_nb_components(max_nb_components)
    if component_limit == 0:
        return "\n\n".join(["## Current frame components", "none"])
    groups = _component_groups(
        _components(_observation_symbol_grid(observation, crop_edges=crop_edges)),
        max_nb_components=component_limit,
    )
    lines = ["## Current frame components"]
    if not groups:
        lines.append("none")
    else:
        lines.extend(_component_group_line(group) for group in groups)
    return "\n\n".join([lines[0], "\n".join(lines[1:])])


def arc_rendered_color_map() -> dict[int, tuple[int, int, int]]:
    """Return the rendered RGB color for each ARC symbol."""

    import numpy as np
    from arc_agi.rendering import frame_to_rgb_array

    symbols = np.arange(ARC_SYMBOL_MIN, ARC_SYMBOL_MAX + 1, dtype=np.uint8).reshape(
        4,
        4,
    )
    rendered = frame_to_rgb_array(steps=0, frame=symbols, scale=1)
    return {
        int(symbols[y, x]): tuple(int(channel) for channel in rendered[y, x])
        for y in range(symbols.shape[0])
        for x in range(symbols.shape[1])
    }


def _frame_components_text(
    frame_index: int,
    groups: Sequence[ChangeFrameComponentGroup],
) -> str:
    lines = [f"frame {frame_index}:"]
    if not groups:
        lines.append("none")
    else:
        lines.extend(_component_group_line(group) for group in groups)
    return "\n".join(lines)


def _component_group_line(
    group: ChangeFrameComponentGroup,
) -> str:
    boxes = ", ".join(_box_text(box) for box in group.boxes)
    parts = [f"color={_arc_symbol_color_name(group.symbol)}"]
    parts.extend([f"nb={len(group.boxes)}", f"box=[{boxes}]"])
    return "- " + " ".join(parts)


def _box_text(box: tuple[int, int, int, int]) -> str:
    return f"({','.join(str(value) for value in box)})"


def _observation_symbol_grid(
    observation: Observation,
    *,
    crop_edges: Any | None,
) -> tuple[tuple[int, ...], ...]:
    frame = observation.frame
    if frame is None and observation.frames:
        frame = observation.frames[-1]
    if frame is None:
        raise ValueError(f"observation {observation.id!r} does not contain a frame")

    native_grid = _native_arc_grid(frame)
    if native_grid is not None:
        return _crop_native_grid(native_grid, crop_edges=crop_edges)

    image = frame_to_pil_image(frame, step=observation.step, label=observation.id)
    if not _observation_already_cropped(observation, crop_edges):
        image = crop_image_arc_grid_edges(image, crop_edges)
    return _arc_symbols_from_rendered_image(image)


def _native_arc_grid(frame: Any) -> tuple[tuple[int, ...], ...] | None:
    try:
        import numpy as np

        array = np.asarray(frame)
    except Exception:
        return None

    if array.ndim != 2:
        return None
    rows = array.tolist()
    normalized: list[tuple[int, ...]] = []
    for y, row in enumerate(rows):
        if not isinstance(row, list):
            return None
        normalized_row: list[int] = []
        for x, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"frame cell ({x},{y}) must be an ARC symbol")
            if not ARC_SYMBOL_MIN <= value <= ARC_SYMBOL_MAX:
                raise ValueError(
                    f"frame cell ({x},{y}) has ARC symbol {value}; expected 0..15"
                )
            normalized_row.append(value)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _crop_native_grid(
    grid: tuple[tuple[int, ...], ...],
    *,
    crop_edges: Any | None,
) -> tuple[tuple[int, ...], ...]:
    if not grid or not grid[0]:
        raise ValueError("component extraction requires a non-empty frame")
    edges = normalize_arc_grid_crop_edges(crop_edges)
    left, top, right, bottom = edges
    height = len(grid)
    width = len(grid[0])
    if (width, height) == (ARC_GRID_SIZE, ARC_GRID_SIZE):
        x0, y0 = left, top
        x1, y1 = width - right, height - bottom
    else:
        x0, y0 = 0, 0
        x1, y1 = width, height
    if x0 >= x1 or y0 >= y1:
        raise ValueError("component crop resolves to an empty frame")
    return tuple(row[x0:x1] for row in grid[y0:y1])


def _arc_symbols_from_rendered_image(image: Any) -> tuple[tuple[int, ...], ...]:
    reverse_palette = {
        color: symbol for symbol, color in arc_rendered_color_map().items()
    }
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    rows: list[tuple[int, ...]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            color = rgb_image.getpixel((x, y))
            try:
                row.append(reverse_palette[color])
            except KeyError as exc:
                raise ValueError(
                    "component extraction requires native ARC grids or unmodified "
                    f"ARC-rendered images; pixel ({x},{y}) has unknown color "
                    f"{_rgb_text(color)}"
                ) from exc
        rows.append(tuple(row))
    return tuple(rows)


def _components(grid: tuple[tuple[int, ...], ...]) -> tuple[ChangeFrameComponent, ...]:
    height = len(grid)
    width = len(grid[0])
    seen: set[tuple[int, int]] = set()
    components: list[ChangeFrameComponent] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen:
                continue
            cells = _component_cells(grid, x, y, seen)
            symbol = grid[y][x]
            components.append(
                _component_from_cells(
                    symbol=symbol,
                    cells=cells,
                    width=width,
                    height=height,
                )
            )
    return tuple(components)


def _component_groups(
    components: Sequence[ChangeFrameComponent],
    *,
    max_nb_components: int,
) -> tuple[ChangeFrameComponentGroup, ...]:
    grouped: dict[tuple[int, tuple[tuple[int, int], ...]], list[ChangeFrameComponent]]
    grouped = {}
    for component in components:
        grouped.setdefault((component.symbol, component.shape), []).append(component)

    groups = tuple(
        ChangeFrameComponentGroup(
            symbol=symbol,
            size=len(shape),
            boxes=tuple(sorted(component.bbox for component in group_components)),
        )
        for (symbol, shape), group_components in grouped.items()
    )
    ordered_groups = sorted(
        groups,
        key=lambda group: (-group.size, len(group.boxes), group.symbol, group.boxes),
    )

    selected: list[ChangeFrameComponentGroup] = []
    cumulative_count = 0
    for group in ordered_groups:
        next_count = cumulative_count + len(group.boxes)
        if next_count > max_nb_components:
            break
        selected.append(group)
        cumulative_count = next_count
    return tuple(selected)


def _component_cells(
    grid: tuple[tuple[int, ...], ...],
    start_x: int,
    start_y: int,
    seen: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    height = len(grid)
    width = len(grid[0])
    symbol = grid[start_y][start_x]
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    seen.add((start_x, start_y))
    cells: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        cells.append((x, y))
        for next_x, next_y in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
            if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                continue
            if (next_x, next_y) in seen or grid[next_y][next_x] != symbol:
                continue
            seen.add((next_x, next_y))
            queue.append((next_x, next_y))
    return tuple(cells)


def _component_from_cells(
    *,
    symbol: int,
    cells: tuple[tuple[int, int], ...],
    width: int,
    height: int,
) -> ChangeFrameComponent:
    xs = [x for x, _y in cells]
    ys = [y for _x, y in cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return ChangeFrameComponent(
        symbol=symbol,
        bbox=(
            _scale_edge(min_x, width),
            _scale_edge(min_y, height),
            _scale_edge(max_x + 1, width),
            _scale_edge(max_y + 1, height),
        ),
        shape=tuple(sorted((x - min_x, y - min_y) for x, y in cells)),
    )


def _observation_already_cropped(
    observation: Observation,
    crop_edges: Any | None,
) -> bool:
    metadata_edges = observation.metadata.get("change_summary_crop_edges")
    if metadata_edges is None:
        return False
    try:
        return tuple(metadata_edges) == normalize_arc_grid_crop_edges(crop_edges)
    except TypeError:
        return False


def _scale_edge(value: int, axis_size: int) -> int:
    return _clamp(round(value * 1000 / axis_size), 0, 1000)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _normalized_max_nb_components(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_nb_components must be a non-negative int")
    return value


def _normalized_change_summary_component_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_nb_components must be a positive int")
    return value


def _arc_symbol_color_name(symbol: int) -> str:
    try:
        return ARC_SYMBOL_COLOR_NAMES[symbol]
    except KeyError as exc:
        raise ValueError(f"unknown ARC symbol {symbol}") from exc


def _rgb_text(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"
