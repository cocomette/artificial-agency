"""Text serialization for native ARC observation frames.

The model-facing representation is deliberately independent of provider image
APIs. It serializes the cropped native ARC grid losslessly, then adds derived
component and delta sections for easier reasoning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


ARC_GRID_SIZE = 64
ARC_SYMBOL_MIN = 0
ARC_SYMBOL_MAX = 15


@dataclass(frozen=True, slots=True)
class ObservationTextConfig:
    """Shared model-input text serialization settings."""

    crop_cells: int = 3
    overflow_chars_per_frame: int = 12000
    include_rows: bool = True
    include_components: bool = True
    include_component_runs: bool = True
    compact_components: bool = False


@dataclass(frozen=True, slots=True)
class ComponentRun:
    """One horizontal run inside a connected component."""

    y: int
    x0: int
    x1: int


@dataclass(frozen=True, slots=True)
class ComponentText:
    """Derived 4-connected same-symbol component metadata."""

    id: str
    symbol: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    runs: tuple[ComponentRun, ...]


@dataclass(frozen=True, slots=True)
class FrameText:
    """Rendered text and metadata for one cropped ARC frame."""

    index: int
    rows_text: str
    include_rows: bool
    components: tuple[ComponentText, ...]
    components_text: str
    overflow: bool
    omitted_component_count: int = 0
    component_runs_omitted: bool = False


@dataclass(frozen=True, slots=True)
class DeltaText:
    """Component-level change summary between adjacent frames."""

    from_index: int
    to_index: int
    changed_cell_count: int
    old_component_ids: tuple[str, ...]
    new_component_ids: tuple[str, ...]
    text: str
    component_ids_available: bool = True


@dataclass(frozen=True, slots=True)
class ObservationText:
    """Typed model-facing observation serialization."""

    text: str
    frame_texts: tuple[FrameText, ...]
    deltas: tuple[DeltaText, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_observation(
    observation: Any,
    *,
    config: ObservationTextConfig | None = None,
    label: str = "current_observation",
    include_header_metadata: bool = True,
) -> ObservationText:
    """Serialize a framework Observation or frame-like object as text."""

    frames = _observation_frames(observation)
    return serialize_frames(
        frames,
        config=config,
        label=label,
        observation_id=getattr(observation, "id", None),
        step=getattr(observation, "step", None),
        include_header_metadata=include_header_metadata,
    )


def serialize_observations(
    observations: Sequence[Any],
    *,
    config: ObservationTextConfig | None = None,
    label: str = "observation_frames",
    include_header_metadata: bool = True,
) -> ObservationText:
    """Serialize a sequence of single-frame observations as text."""

    frames: list[Any] = []
    ids: list[str] = []
    steps: list[Any] = []
    for observation in observations:
        observation_frames = _observation_frames(observation)
        if len(observation_frames) != 1:
            raise ValueError(
                "observation text sequence expects each observation to carry "
                "exactly one frame"
            )
        frames.append(observation_frames[0])
        ids.append(str(getattr(observation, "id", "")))
        steps.append(getattr(observation, "step", None))
    return serialize_frames(
        frames,
        config=config,
        label=label,
        observation_id=", ".join(item for item in ids if item),
        step=steps[-1] if steps else None,
        include_header_metadata=include_header_metadata,
    )


def serialize_frames(
    frames: Sequence[Any],
    *,
    config: ObservationTextConfig | None = None,
    label: str,
    observation_id: str | None = None,
    step: Any | None = None,
    include_header_metadata: bool = True,
) -> ObservationText:
    """Serialize one or more native ARC frames."""

    if not frames:
        raise ValueError("observation text requires at least one frame")
    resolved_config = config or ObservationTextConfig()
    arrays = tuple(_native_arc_grid(frame) for frame in frames)
    crop = _crop_bounds(arrays[0], resolved_config.crop_cells)
    for index, array in enumerate(arrays[1:], start=1):
        if (len(array), len(array[0])) != (ARC_GRID_SIZE, ARC_GRID_SIZE):
            raise ValueError(f"frame {index} is not a 64x64 ARC grid")

    frame_texts = tuple(
        _frame_text(
            array,
            index=index,
            crop=crop,
            config=resolved_config,
        )
        for index, array in enumerate(arrays)
    )
    deltas = tuple(
        _delta_text(
            arrays[index],
            arrays[index + 1],
            frame_texts[index].components,
            frame_texts[index + 1].components,
            from_index=index,
            crop=crop,
            component_ids_available=(
                resolved_config.include_components
                and not resolved_config.compact_components
                and not frame_texts[index].overflow
                and not frame_texts[index + 1].overflow
            ),
        )
        for index in range(len(arrays) - 1)
    )
    text = "\n\n".join(
        part
        for part in (
            _header_text(
                label=label,
                observation_id=observation_id,
                step=step,
                frame_count=len(arrays),
                crop=crop,
                config=resolved_config,
                include_metadata=include_header_metadata,
            ),
            *(_render_frame(frame_text) for frame_text in frame_texts),
            _render_deltas(deltas),
        )
        if part
    )
    return ObservationText(
        text=text,
        frame_texts=frame_texts,
        deltas=deltas,
        metadata={
            "label": label,
            "observation_id": observation_id,
            "step": step,
            "frame_count": len(arrays),
            "crop_bounds": crop,
            "coordinate_system": "original_arc_grid_0_63",
        },
    )


def cropped_changed_cell_count(
    left: Any,
    right: Any,
    *,
    config: ObservationTextConfig | None = None,
) -> int:
    """Return changed ARC cells within the model-visible crop."""

    resolved_config = config or ObservationTextConfig()
    left_grid = _native_arc_grid(left)
    right_grid = _native_arc_grid(right)
    crop = _crop_bounds(left_grid, resolved_config.crop_cells)
    if (len(right_grid), len(right_grid[0])) != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError("right frame is not a 64x64 ARC grid")
    x0, y0, x1, y1 = crop
    changed = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if left_grid[y][x] != right_grid[y][x]:
                changed += 1
    return changed


def _observation_frames(observation: Any) -> tuple[Any, ...]:
    frames = tuple(getattr(observation, "frames", ()) or ())
    if frames:
        return frames
    frame = getattr(observation, "frame", observation)
    if frame is None:
        raise ValueError("observation text requires a frame")
    return (frame,)


def _native_arc_grid(frame: Any) -> tuple[tuple[int, ...], ...]:
    try:
        import numpy as np

        array = np.asarray(frame)
        if array.ndim != 2:
            raise ValueError("frame must be a native 2D ARC grid")
        rows = array.tolist()
    except ImportError:
        rows = frame

    if not isinstance(rows, (list, tuple)):
        raise ValueError("frame must be a native 2D ARC grid")
    if len(rows) != ARC_GRID_SIZE:
        raise ValueError(f"frame must have {ARC_GRID_SIZE} rows")

    normalized_rows: list[tuple[int, ...]] = []
    for y, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != ARC_GRID_SIZE:
            raise ValueError(f"frame row {y} must have {ARC_GRID_SIZE} columns")
        normalized_row: list[int] = []
        for x, value in enumerate(row):
            if isinstance(value, bool):
                raise ValueError(f"frame cell ({x},{y}) must be an ARC symbol")
            if not isinstance(value, int):
                raise ValueError(f"frame cell ({x},{y}) must be an integer")
            symbol = value
            if not ARC_SYMBOL_MIN <= symbol <= ARC_SYMBOL_MAX:
                raise ValueError(
                    f"frame cell ({x},{y}) has ARC symbol {symbol}; expected 0..15"
                )
            normalized_row.append(symbol)
        normalized_rows.append(tuple(normalized_row))
    return tuple(normalized_rows)


def _crop_bounds(
    grid: tuple[tuple[int, ...], ...],
    crop_cells: int,
) -> tuple[int, int, int, int]:
    if crop_cells < 0:
        raise ValueError("observation_text.crop_cells must be non-negative")
    if (len(grid), len(grid[0])) != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError("observation text supports only 64x64 ARC grids")
    x0 = crop_cells
    y0 = crop_cells
    x1 = ARC_GRID_SIZE - crop_cells - 1
    y1 = ARC_GRID_SIZE - crop_cells - 1
    if x0 > x1 or y0 > y1:
        raise ValueError("observation_text.crop_cells leaves an empty crop")
    return (x0, y0, x1, y1)


def _frame_text(
    grid: tuple[tuple[int, ...], ...],
    *,
    index: int,
    crop: tuple[int, int, int, int],
    config: ObservationTextConfig,
) -> FrameText:
    rows_text = _rows_text(grid, crop)
    rendered_rows_text = rows_text if config.include_rows else ""
    if not config.include_components:
        return FrameText(
            index=index,
            rows_text=rows_text,
            include_rows=config.include_rows,
            components=(),
            components_text="",
            overflow=False,
            omitted_component_count=0,
        )
    components = _components(grid, crop)
    components_text = _components_text(
        components,
        include_runs=config.include_component_runs,
        compact=config.compact_components,
    )
    component_runs_omitted = bool(components) and (
        config.compact_components or not config.include_component_runs
    )
    overflow = (
        len(rendered_rows_text) + len(components_text)
        > config.overflow_chars_per_frame
    )
    if overflow and config.include_component_runs and not config.compact_components:
        compact_components_text = _components_text(components, include_runs=False)
        compact_overflow = (
            len(rendered_rows_text) + len(compact_components_text)
            > config.overflow_chars_per_frame
        )
        if not compact_overflow:
            components_text = compact_components_text
            component_runs_omitted = bool(components)
            overflow = False
    if overflow:
        components_text = ""
    return FrameText(
        index=index,
        rows_text=rows_text,
        include_rows=config.include_rows,
        components=components,
        components_text=components_text,
        overflow=overflow,
        omitted_component_count=len(components) if overflow else 0,
        component_runs_omitted=component_runs_omitted and not overflow,
    )


def _rows_text(
    grid: tuple[tuple[int, ...], ...],
    crop: tuple[int, int, int, int],
) -> str:
    x0, y0, x1, y1 = crop
    lines = [
        f"x_range: {x0}..{x1}",
        f"y_range: {y0}..{y1}",
        "rows:",
    ]
    for y in range(y0, y1 + 1):
        symbols = "".join(_symbol_text(grid[y][x]) for x in range(x0, x1 + 1))
        lines.append(f"y={y:02d}: {symbols}")
    return "\n".join(lines)


def _components(
    grid: tuple[tuple[int, ...], ...],
    crop: tuple[int, int, int, int],
) -> tuple[ComponentText, ...]:
    x0, y0, x1, y1 = crop
    seen: set[tuple[int, int]] = set()
    components: list[ComponentText] = []
    symbol_counts: dict[int, int] = {}
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x, y) in seen:
                continue
            symbol = grid[y][x]
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            cells = _component_cells(grid, crop, x, y, seen)
            components.append(_component_from_cells(symbol, symbol_counts[symbol], cells))
    return tuple(components)


def _component_cells(
    grid: tuple[tuple[int, ...], ...],
    crop: tuple[int, int, int, int],
    start_x: int,
    start_y: int,
    seen: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    x0, y0, x1, y1 = crop
    symbol = grid[start_y][start_x]
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    seen.add((start_x, start_y))
    cells: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        cells.append((x, y))
        for nx, ny in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
            if nx < x0 or nx > x1 or ny < y0 or ny > y1:
                continue
            if (nx, ny) in seen or grid[ny][nx] != symbol:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return tuple(sorted(cells, key=lambda item: (item[1], item[0])))


def _component_from_cells(
    symbol: int,
    suffix: int,
    cells: tuple[tuple[int, int], ...],
) -> ComponentText:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    area = len(cells)
    centroid = (sum(xs) / area, sum(ys) / area)
    return ComponentText(
        id=f"s{_symbol_text(symbol)}.{suffix}",
        symbol=symbol,
        area=area,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        centroid=(round(centroid[0], 2), round(centroid[1], 2)),
        runs=_component_runs(cells),
    )


def _component_runs(cells: tuple[tuple[int, int], ...]) -> tuple[ComponentRun, ...]:
    by_y: dict[int, list[int]] = {}
    for x, y in cells:
        by_y.setdefault(y, []).append(x)
    runs: list[ComponentRun] = []
    for y in sorted(by_y):
        sorted_xs = sorted(by_y[y])
        start = previous = sorted_xs[0]
        for x in sorted_xs[1:]:
            if x == previous + 1:
                previous = x
                continue
            runs.append(ComponentRun(y=y, x0=start, x1=previous))
            start = previous = x
        runs.append(ComponentRun(y=y, x0=start, x1=previous))
    return tuple(runs)


def _components_text(
    components: tuple[ComponentText, ...],
    *,
    include_runs: bool,
    compact: bool = False,
) -> str:
    if compact:
        return _compact_components_text(components)
    lines = ["components:"]
    for component in components:
        bbox = ",".join(str(item) for item in component.bbox)
        centroid = f"{component.centroid[0]:.2f},{component.centroid[1]:.2f}"
        line = (
            f"- {component.id} symbol={_symbol_text(component.symbol)} "
            f"area={component.area} "
            f"bbox=({bbox}) centroid=({centroid})"
        )
        if include_runs:
            runs = " ".join(_run_text(run) for run in component.runs)
            line += f" runs={runs}"
        lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _ComponentGroup:
    symbol: int
    size: int
    boxes: tuple[tuple[int, int, int, int], ...]


def _compact_components_text(
    components: tuple[ComponentText, ...],
) -> str:
    lines = ["components:"]
    for group in _component_groups(components):
        boxes = ", ".join(_box_text(box) for box in group.boxes)
        lines.append(
            f"- symbol={_symbol_text(group.symbol)} "
            f"size={group.size} "
            f"nb={len(group.boxes)} "
            f"box=[{boxes}]"
        )
    return "\n".join(lines)


def _component_groups(
    components: tuple[ComponentText, ...],
) -> tuple[_ComponentGroup, ...]:
    grouped: dict[tuple[int, tuple[tuple[int, int], ...]], list[ComponentText]] = {}
    for component in components:
        grouped.setdefault(
            (component.symbol, _component_shape(component)),
            [],
        ).append(component)

    groups = tuple(
        _ComponentGroup(
            symbol=symbol,
            size=len(shape),
            boxes=tuple(sorted(component.bbox for component in group_components)),
        )
        for (symbol, shape), group_components in grouped.items()
    )
    return tuple(
        sorted(
            groups,
            key=lambda group: (-group.size, len(group.boxes), group.symbol, group.boxes),
        )
    )


def _component_shape(component: ComponentText) -> tuple[tuple[int, int], ...]:
    x0, y0, _x1, _y1 = component.bbox
    cells: list[tuple[int, int]] = []
    for run in component.runs:
        cells.extend((x - x0, run.y - y0) for x in range(run.x0, run.x1 + 1))
    return tuple(sorted(cells, key=lambda item: (item[1], item[0])))


def _box_text(box: tuple[int, int, int, int]) -> str:
    return f"({','.join(str(value) for value in box)})"


def _symbol_text(symbol: int) -> str:
    return format(symbol, "X")


def _run_text(run: ComponentRun) -> str:
    if run.x0 == run.x1:
        return f"y{run.y}:x{run.x0}"
    return f"y{run.y}:x{run.x0}-{run.x1}"


def _delta_text(
    old_grid: tuple[tuple[int, ...], ...],
    new_grid: tuple[tuple[int, ...], ...],
    old_components: tuple[ComponentText, ...],
    new_components: tuple[ComponentText, ...],
    *,
    from_index: int,
    crop: tuple[int, int, int, int],
    component_ids_available: bool,
) -> DeltaText:
    x0, y0, x1, y1 = crop
    changed_cells = {
        (x, y)
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
        if old_grid[y][x] != new_grid[y][x]
    }
    old_ids = (
        _components_touching(old_components, changed_cells)
        if component_ids_available
        else ()
    )
    new_ids = (
        _components_touching(new_components, changed_cells)
        if component_ids_available
        else ()
    )
    lines = [
        f"delta {from_index}->{from_index + 1}:",
        f"changed_cell_count: {len(changed_cells)}",
    ]
    if component_ids_available:
        lines.extend(
            [
                "old_components_touching_changed_cells: " + _ids_text(old_ids),
                "new_components_touching_changed_cells: " + _ids_text(new_ids),
            ]
        )
    text = "\n".join(lines)
    return DeltaText(
        from_index=from_index,
        to_index=from_index + 1,
        changed_cell_count=len(changed_cells),
        old_component_ids=old_ids,
        new_component_ids=new_ids,
        text=text,
        component_ids_available=component_ids_available,
    )


def _components_touching(
    components: tuple[ComponentText, ...],
    changed_cells: set[tuple[int, int]],
) -> tuple[str, ...]:
    if not changed_cells:
        return ()
    result: list[str] = []
    for component in components:
        if _component_touches(component, changed_cells):
            result.append(component.id)
    return tuple(result)


def _component_touches(
    component: ComponentText,
    changed_cells: set[tuple[int, int]],
) -> bool:
    for run in component.runs:
        for x in range(run.x0, run.x1 + 1):
            if (x, run.y) in changed_cells:
                return True
    return False


def _ids_text(ids: Sequence[str]) -> str:
    return ", ".join(ids) if ids else "none"


def _header_text(
    *,
    label: str,
    observation_id: str | None,
    step: Any | None,
    frame_count: int,
    crop: tuple[int, int, int, int],
    config: ObservationTextConfig,
    include_metadata: bool,
) -> str:
    if not include_metadata:
        return f"## {label}"

    x0, y0, x1, y1 = crop
    return "\n".join(
        [
            f"## {label}",
            f"observation_id: {observation_id or 'unknown'}",
            f"step: {step if step is not None else 'unknown'}",
            f"frame_count: {frame_count}",
            f"original_size: {ARC_GRID_SIZE}x{ARC_GRID_SIZE}",
            f"crop_cells: {config.crop_cells}",
            f"crop_bounds_original_xyxy: ({x0},{y0},{x1},{y1})",
            f"cropped_size: {x1 - x0 + 1}x{y1 - y0 + 1}",
            "coordinate_system: original ARC grid coordinates, x/y 0..63",
            "symbols: ARC symbols 0..F",
        ]
    )


def _render_frame(frame_text: FrameText) -> str:
    parts = [f"### frame {frame_text.index}"]
    if frame_text.include_rows:
        parts.extend(["#### rows", frame_text.rows_text])
    if frame_text.components_text:
        parts.extend(["#### components", frame_text.components_text])
    return "\n\n".join(parts)


def _render_deltas(deltas: Iterable[DeltaText]) -> str:
    delta_texts = [delta.text for delta in deltas]
    if not delta_texts:
        return ""
    header = (
        "### component deltas"
        if any(delta.component_ids_available for delta in deltas)
        else "### frame deltas"
    )
    return "\n\n".join([header, *delta_texts])
