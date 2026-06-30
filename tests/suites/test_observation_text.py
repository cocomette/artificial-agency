"""Tests for native ARC observation text serialization."""

from __future__ import annotations

import pytest

from face_of_agi.contracts import Observation
from face_of_agi.models.observation_text import (
    ObservationTextConfig,
    cropped_changed_cell_count,
    serialize_observation,
    serialize_observations,
)


def _grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]


def test_serializer_crops_to_original_coordinates_and_lists_components() -> None:
    grid = _grid()
    grid[3][3] = 1
    grid[4][3] = 1
    grid[4][4] = 1
    grid[60][60] = 2

    observation_text = serialize_observation(
        Observation(id="obs", step=7, frame=grid)
    )

    assert observation_text.metadata["crop_bounds"] == (3, 3, 60, 60)
    assert "coordinate_system: original ARC grid coordinates, x/y 0..63" in (
        observation_text.text
    )
    assert "x_range: 3..60" in observation_text.text
    assert "y=03: 100" in observation_text.text
    assert "y=60:" in observation_text.text
    assert "symbol=1 area=3 bbox=(3,3,4,4)" in observation_text.text
    assert "symbol=2 area=1 bbox=(60,60,60,60)" in observation_text.text


def test_serializer_accepts_arc_symbols_10_to_15_as_hex_text() -> None:
    grid = _grid()
    for offset, symbol in enumerate(range(10, 16)):
        grid[3][3 + offset] = symbol

    observation_text = serialize_observation(
        Observation(id="hex", step=0, frame=grid)
    )

    assert "symbols: ARC symbols 0..F" in observation_text.text
    assert "ARC color symbols" not in observation_text.text
    assert "y=03: ABCDEF" in observation_text.text
    assert "- sA.1 symbol=A area=1 bbox=(3,3,3,3)" in observation_text.text
    assert "- sF.1 symbol=F area=1 bbox=(8,3,8,3)" in observation_text.text


def test_serializer_rejects_non_arc_grids() -> None:
    bad_shape = [[0 for _x in range(8)] for _y in range(8)]
    with pytest.raises(ValueError, match="64 rows"):
        serialize_observation(Observation(id="bad", step=0, frame=bad_shape))

    bad_symbol = _grid()
    bad_symbol[3][3] = 16
    with pytest.raises(ValueError, match="expected 0..15"):
        serialize_observation(Observation(id="bad", step=0, frame=bad_symbol))

    bad_float = _grid()
    bad_float[3][3] = 1.0  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must be an integer"):
        serialize_observation(Observation(id="bad", step=0, frame=bad_float))


def test_serializer_overflow_keeps_rows_and_omits_components() -> None:
    grid = _grid()
    for y in range(3, 61):
        for x in range(3, 61):
            grid[y][x] = (x + y) % 2

    observation_text = serialize_observation(
        Observation(id="dense", step=0, frame=grid),
        config=ObservationTextConfig(overflow_chars_per_frame=1),
    )

    assert "rows:" in observation_text.text
    assert "#### components" not in observation_text.text
    assert "components_omitted" not in observation_text.text
    assert observation_text.frame_texts[0].overflow is True
    assert observation_text.frame_texts[0].omitted_component_count > 0


def test_serializer_can_omit_rows_by_config() -> None:
    grid = _grid()
    grid[3][3] = 1
    grid[4][3] = 1

    observation_text = serialize_observation(
        Observation(id="no-rows", step=0, frame=grid),
        config=ObservationTextConfig(include_rows=False),
    )

    assert "### frame 0" in observation_text.text
    assert "#### rows" not in observation_text.text
    assert "x_range: 3..60" not in observation_text.text
    assert "y=03:" not in observation_text.text
    assert "#### components" in observation_text.text
    assert "symbol=1 area=2 bbox=(3,3,3,4)" in observation_text.text
    assert observation_text.frame_texts[0].rows_text
    assert observation_text.frame_texts[0].include_rows is False


def test_row_omission_excludes_rows_from_overflow_budget() -> None:
    grid = _grid()
    grid[3][3] = 1

    with_rows = serialize_observation(
        Observation(id="with-rows", step=0, frame=grid),
        config=ObservationTextConfig(overflow_chars_per_frame=200),
    )
    without_rows = serialize_observation(
        Observation(id="without-rows", step=0, frame=grid),
        config=ObservationTextConfig(
            overflow_chars_per_frame=200,
            include_rows=False,
        ),
    )

    assert "#### components" not in with_rows.text
    assert with_rows.frame_texts[0].overflow is True
    assert "#### rows" not in without_rows.text
    assert "#### components" in without_rows.text
    assert without_rows.frame_texts[0].overflow is False


def test_serializer_can_omit_components_by_config() -> None:
    grid = _grid()
    grid[3][3] = 1
    grid[4][3] = 1

    observation_text = serialize_observation(
        Observation(id="no-components", step=0, frame=grid),
        config=ObservationTextConfig(include_components=False),
    )

    assert "rows:" in observation_text.text
    assert "#### components" not in observation_text.text
    assert "symbol=1" not in observation_text.text
    assert observation_text.frame_texts[0].components == ()
    assert observation_text.frame_texts[0].components_text == ""
    assert observation_text.frame_texts[0].overflow is False
    assert observation_text.frame_texts[0].omitted_component_count == 0


def test_serializer_can_omit_component_runs_by_config() -> None:
    grid = _grid()
    grid[3][3] = 1
    grid[4][3] = 1
    grid[4][4] = 1

    observation_text = serialize_observation(
        Observation(id="no-runs", step=0, frame=grid),
        config=ObservationTextConfig(include_component_runs=False),
    )

    assert "#### components" in observation_text.text
    assert "symbol=1 area=3 bbox=(3,3,4,4)" in observation_text.text
    assert "centroid=(3.33,3.67)" in observation_text.text
    assert "runs=" not in observation_text.text
    assert observation_text.frame_texts[0].components
    assert observation_text.frame_texts[0].overflow is False
    assert observation_text.frame_texts[0].component_runs_omitted is True


def test_serializer_can_compact_components_by_config() -> None:
    grid = _grid()
    grid[3][3] = 1
    grid[5][5] = 1
    grid[7][7] = 2
    grid[7][8] = 2

    observation_text = serialize_observation(
        Observation(id="compact", step=0, frame=grid),
        config=ObservationTextConfig(compact_components=True),
    )

    assert "#### components" in observation_text.text
    assert "symbol=1 size=1 nb=2 box=[(3,3,3,3), (5,5,5,5)]" in (
        observation_text.text
    )
    assert "symbol=2 size=2 nb=1 box=[(7,7,8,7)]" in observation_text.text
    assert "centroid=" not in observation_text.text
    assert "runs=" not in observation_text.text
    assert "s1.1" not in observation_text.text
    assert observation_text.frame_texts[0].component_runs_omitted is True


def test_compact_component_deltas_keep_changed_counts_only() -> None:
    first = _grid()
    second = _grid()
    second[10][10] = 1

    observation_text = serialize_observations(
        (
            Observation(id="first", step=0, frame=first),
            Observation(id="second", step=1, frame=second),
        ),
        config=ObservationTextConfig(compact_components=True),
    )

    assert observation_text.deltas[0].changed_cell_count == 1
    assert observation_text.deltas[0].component_ids_available is False
    assert "### frame deltas" in observation_text.text
    assert "changed_cell_count: 1" in observation_text.text
    assert "old_components_touching_changed_cells:" not in observation_text.text
    assert "new_components_touching_changed_cells:" not in observation_text.text


def test_serializer_falls_back_to_compact_components_on_full_run_overflow() -> None:
    grid = _grid(fill=1)

    observation_text = serialize_observation(
        Observation(id="compact-fallback", step=0, frame=grid),
        config=ObservationTextConfig(
            overflow_chars_per_frame=200,
            include_rows=False,
            include_component_runs=True,
        ),
    )

    assert "#### rows" not in observation_text.text
    assert "#### components" in observation_text.text
    assert "symbol=1 area=3364 bbox=(3,3,60,60)" in observation_text.text
    assert "runs=" not in observation_text.text
    assert observation_text.frame_texts[0].overflow is False
    assert observation_text.frame_texts[0].omitted_component_count == 0
    assert observation_text.frame_texts[0].component_runs_omitted is True


def test_compact_component_fallback_keeps_component_delta_ids() -> None:
    first = _grid(fill=1)
    second = _grid(fill=2)

    observation_text = serialize_observations(
        (
            Observation(id="first", step=0, frame=first),
            Observation(id="second", step=1, frame=second),
        ),
        config=ObservationTextConfig(
            overflow_chars_per_frame=200,
            include_rows=False,
            include_component_runs=True,
        ),
    )

    assert "#### rows" not in observation_text.text
    assert "runs=" not in observation_text.text
    assert observation_text.deltas[0].changed_cell_count == 3364
    assert observation_text.deltas[0].component_ids_available is True
    assert "### component deltas" in observation_text.text
    assert "old_components_touching_changed_cells: s1.1" in observation_text.text
    assert "new_components_touching_changed_cells: s2.1" in observation_text.text


def test_serializer_omits_components_when_compact_fallback_still_overflows() -> None:
    grid = _grid()
    for y in range(3, 61):
        for x in range(3, 61):
            grid[y][x] = (x + y) % 2

    observation_text = serialize_observation(
        Observation(id="dense-compact-overflow", step=0, frame=grid),
        config=ObservationTextConfig(
            overflow_chars_per_frame=1,
            include_rows=False,
            include_component_runs=True,
        ),
    )

    assert "#### rows" not in observation_text.text
    assert "#### components" not in observation_text.text
    assert observation_text.frame_texts[0].overflow is True
    assert observation_text.frame_texts[0].omitted_component_count > 0
    assert observation_text.frame_texts[0].component_runs_omitted is False


def test_frame_bundle_deltas_list_changed_components_without_matching() -> None:
    first = _grid()
    second = _grid()
    first[10][10] = 1
    second[10][10] = 2
    second[11][10] = 2

    observation_text = serialize_observations(
        (
            Observation(id="first", step=0, frame=first),
            Observation(id="second", step=1, frame=second),
        )
    )

    assert observation_text.deltas[0].changed_cell_count == 2
    assert "### component deltas" in observation_text.text
    assert "delta 0->1:" in observation_text.text
    assert "changed_cell_count: 2" in observation_text.text
    assert "old_components_touching_changed_cells:" in observation_text.text
    assert "new_components_touching_changed_cells:" in observation_text.text
    assert observation_text.deltas[0].component_ids_available is True


def test_frame_bundle_deltas_omit_component_ids_when_components_overflow() -> None:
    first = _grid()
    second = _grid()
    second[10][10] = 1

    observation_text = serialize_observations(
        (
            Observation(id="first", step=0, frame=first),
            Observation(id="second", step=1, frame=second),
        ),
        config=ObservationTextConfig(overflow_chars_per_frame=1),
    )

    assert observation_text.deltas[0].changed_cell_count == 1
    assert observation_text.deltas[0].component_ids_available is False
    assert "### frame deltas" in observation_text.text
    assert "changed_cell_count: 1" in observation_text.text
    assert "old_components_touching_changed_cells:" not in observation_text.text
    assert "new_components_touching_changed_cells:" not in observation_text.text


def test_frame_bundle_deltas_omit_component_ids_when_components_disabled() -> None:
    first = _grid()
    second = _grid()
    second[10][10] = 1

    observation_text = serialize_observations(
        (
            Observation(id="first", step=0, frame=first),
            Observation(id="second", step=1, frame=second),
        ),
        config=ObservationTextConfig(include_components=False),
    )

    assert observation_text.deltas[0].changed_cell_count == 1
    assert observation_text.deltas[0].component_ids_available is False
    assert "#### components" not in observation_text.text
    assert "### frame deltas" in observation_text.text
    assert "changed_cell_count: 1" in observation_text.text
    assert "old_components_touching_changed_cells:" not in observation_text.text
    assert "new_components_touching_changed_cells:" not in observation_text.text


def test_cropped_changed_cell_count_uses_model_visible_crop() -> None:
    first = _grid()
    second = _grid()
    second[0][0] = 14
    second[63][63] = 15
    second[3][3] = 8

    assert cropped_changed_cell_count(first, second) == 1
