"""Tests for prompt-facing action history rendering."""

from __future__ import annotations

from arcengine import GameAction

from face_of_agi.contracts import ActionHistoryEntry, ActionSpec
from face_of_agi.debug.sinks.terminal import _action_history_payload, _action_payload
from face_of_agi.models.action_history import (
    action_history_entry_text,
    model_facing_action_text,
)


def test_action_history_renders_cell_metrics_and_zero_change_wording() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec("ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        changed_cell_percent=0.0,
        completed_levels=2,
        action_count=7,
        change_summary="no changes",
    )

    text = action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    )

    assert "[changed_cells=0]" in text
    assert "[changed_cells_pct=0%]" in text
    assert "[completed_levels=2]" in text
    assert "[action_count=7]" in text
    assert "change: First and final frames are identical." in text
    assert "changed_pixels" not in text


def test_action_history_renders_action6_coordinates_and_optional_target() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec(
            "ACTION6",
            data={"x": 11, "y": 12},
            target="symbol 4 cell",
        ),
        controllable=True,
        changed_pixel_count=3,
        changed_cell_percent=0.0892,
        change_summary="symbol 4 changed",
    )

    text = action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    )

    assert 'ACTION6 {"x": 11, "y": 12} target="symbol 4 cell"' in text
    assert "[changed_cells=3]" in text
    assert "[changed_cells_pct=0.0892%]" in text


def test_action_history_allows_historical_action6_without_target() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec("ACTION6", data={"x": 11, "y": 12}),
        controllable=True,
        changed_pixel_count=1,
        change_summary="changed",
    )

    text = action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    )

    assert 'ACTION6 {"x": 11, "y": 12}' in text
    assert "target=" not in text


def test_debug_action_payload_includes_target_and_history_metrics() -> None:
    action = ActionSpec(GameAction.ACTION6, data={"x": 1, "y": 2}, target="corner")
    entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=4,
        changed_cell_percent=0.12,
        completed_levels=1,
        action_count=9,
        change_summary="changed",
    )

    assert _action_payload(action) == {
        "action_id": "ACTION6",
        "data": {"x": 1, "y": 2},
        "target": "corner",
        "requires_data": True,
    }
    payload = _action_history_payload(entry)
    assert payload["changed_cell_percent"] == 0.12
    assert payload["completed_levels"] == 1
    assert payload["action_count"] == 9
