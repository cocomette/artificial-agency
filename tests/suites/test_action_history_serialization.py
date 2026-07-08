"""Current-shape serialization tests for action metadata."""

from __future__ import annotations

from face_of_agi.contracts import ActionHistoryEntry, ActionSpec, ChangeSummaryElement
from face_of_agi.debug.sinks.terminal import (
    _action_history_payload,
    _action_payload,
)
from face_of_agi.frames import to_memory_jsonable


def test_action_payload_serializes_target() -> None:
    action = ActionSpec(
        "ACTION6",
        data={"x": 4, "y": 5},
        target="blue center tile",
    )

    assert _action_payload(action) == {
        "action_id": "ACTION6",
        "data": {"x": 4, "y": 5},
        "target": "blue center tile",
        "requires_data": False,
    }


def test_action_history_serializes_new_metadata_fields() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec(
            "ACTION6",
            data={"x": 4, "y": 5},
            target="blue center tile",
            target_value=9,
            target_bbox=(200, 300, 400, 500),
        ),
        controllable=True,
        changed_pixel_count=12,
        changed_pixel_percent=3.125,
        completed_levels=2,
        action_count=9,
        change_summary="blue tile brightened",
        change_elements=(
            ChangeSummaryElement(
                element_name="blue tile",
                element_description="blue center tile",
                element_mutation="brightened",
            ),
        ),
    )

    debug_payload = _action_history_payload(entry)
    memory_payload = to_memory_jsonable(entry)

    assert debug_payload["action"]["target"] == "blue center tile"
    assert debug_payload["changed_pixel_percent"] == 3.125
    assert debug_payload["completed_levels"] == 2
    assert debug_payload["action_count"] == 9
    assert debug_payload["change_elements"][0]["element_name"] == "blue tile"
    assert memory_payload["action"]["target"] == "blue center tile"
    assert memory_payload["action"]["target_value"] == 9
    assert "target_bbox" not in memory_payload["action"]
    assert memory_payload["changed_pixel_percent"] == 3.125
    assert memory_payload["completed_levels"] == 2
    assert memory_payload["action_count"] == 9
    assert memory_payload["change_elements"][0]["element_mutation"] == "brightened"
