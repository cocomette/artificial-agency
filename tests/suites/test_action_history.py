"""Tests for prompt-facing action history rendering."""

from __future__ import annotations

import pytest

from face_of_agi.contracts import ActionHistoryEntry, ActionSpec, ChangeSummaryElement
from face_of_agi.models.action_history import (
    action_history_entry_text,
    model_facing_action_text,
)


def test_action6_history_renders_target_only() -> None:
    assert (
        model_facing_action_text(
            ActionSpec(
                "ACTION6",
                data={"x": 2, "y": 3},
                target="red center tile",
            )
        )
        == 'ACTION6 target="red center tile"'
    )


def test_action6_history_requires_target() -> None:
    with pytest.raises(ValueError, match="ACTION6 action history entries require"):
        model_facing_action_text(ActionSpec("ACTION6", data={"x": 2, "y": 3}))


def test_action_history_renders_count_percent_progress_and_zero_text() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec("ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        changed_pixel_percent=0.0,
        completed_levels=2,
        action_count=7,
        change_summary="middle frame flashed white",
    )

    text = action_history_entry_text(
        entry,
        action_text=lambda action: action.name,
    )

    assert "[changed_pixels=0]" in text
    assert "[changed_area=0%]" in text
    assert "[completed_levels=2]" in text
    assert "[action_count=7]" in text
    assert (
        "change: First and final frames are identical. "
        "middle frame flashed white"
    ) in text


def test_action_history_zero_text_replaces_generic_no_change_summary() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec("ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        change_summary="no changes",
    )

    text = action_history_entry_text(
        entry,
        action_text=lambda action: action.name,
    )

    assert "change: First and final frames are identical." in text
    assert "no changes" not in text


def test_action_history_renders_change_elements_as_bullets() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec("ACTION1"),
        controllable=True,
        changed_pixel_count=4,
        change_summary="ignored when elements exist",
        change_elements=(
            ChangeSummaryElement(
                element_name="cursor",
                element_description="small white square",
                element_mutation="moved right",
            ),
        ),
    )

    text = action_history_entry_text(
        entry,
        action_text=lambda action: action.name,
    )

    assert "Elements and associated changes:" in text
    assert "- cursor: small white square; mutations: moved right" in text
    assert "change:" not in text
    assert "ignored when elements exist" not in text
