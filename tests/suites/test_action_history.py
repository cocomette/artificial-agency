"""Tests for model-facing action-history rendering helpers."""

from __future__ import annotations

from arcengine import GameAction
import pytest

from face_of_agi.contracts import ActionHistoryEntry, ActionSpec, ChangeSummaryElement
from face_of_agi.models.action_history import (
    action_history_entry_text,
    grouped_action_history_text,
    model_facing_action_text,
)


def test_model_facing_action_text_renders_action6_history_as_target() -> None:
    action = ActionSpec(
        action_id="ACTION6",
        data={"x": 32, "y": 20},
        target="the blue square near the lower center",
    )

    assert (
        model_facing_action_text(action)
        == 'ACTION6 target="the blue square near the lower center"'
    )


def test_model_facing_action_text_rejects_action6_without_target() -> None:
    action = ActionSpec(action_id="ACTION6", data={"x": 32, "y": 43})

    with pytest.raises(ValueError, match="require a target"):
        model_facing_action_text(
            action,
            crop_edges=(4, 4, 4, 4),
        )


def test_model_facing_action_text_renders_simple_actions_unchanged() -> None:
    assert model_facing_action_text(ActionSpec(action_id="ACTION1")) == "ACTION1"
    assert (
        model_facing_action_text(ActionSpec(action_id="ACTION5", data={"x": 1}))
        == 'ACTION5 {"x": 1}'
    )


def test_model_facing_action_text_renders_action6_placeholder() -> None:
    action = ActionSpec(action_id=GameAction.ACTION6)

    assert model_facing_action_text(action) == "ACTION6(x,y normalized_0_1000,target)"


def test_action_history_entry_text_renders_completed_levels_and_action_count() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=4,
        change_summary="A tile moved.",
        completed_levels=2,
        action_count=17,
    )

    assert action_history_entry_text(
        entry,
        latest=True,
        action_text=model_facing_action_text,
    ) == (
        "ACTION1 [latest] [completed_levels=2] "
        "[action_count=17] [changed_pixels=4%] "
        "Elements and associated changes:\nA tile moved."
    )


def test_action_history_entry_text_overrides_summary_when_frames_are_identical() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        change_summary="No visible change.",
    )

    assert action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    ) == (
        "ACTION1 [changed_pixels=0%] Elements and associated changes:\n"
        "No changes happened for this "
        "transition. The previous and current frames are identical"
    )


def test_action_history_entry_text_includes_carried_elements_for_noop() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        change_summary="No visible change.",
        change_elements=(
            ChangeSummaryElement(
                element_name="blue_block",
                element_description="blue block near the left wall",
                element_mutation="",
            ),
        ),
    )

    rendered = action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    )

    assert "No changes happened for this transition" in rendered
    assert "- blue_block: blue block near the left wall" in rendered


def test_action_history_entry_text_renders_animation_change_summary() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec.none(),
        controllable=False,
        changed_pixel_count=11593,
        change_summary=(
            "The white borders surrounding the square objects at the top right "
            "and bottom left corners have disappeared in the second image."
        ),
        completed_levels=0,
    )

    assert action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    ) == (
        "[animation] [completed_levels=0] [changed_pixels=11593%] "
        "Elements and associated changes:\n"
        "The white borders surrounding the square objects at the top "
        "right and bottom left corners have disappeared in the second image."
    )


def test_action_history_entry_text_renders_bundled_animation_average() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec.none(),
        controllable=False,
        changed_pixel_count=0,
        change_summary="The objects pulsed during the transition.",
        completed_levels=0,
        animation_frame_count=4,
        avg_changed_pixel_count=12.3456,
    )

    assert action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    ) == (
        "[animation: 4 frames] [completed_levels=0] "
        "[animation_avg_changed_pixels=12.3456%] "
        "Elements and associated changes:\n"
        "The objects pulsed during "
        "the transition."
    )


def test_grouped_action_history_merges_animation_into_action_line() -> None:
    action_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        change_summary="No visible change.",
        completed_levels=0,
        action_count=16,
    )
    animation_entry = ActionHistoryEntry(
        action=ActionSpec.none(),
        controllable=False,
        changed_pixel_count=0,
        change_summary="The square disappears during the animation.",
        completed_levels=0,
        animation_frame_count=2,
        avg_changed_pixel_count=1.1891,
    )

    assert grouped_action_history_text(
        (action_entry, animation_entry),
        action_text=model_facing_action_text,
        numbered=True,
    ) == (
        "1. ACTION1 [latest] [animation: 2 frames] "
        "[completed_levels=0] [action_count=16] "
        "[animation_avg_changed_pixels=1.1891%] This action triggered an animation "
        "feedback, but previous and current frame remain identical. So there is "
        "no progress but the animation teach you something. Elements and "
        "associated animation feedback changes:\n"
        "The square disappears during the animation."
    )
