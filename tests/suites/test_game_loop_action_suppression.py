"""Tests for prompt-facing no-change action suppression."""

from __future__ import annotations

from face_of_agi.contracts import ActionHistoryEntry, ActionSpec
from face_of_agi.orchestration.game_loop.helpers import prompt_action_outcome


def _entry(action: ActionSpec, *, changed_cells: int = 0) -> ActionHistoryEntry:
    return ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=changed_cells,
        change_summary="no changes",
    )


def _action6(x: int, y: int, *, target: str | None = None) -> ActionSpec:
    return ActionSpec("ACTION6", data={"x": x, "y": y}, target=target)


def test_mixed_action_space_suppresses_only_repeated_action6_coordinate() -> None:
    action_space = (ActionSpec("ACTION1"), ActionSpec("ACTION6"))
    history = tuple(
        _entry(_action6(11, 12, target=f"target {index}")) for index in range(3)
    )

    outcome = prompt_action_outcome(
        action_space=action_space,
        action_history=history,
        action_suppression_zero_changed_pixel_turns=3,
        updater_stagnation_warning_zero_changed_pixel_turns=0,
    )

    assert outcome.allowed_actions == action_space
    assert outcome.evidence.suppressed_actions == (
        'ACTION6 {"x": 11, "y": 12} target="target 2"',
    )
    assert "that coordinate" in outcome.evidence.suppression_reason
    assert "ACTION6 remains available" in outcome.evidence.suppression_reason


def test_action6_only_space_uses_coordinate_specific_suppression() -> None:
    action_space = (ActionSpec("ACTION6"),)
    history = (_entry(_action6(7, 8)), _entry(_action6(7, 8)))

    outcome = prompt_action_outcome(
        action_space=action_space,
        action_history=history,
        action_suppression_zero_changed_pixel_turns=2,
        updater_stagnation_warning_zero_changed_pixel_turns=0,
    )

    assert outcome.allowed_actions == action_space
    assert outcome.evidence.suppressed_actions == ('ACTION6 {"x": 7, "y": 8}',)
    assert outcome.evidence.suppression_disabled_reason == ""


def test_different_action6_coordinates_break_latest_streak() -> None:
    action_space = (ActionSpec("ACTION1"), ActionSpec("ACTION6"))
    history = (
        _entry(_action6(11, 12)),
        _entry(_action6(13, 12)),
        _entry(_action6(11, 12)),
    )

    outcome = prompt_action_outcome(
        action_space=action_space,
        action_history=history,
        action_suppression_zero_changed_pixel_turns=2,
        updater_stagnation_warning_zero_changed_pixel_turns=0,
    )

    assert outcome.allowed_actions == action_space
    assert outcome.evidence.suppressed_actions == ()
    assert outcome.evidence.latest_repeated_action_count == 1


def test_action6_suppression_identity_ignores_target_text() -> None:
    action_space = (ActionSpec("ACTION1"), ActionSpec("ACTION6"))
    history = (
        _entry(_action6(11, 12, target="symbol 1")),
        _entry(_action6(11, 12, target="symbol 2")),
        _entry(_action6(11, 12, target="symbol 3")),
    )

    outcome = prompt_action_outcome(
        action_space=action_space,
        action_history=history,
        action_suppression_zero_changed_pixel_turns=3,
        updater_stagnation_warning_zero_changed_pixel_turns=0,
    )

    assert outcome.allowed_actions == action_space
    assert outcome.evidence.suppressed_actions == (
        'ACTION6 {"x": 11, "y": 12} target="symbol 3"',
    )
    assert outcome.evidence.latest_repeated_action_count == 3


def test_simple_action_suppression_remains_class_level() -> None:
    action_space = (ActionSpec("ACTION1"), ActionSpec("ACTION2"), ActionSpec("ACTION6"))
    history = tuple(_entry(ActionSpec("ACTION1")) for _index in range(3))

    outcome = prompt_action_outcome(
        action_space=action_space,
        action_history=history,
        action_suppression_zero_changed_pixel_turns=3,
        updater_stagnation_warning_zero_changed_pixel_turns=0,
    )

    assert outcome.allowed_actions == (ActionSpec("ACTION2"), ActionSpec("ACTION6"))
    assert outcome.evidence.suppressed_actions == ("ACTION1",)
    assert "omitted" in outcome.evidence.suppression_reason
