"""Tests for model-facing action-history rendering helpers."""

from __future__ import annotations

from arcengine import GameAction
import pytest

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    TurnReward,
)
from face_of_agi.models.action_history import (
    action_history_entry_text,
    model_facing_action_text,
)
from face_of_agi.orchestration.game_loop.helpers import (
    build_action_history_entry,
    prompt_action_outcome,
)


def test_model_facing_action_text_renders_action6_history_as_normalized() -> None:
    action = ActionSpec(action_id="ACTION6", data={"x": 32, "y": 20})

    assert model_facing_action_text(action) == 'ACTION6 {"x": 500, "y": 313}'


def test_model_facing_action_text_renders_action6_history_relative_to_crop() -> None:
    crop_edges = 4

    assert (
        model_facing_action_text(
            ActionSpec(action_id="ACTION6", data={"x": 4, "y": 60}),
            crop_edges=crop_edges,
        )
        == 'ACTION6 {"x": 0, "y": 1000}'
    )
    assert (
        model_facing_action_text(
            ActionSpec(action_id="ACTION6", data={"x": 32, "y": 32}),
            crop_edges=crop_edges,
        )
        == 'ACTION6 {"x": 500, "y": 500}'
    )


def test_model_facing_action_text_renders_simple_actions_unchanged() -> None:
    assert model_facing_action_text(ActionSpec(action_id="ACTION1")) == "ACTION1"
    assert (
        model_facing_action_text(ActionSpec(action_id="ACTION5", data={"x": 1}))
        == 'ACTION5 {"x": 1}'
    )


def test_model_facing_action_text_renders_action6_placeholder() -> None:
    action = ActionSpec(action_id=GameAction.ACTION6)

    assert model_facing_action_text(action) == "ACTION6(x,y normalized_0_1000)"


def test_action_history_entry_text_renders_skipped_animation_count() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec.none(),
        controllable=False,
        changed_pixel_percent=4,
        change_summary="Animation jumped to the final frame.",
        skipped_intermediate_animation_frame_count=3,
    )

    assert action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    ) == (
        "NONE [animation] [skipped_intermediate_animation_frames=3] "
        "[changed_pixel_percent=4] change: Animation jumped to the final frame."
    )


def test_action_history_entry_text_renders_reward_feedback() -> None:
    entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_percent=4,
        change_summary="The door opened.",
        reward=TurnReward(
            prediction_accuracy=0.75,
            learning_progress=0.75,
            goal_delta=0.25,
            progress_bonus=0.0,
            resource_cost=0.1,
            lp_weight=0.6,
            goal_weight=0.4,
            total=0.45,
        ),
        reward_judge_notes="close prediction",
        reward_error_tags=("minor_detail",),
    )

    assert action_history_entry_text(
        entry,
        action_text=model_facing_action_text,
    ) == (
        "ACTION1 [changed_pixel_percent=4] "
        "[reward total=0.45 learning_progress=0.75 prediction_accuracy=0.75 "
        "goal_delta=0.25 progress_bonus=0 resource_cost=0.1 "
        "errors=minor_detail notes=close prediction] change: The door opened."
    )


def test_prompt_action_outcome_suppresses_repeated_zero_percent_action() -> None:
    action1 = ActionSpec(action_id="ACTION1")
    action2 = ActionSpec(action_id="ACTION2")
    history = (
        ActionHistoryEntry(
            action=action1,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
        ActionHistoryEntry(
            action=action1,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
    )

    result = prompt_action_outcome(
        action_space=(action1, action2),
        action_history=history,
        action_suppression_zero_changed_pixel_turns=2,
        updater_stagnation_warning_zero_changed_pixel_turns=2,
    )

    assert result.allowed_actions == (action2,)
    assert result.evidence.suppressed_actions == ("ACTION1",)
    assert "changed_pixel_percent=0" in result.evidence.suppression_reason
    assert result.evidence.latest_same_action_zero_changed_pixel_turn_count == 2
    assert result.evidence.stagnation_warning is True


def test_prompt_action_outcome_nonzero_percent_breaks_zero_percent_streak() -> None:
    action1 = ActionSpec(action_id="ACTION1")
    action2 = ActionSpec(action_id="ACTION2")
    history = (
        ActionHistoryEntry(
            action=action1,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
        ActionHistoryEntry(
            action=action1,
            controllable=True,
            changed_pixel_percent=3,
            change_summary="The tile moved.",
        ),
        ActionHistoryEntry(
            action=action1,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
    )

    result = prompt_action_outcome(
        action_space=(action1, action2),
        action_history=history,
        action_suppression_zero_changed_pixel_turns=2,
        updater_stagnation_warning_zero_changed_pixel_turns=2,
    )

    assert result.allowed_actions == (action1, action2)
    assert result.evidence.suppressed_actions == ()
    assert result.evidence.latest_same_action_zero_changed_pixel_turn_count == 1
    assert result.evidence.stagnation_warning is False


def test_prompt_action_outcome_suppresses_only_repeated_action6_coordinate() -> None:
    action1 = ActionSpec(action_id="ACTION1")
    action6 = ActionSpec(action_id=GameAction.ACTION6)
    used_coordinate = ActionSpec(
        action_id=GameAction.ACTION6,
        data={"x": 32, "y": 32},
    )
    history = (
        ActionHistoryEntry(
            action=used_coordinate,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
        ActionHistoryEntry(
            action=used_coordinate,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
    )

    result = prompt_action_outcome(
        action_space=(action1, action6),
        action_history=history,
        action_suppression_zero_changed_pixel_turns=2,
        updater_stagnation_warning_zero_changed_pixel_turns=2,
    )

    assert result.allowed_actions == (action1, action6)
    assert result.evidence.suppressed_actions == ('ACTION6 {"x": 500, "y": 500}',)
    assert "ACTION6 remains available" in result.evidence.suppression_reason


def test_prompt_action_outcome_does_not_suppress_action6_without_coordinate() -> None:
    action6 = ActionSpec(action_id=GameAction.ACTION6)
    history = (
        ActionHistoryEntry(
            action=action6,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
        ActionHistoryEntry(
            action=action6,
            controllable=True,
            changed_pixel_percent=0,
            change_summary="no changes",
        ),
    )

    result = prompt_action_outcome(
        action_space=(action6,),
        action_history=history,
        action_suppression_zero_changed_pixel_turns=2,
        updater_stagnation_warning_zero_changed_pixel_turns=2,
    )

    assert result.allowed_actions == (action6,)
    assert result.evidence.suppressed_actions == ()


@pytest.mark.parametrize("value", [-0.1, 100.1, float("nan"), float("inf")])
def test_build_action_history_entry_rejects_invalid_percent(value: float) -> None:
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ObservationRef(memory="state", id="first"),
        current_observation_ref=ObservationRef(memory="state", id="current"),
        current_observation=Observation(id="current", step=1),
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id="ACTION1"),)
        ),
    )

    with pytest.raises(ValueError, match="changed_pixel_percent"):
        build_action_history_entry(
            frame_context=frame_context,
            final_action=ActionSpec(action_id="ACTION1"),
            next_observation=Observation(id="next", step=2),
            changed_pixel_percent=value,
            change_summary="changed",
        )
