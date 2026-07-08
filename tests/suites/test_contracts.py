"""Smoke tests for public architecture contracts."""

import pytest

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DescriptionPredictionError,
    FrameControlMode,
    NONE_ACTION_ID,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RuntimeConfig,
    ToolResult,
    parse_description_prediction,
    validate_description_prediction,
)


def test_contracts_import_and_compose_context() -> None:
    contexts = ContextDocuments()
    contexts.agent.general = "general"
    contexts.agent.game = "game"

    observation = Observation(id="obs-0", step=0, frame=object())
    action = ActionSpec(action_id="ACTION1")
    trace = AgentTrace(
        step=0,
        first_observation_ref=ObservationRef(memory="state", id=observation.id),
        current_observation_ref=ObservationRef(memory="state", id=observation.id),
        final_action=action,
    )
    config = RuntimeConfig(run_id="run-1", game_ids=("game-1",))
    predictions = PostDecisionPredictions(
        world_prediction=ToolResult(
            id="world-post",
            tool="world",
            predicted_description={"frame": 1},
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action,
        )
    )

    assert contexts.agent.composed() == "general\n\ngame"
    assert trace.final_action.action_id == "ACTION1"
    assert predictions.world_prediction.action is action
    assert config.game_ids == ("game-1",)


def test_none_action_marks_non_controllable_frame() -> None:
    control_mode = FrameControlMode.animation_unroll()

    assert control_mode.controllable is False
    assert control_mode.allowed_actions == (ActionSpec.none(),)
    assert control_mode.allowed_actions[0].action_id == NONE_ACTION_ID
    assert control_mode.allowed_actions[0].is_none()


def test_description_prediction_scales_normalized_1000_bboxes_to_pixels() -> None:
    prediction = parse_description_prediction(
        (
            '[{"bbox_2d":[125,250,1000,500],'
            '"description":"scaled area"}]'
        ),
        image_size=(80, 40),
        coordinate_space="normalized_1000",
    )

    assert prediction == [
        {
            "bbox_2d": [10.0, 10.0, 80.0, 20.0],
            "description": "scaled area",
        }
    ]


def test_description_prediction_rejects_object_bbox_2d() -> None:
    with pytest.raises(DescriptionPredictionError, match="expected array"):
        parse_description_prediction(
            (
                '[{"bbox_2d":{"x0":0,"y0":0,"x1":8,"y1":8},'
                '"description":"object area"}]'
            ),
            image_size=(80, 40),
        )


@pytest.mark.parametrize(
    ("bbox_2d", "expected_error"),
    [
        ([0, 0, 8], "expected 4 coordinates"),
        ([False, 0, 8, 8], r"bbox_2d\[0\]: expected number"),
        ([0, "bad", 8, 8], r"bbox_2d\[1\]: expected number"),
        ([0, 0, float("inf"), 8], r"bbox_2d\[2\]: expected number"),
    ],
)
def test_description_prediction_rejects_invalid_bbox_2d_arrays(
    bbox_2d: object,
    expected_error: str,
) -> None:
    with pytest.raises(DescriptionPredictionError, match=expected_error):
        validate_description_prediction(
            [{"bbox_2d": bbox_2d, "description": "bad area"}],
            image_size=(80, 40),
        )
