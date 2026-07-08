"""Smoke tests for current public architecture contracts."""

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    NONE_ACTION_ID,
    Observation,
    ObservationRef,
    RuntimeConfig,
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

    assert contexts.agent.composed() == "general\n\ngame"
    assert trace.final_action.action_id == "ACTION1"
    assert config.game_ids == ("game-1",)


def test_none_action_marks_non_controllable_frame() -> None:
    control_mode = FrameControlMode.animation_unroll((ActionSpec.none(),))

    assert control_mode.controllable is False
    assert control_mode.allowed_actions == (ActionSpec.none(),)
    assert control_mode.allowed_actions[0].action_id == NONE_ACTION_ID
    assert control_mode.allowed_actions[0].is_none()
