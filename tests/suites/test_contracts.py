"""Smoke tests for public architecture contracts."""

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    Observation,
    ObservationRef,
    RuntimeConfig,
    ToolResult,
)


def test_contracts_import_and_compose_agent_context() -> None:
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
    tool_result = ToolResult(
        id="tool-1",
        tool="inspect",
        output={"frame": 1},
        source_observation_ref=ObservationRef(memory="state", id=observation.id),
        action=action,
    )

    assert contexts.agent.composed() == "general\n\ngame"
    assert trace.final_action.action_id == "ACTION1"
    assert tool_result.action is action
    assert tool_result.output == {"frame": 1}
    assert config.game_ids == ("game-1",)


def test_animation_unroll_keeps_real_allowed_actions() -> None:
    action = ActionSpec(action_id="ACTION1")
    control_mode = FrameControlMode.animation_unroll((action,))

    assert control_mode.controllable is False
    assert control_mode.allowed_actions == (action,)
