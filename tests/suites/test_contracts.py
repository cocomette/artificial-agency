"""Smoke tests for public architecture contracts."""

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    NONE_ACTION_ID,
    Observation,
    ObservationRef,
    RuntimeConfig,
    ToolResult,
    TurnMetrics,
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
        tool_results=[
            ToolResult(
                id="tool-1",
                tool="world",
                output={"summary": "unused test double"},
                source_observation_ref=ObservationRef(
                    memory="state",
                    id=observation.id,
                ),
            )
        ],
    )
    config = RuntimeConfig(run_id="run-1", game_ids=("game-1",))
    metrics = TurnMetrics(time_cost=1.0, trace_cost=0.25, cumulative_score=2.0)

    assert contexts.agent.composed() == "general\n\ngame"
    assert trace.final_action.action_id == "ACTION1"
    assert trace.tool_results[0].output == {"summary": "unused test double"}
    assert config.game_ids == ("game-1",)
    assert metrics.cumulative_score == 2.0


def test_none_action_marks_non_controllable_frame() -> None:
    action = ActionSpec.none()
    control_mode = FrameControlMode.animation_unroll((action,))

    assert control_mode.controllable is False
    assert control_mode.allowed_actions == (action,)
    assert control_mode.allowed_actions[0].action_id == NONE_ACTION_ID
    assert control_mode.allowed_actions[0].is_none()
