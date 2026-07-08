"""Tests for the updater model shell."""

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ObservationRef,
    RoleContext,
    ToolResult,
)
from face_of_agi.models.updater import (
    AgentContextUpdateInput,
    ToolContextUpdateInput,
    UpdaterAdapter,
)


def test_noop_updater_returns_same_tool_contexts() -> None:
    updater = UpdaterAdapter()
    observation_ref = ObservationRef(memory="state", id="obs-0")
    world_context = RoleContext(general="K^S", game="L^S")
    goal_context = RoleContext(general="K^G", game="L^G")

    world_result = updater.update_tool_context(
        ToolContextUpdateInput(
            role="world",
            previous_context=world_context,
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            tool_results=(
                ToolResult(
                    id="world-out",
                    tool="world",
                    predicted_observation={"frame": 1},
                    source_observation_ref=observation_ref,
                ),
            ),
        )
    )
    goal_result = updater.update_tool_context(
        ToolContextUpdateInput(
            role="goal",
            previous_context=goal_context,
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
        )
    )

    assert world_result is world_context
    assert goal_result is goal_context
    assert world_result.general == "K^S"
    assert world_result.game == "L^S"
    assert goal_result.general == "K^G"
    assert goal_result.game == "L^G"


def test_noop_updater_returns_same_agent_context() -> None:
    updater = UpdaterAdapter()
    observation_ref = ObservationRef(memory="state", id="obs-0")
    action = ActionSpec(action_id="ACTION1")
    agent_context = RoleContext(general="K^X", game="L^X")

    result = updater.update_agent_context(
        AgentContextUpdateInput(
            previous_context=agent_context,
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            trace=AgentTrace(
                step=0,
                first_observation_ref=observation_ref,
                current_observation_ref=observation_ref,
                final_action=action,
            ),
        )
    )

    assert result is agent_context
    assert result.general == "K^X"
    assert result.game == "L^X"
