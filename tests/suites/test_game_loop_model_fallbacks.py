"""Tests for narrowed model-role fallbacks in the game loop."""

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    AgentTrace,
    DecisionResult,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    RuntimeConfig,
    UpdaterFrameTransitionInput,
)
from face_of_agi.debug import DebugBus
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.models.change.adapter import ChangeSummaryOutputError
from face_of_agi.models.historizer.adapter import HistorizerOutputError
from face_of_agi.models.memory import GameMemoryDocument, GameMemoryInput
from face_of_agi.models.memory.adapter import GameMemoryOutputError
from face_of_agi.models.orchestrator_agent.tooling import AgentOutputError
from face_of_agi.models.updater.adapter import UpdaterOutputError
from face_of_agi.orchestration.game_loop.fallbacks import (
    fallback_action,
    fallback_decision_result,
    is_agent_model_failure,
    is_change_model_failure,
    is_historizer_model_failure,
    is_memory_model_failure,
    is_updater_model_failure,
)
from face_of_agi.orchestration.game_loop.actions import steps
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)


def _frame_context(actions: tuple[ActionSpec, ...]) -> FrameTurnContext:
    observation = Observation(id="obs-0", step=0, frame=object())
    observation_ref = ObservationRef(memory="state", id=observation.id)
    return FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn(actions),
    )


def test_fallback_action_prefers_simple_real_action() -> None:
    action = fallback_action(
        turn_id=1,
        action_space=(
            ActionSpec.none(),
            ActionSpec(action_id="ACTION6"),
            ActionSpec(action_id="ACTION1"),
        ),
    )

    assert action == ActionSpec(action_id="ACTION1")


def test_fallback_action6_includes_environment_ready_data_and_target() -> None:
    action = fallback_action(
        turn_id=2,
        action_space=(ActionSpec(action_id="ACTION6"),),
    )

    assert action.name == "ACTION6"
    assert action.data == {"x": 16, "y": 32}
    assert action.target == "fallback probe at (16,32)"


def test_fallback_decision_result_records_agent_error_metadata() -> None:
    error = AgentOutputError("bad structured output")
    result = fallback_decision_result(
        frame_context=_frame_context((ActionSpec(action_id="ACTION1"),)),
        turn_id=1,
        action_space=(ActionSpec(action_id="ACTION1"),),
        error=error,
    )

    assert result.final_action == ActionSpec(action_id="ACTION1")
    assert result.trace.metadata["fallback"] == "agent_decision_error"
    assert result.trace.metadata["fallback_error_type"] == "AgentOutputError"


def test_model_failure_classifiers_only_accept_model_output_errors() -> None:
    assert is_agent_model_failure(AgentOutputError("bad"))
    assert is_change_model_failure(ChangeSummaryOutputError("bad"))
    assert is_historizer_model_failure(HistorizerOutputError("bad"))
    assert is_memory_model_failure(GameMemoryOutputError("bad"))
    assert is_updater_model_failure(UpdaterOutputError("bad"))

    assert not is_agent_model_failure(ValueError("programming error"))
    assert not is_change_model_failure(ValueError("programming error"))
    assert not is_historizer_model_failure(ValueError("programming error"))
    assert not is_memory_model_failure(ValueError("programming error"))
    assert not is_updater_model_failure(ValueError("programming error"))


class FailingMemoryModel:
    def summarize_game_memory(
        self,
        memory_input: GameMemoryInput,
    ) -> GameMemoryDocument:
        del memory_input
        raise GameMemoryOutputError("invalid memory JSON")


def test_memory_model_failure_keeps_previous_memory() -> None:
    action = ActionSpec(action_id="ACTION1")
    current_observation = Observation(id="obs-0", step=0, frame=object())
    next_observation = Observation(id="obs-1", step=1, frame=object())
    current_ref = ObservationRef(memory="state", id=current_observation.id)
    next_ref = ObservationRef(memory="state", id=next_observation.id)
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=1,
        ),
        game_id="game-1",
        latest_environment_observation=current_observation,
        remaining_actions=1,
        first_observation=current_observation,
        current=FrameTurnSnapshot(
            run_id="run-1",
            game_id="game-1",
            turn_id=1,
            observation=current_observation,
            observation_ref=current_ref,
            source_state_id=None,
            frame_index=0,
            frame_count=1,
            control_mode=FrameControlMode.real_environment_turn((action,)),
            first_observation_ref=current_ref,
        ),
        next=FrameTurnSnapshot(
            run_id="run-1",
            game_id="game-1",
            turn_id=2,
            observation=next_observation,
            observation_ref=next_ref,
            source_state_id=None,
            frame_index=0,
            frame_count=1,
            control_mode=None,
            first_observation_ref=current_ref,
        ),
        decision=_decision(current_observation, action),
        update_input=UpdaterFrameTransitionInput(
            current_observation_ref=current_ref,
            actual_next_observation_ref=next_ref,
            decision_trace=_decision(current_observation, action).trace,
            actual_next_observation=next_observation,
            submitted_action=action,
            action_history_entry=ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=1,
                change_summary="changed",
            ),
        ),
        game_memory=GameMemoryDocument("previous memory", metadata={"available": True}),
        game_memory_updated_this_turn=True,
    )

    steps.summarize_game_memory(
        session,
        memory_model=FailingMemoryModel(),
        debug=DebugBus.disabled(),
    )

    assert session.game_memory.markdown == "previous memory"
    assert session.game_memory.metadata["fallback"] == "game_memory_error"
    assert session.game_memory_updated_this_turn is False


def _decision(observation: Observation, action: ActionSpec) -> DecisionResult:
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=observation.step,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )
    return DecisionResult(final_action=action, trace=trace)
