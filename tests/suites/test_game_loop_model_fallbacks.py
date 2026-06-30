"""Tests for model-error fallbacks in the orchestration game loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from arcengine import GameAction

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DecisionResult,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    RoleContext,
    RuntimeConfig,
    TurnMetrics,
    UpdaterFrameTransitionInput,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.models.historizer import (
    AgentContextHistoryInput,
    AgentContextHistorySummary,
)
from face_of_agi.models.change.adapter import ChangeSummaryOutputError
from face_of_agi.models.historizer.adapter import HistorizerOutputError
from face_of_agi.models.observation_text import ObservationTextConfig
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.models.updater.adapter import UpdaterOutputError
from face_of_agi.orchestration.game_loop import GameLoopStateMachine
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_context_updates,
    summarize_agent_context_history,
)
from face_of_agi.orchestration.game_loop.actions.steps import (
    enter_frame_turn,
    persist as persist_step,
    summarize_change_model,
)
from face_of_agi.orchestration.game_loop.actions.steps import decide as decide_step
from face_of_agi.orchestration.game_loop.helpers import decide_frame_turn
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)
from face_of_agi.debug.bus import DebugBus


class FailingAgent:
    """Agent test double that always fails before returning a decision."""

    def __init__(self, observation_text: ObservationTextConfig | None = None) -> None:
        self.config = type(
            "FakeAgentConfig",
            (),
            {"observation_text": observation_text or ObservationTextConfig()},
        )()
        self.last_provider_requests: list[dict[str, str]] = []

    def decide(self, *args: Any, **kwargs: Any) -> DecisionResult:
        raise RuntimeError(
            "vllm X produced invalid structured agent step after 3 repair "
            "attempt(s): invalid model output"
        )


class NonModelFailingAgent(FailingAgent):
    """Agent test double that fails before the model/provider boundary."""

    def decide(self, *args: Any, **kwargs: Any) -> DecisionResult:
        raise ValueError("prompt construction bug")


class InvalidActionAgent:
    """Agent test double that returns a structurally invalid final action."""

    config = type(
        "FakeAgentConfig",
        (),
        {"observation_text": ObservationTextConfig()},
    )()
    last_provider_requests: list[dict[str, str]] = []

    def decide(self, *args: Any, **kwargs: Any) -> DecisionResult:
        observation = kwargs["current_observation"]
        ref = ObservationRef(memory="state", id=observation.id)
        action = ActionSpec("NOT_ALLOWED")
        return DecisionResult(
            final_action=action,
            trace=AgentTrace(
                step=observation.step,
                first_observation_ref=ref,
                current_observation_ref=ref,
                final_action=action,
            ),
        )


class FailingChangeModel:
    """Change-summary test double that simulates provider/output failure."""

    config = type(
        "FakeChangeConfig",
        (),
        {"observation_text": ObservationTextConfig()},
    )()

    def summarize(self, *args: Any, **kwargs: Any) -> Any:
        raise ChangeSummaryOutputError("bad change JSON")


class NonModelFailingChangeModel(FailingChangeModel):
    """Change model test double that simulates deterministic framework failure."""

    def summarize(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad frame")


class FailingHistorizer:
    """Historizer test double that simulates unrepairable output."""

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        raise HistorizerOutputError("bad history JSON")


class NonModelFailingHistorizer(FailingHistorizer):
    """Historizer test double that simulates deterministic framework failure."""

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        raise ValueError("bad context history")


class FakeStateMemory:
    """State-memory test double with enough context history for historizer use."""

    def read_agent_game_context_history(self, **kwargs: Any) -> tuple[str, ...]:
        return ('{"goals": "old"}', '{"goals": "new"}')


class FailingAgentUpdater:
    """Updater test double that always fails."""

    def update_agent_game_context(self, update_input: Any) -> RoleContext:
        raise UpdaterOutputError("bad updater JSON")


class NonModelFailingAgentUpdater(FailingAgentUpdater):
    """Updater test double that simulates deterministic framework failure."""

    def update_agent_game_context(self, update_input: Any) -> RoleContext:
        raise ValueError("bad updater input")


class PrewriteFailingStateMemory:
    """State-memory test double that fails before a frame turn is recorded."""

    def prewrite_frame_turn_source(self, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("prewrite failed")


class CompleteFailingStateMemory:
    """State-memory test double that fails while completing a frame turn."""

    def complete_frame_turn_state(self, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("complete failed")


class LifecycleFailingEnvironment:
    """Environment test double that fails after startup."""

    def __init__(self) -> None:
        self.game_id = "game-1"

    def select_game_by_id(self, game_id: str) -> str:
        self.game_id = game_id
        return game_id

    def reset(self) -> Observation:
        return Observation(id="initial", step=0, frame=_grid())

    def get_info(self) -> Any:
        raise RuntimeError("lifecycle failed")


def _grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]


def _frame_context(
    *,
    action_space: Sequence[ActionSpec],
    turn_id: int = 1,
) -> FrameTurnContext:
    observation = Observation(id=f"obs-{turn_id}", step=turn_id, frame=_grid())
    ref = ObservationRef(memory="state", id=observation.id)
    return FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=observation,
        current_source_state_id=turn_id,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn(tuple(action_space)),
    )


def _transition_session() -> GameLoopSession:
    current_grid = _grid(0)
    next_grid = _grid(0)
    next_grid[10][10] = 4
    current = Observation(id="current", step=0, frame=current_grid)
    next_observation = Observation(id="next", step=1, frame=next_grid)
    current_ref = ObservationRef(memory="state", id="current")
    next_ref = ObservationRef(memory="state", id="next")
    action = ActionSpec("ACTION1")
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(game_index=0, max_actions_per_level=10),
        game_id="game-1",
        latest_environment_observation=current,
        remaining_actions=10,
        real_actions=(action,),
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=current,
        observation_ref=current_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=current_ref,
    )
    session.next = FrameTurnSnapshot(
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
        previous_observation_ref=current_ref,
    )
    session.decision = DecisionResult(
        final_action=action,
        trace=AgentTrace(
            step=0,
            first_observation_ref=current_ref,
            current_observation_ref=current_ref,
            final_action=action,
        ),
    )
    session.transition_frame_observations = (current, next_observation)
    return session


def _apply_context_update_with(updater: Any, *, contexts: ContextDocuments) -> None:
    action = ActionSpec("ACTION1")
    current = Observation(id="current", step=0, frame=_grid())
    next_observation = Observation(id="next", step=1, frame=_grid())
    current_ref = ObservationRef(memory="state", id="current")
    next_ref = ObservationRef(memory="state", id="next")
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=current_ref,
        current_observation_ref=current_ref,
        current_observation=current,
        current_source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )
    update_input = UpdaterFrameTransitionInput(
        current_observation_ref=current_ref,
        actual_next_observation_ref=next_ref,
        decision_trace=AgentTrace(
            step=0,
            first_observation_ref=current_ref,
            current_observation_ref=current_ref,
            final_action=action,
        ),
        actual_next_observation=next_observation,
        turn_metrics=TurnMetrics(),
        submitted_action=action,
        action_history_entry=ActionHistoryEntry(
            action=action,
            controllable=True,
            changed_pixel_count=0,
            change_summary="no changes",
        ),
    )
    apply_context_updates(
        update_input,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_game_updater=updater),
        debug=DebugBus.disabled(),
        state_memory=None,
        frame_context=frame_context,
        prior_action_history=(),
        agent_updater_action_history_window=3,
        agent_context_history=AgentContextHistorySummary.not_available(),
        action_suppression_zero_changed_pixel_turns=3,
        updater_stagnation_warning_zero_changed_pixel_turns=3,
        game_last_started_turns_ago=0,
        score_last_advanced_turns_ago=None,
        game_start_reason="initial_start",
        game_restart_count=0,
        turn_id=1,
    )


def test_agent_failure_uses_first_simple_fallback_action() -> None:
    action_space = (ActionSpec("ACTION1"), ActionSpec("ACTION2"))
    frame_context = _frame_context(action_space=action_space)

    decision, duration = decide_frame_turn(
        agent=FailingAgent(),
        contexts=ContextDocuments(),
        debug=DebugBus.disabled(),
        frame_context=frame_context,
        recent_action_history_available=True,
        tool_runtime=None,
        turn_id=1,
        action_suppression_zero_changed_pixel_turns=3,
    )

    assert duration >= 0.0
    assert decision.final_action == ActionSpec("ACTION1")
    assert decision.trace.metadata["fallback"] == "agent_decision_error"
    assert decision.trace.metadata["fallback_error_type"] == "RuntimeError"


def test_non_model_agent_failure_uses_fallback_action() -> None:
    action_space = (ActionSpec("ACTION1"),)
    frame_context = _frame_context(action_space=action_space)

    decision, _duration = decide_frame_turn(
        agent=NonModelFailingAgent(),
        contexts=ContextDocuments(),
        debug=DebugBus.disabled(),
        frame_context=frame_context,
        recent_action_history_available=True,
        tool_runtime=None,
        turn_id=1,
        action_suppression_zero_changed_pixel_turns=3,
    )

    assert decision.final_action == ActionSpec("ACTION1")
    assert decision.trace.metadata["fallback"] == "agent_decision_error"
    assert decision.trace.metadata["fallback_error_type"] == "ValueError"


def test_agent_failure_synthesizes_action6_data_when_needed() -> None:
    action_space = (ActionSpec(GameAction.ACTION6),)
    frame_context = _frame_context(action_space=action_space, turn_id=2)

    decision, _duration = decide_frame_turn(
        agent=FailingAgent(ObservationTextConfig(crop_cells=2)),
        contexts=ContextDocuments(),
        debug=DebugBus.disabled(),
        frame_context=frame_context,
        recent_action_history_available=True,
        tool_runtime=None,
        turn_id=2,
        action_suppression_zero_changed_pixel_turns=3,
    )

    assert decision.final_action.name == "ACTION6"
    assert decision.final_action.data == {"x": 17, "y": 32}
    assert decision.final_action.target == "fallback probe at (17,32)"
    assert decision.trace.metadata["decision_source"] == "orchestration_fallback"


def test_invalid_agent_decision_is_replaced_before_environment_step() -> None:
    action = ActionSpec("ACTION1")
    current = Observation(id="current", step=0, frame=_grid())
    current_ref = ObservationRef(memory="state", id="current")
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(game_index=0, max_actions_per_level=10),
        game_id="game-1",
        latest_environment_observation=current,
        remaining_actions=10,
        real_actions=(action,),
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=current,
        observation_ref=current_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=current_ref,
    )

    decide_step(
        session,
        agent=InvalidActionAgent(),
        contexts=ContextDocuments(),
        debug=DebugBus.disabled(),
    )

    assert session.decision is not None
    assert session.decision.final_action == action
    assert session.decision.trace.metadata["fallback"] == "agent_decision_error"


def test_change_summary_failure_uses_deterministic_transition_fallback() -> None:
    session = _transition_session()

    result = summarize_change_model(
        session,
        change_model=FailingChangeModel(),
        debug=DebugBus.disabled(),
    )

    assert result.changed_pixel_count == 1
    assert result.change_detected is True
    assert result.metadata["fallback"] == "change_summary_error"


def test_non_model_change_summary_failure_uses_deterministic_fallback() -> None:
    result = summarize_change_model(
        _transition_session(),
        change_model=NonModelFailingChangeModel(),
        debug=DebugBus.disabled(),
    )

    assert result.changed_pixel_count == 1
    assert result.metadata["fallback"] == "change_summary_error"
    assert result.metadata["fallback_error_type"] == "ValueError"


def test_historizer_failure_returns_not_available_summary() -> None:
    frame_context = _frame_context(action_space=(ActionSpec("ACTION1"),))

    summary = summarize_agent_context_history(
        state_memory=FakeStateMemory(),
        frame_context=frame_context,
        historizer=FailingHistorizer(),
        context_window=2,
        turn_id=1,
        debug=DebugBus.disabled(),
    )

    assert summary.is_available() is False
    assert summary.metadata["fallback"] == "historizer_error"


def test_non_model_historizer_failure_returns_not_available_summary() -> None:
    frame_context = _frame_context(action_space=(ActionSpec("ACTION1"),))

    summary = summarize_agent_context_history(
        state_memory=FakeStateMemory(),
        frame_context=frame_context,
        historizer=NonModelFailingHistorizer(),
        context_window=2,
        turn_id=1,
        debug=DebugBus.disabled(),
    )

    assert summary.is_available() is False
    assert summary.metadata["fallback"] == "historizer_error"
    assert summary.metadata["fallback_error_type"] == "ValueError"


def test_missing_historizer_registration_returns_not_available_summary() -> None:
    frame_context = _frame_context(action_space=(ActionSpec("ACTION1"),))

    summary = summarize_agent_context_history(
        state_memory=FakeStateMemory(),
        frame_context=frame_context,
        historizer=None,
        context_window=2,
        turn_id=1,
        debug=DebugBus.disabled(),
    )

    assert summary.is_available() is False
    assert summary.metadata["fallback"] == "historizer_error"
    assert summary.metadata["fallback_error_type"] == "RuntimeError"


def test_agent_updater_failure_keeps_previous_context() -> None:
    previous_context = RoleContext(general="general", game="old game context")
    contexts = ContextDocuments(agent=previous_context)

    _apply_context_update_with(FailingAgentUpdater(), contexts=contexts)

    assert contexts.agent == previous_context


def test_non_model_agent_updater_failure_keeps_previous_context() -> None:
    contexts = ContextDocuments(
        agent=RoleContext(general="general", game="old game context")
    )
    previous_context = contexts.agent

    _apply_context_update_with(
        NonModelFailingAgentUpdater(),
        contexts=contexts,
    )

    assert contexts.agent == previous_context


def test_state_memory_prewrite_failure_does_not_block_frame_turn() -> None:
    action = ActionSpec("ACTION1")
    observation = Observation(id="current", step=0, frame=_grid())
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(game_index=0, max_actions_per_level=10),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=10,
        real_actions=(action,),
        frame_buffer=(observation,),
    )

    enter_frame_turn(
        session,
        contexts=ContextDocuments(),
        state_memory=PrewriteFailingStateMemory(),
        tool_runtime_factory=None,
        debug=DebugBus.disabled(),
    )

    assert session.current is not None
    assert session.current.source_state_id is None


def test_state_memory_persist_failure_does_not_block_turn() -> None:
    session = _transition_session()
    assert session.current is not None
    session.current = replace(session.current, source_state_id=123)
    assert session.decision is not None
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=session.current.observation_ref,
        actual_next_observation_ref=session.next.observation_ref if session.next else None,
        decision_trace=session.decision.trace,
        actual_next_observation=session.next.observation if session.next else None,
        turn_metrics=TurnMetrics(),
    )

    persist_step(
        session,
        contexts=ContextDocuments(),
        state_memory=CompleteFailingStateMemory(),
        debug=DebugBus.disabled(),
    )

    assert session.state_record_ids == []


def test_state_machine_returns_terminal_fallback_for_lifecycle_exception() -> None:
    result = GameLoopStateMachine(
        state_memory=None,
        contexts=ContextDocuments(),
        agent=object(),
        change_summary_model=object(),
        agent_context_historizer=None,
        updater_tasks=UpdaterTaskRegistry(),
        debug=DebugBus.disabled(),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=LifecycleFailingEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert result.stop_reason == "framework_error_fallback"
    assert result.metadata["fallback_error_type"] == "RuntimeError"
    assert result.metadata["fallback_error"] == "lifecycle failed"
