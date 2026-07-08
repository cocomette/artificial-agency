"""Frame-unrolled game-loop state machine owned by orchestration."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from face_of_agi.contracts import ContextDocuments, FrameTurnContext, GameRunResult
from face_of_agi.contracts import RuntimeConfig
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import StateMemory
from face_of_agi.models.action_coordinates import normalized_crop_box_to_arc_grid_edges
from face_of_agi.models.change import ChangeSummaryModel
from face_of_agi.models.memory import GameMemoryModel
from face_of_agi.models.adapters import (
    AgentContextHistorizerModel,
    OrchestratorAgentModel,
)
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import FrameTurnCompleted
from face_of_agi.orchestration.game_loop.actions import steps
from face_of_agi.orchestration.game_loop import simulation
from face_of_agi.orchestration.game_loop.helpers import model_input_crop_box_normalized
from face_of_agi.orchestration.game_loop.lifecycle import (
    check_lifecycle,
    check_runtime_deadline,
    finish_run,
    start_run,
)

AgentToolRuntimeFactory = Callable[
    [str, str, int, FrameTurnContext],
    AgentToolRuntime,
]

class GameLoopStateMachine:
    """Run one ARC game through the target frame-turn state machine.

    This component owns the game-loop mechanics. The top-level `Orchestrator`
    remains the coordinator that wires dependencies and invokes this component.
    """

    def __init__(
        self,
        *,
        state_memory: StateMemory | None,
        contexts: ContextDocuments,
        agent: OrchestratorAgentModel,
        change_summary_model: ChangeSummaryModel,
        agent_context_historizer: AgentContextHistorizerModel | None,
        game_memory_model: GameMemoryModel,
        updater_tasks: UpdaterTaskRegistry,
        tool_runtime_factory: AgentToolRuntimeFactory | None = None,
        debug: DebugBus,
    ) -> None:
        self.state_memory = state_memory
        self.contexts = contexts
        self.agent = agent
        self.change_summary_model = change_summary_model
        self.agent_context_historizer = agent_context_historizer
        self.game_memory_model = game_memory_model
        self.updater_tasks = updater_tasks
        self.tool_runtime_factory = tool_runtime_factory
        self.debug = debug

    def run(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
    ) -> GameRunResult:
        """Run one selected ARC game until a terminal loop condition."""

        frame_hash_crop_edges = _frame_hash_crop_edges(self.change_summary_model)
        session = start_run(
            config=config,
            environment=environment,
            environment_config=environment_config,
            contexts=self.contexts,
            state_memory=self.state_memory,
            debug=self.debug,
        )

        while session.running:
            session.process_turn = True
            if check_runtime_deadline(session):
                continue
            check_lifecycle(session)
            if not session.process_turn:
                continue

            turn_started_at = perf_counter()
            steps.load_frame_buffer_if_needed(session)
            steps.enter_frame_turn(
                session,
                contexts=self.contexts,
                state_memory=self.state_memory,
                tool_runtime_factory=self.tool_runtime_factory,
                frame_hash_crop_edges=frame_hash_crop_edges,
                debug=self.debug,
            )
            current = steps.require_current(session)
            if current.control_mode is None:
                raise RuntimeError("current frame snapshot is missing control mode")
            if check_runtime_deadline(session):
                continue

            steps.decide(
                session,
                agent=self.agent,
                contexts=self.contexts,
                debug=self.debug,
            )
            if check_runtime_deadline(session):
                continue
            if simulation.maybe_run_known_state_simulation(
                session,
                contexts=self.contexts,
                agent=self.agent,
                agent_context_historizer=self.agent_context_historizer,
                game_memory_model=self.game_memory_model,
                updater_tasks=self.updater_tasks,
                tool_runtime_factory=self.tool_runtime_factory,
                state_memory=self.state_memory,
                frame_hash_crop_edges=frame_hash_crop_edges,
                debug=self.debug,
            ):
                continue
            if check_runtime_deadline(session):
                continue
            steps.resolve_next_snapshot(session, debug=self.debug)
            if check_runtime_deadline(session):
                continue

            current = steps.require_current(session)
            try:
                change_result = steps.summarize_change_model(
                    session,
                    change_model=self.change_summary_model,
                    debug=self.debug,
                )
            finally:
                steps.capture_change_summary_inputs(
                    session,
                    change_model=self.change_summary_model,
                    debug=self.debug,
                )
            steps.attach_change_summary(session, result=change_result)
            if check_runtime_deadline(session):
                continue
            if current.control_mode.controllable:
                steps.summarize_game_memory(
                    session,
                    memory_model=self.game_memory_model,
                    debug=self.debug,
                )
                if check_runtime_deadline(session):
                    continue

            try:
                agent_context_history = steps.summarize_agent_context_history(
                    session,
                    state_memory=self.state_memory,
                    agent_context_historizer=self.agent_context_historizer,
                    debug=self.debug,
                )
            finally:
                _capture_agent_context_history_inputs(
                    session,
                    historizer=self.agent_context_historizer,
                    debug=self.debug,
                )
            if check_runtime_deadline(session):
                continue

            steps.run_updaters(
                session,
                contexts=self.contexts,
                agent_context_history=agent_context_history,
                updater_tasks=self.updater_tasks,
                state_memory=self.state_memory,
                debug=self.debug,
            )
            steps.persist(
                session,
                contexts=self.contexts,
                state_memory=self.state_memory,
                debug=self.debug,
            )
            current = steps.require_current(session)
            decision = steps.require_decision(session)
            if current.control_mode is None:
                raise RuntimeError("completed frame turn is missing control mode")
            self.debug.emit(
                FrameTurnCompleted(
                    run_id=session.config.run_id,
                    game_id=session.game_id,
                    game_index=session.environment_config.game_index,
                    turn_id=current.turn_id,
                    env_step=current.observation.step,
                    frame_index=current.frame_index,
                    frame_count=current.frame_count,
                    controllable=current.control_mode.controllable,
                    action=decision.final_action,
                    turn_duration_seconds=perf_counter() - turn_started_at,
                    completed_levels=_completed_levels_after_turn(session),
                    remaining_actions=session.remaining_actions,
                )
            )
            steps.advance(session)

        return finish_run(
            session,
            contexts=self.contexts,
            updater_tasks=self.updater_tasks,
            state_memory=self.state_memory,
            debug=self.debug,
        )


def _capture_agent_context_history_inputs(
    session,
    *,
    historizer: AgentContextHistorizerModel | None,
    debug: DebugBus,
) -> None:
    current = session.current
    if current is None:
        return
    debug.capture_model_inputs(
        current.to_frame_context(),
        current.turn_id,
        historizer,
    )


def _completed_levels_after_turn(session) -> int:
    metrics = session.turn_metrics
    if metrics is not None and metrics.cumulative_score is not None:
        return int(metrics.cumulative_score)
    return int(session.completed_levels)


def _frame_hash_crop_edges(
    change_summary_model: ChangeSummaryModel,
) -> tuple[int, int, int, int]:
    """Return the ARC-grid crop edges used for known-state frame hashes."""

    return normalized_crop_box_to_arc_grid_edges(
        model_input_crop_box_normalized(change_summary_model)
    )
