"""Frame-unrolled game-loop state machine owned by orchestration."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import TypeVar

from face_of_agi.contracts import ContextDocuments, FrameTurnContext, GameRunResult
from face_of_agi.contracts import RuntimeConfig
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import StateMemory
from face_of_agi.models.change import ChangeSummaryModel, ChangeSummaryResult
from face_of_agi.models.adapters import (
    AgentContextHistorizerModel,
)
from face_of_agi.models.level_summary import LevelSolutionSummarizerModel
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.models.world import AgentWorldModel
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import FrameTurnCompleted
from face_of_agi.orchestration.game_loop.actions import steps
from face_of_agi.orchestration.game_loop.lifecycle import (
    check_lifecycle,
    check_runtime_deadline,
    finish_run,
    reset_after_game_over,
    start_run,
)
from face_of_agi.runtime import timing as runtime_timing

AgentToolRuntimeFactory = Callable[
    [str, str, int, FrameTurnContext],
    AgentToolRuntime,
]
T = TypeVar("T")


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
        change_summary_model: ChangeSummaryModel,
        world_model: AgentWorldModel,
        agent_context_historizer: AgentContextHistorizerModel | None,
        level_solution_summarizer: LevelSolutionSummarizerModel | None = None,
        updater_tasks: UpdaterTaskRegistry,
        tool_runtime_factory: AgentToolRuntimeFactory | None = None,
        debug: DebugBus,
    ) -> None:
        self.state_memory = state_memory
        self.contexts = contexts
        self.change_summary_model = change_summary_model
        self.world_model = world_model
        self.agent_context_historizer = agent_context_historizer
        self.level_solution_summarizer = level_solution_summarizer
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

        session = start_run(
            config=config,
            environment=environment,
            environment_config=environment_config,
            contexts=self.contexts,
            state_memory=self.state_memory,
            debug=self.debug,
        )

        with ThreadPoolExecutor(max_workers=2) as turn_executor:
            while session.running:
                change_future: Future[ChangeSummaryResult] | None = None

                session.process_turn = True
                if check_runtime_deadline(session):
                    continue
                check_lifecycle(
                    session,
                    state_memory=self.state_memory,
                    level_solution_summarizer=self.level_solution_summarizer,
                    debug=self.debug,
                )
                if not session.process_turn:
                    continue

                turn_started_at = perf_counter()
                try:
                    steps.load_frame_buffer_if_needed(session)
                    steps.enter_frame_turn(
                        session,
                        contexts=self.contexts,
                        state_memory=self.state_memory,
                        tool_runtime_factory=self.tool_runtime_factory,
                        change_model=self.change_summary_model,
                        debug=self.debug,
                    )
                    current = steps.require_current(session)
                    if current.control_mode is None:
                        raise RuntimeError(
                            "current frame snapshot is missing control mode"
                        )
                    run_context_updates = True
                    if check_runtime_deadline(session):
                        continue

                    if steps.has_observed_transition(session):
                        steps.prepare_observed_transition(session)
                        changed_pixel_count = steps.change_summary_changed_pixel_count(
                            session,
                            change_model=self.change_summary_model,
                        )
                        if changed_pixel_count == 0 and not (
                            steps.observed_transition_has_animation_bundle(session)
                        ):
                            steps.attach_unchanged_frame_summary(
                                session,
                                change_model=self.change_summary_model,
                                changed_pixel_count=changed_pixel_count,
                            )
                            steps.clear_queued_actions_after_net_noop_transition(
                                session
                            )
                            if check_runtime_deadline(session):
                                continue
                        else:
                            current = steps.require_current(session)
                            change_future = turn_executor.submit(
                                steps.summarize_change_model,
                                session,
                                change_model=self.change_summary_model,
                                debug=self.debug,
                            )
                            try:
                                change_result = _wait_for_future(
                                    change_future,
                                    span_name="game_loop.change_summary.wait",
                                    turn_id=current.turn_id,
                                    step=current.observation.step,
                                )
                            finally:
                                steps.capture_change_summary_inputs(
                                    session,
                                    change_model=self.change_summary_model,
                                    debug=self.debug,
                                )
                                change_future = None
                            steps.attach_change_summary(
                                session,
                                result=change_result,
                                change_model=self.change_summary_model,
                                changed_pixel_count=changed_pixel_count,
                            )
                            steps.clear_queued_actions_after_net_noop_transition(
                                session
                            )
                            if check_runtime_deadline(session):
                                continue

                        if run_context_updates:
                            if current.control_mode.controllable:
                                if steps.should_run_updaters(session):
                                    steps.run_updaters(
                                        session,
                                        contexts=self.contexts,
                                        world_model=self.world_model,
                                        agent_context_historizer=(
                                            self.agent_context_historizer
                                        ),
                                        updater_tasks=self.updater_tasks,
                                        state_memory=self.state_memory,
                                        debug=self.debug,
                                    )
                    elif run_context_updates:
                        steps.bootstrap_agent_updater_decision(
                            session,
                            contexts=self.contexts,
                            updater_tasks=self.updater_tasks,
                            state_memory=self.state_memory,
                            debug=self.debug,
                        )

                    if session.pending_game_over_reset:
                        steps.attach_game_over_reset_decision(session)
                        steps.persist(
                            session,
                            contexts=self.contexts,
                            state_memory=self.state_memory,
                            debug=self.debug,
                        )
                        current = steps.require_current(session)
                        decision = steps.require_decision(session)
                        if current.control_mode is None:
                            raise RuntimeError(
                                "completed frame turn is missing control mode"
                            )
                        self.debug.emit(
                            FrameTurnCompleted(
                                run_id=session.config.run_id,
                                game_id=session.game_id,
                                game_index=session.environment_config.game_index,
                                turn_id=current.turn_id,
                                env_step=current.observation.step,
                                frame_index=current.frame_index,
                                frame_count=current.frame_count,
                                controllable=False,
                                action=decision.final_action,
                                turn_duration_seconds=perf_counter()
                                - turn_started_at,
                                completed_levels=_completed_levels_after_turn(
                                    session
                                ),
                                remaining_actions=session.remaining_actions,
                                max_actions_per_level=(
                                    session.environment_config.max_actions_per_level
                                ),
                            )
                        )
                        steps.record_action_history(session)
                        reset_after_game_over(session)
                        continue

                    steps.decide(
                        session,
                        debug=self.debug,
                    )
                    if check_runtime_deadline(session):
                        continue
                    steps.resolve_next_snapshot(
                        session,
                        debug=self.debug,
                        change_model=self.change_summary_model,
                    )
                    if check_runtime_deadline(session):
                        continue
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
                            max_actions_per_level=(
                                session.environment_config.max_actions_per_level
                            ),
                        )
                    )
                    steps.advance(session)
                finally:
                    _settle_abandoned_turn_future(
                        change_future,
                        capture=lambda: steps.capture_change_summary_inputs(
                            session,
                            change_model=self.change_summary_model,
                            debug=self.debug,
                        ),
                    )
                    _settle_abandoned_turn_future(
                        None,
                        capture=lambda: None,
                    )

        return finish_run(
            session,
            contexts=self.contexts,
            updater_tasks=self.updater_tasks,
            state_memory=self.state_memory,
            debug=self.debug,
        )


def _wait_for_future(
    future: Future[T],
    *,
    span_name: str,
    turn_id: int,
    step: int | None,
) -> T:
    """Wait for a model prerequisite future with optional timing output."""

    with runtime_timing.span(span_name, turn_id=turn_id, step=step):
        return future.result()


def _settle_abandoned_turn_future(
    future: Future[object] | None,
    *,
    capture: Callable[[], None],
) -> None:
    """Cancel or drain an abandoned turn future before the next turn starts."""

    if future is None:
        return
    if future.cancel():
        return
    try:
        future.result()
    except Exception:
        pass
    finally:
        capture()


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
