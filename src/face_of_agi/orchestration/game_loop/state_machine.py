"""Frame-unrolled game-loop state machine owned by orchestration."""

from __future__ import annotations

from collections.abc import Callable

from face_of_agi.contracts import ContextDocuments, FrameTurnContext, GameRunResult
from face_of_agi.contracts import RuntimeConfig
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import StateMemory
from face_of_agi.models.adapters import OrchestratorAgentModel
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.orchestration.game_loop.actions import steps
from face_of_agi.orchestration.game_loop.actions.post_decision_predictions import (
    PostDecisionPredictionRunner,
)
from face_of_agi.orchestration.game_loop.lifecycle import (
    check_lifecycle,
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
        updater_tasks: UpdaterTaskRegistry,
        post_decision_prediction_runner: PostDecisionPredictionRunner,
        tool_runtime_factory: AgentToolRuntimeFactory | None = None,
        debug: DebugBus,
    ) -> None:
        self.state_memory = state_memory
        self.contexts = contexts
        self.agent = agent
        self.updater_tasks = updater_tasks
        self.post_decision_prediction_runner = post_decision_prediction_runner
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

        while session.running:
            session.process_turn = True
            check_lifecycle(session)
            if not session.process_turn:
                continue

            steps.load_frame_buffer_if_needed(session)
            steps.enter_frame_turn(
                session,
                contexts=self.contexts,
                state_memory=self.state_memory,
                tool_runtime_factory=self.tool_runtime_factory,
                debug=self.debug,
            )
            steps.decide(
                session,
                agent=self.agent,
                contexts=self.contexts,
                debug=self.debug,
            )
            steps.run_post_decision_predictions(
                session,
                contexts=self.contexts,
                runner=self.post_decision_prediction_runner,
                debug=self.debug,
            )
            steps.resolve_next_snapshot(session, debug=self.debug)
            steps.run_updaters(
                session,
                contexts=self.contexts,
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
            steps.advance(session)

        return finish_run(
            session,
            contexts=self.contexts,
            updater_tasks=self.updater_tasks,
            state_memory=self.state_memory,
            debug=self.debug,
        )
