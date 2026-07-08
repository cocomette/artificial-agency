"""Central orchestration boundary."""

from __future__ import annotations

from typing import TextIO

from face_of_agi.agent_creator.service import AgentCreatorService
from face_of_agi.contracts import (
    ContextDocuments,
    FrameTurnContext,
    GameRunResult,
    RuntimeConfig,
    ToolName,
)
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import ExperimentalMemory, StateMemory
from face_of_agi.models.adapters import ModelRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.sinks import DebugTrace
from face_of_agi.orchestration.game_loop import (
    GameLoopStateMachine,
)
from face_of_agi.orchestration.tool_runtime import OrchestrationAgentToolRuntime


class Orchestrator:
    """Coordinate environment, memory, and model boundaries.

    Sub-orchestration components own concrete workflows such as the game-loop
    state machine. This class wires dependencies and keeps those workflows
    behind a single orchestration boundary.
    """

    def __init__(
        self,
        *,
        state_memory: StateMemory | None = None,
        experimental_memory: ExperimentalMemory | None = None,
        models: ModelRegistry | None = None,
        contexts: ContextDocuments | None = None,
        experimental_memory_turn_buffer: int = 2,
        agent_creator_service: AgentCreatorService | None = None,
    ) -> None:
        self.state_memory = state_memory
        self.experimental_memory = experimental_memory
        self.models = self._ensure_models(models)
        self.contexts = contexts or ContextDocuments()
        self.agent_creator_service = agent_creator_service
        if experimental_memory_turn_buffer < 1:
            raise ValueError("experimental memory turn buffer must be at least 1")
        self.experimental_memory_turn_buffer = experimental_memory_turn_buffer
        self.debug = DebugBus.disabled()

    def run_environment_shell(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
        trace_output: TextIO | None = None,
        debug_trace: DebugTrace | None = None,
        debug: DebugBus | None = None,
    ) -> GameRunResult:
        """Run one ARC game through the dedicated game-loop component."""

        active_debug = debug or DebugBus(
            sink=(
                debug_trace
                or DebugTrace.from_config(
                    environment_config,
                    output=trace_output,
                )
            ),
            state_memory=self.state_memory,
        )
        previous_debug = self.debug
        self.debug = active_debug
        try:
            return GameLoopStateMachine(
                state_memory=self.state_memory,
                contexts=self.contexts,
                change_summary_model=self.models.require_change_summary_model(),
                world_model=self.models.require_world_model(),
                agent_context_historizer=(
                    self.models.agent_context_historizer_model
                ),
                level_solution_summarizer=(
                    self.models.level_solution_summarizer
                ),
                updater_tasks=self.models.require_updater_tasks(),
                tool_runtime_factory=None,
                agent_creator_service=self.agent_creator_service,
                debug=active_debug,
            ).run(
                config=config,
                environment=environment,
                environment_config=environment_config,
            )
        finally:
            self.debug = previous_debug

    def cleanup_state_memory_keep_latest(self) -> None:
        """Prune dedicated M state rows after a normal run finishes."""

        if self.state_memory is None:
            return
        self.state_memory.cleanup_keep_latest_per_game()

    def _ensure_models(self, models: ModelRegistry | None) -> ModelRegistry:
        return models or ModelRegistry()

    def _build_agent_tool_runtime(
        self,
        run_id: str,
        game_id: str,
        turn_id: int,
        frame_context: FrameTurnContext,
    ) -> OrchestrationAgentToolRuntime:
        """Build the controlled tool interface for one X decision turn."""

        return OrchestrationAgentToolRuntime(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            frame_context=frame_context,
            available_tool_names=self._available_tool_names(),
            tools_enabled=frame_context.control_mode.controllable,
        )

    def _available_tool_names(self) -> tuple[ToolName, ...]:
        """Return configured tools exposed to X on this frame."""

        return ()
