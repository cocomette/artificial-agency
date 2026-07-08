"""Central orchestration boundary for the online learner runtime."""

from __future__ import annotations

from typing import TextIO

from face_of_agi.contracts import GameRunResult, RuntimeConfig
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.sinks import DebugTrace
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import ExperimentalMemory, StateMemory
from face_of_agi.online.agent import OnlineLearnerAgent
from face_of_agi.orchestration.game_loop import GameLoopStateMachine


class Orchestrator:
    """Coordinate environment, online learner, memory, and debug tracing."""

    def __init__(
        self,
        *,
        agent: OnlineLearnerAgent,
        state_memory: StateMemory | None = None,
        experimental_memory: ExperimentalMemory | None = None,
        experimental_memory_turn_buffer: int = 2,
    ) -> None:
        self.agent = agent
        self.state_memory = state_memory
        self.experimental_memory = experimental_memory
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
        """Run one ARC game through the online learner game loop."""

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
                agent=self.agent,
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
