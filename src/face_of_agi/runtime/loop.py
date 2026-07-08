"""Top-level runtime loop boundary."""

from __future__ import annotations

import sys
from typing import TextIO

from face_of_agi.contracts import GameRunResult, RuntimeConfig
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.sinks import DebugTrace
from face_of_agi.orchestration.orchestrator import Orchestrator


class RuntimeLoop:
    """Run one configured environment shell through orchestration."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        trace_output: TextIO | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.trace_output = trace_output or sys.stdout

    def run(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
    ) -> GameRunResult:
        """Run the configured game loop shell."""

        return self._run_environment_shell(
            config=config,
            environment=environment,
            environment_config=environment_config,
        )

    def _run_environment_shell(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
    ) -> GameRunResult:
        """Delegate the single-game ARC loop to orchestration."""

        debug = DebugBus(
            sink=DebugTrace.from_config(
                environment_config,
                output=self.trace_output,
            ),
            state_memory=self.orchestrator.state_memory,
        )
        result = self.orchestrator.run_environment_shell(
            config=config,
            environment=environment,
            environment_config=environment_config,
            debug=debug,
        )
        if not environment_config.debug_keep_all_m_states:
            self.orchestrator.cleanup_state_memory_keep_latest()
        return result
