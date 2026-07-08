"""Top-level runtime loop boundary."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

from face_of_agi.contracts import GameRunResult, RuntimeConfig
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.orchestration.orchestrator import Orchestrator


class RuntimeLoop:
    """Run the starter environment shell or the older reset-only fallback."""

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
        environment: EnvironmentAdapter | None = None,
        environment_config: EnvironmentConfig | None = None,
        environments: Mapping[str, EnvironmentAdapter] | None = None,
    ) -> GameRunResult | list[GameRunResult]:
        """Run the starter shell or the older reset-only multi-game fallback."""

        if environment is not None and environment_config is not None:
            return self._run_environment_shell(
                config=config,
                environment=environment,
                environment_config=environment_config,
            )

        if environments is None:
            raise RuntimeError(
                "runtime requires either one environment shell or legacy environments"
            )

        game_ids: Sequence[str] = config.game_ids or tuple(environments.keys())
        results: list[GameRunResult] = []
        for game_id in game_ids:
            results.append(
                self.orchestrator.run_reset_only(
                    run_id=config.run_id,
                    game_id=game_id,
                    environment=environments[game_id],
                )
            )
        return results

    def _run_environment_shell(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
    ) -> GameRunResult:
        """Delegate the single-game ARC loop to orchestration."""

        result = self.orchestrator.run_environment_shell(
            config=config,
            environment=environment,
            environment_config=environment_config,
            trace_output=self.trace_output,
        )
        self.orchestrator.cleanup_state_memory_keep_latest()
        return result
