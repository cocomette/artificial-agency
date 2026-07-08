"""Kaggle ARC Prize 2026 runtime entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, TextIO
import urllib.error
import urllib.request

from arc_agi import OperationMode

from face_of_agi.contracts import ContextDocuments, GameRunResult, RuntimeConfig
from face_of_agi.debug.sinks import LiveTurnMonitor
from face_of_agi.environment.adapter import (
    ArcEnvironmentWrapperAdapter,
    KaggleArcadeAdapter,
)
from face_of_agi.environment.config import EnvironmentConfig, load_environment_config
from face_of_agi.runtime.loop import RuntimeLoop
from face_of_agi.runtime.parallel import (
    ParallelGameRunSpec,
    ParallelRuntimeLoop,
    retry_parallel_game_spec,
)
from face_of_agi.runtime.shell import (
    _build_model_registry,
    _build_orchestrator,
    _print_parallel_result,
)

DEFAULT_KAGGLE_CONFIG = (
    "src/face_of_agi/runtime/configs/vllm/"
    "vllm_rtx6000_qwen36_35b_fp8_parallel.yaml"
)
DEFAULT_DATABASE_DIR = Path("/kaggle/working/runs")


def main(argv: list[str] | None = None) -> None:
    """Run FACE-OF-AGI directly inside a Kaggle competition notebook."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    tags = _parse_tags(args.tags)
    result = run_config(
        config_path=Path(args.config),
        database_dir=Path(args.database_dir),
        tags=tags,
        deadline_monotonic=_deadline_monotonic_from_epoch(
            args.deadline_epoch_seconds
        ),
    )
    _print_parallel_result(result)
    if result.failures:
        print(
            "kaggle completed with failed games after retries; "
            "leaving notebook exit status at 0"
        )


def run_config(
    *,
    config_path: Path,
    database_dir: Path,
    tags: tuple[str, ...] = (),
    arcade: Any | None = None,
    trace_output: TextIO | None = None,
    deadline_monotonic: float | None = None,
) -> Any:
    """Run all selected Kaggle games through isolated runtime workers."""

    _configure_gateway_environment()
    environment_config = load_environment_config(config_path)
    environment_config.operation_mode = OperationMode.COMPETITION
    database_dir.mkdir(parents=True, exist_ok=True)

    if arcade is None:
        _wait_for_gateway()
    arc = arcade or _build_arcade(environment_config)
    card_id = arc.open_scorecard(tags=tags)
    print(f"kaggle scorecard opened: {card_id}")
    try:
        selected_games = _selected_game_ids(environment_config, arc)
        if not selected_games:
            raise RuntimeError("Kaggle runner found no ARC games to run")

        batch_run_id = _build_kaggle_batch_run_id()
        live_turn_monitor = (
            LiveTurnMonitor(
                selected_game_count=len(selected_games),
                output=trace_output,
            )
            if environment_config.live_turn_monitor
            else None
        )
        specs = tuple(
            _build_kaggle_spec(
                batch_run_id=batch_run_id,
                selection_index=selection_index,
                game_id=game_id,
                card_id=card_id,
                arc=arc,
                database_dir=database_dir,
                base_environment_config=environment_config,
                deadline_monotonic=deadline_monotonic,
                live_turn_monitor=live_turn_monitor,
            )
            for selection_index, game_id in selected_games
        )
        return ParallelRuntimeLoop(
            _run_kaggle_game,
            trace_output=trace_output,
        ).run(
            batch_run_id=batch_run_id,
            specs=specs,
            max_parallel_games=environment_config.max_parallel_games,
            max_game_retries=environment_config.max_game_retries,
            retry_spec_factory=_build_kaggle_retry_spec,
            deadline_monotonic=deadline_monotonic,
        )
    finally:
        scorecard = arc.close_scorecard(card_id)
        print(f"kaggle scorecard closed: {card_id}")
        if scorecard is not None:
            print(scorecard)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FACE-OF-AGI against Kaggle ARC-AGI-3 gateway games.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_KAGGLE_CONFIG,
        help="Runtime YAML config path.",
    )
    parser.add_argument(
        "--database-dir",
        default=str(DEFAULT_DATABASE_DIR),
        help="Directory for per-game SQLite memory files.",
    )
    parser.add_argument(
        "--tags",
        default="face-of-agi,kaggle",
        help="Comma-separated scorecard tags.",
    )
    parser.add_argument(
        "--deadline-epoch-seconds",
        type=float,
        default=None,
        help="Unix epoch time when the Kaggle runtime should stop cleanly.",
    )
    return parser


def _configure_gateway_environment() -> None:
    """Set the ARC toolkit environment to Kaggle's gateway sidecar."""

    os.environ["SCHEME"] = "http"
    os.environ["HOST"] = "gateway"
    os.environ["PORT"] = "8001"
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ["ARC_BASE_URL"] = "http://gateway:8001/"
    os.environ["OPERATION_MODE"] = "competition"
    os.environ.setdefault("ENVIRONMENTS_DIR", "")
    os.environ.setdefault("RECORDINGS_DIR", "/kaggle/working/server_recording")
    os.environ.setdefault("MPLBACKEND", "agg")


def _wait_for_gateway(timeout_seconds: float = 600.0) -> None:
    """Wait for Kaggle's local ARC gateway before discovering games."""

    deadline = time.monotonic() + timeout_seconds
    url = os.environ["ARC_BASE_URL"].rstrip("/") + "/api/games"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (TimeoutError, urllib.error.URLError):
            time.sleep(2)
    raise RuntimeError("Kaggle ARC gateway did not become ready")


def _build_arcade(config: EnvironmentConfig) -> KaggleArcadeAdapter:
    return KaggleArcadeAdapter.from_config(config)


def _selected_game_ids(
    config: EnvironmentConfig,
    arcade: Any,
) -> tuple[tuple[int, str], ...]:
    """Return selected game ids using the live Kaggle gateway catalog."""

    if config.game_ids:
        return tuple(enumerate(config.game_ids))

    available = arcade.list_available_game_ids()
    if config.game_selection == "all_available":
        return tuple(enumerate(available))

    indices = config.game_indices
    if config.game_index is not None:
        indices = (config.game_index,)
    resolved: list[tuple[int, str]] = []
    missing: list[int] = []
    for index in indices:
        if 0 <= index < len(available):
            resolved.append((index, available[index]))
        else:
            missing.append(index)
    if missing:
        missing_text = ", ".join(str(index) for index in missing)
        raise RuntimeError(
            f"Kaggle game indices out of range for live catalog: {missing_text}"
        )
    return tuple(resolved)


def _build_kaggle_spec(
    *,
    batch_run_id: str,
    selection_index: int,
    game_id: str,
    card_id: str,
    arc: Any,
    database_dir: Path,
    base_environment_config: EnvironmentConfig,
    deadline_monotonic: float | None,
    live_turn_monitor: LiveTurnMonitor | None,
) -> ParallelGameRunSpec:
    """Create one isolated worker spec bound to a Kaggle ARC environment."""

    arc_environment = arc.make_scorecard_environment(
        game_id,
        scorecard_id=card_id,
    )

    safe_game_id = _safe_game_id(game_id)
    return ParallelGameRunSpec(
        game_index=selection_index,
        game_id=game_id,
        run_id=f"{batch_run_id}-{safe_game_id}",
        database_path=database_dir / f"memory-{safe_game_id}.sqlite",
        environment_config=replace(
            base_environment_config,
            game_index=selection_index,
            game_indices=(),
            game_ids=(),
            game_selection=None,
            game_id=game_id,
            operation_mode=OperationMode.COMPETITION,
        ),
        arc_environment=arc_environment,
        live_turn_monitor=live_turn_monitor,
        deadline_monotonic=deadline_monotonic,
    )


def _build_kaggle_retry_spec(
    spec: ParallelGameRunSpec,
    attempt_index: int,
) -> ParallelGameRunSpec:
    """Create an isolated retry spec that reuses the Competition Mode wrapper."""

    retry_spec = retry_parallel_game_spec(spec, attempt_index)
    return replace(
        retry_spec,
        arc_environment=spec.arc_environment,
    )


def _run_kaggle_game(
    spec: ParallelGameRunSpec,
    trace_output: TextIO,
) -> GameRunResult:
    """Run one Kaggle game wrapper through the normal runtime loop."""

    if spec.arc_environment is None:
        raise RuntimeError("Kaggle worker spec is missing an ARC environment")
    environment_config = spec.environment_config
    runtime_config = RuntimeConfig(
        run_id=spec.run_id,
        database_path=spec.database_path,
        deadline_monotonic=spec.deadline_monotonic,
    )
    model_registry = _build_model_registry(
        agent_config=environment_config.models.agent,
        change_config=environment_config.models.change,
        memory_config=environment_config.models.memory,
        world_config=environment_config.models.world,
        goal_config=environment_config.models.goal,
        interest_config=environment_config.models.interest,
        reward_judge_config=environment_config.models.reward_judge,
        shared_vlm_config=environment_config.models.shared_vlm,
    )
    runtime = RuntimeLoop(
        _build_orchestrator(
            spec.database_path,
            experimental_memory_turn_buffer=(
                environment_config.experimental_memory_turn_buffer
            ),
            models=model_registry,
            contexts=ContextDocuments(),
        ),
        trace_output=trace_output,
        live_turn_monitor=spec.live_turn_monitor,
    )
    return runtime.run(
        config=runtime_config,
        environment=ArcEnvironmentWrapperAdapter(
            game_id=spec.game_id,
            environment=spec.arc_environment,
        ),
        environment_config=environment_config,
    )


def _parse_tags(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(tag.strip() for tag in value.split(",") if tag.strip())


def _deadline_monotonic_from_epoch(value: float | None) -> float | None:
    if value is None:
        return None
    return time.monotonic() + max(0.0, float(value) - time.time())


def _build_kaggle_batch_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"kaggle-{timestamp}"


def _safe_game_id(game_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", game_id).strip("-") or "game"


if __name__ == "__main__":
    main(sys.argv[1:])
