"""Manual one-step OpenAI E2E check for the full ARC game loop.

This script calls the real ARC environment and the real OpenAI API. It runs one
real environment action through orchestration with OpenAI-backed X, world, and
goal roles, while keeping post-decision predictions mocked via
`prompt_model_calls_enabled=False`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
import json
import os
from pathlib import Path
import shutil
from typing import Any

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    ContextDocuments,
    EnvironmentInfo,
    GameRunResult,
    Observation,
    RuntimeConfig,
)
from face_of_agi.environment import ArcEnvironmentAdapter
from face_of_agi.environment.cheat_context import load_cheat_action_context
from face_of_agi.environment.config import (
    EnvironmentConfig,
    ModelRoleConfig,
    ModelRuntimeConfig,
    load_environment_config,
    load_game_catalog,
)
from face_of_agi.frames import frame_to_pil_image, observation_to_pil_image
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import ModelRegistry
from face_of_agi.models.orchestrator_agent import (
    OpenAIOrchestratorAgentConfig,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.tools.goal import OpenAIGoalToolConfig
from face_of_agi.models.tools.goal.providers.openai import OpenAIGoalToolAdapter
from face_of_agi.models.tools.world import OpenAIWorldToolConfig
from face_of_agi.models.tools.world.providers.openai import OpenAIWorldToolAdapter
from face_of_agi.orchestration import Orchestrator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "src" / "face_of_agi" / "runtime" / "starter_loop.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "openai_full_game_loop_e2e"
DEFAULT_GAME_INDEX = 4
DEFAULT_GAME_ID = "ls20-9607627b"
DEFAULT_GAME_DIR = ROOT / "environment_files" / "ls20" / "9607627b"


class RecordingEnvironment:
    """Thin wrapper that records observations returned by the real environment."""

    def __init__(self, wrapped: ArcEnvironmentAdapter) -> None:
        self.wrapped = wrapped
        self.selected_game_id: str | None = None
        self.reset_observations: list[Observation] = []
        self.step_actions: list[ActionSpec] = []
        self.step_observations: list[Observation] = []

    def list_available_games(self) -> Sequence[Any]:
        return self.wrapped.list_available_games()

    def list_local_games(self) -> Sequence[Any]:
        return self.wrapped.list_local_games()

    def resolve_game_id(self, game_index: int) -> str:
        return self.wrapped.resolve_game_id(game_index)

    def select_game_by_id(self, game_id: str) -> str:
        self.selected_game_id = self.wrapped.select_game_by_id(game_id)
        return self.selected_game_id

    def reset(self) -> Observation:
        observation = self.wrapped.reset()
        self.reset_observations.append(observation)
        return observation

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, Any] | None = None,
    ) -> Observation:
        observation = self.wrapped.step(action, reasoning=reasoning)
        self.step_actions.append(action)
        self.step_observations.append(observation)
        return observation

    def get_action_space(self) -> Sequence[ActionSpec]:
        return self.wrapped.get_action_space()

    def get_info(self) -> EnvironmentInfo:
        return self.wrapped.get_info()


class ArtifactWriter:
    """Serialize nested records and save image-like objects as PNG artifacts."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        if self.image_dir.exists():
            shutil.rmtree(self.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._image_paths: dict[int, str] = {}
        self._image_index = 0

    def image_reference(self, image: Image.Image, *, label: str) -> dict[str, Any]:
        """Save one image once and return a JSON-safe reference."""

        image_key = id(image)
        if image_key not in self._image_paths:
            self._image_index += 1
            filename = f"{self._image_index:03d}_{_safe_slug(label)}.png"
            path = self.image_dir / filename
            image.convert("RGB").save(path)
            self._image_paths[image_key] = str(path.relative_to(self.output_dir))
        return {
            "image_path": self._image_paths[image_key],
            "mode": image.mode,
            "size": list(image.size),
        }


def main() -> None:
    """Run one live OpenAI-backed ARC step and write inspection artifacts."""

    args = _parse_args()
    _load_env_file(args.env_file)
    run_id = args.run_id or _build_run_id(args)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = _resolve_database_path(args.database, output_dir)
    _prepare_database(
        database_path,
        database_arg=args.database,
        overwrite=args.overwrite_database,
    )

    environment_config = build_e2e_environment_config(args)
    database = SQLiteDatabase(database_path)
    state_memory = StateMemory(database)
    experimental_memory = ExperimentalMemory(database)
    contexts = build_context_documents(
        args=args,
        environment_config=environment_config,
    )
    models = build_model_registry(args)
    orchestrator = Orchestrator(
        state_memory=state_memory,
        experimental_memory=experimental_memory,
        models=models,
        contexts=contexts,
        experimental_memory_turn_buffer=(
            environment_config.experimental_memory_turn_buffer
        ),
        prompt_model_calls_enabled=False,
    )
    environment = RecordingEnvironment(
        ArcEnvironmentAdapter.from_config(environment_config)
    )
    trace_output = StringIO()

    result = orchestrator.run_environment_shell(
        config=RuntimeConfig(run_id=run_id, database_path=database_path),
        environment=environment,
        environment_config=environment_config,
        trace_output=trace_output,
    )

    game_id = environment_config.game_id
    if game_id is None:
        raise RuntimeError("E2E environment config did not resolve a game id")
    states = state_memory.list_states(game_id=game_id)
    experiments = experimental_memory.list_experiments(
        run_id=run_id,
        game_id=game_id,
    )
    validate_live_run(
        result=result,
        environment=environment,
        states=states,
        experiments=experiments,
    )
    write_inspection_artifacts(
        output_dir=output_dir,
        database_path=database_path,
        args=args,
        run_id=run_id,
        environment_config=environment_config,
        result=result,
        trace_text=trace_output.getvalue(),
        recording=environment,
        states=states,
        experiments=experiments,
    )

    print(f"saved one-step OpenAI full-loop artifacts to {output_dir}")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "game_id": game_id,
                "stop_reason": result.stop_reason,
                "real_step_count": result.step_count,
                "state_rows": len(states),
                "e_experiments": len(experiments),
                "experiment_tools": sorted({item.tool_name for item in experiments}),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_e2e_environment_config(args: argparse.Namespace) -> EnvironmentConfig:
    """Load starter config and force the one-step OpenAI E2E settings."""

    config = load_environment_config(_resolve_path(args.config))
    if args.game_id:
        config.game_id = args.game_id
        if args.game_index is not None:
            config.game_index = args.game_index
    elif args.game_index is not None:
        config.game_index = args.game_index
        config.game_id = _resolve_game_id_from_catalog(config)
    else:
        config.game_index = DEFAULT_GAME_INDEX
        config.game_id = DEFAULT_GAME_ID

    config.max_actions_per_level = 1
    config.enable_visualization = False
    config.render_mode = None
    config.save_recording = False
    config.include_frame_data = True
    config.models = ModelRuntimeConfig(
        prompt_model_calls_enabled=False,
        agent=ModelRoleConfig(
            backend="openai",
            model=args.agent_model,
            max_tool_calls=args.max_tool_calls,
            repair_attempts=args.repair_attempts,
            options={"reasoning": {"effort": args.reasoning_effort}},
        ),
        world=ModelRoleConfig(
            backend="openai",
            model=args.world_model,
            options=_tool_model_options(args, role="world"),
        ),
        goal=ModelRoleConfig(
            backend="openai",
            model=args.goal_model,
            options=_tool_model_options(args, role="goal"),
        ),
    )
    return config


def build_context_documents(
    *,
    args: argparse.Namespace | None = None,
    environment_config: EnvironmentConfig | None = None,
    cheat_action_context: str | None = None,
) -> ContextDocuments:
    """Return contexts that instruct X to exercise both real tools."""

    if (
        cheat_action_context is None
        and args is not None
        and getattr(args, "cheat_action_context", False)
    ):
        cheat_action_context = load_cheat_action_context(_resolve_path(args.game_dir))

    game_id = (
        environment_config.game_id
        if environment_config is not None and environment_config.game_id is not None
        else DEFAULT_GAME_ID
    )

    return ContextDocuments(
        agent=(
            _role_context(
                general=(
                    "You are running a live full-loop E2E check. Follow the "
                    "tool policy exactly and choose only actions from the "
                    "provided action space."
                ),
                game=_compose_context_game_text(
                    game_id=game_id,
                    base=(
                        "On non-controllable animation frames, submit the "
                        "internal NONE action and do not call tools. On the "
                        "controllable frame, call the goal tool once using the "
                        "current observation reference, then call the world "
                        "tool once using the current observation reference and "
                        "one valid candidate action. After both tool results "
                        "are returned, submit one valid final action."
                    ),
                    cheat_action_context=cheat_action_context,
                ),
            )
        ),
        world=_role_context(
            general=(
                "Predict the next ARC observation from the supplied source "
                "frame and candidate action."
            ),
            game=_compose_context_game_text(
                game_id=game_id,
                base=(
                    "This is a live one-step E2E check; preserve concrete "
                    "visual details."
                ),
                cheat_action_context=cheat_action_context,
            ),
        ),
        goal=_role_context(
            general=(
                "Infer a goal-relevant visual observation from the current ARC "
                "frame and explain progress-relevant visual evidence."
            ),
            game=_compose_context_game_text(
                game_id=game_id,
                base="This is a live one-step E2E check; focus on the current frame.",
                cheat_action_context=cheat_action_context,
            ),
        ),
    )


def build_model_registry(args: argparse.Namespace) -> ModelRegistry:
    """Build OpenAI-backed agent, world, and goal roles."""

    return ModelRegistry(
        orchestrator_agent=OpenAIOrchestratorAgentAdapter(
            OpenAIOrchestratorAgentConfig(
                model=args.agent_model,
                max_tool_calls=args.max_tool_calls,
                repair_attempts=args.repair_attempts,
                reasoning={"effort": args.reasoning_effort},
                metadata={"role": "agent", "script": Path(__file__).name},
            )
        ),
        world_tool=OpenAIWorldToolAdapter(
            OpenAIWorldToolConfig(
                model=args.world_model,
                image_model=args.image_model,
                reasoning={"effort": args.reasoning_effort},
                image_size=args.image_size,
                image_quality=args.image_quality,
                metadata={"role": "world", "script": Path(__file__).name},
            )
        ),
        goal_tool=OpenAIGoalToolAdapter(
            OpenAIGoalToolConfig(
                model=args.goal_model,
                image_model=args.image_model,
                reasoning={"effort": args.reasoning_effort},
                image_size=args.image_size,
                image_quality=args.image_quality,
                metadata={"role": "goal", "script": Path(__file__).name},
            )
        ),
    )


def validate_live_run(
    *,
    result: GameRunResult,
    environment: RecordingEnvironment,
    states: Sequence[Any],
    experiments: Sequence[Any],
) -> None:
    """Fail fast when the live run did not exercise the intended boundaries."""

    if result.step_count != 1:
        raise RuntimeError(
            f"expected exactly one real environment step, got {result.step_count}"
        )
    if len(environment.step_actions) != 1:
        raise RuntimeError(
            f"expected exactly one submitted environment action, got "
            f"{len(environment.step_actions)}"
        )
    if environment.step_actions[0].is_none():
        raise RuntimeError("synthetic NONE was submitted to the real environment")
    if not states:
        raise RuntimeError("expected at least one M state row")

    experiment_tools = {experiment.tool_name for experiment in experiments}
    missing = {"world", "goal"} - experiment_tools
    if missing:
        raise RuntimeError(
            "expected agent-requested world and goal experiments in E; "
            f"missing: {', '.join(sorted(missing))}"
        )

    for state in states:
        for prediction_name, prediction in (
            ("world_prediction", state.world_prediction),
            ("goal_prediction", state.goal_prediction),
        ):
            if prediction is None:
                continue
            metadata = dict(prediction.get("metadata") or {})
            if metadata.get("prompt_model_calls_enabled") is not False:
                raise RuntimeError(
                    f"{prediction_name} was expected to be mocked because "
                    "prompt_model_calls_enabled=false"
                )


def write_inspection_artifacts(
    *,
    output_dir: Path,
    database_path: Path,
    args: argparse.Namespace,
    run_id: str,
    environment_config: EnvironmentConfig,
    result: GameRunResult,
    trace_text: str,
    recording: RecordingEnvironment,
    states: Sequence[Any],
    experiments: Sequence[Any],
) -> None:
    """Write JSON, text, SQLite, and PNG artifacts for manual inspection."""

    writer = ArtifactWriter(output_dir)
    _save_recorded_observation_images(writer, recording)
    _save_state_and_experiment_images(writer, states, experiments)

    _write_text(output_dir / "frame_trace.txt", trace_text)
    _write_artifact_json(output_dir / "run_result.json", result, writer)
    _write_artifact_json(
        output_dir / "config_snapshot.json",
        _config_snapshot(
            args=args,
            run_id=run_id,
            environment_config=environment_config,
            database_path=database_path,
            output_dir=output_dir,
        ),
        writer,
    )
    _write_artifact_json(output_dir / "m_states.json", list(states), writer)
    _write_artifact_json(output_dir / "e_experiments.json", list(experiments), writer)
    _write_artifact_json(
        output_dir / "decision_trace.json",
        result.decision.trace if result.decision is not None else None,
        writer,
    )
    _write_artifact_json(
        output_dir / "tool_results.json",
        _tool_results_payload(states=states, experiments=experiments),
        writer,
    )
    _write_artifact_json(
        output_dir / "provider_usage.json",
        _provider_usage_payload(states=states, experiments=experiments),
        writer,
    )
    _copy_database(database_path, output_dir / "memory.sqlite")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--database", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--game-index",
        type=int,
        default=None,
        help=f"Game catalog index. Defaults to ls20 index {DEFAULT_GAME_INDEX}.",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help=f"Game id. Defaults to {DEFAULT_GAME_ID}.",
    )
    parser.add_argument("--agent-model", default="gpt-5-nano")
    parser.add_argument("--world-model", default="gpt-5-nano")
    parser.add_argument("--goal-model", default="gpt-5-nano")
    parser.add_argument("--image-model", default="gpt-image-1-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-quality", default="low")
    parser.add_argument("--max-tool-calls", type=int, default=2)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument(
        "--cheat-action-context",
        action="store_true",
        help="Append action semantics parsed from the local game source.",
    )
    parser.add_argument(
        "--game-dir",
        default=str(DEFAULT_GAME_DIR),
        help="Local ARC game directory used for --cheat-action-context.",
    )
    parser.add_argument(
        "--overwrite-database",
        action="store_true",
        help="Allow replacing an existing database path supplied with --database.",
    )
    return parser.parse_args()


def _role_context(*, general: str, game: str) -> Any:
    from face_of_agi.contracts import RoleContext

    return RoleContext(general=general, game=game)


def _compose_context_game_text(
    *,
    game_id: str,
    base: str,
    cheat_action_context: str | None,
) -> str:
    parts = [f"This E2E check uses game {game_id}.", base]
    if cheat_action_context:
        parts.append(
            "Cheat action context from the local game source:\n"
            f"{cheat_action_context}"
        )
    return "\n\n".join(parts)


def _tool_model_options(args: argparse.Namespace, *, role: str) -> dict[str, Any]:
    return {
        "image_model": args.image_model,
        "reasoning": {"effort": args.reasoning_effort},
        "image_size": args.image_size,
        "image_quality": args.image_quality,
        "metadata": {"role": role, "script": Path(__file__).name},
    }


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _resolve_output_dir(output_dir: str | Path) -> Path:
    return _resolve_path(output_dir)


def _resolve_database_path(database: str | None, output_dir: Path) -> Path:
    if database is None:
        return output_dir / "memory.sqlite"
    return _resolve_path(database)


def _prepare_database(
    database_path: Path,
    *,
    database_arg: str | None,
    overwrite: bool,
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if not database_path.exists():
        return
    if database_arg is None or overwrite:
        database_path.unlink()
        return
    raise RuntimeError(
        f"database already exists: {database_path}. Pass a fresh --database path "
        "or remove it before running the E2E script."
    )


def _resolve_game_id_from_catalog(config: EnvironmentConfig) -> str:
    catalog = load_game_catalog(config.game_catalog_path)
    key = str(config.game_index)
    if key not in catalog:
        raise RuntimeError(
            f"game index {config.game_index} was not found in "
            f"{config.game_catalog_path}; run --list-games first"
        )
    return catalog[key]


def _build_run_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.game_id:
        game = args.game_id
    elif args.game_index is not None:
        game = f"game-index-{args.game_index}"
    else:
        game = DEFAULT_GAME_ID
    return f"openai-full-loop-{game}-{timestamp}"


def _load_env_file(env_file: str) -> None:
    if not env_file:
        return
    path = _resolve_path(env_file)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _save_recorded_observation_images(
    writer: ArtifactWriter,
    recording: RecordingEnvironment,
) -> None:
    for index, observation in enumerate(recording.reset_observations):
        _save_observation_images(writer, observation, label=f"reset_{index}")
    for index, observation in enumerate(recording.step_observations):
        _save_observation_images(writer, observation, label=f"post_step_{index}")


def _save_state_and_experiment_images(
    writer: ArtifactWriter,
    states: Sequence[Any],
    experiments: Sequence[Any],
) -> None:
    for state in states:
        frame = dict(state.current_observation).get("frame")
        if isinstance(frame, Image.Image):
            writer.image_reference(frame, label=f"m_state_{state.id}_current")
    for experiment in experiments:
        output_frame = dict(experiment.output_observation).get("frame")
        if isinstance(output_frame, Image.Image):
            writer.image_reference(
                output_frame,
                label=f"e_{experiment.id}_{experiment.tool_name}_output",
            )


def _save_observation_images(
    writer: ArtifactWriter,
    observation: Observation,
    *,
    label: str,
) -> None:
    frames = observation.frames or (
        (observation.frame,) if observation.frame is not None else ()
    )
    if not frames:
        return
    for index, frame in enumerate(frames):
        try:
            image = frame_to_pil_image(
                frame,
                step=observation.step,
                label=f"{label}_{index}",
            )
        except Exception:
            continue
        writer.image_reference(image, label=f"{label}_frame_{index}")
    try:
        writer.image_reference(
            observation_to_pil_image(observation),
            label=f"{label}_visible",
        )
    except Exception:
        pass


def _write_artifact_json(path: Path, value: Any, writer: ArtifactWriter) -> None:
    payload = _artifact_jsonable(value, writer, label=path.stem)
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_jsonable(value: Any, writer: ArtifactWriter, *, label: str) -> Any:
    if isinstance(value, Image.Image):
        return writer.image_reference(value, label=label)
    if is_dataclass(value):
        return {
            field.name: _artifact_jsonable(
                getattr(value, field.name),
                writer,
                label=f"{label}_{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _artifact_jsonable(item, writer, label=f"{label}_{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _artifact_jsonable(item, writer, label=f"{label}_{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _config_snapshot(
    *,
    args: argparse.Namespace,
    run_id: str,
    environment_config: EnvironmentConfig,
    database_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_path": str(_resolve_path(args.config)),
        "output_dir": str(output_dir),
        "database_path": str(database_path),
        "game_index": environment_config.game_index,
        "game_id": environment_config.game_id,
        "max_actions_per_level": environment_config.max_actions_per_level,
        "enable_visualization": environment_config.enable_visualization,
        "save_recording": environment_config.save_recording,
        "include_frame_data": environment_config.include_frame_data,
        "cheat_action_context_enabled": args.cheat_action_context,
        "cheat_action_context_game_dir": str(_resolve_path(args.game_dir)),
        "experimental_memory_turn_buffer": (
            environment_config.experimental_memory_turn_buffer
        ),
        "models": {
            "prompt_model_calls_enabled": (
                environment_config.models.prompt_model_calls_enabled
            ),
            "agent": _model_role_payload(environment_config.models.agent),
            "world": _model_role_payload(environment_config.models.world),
            "goal": _model_role_payload(environment_config.models.goal),
        },
    }


def _model_role_payload(config: ModelRoleConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "model": config.model,
        "max_tool_calls": config.max_tool_calls,
        "repair_attempts": config.repair_attempts,
        "options": config.options,
    }


def _tool_results_payload(
    *,
    states: Sequence[Any],
    experiments: Sequence[Any],
) -> dict[str, Any]:
    trace_results = []
    for state in states:
        trace_results.extend(dict(state.agent_trace).get("tool_results") or [])
    return {
        "agent_trace_tool_results": trace_results,
        "experimental_tool_results": [
            experiment.tool_result for experiment in experiments
        ],
    }


def _provider_usage_payload(
    *,
    states: Sequence[Any],
    experiments: Sequence[Any],
) -> dict[str, Any]:
    agent_calls = []
    for state in states:
        metadata = dict(dict(state.agent_trace).get("metadata") or {})
        if metadata:
            agent_calls.append(
                {
                    "state_id": state.id,
                    "backend": metadata.get("backend"),
                    "model": metadata.get("model"),
                    "provider_response_ids": metadata.get("provider_response_ids"),
                    "tool_call_count": metadata.get("tool_call_count"),
                    "repair_count": metadata.get("repair_count"),
                    "usage": metadata.get("usage"),
                }
            )

    tool_calls = []
    for experiment in experiments:
        tool_result = dict(experiment.tool_result)
        metadata = dict(tool_result.get("metadata") or {})
        tool_calls.append(
            {
                "experiment_id": experiment.id,
                "tool": experiment.tool_name,
                "response_id": metadata.get("response_id"),
                "response_model": metadata.get("response_model"),
                "image_generation_call_id": metadata.get("image_generation_call_id"),
                "usage": metadata.get("usage"),
            }
        )

    return {"agent": agent_calls, "tools": tool_calls}


def _copy_database(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _safe_slug(value: str, *, max_length: int = 80) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    slug = "".join(allowed).strip("_") or "image"
    return slug[:max_length]


if __name__ == "__main__":
    main()
