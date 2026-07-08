"""Runnable starter shell for the ARC-AGI environment loop."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from face_of_agi.contracts import (
    ContextDocuments,
    GameRunResult,
    RoleContext,
    RuntimeConfig,
)
from face_of_agi.environment import ArcEnvironmentAdapter, load_environment_config
from face_of_agi.environment.cheat_context import (
    load_cheat_action_context,
    resolve_cheat_action_context_game_dir,
)
from face_of_agi.environment.config import (
    EnvironmentConfig,
    ModelRoleConfig,
    load_game_catalog,
    write_game_catalog,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import (
    GoalToolAdapter,
    GoalToolConfig,
    ModelRegistry,
    OllamaOrchestratorAgentConfig,
    OpenAIGoalToolConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIWorldToolConfig,
    OrchestratorAgentConfig,
    WorldToolAdapter,
    WorldToolConfig,
)
from face_of_agi.models.orchestrator_agent.providers import (
    OllamaOrchestratorAgentAdapter,
    OpenAIOrchestratorAgentAdapter,
    RandomOrchestratorAgentAdapter,
)
from face_of_agi.models.tools.goal.providers.openai import OpenAIGoalToolAdapter
from face_of_agi.models.tools.world.providers.openai import OpenAIWorldToolAdapter
from face_of_agi.orchestration import Orchestrator
from face_of_agi.runtime.loop import RuntimeLoop

DEFAULT_DATABASE_PATH = Path("runs/memory.sqlite")


def main() -> None:
    """Run the starter ARC shell from a YAML config file."""

    args = _build_parser().parse_args()
    database_path = Path(args.database)
    if args.clean_db:
        state = StateMemory(SQLiteDatabase(database_path))
        state.clear_memory_tables()
        print(f"cleared memory database rows from {database_path}")
        return

    config_path = Path(args.config)
    environment_config = load_environment_config(config_path)
    environment = ArcEnvironmentAdapter.from_config(environment_config)

    if args.list_games:
        games = tuple(environment.list_available_games())
        if not games:
            print("no ARC games discovered by Arcade().get_environments()")
            return

        catalog = {
            str(index): game.game_id
            for index, game in enumerate(games)
        }
        write_game_catalog(environment_config.game_catalog_path, catalog)
        for index, game in enumerate(games):
            print(f"{index}: {game.game_id}: {game.title}")
        print(f"wrote game catalog to {environment_config.game_catalog_path}")
        return

    environment_config.game_id = _resolve_selected_game_id(environment_config)
    runtime_config = RuntimeConfig(
        run_id=_build_run_id(environment_config),
        database_path=database_path,
    )
    runtime = RuntimeLoop(
        _build_orchestrator(
            database_path,
            experimental_memory_turn_buffer=(
                environment_config.experimental_memory_turn_buffer
            ),
            agent_config=environment_config.models.agent,
            world_config=environment_config.models.world,
            goal_config=environment_config.models.goal,
            prompt_model_calls_enabled=(
                environment_config.models.prompt_model_calls_enabled
            ),
            contexts=_build_context_documents(environment_config),
        )
    )

    try:
        result = runtime.run(
            config=runtime_config,
            environment=environment,
            environment_config=environment_config,
        )
    except Exception as exc:
        print(f"starter shell failed: {exc}")
        raise SystemExit(1) from exc

    if not isinstance(result, GameRunResult):
        raise RuntimeError("starter shell expected a single-game runtime result")

    print(
        "stop:"
        f" reason={result.stop_reason}"
        f" steps={result.step_count}"
        f" completed_levels={result.completed_levels}"
        f" last_state={result.last_state}"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the small CLI for the starter shell."""

    parser = argparse.ArgumentParser(
        description="Run the minimal ARC-AGI starter shell.",
    )
    parser.add_argument(
        "--config",
        default="src/face_of_agi/runtime/starter_loop.yaml",
        help="Path to the environment YAML config.",
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="Print the live ARC toolkit environment list with indices and exit.",
    )
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DATABASE_PATH),
        help="Path to the SQLite memory database.",
    )
    parser.add_argument(
        "--clean-db",
        action="store_true",
        help="Clear memory database rows and exit without starting ARC.",
    )
    return parser


def _build_orchestrator(
    database_path: Path,
    *,
    experimental_memory_turn_buffer: int = 2,
    agent_config: ModelRoleConfig | None = None,
    world_config: ModelRoleConfig | None = None,
    goal_config: ModelRoleConfig | None = None,
    prompt_model_calls_enabled: bool = False,
    contexts: ContextDocuments | None = None,
) -> Orchestrator:
    """Assemble orchestration with persistent SQLite-backed memory."""

    database = SQLiteDatabase(database_path)
    return Orchestrator(
        state_memory=StateMemory(database),
        experimental_memory=ExperimentalMemory(database),
        models=_build_model_registry(
            agent_config=agent_config or ModelRoleConfig(),
            world_config=world_config or ModelRoleConfig(),
            goal_config=goal_config or ModelRoleConfig(),
        ),
        contexts=contexts,
        experimental_memory_turn_buffer=experimental_memory_turn_buffer,
        prompt_model_calls_enabled=prompt_model_calls_enabled,
    )


def _build_context_documents(config: EnvironmentConfig) -> ContextDocuments:
    """Build configured role contexts for the runtime shell."""

    if not config.cheat_action_context:
        return ContextDocuments()
    if config.game_id is None:
        raise RuntimeError("cheat action context requires a resolved game_id")

    game_dir = (
        Path(config.cheat_action_context_game_dir)
        if config.cheat_action_context_game_dir
        else resolve_cheat_action_context_game_dir(
            environments_dir=config.environments_dir,
            game_id=config.game_id,
        )
    )
    action_context = load_cheat_action_context(game_dir)
    game_context = _compose_cheat_action_context_text(
        game_id=config.game_id,
        action_context=action_context,
    )
    return ContextDocuments(
        agent=RoleContext(game=game_context),
        world=RoleContext(game=game_context),
        goal=RoleContext(game=game_context),
    )


def _compose_cheat_action_context_text(
    *,
    game_id: str,
    action_context: str,
) -> str:
    return "\n\n".join(
        [
            f"This runtime run uses game {game_id}.",
            "Use these action semantics when reasoning about GameAction values.",
            "Cheat action context from the local game source:\n"
            f"{action_context}",
        ]
    )


def _build_model_registry(
    *,
    agent_config: ModelRoleConfig,
    world_config: ModelRoleConfig,
    goal_config: ModelRoleConfig,
) -> ModelRegistry:
    """Build model role adapters from starter YAML config."""

    return ModelRegistry(
        orchestrator_agent=_build_agent(agent_config),
        world_tool=_build_world_tool(world_config),
        goal_tool=_build_goal_tool(goal_config),
    )


def _build_agent(config: ModelRoleConfig) -> object | None:
    """Build the selected X agent adapter."""

    backend = (config.backend or "random").lower()
    if backend in {"", "none", "random"}:
        return RandomOrchestratorAgentAdapter(
            OrchestratorAgentConfig(**_config_kwargs(config, OrchestratorAgentConfig))
        )
    if backend == "openai":
        return OpenAIOrchestratorAgentAdapter(
            OpenAIOrchestratorAgentConfig(
                **_config_kwargs(config, OpenAIOrchestratorAgentConfig)
            )
        )
    if backend == "ollama":
        return OllamaOrchestratorAgentAdapter(
            OllamaOrchestratorAgentConfig(
                **_config_kwargs(config, OllamaOrchestratorAgentConfig)
            )
        )
    if backend in {"huggingface", "huggingface-diffusers"}:
        raise NotImplementedError("Hugging Face Agent X provider is not implemented yet")
    if backend == "configurable":
        raise NotImplementedError("Configurable Agent X provider is not implemented yet")
    raise ValueError(f"unknown agent backend: {config.backend}")


def _build_world_tool(config: ModelRoleConfig) -> object | None:
    """Build the selected world tool adapter."""

    backend = (config.backend or "none").lower()
    if backend in {"", "none"}:
        return None
    if backend == "openai":
        return OpenAIWorldToolAdapter(
            OpenAIWorldToolConfig(**_config_kwargs(config, OpenAIWorldToolConfig))
        )
    if backend in {"huggingface-diffusers", "diffusers"}:
        return WorldToolAdapter(
            WorldToolConfig(**_config_kwargs(config, WorldToolConfig))
        )
    raise ValueError(f"unknown world backend: {config.backend}")


def _build_goal_tool(config: ModelRoleConfig) -> object | None:
    """Build the selected goal tool adapter."""

    backend = (config.backend or "none").lower()
    if backend in {"", "none"}:
        return None
    if backend == "openai":
        return OpenAIGoalToolAdapter(
            OpenAIGoalToolConfig(**_config_kwargs(config, OpenAIGoalToolConfig))
        )
    if backend in {"huggingface-diffusers", "diffusers"}:
        return GoalToolAdapter(GoalToolConfig(**_config_kwargs(config, GoalToolConfig)))
    raise ValueError(f"unknown goal backend: {config.backend}")


def _config_kwargs(config: ModelRoleConfig, config_type: type) -> dict[str, Any]:
    """Return dataclass kwargs supported by one config class."""

    allowed = {field.name for field in fields(config_type)}
    kwargs: dict[str, Any] = {key: value for key, value in config.options.items()}
    if config.backend is not None:
        kwargs["backend"] = config.backend
    if config.model is not None:
        kwargs["model"] = config.model
    if config.max_tool_calls is not None:
        kwargs["max_tool_calls"] = config.max_tool_calls
    if config.repair_attempts is not None:
        kwargs["repair_attempts"] = config.repair_attempts
    return {key: value for key, value in kwargs.items() if key in allowed}


def _build_run_id(config: object) -> str:
    """Create a stable enough run id for local shell runs."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    game_index = getattr(config, "game_index", "unknown")
    return f"game-index-{game_index}-{timestamp}"


def _resolve_selected_game_id(config: object) -> str:
    """Resolve the chosen game index from the stored local catalog file."""

    game_index = getattr(config, "game_index")
    game_catalog_path = getattr(config, "game_catalog_path")
    catalog = load_game_catalog(game_catalog_path)
    key = str(game_index)
    if key not in catalog:
        raise RuntimeError(
            f"game index {game_index} was not found in {game_catalog_path}; "
            "run --list-games first"
        )
    return catalog[key]


if __name__ == "__main__":
    main()
