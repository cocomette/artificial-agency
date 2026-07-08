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
    RuntimeConfig,
)
from face_of_agi.environment import ArcEnvironmentAdapter, load_environment_config
from face_of_agi.environment.config import (
    ModelRoleConfig,
    UpdaterRuntimeConfig,
    load_game_catalog,
    write_game_catalog,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import (
    GoalPredictionAdapter,
    ModelRegistry,
    OllamaDescriptionConfig,
    OllamaOrchestratorAgentConfig,
    OpenAIDescriptionConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIUpdaterConfig,
    OllamaUpdaterConfig,
    OrchestratorAgentConfig,
    UpdaterConfig,
    UpdaterTaskRegistry,
    VLLMDescriptionConfig,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterConfig,
    WorldPredictionAdapter,
)
from face_of_agi.models.orchestrator_agent.providers import (
    OllamaOrchestratorAgentAdapter,
    OpenAIOrchestratorAgentAdapter,
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.updater.providers import (
    ConfigurableUpdaterAdapter,
    HuggingFaceUpdaterAdapter,
    OllamaUpdaterAdapter,
    OpenAIUpdaterAdapter,
    VLLMUpdaterAdapter,
)
from face_of_agi.orchestration import Orchestrator
from face_of_agi.runtime.loop import RuntimeLoop

DEFAULT_DATABASE_PATH = Path("runs/memory.sqlite")


def main() -> None:
    """Run the starter ARC shell from a YAML config file."""

    parser = _build_parser()
    args = parser.parse_args()
    playback_request = _playback_request_from_args(parser, args)
    database_path = Path(args.database)
    if args.clean_db:
        state = StateMemory(SQLiteDatabase(database_path))
        state.clear_memory_tables()
        print(f"cleared memory database rows from {database_path}")
        return

    config_path = Path(args.config)
    environment_config = load_environment_config(config_path)
    if args.debug_keep_all_m_states or playback_request is not None:
        environment_config.debug_keep_all_m_states = True
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

    if playback_request is not None:
        environment_config.game_id = playback_request.game_id
        environment_config.use_learned_contexts = False
    else:
        environment_config.game_id = _resolve_selected_game_id(environment_config)
    runtime_config = RuntimeConfig(
        run_id=_build_run_id(environment_config),
        database_path=database_path,
    )
    model_registry = _build_model_registry(
        agent_config=environment_config.models.agent,
        world_config=environment_config.models.world,
        goal_config=environment_config.models.goal,
        shared_vlm_config=environment_config.models.shared_vlm,
        updater_config=environment_config.models.updater,
    )
    contexts = None
    if playback_request is not None:
        from debug.playback import prepare_playback

        playback = prepare_playback(
            state_memory=StateMemory(SQLiteDatabase(database_path)),
            request=playback_request,
            live_models=model_registry,
        )
        model_registry = playback.models
        contexts = playback.contexts
        print(
            "playback:"
            f" source_run={playback_request.source_run_id}"
            f" game={playback_request.game_id}"
            f" handoff_turn={playback_request.turn_id}"
            f" replay_turns={playback.replay_turn_count}"
        )

    runtime = RuntimeLoop(
        _build_orchestrator(
            database_path,
            experimental_memory_turn_buffer=(
                environment_config.experimental_memory_turn_buffer
            ),
            models=model_registry,
            contexts=contexts,
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
        default="src/face_of_agi/runtime/configs/starter_loop.yaml",
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
    parser.add_argument(
        "--debug-keep-all-m-states",
        action="store_true",
        help="Preserve every M state row after a successful run for debugging.",
    )
    parser.add_argument(
        "--playback-run-id",
        help="Debug playback source run id. Requires playback game and turn ids.",
    )
    parser.add_argument(
        "--playback-game-id",
        help="Debug playback source game id. Overrides the config-selected game.",
    )
    parser.add_argument(
        "--playback-turn-id",
        type=int,
        help="Debug playback handoff turn id selected in persisted M memory.",
    )
    return parser


def _playback_request_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Any | None:
    """Return a debug playback request when all playback flags are present."""

    values = (
        args.playback_run_id,
        args.playback_game_id,
        args.playback_turn_id,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        parser.error(
            "--playback-run-id, --playback-game-id, and --playback-turn-id "
            "must be provided together"
        )
    if args.clean_db or args.list_games:
        parser.error("debug playback flags cannot be combined with setup commands")

    from debug.playback import PlaybackRequest

    return PlaybackRequest(
        source_run_id=str(args.playback_run_id),
        game_id=str(args.playback_game_id),
        turn_id=int(args.playback_turn_id),
    )


def _build_orchestrator(
    database_path: Path,
    *,
    experimental_memory_turn_buffer: int = 2,
    agent_config: ModelRoleConfig | None = None,
    world_config: ModelRoleConfig | None = None,
    goal_config: ModelRoleConfig | None = None,
    shared_vlm_config: ModelRoleConfig | None = None,
    updater_config: UpdaterRuntimeConfig | None = None,
    contexts: ContextDocuments | None = None,
    models: ModelRegistry | None = None,
) -> Orchestrator:
    """Assemble orchestration with persistent SQLite-backed memory."""

    database = SQLiteDatabase(database_path)
    return Orchestrator(
        state_memory=StateMemory(database),
        experimental_memory=ExperimentalMemory(database),
        models=models
        or _build_model_registry(
            agent_config=agent_config or ModelRoleConfig(),
            world_config=world_config or ModelRoleConfig(),
            goal_config=goal_config or ModelRoleConfig(),
            shared_vlm_config=shared_vlm_config or ModelRoleConfig(),
            updater_config=updater_config,
        ),
        contexts=contexts,
        experimental_memory_turn_buffer=experimental_memory_turn_buffer,
    )


def _build_model_registry(
    *,
    agent_config: ModelRoleConfig,
    world_config: ModelRoleConfig,
    goal_config: ModelRoleConfig | None = None,
    shared_vlm_config: ModelRoleConfig | None = None,
    updater_config: UpdaterRuntimeConfig | None = None,
) -> ModelRegistry:
    """Build model role adapters from starter YAML config."""

    del goal_config
    shared_vlm_config = shared_vlm_config or ModelRoleConfig()
    shared_ollama_client = _shared_ollama_client(shared_vlm_config)
    world_role_config = _with_shared_vlm_role_config(world_config, shared_vlm_config)
    return ModelRegistry(
        orchestrator_agent=_build_agent(
            _with_shared_vlm_role_config(agent_config, shared_vlm_config),
            ollama_client=shared_ollama_client,
        ),
        world_prediction_model=_build_world_prediction_model(
            world_role_config,
            ollama_client=shared_ollama_client,
        ),
        goal_prediction_model=None,
        updater_tasks=_build_updater_tasks(
            _with_shared_vlm_updater_config(updater_config, shared_vlm_config),
            ollama_client=shared_ollama_client,
        ),
    )


def _build_agent(
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build the selected X agent adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError("models.agent.backend is required")
    backend = config.backend.lower()
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
            ),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_role_model("models.agent", backend, config)
        return VLLMOrchestratorAgentAdapter(
            VLLMOrchestratorAgentConfig(
                **_config_kwargs(config, VLLMOrchestratorAgentConfig)
            )
        )
    if backend in {"huggingface", "huggingface-diffusers"}:
        raise NotImplementedError("Hugging Face Agent X provider is not implemented yet")
    if backend == "configurable":
        raise NotImplementedError("Configurable Agent X provider is not implemented yet")
    raise ValueError(f"unknown agent backend: {config.backend}")


def _build_world_prediction_model(
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build the selected world prediction adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError("models.world.backend is required")
    backend = config.backend.lower()
    if backend == "openai":
        return WorldPredictionAdapter(
            OpenAIDescriptionConfig(**_config_kwargs(config, OpenAIDescriptionConfig))
        )
    if backend == "ollama":
        return WorldPredictionAdapter(
            OllamaDescriptionConfig(**_config_kwargs(config, OllamaDescriptionConfig)),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_role_model("models.world", backend, config)
        return WorldPredictionAdapter(
            VLLMDescriptionConfig(**_config_kwargs(config, VLLMDescriptionConfig))
        )
    if backend in {"huggingface-diffusers", "diffusers", "huggingface"}:
        raise NotImplementedError(
            "Hugging Face/Diffusers world prediction image generation was removed; "
            "use backend openai, ollama, or vllm."
        )
    raise ValueError(f"unknown world backend: {config.backend}")


def _build_goal_prediction_model(
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build the selected goal prediction adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError("models.goal.backend is required")
    backend = config.backend.lower()
    if backend == "openai":
        return GoalPredictionAdapter(
            OpenAIDescriptionConfig(**_config_kwargs(config, OpenAIDescriptionConfig))
        )
    if backend == "ollama":
        return GoalPredictionAdapter(
            OllamaDescriptionConfig(**_config_kwargs(config, OllamaDescriptionConfig)),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_role_model("models.goal", backend, config)
        return GoalPredictionAdapter(
            VLLMDescriptionConfig(**_config_kwargs(config, VLLMDescriptionConfig))
        )
    if backend in {"huggingface-diffusers", "diffusers", "huggingface"}:
        raise NotImplementedError(
            "Hugging Face/Diffusers goal prediction image generation was removed; "
            "use backend openai, ollama, or vllm."
        )
    raise ValueError(f"unknown goal backend: {config.backend}")


def _build_updater_tasks(
    config: UpdaterRuntimeConfig | None,
    *,
    ollama_client: object | None = None,
) -> UpdaterTaskRegistry:
    """Build configured updater P task adapters."""

    if config is None:
        raise ValueError("models.updater config is required")
    return UpdaterTaskRegistry(
        world_game_updater=_build_updater_task(
            "world",
            config.world,
            ollama_client=ollama_client,
        ),
        goal_game_updater=None,
        agent_game_updater=_build_updater_task(
            "agent",
            config.agent,
            ollama_client=ollama_client,
        ),
        general_updater=_build_updater_task(
            "general",
            config.general,
            ollama_client=ollama_client,
        ),
    )


def _build_updater_task(
    task_name: str,
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build one selected updater P task adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError(f"models.updater.{task_name}.backend is required")
    backend = config.backend.lower()
    updater_config = UpdaterConfig(**_config_kwargs(config, UpdaterConfig))
    if backend == "openai":
        _require_prompt_updater_task(task_name, backend)
        _require_updater_model(task_name, backend, config)
        return OpenAIUpdaterAdapter(
            OpenAIUpdaterConfig(**_config_kwargs(config, OpenAIUpdaterConfig))
        )
    if backend == "ollama":
        _require_prompt_updater_task(task_name, backend)
        _require_updater_model(task_name, backend, config)
        return OllamaUpdaterAdapter(
            OllamaUpdaterConfig(**_config_kwargs(config, OllamaUpdaterConfig)),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_prompt_updater_task(task_name, backend)
        _require_updater_model(task_name, backend, config)
        return VLLMUpdaterAdapter(
            VLLMUpdaterConfig(**_config_kwargs(config, VLLMUpdaterConfig))
        )
    if backend in {"huggingface", "huggingface-diffusers"}:
        return HuggingFaceUpdaterAdapter(updater_config)
    if backend == "configurable":
        return ConfigurableUpdaterAdapter(updater_config)
    raise ValueError(f"unknown updater backend: {config.backend}")


def _require_prompt_updater_task(task_name: str, backend: str) -> None:
    """Fail clearly for real updater slots that are not implemented yet."""

    if task_name not in {"world", "goal", "agent", "general"}:
        raise NotImplementedError(
            f"{backend} updater is implemented only for world, goal, agent, "
            "and general prompt tasks"
        )


def _require_updater_model(
    task_name: str,
    backend: str,
    config: ModelRoleConfig,
) -> None:
    """Require explicit model names for real updater providers."""

    if not config.model:
        raise ValueError(
            f"models.updater.{task_name}.model is required for backend {backend}"
        )


def _require_role_model(role_path: str, backend: str, config: ModelRoleConfig) -> None:
    """Require explicit model names for real model providers without defaults."""

    if not config.model:
        raise ValueError(f"{role_path}.model is required for backend {backend}")


def _shared_ollama_client(config: ModelRoleConfig) -> object | None:
    """Return one shared Ollama client for local VLM roles when configured."""

    if (config.backend or "").lower() != "ollama":
        return None
    try:
        import ollama
    except ImportError:
        return None

    host = config.options.get("host")
    if host:
        return ollama.Client(host=host)
    return ollama


def _with_shared_vlm_role_config(
    config: ModelRoleConfig,
    shared: ModelRoleConfig,
) -> ModelRoleConfig:
    """Apply shared local VLM defaults to matching local role configs."""

    backend = (config.backend or "").lower()
    if backend not in {"ollama", "vllm"}:
        return config
    if backend != (shared.backend or "").lower():
        return config
    shared_options = (
        _shared_ollama_runtime_options(shared)
        if backend == "ollama"
        else _shared_vllm_runtime_options(shared)
    )

    return ModelRoleConfig(
        backend=config.backend,
        model=config.model or shared.model,
        max_tool_calls=config.max_tool_calls,
        repair_attempts=config.repair_attempts,
        options=_deep_merge_dicts(
            shared_options,
            config.options,
        ),
    )


def _with_shared_vlm_updater_config(
    config: UpdaterRuntimeConfig | None,
    shared: ModelRoleConfig,
) -> UpdaterRuntimeConfig | None:
    """Apply shared local VLM defaults to matching updater task configs."""

    if config is None:
        return None
    return UpdaterRuntimeConfig(
        world=_with_shared_vlm_role_config(config.world, shared),
        goal=_with_shared_vlm_role_config(config.goal, shared),
        agent=_with_shared_vlm_role_config(config.agent, shared),
        general=_with_shared_vlm_role_config(config.general, shared),
    )


def _shared_ollama_runtime_options(config: ModelRoleConfig) -> dict[str, Any]:
    """Return shared Ollama behavior options without changing role prompts."""

    recognized = {"host", "think", "keep_alive", "options"}
    options = {
        key: value
        for key, value in config.options.items()
        if key in recognized
    }
    generation_options = {
        key: value
        for key, value in config.options.items()
        if key not in recognized
    }
    if generation_options:
        existing_options = options.get("options")
        if isinstance(existing_options, dict):
            options["options"] = _deep_merge_dicts(
                existing_options,
                generation_options,
            )
        elif "options" not in options:
            options["options"] = generation_options
    return options


def _shared_vllm_runtime_options(config: ModelRoleConfig) -> dict[str, Any]:
    """Return shared vLLM behavior options without changing role prompts."""

    modal_server_keys = {"server", "server_args"}
    return {
        key: value
        for key, value in config.options.items()
        if key not in modal_server_keys
    }


def _deep_merge_dicts(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge dictionaries while replacing scalars and lists."""

    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _config_kwargs(config: ModelRoleConfig, config_type: type) -> dict[str, Any]:
    """Return dataclass kwargs supported by one config class."""

    allowed = {field.name for field in fields(config_type)}
    kwargs: dict[str, Any] = {}
    provider_options: dict[str, Any] = {}
    explicit_options = config.options.get("options")
    if "options" in allowed and isinstance(explicit_options, dict):
        provider_options = dict(explicit_options)
    elif "options" in allowed and explicit_options is not None:
        kwargs["options"] = explicit_options

    for key, value in config.options.items():
        if key == "options":
            continue
        if key in allowed:
            kwargs[key] = value
        elif "options" in allowed:
            provider_options[key] = value

    if "options" in allowed and provider_options:
        existing_options = kwargs.get("options")
        if isinstance(existing_options, dict):
            kwargs["options"] = _deep_merge_dicts(existing_options, provider_options)
        elif existing_options is None:
            kwargs["options"] = provider_options
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
