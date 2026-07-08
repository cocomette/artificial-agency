"""Runnable starter shell for the ARC-AGI environment loop."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, TextIO

from face_of_agi.contracts import (
    ContextDocuments,
    GameRunResult,
    ParallelGameRunResult,
    RuntimeConfig,
)
from face_of_agi.debug.sinks import LiveTurnMonitor
from face_of_agi.environment import ArcEnvironmentAdapter, load_environment_config
from face_of_agi.environment.config import (
    ModelSchedulerConfig,
    ModelRoleConfig,
    UpdaterRuntimeConfig,
    load_game_catalog,
    write_game_catalog,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import (
    ChangeSummaryAdapter,
    DisabledGameMemoryAdapter,
    ModelRegistry,
    OllamaChangeSummaryConfig,
    OllamaGameMemoryConfig,
    OllamaHistorizerConfig,
    OllamaOrchestratorAgentConfig,
    OpenAIChangeSummaryConfig,
    OpenAIGameMemoryConfig,
    OpenAIHistorizerConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIUpdaterConfig,
    OllamaUpdaterConfig,
    OrchestratorAgentConfig,
    UpdaterConfig,
    UpdaterTaskRegistry,
    VLLMChangeSummaryConfig,
    VLLMGameMemoryConfig,
    VLLMHistorizerConfig,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterConfig,
)
from face_of_agi.models.orchestrator_agent.providers import (
    OllamaOrchestratorAgentAdapter,
    OpenAIOrchestratorAgentAdapter,
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.historizer.providers import (
    OllamaHistorizerAdapter,
    OpenAIHistorizerAdapter,
    VLLMHistorizerAdapter,
)
from face_of_agi.models.memory.providers import (
    OllamaGameMemoryAdapter,
    OpenAIGameMemoryAdapter,
    VLLMGameMemoryAdapter,
)
from face_of_agi.models.updater.providers import (
    ConfigurableUpdaterAdapter,
    HuggingFaceUpdaterAdapter,
    OllamaUpdaterAdapter,
    OpenAIUpdaterAdapter,
    VLLMUpdaterAdapter,
)
from face_of_agi.models.providers.scheduler import ModelCallScheduler
from face_of_agi.orchestration import Orchestrator
from face_of_agi.runtime.loop import RuntimeLoop
from face_of_agi.runtime.parallel import ParallelGameRunSpec, ParallelRuntimeLoop

DEFAULT_DATABASE_PATH = Path("runs/memory.sqlite")
_SCHEDULERS_LOCK = threading.Lock()
_SCHEDULERS: dict[tuple[int, int], ModelCallScheduler] = {}


def main() -> None:
    """Run the starter ARC shell from a YAML config file."""

    parser = _build_parser()
    args = parser.parse_args()
    playback_request = _playback_request_from_args(parser, args)
    database_path = Path(args.database)
    config_path = Path(args.config)
    if args.clean_db:
        environment_config = _optional_environment_config_for_clean_db(config_path)
        cleared_paths = _clean_database_paths(
            database_path=database_path,
            environment_config=environment_config,
        )
        for cleared_path in cleared_paths:
            print(f"cleared memory database rows from {cleared_path}")
        return

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

    if _is_parallel_selection(environment_config):
        if playback_request is not None:
            parser.error("debug playback supports only single-game configs")
        result = _run_parallel_config(
            base_environment_config=environment_config,
            base_database_path=database_path,
            trace_output=None,
        )
        _print_parallel_result(result)
        if result.failures:
            raise SystemExit(1)
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
        change_config=environment_config.models.change,
        historizer_config=environment_config.models.historizer,
        memory_config=environment_config.models.memory,
        shared_vlm_config=environment_config.models.shared_vlm,
        scheduler_config=environment_config.models.scheduler,
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


def _run_parallel_config(
    *,
    base_environment_config: Any,
    base_database_path: Path,
    trace_output: TextIO | None,
) -> ParallelGameRunResult:
    """Run all configured games with isolated runtime state."""

    batch_run_id = _build_parallel_batch_run_id()
    selected_games = _resolve_selected_game_ids(base_environment_config)
    live_turn_monitor = (
        LiveTurnMonitor(
            selected_game_count=len(selected_games),
            output=trace_output,
        )
        if base_environment_config.live_turn_monitor
        else None
    )
    specs = tuple(
        ParallelGameRunSpec(
            game_index=game_index,
            game_id=game_id,
            run_id=f"{batch_run_id}-game-index-{game_index}",
            database_path=_database_path_for_game(base_database_path, game_index),
            environment_config=replace(
                base_environment_config,
                game_index=game_index,
                game_indices=(),
                game_ids=(),
                game_selection=None,
                game_id=game_id,
            ),
            live_turn_monitor=live_turn_monitor,
        )
        for game_index, game_id in selected_games
    )
    return ParallelRuntimeLoop(
        _run_parallel_game,
        trace_output=trace_output,
    ).run(
        batch_run_id=batch_run_id,
        specs=specs,
        max_parallel_games=base_environment_config.max_parallel_games,
        max_game_retries=base_environment_config.max_game_retries,
    )


def _is_parallel_selection(environment_config: Any) -> bool:
    """Return whether a config selects a multi-game runtime path."""

    return bool(
        environment_config.game_indices
        or environment_config.game_ids
        or environment_config.game_selection == "all_available"
    )


def _run_parallel_game(
    spec: ParallelGameRunSpec,
    trace_output: TextIO,
) -> GameRunResult:
    """Run one worker game with fresh adapters and its own SQLite database."""

    environment_config = spec.environment_config
    environment = ArcEnvironmentAdapter.from_config(environment_config)
    runtime_config = RuntimeConfig(
        run_id=spec.run_id,
        database_path=spec.database_path,
    )
    model_registry = _build_model_registry(
        agent_config=environment_config.models.agent,
        change_config=environment_config.models.change,
        historizer_config=environment_config.models.historizer,
        memory_config=environment_config.models.memory,
        shared_vlm_config=environment_config.models.shared_vlm,
        scheduler_config=environment_config.models.scheduler,
        updater_config=environment_config.models.updater,
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
        environment=environment,
        environment_config=environment_config,
    )


def _print_parallel_result(result: ParallelGameRunResult) -> None:
    """Print concise worker outcomes after a parallel shell run."""

    for success in result.successes:
        game_result = success.result
        print(
            "stop:"
            f" game_index={success.game_index}"
            f" game_id={success.game_id}"
            f" run_id={success.run_id}"
            f" database={success.database_path}"
            f" attempts={success.attempt_count}"
            f" reason={game_result.stop_reason}"
            f" steps={game_result.step_count}"
            f" completed_levels={game_result.completed_levels}"
            f" last_state={game_result.last_state}"
        )
    for failure in result.failures:
        print(
            "failed:"
            f" game_index={failure.game_index}"
            f" game_id={failure.game_id}"
            f" run_id={failure.run_id}"
            f" database={failure.database_path}"
            f" attempts={failure.attempt_count}"
            f" error={failure.exception_type}: {failure.message}"
        )
    print(
        "parallel stop:"
        f" batch_run_id={result.batch_run_id}"
        f" succeeded={len(result.successes)}"
        f" failed={len(result.failures)}"
    )


def _clean_database_paths(
    *,
    database_path: Path,
    environment_config: Any | None,
) -> tuple[Path, ...]:
    """Clear the single-game database or all derived parallel databases."""

    paths = (
        tuple(
            _database_path_for_game(database_path, game_index)
            for game_index, _game_id in _resolve_selected_game_ids(environment_config)
        )
        if environment_config is not None and _is_parallel_selection(environment_config)
        else (database_path,)
    )
    for path in paths:
        state = StateMemory(SQLiteDatabase(path))
        state.clear_memory_tables()
    return paths


def _optional_environment_config_for_clean_db(config_path: Path) -> Any | None:
    """Load config for parallel clean-db, preserving legacy missing-file cleanup."""

    try:
        return load_environment_config(config_path)
    except FileNotFoundError:
        return None


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
    change_config: ModelRoleConfig | None = None,
    historizer_config: ModelRoleConfig | None = None,
    memory_config: ModelRoleConfig | None = None,
    shared_vlm_config: ModelRoleConfig | None = None,
    scheduler_config: ModelSchedulerConfig | None = None,
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
            change_config=change_config or ModelRoleConfig(),
            historizer_config=historizer_config or ModelRoleConfig(),
            memory_config=memory_config or ModelRoleConfig(),
            shared_vlm_config=shared_vlm_config or ModelRoleConfig(),
            scheduler_config=scheduler_config or ModelSchedulerConfig(),
            updater_config=updater_config,
        ),
        contexts=contexts,
        experimental_memory_turn_buffer=experimental_memory_turn_buffer,
    )


def _build_model_registry(
    *,
    agent_config: ModelRoleConfig,
    change_config: ModelRoleConfig,
    historizer_config: ModelRoleConfig | None = None,
    memory_config: ModelRoleConfig | None = None,
    shared_vlm_config: ModelRoleConfig | None = None,
    scheduler_config: ModelSchedulerConfig | None = None,
    updater_config: UpdaterRuntimeConfig | None = None,
) -> ModelRegistry:
    """Build model role adapters from starter YAML config."""

    shared_vlm_config = shared_vlm_config or ModelRoleConfig()
    scheduler = _shared_model_call_scheduler(scheduler_config)
    shared_ollama_client = _shared_ollama_client(shared_vlm_config)
    change_role_config = _with_shared_vlm_role_config(
        change_config,
        shared_vlm_config,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )
    historizer_role_config = _with_shared_vlm_role_config(
        historizer_config or ModelRoleConfig(),
        shared_vlm_config,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )
    memory_role_config = _with_shared_vlm_role_config(
        memory_config or ModelRoleConfig(),
        shared_vlm_config,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )
    return ModelRegistry(
        orchestrator_agent=_build_agent(
            _with_shared_vlm_role_config(
                agent_config,
                shared_vlm_config,
                scheduler=scheduler,
                scheduler_config=scheduler_config,
            ),
            ollama_client=shared_ollama_client,
        ),
        change_summary_model=_build_change_summary_model(
            change_role_config,
            ollama_client=shared_ollama_client,
        ),
        agent_context_historizer_model=_build_historizer_model(
            historizer_role_config,
            ollama_client=shared_ollama_client,
        ),
        game_memory_model=_build_game_memory_model(
            memory_role_config,
            ollama_client=shared_ollama_client,
        ),
        updater_tasks=_build_updater_tasks(
            _with_shared_vlm_updater_config(
                updater_config,
                shared_vlm_config,
                scheduler=scheduler,
                scheduler_config=scheduler_config,
            ),
            ollama_client=shared_ollama_client,
        ),
    )


def _build_change_summary_model(
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build the selected transition change summary adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError("models.change.backend is required")
    backend = config.backend.lower()
    if backend == "openai":
        _require_model("change", backend, config)
        return ChangeSummaryAdapter(
            OpenAIChangeSummaryConfig(
                **_config_kwargs(config, OpenAIChangeSummaryConfig)
            )
        )
    if backend == "ollama":
        _require_model("change", backend, config)
        return ChangeSummaryAdapter(
            OllamaChangeSummaryConfig(
                **_config_kwargs(config, OllamaChangeSummaryConfig)
            ),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_role_model("models.change", backend, config)
        return ChangeSummaryAdapter(
            VLLMChangeSummaryConfig(
                **_config_kwargs(config, VLLMChangeSummaryConfig)
            )
        )
    if backend in {"huggingface", "huggingface-diffusers", "diffusers"}:
        raise NotImplementedError(
            "Hugging Face change summary provider is not implemented yet"
        )
    if backend == "configurable":
        raise NotImplementedError(
            "Configurable change summary provider is not implemented yet"
        )
    raise ValueError(f"unknown change backend: {config.backend}")


def _build_historizer_model(
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build the selected agent context historizer adapter."""

    if config.backend is None or config.backend == "":
        return None
    backend = config.backend.lower()
    if backend == "openai":
        _require_model("historizer", backend, config)
        return OpenAIHistorizerAdapter(
            OpenAIHistorizerConfig(
                **_config_kwargs(config, OpenAIHistorizerConfig)
            )
        )
    if backend == "ollama":
        _require_model("historizer", backend, config)
        return OllamaHistorizerAdapter(
            OllamaHistorizerConfig(
                **_config_kwargs(config, OllamaHistorizerConfig)
            ),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_role_model("models.historizer", backend, config)
        return VLLMHistorizerAdapter(
            VLLMHistorizerConfig(
                **_config_kwargs(config, VLLMHistorizerConfig)
            )
        )
    if backend in {"huggingface", "huggingface-diffusers", "diffusers"}:
        raise NotImplementedError(
            "Hugging Face historizer provider is not implemented yet"
        )
    if backend == "configurable":
        raise NotImplementedError(
            "Configurable historizer provider is not implemented yet"
        )
    raise ValueError(f"unknown historizer backend: {config.backend}")


def _build_game_memory_model(
    config: ModelRoleConfig,
    *,
    ollama_client: object | None = None,
) -> object | None:
    """Build the selected game memory adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError("models.memory.backend is required")
    backend = config.backend.lower()
    if backend in {"none", "off", "disabled", "false"}:
        return DisabledGameMemoryAdapter()
    if backend == "openai":
        _require_model("memory", backend, config)
        return OpenAIGameMemoryAdapter(
            OpenAIGameMemoryConfig(
                **_config_kwargs(config, OpenAIGameMemoryConfig)
            )
        )
    if backend == "ollama":
        _require_model("memory", backend, config)
        return OllamaGameMemoryAdapter(
            OllamaGameMemoryConfig(
                **_config_kwargs(config, OllamaGameMemoryConfig)
            ),
            client=ollama_client,
        )
    if backend == "vllm":
        _require_role_model("models.memory", backend, config)
        return VLLMGameMemoryAdapter(
            VLLMGameMemoryConfig(
                **_config_kwargs(config, VLLMGameMemoryConfig)
            )
        )
    if backend in {"huggingface", "huggingface-diffusers", "diffusers"}:
        raise NotImplementedError(
            "Hugging Face game memory provider is not implemented yet"
        )
    if backend == "configurable":
        raise NotImplementedError(
            "Configurable game memory provider is not implemented yet"
        )
    raise ValueError(f"unknown memory backend: {config.backend}")


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


def _build_updater_tasks(
    config: UpdaterRuntimeConfig | None,
    *,
    ollama_client: object | None = None,
) -> UpdaterTaskRegistry:
    """Build configured updater P task adapters."""

    if config is None:
        raise ValueError("models.updater config is required")
    return UpdaterTaskRegistry(
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
        _require_model(f"updater.{task_name}", backend, config)
        return OpenAIUpdaterAdapter(
            OpenAIUpdaterConfig(**_config_kwargs(config, OpenAIUpdaterConfig))
        )
    if backend == "ollama":
        _require_prompt_updater_task(task_name, backend)
        _require_model(f"updater.{task_name}", backend, config)
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


def _require_updater_task_config(
    config: ModelRoleConfig | None,
    task_name: str,
) -> ModelRoleConfig:
    """Return an active updater task config, failing if it is missing."""

    if config is None:
        raise ValueError(f"models.updater.{task_name} config is required")
    return config


def _require_prompt_updater_task(task_name: str, backend: str) -> None:
    """Fail clearly for real updater slots that are not implemented yet."""

    if task_name not in {"agent", "general"}:
        raise NotImplementedError(
            f"{backend} updater is implemented only for agent and general "
            "prompt tasks"
        )


def _require_model(
    config_name: str,
    backend: str,
    config: ModelRoleConfig,
) -> None:
    """Require explicit model names for real model providers."""

    if not config.model:
        raise ValueError(
            f"models.{config_name}.model is required for backend {backend}"
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
    *,
    scheduler: ModelCallScheduler | None = None,
    scheduler_config: ModelSchedulerConfig | None = None,
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

    options = _deep_merge_dicts(
        shared_options,
        config.options,
    )
    if backend == "vllm" and scheduler is not None:
        options = _with_scheduler_options(
            options,
            role_options=config.options,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        )

    return ModelRoleConfig(
        backend=config.backend,
        model=config.model or shared.model,
        max_tool_calls=(
            config.max_tool_calls
            if config.max_tool_calls is not None
            else shared.max_tool_calls
        ),
        repair_attempts=(
            config.repair_attempts
            if config.repair_attempts is not None
            else shared.repair_attempts
        ),
        options=options,
    )


def _with_shared_vlm_updater_config(
    config: UpdaterRuntimeConfig | None,
    shared: ModelRoleConfig,
    *,
    scheduler: ModelCallScheduler | None = None,
    scheduler_config: ModelSchedulerConfig | None = None,
) -> UpdaterRuntimeConfig | None:
    """Apply shared local VLM defaults to matching updater task configs."""

    if config is None:
        return None
    return UpdaterRuntimeConfig(
        agent=_with_shared_vlm_role_config(
            config.agent,
            shared,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        ),
        general=_with_shared_vlm_role_config(
            config.general,
            shared,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        ),
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


def _shared_model_call_scheduler(
    config: ModelSchedulerConfig | None,
) -> ModelCallScheduler | None:
    """Return the process-wide scheduler for one scheduler config."""

    if config is None or not config.enabled:
        return None
    key = (config.max_concurrent_calls, config.max_concurrent_calls_per_game)
    with _SCHEDULERS_LOCK:
        scheduler = _SCHEDULERS.get(key)
        if scheduler is None:
            scheduler = ModelCallScheduler(
                max_concurrent_calls=config.max_concurrent_calls,
                max_concurrent_calls_per_game=(
                    config.max_concurrent_calls_per_game
                ),
            )
            _SCHEDULERS[key] = scheduler
        return scheduler


def _with_scheduler_options(
    options: dict[str, Any],
    *,
    role_options: dict[str, Any],
    scheduler: ModelCallScheduler,
    scheduler_config: ModelSchedulerConfig | None,
) -> dict[str, Any]:
    """Attach scheduler runtime controls to one vLLM role config."""

    merged = dict(options)
    merged["scheduler"] = scheduler
    if scheduler_config is not None:
        merged["scheduler_queue_timeout_seconds"] = (
            scheduler_config.queue_timeout_seconds
            if scheduler_config.queue_timeout_seconds is not None
            else _request_timeout_from_reasoning_budget(merged)
        )
    if "timeout" not in role_options:
        merged["timeout"] = _request_timeout_from_reasoning_budget(merged)
    return merged


def _request_timeout_from_reasoning_budget(options: dict[str, Any]) -> float:
    """Return clamp(90, 300, 60 + thinking_token_budget / 16)."""

    budget = _float_or_zero(options.get("thinking_token_budget"))
    return max(90.0, min(300.0, 60.0 + budget / 16.0))


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
    runtime_only_keys = {"scheduler", "scheduler_queue_timeout_seconds"}
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
        elif "options" in allowed and key not in runtime_only_keys:
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
    game_index = getattr(config, "game_index", None)
    if game_index is None:
        game_index = "unknown"
    return f"game-index-{game_index}-{timestamp}"


def _build_parallel_batch_run_id() -> str:
    """Create a batch run id shared by one parallel shell execution."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"parallel-{timestamp}"


def _resolve_selected_game_id(config: object) -> str:
    """Resolve the chosen game index from the stored local catalog file."""

    game_index = getattr(config, "game_index")
    if game_index is None:
        raise RuntimeError("single-game runtime config is missing game_index")
    game_catalog_path = getattr(config, "game_catalog_path")
    catalog = load_game_catalog(game_catalog_path)
    key = str(game_index)
    if key not in catalog:
        raise RuntimeError(
            f"game index {game_index} was not found in {game_catalog_path}; "
            "run --list-games first"
        )
    return catalog[key]


def _resolve_selected_game_ids(config: object) -> tuple[tuple[int, str], ...]:
    """Resolve all configured parallel game ids."""

    explicit_game_ids = tuple(getattr(config, "game_ids", ()) or ())
    if explicit_game_ids:
        return tuple(enumerate(explicit_game_ids))

    if getattr(config, "game_selection", None) == "all_available":
        return _resolve_catalog_game_ids(
            getattr(config, "game_catalog_path"),
        )

    game_indices = tuple(getattr(config, "game_indices", ()) or ())
    if not game_indices:
        game_index = getattr(config, "game_index", None)
        if game_index is None:
            raise RuntimeError(
                "runtime config is missing game_index, game_indices, "
                "game_ids, or game_selection"
            )
        game_indices = (int(game_index),)

    game_catalog_path = getattr(config, "game_catalog_path")
    catalog = load_game_catalog(game_catalog_path)
    resolved: list[tuple[int, str]] = []
    missing: list[int] = []
    for game_index in game_indices:
        key = str(game_index)
        if key not in catalog:
            missing.append(game_index)
            continue
        resolved.append((game_index, catalog[key]))
    if missing:
        missing_text = ", ".join(str(index) for index in missing)
        raise RuntimeError(
            f"game indices not found in {game_catalog_path}: {missing_text}; "
            "run --list-games first"
        )
    return tuple(resolved)


def _resolve_catalog_game_ids(game_catalog_path: str) -> tuple[tuple[int, str], ...]:
    """Resolve every game id in the local catalog in numeric index order."""

    catalog = load_game_catalog(game_catalog_path)
    resolved: list[tuple[int, str]] = []
    invalid_keys: list[str] = []
    for key, game_id in catalog.items():
        try:
            game_index = int(key)
        except ValueError:
            invalid_keys.append(key)
            continue
        if game_index < 0:
            invalid_keys.append(key)
            continue
        resolved.append((game_index, game_id))

    if invalid_keys:
        invalid_text = ", ".join(sorted(invalid_keys))
        raise RuntimeError(
            f"game catalog contains invalid indices in {game_catalog_path}: "
            f"{invalid_text}"
        )
    if not resolved:
        raise RuntimeError(f"game catalog is empty: {game_catalog_path}")
    return tuple(sorted(resolved, key=lambda item: item[0]))


def _database_path_for_game(base_path: Path, game_index: int) -> Path:
    """Return the per-game SQLite path derived from one CLI database path."""

    return base_path.with_name(
        f"{base_path.stem}-game-index-{game_index}{base_path.suffix}"
    )


if __name__ == "__main__":
    main()
