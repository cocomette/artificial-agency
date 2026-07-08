"""Runnable starter shell for the ARC-AGI environment loop."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
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
    ModelRoleConfig,
    load_game_catalog,
    write_game_catalog,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import (
    ChangeSummaryAdapter,
    HFChangeSummaryConfig,
    HFGoalAdapter,
    HFGoalConfig,
    HFInterestAdapter,
    HFInterestConfig,
    HFMemoryAdapter,
    HFMemoryConfig,
    HFOrchestratorAgentAdapter,
    HFOrchestratorAgentConfig,
    HFRewardJudgeAdapter,
    HFRewardJudgeConfig,
    HFWorldAdapter,
    HFWorldConfig,
    ModelRegistry,
    VLLMChangeSummaryConfig,
    VLLMGoalAdapter,
    VLLMGoalConfig,
    VLLMInterestAdapter,
    VLLMInterestConfig,
    VLLMMemoryAdapter,
    VLLMMemoryConfig,
    VLLMOrchestratorAgentConfig,
    VLLMRewardJudgeAdapter,
    VLLMRewardJudgeConfig,
    VLLMWorldAdapter,
    VLLMWorldConfig,
)
from face_of_agi.models.orchestrator_agent.providers import VLLMOrchestratorAgentAdapter
from face_of_agi.models.providers.hf_transformers import (
    HFVLMEngine,
    shared_hf_vlm_engine,
)
from face_of_agi.orchestration import Orchestrator
from face_of_agi.orchestration.online_lora import OnlineLoRAManager
from face_of_agi.runtime.loop import RuntimeLoop
from face_of_agi.runtime.parallel import ParallelGameRunSpec, ParallelRuntimeLoop

DEFAULT_DATABASE_PATH = Path("runs/memory.sqlite")


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
        memory_config=environment_config.models.memory,
        world_config=environment_config.models.world,
        goal_config=environment_config.models.goal,
        interest_config=environment_config.models.interest,
        reward_judge_config=environment_config.models.reward_judge,
        shared_vlm_config=environment_config.models.shared_vlm,
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

    online_lora_manager = _build_online_lora_manager(
        environment_config,
        hf_engine=_hf_engine_from_registry(model_registry),
    )
    try:
        runtime = RuntimeLoop(
            _build_orchestrator(
                database_path,
                experimental_memory_turn_buffer=(
                    environment_config.experimental_memory_turn_buffer
                ),
                models=model_registry,
                contexts=contexts,
                online_lora_manager=online_lora_manager,
            )
        )
        result = runtime.run(
            config=runtime_config,
            environment=environment,
            environment_config=environment_config,
        )
    except Exception as exc:
        print(f"starter shell failed: {exc}")
        raise SystemExit(1) from exc
    finally:
        if online_lora_manager is not None:
            online_lora_manager.shutdown()

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
    online_lora_manager = _build_online_lora_manager(
        base_environment_config,
        hf_engine=_build_hf_engine_for_environment_config(base_environment_config),
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
            online_lora_manager=online_lora_manager,
        )
        for game_index, game_id in selected_games
    )
    try:
        return ParallelRuntimeLoop(
            _run_parallel_game,
            trace_output=trace_output,
        ).run(
            batch_run_id=batch_run_id,
            specs=specs,
            max_parallel_games=base_environment_config.max_parallel_games,
            max_game_retries=base_environment_config.max_game_retries,
        )
    finally:
        if online_lora_manager is not None:
            online_lora_manager.shutdown()


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
            online_lora_manager=spec.online_lora_manager,
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
    memory_config: ModelRoleConfig | None = None,
    world_config: ModelRoleConfig | None = None,
    goal_config: ModelRoleConfig | None = None,
    interest_config: ModelRoleConfig | None = None,
    reward_judge_config: ModelRoleConfig | None = None,
    shared_vlm_config: ModelRoleConfig | None = None,
    contexts: ContextDocuments | None = None,
    models: ModelRegistry | None = None,
    online_lora_manager: OnlineLoRAManager | None = None,
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
            memory_config=memory_config or ModelRoleConfig(),
            world_config=world_config or ModelRoleConfig(),
            goal_config=goal_config or ModelRoleConfig(),
            interest_config=interest_config or ModelRoleConfig(),
            reward_judge_config=reward_judge_config or ModelRoleConfig(),
            shared_vlm_config=shared_vlm_config or ModelRoleConfig(),
        ),
        contexts=contexts,
        experimental_memory_turn_buffer=experimental_memory_turn_buffer,
        online_lora_manager=online_lora_manager,
    )


def _build_online_lora_manager(
    environment_config: Any,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> OnlineLoRAManager | None:
    if not environment_config.online_lora.enabled:
        return None
    return OnlineLoRAManager(
        config=environment_config.online_lora,
        vllm_base_url=_vllm_base_url(environment_config),
        hf_engine=hf_engine,
    )


def _vllm_base_url(environment_config: Any) -> str:
    for role in (
        environment_config.models.shared_vlm,
        environment_config.models.world,
        environment_config.models.interest,
        environment_config.models.agent,
    ):
        base_url = role.options.get("base_url")
        if base_url:
            return str(base_url).rstrip("/")
    return "http://127.0.0.1:8000/v1"


def _build_model_registry(
    *,
    agent_config: ModelRoleConfig,
    change_config: ModelRoleConfig,
    memory_config: ModelRoleConfig,
    world_config: ModelRoleConfig,
    goal_config: ModelRoleConfig,
    interest_config: ModelRoleConfig,
    reward_judge_config: ModelRoleConfig,
    shared_vlm_config: ModelRoleConfig | None = None,
) -> ModelRegistry:
    """Build model role adapters from starter YAML config."""

    shared_vlm_config = shared_vlm_config or ModelRoleConfig()
    role_configs = {
        "agent": _with_shared_vlm_role_config(agent_config, shared_vlm_config),
        "change": _with_shared_vlm_role_config(change_config, shared_vlm_config),
        "memory": _with_shared_vlm_role_config(memory_config, shared_vlm_config),
        "world": _with_shared_vlm_role_config(world_config, shared_vlm_config),
        "goal": _with_shared_vlm_role_config(goal_config, shared_vlm_config),
        "interest": _with_shared_vlm_role_config(interest_config, shared_vlm_config),
        "reward_judge": _with_shared_vlm_role_config(
            reward_judge_config,
            shared_vlm_config,
        ),
    }
    hf_engine = _build_hf_engine_for_roles(role_configs)
    return ModelRegistry(
        orchestrator_agent=_build_agent(
            role_configs["agent"],
            hf_engine=hf_engine,
        ),
        change_summary_model=_build_change_summary_model(
            role_configs["change"],
            hf_engine=hf_engine,
        ),
        memory_model=_build_memory_model(
            role_configs["memory"],
            hf_engine=hf_engine,
        ),
        world_model=_build_world_model(
            role_configs["world"],
            hf_engine=hf_engine,
        ),
        goal_model=_build_goal_model(
            role_configs["goal"],
            hf_engine=hf_engine,
        ),
        interest_model=_build_interest_model(
            role_configs["interest"],
            hf_engine=hf_engine,
        ),
        reward_judge_model=_build_reward_judge_model(
            role_configs["reward_judge"],
            hf_engine=hf_engine,
        ),
    )


def _build_change_summary_model(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object | None:
    """Build the transition change summary adapter."""

    backend = _backend_name(config)
    if backend == "hf_transformers":
        from face_of_agi.models.change.providers.hf_transformers import (
            HFChangeSummaryProvider,
        )

        hf_config = HFChangeSummaryConfig(
            **_config_kwargs(config, HFChangeSummaryConfig)
        )
        return ChangeSummaryAdapter(
            hf_config,
            provider=HFChangeSummaryProvider(hf_config, engine=hf_engine),
        )
    _require_vllm_role("models.change", config)
    return ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(**_config_kwargs(config, VLLMChangeSummaryConfig))
    )


def _build_memory_model(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object:
    """Build the Memory role adapter."""

    backend = _backend_name(config)
    if backend == "hf_transformers":
        return HFMemoryAdapter(
            HFMemoryConfig(**_config_kwargs(config, HFMemoryConfig)),
            engine=hf_engine,
        )
    _require_vllm_role("models.memory", config)
    return VLLMMemoryAdapter(
        VLLMMemoryConfig(**_config_kwargs(config, VLLMMemoryConfig))
    )


def _build_world_model(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object:
    """Build the World role adapter."""

    backend = _backend_name(config)
    if backend == "hf_transformers":
        return HFWorldAdapter(
            HFWorldConfig(**_config_kwargs(config, HFWorldConfig)),
            engine=hf_engine,
        )
    _require_vllm_role("models.world", config)
    return VLLMWorldAdapter(
        VLLMWorldConfig(**_config_kwargs(config, VLLMWorldConfig))
    )


def _build_goal_model(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object:
    """Build the Goal role adapter."""

    backend = _backend_name(config)
    if backend == "hf_transformers":
        return HFGoalAdapter(
            HFGoalConfig(**_config_kwargs(config, HFGoalConfig)),
            engine=hf_engine,
        )
    _require_vllm_role("models.goal", config)
    return VLLMGoalAdapter(
        VLLMGoalConfig(**_config_kwargs(config, VLLMGoalConfig))
    )


def _build_interest_model(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object:
    """Build the Interest role adapter."""

    backend = _backend_name(config)
    if backend == "hf_transformers":
        return HFInterestAdapter(
            HFInterestConfig(**_config_kwargs(config, HFInterestConfig)),
            engine=hf_engine,
        )
    _require_vllm_role("models.interest", config)
    return VLLMInterestAdapter(
        VLLMInterestConfig(**_config_kwargs(config, VLLMInterestConfig))
    )


def _build_reward_judge_model(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object:
    """Build the Reward Judge role adapter."""

    backend = _backend_name(config)
    if backend == "hf_transformers":
        return HFRewardJudgeAdapter(
            HFRewardJudgeConfig(**_config_kwargs(config, HFRewardJudgeConfig)),
            engine=hf_engine,
        )
    _require_vllm_role("models.reward_judge", config)
    return VLLMRewardJudgeAdapter(
        VLLMRewardJudgeConfig(**_config_kwargs(config, VLLMRewardJudgeConfig))
    )


def _build_agent(
    config: ModelRoleConfig,
    *,
    hf_engine: HFVLMEngine | None = None,
) -> object | None:
    """Build the selected X agent adapter."""

    if config.backend is None or config.backend == "":
        raise ValueError("models.agent.backend is required")
    backend = _backend_name(config)
    if backend == "vllm":
        _require_role_model("models.agent", backend, config)
        return VLLMOrchestratorAgentAdapter(
            VLLMOrchestratorAgentConfig(
                **_config_kwargs(config, VLLMOrchestratorAgentConfig)
            )
        )
    if backend == "hf_transformers":
        _require_hf_role_model("models.agent", config)
        return HFOrchestratorAgentAdapter(
            HFOrchestratorAgentConfig(
                **_config_kwargs(config, HFOrchestratorAgentConfig)
            ),
            engine=hf_engine,
        )
    raise ValueError("models.agent.backend must be vllm or hf_transformers")


def _require_role_model(role_path: str, backend: str, config: ModelRoleConfig) -> None:
    """Require explicit model names for real model providers without defaults."""

    if not config.model:
        raise ValueError(f"{role_path}.model is required for backend {backend}")


def _require_hf_role_model(role_path: str, config: ModelRoleConfig) -> None:
    """Require an explicit HF model id or model_path."""

    if config.model or config.options.get("model_path"):
        return
    raise ValueError(
        f"{role_path}.model or {role_path}.model_path is required for "
        "backend hf_transformers"
    )


def _require_vllm_role(role_path: str, config: ModelRoleConfig) -> None:
    """Require vLLM for a new v1 role."""

    backend = (config.backend or "").lower()
    if backend != "vllm":
        raise ValueError(f"{role_path}.backend must be vllm")
    _require_role_model(role_path, backend, config)


def _backend_name(config: ModelRoleConfig) -> str:
    return (config.backend or "").lower()


def _with_shared_vlm_role_config(
    config: ModelRoleConfig,
    shared: ModelRoleConfig,
) -> ModelRoleConfig:
    """Apply shared VLM defaults to matching role configs."""

    backend = (config.backend or "").lower()
    if backend not in {"vllm", "hf_transformers"}:
        return config
    if backend != (shared.backend or "").lower():
        return config

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
        options=_deep_merge_dicts(
            _shared_vlm_runtime_options(shared),
            config.options,
        ),
    )


def _shared_vlm_runtime_options(config: ModelRoleConfig) -> dict[str, Any]:
    """Return shared VLM behavior options without changing role prompts."""

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


def _build_hf_engine_for_roles(
    role_configs: dict[str, ModelRoleConfig],
) -> HFVLMEngine | None:
    """Return the shared HF engine when any active v1 role uses HF."""

    hf_configs = [
        config
        for config in role_configs.values()
        if _backend_name(config) == "hf_transformers"
    ]
    if not hf_configs:
        return None
    first = hf_configs[0]
    _require_hf_role_model("models.shared_vlm", first)
    hf_config = HFWorldConfig(**_config_kwargs(first, HFWorldConfig))
    return shared_hf_vlm_engine(hf_config)


def _build_hf_engine_for_environment_config(environment_config: Any) -> HFVLMEngine | None:
    """Build/reuse the shared HF engine from an environment config."""

    role_configs = {
        "agent": _with_shared_vlm_role_config(
            environment_config.models.agent,
            environment_config.models.shared_vlm,
        ),
        "change": _with_shared_vlm_role_config(
            environment_config.models.change,
            environment_config.models.shared_vlm,
        ),
        "memory": _with_shared_vlm_role_config(
            environment_config.models.memory,
            environment_config.models.shared_vlm,
        ),
        "world": _with_shared_vlm_role_config(
            environment_config.models.world,
            environment_config.models.shared_vlm,
        ),
        "goal": _with_shared_vlm_role_config(
            environment_config.models.goal,
            environment_config.models.shared_vlm,
        ),
        "interest": _with_shared_vlm_role_config(
            environment_config.models.interest,
            environment_config.models.shared_vlm,
        ),
        "reward_judge": _with_shared_vlm_role_config(
            environment_config.models.reward_judge,
            environment_config.models.shared_vlm,
        ),
    }
    return _build_hf_engine_for_roles(role_configs)


def _hf_engine_from_registry(registry: ModelRegistry) -> HFVLMEngine | None:
    """Return the HF engine captured by any HF-backed role in a registry."""

    for model in (
        registry.world_model,
        registry.interest_model,
        registry.orchestrator_agent,
        registry.memory_model,
        registry.goal_model,
        registry.change_summary_model,
        registry.reward_judge_model,
    ):
        provider = getattr(model, "provider", None)
        client = getattr(provider, "_client", None)
        engine = getattr(client, "engine", None)
        if isinstance(engine, HFVLMEngine):
            return engine
    return None


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
