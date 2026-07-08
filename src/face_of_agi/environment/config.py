"""Environment-local configuration for the starter ARC shell."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from arc_agi import OperationMode

DebugTraceMode = Literal[
    "off",
    "minimal",
    "agent_decision",
    "verbose",
    "model_inputs",
]
DebugColorMode = Literal["auto", "always", "never"]


@dataclass(slots=True)
class ModelRoleConfig:
    """Runtime-selected model role backend configuration."""

    backend: str | None = None
    model: str | None = None
    max_tool_calls: int | None = None
    repair_attempts: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelRuntimeConfig:
    """Runtime model backend config for agent and tool roles."""

    shared_vlm: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    agent: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    world: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    goal: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    updater: "UpdaterRuntimeConfig | None" = None


@dataclass(slots=True)
class UpdaterRuntimeConfig:
    """Runtime backend config for updater task slots."""

    world: ModelRoleConfig
    goal: ModelRoleConfig
    agent: ModelRoleConfig
    general: ModelRoleConfig


@dataclass(slots=True)
class EnvironmentConfig:
    """Minimal environment config for the starter runtime shell."""

    game_index: int
    max_actions_per_level: int
    game_id: str | None = None
    operation_mode: OperationMode = OperationMode.OFFLINE
    game_catalog_path: str = "src/face_of_agi/environment/local_games.json"
    environments_dir: str = "environment_files"
    recordings_dir: str = "recordings"
    enable_visualization: bool = False
    render_mode: str | None = None
    seed: int = 0
    save_recording: bool = False
    use_learned_contexts: bool = True
    experimental_memory_turn_buffer: int = 2
    action_history_window: int = 8
    debug_keep_all_m_states: bool = False
    debug_trace: DebugTraceMode = "minimal"
    debug_color: DebugColorMode = "auto"
    models: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """Load the starter environment config from YAML."""

    raw_data = _read_yaml(path)
    game_index = int(raw_data["game_index"])
    max_actions_per_level = int(raw_data["max_actions_per_level"])
    operation_mode = OperationMode(str(raw_data.get("operation_mode", "offline")))
    return EnvironmentConfig(
        game_index=game_index,
        max_actions_per_level=max_actions_per_level,
        game_id=_optional_string(raw_data.get("game_id")),
        operation_mode=operation_mode,
        game_catalog_path=str(
            raw_data.get(
                "game_catalog_path",
                "src/face_of_agi/environment/local_games.json",
            )
        ),
        environments_dir=str(raw_data.get("environments_dir", "environment_files")),
        recordings_dir=str(raw_data.get("recordings_dir", "recordings")),
        enable_visualization=bool(raw_data.get("enable_visualization", False)),
        render_mode=_optional_string(raw_data.get("render_mode")),
        seed=int(raw_data.get("seed", 0)),
        save_recording=bool(raw_data.get("save_recording", False)),
        use_learned_contexts=_optional_bool(
            raw_data.get("use_learned_contexts"),
            default=True,
        ),
        experimental_memory_turn_buffer=int(
            raw_data.get("experimental_memory_turn_buffer", 2)
        ),
        action_history_window=_non_negative_int(
            raw_data.get("action_history_window", 8),
            key="action_history_window",
        ),
        debug_keep_all_m_states=_optional_bool(
            raw_data.get("debug_keep_all_m_states"),
            default=False,
        ),
        debug_trace=_choice(
            raw_data.get("debug_trace"),
            key="debug_trace",
            default="minimal",
            allowed=("off", "minimal", "agent_decision", "verbose", "model_inputs"),
        ),
        debug_color=_choice(
            raw_data.get("debug_color"),
            key="debug_color",
            default="auto",
            allowed=("auto", "always", "never"),
        ),
        models=_load_model_runtime_config(raw_data.get("models")),
    )


def _load_model_runtime_config(value: Any) -> ModelRuntimeConfig:
    """Load optional model role backend config from YAML."""

    if value is None:
        raise ValueError("models.updater config is required")
    if not isinstance(value, dict):
        raise ValueError("models config must be a mapping")

    return ModelRuntimeConfig(
        shared_vlm=_load_model_role_config(value.get("shared_vlm")),
        agent=_load_model_role_config(value.get("agent")),
        world=_load_model_role_config(value.get("world")),
        goal=_load_model_role_config(value.get("goal")),
        updater=_load_updater_runtime_config(value.get("updater")),
    )


def _load_updater_runtime_config(value: Any) -> UpdaterRuntimeConfig:
    """Load updater task backend configs from YAML."""

    if value is None:
        raise ValueError("models.updater config is required")
    if not isinstance(value, dict):
        raise ValueError("updater config must be a mapping")

    return UpdaterRuntimeConfig(
        world=_load_required_updater_task_config(value, "world"),
        goal=_load_required_updater_task_config(value, "goal"),
        agent=_load_required_updater_task_config(value, "agent"),
        general=_load_required_updater_task_config(value, "general"),
    )


def _load_required_updater_task_config(
    value: dict[str, Any],
    task_name: str,
) -> ModelRoleConfig:
    """Load one required updater task config."""

    if task_name not in value:
        raise ValueError(f"models.updater.{task_name} config is required")
    config = _load_model_role_config(value[task_name])
    if config.backend is None or config.backend == "":
        raise ValueError(f"models.updater.{task_name}.backend is required")
    return config


def _load_model_role_config(value: Any) -> ModelRoleConfig:
    """Load one role config from a YAML mapping."""

    if value is None:
        return ModelRoleConfig()
    if not isinstance(value, dict):
        raise ValueError("model role config must be a mapping")

    known_keys = {"backend", "model", "max_tool_calls", "repair_attempts"}
    options = dict(value.get("options") or {})
    for key, item in value.items():
        if key not in known_keys and key != "options":
            options[key] = item

    return ModelRoleConfig(
        backend=_optional_string(value.get("backend")),
        model=_optional_string(value.get("model")),
        max_tool_calls=(
            int(value["max_tool_calls"])
            if value.get("max_tool_calls") is not None
            else None
        ),
        repair_attempts=(
            int(value["repair_attempts"])
            if value.get("repair_attempts") is not None
            else None
        ),
        options=options,
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"environment config must be a mapping: {config_path}")

    missing_keys = {"game_index", "max_actions_per_level"} - loaded.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"environment config is missing required keys: {missing}")

    return loaded


def _optional_string(value: Any) -> str | None:
    """Normalize optional scalar config values."""

    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any, *, default: bool = False) -> bool:
    """Normalize optional boolean config values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"expected boolean config value, got {value!r}")


def _non_negative_int(value: Any, *, key: str) -> int:
    """Normalize one non-negative integer config value."""

    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _choice(
    value: Any,
    *,
    key: str,
    default: str,
    allowed: tuple[str, ...],
) -> Any:
    """Normalize one string enum config value."""

    if value is None:
        parsed = default
    elif isinstance(value, bool) and "off" in allowed:
        parsed = "off" if value is False else "on"
    else:
        parsed = str(value).strip()
    if parsed not in allowed:
        options = ", ".join(allowed)
        raise ValueError(f"{key} must be one of: {options}")
    return parsed


def load_game_catalog(path: str | Path) -> dict[str, str]:
    """Load the locally stored game catalog written by `--list-games`."""

    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"game catalog must be a JSON object: {catalog_path}")

    return {str(key): str(value) for key, value in loaded.items()}


def write_game_catalog(path: str | Path, games: dict[str, str]) -> None:
    """Write the indexed game catalog to a local JSON file."""

    catalog_path = Path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8") as handle:
        json.dump(games, handle, indent=2, sort_keys=True)
        handle.write("\n")
