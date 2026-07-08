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
GameSelectionMode = Literal["all_available"]
BackboneModelFamily = Literal["generic_vision", "qwen3_5_moe_multimodal"]
DEFAULT_BACKBONE_FEATURE_PROMPT = (
    "Encode this ARC-AGI game frame as a compact visual state representation."
)


@dataclass(slots=True)
class BackboneRuntimeConfig:
    """Frozen local Transformers backbone configuration."""

    backend: str = "transformers"
    model_family: BackboneModelFamily = "generic_vision"
    model_path: str = ""
    processor_path: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    image_size: str | None = "224x224"
    local_files_only: bool = True
    representation_layer: str = "pooled"
    feature_prompt: str = DEFAULT_BACKBONE_FEATURE_PROMPT
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OnlineRuntimeConfig:
    """Small online learner component configuration."""

    buffer_size: int = 512
    adapter_rank: int = 16
    ensemble_size: int = 5
    hidden_dim: int = 512
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass(slots=True)
class ReplayRuntimeConfig:
    """Bounded replay configuration."""

    max_updates_per_turn: int = 8
    max_seconds_per_turn: float = 0.5
    solved_level_updates: int = 32


@dataclass(slots=True)
class PlannerRuntimeConfig:
    """Short-horizon planner configuration."""

    horizon: int = 3
    candidate_count: int = 64
    coordinate_candidates: int = 16
    diagnostic_turns: int = 4


@dataclass(slots=True)
class AgentRuntimeConfig:
    """Runtime configuration for the online learner agent."""

    backbone: BackboneRuntimeConfig = field(default_factory=BackboneRuntimeConfig)
    online: OnlineRuntimeConfig = field(default_factory=OnlineRuntimeConfig)
    replay: ReplayRuntimeConfig = field(default_factory=ReplayRuntimeConfig)
    planner: PlannerRuntimeConfig = field(default_factory=PlannerRuntimeConfig)


@dataclass(slots=True)
class EnvironmentConfig:
    """Minimal environment config for the starter runtime shell."""

    game_index: int | None = None
    max_actions_per_level: int = 0
    max_levels_per_game: int | None = None
    game_indices: tuple[int, ...] = ()
    game_ids: tuple[str, ...] = ()
    game_selection: GameSelectionMode | None = None
    max_parallel_games: int | None = None
    max_game_retries: int = 0
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
    animation_keyframe_pixel_threshold: int = 8
    debug_keep_all_m_states: bool = False
    debug_trace: DebugTraceMode = "minimal"
    debug_color: DebugColorMode = "auto"
    live_turn_monitor: bool = False
    agent: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """Load the starter environment config from YAML."""

    raw_data = _read_yaml(path)
    game_index, game_indices, game_ids, game_selection = _load_game_selection(raw_data)
    game_id = _optional_string(raw_data.get("game_id"))
    if (game_indices or game_ids or game_selection is not None) and game_id is not None:
        raise ValueError(
            "game_id cannot be set when game_indices, game_ids, or "
            "game_selection is configured"
        )
    max_actions_per_level = int(raw_data["max_actions_per_level"])
    operation_mode = OperationMode(str(raw_data.get("operation_mode", "offline")))
    return EnvironmentConfig(
        max_actions_per_level=max_actions_per_level,
        max_levels_per_game=_optional_positive_int(
            raw_data.get("max_levels_per_game"),
            key="max_levels_per_game",
        ),
        game_index=game_index,
        game_indices=game_indices,
        game_ids=game_ids,
        game_selection=game_selection,
        max_parallel_games=_optional_positive_int(
            raw_data.get("max_parallel_games"),
            key="max_parallel_games",
        ),
        max_game_retries=_non_negative_int(
            raw_data.get("max_game_retries", 0),
            key="max_game_retries",
        ),
        game_id=game_id,
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
        animation_keyframe_pixel_threshold=_non_negative_int(
            raw_data.get("animation_keyframe_pixel_threshold", 8),
            key="animation_keyframe_pixel_threshold",
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
        live_turn_monitor=_optional_bool(
            raw_data.get("live_turn_monitor"),
            default=False,
        ),
        agent=_load_agent_runtime_config(raw_data.get("agent")),
    )


def _load_agent_runtime_config(value: Any) -> AgentRuntimeConfig:
    """Load the required online learner config from YAML."""

    if value is None:
        raise ValueError("agent config is required")
    if not isinstance(value, dict):
        raise ValueError("agent config must be a mapping")
    return AgentRuntimeConfig(
        backbone=_load_backbone_runtime_config(value.get("backbone")),
        online=_load_online_runtime_config(value.get("online")),
        replay=_load_replay_runtime_config(value.get("replay")),
        planner=_load_planner_runtime_config(value.get("planner")),
    )


def _load_backbone_runtime_config(value: Any) -> BackboneRuntimeConfig:
    if not isinstance(value, dict):
        raise ValueError("agent.backbone config is required")
    backend = str(value.get("backend") or "transformers")
    if backend != "transformers":
        raise ValueError("agent.backbone.backend must be transformers")
    model_path = _optional_string(value.get("model_path")) or ""
    if not model_path:
        raise ValueError("agent.backbone.model_path is required")
    known = {
        "backend",
        "model_family",
        "model_path",
        "processor_path",
        "device",
        "dtype",
        "image_size",
        "local_files_only",
        "representation_layer",
        "feature_prompt",
        "processor_kwargs",
        "model_kwargs",
        "options",
    }
    return BackboneRuntimeConfig(
        backend=backend,
        model_family=_choice(
            value.get("model_family"),
            key="agent.backbone.model_family",
            default="generic_vision",
            allowed=("generic_vision", "qwen3_5_moe_multimodal"),
        ),
        model_path=model_path,
        processor_path=_optional_string(value.get("processor_path")),
        device=str(value.get("device") or "auto"),
        dtype=str(value.get("dtype") or "auto"),
        image_size=_optional_string(value.get("image_size")) or "224x224",
        local_files_only=_optional_bool(
            value.get("local_files_only"),
            default=True,
        ),
        representation_layer=str(value.get("representation_layer") or "pooled"),
        feature_prompt=str(
            value.get("feature_prompt") or DEFAULT_BACKBONE_FEATURE_PROMPT
        ),
        processor_kwargs=_optional_mapping(
            value.get("processor_kwargs"),
            "agent.backbone.processor_kwargs",
        ),
        model_kwargs=_optional_mapping(
            value.get("model_kwargs"),
            "agent.backbone.model_kwargs",
        ),
        options=_options(value, known),
    )


def _load_online_runtime_config(value: Any) -> OnlineRuntimeConfig:
    data = _optional_mapping(value, "agent.online")
    return OnlineRuntimeConfig(
        buffer_size=_positive_int(data.get("buffer_size", 512), key="buffer_size"),
        adapter_rank=_positive_int(data.get("adapter_rank", 16), key="adapter_rank"),
        ensemble_size=_positive_int(data.get("ensemble_size", 5), key="ensemble_size"),
        hidden_dim=_positive_int(data.get("hidden_dim", 512), key="hidden_dim"),
        learning_rate=_positive_float(
            data.get("learning_rate", 0.001),
            key="learning_rate",
        ),
        batch_size=_positive_int(data.get("batch_size", 32), key="batch_size"),
    )


def _load_replay_runtime_config(value: Any) -> ReplayRuntimeConfig:
    data = _optional_mapping(value, "agent.replay")
    return ReplayRuntimeConfig(
        max_updates_per_turn=_non_negative_int(
            data.get("max_updates_per_turn", 8),
            key="max_updates_per_turn",
        ),
        max_seconds_per_turn=_non_negative_float(
            data.get("max_seconds_per_turn", 0.5),
            key="max_seconds_per_turn",
        ),
        solved_level_updates=_non_negative_int(
            data.get("solved_level_updates", 32),
            key="solved_level_updates",
        ),
    )


def _load_planner_runtime_config(value: Any) -> PlannerRuntimeConfig:
    data = _optional_mapping(value, "agent.planner")
    return PlannerRuntimeConfig(
        horizon=_positive_int(data.get("horizon", 3), key="horizon"),
        candidate_count=_positive_int(
            data.get("candidate_count", 64),
            key="candidate_count",
        ),
        coordinate_candidates=_positive_int(
            data.get("coordinate_candidates", 16),
            key="coordinate_candidates",
        ),
        diagnostic_turns=_non_negative_int(
            data.get("diagnostic_turns", 4),
            key="diagnostic_turns",
        ),
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"environment config must be a mapping: {config_path}")

    missing_keys = {"max_actions_per_level"} - loaded.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"environment config is missing required keys: {missing}")

    if "max_actions_per_game" in loaded:
        raise ValueError(
            "max_actions_per_game has been removed; use max_actions_per_level "
            "for the per-level action budget"
        )
    if "models" in loaded:
        raise ValueError(
            "models config has been removed; configure the online learner under "
            "agent"
        )
    return loaded


def _load_game_selection(
    value: dict[str, Any],
) -> tuple[int | None, tuple[int, ...], tuple[str, ...], GameSelectionMode | None]:
    """Return exactly one configured runtime game selector."""

    has_single = value.get("game_index") is not None
    game_indices = _optional_game_indices(value.get("game_indices"))
    game_ids = _optional_game_ids(value.get("game_ids"))
    game_selection = _optional_game_selection(value.get("game_selection"))
    selected_count = sum(
        (
            1 if has_single else 0,
            1 if game_indices else 0,
            1 if game_ids else 0,
            1 if game_selection is not None else 0,
        )
    )
    if selected_count > 1:
        raise ValueError(
            "configure exactly one of game_index, game_indices, game_ids, "
            "or game_selection"
        )
    if selected_count == 0:
        raise ValueError(
            "environment config requires game_index, game_indices, game_ids, "
            "or game_selection"
        )
    return (
        _non_negative_int(value["game_index"], key="game_index")
        if has_single
        else None,
        game_indices,
        game_ids,
        game_selection,
    )


def _optional_game_indices(value: Any) -> tuple[int, ...]:
    """Normalize an optional list of selected game catalog indices."""

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("game_indices must be a list of non-negative integers")
    if not value:
        raise ValueError("game_indices must not be empty")

    indices = tuple(_non_negative_int(item, key="game_indices") for item in value)
    if len(set(indices)) != len(indices):
        raise ValueError("game_indices must not contain duplicates")
    return indices


def _optional_game_ids(value: Any) -> tuple[str, ...]:
    """Normalize an optional list of explicit ARC game ids."""

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("game_ids must be a list of non-empty strings")
    if not value:
        raise ValueError("game_ids must not be empty")

    game_ids = tuple(str(item).strip() for item in value)
    if any(not game_id for game_id in game_ids):
        raise ValueError("game_ids must contain non-empty strings")
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("game_ids must not contain duplicates")
    return game_ids


def _optional_game_selection(value: Any) -> GameSelectionMode | None:
    """Normalize the optional named runtime game selector."""

    if value is None:
        return None
    parsed = str(value).strip()
    if parsed != "all_available":
        raise ValueError("game_selection must be all_available")
    return "all_available"


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


def _positive_int(value: Any, *, key: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{key} must be at least 1")
    return parsed


def _non_negative_float(value: Any, *, key: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _positive_float(value: Any, *, key: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def _optional_positive_int(value: Any, *, key: str) -> int | None:
    """Normalize one optional positive integer config value."""

    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{key} must be at least 1")
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


def _optional_mapping(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _options(value: dict[str, Any], known: set[str]) -> dict[str, Any]:
    options = dict(value.get("options") or {})
    for key, item in value.items():
        if key not in known:
            options[key] = item
    return options


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
