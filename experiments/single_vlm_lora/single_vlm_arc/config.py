"""Configuration for the isolated single-VLM LoRA experiment."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


ModelBackend = Literal["hf", "fake"]
ActionSelectionMode = Literal["sample", "argmax"]
HiddenPoolingMode = Literal["last", "attention"]
PolicyAdvantageBaselineMode = Literal["ema", "zero"]
PolicyAdvantageNormalizationMode = Literal["none", "ema_abs"]
WorldLossMode = Literal["pixel_ce", "latent_grid", "hybrid"]


@dataclass(slots=True)
class LoRAConfig:
    enabled: bool = True
    separate_role_adapters: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    save_every: int = 10


@dataclass(slots=True)
class ModelConfig:
    backend: ModelBackend = "hf"
    model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    processor_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    image_size: tuple[int, int] = (224, 224)
    action_selection: ActionSelectionMode = "sample"
    temperature: float = 1.0
    hidden_size: int = 128
    hidden_pooling: HiddenPoolingMode = "last"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(slots=True)
class EnvironmentExperimentConfig:
    game_id: str | None = "ls20-9607627b"
    game_index: int = 3
    max_turns: int = 10
    seed: int = 0
    operation_mode: str = "offline"
    environments_dir: str = "environment_files"
    recordings_dir: str = "recordings"
    save_recording: bool = False


@dataclass(slots=True)
class RewardConfig:
    score_weight: float = 10.0
    learning_progress_weight: float = 10.0
    action_cost: float = 0.02
    time_cost_weight: float = 0.01
    update_cost: float = 0.01


@dataclass(slots=True)
class UpdateConfig:
    learning_rate: float = 1e-4
    update_steps: int = 1
    gradient_clip_norm: float = 1.0
    world_loss_mode: WorldLossMode = "pixel_ce"
    next_frame_loss_weight: float = 1.0
    latent_loss_weight: float = 1.0
    latent_changed_patch_weight: float = 6.0
    latent_huber_beta: float = 1.0
    latent_cosine_loss_weight: float = 0.1
    latent_cosine_min_delta_norm: float = 1e-4
    latent_learning_progress_normalization: bool = True
    latent_learning_progress_normalization_floor: float = 0.01
    policy_loss_weight: float = 0.05
    coord_loss_weight: float = 0.01
    holdout_min_transitions: int = 1
    learning_progress_horizon: int = 4
    learning_progress_discount: float = 0.8
    learning_progress_rate_beta: float = 0.5
    action_conditioned_learning_progress_baseline: bool = False
    reward_baseline_beta: float = 0.9
    policy_warmup_turns: int = 3
    policy_clip_epsilon: float = 0.2
    policy_advantage_baseline: PolicyAdvantageBaselineMode = "ema"
    policy_advantage_normalization: PolicyAdvantageNormalizationMode = "none"
    policy_advantage_normalization_beta: float = 0.5
    policy_advantage_normalization_floor: float = 0.01
    residual_frame_prediction: bool = False
    residual_frame_logit_bias: float = 4.0
    policy_update_accumulation_steps: int = 1
    policy_learning_progress_return_horizon: int = 12
    policy_learning_progress_return_discount: float = 0.93
    policy_adapter_trainable: bool = True


@dataclass(slots=True)
class LoggingConfig:
    output_dir: str = "runs/single_vlm_lora"
    save_video: bool = False
    video_fps: int = 4
    video_frame_scale: int = 8
    save_frame_predictions: bool = True
    frame_prediction_save_every: int = 1
    frame_prediction_frame_scale: int = 8
    save_latent_predictions: bool = True
    latent_prediction_save_every: int = 1
    latent_prediction_frame_scale: int = 16


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str = "single-vlm-lora"
    dry_run: bool = False
    frame_history_n: int = 4
    palette_size: int = 16
    frame_size: tuple[int, int] = (64, 64)
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentExperimentConfig = field(
        default_factory=EnvironmentExperimentConfig
    )
    rewards: RewardConfig = field(default_factory=RewardConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load one YAML experiment config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment config must be a mapping: {config_path}")
    return _dataclass_from_mapping(ExperimentConfig, raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return a YAML/JSON-safe config mapping."""

    return _jsonable(asdict(config))


def apply_cli_overrides(
    config: ExperimentConfig,
    *,
    game_id: str | None = None,
    game_index: int | None = None,
    max_turns: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    dry_run: bool = False,
    save_video: bool = False,
    video_fps: int | None = None,
    video_frame_scale: int | None = None,
    world_loss_mode: str | None = None,
    latent_loss_weight: float | None = None,
    latent_changed_patch_weight: float | None = None,
    latent_huber_beta: float | None = None,
    latent_cosine_loss_weight: float | None = None,
    latent_cosine_min_delta_norm: float | None = None,
    latent_learning_progress_normalization: bool | None = None,
    latent_learning_progress_normalization_floor: float | None = None,
    save_frame_predictions: bool | None = None,
    frame_prediction_save_every: int | None = None,
    frame_prediction_frame_scale: int | None = None,
    save_latent_predictions: bool | None = None,
    latent_prediction_save_every: int | None = None,
    latent_prediction_frame_scale: int | None = None,
) -> ExperimentConfig:
    """Apply CLI overrides in-place and return the config."""

    if game_id is not None:
        config.environment.game_id = game_id
    if game_index is not None:
        config.environment.game_index = game_index
        if game_id is None:
            config.environment.game_id = None
    if max_turns is not None:
        config.environment.max_turns = max_turns
    if seed is not None:
        config.environment.seed = seed
    if output_dir is not None:
        config.logging.output_dir = output_dir
    if dry_run:
        config.dry_run = True
    if save_video:
        config.logging.save_video = True
    if video_fps is not None:
        config.logging.video_fps = video_fps
    if video_frame_scale is not None:
        config.logging.video_frame_scale = video_frame_scale
    if world_loss_mode is not None:
        if world_loss_mode not in ("pixel_ce", "latent_grid", "hybrid"):
            raise ValueError(f"unsupported world_loss_mode: {world_loss_mode!r}")
        config.update.world_loss_mode = world_loss_mode
    if latent_loss_weight is not None:
        config.update.latent_loss_weight = float(latent_loss_weight)
    if latent_changed_patch_weight is not None:
        config.update.latent_changed_patch_weight = float(latent_changed_patch_weight)
    if latent_huber_beta is not None:
        config.update.latent_huber_beta = float(latent_huber_beta)
    if latent_cosine_loss_weight is not None:
        config.update.latent_cosine_loss_weight = float(latent_cosine_loss_weight)
    if latent_cosine_min_delta_norm is not None:
        config.update.latent_cosine_min_delta_norm = float(latent_cosine_min_delta_norm)
    if latent_learning_progress_normalization is not None:
        config.update.latent_learning_progress_normalization = bool(
            latent_learning_progress_normalization
        )
    if latent_learning_progress_normalization_floor is not None:
        config.update.latent_learning_progress_normalization_floor = float(
            latent_learning_progress_normalization_floor
        )
    if save_frame_predictions is not None:
        config.logging.save_frame_predictions = bool(save_frame_predictions)
    if frame_prediction_save_every is not None:
        config.logging.frame_prediction_save_every = int(frame_prediction_save_every)
    if frame_prediction_frame_scale is not None:
        config.logging.frame_prediction_frame_scale = int(frame_prediction_frame_scale)
    if save_latent_predictions is not None:
        config.logging.save_latent_predictions = bool(save_latent_predictions)
    if latent_prediction_save_every is not None:
        config.logging.latent_prediction_save_every = int(latent_prediction_save_every)
    if latent_prediction_frame_scale is not None:
        config.logging.latent_prediction_frame_scale = int(latent_prediction_frame_scale)
    return config


def _dataclass_from_mapping(cls: type[Any], raw: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in raw:
            continue
        value = raw[field_info.name]
        default_value = _field_default(field_info)
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[field_info.name] = _dataclass_from_mapping(type(default_value), value)
        elif field_info.name in {"image_size", "frame_size"}:
            kwargs[field_info.name] = _parse_size(value, field_info.name)
        elif field_info.name == "target_modules":
            kwargs[field_info.name] = tuple(str(item) for item in value)
        else:
            kwargs[field_info.name] = value
    return cls(**kwargs)


def _field_default(field_info: Any) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return MISSING


def _parse_size(value: Any, field_name: str) -> tuple[int, int]:
    if isinstance(value, str) and "x" in value:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
    else:
        raise ValueError(f"{field_name} must be like '224x224' or [224, 224]")
    if width <= 0 or height <= 0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return (width, height)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
