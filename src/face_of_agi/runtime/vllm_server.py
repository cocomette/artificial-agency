"""Shared helpers for launching local OpenAI-compatible vLLM servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_VLLM_PORT = 8000


@dataclass(frozen=True)
class VLLMServerConfig:
    """vLLM server settings parsed from one runtime config."""

    model: str
    host: str = DEFAULT_VLLM_HOST
    port: int = DEFAULT_VLLM_PORT
    model_path: str | None = None
    max_model_len: int | None = None
    reasoning_parser: str | None = None
    enable_lora: bool = True
    max_loras: int | None = 3
    max_lora_rank: int | None = None
    allow_runtime_lora_updating: bool = True
    extra_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        """Return the local OpenAI-compatible base URL for the server."""

        return f"http://{self.host}:{self.port}/v1"


def vllm_server_config_from_config_text(
    config_text: str,
) -> VLLMServerConfig | None:
    """Return local vLLM server settings referenced by a YAML config."""

    raw = yaml.safe_load(config_text) or {}
    models = raw.get("models") or {}
    if not isinstance(models, dict):
        return None

    shared = models.get("shared_vlm") or {}
    shared_backend = _backend(shared)
    found: list[str] = []
    if shared_backend == "vllm":
        _append_model(found, shared.get("model"))
    for role_name in ("agent", "change", "memory", "world", "goal", "reward_judge"):
        role = models.get(role_name) or {}
        if _backend(role) == "vllm":
            _append_model(found, role.get("model") or shared.get("model"))
    if not found:
        return None

    server = shared.get("server") if isinstance(shared, dict) else None
    if server is None:
        server = {}
    if not isinstance(server, dict):
        raise ValueError("models.shared_vlm.server must be a mapping")
    extra_args = server.get("extra_args") or ()
    if isinstance(extra_args, str):
        extra_args = (extra_args,)
    if not isinstance(extra_args, (list, tuple)):
        raise ValueError("models.shared_vlm.server.extra_args must be a list")

    return VLLMServerConfig(
        model=found[0],
        host=str(server.get("host", DEFAULT_VLLM_HOST)),
        port=int(server.get("port", DEFAULT_VLLM_PORT)),
        model_path=_optional_server_string(server.get("model_path")),
        max_model_len=_optional_server_int(server.get("max_model_len")),
        reasoning_parser=_optional_server_string(server.get("reasoning_parser")),
        enable_lora=_optional_server_bool(server.get("enable_lora"), default=True),
        max_loras=_optional_server_int(server.get("max_loras", 3)),
        max_lora_rank=_optional_server_int(server.get("max_lora_rank")),
        allow_runtime_lora_updating=_optional_server_bool(
            server.get("allow_runtime_lora_updating"),
            default=True,
        ),
        extra_args=tuple(str(arg) for arg in extra_args),
    )


def vllm_server_command(config: VLLMServerConfig) -> tuple[str, ...]:
    """Return the `vllm serve` command for one parsed server config."""

    served_target = config.model_path or config.model
    command = [
        "vllm",
        "serve",
        served_target,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    if config.model_path is not None:
        command.extend(["--served-model-name", config.model])
    if config.max_model_len is not None:
        command.extend(["--max-model-len", str(config.max_model_len)])
    if config.reasoning_parser:
        command.extend(["--reasoning-parser", config.reasoning_parser])
    if config.enable_lora:
        command.append("--enable-lora")
    if config.max_loras is not None:
        command.extend(["--max-loras", str(config.max_loras)])
    if config.max_lora_rank is not None:
        command.extend(["--max-lora-rank", str(config.max_lora_rank)])
    command.extend(config.extra_args)
    return tuple(command)


def _append_model(target: list[str], model: Any) -> None:
    if isinstance(model, str) and model and model not in target:
        target.append(model)


def _backend(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("backend") or "").lower()


def _optional_server_string(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _optional_server_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_server_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)
