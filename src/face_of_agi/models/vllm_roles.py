"""Shared helpers for vLLM-only ARC role adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from face_of_agi.contracts import ActionSpec, Observation
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.action_history import model_facing_action_text
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.structured_output import append_output_schema_to_instructions
from face_of_agi.debug.capture import capture_vllm_model_input


@dataclass(slots=True)
class VLLMRoleConfig:
    """Shared vLLM Chat Completions config for new v1 roles."""

    backend: str | None = "vllm"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = "VLLM_API_KEY"
    base_url: str = "http://127.0.0.1:8000/v1"
    timeout: float | None = None
    max_retries: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    use_response_format: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    extra_request_options: dict[str, Any] = field(default_factory=dict)
    include_output_schema_in_instructions: bool = False
    repair_attempts: int = 1
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | list[int] | None = None
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    lora_adapter_name: str | None = None


class VLLMJsonRoleClient:
    """Small JSON-schema vLLM role caller with debug capture."""

    backend = "vllm"

    def __init__(
        self,
        *,
        config: VLLMRoleConfig,
        call_slot: str,
        instruction_path: Path,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError(f"models.{call_slot}.model is required for backend vllm")
        self.config = config
        self.model = config.model
        self.call_slot = call_slot
        self.instructions = instruction_path.read_text(encoding="utf-8").strip()
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_usage: Any | None = None
        self.active_lora_adapter_name: str | None = None

    def activate_lora_adapter(self, adapter_name: str) -> None:
        """Use a runtime-loaded vLLM LoRA adapter for future role calls."""

        self.active_lora_adapter_name = adapter_name

    def complete_json(
        self,
        *,
        prompt_text: str,
        output_schema: dict[str, Any],
        schema_name: str,
        images: tuple[Any, ...] = (),
        phase: str = "complete",
    ) -> str:
        """Call vLLM and return raw JSON text."""

        instructions = append_output_schema_to_instructions(
            self.instructions,
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        messages = self._messages(
            instructions=instructions,
            prompt_text=prompt_text,
            images=images,
        )
        max_repairs = max(0, self.config.repair_attempts)
        for attempt in range(max_repairs + 1):
            response = self._client.chat(
                model=self.active_lora_adapter_name or self.config.model,
                messages=messages,
                response_format=json_schema_response_format(
                    name=schema_name,
                    schema=output_schema,
                ),
            )
            self.last_request = self._client.last_request
            self.last_response_text = chat_message_optional_content(response) or ""
            response_metadata = chat_response_metadata(response)
            self.last_usage = response_metadata.get("usage")
            if self.last_request is not None:
                capture_vllm_model_input(
                    self,
                    call_slot=self.call_slot,
                    provider=self.backend,
                    model=self.model,
                    phase=phase,
                    request=self.last_request,
                    response=response,
                    metadata={"response_metadata": response_metadata},
                    attempt=attempt,
                )
            try:
                parse_json_object(self.last_response_text, label=self.call_slot)
            except RuntimeError as exc:
                if attempt >= max_repairs:
                    raise
                messages.extend(
                    [
                        {"role": "assistant", "content": self.last_response_text},
                        {
                            "role": "user",
                            "content": _json_repair_prompt(
                                schema_name=schema_name,
                                validation_error=str(exc),
                                invalid_text=self.last_response_text,
                                attempt=attempt + 1,
                            ),
                        },
                    ]
                )
                continue
            return self.last_response_text
        raise RuntimeError("unreachable vLLM role repair state")

    def _messages(
        self,
        *,
        instructions: str,
        prompt_text: str,
        images: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        content.extend(
            vllm_image_content(
                images,
                detail=self.config.input_image_detail,
                size=self.config.input_image_size,
                resample=self.config.input_image_resample,
                mime_type=self.config.image_mime_type,
                crop_edges=self.config.input_image_crop_arc_grid_edges,
            )
        )
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ]

def observation_image(config: VLLMRoleConfig, observation: Observation) -> Any:
    """Return one PIL image for a role-visible observation."""

    return observation_to_pil_image(
        observation,
        frame_scale=config.frame_scale,
    )


def action_text(action: ActionSpec, *, crop_edges: Any | None = None) -> str:
    """Return prompt-facing action text."""

    return model_facing_action_text(action, crop_edges=crop_edges)


def parse_json_object(text: str, *, label: str) -> dict[str, Any]:
    """Parse a provider JSON object or fail with a role-specific message."""

    import json

    loaded = _decode_json_object_text(text, label=label)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{label} response must be a JSON object")
    return loaded


def _decode_json_object_text(text: str, *, label: str) -> Any:
    """Decode a JSON object from model text that may include light wrapping."""

    import json

    stripped = text.strip()
    candidates = [stripped]
    fenced = _strip_markdown_json_fence(stripped)
    if fenced != stripped:
        candidates.append(fenced)
    candidates.extend(_json_object_suffixes(stripped))

    last_error: json.JSONDecodeError | None = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if candidate[end:].strip():
            continue
        return value
    if last_error is None:
        raise RuntimeError(f"{label} response was empty")
    raise RuntimeError(
        f"{label} response was not valid JSON: {last_error}; "
        f"raw response preview: {_preview_text(text)!r}"
    ) from last_error


def _strip_markdown_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def _json_object_suffixes(text: str) -> list[str]:
    return [text[index:].strip() for index, char in enumerate(text) if char == "{"]


def _preview_text(text: str, *, limit: int = 300) -> str:
    return text.strip().replace("\n", "\\n")[:limit]


def _json_repair_prompt(
    *,
    schema_name: str,
    validation_error: str,
    invalid_text: str,
    attempt: int,
) -> str:
    return "\n\n".join(
        [
            f"Repair attempt {attempt}: the previous {schema_name} output was invalid.",
            "Validation error:\n" + validation_error,
            "Invalid output:\n" + invalid_text,
            "Return only the corrected JSON object. Do not include prose, Markdown fences, or the JSON schema.",
        ]
    )


def bounded_float(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    """Return a finite float within an inclusive range."""

    from math import isfinite

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    numeric = float(value)
    if not isfinite(numeric) or not minimum <= numeric <= maximum:
        raise RuntimeError(f"{label} must be within {minimum}..{maximum}")
    return numeric


def non_negative_int(value: Any, *, label: str) -> int:
    """Return a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    if value < 0:
        raise RuntimeError(f"{label} must be non-negative")
    return value
