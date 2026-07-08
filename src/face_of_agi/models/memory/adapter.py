"""Provider-neutral adapter for the game memory model role."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from face_of_agi.contracts import ActionHistoryItem
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text_for_crop,
)
from face_of_agi.models.memory.config import GameMemoryConfig
from face_of_agi.models.memory.contracts import (
    GAME_MEMORY_MAX_CHARS,
    GameMemoryDocument,
    GameMemoryInput,
    PromptGameMemoryImage,
    PromptGameMemoryProvider,
    PromptGameMemoryProviderResponse,
    PromptGameMemoryRequest,
    game_memory_json_schema,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    emit_repair_attempt_event,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class GameMemoryOutputError(RuntimeError):
    """Raised when a memory backend returns unusable structured output."""


class DisabledGameMemoryAdapter:
    """Game-memory role implementation that intentionally performs no model call."""

    backend = "none"
    model = None

    def summarize_game_memory(
        self,
        memory_input: GameMemoryInput,
    ) -> GameMemoryDocument:
        """Return the unavailable sentinel without reading prompt evidence."""

        del memory_input
        document = GameMemoryDocument.not_available()
        document.metadata = {
            **document.metadata,
            "backend": self.backend,
            "disabled": True,
        }
        return document


class GameMemoryAdapter:
    """Provider-neutral game memory adapter that delegates model calls."""

    def __init__(
        self,
        provider: PromptGameMemoryProvider,
        config: GameMemoryConfig | None = None,
    ) -> None:
        self.config = config or GameMemoryConfig()
        self.provider = provider

    def summarize_game_memory(
        self,
        memory_input: GameMemoryInput,
    ) -> GameMemoryDocument:
        """Summarize same-run action and observation evidence as memory text."""

        memory_max_chars = self.config.memory_max_chars
        output_schema = game_memory_json_schema(
            memory_max_chars=memory_max_chars,
        )
        request = PromptGameMemoryRequest(
            instructions=append_output_schema_to_instructions(
                load_game_memory_instructions(self.config.instruction_path),
                output_schema,
                include=self.config.include_output_schema_in_instructions,
            ),
            text=build_game_memory_prompt(
                memory_input,
                crop_box_normalized=self.config.input_image_crop_box_normalized,
                memory_max_chars=memory_max_chars,
            ),
            images=game_memory_images(
                memory_input,
                frame_scale=self.config.frame_scale,
            ),
            output_schema=output_schema,
            metadata={
                **memory_input.metadata,
                "backend": self.provider.backend,
                "model": self.provider.model,
                "action_history_count": len(memory_input.action_history),
            },
        )
        response = self.provider.summarize_game_memory(request)
        validated = validate_with_repair(
            label=f"{self.provider.backend} game memory",
            response=response,
            text_of=lambda item: item.text,
            validate=parse_game_memory_output,
            repair=provider_repair_callback(
                self.provider,
                "repair_game_memory",
                args=(request,),
            ),
            max_repair_attempts=self.config.repair_attempts,
            error_factory=GameMemoryOutputError,
        )
        memory, final_response, length_metadata = _repair_oversized_memory(
            self.provider,
            request,
            memory=validated.value,
            response=validated.response,
            repair_attempts=validated.repair_attempts,
            max_repair_attempts=self.config.repair_attempts,
            memory_max_chars=memory_max_chars,
        )
        return GameMemoryDocument(
            markdown=memory,
            metadata={
                **request.metadata,
                **final_response.metadata,
                "available": True,
                "repair_attempts": length_metadata["repair_attempts"],
                "structural_repair_attempts": validated.repair_attempts,
                **length_metadata,
            },
        )


def load_game_memory_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable memory instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def build_game_memory_prompt(
    memory_input: GameMemoryInput,
    *,
    crop_box_normalized: Any | None = None,
    memory_max_chars: int | None = GAME_MEMORY_MAX_CHARS,
) -> str:
    """Build the provider-neutral text prompt for the memory role."""

    return "\n\n".join(
        [
            "## Attached frames\n\n"
            "- first_game_frame: first visible frame for this game run\n"
            "- current_game_frame: latest frame after the newest real action",
            "## Action history\n\n"
            + _numbered_action_history_text(
                memory_input.action_history,
                crop_box_normalized=crop_box_normalized,
            ),
            "## Memory task\n\n"
            "Write the updated compact game memory text for the whole "
            "same-run game history so far in the JSON `memory` field. "
            + _memory_limit_instruction(memory_max_chars),
        ]
    )


def game_memory_images(
    memory_input: GameMemoryInput,
    *,
    frame_scale: int,
) -> tuple[PromptGameMemoryImage, ...]:
    """Return first/latest frame images attached to the memory prompt."""

    return (
        PromptGameMemoryImage(
            label="first_game_frame",
            image=observation_to_pil_image(
                memory_input.first_observation,
                frame_scale=frame_scale,
            ),
        ),
        PromptGameMemoryImage(
            label="current_game_frame",
            image=observation_to_pil_image(
                memory_input.current_observation,
                frame_scale=frame_scale,
            ),
        ),
    )


def parse_game_memory_output(text: str) -> str:
    """Parse the required JSON game memory output contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise GameMemoryOutputError(
            "game memory response must be JSON with a non-empty string "
            f"'memory' field; raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise GameMemoryOutputError("game memory response must be a JSON object")
    keys = set(loaded)
    if keys != {"memory"}:
        missing = ["memory"] if "memory" not in keys else []
        unexpected = sorted(keys - {"memory"})
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(unexpected))
        raise GameMemoryOutputError(
            "game memory response must contain exactly one top-level "
            "'memory' field"
            + (": " + "; ".join(details) if details else "")
        )
    memory = loaded["memory"]
    if not isinstance(memory, str):
        raise GameMemoryOutputError("game memory 'memory' field must be a string")
    memory = memory.strip()
    if not memory:
        raise GameMemoryOutputError("game memory 'memory' field must be non-empty")
    return memory


def build_game_memory_repair_prompt(
    *,
    invalid_text: str,
    validation_error: str,
    attempt: int,
    memory_max_chars: int | None = GAME_MEMORY_MAX_CHARS,
) -> str:
    """Return compact provider-neutral memory repair instructions."""

    return "\n\n".join(
        [
            f"Repair attempt {attempt}: the previous game memory output was invalid.",
            "Validation error:\n" + validation_error,
            "Previous output excerpt:\n" + _repair_excerpt(invalid_text),
            (
                "Return only corrected JSON with exactly one top-level `memory` "
                "field. The `memory` value must be non-empty and compact. "
                + _memory_limit_instruction(memory_max_chars)
                + " Preserve "
                "specific action effects, progress, failed patterns, current "
                "objective hypotheses, and uncertainty. Do not add any other "
                "top-level fields."
            ),
        ]
    )


def game_memory_length_validation_error(
    memory: str,
    *,
    memory_max_chars: int | None = GAME_MEMORY_MAX_CHARS,
) -> str:
    """Return the soft validation error text for over-limit memory."""

    if memory_max_chars is None:
        return "game memory length validation is disabled"
    return (
        f"game memory 'memory' field is {len(memory)} characters; expected at "
        f"most {memory_max_chars}"
    )


def _repair_oversized_memory(
    provider: PromptGameMemoryProvider,
    request: PromptGameMemoryRequest,
    *,
    memory: str,
    response: PromptGameMemoryProviderResponse,
    repair_attempts: int,
    max_repair_attempts: int,
    memory_max_chars: int | None,
) -> tuple[str, PromptGameMemoryProviderResponse, dict[str, Any]]:
    original_char_count = len(memory)
    length_repair_attempted = False
    length_repair_succeeded = False
    skipped_reason: str | None = None
    final_memory = memory
    final_response = response
    final_repair_attempts = repair_attempts

    if memory_max_chars is not None and original_char_count > memory_max_chars:
        if repair_attempts < max_repair_attempts:
            length_repair_attempted = True
            final_repair_attempts = repair_attempts + 1
            length_error = game_memory_length_validation_error(
                memory,
                memory_max_chars=memory_max_chars,
            )
            emit_repair_attempt_event(
                provider,
                validation_error=length_error,
                attempt=final_repair_attempts,
            )
            repair_response = provider.repair_game_memory(
                request,
                invalid_text=json.dumps({"memory": memory}, ensure_ascii=False),
                validation_error=length_error,
                attempt=final_repair_attempts,
            )
            try:
                final_memory = parse_game_memory_output(repair_response.text)
            except Exception as exc:
                raise GameMemoryOutputError(
                    "game memory length repair produced invalid structured "
                    f"output: {exc}"
                ) from exc
            final_response = repair_response
            length_repair_succeeded = True
        else:
            skipped_reason = "repair_budget_exhausted"

    return (
        final_memory,
        final_response,
        {
            "repair_attempts": final_repair_attempts,
            "memory_char_count": len(final_memory),
            "memory_initial_char_count": original_char_count,
            "memory_char_limit": memory_max_chars,
            "memory_oversize": (
                False
                if memory_max_chars is None
                else len(final_memory) > memory_max_chars
            ),
            "memory_initial_oversize": (
                False
                if memory_max_chars is None
                else original_char_count > memory_max_chars
            ),
            "memory_length_repair_attempted": length_repair_attempted,
            "memory_length_repair_succeeded": length_repair_succeeded,
            "memory_length_repair_skipped_reason": skipped_reason,
        },
    )


def _numbered_action_history_text(
    history: tuple[ActionHistoryItem, ...],
    *,
    crop_box_normalized: Any | None,
) -> str:
    if not history:
        return "none"
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text_for_crop(crop_box_normalized),
        numbered=True,
        latest_description=(
            "Numbered oldest-to-newest. Controllable action rows may include "
            "nested animation_after rows; GAME_RESET rows mark environment "
            "resets between action groups, and SCORE_ADVANCE rows mark score "
            "or progress increases. The [latest] marker identifies the newest "
            "transition included in this memory update."
        ),
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return stripped
    return match.group(1).strip()


def _repair_excerpt(text: str, *, limit: int = 6_000) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    head_count = limit // 2
    tail_count = limit - head_count
    return (
        stripped[:head_count]
        + "\n...[truncated for compact repair]...\n"
        + stripped[-tail_count:]
    )


def _memory_limit_instruction(memory_max_chars: int | None) -> str:
    if memory_max_chars is None:
        return "Keep `memory` compact and focused."
    return f"Keep `memory` at or below {int(memory_max_chars):,} characters."
