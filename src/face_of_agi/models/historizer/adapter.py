"""Provider-neutral adapter for the agent context historizer role."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from face_of_agi.models.historizer.config import HistorizerConfig
from face_of_agi.models.historizer.contracts import (
    AGENT_CONTEXT_HISTORY_KEYS,
    AGENT_CONTEXT_HISTORY_FIELD_MAX_CHARS,
    AgentContextHistoryInput,
    AgentContextHistorySummary,
    PromptHistorizerProvider,
    PromptHistorizerProviderResponse,
    PromptHistorizerRequest,
    agent_context_history_json_schema,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class HistorizerOutputError(RuntimeError):
    """Raised when a historizer backend returns invalid output."""


class AgentContextHistorizerAdapter:
    """Provider-neutral historizer that delegates only the model call."""

    def __init__(
        self,
        provider: PromptHistorizerProvider,
        config: HistorizerConfig | None = None,
    ) -> None:
        self.config = config or HistorizerConfig()
        self.provider = provider

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        """Summarize how prior agent context fields evolved."""

        field_max_chars = getattr(
            self.config,
            "field_max_chars",
            AGENT_CONTEXT_HISTORY_FIELD_MAX_CHARS,
        )
        output_schema = agent_context_history_json_schema(
            field_max_chars=field_max_chars,
        )
        instructions = append_output_schema_to_instructions(
            load_historizer_instructions(self.config.instruction_path),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptHistorizerRequest(
            instructions=instructions,
            text=_history_prompt_text(history_input),
            output_schema=output_schema,
            metadata={
                "backend": self.provider.backend,
                "model": self.provider.model,
                "game_id": history_input.game_id,
                "context_window": history_input.context_window,
                "context_count": len(history_input.contexts),
            },
        )
        response = self.provider.summarize_context_history(request)
        validated = validate_with_repair(
            label=f"{self.provider.backend} historizer",
            response=response,
            text_of=lambda item: item.text,
            validate=lambda text: parse_agent_context_history_output(
                text,
                field_max_chars=field_max_chars,
            ),
            repair=provider_repair_callback(
                self.provider,
                "repair_context_history",
                args=(request,),
            ),
            max_repair_attempts=self.config.repair_attempts,
            error_factory=HistorizerOutputError,
        )
        summary = validated.value
        summary.metadata = {
            **request.metadata,
            **validated.response.metadata,
            "available": True,
            "repair_attempts": validated.repair_attempts,
        }
        return summary


def load_historizer_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable historizer instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def parse_agent_context_history_output(
    text: str,
    *,
    field_max_chars: int | None = AGENT_CONTEXT_HISTORY_FIELD_MAX_CHARS,
) -> AgentContextHistorySummary:
    """Parse the required JSON historizer output contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise HistorizerOutputError(
            "historizer response must be JSON with a 'field_evolution' object; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise HistorizerOutputError("historizer response must be a JSON object")
    field_evolution = loaded.get("field_evolution")
    if not isinstance(field_evolution, dict):
        raise HistorizerOutputError(
            "historizer response JSON is missing object field 'field_evolution'"
        )
    missing = [
        key for key in AGENT_CONTEXT_HISTORY_KEYS if key not in field_evolution
    ]
    if missing:
        raise HistorizerOutputError(
            "historizer field_evolution is missing keys: " + ", ".join(missing)
        )
    unexpected = sorted(
        set(field_evolution) - set(AGENT_CONTEXT_HISTORY_KEYS)
    )
    if unexpected:
        raise HistorizerOutputError(
            "historizer field_evolution has unexpected keys: "
            + ", ".join(unexpected)
        )
    invalid = [
        key for key, value in field_evolution.items() if not isinstance(value, str)
    ]
    if invalid:
        raise HistorizerOutputError(
            "historizer field_evolution values must be strings: "
            + ", ".join(sorted(invalid))
        )
    oversized = [
        key
        for key, value in field_evolution.items()
        if (
            field_max_chars is not None
            and isinstance(value, str)
            and len(value) > field_max_chars
        )
    ]
    if oversized:
        details = ", ".join(
            f"{key}={len(field_evolution[key])}"
            for key in sorted(oversized)
        )
        raise HistorizerOutputError(
            "historizer field_evolution values exceed the "
            f"{field_max_chars} character cap: {details}"
        )
    ordered = {key: field_evolution[key] for key in AGENT_CONTEXT_HISTORY_KEYS}
    return AgentContextHistorySummary(field_evolution=ordered)


def _history_prompt_text(history_input: AgentContextHistoryInput) -> str:
    return "\n\n".join(
        [
            "## Game\n\n" + history_input.game_id,
            "## Context history window\n\n"
            + f"- prior_agent_context_window: {history_input.context_window}",
            "## Agent game context history\n\n"
            + _numbered_context_history_text(history_input.contexts),
        ]
    )


def _numbered_context_history_text(contexts: tuple[str, ...]) -> str:
    if not contexts:
        return "not available"
    lines = [
        (
            "Numbered oldest-to-newest. Each item is a complete prior agent "
            "game context returned by the updater."
        )
    ]
    for index, context in enumerate(contexts, start=1):
        lines.append(f"{index}. {_text_or_none(context)}")
    return "\n\n".join(lines)


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


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
