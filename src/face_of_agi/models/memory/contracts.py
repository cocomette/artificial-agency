"""Contracts for the game memory model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import ActionHistoryItem, Observation

_FORBIDDEN_MEMORY_METADATA_KEYS = frozenset({"game_id", "run_id"})
GAME_MEMORY_MAX_CHARS = 10_000


def game_memory_json_schema(
    *,
    memory_max_chars: int | None = GAME_MEMORY_MAX_CHARS,
) -> dict[str, Any]:
    """Return the provider-neutral game memory output JSON schema."""

    memory_schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "description": (
            "Compact same-run game memory text to pass to the agent "
            "and agent-game updater."
        ),
    }
    if memory_max_chars is not None:
        memory_schema["maxLength"] = int(memory_max_chars)

    return {
        "type": "object",
        "properties": {
            "memory": memory_schema,
        },
        "required": ["memory"],
        "additionalProperties": False,
    }


def openai_game_memory_text_format(
    *,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return OpenAI Responses text format for game memory outputs."""

    return {
        "format": {
            "type": "json_schema",
            "name": "game_memory",
            "strict": True,
            "schema": schema or game_memory_json_schema(),
        }
    }


@dataclass(slots=True)
class GameMemoryInput:
    """Input for producing a compact same-run game memory document."""

    action_history: tuple[ActionHistoryItem, ...]
    first_observation: Observation
    current_observation: Observation
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject identifying metadata at the role boundary."""

        forbidden = sorted(_FORBIDDEN_MEMORY_METADATA_KEYS.intersection(self.metadata))
        if forbidden:
            raise ValueError(
                "GameMemoryInput metadata must not include identifying keys: "
                + ", ".join(forbidden)
            )


@dataclass(slots=True)
class GameMemoryDocument:
    """Compact prompt-facing memory of how the game has evolved so far."""

    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_available(cls) -> "GameMemoryDocument":
        """Return the prompt-facing empty memory sentinel."""

        return cls(markdown="not available", metadata={"available": False})

    def is_available(self) -> bool:
        """Return whether the document came from a memory model call."""

        return bool(self.metadata.get("available", True))


@dataclass(slots=True)
class PromptGameMemoryImage:
    """Provider-neutral image attached to a memory prompt."""

    label: str
    image: Any


@dataclass(slots=True)
class PromptGameMemoryRequest:
    """Provider-neutral prompt request for the game memory role."""

    instructions: str
    text: str
    images: tuple[PromptGameMemoryImage, ...]
    output_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptGameMemoryProviderResponse:
    """Raw provider output for one game memory request."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptGameMemoryProvider(Protocol):
    """Thin backend boundary for one memory generation request."""

    backend: str
    model: str | None

    def summarize_game_memory(
        self,
        request: PromptGameMemoryRequest,
    ) -> PromptGameMemoryProviderResponse:
        """Return raw provider text for a game memory document."""
        ...

    def repair_game_memory(
        self,
        request: PromptGameMemoryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptGameMemoryProviderResponse:
        """Return repaired raw provider text for invalid structured output."""
        ...


class GameMemoryModel(Protocol):
    """Model role that summarizes same-run game evolution."""

    def summarize_game_memory(
        self,
        memory_input: GameMemoryInput,
    ) -> GameMemoryDocument:
        """Return a compact prompt-facing game memory document."""
        ...
