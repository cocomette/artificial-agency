"""Contracts for the regenerated Memory role."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import MemoryDocument, Observation


def memory_output_json_schema() -> dict[str, Any]:
    """Return the provider-neutral Memory output schema."""

    return {
        "type": "object",
        "properties": {
            "document": {
                "type": "string",
                "description": "Fresh detailed memory document for the run so far.",
            },
        },
        "required": ["document"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class MemoryLedgerEntry:
    """Sanitized Memory-facing row from the richer orchestration ledger."""

    turn_id: int
    action: str
    change_summary: str


@dataclass(frozen=True, slots=True)
class MemoryBuildInput:
    """Input for regenerating the model-facing memory document."""

    run_id: str
    game_id: str
    first_observation: Observation
    current_observation: Observation
    ledger: Sequence[MemoryLedgerEntry] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryModel(Protocol):
    """Model role that regenerates run memory from a sanitized ledger."""

    def build_memory(self, build_input: MemoryBuildInput) -> MemoryDocument:
        """Return a fresh memory document."""
        ...
