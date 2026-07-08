"""Shared contracts for structured description prediction providers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from face_of_agi.contracts import (
    DESCRIPTION_SCHEMA,
    ToolName,
    VisualCoordinateSpace,
)


def description_schema_for_coordinate_space(
    coordinate_space: VisualCoordinateSpace,
) -> dict[str, Any]:
    """Return a fresh description schema for a model's visual coordinates."""

    return deepcopy(DESCRIPTION_SCHEMA)


@dataclass(frozen=True, slots=True)
class DescriptionRoleSpec:
    """Role-specific description prediction details."""

    tool_name: ToolName
    id_prefix: str
    instruction_dir: Path
    validation_label: str
    provider_label: str
    explanation: str
    include_action: bool


@dataclass(frozen=True, slots=True)
class DescriptionProviderResponse:
    """Provider text output plus normalized description-provider metadata."""

    text: str
    metadata: dict[str, Any]
    request: dict[str, Any] | None = None


class DescriptionProvider(Protocol):
    """Provider transport for structured description prompts."""

    coordinate_space: str

    def complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        image: Any | None,
    ) -> DescriptionProviderResponse:
        """Return provider text, metadata, and request diagnostics."""
        ...

    def repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        image: Any | None,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> DescriptionProviderResponse:
        """Return a repaired provider response for invalid structured output."""
        ...
