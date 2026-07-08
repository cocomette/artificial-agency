"""Configuration for updater model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UpdaterConfig:
    """Provider-neutral config for the updater model."""

    backend: str | None = None
    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
