"""Debug-only data contracts.

These contracts describe persisted debug instrumentation, not runtime game
state or model-role behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelInputDebugRecord:
    """Raw provider request captured for one model call in a frame turn."""

    id: int
    m_state_id: int
    run_id: str
    game_id: str
    turn_id: int
    call_slot: str
    provider: str
    model: str | None
    phase: str
    attempt: int
    request: dict[str, Any]
    usage: Any | None
    metadata: dict[str, Any]
    created_at: str
