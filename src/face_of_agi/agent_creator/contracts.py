"""Shared contracts for dynamic agent-updater roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.contracts import ActionHistoryItem, Observation


@dataclass(frozen=True, slots=True)
class AgentRoleDefinition:
    """One shared updater role available across games."""

    role: str
    meta_description: str
    role_instructions: str


@dataclass(frozen=True, slots=True)
class AgentRoleSnapshot:
    """Immutable complete role-set snapshot produced by the agent creator."""

    id: int | None
    roles: tuple[AgentRoleDefinition, ...]
    general_system_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class AgentStrategySnapshot:
    """One chronological strategy-history item for a game."""

    role: str
    strategy: str


@dataclass(frozen=True, slots=True)
class AgentCreatorBatchItem:
    """Creator-facing payload built from one latest complete game state."""

    run_id: str
    game_id: str
    strategy_history: tuple[AgentStrategySnapshot, ...]
    current_observation: Observation
    action_history: tuple[ActionHistoryItem, ...]
    roles: tuple[AgentRoleDefinition, ...]
    general_system_prompt: str
    world_model_context: dict[str, Any] = field(default_factory=dict)
    role_snapshot_id: int | None = None
    state_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClaimedAgentCreatorBatch:
    """A full pending creator game-request batch claimed for processing."""

    request_ids: tuple[int, ...]
    requests: tuple["AgentCreatorGameRequest", ...]


@dataclass(frozen=True, slots=True)
class AgentCreatorGameRequest:
    """One queued game asking the creator to review its latest complete state."""

    id: int
    run_id: str
    game_id: str
    memory_database_path: str
    status: str = "pending"
    created_at: str = ""
    claimed_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentCreatorRun:
    """One claimed queued-game batch run for role mutation tools."""

    id: int
    status: str
    request_ids: tuple[int, ...]
    max_tool_calls: int
    created_at: str = ""
    completed_at: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentCreatorToolResult:
    """Result content returned to the creator orchestrator after one tool call."""

    status: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Return whether the tool call succeeded."""

        return self.status == "ok"
