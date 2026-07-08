"""Shared agent-creator role store and contracts."""

from importlib import import_module
from typing import Any

from face_of_agi.agent_creator.contracts import (
    AgentCreatorBatchItem,
    AgentCreatorGameRequest,
    AgentCreatorRun,
    AgentCreatorToolResult,
    AgentRoleDefinition,
    AgentRoleSnapshot,
    AgentStrategySnapshot,
    ClaimedAgentCreatorBatch,
)
from face_of_agi.agent_creator.defaults import (
    default_agent_roles,
    default_general_agent_system_prompt,
)
from face_of_agi.agent_creator.store import AgentCreatorStore
from face_of_agi.agent_creator.mutations import RoleMutationToolExecutor

__all__ = [
    "AgentCreatorBatchItem",
    "AgentCreatorGameRequest",
    "AgentCreatorRun",
    "AgentCreatorService",
    "AgentCreatorStore",
    "AgentCreatorToolResult",
    "AgentRoleDefinition",
    "AgentRoleSnapshot",
    "AgentStrategySnapshot",
    "ClaimedAgentCreatorBatch",
    "RoleMutationToolExecutor",
    "default_agent_roles",
    "default_general_agent_system_prompt",
]


def __getattr__(name: str) -> Any:
    """Load the service lazily so memory can import creator contracts safely."""

    if name == "AgentCreatorService":
        value = import_module(
            "face_of_agi.agent_creator.service"
        ).AgentCreatorService
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
