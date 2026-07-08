"""Adapter shell for updater model backends."""

from __future__ import annotations

from face_of_agi.contracts import RoleContext
from face_of_agi.models.updater.config import UpdaterConfig
from face_of_agi.models.updater.contracts import (
    AgentContextUpdateInput,
    ToolContextUpdateInput,
)


class UpdaterAdapter:
    """No-op adapter shell for a replaceable updater backend."""

    def __init__(self, config: UpdaterConfig | None = None) -> None:
        self.config = config or UpdaterConfig()

    def update_tool_context(
        self,
        update_input: ToolContextUpdateInput,
    ) -> RoleContext:
        """Return the world or goal context unchanged."""

        return update_input.previous_context

    def update_agent_context(
        self,
        update_input: AgentContextUpdateInput,
    ) -> RoleContext:
        """Return the agent context unchanged."""

        return update_input.previous_context
