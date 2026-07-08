"""Contracts for the updater model P."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from face_of_agi.contracts import (
    ActionOutcomeEvidence,
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.models.historizer import AgentContextHistorySummary
from face_of_agi.models.memory import GameMemoryDocument

UpdaterRole = Literal["agent"]
ContextSegment = Literal["general", "game"]
UpdaterTask = Literal["agent_game", "general"]
AGENT_GAME_CONTEXT_KEYS = (
    "goals",
    "game_mechanics",
    "policy",
    "history",
    "extras",
)
AGENT_GAME_CONTEXT_MAX_CHARS = 12000
GENERAL_CONTEXT_MAX_CHARS = 20_000
AGENT_GAME_CONTEXT_FIELD_MAX_CHARS = 6_000


def updated_context_json_schema(
    *,
    general_context_max_chars: int | None = GENERAL_CONTEXT_MAX_CHARS,
) -> dict[str, Any]:
    """Return the provider-neutral updater output JSON schema."""

    updated_context_schema: dict[str, Any] = {
        "type": "string",
        "description": "The complete revised context text.",
    }
    if general_context_max_chars is not None:
        updated_context_schema["maxLength"] = int(general_context_max_chars)

    return {
        "type": "object",
        "properties": {
            "updated_context": updated_context_schema,
        },
        "required": ["updated_context"],
        "additionalProperties": False,
    }


def agent_game_updated_context_json_schema(
    *,
    agent_game_context_max_chars: int | None = AGENT_GAME_CONTEXT_MAX_CHARS,
    agent_game_context_field_max_chars: int | None = (
        AGENT_GAME_CONTEXT_FIELD_MAX_CHARS
    ),
) -> dict[str, Any]:
    """Return the agent game updater output JSON schema."""

    descriptions = {
        "goals": "Current objective, progress target, and goal hypothesis.",
        "game_mechanics": "Useful world/action dynamics and uncertainty.",
        "policy": (
            "Action-selection guidance for the next decision; under "
            "stagnation, an explicit action-forcing directive."
        ),
        "history": "Learnings from past outcomes and progress evidence.",
        "extras": "Other useful agent guidance.",
    }
    return {
        "type": "object",
        "properties": {
            "updated_context": {
                "type": "object",
                "description": (
                    "Complete latest agent game context. The serialized "
                    f"context must be at most {agent_game_context_max_chars} "
                    "characters."
                ),
                "properties": {
                    key: {
                        "type": "string",
                        "description": descriptions[key],
                        **(
                            {"maxLength": int(agent_game_context_field_max_chars)}
                            if agent_game_context_field_max_chars is not None
                            else {}
                        ),
                    }
                    for key in AGENT_GAME_CONTEXT_KEYS
                },
                "required": list(AGENT_GAME_CONTEXT_KEYS),
                "additionalProperties": False,
            },
        },
        "required": ["updated_context"],
        "additionalProperties": False,
    }


def updater_output_json_schema(
    task: UpdaterTask,
    *,
    general_context_max_chars: int | None = GENERAL_CONTEXT_MAX_CHARS,
    agent_game_context_max_chars: int | None = AGENT_GAME_CONTEXT_MAX_CHARS,
    agent_game_context_field_max_chars: int | None = (
        AGENT_GAME_CONTEXT_FIELD_MAX_CHARS
    ),
) -> dict[str, Any]:
    """Return the provider-neutral output schema for one updater task."""

    if task == "agent_game":
        return agent_game_updated_context_json_schema(
            agent_game_context_max_chars=agent_game_context_max_chars,
            agent_game_context_field_max_chars=agent_game_context_field_max_chars,
        )
    return updated_context_json_schema(
        general_context_max_chars=general_context_max_chars,
    )


@dataclass(slots=True)
class UpdaterContextTarget:
    """Specific role context segment selected by orchestration."""

    role: UpdaterRole
    segment: ContextSegment
    task: UpdaterTask
    previous_context: RoleContext


@dataclass(slots=True)
class PromptImage:
    """Provider-neutral image attached to a prompt updater request."""

    label: str
    image: Any


@dataclass(slots=True)
class PromptUpdateRequest:
    """Provider-neutral prompt update request built for a concrete target."""

    target: UpdaterContextTarget
    instructions: str
    text: str
    output_schema: dict[str, Any]
    images: tuple[PromptImage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptUpdateResult:
    """Provider-neutral prompt update result from one updater backend."""

    target: UpdaterContextTarget
    updated_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptUpdateProviderResponse:
    """Raw provider output for one prompt update request."""

    target: UpdaterContextTarget
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentProgressFeedback:
    """Progress feedback visible to the agent updater."""

    time_cost: float | None = None
    cumulative_score: float | None = None
    game_last_started_turns_ago: int | None = None
    score_last_advanced_turns_ago: int | None = None
    game_start_reason: str | None = None
    game_restart_count: int = 0


@dataclass(slots=True)
class AgentContextRevisionFeedback:
    """Deterministic staleness signal for the agent updater."""

    compared_turns: int = 0
    goals_unchanged_turns: int = 0
    game_mechanics_unchanged_turns: int = 0
    policy_unchanged_turns: int = 0
    history_unchanged_turns: int = 0
    extras_unchanged_turns: int = 0


@dataclass(slots=True)
class AgentGameContextUpdateInput:
    """Input for updating the agent game-specific context document."""

    previous_context: RoleContext
    current_observation: Observation
    allowed_actions: tuple[ActionSpec, ...]
    glossary_actions: tuple[ActionSpec, ...]
    action_history_window: int
    game_memory: GameMemoryDocument = field(
        default_factory=GameMemoryDocument.not_available
    )
    context_history: AgentContextHistorySummary = field(
        default_factory=AgentContextHistorySummary.not_available
    )
    action_history: tuple[ActionHistoryItem, ...] = ()
    turn_metrics: AgentProgressFeedback = field(
        default_factory=AgentProgressFeedback
    )
    context_revision_feedback: AgentContextRevisionFeedback = field(
        default_factory=AgentContextRevisionFeedback
    )
    action_outcome_evidence: ActionOutcomeEvidence = field(
        default_factory=ActionOutcomeEvidence
    )


@dataclass(slots=True)
class GeneralKnowledgeUpdateInput:
    """Run-level input for updating one role's game-agnostic context."""

    role: UpdaterRole
    previous_context: RoleContext = field(default_factory=RoleContext)
    run_id: str = ""
    game_id: str = ""
    stop_reason: str | None = None
    step_count: int = 0
    completed_levels: int = 0
    final_state: str | None = None
    state_record_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentGameContextUpdaterModel(Protocol):
    """Updater task for agent game context `L^X`."""

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        """Return the next agent game context document."""
        ...


class GeneralKnowledgeUpdaterModel(Protocol):
    """Shared updater task for role-specific general knowledge `K`."""

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        """Return the next role context after an end-of-run general update."""
        ...


@dataclass(slots=True)
class UpdaterTaskRegistry:
    """Configured updater task instances for runtime orchestration."""

    agent_game_updater: AgentGameContextUpdaterModel | None = None
    general_updater: GeneralKnowledgeUpdaterModel | None = None

    def require_agent_game_updater(self) -> AgentGameContextUpdaterModel:
        """Return the agent game updater, failing if not wired."""

        if self.agent_game_updater is None:
            raise RuntimeError("agent game updater is not registered")
        return self.agent_game_updater

    def require_general_updater(self) -> GeneralKnowledgeUpdaterModel:
        """Return the shared general updater, failing if not wired."""

        if self.general_updater is None:
            raise RuntimeError("general updater is not registered")
        return self.general_updater
