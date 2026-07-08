"""Contracts for the agent creator model roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.agent_creator.contracts import (
    AgentCreatorBatchItem,
    AgentRoleDefinition,
)

CREATOR_ORCHESTRATOR_ACTIONS = ("delete", "add", "update")


def agent_creator_tool_schemas() -> tuple[dict[str, Any], ...]:
    """Compatibility schemas for the former creator tool interface."""

    return (
        _tool_schema(
            "delete",
            "Deactivate an existing role that is not useful.",
            {
                "role_name": {
                    "type": "string",
                    "description": "Exact active role name to deactivate.",
                },
            },
        ),
        _tool_schema(
            "add",
            "Create a new role for a recurring failure mode.",
            {
                "role_name": {
                    "type": "string",
                    "description": "Short stable role name to create.",
                },
                "instruction_guidance": {
                    "type": "string",
                    "description": (
                        "Behavioral guidance for the role instructions."
                    ),
                },
                "meta_description": {
                    "type": "string",
                    "description": (
                        "Concise role summary used by the historizer when "
                        "selecting this role, including when it should be "
                        "called."
                    ),
                },
            },
        ),
        _tool_schema(
            "update",
            "Revise an existing role based on identified failures.",
            {
                "role_name": {
                    "type": "string",
                    "description": "Exact active role name to update.",
                },
                "identified_failures": {
                    "type": "string",
                    "description": (
                        "Failures identified for this role and how the role "
                        "should solve them."
                    ),
                },
                "meta_description": {
                    "type": "string",
                    "description": (
                        "Optional revised concise role summary used by the "
                        "historizer when selecting this role."
                    ),
                },
            },
            required=("role_name", "identified_failures"),
            strict=False,
        ),
    )


def creator_orchestrator_plan_json_schema(max_mutations: int = 4) -> dict[str, Any]:
    """Return the structured schema for one creator mutation plan."""

    return {
        "type": "object",
        "properties": {
            "mutations": {
                "type": "array",
                "maxItems": max_mutations,
                "items": {
                    "anyOf": [
                        _delete_mutation_json_schema(),
                        _add_mutation_json_schema(),
                        _update_mutation_json_schema(),
                    ],
                },
            },
        },
        "required": ["mutations"],
        "additionalProperties": False,
    }


def _delete_mutation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["delete"]},
            "role_name": {"type": "string"},
        },
        "required": ["action", "role_name"],
        "additionalProperties": False,
    }


def _add_mutation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add"]},
            "role_name": {"type": "string"},
            "instruction_guidance": {"type": "string"},
            "meta_description": {"type": "string"},
        },
        "required": [
            "action",
            "role_name",
            "instruction_guidance",
            "meta_description",
        ],
        "additionalProperties": False,
    }


def _update_mutation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["update"]},
            "role_name": {"type": "string"},
            "identified_failures": {"type": "string"},
            "meta_description": {"type": "string"},
        },
        "required": ["action", "role_name", "identified_failures"],
        "additionalProperties": False,
    }


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: tuple[str, ...] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required or properties),
                "additionalProperties": False,
            },
            "strict": strict,
        },
    }


def role_instructions_json_schema() -> dict[str, Any]:
    """Return the structured output schema for role instructions only."""

    return {
        "type": "object",
        "properties": {
            "role_instructions": {"type": "string"},
        },
        "required": ["role_instructions"],
        "additionalProperties": False,
    }


def agent_role_json_schema() -> dict[str, Any]:
    """Return the structured output schema for one role definition."""

    return {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "meta_description": {"type": "string"},
            "role_instructions": {"type": "string"},
        },
        "required": ["role", "meta_description", "role_instructions"],
        "additionalProperties": False,
    }


def agent_creator_roles_json_schema() -> dict[str, Any]:
    """Compatibility schema for complete role arrays."""

    return {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "minItems": 1,
                "items": agent_role_json_schema(),
            },
        },
        "required": ["roles"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class AgentCreatorInput:
    """Input for one full role-set update workflow."""

    batch_items: tuple[AgentCreatorBatchItem, ...]
    current_roles: tuple[AgentRoleDefinition, ...]
    general_system_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CreatorOrchestratorRequest:
    """Provider-neutral creator-orchestrator request."""

    instructions: str
    text: str
    tools: tuple[dict[str, Any], ...]
    images: tuple["PromptAgentCreatorImage", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CreatorMutation:
    """One planned creator role mutation."""

    action: str
    role_name: str
    instruction_guidance: str = ""
    identified_failures: str = ""
    meta_description: str = ""


@dataclass(frozen=True, slots=True)
class CreatorMutationPlan:
    """A bounded set of role mutations selected in one creator pass."""

    mutations: tuple[CreatorMutation, ...]


@dataclass(slots=True)
class PromptAgentCreatorImage:
    """Provider-neutral image attached to an agent-creator request."""

    label: str
    image: Any


@dataclass(slots=True)
class CreatorOrchestratorResponse:
    """Provider response metadata for one creator mutation plan."""

    text: str = ""
    tool_call_count: int = 0
    mutations: tuple[CreatorMutation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoleAuthorInput:
    """Input for adding or updating role instructions."""

    role_name: str
    general_system_prompt: str
    instruction_guidance: str = ""
    identified_failures: str = ""
    current_role: AgentRoleDefinition | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoleAuthorRequest:
    """Provider-neutral role-authoring request."""

    instructions: str
    text: str
    output_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCreatorProviderResponse:
    """Raw provider output for one agent-creator request."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


AgentCreatorRequest = RoleAuthorRequest


class RoleMutationToolExecutor(Protocol):
    """Tool executor used by the runtime to apply creator mutation plans."""

    def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute one creator tool call and return JSON content."""
        ...

    def execute_plan(self, mutations: tuple[CreatorMutation, ...]) -> tuple[str, ...]:
        """Execute one creator mutation plan and return compact JSON results."""
        ...


class PromptAgentCreatorProvider(Protocol):
    """Thin backend boundary for agent-creator model calls."""

    backend: str
    model: str | None

    def run_orchestrator(
        self,
        request: CreatorOrchestratorRequest,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        """Return one creator mutation plan."""
        ...

    def author_role(
        self,
        request: RoleAuthorRequest,
    ) -> AgentCreatorProviderResponse:
        """Return raw provider text for one role definition."""
        ...

    def repair_role(
        self,
        request: RoleAuthorRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> AgentCreatorProviderResponse:
        """Return repaired raw provider text."""
        ...


class CreatorOrchestratorModel(Protocol):
    """Model role that returns role mutation plans."""

    def run_creator(
        self,
        creator_input: AgentCreatorInput,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        """Run creator analysis and return a mutation plan."""
        ...


class RoleAuthorModel(Protocol):
    """Model role that writes role-specific instructions."""

    def create_role_instructions(self, author_input: RoleAuthorInput) -> str:
        """Create role instructions."""
        ...

    def update_role_instructions(self, author_input: RoleAuthorInput) -> str:
        """Update role instructions."""
        ...


AgentCreatorModel = CreatorOrchestratorModel
