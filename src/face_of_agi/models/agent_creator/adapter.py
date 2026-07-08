"""Provider-neutral adapters for the agent creator roles."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import re
from typing import Any

from face_of_agi.agent_creator.contracts import AgentRoleDefinition
from face_of_agi.agent_creator.store import validate_role_definitions
from face_of_agi.frames import observation_to_pil_image, to_memory_jsonable
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.agent_creator.config import AgentCreatorConfig
from face_of_agi.models.agent_creator.contracts import (
    AgentCreatorInput,
    AgentCreatorProviderResponse,
    CreatorMutation,
    CreatorMutationPlan,
    CreatorOrchestratorResponse,
    CreatorOrchestratorRequest,
    PromptAgentCreatorImage,
    PromptAgentCreatorProvider,
    RoleAuthorInput,
    RoleAuthorRequest,
    agent_creator_roles_json_schema,
    creator_orchestrator_plan_json_schema,
    agent_role_json_schema,
    role_instructions_json_schema,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)
from face_of_agi.models.arc_grid_crop import (
    crop_image_arc_grid_edges,
    normalize_arc_grid_crop_edges,
)
from face_of_agi.models.image_inputs import resize_image

INSTRUCTION_DIR = Path(__file__).parent / "instructions"
DEFAULT_INSTRUCTION_PATH = INSTRUCTION_DIR / "instruction_prompt.md"
DEFAULT_ROLE_ADD_INSTRUCTION_PATH = INSTRUCTION_DIR / "role_add_prompt.md"
DEFAULT_ROLE_UPDATE_INSTRUCTION_PATH = INSTRUCTION_DIR / "role_update_prompt.md"
LOGGER = logging.getLogger(__name__)


class AgentCreatorOutputError(RuntimeError):
    """Raised when the agent creator returns invalid structured output."""


class AgentCreatorAdapter:
    """Provider-neutral creator orchestrator and role author."""

    def __init__(
        self,
        provider: PromptAgentCreatorProvider,
        config: AgentCreatorConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or AgentCreatorConfig()
        self._arc_grid_crop_edges = normalize_arc_grid_crop_edges(
            self.config.input_image_crop_arc_grid_edges
        )

    def run_creator(
        self,
        creator_input: AgentCreatorInput,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        """Run the creator orchestrator planning pass."""

        output_schema = agent_creator_orchestrator_output_schema(max_tool_calls)
        request = CreatorOrchestratorRequest(
            instructions=append_output_schema_to_instructions(
                load_agent_creator_instructions(self.config.instruction_path),
                output_schema,
                include=self.config.include_output_schema_in_instructions,
            ),
            text=json.dumps(
                _creator_orchestrator_payload(creator_input),
                ensure_ascii=False,
            ),
            tools=(),
            images=_creator_orchestrator_images(
                creator_input,
                crop_edges=self._arc_grid_crop_edges,
                size=self.config.input_image_size,
                resample=self.config.input_image_resample,
            ),
            metadata={
                "backend": self.provider.backend,
                "model": self.provider.model,
                "batch_size": len(creator_input.batch_items),
                "max_mutations": max_tool_calls,
            },
        )
        return self.provider.run_orchestrator(
            request,
            max_tool_calls=max_tool_calls,
        )

    def create_role_instructions(self, author_input: RoleAuthorInput) -> str:
        """Create role-specific instructions."""

        return self._author_role(
            author_input,
            instruction_path=DEFAULT_ROLE_ADD_INSTRUCTION_PATH,
            operation="add",
        )

    def update_role_instructions(self, author_input: RoleAuthorInput) -> str:
        """Update role-specific instructions."""

        return self._author_role(
            author_input,
            instruction_path=DEFAULT_ROLE_UPDATE_INSTRUCTION_PATH,
            operation="update",
        )

    def _author_role(
        self,
        author_input: RoleAuthorInput,
        *,
        instruction_path: Path,
        operation: str,
    ) -> str:
        output_schema = role_instructions_json_schema()
        instructions = append_output_schema_to_instructions(
            instruction_path.read_text(encoding="utf-8").strip(),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = RoleAuthorRequest(
            instructions=instructions,
            text=json.dumps(
                _role_author_payload(author_input, operation=operation),
                ensure_ascii=False,
            ),
            output_schema=output_schema,
            metadata={
                "backend": self.provider.backend,
                "model": self.provider.model,
                "operation": operation,
                "role_name": author_input.role_name,
            },
        )
        response = self.provider.author_role(request)
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} agent creator role author",
                response=response,
                text_of=lambda item: item.text,
                validate=parse_role_instructions_output,
                repair=provider_repair_callback(
                    self.provider,
                    "repair_role",
                    args=(request,),
                ),
                max_repair_attempts=self.config.repair_attempts,
                error_factory=AgentCreatorOutputError,
            )
        except AgentCreatorOutputError:
            LOGGER.error(
                "agent creator role-author structured output repair exhausted; "
                "backend=%s model=%s operation=%s role=%s",
                self.provider.backend,
                self.provider.model,
                operation,
                author_input.role_name,
                exc_info=True,
            )
            raise
        return validated.value

    def update_roles(
        self,
        creator_input: AgentCreatorInput,
    ) -> tuple[AgentRoleDefinition, ...]:
        """Compatibility API; use run_creator for the active workflow."""

        del creator_input
        raise NotImplementedError(
            "agent creator now updates roles through mutation plans"
        )


def load_agent_creator_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable creator-orchestrator instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def agent_creator_orchestrator_output_schema(
    max_mutations: int = 4,
) -> dict[str, Any]:
    """Return the structured decision schema for creator-orchestrator calls."""

    return creator_orchestrator_plan_json_schema(max_mutations=max_mutations)


def parse_agent_creator_role_output(text: str) -> AgentRoleDefinition:
    """Parse and validate one role definition JSON contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise AgentCreatorOutputError(
            "agent creator role response must be JSON; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise AgentCreatorOutputError(
            "agent creator role response must be a JSON object"
        )
    role = _role_from_output(loaded)
    try:
        validate_role_definitions((role,))
    except ValueError as exc:
        raise AgentCreatorOutputError(str(exc)) from exc
    return role


def parse_role_instructions_output(text: str) -> str:
    """Parse and validate one role-instructions JSON contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise AgentCreatorOutputError(
            "agent creator role-instructions response must be JSON; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise AgentCreatorOutputError(
            "agent creator role-instructions response must be a JSON object"
        )
    instructions = loaded.get("role_instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise AgentCreatorOutputError(
            "agent creator role-instructions response JSON is missing "
            "non-empty string field 'role_instructions'"
        )
    return instructions


def parse_creator_orchestrator_plan_output(
    text: str,
    *,
    max_mutations: int = 4,
) -> CreatorMutationPlan:
    """Parse and validate one creator-orchestrator mutation plan."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise AgentCreatorOutputError(
            "agent creator orchestrator response must be JSON; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise AgentCreatorOutputError(
            "agent creator orchestrator response must be a JSON object"
        )

    extra = set(loaded) - {"mutations"}
    if extra:
        extra_text = ", ".join(sorted(extra))
        raise AgentCreatorOutputError(
            f"agent creator orchestrator response has unexpected keys: {extra_text}"
        )

    raw_mutations = loaded.get("mutations")
    if not isinstance(raw_mutations, list):
        raise AgentCreatorOutputError(
            "agent creator orchestrator response must include a mutations array"
        )
    if len(raw_mutations) > max_mutations:
        raise AgentCreatorOutputError(
            "agent creator orchestrator response includes too many mutations: "
            f"{len(raw_mutations)} > {max_mutations}"
        )

    mutations: list[CreatorMutation] = []
    seen_roles: set[str] = set()
    for index, raw_mutation in enumerate(raw_mutations, start=1):
        mutations.append(_parse_creator_mutation(raw_mutation, index=index))
        if mutations[-1].role_name in seen_roles:
            raise AgentCreatorOutputError(
                "agent creator orchestrator response includes more than one "
                f"mutation for role {mutations[-1].role_name!r}"
            )
        seen_roles.add(mutations[-1].role_name)

    return CreatorMutationPlan(mutations=tuple(mutations))


def parse_creator_orchestrator_step_output(text: str) -> CreatorMutation:
    """Compatibility parser for one legacy creator mutation decision."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise AgentCreatorOutputError(
            "agent creator orchestrator response must be JSON; "
            f"raw response preview: {preview!r}"
        ) from exc
    plan = parse_creator_orchestrator_plan_output(
        json.dumps({"mutations": [loaded]}),
        max_mutations=1,
    )
    return plan.mutations[0]


def _parse_creator_mutation(value: Any, *, index: int) -> CreatorMutation:
    if not isinstance(value, dict):
        raise AgentCreatorOutputError(
            f"agent creator orchestrator mutation {index} must be an object"
        )
    expected = {
        "action",
        "role_name",
        "instruction_guidance",
        "identified_failures",
        "meta_description",
    }
    extra = set(value) - expected
    if extra:
        extra_text = ", ".join(sorted(extra))
        raise AgentCreatorOutputError(
            "agent creator orchestrator mutation "
            f"{index} has unexpected keys: {extra_text}"
        )

    values: dict[str, str] = {}
    for key in expected:
        raw = value.get(key, "")
        if not isinstance(raw, str):
            raise AgentCreatorOutputError(
                "agent creator orchestrator mutation "
                f"{index} field {key!r} must be a string"
            )
        values[key] = raw.strip()

    action = values["action"]
    if action not in {"delete", "add", "update"}:
        raise AgentCreatorOutputError(
            "agent creator orchestrator response field 'action' must be one of "
            "delete, add, update"
        )
    if not values["role_name"]:
        raise AgentCreatorOutputError(
            f"agent creator orchestrator action {action!r} requires role_name"
        )
    if action == "add":
        for key in ("instruction_guidance", "meta_description"):
            if not values[key]:
                raise AgentCreatorOutputError(
                    f"agent creator orchestrator action 'add' requires {key}"
                )
    if action == "update" and not values["identified_failures"]:
        raise AgentCreatorOutputError(
            "agent creator orchestrator action 'update' requires identified_failures"
        )
    return CreatorMutation(
        action=action,
        role_name=values["role_name"],
        instruction_guidance=values["instruction_guidance"] if action == "add" else "",
        identified_failures=(
            values["identified_failures"] if action == "update" else ""
        ),
        meta_description=values["meta_description"] if action != "delete" else "",
    )


def creator_mutation_tool_call(
    mutation: CreatorMutation,
) -> tuple[str, dict[str, Any]] | None:
    """Convert a structured creator mutation into an executable call."""

    if mutation.action == "delete":
        return "delete", {"role_name": mutation.role_name}
    if mutation.action == "add":
        return (
            "add",
            {
                "role_name": mutation.role_name,
                "instruction_guidance": mutation.instruction_guidance,
                "meta_description": mutation.meta_description,
            },
        )
    if mutation.action == "update":
        arguments = {
            "role_name": mutation.role_name,
            "identified_failures": mutation.identified_failures,
        }
        if mutation.meta_description:
            arguments["meta_description"] = mutation.meta_description
        return "update", arguments
    raise AgentCreatorOutputError(
        f"unsupported agent creator orchestrator action: {mutation.action!r}"
    )


def creator_orchestrator_step_tool_call(
    step: CreatorMutation,
) -> tuple[str, dict[str, Any]] | None:
    """Compatibility wrapper for legacy single-mutation callers."""

    return creator_mutation_tool_call(step)


def parse_agent_creator_roles_output(text: str) -> tuple[AgentRoleDefinition, ...]:
    """Parse the legacy complete-role-array JSON contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise AgentCreatorOutputError(
            "agent creator response must be JSON; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise AgentCreatorOutputError("agent creator response must be a JSON object")
    roles_value = loaded.get("roles")
    if not isinstance(roles_value, list):
        raise AgentCreatorOutputError(
            "agent creator response JSON is missing array field 'roles'"
        )
    roles = tuple(_role_from_output(item) for item in roles_value)
    try:
        validate_role_definitions(roles)
    except ValueError as exc:
        raise AgentCreatorOutputError(str(exc)) from exc
    return roles


def _role_from_output(value: Any) -> AgentRoleDefinition:
    if not isinstance(value, dict):
        raise AgentCreatorOutputError("agent creator role entries must be objects")
    role = value.get("role")
    meta_description = value.get("meta_description")
    role_instructions = value.get("role_instructions")
    if not isinstance(role, str):
        raise AgentCreatorOutputError("agent creator role.role must be a string")
    if not isinstance(meta_description, str):
        raise AgentCreatorOutputError(
            "agent creator role.meta_description must be a string"
        )
    if not isinstance(role_instructions, str):
        raise AgentCreatorOutputError(
            "agent creator role.role_instructions must be a string"
        )
    return AgentRoleDefinition(
        role=role,
        meta_description=meta_description,
        role_instructions=role_instructions,
    )


def _creator_orchestrator_payload(creator_input: AgentCreatorInput) -> dict[str, Any]:
    return {
        "available_roles": [
            {
                "role": role.role,
                "meta_description": role.meta_description,
            }
            for role in creator_input.current_roles
        ],
        "current_frames": (
            "One current-frame image is attached for each batch item in the same "
            "order as batch_items."
        ),
        "batch_items": [
            {
                "current_frame": f"attached image {index}",
                "world_model_context": to_memory_jsonable(
                    item.world_model_context
                ),
                "strategy_history": to_memory_jsonable(item.strategy_history),
                "action_history": grouped_action_history_text(
                    item.action_history,
                    action_text=model_facing_action_text,
                    numbered=True,
                ),
            }
            for index, item in enumerate(creator_input.batch_items, start=1)
        ],
        "metadata": {
            key: value
            for key, value in to_memory_jsonable(creator_input.metadata).items()
            if key == "max_roles"
        },
    }


def _creator_orchestrator_images(
    creator_input: AgentCreatorInput,
    *,
    crop_edges: tuple[int, int, int, int],
    size: str | tuple[int, int] | None,
    resample: str,
) -> tuple[PromptAgentCreatorImage, ...]:
    images: list[PromptAgentCreatorImage] = []
    for index, item in enumerate(creator_input.batch_items, start=1):
        try:
            image = crop_image_arc_grid_edges(
                observation_to_pil_image(item.current_observation),
                crop_edges,
            )
        except Exception:
            LOGGER.warning(
                "agent creator skipped unavailable current-frame image for "
                "batch item %s",
                index,
                exc_info=True,
            )
            continue
        images.append(
            PromptAgentCreatorImage(
                label=f"batch_item_{index}_current_frame",
                image=resize_image(image, size=size, resample=resample),
            )
        )
    return tuple(images)


def _role_author_payload(
    author_input: RoleAuthorInput,
    *,
    operation: str,
) -> dict[str, Any]:
    if operation == "add":
        return _role_add_author_payload(author_input)
    if operation == "update":
        return _role_update_author_payload(author_input)
    raise ValueError(f"unsupported role-author operation: {operation}")


def _role_add_author_payload(author_input: RoleAuthorInput) -> dict[str, Any]:
    return {
        "role_name": author_input.role_name,
        "instruction_guidance": author_input.instruction_guidance,
        "general_system_prompt": author_input.general_system_prompt,
    }


def _role_update_author_payload(author_input: RoleAuthorInput) -> dict[str, Any]:
    return {
        "role_name": author_input.role_name,
        "current_role": (
            asdict(author_input.current_role)
            if author_input.current_role is not None
            else None
        ),
        "identified_failures": author_input.identified_failures,
        "general_system_prompt": author_input.general_system_prompt,
    }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return stripped
    return match.group(1).strip()
