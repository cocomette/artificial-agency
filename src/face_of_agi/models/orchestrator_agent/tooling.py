"""Shared helpers for orchestrator-agent tool-calling backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from face_of_agi.contracts import (
    ActionSpec,
    ActionHistoryItem,
    AgentTrace,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
    VisualCoordinateSpace,
)
from face_of_agi.frames import observation_to_pil_image, to_memory_jsonable
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.arc_grid_crop import normalized_1000_to_arc_grid
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime

INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "system_prompt.md"


class AgentOutputError(RuntimeError):
    """Raised when a provider response violates the X agent contract."""


def load_agent_instructions() -> str:
    """Load the fixed X system prompt."""

    return INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def build_agent_instructions(
    *,
    allowed_actions: Sequence[ActionSpec],
) -> str:
    """Return X instructions with the current allowed-action glossary."""

    return append_action_glossary(
        load_agent_instructions(),
        allowed_actions,
        mode="agent_decision",
    )


def build_decision_prompt(
    *,
    context: RoleContext,
    action_space: Sequence[ActionSpec],
    recent_action_history: Sequence[ActionHistoryItem] = (),
    recent_action_history_available: bool = True,
) -> str:
    """Build the provider-neutral Markdown text sent beside X images."""

    return "\n\n".join(
        [
            "## Game context\n\n" + _text_or_none(context.composed()),
            "## Allowed actions\n\n" + _allowed_actions_text(action_space),
            "## Recent actions\n\n"
            + _recent_actions_text(
                recent_action_history,
                available=recent_action_history_available,
            ),
        ]
    )


def observation_images(
    *,
    current_observation: Observation,
) -> tuple[Any, ...]:
    """Return the current observation image for X."""

    return (
        observation_to_pil_image(current_observation),
    )


def final_action_schema(action_space: Sequence[ActionSpec]) -> dict[str, Any]:
    """Return the structured final-action schema for one X frame turn."""

    simple_actions = [action for action in action_space if not action.is_complex()]
    complex_actions = [action for action in action_space if action.is_complex()]
    return {
        "type": "object",
        "properties": {
            "action": _action_output_schema(
                simple_actions=simple_actions,
                complex_actions=complex_actions,
            ),
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def _action_output_schema(
    *,
    simple_actions: Sequence[ActionSpec],
    complex_actions: Sequence[ActionSpec],
) -> dict[str, Any]:
    """Return the final action branch schema for simple/complex ARC actions."""

    branches: list[dict[str, Any]] = []
    if simple_actions:
        branches.append(
            _action_object_schema(
                action_ids=[action.name for action in simple_actions],
                include_data=False,
            )
        )
    if complex_actions:
        branches.append(
            _action_object_schema(
                action_ids=[action.name for action in complex_actions],
                include_data=True,
            )
        )
    if len(branches) == 1:
        return branches[0]
    return {"anyOf": branches}


def _action_object_schema(
    *,
    action_ids: Sequence[str],
    include_data: bool,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action_id": {
            "type": "string",
            "enum": list(action_ids),
        },
    }
    required = ["action_id"]
    if include_data:
        properties["data"] = {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        }
        properties["target"] = {
            "type": "string",
            "description": (
                "Concise visual description of the object or area targeted by "
                "these coordinates."
            ),
        }
        required.extend(["data", "target"])
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def openai_final_action_text_format(
    action_space: Sequence[ActionSpec],
) -> dict[str, Any]:
    """Return OpenAI Responses structured-output config for final X action."""

    return {
        "format": {
            "type": "json_schema",
            "name": "agent_final_action",
            "strict": True,
            "schema": final_action_schema(action_space),
        }
    }


def final_action_repair_prompt(
    action_space: Sequence[ActionSpec],
    *,
    validation_error: str,
    invalid_text: str | None,
    attempt: int,
) -> str:
    """Return a provider-neutral final-action repair request."""

    allowed = ", ".join(action.name for action in action_space)
    repair_parts = [
        f"Repair attempt {attempt}: the previous Agent X output was invalid.",
        "Validation error:\n" + validation_error,
    ]
    if invalid_text is not None:
        repair_parts.append("Invalid output:\n" + invalid_text)
    repair_parts.extend(
        [
            "Return only corrected final action JSON for one allowed final action. "
            "The top-level JSON must contain exactly one field named `action`. "
            "The `action` field value must be an object, never a string. "
            "The action object must contain `action_id`; simple actions must not "
            "include `data`; ACTION6 must include a `data` object with numeric "
            "`x` and `y` in normalized visual 0..1000 coordinates and a "
            "non-empty `target` string describing the targeted object or area. "
            "Do not include prose.",
            f"Allowed final actions: {allowed}.",
        ]
    )
    return "\n\n".join(repair_parts)


def build_decision_result(
    *,
    final_action: ActionSpec,
    current_observation: Observation,
    first_observation_ref: ObservationRef | None,
    tool_calls: list[ToolCall],
    tool_results: list[ToolResult],
    metadata: dict[str, Any],
) -> DecisionResult:
    """Build the provider-neutral decision output for X."""

    current_ref = ObservationRef(memory="state", id=current_observation.id)
    trace = AgentTrace(
        step=current_observation.step,
        first_observation_ref=first_observation_ref or current_ref,
        current_observation_ref=current_ref,
        final_action=final_action,
        tool_calls=tool_calls,
        tool_results=tool_results,
        metadata=metadata,
    )
    return DecisionResult(final_action=final_action, trace=trace)


def parse_final_action(
    arguments: Any,
    action_space: Sequence[ActionSpec],
    coordinate_space: VisualCoordinateSpace = "normalized_1000",
) -> ActionSpec:
    """Parse the terminal structured final-action payload."""

    args = parse_arguments(arguments)
    return parse_action(
        args.get("action"),
        action_space,
        coordinate_space=coordinate_space,
    )


def parse_arguments(arguments: Any) -> dict[str, Any]:
    """Return provider tool-call arguments as a dictionary."""

    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            loaded = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise AgentOutputError(
                f"tool arguments were not valid JSON: {exc}"
            ) from exc
        if isinstance(loaded, dict):
            return loaded
    raise AgentOutputError("tool arguments must be a JSON object")


def parse_action(
    value: Any,
    action_space: Sequence[ActionSpec],
    *,
    coordinate_space: VisualCoordinateSpace = "normalized_1000",
    arc_grid_crop_edges: object | None = None,
) -> ActionSpec:
    """Parse and validate a provider action against the current action space."""

    if not isinstance(value, dict):
        raise AgentOutputError("action must be an object")

    raw_action_id = value.get("action_id")
    if raw_action_id is None:
        raise AgentOutputError("action.action_id is required")

    matched = _match_allowed_action(str(raw_action_id), action_space)
    data = value.get("data")
    if data is not None and not isinstance(data, dict):
        raise AgentOutputError("action.data must be an object when provided")
    target = value.get("target")
    if target is not None and not isinstance(target, str):
        raise AgentOutputError("action.target must be a string when provided")

    if matched.is_complex():
        target = _action_target(target)
        if matched.name == "ACTION6" and (
            "bbox" in value or "target_rgb_color" in value
        ):
            data = _action6_targeting_data(value)
        else:
            if data is None:
                raise AgentOutputError("complex actions require action.data")
            data = _normalized_action_data(
                data,
                coordinate_space=coordinate_space,
                arc_grid_crop_edges=arc_grid_crop_edges,
            )
    elif data is not None:
        raise AgentOutputError("simple actions must not include action.data")
    elif target is not None:
        raise AgentOutputError("simple actions must not include action.target")

    return ActionSpec(action_id=matched.action_id, data=data, target=target)


def tool_result_feedback(invocation: ExperimentToolInvocationResult) -> dict[str, Any]:
    """Return the JSON payload sent back after a tool call."""

    return {
        "tool": invocation.tool_result.tool,
        "output": to_memory_jsonable(invocation.tool_result.output),
        "explanation": invocation.tool_result.explanation,
    }


def object_get(value: Any, key: str, default: Any = None) -> Any:
    """Read a key from SDK objects or dictionaries."""

    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def function_call_name_and_arguments(call: Any) -> tuple[str, Any]:
    """Read one Ollama-style function call."""

    function = object_get(call, "function", {})
    name = object_get(function, "name")
    arguments = object_get(function, "arguments", {})
    if not name:
        raise AgentOutputError("tool call did not include a function name")
    return str(name), arguments


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _allowed_actions_text(action_space: Sequence[ActionSpec]) -> str:
    if not action_space:
        return "none"
    return "\n".join(f"- {_action_text(action)}" for action in action_space)


def _recent_actions_text(
    history: Sequence[ActionHistoryItem],
    *,
    available: bool,
) -> str:
    if not available:
        return "not available"
    if not history:
        return "none"
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text,
        numbered=True,
    )


def _action_text(action: ActionSpec) -> str:
    if action.is_complex() and not action.data:
        return f"{action.name}(x,y normalized_0_1000,target)"
    if action.data:
        return f"{action.name} {json.dumps(action.data, sort_keys=True)}"
    return action.name


def _action_target(target: str | None) -> str:
    if target is None or not target.strip():
        raise AgentOutputError("ACTION6 requires non-empty action.target")
    return target.strip()


def _match_allowed_action(
    action_id: str,
    action_space: Sequence[ActionSpec],
) -> ActionSpec:
    for candidate in action_space:
        if action_id in {candidate.name, str(candidate.action_id)}:
            return candidate
    allowed = ", ".join(action.name for action in action_space)
    raise AgentOutputError(f"action {action_id!r} is not allowed; allowed: {allowed}")


def _normalized_action_data(
    data: dict[str, Any],
    *,
    coordinate_space: VisualCoordinateSpace,
    arc_grid_crop_edges: object | None = None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in ("x", "y"):
        if key not in data:
            raise AgentOutputError(f"complex action.data.{key} is required")
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentOutputError(f"complex action.data.{key} must be numeric")
        numeric = float(value)
        if coordinate_space == "normalized_1000":
            if not 0 <= numeric <= 1000:
                raise AgentOutputError(
                    f"complex action.data.{key} must be in normalized 0..1000"
                )
            normalized[key] = normalized_1000_to_arc_grid(
                numeric,
                key,
                crop_edges=arc_grid_crop_edges,
            )
            continue
        raise AgentOutputError(
            "pixel visual coordinates cannot be converted to ARC coordinates "
            "without an image size; use a normalized_1000 model profile"
        )
    return normalized


def _action6_targeting_data(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": _normalized_action6_bbox(value.get("bbox")),
        "target_rgb_color": _normalized_action6_rgb(value.get("target_rgb_color")),
    }


def _normalized_action6_bbox(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise AgentOutputError("ACTION6 bbox must be [x0, y0, x1, y1]")
    normalized: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AgentOutputError(f"ACTION6 bbox[{index}] must be numeric")
        numeric = float(item)
        if not 0 <= numeric <= 1000:
            raise AgentOutputError("ACTION6 bbox values must be in normalized 0..1000")
        normalized.append(round(numeric))
    x0, y0, x1, y1 = normalized
    if x0 >= x1 or y0 >= y1:
        raise AgentOutputError("ACTION6 bbox must have positive width and height")
    return normalized


def _normalized_action6_rgb(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise AgentOutputError("ACTION6 target_rgb_color must be [r, g, b]")
    normalized: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise AgentOutputError(
                f"ACTION6 target_rgb_color[{index}] must be an integer"
            )
        if not 0 <= item <= 255:
            raise AgentOutputError("ACTION6 target_rgb_color values must be 0..255")
        normalized.append(item)
    return normalized
