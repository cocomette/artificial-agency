"""Shared helpers for orchestrator-agent tool-calling backends."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    DecisionResult,
    ExperimentToolInvocationResult,
    NONE_ACTION_ID,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolName,
    ToolResult,
)
from face_of_agi.frames import (
    image_to_base64_png,
    image_to_data_url,
    observation_to_pil_image,
)
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime

INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "system_prompt.md"


class AgentOutputError(RuntimeError):
    """Raised when a provider response violates the X agent contract."""


def load_agent_instructions() -> str:
    """Load the fixed X system prompt."""

    return INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def build_decision_prompt(
    *,
    context: RoleContext,
    first_observation: Observation,
    current_observation: Observation,
    action_space: Sequence[ActionSpec],
    tool_runtime: AgentToolRuntime | None,
) -> str:
    """Build the provider-neutral text sent beside first/current images."""

    tool_metadata = (
        tool_runtime.tool_metadata()
        if tool_runtime is not None
        else {"tools_enabled": False, "available_tools": []}
    )
    visible_refs = (
        [ref_payload(ref) for ref in tool_runtime.available_observation_refs()]
        if tool_runtime is not None
        else []
    )
    prompt_payload = {
        "role_context": context.composed(),
        "first_observation": observation_payload(first_observation),
        "current_observation": observation_payload(current_observation),
        "allowed_actions": [action_payload(action) for action in action_space],
        "visible_observation_refs": visible_refs,
        "tool_policy": tool_metadata,
        "instructions": (
            "Use native tools. Call world/goal only with listed refs and only "
            "when the tool is available. Finish with submit_action."
        ),
    }
    return json.dumps(prompt_payload, indent=2, sort_keys=True)


def observation_images(
    *,
    first_observation: Observation,
    current_observation: Observation,
    frame_scale: int,
) -> tuple[Any, ...]:
    """Return the first/current images to include in provider messages."""

    first_image = observation_to_pil_image(first_observation, frame_scale=frame_scale)
    current_image = observation_to_pil_image(
        current_observation,
        frame_scale=frame_scale,
    )
    if first_observation.id == current_observation.id:
        return (current_image,)
    return (first_image, current_image)


def openai_image_content(images: Sequence[Any], *, detail: str) -> list[dict[str, Any]]:
    """Return OpenAI Responses content items for images."""

    return [
        {
            "type": "input_image",
            "image_url": image_to_data_url(image),
            "detail": detail,
        }
        for image in images
    ]


def ollama_image_payloads(images: Sequence[Any]) -> list[str]:
    """Return base64 PNG images for Ollama chat messages."""

    return [image_to_base64_png(image) for image in images]


def openai_tool_definitions(available_tools: Sequence[ToolName]) -> list[dict[str, Any]]:
    """Return OpenAI Responses function-tool definitions for X."""

    tools = []
    if "world" in available_tools:
        tools.append(_openai_function_tool("world", WORLD_TOOL_DESCRIPTION, _world_schema()))
    if "goal" in available_tools:
        tools.append(_openai_function_tool("goal", GOAL_TOOL_DESCRIPTION, _goal_schema()))
    tools.append(
        _openai_function_tool(
            "submit_action",
            "Submit the final real or synthetic action for this frame.",
            _submit_action_schema(),
        )
    )
    return tools


def ollama_tool_definitions(available_tools: Sequence[ToolName]) -> list[dict[str, Any]]:
    """Return Ollama chat function-tool definitions for X."""

    tools = []
    if "world" in available_tools:
        tools.append(_ollama_function_tool("world", WORLD_TOOL_DESCRIPTION, _world_schema()))
    if "goal" in available_tools:
        tools.append(_ollama_function_tool("goal", GOAL_TOOL_DESCRIPTION, _goal_schema()))
    tools.append(
        _ollama_function_tool(
            "submit_action",
            "Submit the final real or synthetic action for this frame.",
            _submit_action_schema(),
        )
    )
    return tools


def build_tool_call(
    *,
    name: str,
    arguments: Any,
    action_space: Sequence[ActionSpec],
) -> ToolCall:
    """Convert provider tool-call arguments into a local ToolCall."""

    args = parse_arguments(arguments)
    if name == "world":
        return ToolCall(
            tool="world",
            observation_ref=parse_observation_ref(args.get("observation_ref")),
            action=parse_action(args.get("action"), action_space),
        )
    if name == "goal":
        return ToolCall(
            tool="goal",
            observation_ref=parse_observation_ref(args.get("observation_ref")),
        )
    raise AgentOutputError(f"unknown model tool call: {name}")


def build_decision_result(
    *,
    final_action: ActionSpec,
    reasoning_summary: str | None,
    first_observation: Observation,
    current_observation: Observation,
    tool_calls: list[ToolCall],
    tool_results: list[ToolResult],
    metadata: dict[str, Any],
) -> DecisionResult:
    """Build the provider-neutral decision output for X."""

    first_ref = ObservationRef(memory="state", id=first_observation.id)
    current_ref = ObservationRef(memory="state", id=current_observation.id)
    trace = AgentTrace(
        step=current_observation.step,
        first_observation_ref=first_ref,
        current_observation_ref=current_ref,
        final_action=final_action,
        tool_calls=tool_calls,
        tool_results=tool_results,
        reasoning_summary=reasoning_summary,
        metadata=metadata,
    )
    return DecisionResult(final_action=final_action, trace=trace)


def parse_submit_action(
    arguments: Any,
    action_space: Sequence[ActionSpec],
) -> tuple[ActionSpec, str | None]:
    """Parse the terminal submit_action call."""

    args = parse_arguments(arguments)
    final_action = parse_action(args.get("action"), action_space)
    reasoning_summary = args.get("reasoning_summary")
    if reasoning_summary is not None:
        reasoning_summary = str(reasoning_summary)
    return final_action, reasoning_summary


def parse_arguments(arguments: Any) -> dict[str, Any]:
    """Return provider tool-call arguments as a dictionary."""

    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            loaded = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise AgentOutputError(f"tool arguments were not valid JSON: {exc}") from exc
        if isinstance(loaded, dict):
            return loaded
    raise AgentOutputError("tool arguments must be a JSON object")


def parse_observation_ref(value: Any) -> ObservationRef:
    """Parse and validate an observation reference."""

    if not isinstance(value, dict):
        raise AgentOutputError("observation_ref must be an object")
    memory = value.get("memory")
    if memory not in {"state", "experimental"}:
        raise AgentOutputError("observation_ref.memory must be state or experimental")
    ref_id = value.get("id")
    if ref_id is None or str(ref_id) == "":
        raise AgentOutputError("observation_ref.id is required")
    return ObservationRef(memory=memory, id=str(ref_id))


def parse_action(value: Any, action_space: Sequence[ActionSpec]) -> ActionSpec:
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

    if matched.is_complex():
        if data is None:
            raise AgentOutputError("complex actions require action.data")
        _validate_coordinates(data)
    elif data == {}:
        data = None

    return ActionSpec(action_id=matched.action_id, data=data)


def tool_result_feedback(invocation: ExperimentToolInvocationResult) -> dict[str, Any]:
    """Return the minimal JSON payload sent back after a tool call."""

    return {
        "tool": invocation.tool_result.tool,
        "observation_ref": ref_payload(invocation.observation_ref),
    }


def action_payload(action: ActionSpec) -> dict[str, Any]:
    """Return a JSON-friendly action description."""

    return {
        "action_id": action.name,
        "data": action.data,
        "requires_data": action.is_complex(),
    }


def observation_payload(observation: Observation) -> dict[str, Any]:
    """Return text metadata for an observation image."""

    return {
        "id": observation.id,
        "step": observation.step,
        "frame_count": observation.frame_count(),
        "metadata": {
            key: _safe_metadata_value(key, value)
            for key, value in observation.metadata.items()
        },
    }


def ref_payload(ref: ObservationRef) -> dict[str, str]:
    """Return a JSON-friendly observation reference."""

    return asdict(ref)


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


def _match_allowed_action(
    action_id: str,
    action_space: Sequence[ActionSpec],
) -> ActionSpec:
    for candidate in action_space:
        if action_id in {candidate.name, str(candidate.action_id)}:
            return candidate
    allowed = ", ".join(action.name for action in action_space)
    raise AgentOutputError(f"action {action_id!r} is not allowed; allowed: {allowed}")


def _validate_coordinates(data: dict[str, Any]) -> None:
    for key in ("x", "y"):
        if key not in data:
            raise AgentOutputError(f"complex action.data.{key} is required")
        value = data[key]
        if not isinstance(value, int) or not 0 <= value <= 63:
            raise AgentOutputError(f"complex action.data.{key} must be an int 0..63")


def _safe_metadata_value(key: str, value: Any) -> Any:
    if key == "raw_frame_data":
        return repr(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _openai_function_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": True,
    }


def _ollama_function_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


WORLD_TOOL_DESCRIPTION = (
    "Predict a next observation from a memory reference and proposed action."
)
GOAL_TOOL_DESCRIPTION = "Predict or evaluate a goal-relevant observation."


def _observation_ref_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "enum": ["state", "experimental"]},
            "id": {"type": "string"},
        },
        "required": ["memory", "id"],
        "additionalProperties": False,
    }


def _action_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action_id": {"type": "string"},
            "data": {
                "type": ["object", "null"],
                "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 63},
                    "y": {"type": "integer", "minimum": 0, "maximum": 63},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
        "required": ["action_id", "data"],
        "additionalProperties": False,
    }


def _world_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observation_ref": _observation_ref_schema(),
            "action": _action_schema(),
        },
        "required": ["observation_ref", "action"],
        "additionalProperties": False,
    }


def _goal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observation_ref": _observation_ref_schema(),
        },
        "required": ["observation_ref"],
        "additionalProperties": False,
    }


def _submit_action_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": _action_schema(),
            "reasoning_summary": {"type": "string"},
        },
        "required": ["action", "reasoning_summary"],
        "additionalProperties": False,
    }


def none_submit_arguments() -> dict[str, Any]:
    """Return a valid terminal action for no-control animation frames."""

    return {
        "action": {"action_id": NONE_ACTION_ID, "data": None},
        "reasoning_summary": "non-controllable animation frame",
    }
