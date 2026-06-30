"""Shared helpers for orchestrator-agent tool-calling backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionOutcomeEvidence,
    ActionSpec,
    AgentTrace,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.frames import to_memory_jsonable
from face_of_agi.models.color_glossary import append_arc_color_glossary
from face_of_agi.models.action_coordinates import (
    action6_coordinate_bounds,
    action6_coordinate_range_phrase,
    action6_coordinate_range_text,
    action6_data_from_visible_crop,
)
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime
from face_of_agi.models.observation_text import (
    ObservationTextConfig,
    serialize_observation,
)

INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "system_prompt.md"


class AgentOutputError(RuntimeError):
    """Raised when a provider response violates the X agent contract."""


def load_agent_instructions() -> str:
    """Load the fixed X system prompt."""

    return INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def build_agent_instructions(
    *,
    glossary_actions: Sequence[ActionSpec],
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    """Return X instructions with the current raw action glossary."""

    return append_arc_color_glossary(
        append_action_glossary(
            load_agent_instructions(),
            glossary_actions,
            mode="agent_decision",
            observation_text_config=observation_text_config,
        )
    )


def build_decision_prompt(
    *,
    context: RoleContext,
    current_observation: Observation,
    action_space: Sequence[ActionSpec],
    recent_action_history: Sequence[ActionHistoryItem] = (),
    recent_action_history_available: bool = True,
    action_outcome_evidence: ActionOutcomeEvidence | None = None,
    observation_text_config: ObservationTextConfig | None = None,
) -> str:
    """Build the provider-neutral Markdown text sent to X."""

    parts = [
        "## Agent context\n\n" + _text_or_none(context.composed()),
        "## Current observation\n\n"
        + serialize_observation(
            current_observation,
            config=observation_text_config,
            label="current_observation",
            include_header_metadata=False,
        ).text,
        "## Allowed actions\n\n"
        + _allowed_actions_text(
            action_space,
            observation_text_config=observation_text_config,
        ),
    ]
    suppression_evidence = _action_suppression_evidence_text(
        action_outcome_evidence
    )
    if suppression_evidence:
        parts.append("## Action suppression evidence\n\n" + suppression_evidence)
    parts.append(
        "## Recent actions\n\n"
        + _recent_actions_text(
            recent_action_history,
            available=recent_action_history_available,
            observation_text_config=observation_text_config,
        )
    )
    return "\n\n".join(parts)


def final_action_schema(
    action_space: Sequence[ActionSpec],
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the structured final-action schema for one X frame turn."""

    simple_actions = [action for action in action_space if not action.is_complex()]
    complex_actions = [action for action in action_space if action.is_complex()]
    return {
        "type": "object",
        "properties": {
            "action": _action_output_schema(
                simple_actions=simple_actions,
                complex_actions=complex_actions,
                observation_text_config=observation_text_config,
            ),
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def _action_output_schema(
    *,
    simple_actions: Sequence[ActionSpec],
    complex_actions: Sequence[ActionSpec],
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
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
                observation_text_config=observation_text_config,
            )
        )
    if len(branches) == 1:
        return branches[0]
    return {"anyOf": branches}


def _action_object_schema(
    *,
    action_ids: Sequence[str],
    include_data: bool,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action_id": {
            "type": "string",
            "enum": list(action_ids),
        },
    }
    required = ["action_id"]
    if include_data:
        minimum, maximum = action6_coordinate_bounds(observation_text_config)
        properties["data"] = {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": minimum, "maximum": maximum},
                "y": {"type": "number", "minimum": minimum, "maximum": maximum},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        }
        properties["target"] = {
            "type": "string",
            "minLength": 1,
            "description": (
                "Concise text description of the visible object, cell, or "
                "region targeted by these ACTION6 coordinates."
            ),
        }
        required.extend(["data", "target"])
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def final_action_repair_prompt(
    action_space: Sequence[ActionSpec],
    *,
    validation_error: str,
    invalid_text: str | None,
    attempt: int,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    """Return a provider-neutral final-action repair request."""

    allowed = ", ".join(action.name for action in action_space)
    action6_range = action6_coordinate_range_phrase(observation_text_config)
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
            "include `data`; ACTION6 must include a `data` object with integer "
            f"`x` and `y` visible cropped coordinates from {action6_range}. "
            "ACTION6 must also include a non-empty top-level `target` string "
            "describing the object, cell, or region targeted by those coordinates. "
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
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> ActionSpec:
    """Parse the terminal structured final-action payload."""

    args = parse_arguments(arguments)
    return parse_action(
        args.get("action"),
        action_space,
        observation_text_config=observation_text_config,
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
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
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
        if data is None:
            raise AgentOutputError("complex actions require action.data")
        data = _visible_crop_action_data(
            data,
            observation_text_config=observation_text_config,
        )
        target = _action_target(target)
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
    """Read one provider function call."""

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


def _allowed_actions_text(
    action_space: Sequence[ActionSpec],
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    if not action_space:
        return "none"
    lines = [
        (
            "These are the only actions you may choose this turn. The action "
            "glossary may include raw game actions that are not allowed in "
            "this turn."
        )
    ]
    lines.extend(
        f"- {_action_text(action, observation_text_config=observation_text_config)}"
        for action in action_space
    )
    return "\n".join(lines)


def _recent_actions_text(
    history: Sequence[ActionHistoryItem],
    *,
    available: bool,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    if not available:
        return "not available"
    if not history:
        return "none"
    action6_range = action6_coordinate_range_text(observation_text_config)
    lines = [
        (
            "Numbered oldest-to-newest. Controllable action rows may include "
            "nested animation_after rows; GAME_RESET rows mark environment "
            "resets between action groups, and SCORE_ADVANCE rows mark score "
            "or progress increases. The [latest] marker identifies the "
            "transition, reset, or score marker that produced the current "
            "frame. ACTION6 data shown in recent actions is rendered as ARC "
            "grid coordinates. New ACTION6 outputs must use visible cropped "
            f"coordinates {action6_range} on both axes and include a target "
            "description."
        )
    ]
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text,
        numbered=True,
        latest_description=lines[0],
    )


def _action_suppression_evidence_text(
    evidence: ActionOutcomeEvidence | None,
) -> str:
    if evidence is None:
        return ""
    lines: list[str] = []
    if evidence.suppressed_actions:
        lines.append(f"- suppression_threshold: {evidence.suppression_threshold}")
        lines.append(
            "- suppressed_action_choices: " + ", ".join(evidence.suppressed_actions)
        )
        if evidence.suppression_reason:
            lines.append("- suppression_reason: " + evidence.suppression_reason)
    elif evidence.suppression_disabled_reason:
        lines.append(f"- suppression_threshold: {evidence.suppression_threshold}")
        lines.append(
            "- suppression_disabled_reason: "
            + evidence.suppression_disabled_reason
        )
    return "\n".join(lines)


def _action_text(
    action: ActionSpec,
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None = None,
) -> str:
    if action.is_complex() and not action.data:
        return (
            f"{action.name}(x,y "
            f"{action6_coordinate_range_text(observation_text_config)},target)"
        )
    if action.name == "ACTION6":
        return model_facing_action_text(action)
    if action.data:
        return f"{action.name} {json.dumps(action.data, sort_keys=True)}"
    return action.name


def _match_allowed_action(
    action_id: str,
    action_space: Sequence[ActionSpec],
) -> ActionSpec:
    for candidate in action_space:
        if action_id in {candidate.name, str(candidate.action_id)}:
            return candidate
    allowed = ", ".join(action.name for action in action_space)
    raise AgentOutputError(f"action {action_id!r} is not allowed; allowed: {allowed}")


def _visible_crop_action_data(
    data: dict[str, Any],
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
) -> dict[str, int]:
    try:
        return action6_data_from_visible_crop(
            data,
            observation_text_config=observation_text_config,
        )
    except ValueError as exc:
        raise AgentOutputError(str(exc)) from exc


def _action_target(target: str | None) -> str:
    if target is None or not target.strip():
        raise AgentOutputError("ACTION6 requires non-empty action.target")
    return target.strip()
