"""Shared helpers for orchestrator-agent tool-calling backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from face_of_agi.contracts import (
    AgentCandidateAction,
    ActionHistoryItem,
    ActionOutcomeEvidence,
    ActionSpec,
    AgentTrace,
    CandidateValuePrediction,
    DecisionResult,
    ExperimentToolInvocationResult,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
    VisualCoordinateSpace,
    WorldPrediction,
)
from face_of_agi.frames import observation_to_pil_image, to_memory_jsonable
from face_of_agi.models.arc_grid_crop import normalized_1000_to_arc_grid
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
    model_facing_action_text_for_crop,
)
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime

INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "system_prompt.md"


class AgentOutputError(RuntimeError):
    """Raised when a provider response violates the X agent contract."""


def load_agent_instructions() -> str:
    """Load the fixed X system prompt."""

    return INSTRUCTION_PATH.read_text(encoding="utf-8").strip()


def build_agent_instructions(
    *,
    glossary_actions: Sequence[ActionSpec],
) -> str:
    """Return X instructions with the current raw action glossary."""

    return append_action_glossary(
        load_agent_instructions(),
        glossary_actions,
        mode="agent_decision",
    )


def build_decision_prompt(
    *,
    context: RoleContext,
    action_space: Sequence[ActionSpec],
    recent_action_history: Sequence[ActionHistoryItem] = (),
    recent_action_history_available: bool = True,
    action_outcome_evidence: ActionOutcomeEvidence | None = None,
    crop_edges: Any | None = None,
) -> str:
    """Build the provider-neutral Markdown text sent beside X images."""

    parts = [
        "## Agent context\n\n" + _text_or_none(context.composed()),
        "## Allowed actions\n\n"
        + _allowed_actions_text(
            action_space,
            crop_edges=crop_edges,
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
            crop_edges=crop_edges,
        )
    )
    return "\n\n".join(parts)


def observation_images(
    *,
    current_observation: Observation,
    frame_scale: int,
) -> tuple[Any, ...]:
    """Return the current observation image for X."""

    return (
        observation_to_pil_image(current_observation, frame_scale=frame_scale),
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


def candidate_actions_schema(action_space: Sequence[ActionSpec]) -> dict[str, Any]:
    """Return the candidate-action proposal schema."""

    return {
        "type": "object",
        "properties": {
            "candidate_actions": {
                "type": "array",
                "items": _action_output_schema(
                    simple_actions=[action for action in action_space if not action.is_complex()],
                    complex_actions=[action for action in action_space if action.is_complex()],
                ),
            },
            "notes": {"type": "string"},
        },
        "required": ["candidate_actions", "notes"],
        "additionalProperties": False,
    }


def build_candidate_prompt(
    *,
    memory: MemoryDocument,
    goal: GoalPrediction,
    action_space: Sequence[ActionSpec],
    max_candidates: int,
    recent_action_history: Sequence[ActionHistoryItem] = (),
    crop_edges: Any | None = None,
) -> str:
    """Build the candidate-proposal prompt."""

    return "\n\n".join(
        [
            "## Memory\n\n" + _text_or_none(memory.document),
            "## Goal prediction\n\n" + goal_prediction_text(goal),
            "## Allowed actions\n\n"
            + _allowed_actions_text(action_space, crop_edges=crop_edges),
            "## Recent actions\n\n"
            + _recent_actions_text(
                recent_action_history,
                available=True,
                crop_edges=crop_edges,
            ),
            "Return up to "
            f"{max_candidates} useful ACTION6 coordinate candidates. "
            "Runtime will include all simple actions automatically, so focus on "
            "coordinates only when ACTION6 is available.",
        ]
    )


def build_selection_prompt(
    *,
    memory: MemoryDocument,
    goal: GoalPrediction,
    candidates: Sequence[AgentCandidateAction],
    world_predictions: Sequence[WorldPrediction],
    interest_prediction: InterestPrediction | None = None,
    recent_action_history: Sequence[ActionHistoryItem] = (),
    crop_edges: Any | None = None,
) -> str:
    """Build the final action-selection prompt."""

    prediction_by_index = {
        prediction.candidate_index: prediction for prediction in world_predictions
    }
    value_by_index = _candidate_value_by_index(interest_prediction)
    candidate_lines: list[str] = []
    for candidate in candidates:
        prediction = prediction_by_index.get(candidate.rank)
        value = value_by_index.get(candidate.rank)
        predicted_change = (
            prediction.predicted_change if prediction is not None else "not available"
        )
        lines = [
            f"- candidate_index: {candidate.rank}",
            "  action: "
            + model_facing_action_text(
                candidate.action,
                crop_edges=crop_edges,
            ),
            f"  source: {candidate.source}",
            f"  predicted_change: {predicted_change}",
            "  interest_value: " + _interest_value_text(value),
            f"  rationale: {candidate.rationale}",
        ]
        candidate_lines.append("\n".join(lines))
    return "\n\n".join(
        [
            "## Memory\n\n" + _text_or_none(memory.document),
            "## Goal prediction\n\n" + goal_prediction_text(goal),
            "## Candidate actions with World predictions and Interest values\n\n"
            + "\n".join(candidate_lines),
            "## Recent actions\n\n"
            + _recent_actions_text(
                recent_action_history,
                available=True,
                crop_edges=crop_edges,
            ),
            "Choose exactly one candidate action as the final environment action.",
        ]
    )


def _candidate_value_by_index(
    interest_prediction: InterestPrediction | None,
) -> dict[int, CandidateValuePrediction]:
    if interest_prediction is None:
        return {}
    return {
        value.candidate_index: value
        for value in interest_prediction.candidate_values
    }


def _interest_value_text(value: CandidateValuePrediction | None) -> str:
    if value is None:
        return "not available"
    confidence_adjusted_lp = value.metadata.get(
        "confidence_adjusted_learning_progress"
    )
    blended_score = value.metadata.get("blended_score")
    fields = [
        f"expected_learning_progress={value.expected_learning_progress}",
        f"confidence={value.confidence}",
        f"confidence_adjusted_learning_progress={confidence_adjusted_lp}",
        f"expected_goal_delta={value.expected_goal_delta}",
        f"blended_score={blended_score}",
    ]
    if value.notes:
        fields.append(f"notes={value.notes}")
    return "; ".join(fields)


def goal_prediction_text(goal: GoalPrediction) -> str:
    """Return prompt-facing Goal output text."""

    return "\n".join(
        [
            f"goal: {goal.goal}",
            f"subgoals: {list(goal.subgoals)}",
            f"steps_remaining: {goal.steps_remaining}",
            f"confidence: {goal.confidence}",
        ]
    )


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
        required.append("data")
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
            "`x` and `y` in normalized visual 0..1000 coordinates. "
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
    crop_edges: Any | None = None,
) -> ActionSpec:
    """Parse the terminal structured final-action payload."""

    args = parse_arguments(arguments)
    return parse_action(
        args.get("action"),
        action_space,
        coordinate_space=coordinate_space,
        crop_edges=crop_edges,
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
    crop_edges: Any | None = None,
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

    if matched.is_complex():
        if data is None:
            raise AgentOutputError("complex actions require action.data")
        data = _normalized_action_data(
            data,
            coordinate_space=coordinate_space,
            crop_edges=crop_edges,
        )
    elif data is not None:
        raise AgentOutputError("simple actions must not include action.data")

    return ActionSpec(action_id=matched.action_id, data=data)


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


def _allowed_actions_text(
    action_space: Sequence[ActionSpec],
    *,
    crop_edges: Any | None,
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
        f"- {_action_text(action, crop_edges=crop_edges)}"
        for action in action_space
    )
    return "\n".join(lines)


def _recent_actions_text(
    history: Sequence[ActionHistoryItem],
    *,
    available: bool,
    crop_edges: Any | None,
) -> str:
    if not available:
        return "not available"
    if not history:
        return "none"
    lines = [
        (
            "Numbered oldest-to-newest. Controllable action rows may include "
            "animation evidence fields; nested animation_after rows mark "
            "synthetic animation-only turns. GAME_RESET rows mark environment "
            "resets between action groups, and SCORE_ADVANCE rows mark score "
            "or progress increases. The [latest] marker identifies the "
            "transition, reset, or score marker that produced the attached "
            "current frame. "
            "ACTION6 data shown in recent actions is rendered as normalized "
            "visual 0..1000 coordinates, matching the coordinate space used "
            "for new ACTION6 outputs."
        )
    ]
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text_for_crop(crop_edges),
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
    crop_edges: Any | None,
) -> str:
    if action.is_complex() and not action.data:
        return f"{action.name}(x,y normalized_0_1000)"
    if action.name == "ACTION6":
        return model_facing_action_text(
            action,
            crop_edges=crop_edges,
        )
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


def _normalized_action_data(
    data: dict[str, Any],
    *,
    coordinate_space: VisualCoordinateSpace,
    crop_edges: Any | None,
) -> dict[str, int]:
    if coordinate_space == "normalized_1000":
        try:
            return {
                "x": normalized_1000_to_arc_grid(
                    _normalized_coordinate(data, "x"),
                    "x",
                    crop_edges=crop_edges,
                ),
                "y": normalized_1000_to_arc_grid(
                    _normalized_coordinate(data, "y"),
                    "y",
                    crop_edges=crop_edges,
                ),
            }
        except ValueError as exc:
            raise AgentOutputError(str(exc)) from exc
    raise AgentOutputError(
        "pixel visual coordinates cannot be converted to ARC coordinates "
        "without an image size; use a normalized_1000 model profile"
    )


def _normalized_coordinate(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"complex action.data.{key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"complex action.data.{key} must be numeric")
    numeric = float(value)
    if not 0 <= numeric <= 1000:
        raise ValueError(f"complex action.data.{key} must be in normalized 0..1000")
    return numeric
