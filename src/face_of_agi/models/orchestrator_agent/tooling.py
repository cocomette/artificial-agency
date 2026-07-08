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
    VisualCoordinateSpace,
)
from face_of_agi.frames import (
    observation_arc_cell_value,
    observation_to_pil_image,
    to_memory_jsonable,
)
from face_of_agi.models.action_coordinates import (
    action6_data_from_normalized_1000,
    cropped_pixel_to_arc_grid_coordinate,
)
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
    model_facing_action_text_for_crop,
)
from face_of_agi.models.memory import GameMemoryDocument
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime
from face_of_agi.models.image_inputs import crop_image_normalized

INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "system_prompt.md"
ACTION6_TARGETING_COORDINATES = "coordinates"
ACTION6_TARGETING_BBOX_COLOR = "bbox_color"
ACTION6_TARGETING_BBOX_CENTER = "bbox_center"


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
    game_memory: GameMemoryDocument | None = None,
    crop_box_normalized: Any | None = None,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> str:
    """Build the provider-neutral Markdown text sent beside X images."""

    parts = [
        "## Agent context\n\n" + _text_or_none(context.composed()),
        "## Allowed actions\n\n"
        + _allowed_actions_text(
            action_space,
            crop_box_normalized=crop_box_normalized,
            action6_targeting_mode=action6_targeting_mode,
        ),
    ]
    suppression_evidence = _action_suppression_evidence_text(
        action_outcome_evidence
    )
    if suppression_evidence:
        parts.append("## Action suppression evidence\n\n" + suppression_evidence)
    parts.append(
        "## Game memory\n\n"
        + _game_memory_text(game_memory or GameMemoryDocument.not_available())
    )
    parts.append(
        "## Recent actions\n\n"
        + _recent_actions_text(
            recent_action_history,
            available=recent_action_history_available,
            crop_box_normalized=crop_box_normalized,
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


def final_action_schema(
    action_space: Sequence[ActionSpec],
    *,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> dict[str, Any]:
    """Return the structured final-action schema for one X frame turn."""

    mode = _normalized_action6_targeting_mode(action6_targeting_mode)
    simple_actions = [action for action in action_space if not action.is_complex()]
    complex_actions = [action for action in action_space if action.is_complex()]
    return {
        "type": "object",
        "properties": {
            "action": _action_output_schema(
                simple_actions=simple_actions,
                complex_actions=complex_actions,
                action6_targeting_mode=mode,
            ),
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def _action_output_schema(
    *,
    simple_actions: Sequence[ActionSpec],
    complex_actions: Sequence[ActionSpec],
    action6_targeting_mode: str,
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
                action6_targeting_mode=action6_targeting_mode,
            )
        )
    if len(branches) == 1:
        return branches[0]
    return {"anyOf": branches}


def _action_object_schema(
    *,
    action_ids: Sequence[str],
    include_data: bool,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action_id": {
            "type": "string",
            "enum": list(action_ids),
        },
    }
    required = ["action_id"]
    if include_data and action6_targeting_mode in {
        ACTION6_TARGETING_BBOX_COLOR,
        ACTION6_TARGETING_BBOX_CENTER,
    }:
        properties["target"] = {
            "type": "string",
            "description": (
                "Concise visual description of the object or area targeted."
            ),
        }
        properties["bbox"] = {
            "type": "array",
            "description": (
                "[x0, y0, x1, y1] tight bounding box around the target area in "
                "normalized visual 0..1000 coordinates."
            ),
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        }
        required.extend(["target", "bbox"])
        if action6_targeting_mode == ACTION6_TARGETING_BBOX_COLOR:
            properties["target_rgb_color"] = {
                "type": "array",
                "description": "RGB color of the target pixels inside bbox.",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            }
            required.append("target_rgb_color")
    elif include_data:
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
    *,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> dict[str, Any]:
    """Return OpenAI Responses structured-output config for final X action."""

    return {
        "format": {
            "type": "json_schema",
            "name": "agent_final_action",
            "strict": True,
            "schema": final_action_schema(
                action_space,
                action6_targeting_mode=action6_targeting_mode,
            ),
        }
    }


def final_action_repair_prompt(
    action_space: Sequence[ActionSpec],
    *,
    validation_error: str,
    invalid_text: str | None,
    attempt: int,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> str:
    """Return a provider-neutral final-action repair request."""

    mode = _normalized_action6_targeting_mode(action6_targeting_mode)
    allowed = ", ".join(action.name for action in action_space)
    if mode == ACTION6_TARGETING_BBOX_COLOR:
        action6_contract = (
            "ACTION6 must include a non-empty `target` string, `bbox` as "
            "[x0,y0,x1,y1] in normalized visual 0..1000 coordinates, and "
            "`target_rgb_color` as [r,g,b]. ACTION6 must not include `data`."
        )
    elif mode == ACTION6_TARGETING_BBOX_CENTER:
        action6_contract = (
            "ACTION6 must include a non-empty `target` string and `bbox` as "
            "[x0,y0,x1,y1] in normalized visual 0..1000 coordinates. "
            "ACTION6 must not include `data` or `target_rgb_color`; the "
            "runtime clicks the bbox center."
        )
    else:
        action6_contract = (
            "ACTION6 must include a `data` object with numeric `x` and `y` "
            "in normalized visual 0..1000 coordinates and a non-empty `target` "
            "string describing the targeted object or area."
        )
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
            f"include `data`, `target`, `bbox`, or `target_rgb_color`; "
            f"{action6_contract} Do not include prose.",
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
    crop_box_normalized: Any | None = None,
    current_observation: Observation | None = None,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> ActionSpec:
    """Parse the terminal structured final-action payload."""

    args = parse_arguments(arguments)
    return parse_action(
        args.get("action"),
        action_space,
        coordinate_space=coordinate_space,
        crop_box_normalized=crop_box_normalized,
        current_observation=current_observation,
        action6_targeting_mode=action6_targeting_mode,
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
    crop_box_normalized: Any | None = None,
    current_observation: Observation | None = None,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> ActionSpec:
    """Parse and validate a provider action against the current action space."""

    if not isinstance(value, dict):
        raise AgentOutputError("action must be an object")

    mode = _normalized_action6_targeting_mode(action6_targeting_mode)
    raw_action_id = value.get("action_id")
    if raw_action_id is None:
        raise AgentOutputError("action.action_id is required")

    matched = _match_allowed_action(str(raw_action_id), action_space)
    allowed_keys = {"action_id", "data", "target"}
    if matched.is_complex() and mode == ACTION6_TARGETING_BBOX_COLOR:
        allowed_keys = {"action_id", "target", "bbox", "target_rgb_color"}
    elif matched.is_complex() and mode == ACTION6_TARGETING_BBOX_CENTER:
        allowed_keys = {"action_id", "target", "bbox"}
    unexpected = set(value) - allowed_keys
    if unexpected:
        raise AgentOutputError(
            "action contains unexpected keys: " + ", ".join(sorted(unexpected))
        )
    data = value.get("data")
    if data is not None and not isinstance(data, dict):
        raise AgentOutputError("action.data must be an object when provided")
    target = value.get("target")
    if target is not None and not isinstance(target, str):
        raise AgentOutputError("action.target must be a string when provided")

    if matched.is_complex():
        target = _action_target(target)
        target_bbox: tuple[int, int, int, int] | None = None
        target_value: int | None = None
        if mode == ACTION6_TARGETING_BBOX_COLOR:
            if data is not None:
                raise AgentOutputError("ACTION6 bbox_color mode must not include data")
            target_bbox = _action6_bbox(value.get("bbox"))
            target_rgb = _action6_target_rgb_color(value.get("target_rgb_color"))
            data = _retarget_action6_bbox_color_data(
                value,
                current_observation=current_observation,
                crop_box_normalized=crop_box_normalized,
                bbox=target_bbox,
                target_rgb=target_rgb,
            )
            target_value = _action6_target_value(current_observation, data)
        elif mode == ACTION6_TARGETING_BBOX_CENTER:
            if data is not None:
                raise AgentOutputError(
                    "ACTION6 bbox_center mode must not include data"
                )
            target_bbox = _action6_bbox(value.get("bbox"))
            data = _action6_bbox_center_data(
                value,
                crop_box_normalized=crop_box_normalized,
                bbox=target_bbox,
            )
            target_value = _action6_target_value(current_observation, data)
        else:
            if data is None:
                raise AgentOutputError("complex actions require action.data")
            data = _normalized_action_data(
                data,
                coordinate_space=coordinate_space,
                crop_box_normalized=crop_box_normalized,
            )
            target_bbox = None
            target_value = _action6_target_value(current_observation, data)
    elif data is not None:
        raise AgentOutputError("simple actions must not include action.data")
    elif target is not None:
        raise AgentOutputError("simple actions must not include action.target")
    else:
        target_bbox = None
        target_value = None

    return ActionSpec(
        action_id=matched.action_id,
        data=data,
        target=target,
        target_value=target_value,
        target_bbox=target_bbox,
    )


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
    crop_box_normalized: Any | None,
    action6_targeting_mode: str,
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
        f"- {_action_text(
            action,
            crop_box_normalized=crop_box_normalized,
            action6_targeting_mode=action6_targeting_mode,
        )}"
        for action in action_space
    )
    return "\n".join(lines)


def _recent_actions_text(
    history: Sequence[ActionHistoryItem],
    *,
    available: bool,
    crop_box_normalized: Any | None,
) -> str:
    if not available:
        return "not available"
    if not history:
        return "none"
    lines = [
        (
            "Numbered oldest-to-newest. Controllable action rows may include "
            "nested animation_after rows; GAME_RESET rows mark environment "
            "resets between action groups, and SCORE_ADVANCE rows mark score "
            "or progress increases. The [latest] marker identifies the "
            "transition, reset, or score marker that produced the attached "
            "current frame. changed_area is a display-only percentage for the "
            "same first-to-final evidence as changed_pixels. "
            "ACTION6 data shown in recent actions is rendered only as a "
            "target string naming the selected visible object or area. New "
            "ACTION6 outputs must use the shape shown in Allowed actions. "
            "Rows with `Elements and associated changes` contain structured "
            "element bullets; elements can be targets, triggers, objects, "
            "characters, buttons, items to collect, obstacles, layout, paths, "
            "or other game-relevant artifacts."
        )
    ]
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text_for_crop(crop_box_normalized),
        numbered=True,
        latest_description=lines[0],
    )


def _game_memory_text(memory: GameMemoryDocument) -> str:
    return _text_or_none(memory.markdown)


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
    crop_box_normalized: Any | None,
    action6_targeting_mode: str = ACTION6_TARGETING_COORDINATES,
) -> str:
    mode = _normalized_action6_targeting_mode(action6_targeting_mode)
    if action.is_complex() and not action.data:
        if action.name == "ACTION6" and mode == ACTION6_TARGETING_BBOX_COLOR:
            return f"{action.name}(target,bbox,target_rgb_color)"
        if action.name == "ACTION6" and mode == ACTION6_TARGETING_BBOX_CENTER:
            return f"{action.name}(target,bbox)"
        return f"{action.name}(x,y normalized_0_1000,target)"
    if action.name == "ACTION6":
        return model_facing_action_text(
            action,
            crop_box_normalized=crop_box_normalized,
        )
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
    crop_box_normalized: Any | None,
) -> dict[str, int]:
    if coordinate_space == "normalized_1000":
        try:
            return action6_data_from_normalized_1000(
                data,
                crop_box_normalized=crop_box_normalized,
            )
        except ValueError as exc:
            raise AgentOutputError(str(exc)) from exc
    raise AgentOutputError(
        "pixel visual coordinates cannot be converted to ARC coordinates "
        "without an image size; use a normalized_1000 model profile"
    )


def _retarget_action6_bbox_color_data(
    value: dict[str, Any],
    *,
    current_observation: Observation | None,
    crop_box_normalized: Any | None,
    bbox: tuple[int, int, int, int] | None = None,
    target_rgb: tuple[int, int, int] | None = None,
) -> dict[str, int]:
    if current_observation is None:
        raise AgentOutputError(
            "ACTION6 bbox_color mode requires the current observation"
        )
    if bbox is None:
        bbox = _action6_bbox(value.get("bbox"))
    if target_rgb is None:
        target_rgb = _action6_target_rgb_color(value.get("target_rgb_color"))
    image = crop_image_normalized(
        observation_to_pil_image(current_observation),
        crop_box_normalized,
    )
    pixel_x, pixel_y = _closest_target_color_pixel(
        image,
        bbox=bbox,
        target_rgb=target_rgb,
    )
    return {
        "x": cropped_pixel_to_arc_grid_coordinate(
            pixel_x,
            image.width,
            "x",
            crop_box_normalized=crop_box_normalized,
        ),
        "y": cropped_pixel_to_arc_grid_coordinate(
            pixel_y,
            image.height,
            "y",
            crop_box_normalized=crop_box_normalized,
        ),
    }


def _action6_bbox_center_data(
    value: dict[str, Any],
    *,
    crop_box_normalized: Any | None,
    bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, int]:
    if bbox is None:
        bbox = _action6_bbox(value.get("bbox"))
    x0, y0, x1, y1 = bbox
    try:
        return action6_data_from_normalized_1000(
            {
                "x": (x0 + x1) / 2,
                "y": (y0 + y1) / 2,
            },
            crop_box_normalized=crop_box_normalized,
        )
    except ValueError as exc:
        raise AgentOutputError(str(exc)) from exc


def _action6_target_value(
    current_observation: Observation | None,
    data: dict[str, int],
) -> int | None:
    if current_observation is None:
        return None
    try:
        return observation_arc_cell_value(
            current_observation,
            x=int(data["x"]),
            y=int(data["y"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentOutputError(
            f"ACTION6 target value could not be derived: {exc}"
        ) from exc


def _closest_target_color_pixel(
    image: Any,
    *,
    bbox: tuple[int, int, int, int],
    target_rgb: tuple[int, int, int],
) -> tuple[int, int]:
    rgb_image = image.convert("RGB")
    left, top, right, bottom = _bbox_pixel_window(bbox, rgb_image.size)
    center_x = (left + right - 1) / 2
    center_y = (top + bottom - 1) / 2
    best: tuple[int, float, int, int] | None = None
    for y in range(top, bottom):
        for x in range(left, right):
            color = rgb_image.getpixel((x, y))
            color_distance = sum(
                (int(color[index]) - target_rgb[index]) ** 2
                for index in range(3)
            )
            center_distance = (x - center_x) ** 2 + (y - center_y) ** 2
            candidate = (color_distance, center_distance, x, y)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise AgentOutputError("ACTION6 bbox did not contain any target pixels")
    return (best[2], best[3])


def _bbox_pixel_window(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise AgentOutputError("ACTION6 retargeting requires a non-empty image")
    x0, y0, x1, y1 = bbox
    left = _clamp(int(x0 * width / 1000), 0, width - 1)
    top = _clamp(int(y0 * height / 1000), 0, height - 1)
    right = _clamp(_ceil_div(x1 * width, 1000), left + 1, width)
    bottom = _clamp(_ceil_div(y1 * height, 1000), top + 1, height)
    return (left, top, right, bottom)


def _action6_bbox(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise AgentOutputError("ACTION6 requires bbox [x0,y0,x1,y1]")
    coordinates: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AgentOutputError(f"ACTION6 bbox item {index} must be numeric")
        numeric = float(item)
        if not 0 <= numeric <= 1000:
            raise AgentOutputError("ACTION6 bbox coordinates must be in 0..1000")
        coordinates.append(int(round(numeric)))
    x0, y0, x1, y1 = coordinates
    if not (x0 < x1 and y0 < y1):
        raise AgentOutputError("ACTION6 bbox must satisfy x0 < x1 and y0 < y1")
    return (x0, y0, x1, y1)


def _action6_target_rgb_color(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise AgentOutputError("ACTION6 requires target_rgb_color [r,g,b]")
    channels: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise AgentOutputError(
                f"ACTION6 target_rgb_color item {index} must be an integer"
            )
        if not 0 <= item <= 255:
            raise AgentOutputError("ACTION6 target_rgb_color values must be 0..255")
        channels.append(int(item))
    return (channels[0], channels[1], channels[2])


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _normalized_action6_targeting_mode(value: str) -> str:
    if value in {
        ACTION6_TARGETING_COORDINATES,
        ACTION6_TARGETING_BBOX_COLOR,
        ACTION6_TARGETING_BBOX_CENTER,
    }:
        return value
    raise AgentOutputError(
        "action6_targeting_mode must be 'coordinates', 'bbox_color', "
        "or 'bbox_center'"
    )
