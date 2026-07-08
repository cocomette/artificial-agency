"""Shared architecture contracts for the ARC-AGI-3 framework.

These contracts are intentionally small. They name the boundaries from the
architecture docs without choosing a model backend, frame representation, or
final database schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

from arcengine import FrameDataRaw, GameAction, GameState

MemoryDomain = Literal["state", "experimental"]
ToolName = Literal["world", "goal"]
ActionId: TypeAlias = GameAction | str
FrameControlReason = Literal["animation_unroll", "real_environment_turn"]
VisualCoordinateSpace = Literal["pixel", "normalized_1000"]


BBox: TypeAlias = list[float]


class DescriptionArea(TypedDict):
    """One structured visual area description produced by S or G."""

    bbox_2d: BBox
    description: str


DescriptionPrediction = list[DescriptionArea]


class DescriptionPredictionError(RuntimeError):
    """Raised when a description prediction violates the shared contract."""

NONE_ACTION_ID = "NONE"

BBOX_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Visual coordinates in [x0, y0, x1, y1] order.",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
}

DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "bbox_2d": BBOX_SCHEMA,
            "description": {
                "type": "string",
                "description": (
                    "Concise expected next-frame change for the currently "
                    "bounded visible image area."
                ),
            },
        },
        "required": ["bbox_2d", "description"],
        "additionalProperties": False,
    },
}


@dataclass(slots=True)
class ActionSpec:
    """Game action selected from the active ARC-AGI action space.

    The starter shell keeps the ARC `GameAction` enum intact instead of
    redefining actions locally.
    """

    action_id: ActionId
    data: dict[str, Any] | None = None

    @classmethod
    def none(cls) -> "ActionSpec":
        """Return the internal no-control action for unrolled animation frames."""

        return cls(action_id=NONE_ACTION_ID)

    @property
    def name(self) -> str:
        """Return a stable display name for ARC and internal actions."""

        if isinstance(self.action_id, GameAction):
            return self.action_id.name
        return str(self.action_id)

    def is_none(self) -> bool:
        """Return whether this is the internal orchestration-only NONE action."""

        return self.name == NONE_ACTION_ID

    def is_complex(self) -> bool:
        """Return whether this action requires ARC action data."""

        return isinstance(self.action_id, GameAction) and self.action_id.is_complex()


@dataclass(slots=True)
class FrameControlMode:
    """Whether a frame turn may submit a real action to the environment."""

    controllable: bool
    allowed_actions: tuple[ActionSpec, ...]
    reason: FrameControlReason

    @classmethod
    def animation_unroll(cls) -> "FrameControlMode":
        """Return the control mode for non-final unrolled frames."""

        return cls(
            controllable=False,
            allowed_actions=(ActionSpec.none(),),
            reason="animation_unroll",
        )

    @classmethod
    def real_environment_turn(
        cls,
        allowed_actions: tuple[ActionSpec, ...],
    ) -> "FrameControlMode":
        """Return the control mode for a frame that can step ARC."""

        return cls(
            controllable=True,
            allowed_actions=allowed_actions,
            reason="real_environment_turn",
        )


@dataclass(slots=True)
class Observation:
    """Observation returned by an ARC-AGI environment adapter."""

    id: str
    step: int
    frame: Any | None = None
    frames: tuple[Any, ...] = ()
    raw_frame_data: FrameDataRaw | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def frame_count(self) -> int:
        """Return the number of incoming frames carried by this observation."""

        if self.frames:
            return len(self.frames)
        if self.frame is None:
            return 0
        return 1


def parse_description_prediction(
    text: str,
    *,
    image_size: tuple[int, int] | None,
    coordinate_space: VisualCoordinateSpace = "pixel",
) -> DescriptionPrediction:
    """Parse provider JSON text into a validated description prediction."""

    try:
        parsed = json.loads(_strip_json_fence(text.strip()))
    except json.JSONDecodeError as exc:
        raise DescriptionPredictionError(
            "description prediction was not valid JSON"
        ) from exc
    return validate_description_prediction(
        parsed,
        image_size=image_size,
        coordinate_space=coordinate_space,
    )


def validate_description_prediction(
    value: Any,
    *,
    image_size: tuple[int, int] | None = None,
    coordinate_space: VisualCoordinateSpace = "pixel",
) -> DescriptionPrediction:
    """Return a normalized description prediction or raise."""

    if isinstance(value, dict) and isinstance(value.get("items"), list):
        value = value["items"]
    if not isinstance(value, list):
        raise DescriptionPredictionError("description prediction must be a JSON array")

    prediction: DescriptionPrediction = []
    errors: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            errors.append(f"item {index}: expected object")
            continue
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"item {index}: missing non-empty description")
            continue
        bbox, bbox_error = _validated_bbox(
            item.get("bbox_2d"),
            label=f"item {index} bbox_2d",
            image_size=image_size,
            coordinate_space=coordinate_space,
        )
        if bbox_error is not None:
            errors.append(bbox_error)
            continue
        unexpected_keys = sorted(set(item) - {"bbox_2d", "description"})
        if unexpected_keys:
            errors.append(
                f"item {index}: unexpected keys {', '.join(unexpected_keys)}"
            )
            continue
        prediction.append({"bbox_2d": bbox, "description": description.strip()})

    if errors:
        raise DescriptionPredictionError("; ".join(errors))
    return prediction


def _validated_bbox(
    value: Any,
    *,
    label: str,
    image_size: tuple[int, int] | None,
    coordinate_space: VisualCoordinateSpace,
) -> tuple[BBox | None, str | None]:
    if not isinstance(value, list):
        return None, f"{label}: expected array [x0, y0, x1, y1]"
    if len(value) != 4:
        return None, f"{label}: expected 4 coordinates [x0, y0, x1, y1]"

    bbox: BBox = []
    for index, raw in enumerate(value):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            return None, f"{label}[{index}]: expected number"
        bbox.append(float(raw))
    if image_size is not None:
        bbox = _bbox_in_pixel_space(
            bbox,
            image_size=image_size,
            coordinate_space=coordinate_space,
        )
    x0, y0, x1, y1 = bbox
    if x1 < x0 or y1 < y0:
        return None, f"{label}: bottom-right must be greater than top-left"
    if image_size is not None:
        width, height = image_size
        if not 0 <= x0 <= width or not 0 <= x1 <= width:
            return None, f"{label}: x coordinates outside image width {width}"
        if not 0 <= y0 <= height or not 0 <= y1 <= height:
            return None, f"{label}: y coordinates outside image height {height}"
    return bbox, None


def _bbox_in_pixel_space(
    bbox: BBox,
    *,
    image_size: tuple[int, int],
    coordinate_space: VisualCoordinateSpace,
) -> BBox:
    if coordinate_space == "pixel":
        return bbox

    width, height = image_size
    x0, y0, x1, y1 = bbox
    return [
        _clamp(x0 * width / 1000, width),
        _clamp(y0 * height / 1000, height),
        _clamp(x1 * width / 1000, width),
        _clamp(y1 * height / 1000, height),
    ]


def _clamp(value: float, size: int) -> float:
    return max(0.0, min(round(value), float(size)))


def _strip_json_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


@dataclass(slots=True)
class FrameTurnContext:
    """One unrolled frame pass through orchestration."""

    run_id: str
    game_id: str
    first_observation_ref: ObservationRef
    current_observation_ref: ObservationRef
    current_observation: Observation
    frame_index: int
    frame_count: int
    control_mode: FrameControlMode
    current_source_state_id: int | None = None
    previous_source_state_id: int | None = None
    previous_observation_ref: ObservationRef | None = None
    recent_action_history: tuple["ActionHistoryEntry", ...] = ()


@dataclass(slots=True)
class FrameSourceRef:
    """Model-facing label for a callable source frame."""

    label: str
    source_state_id: int | None


@dataclass(slots=True)
class PostDecisionPredictions:
    """Committed S/G predictions produced after X chooses a frame action."""

    world_prediction: "PredictionResult | None" = None
    goal_prediction: "PredictionResult | None" = None


@dataclass(slots=True)
class UpdaterFrameTransitionInput:
    """Updater input for an observed frame transition."""

    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    decision_trace: "AgentTrace"
    previous_observation: Observation
    actual_next_observation: Observation | None = None
    post_decision_predictions: PostDecisionPredictions = field(
        default_factory=PostDecisionPredictions
    )
    turn_metrics: "TurnMetrics" = field(
        default_factory=lambda: TurnMetrics()
    )
    submitted_action: ActionSpec | None = None
    synthetic_none_action: ActionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvironmentInfo:
    """ARC-AGI environment metadata exposed to the starter runtime.

    This intentionally mirrors the real ARC toolkit objects closely.
    """

    game_id: str
    state: GameState | None = None
    available_actions: tuple[ActionSpec, ...] = ()
    levels_completed: int = 0
    win_levels: int = 0
    full_reset: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObservationRef:
    """Reference to an observation or prediction stored in a memory domain."""

    memory: MemoryDomain
    id: str


@dataclass(slots=True)
class ActionHistoryEntry:
    """Compact prior frame-turn action exposed to Agent X."""

    action: ActionSpec
    controllable: bool


@dataclass(slots=True)
class ToolCall:
    """Agent X tool-call shape used by the provider loop and E schema."""

    tool: ToolName
    source_state_id: int
    action: ActionSpec | None = None


@dataclass(slots=True)
class ToolResult:
    """Prediction or tool output carrying a description artifact."""

    id: str
    tool: ToolName
    predicted_description: DescriptionPrediction
    source_observation_ref: ObservationRef
    source_state_id: int | None = None
    action: ActionSpec | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PredictionCall: TypeAlias = ToolCall
PredictionResult: TypeAlias = ToolResult


@dataclass(slots=True)
class AgentTrace:
    """Trace produced by the agent for one decision step."""

    step: int
    first_observation_ref: ObservationRef
    current_observation_ref: ObservationRef
    final_action: ActionSpec
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    reasoning_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnMetrics:
    """Frame-turn cost, timing, and score/progress metrics."""

    time_cost: float | None = None
    trace_cost: float | None = None
    score_delta: float | None = None


@dataclass(slots=True)
class RoleContext:
    """Text context documents for one model role.

    `general` corresponds to K in the architecture docs. `game` corresponds to
    the current game-specific L document.
    """

    general: str = ""
    game: str = ""

    def composed(self) -> str:
        """Return the current K + L text used for a model call."""

        return "\n\n".join(part for part in (self.general, self.game) if part)


@dataclass(slots=True)
class ContextDocuments:
    """Context documents for the world, goal, and agent roles."""

    world: RoleContext = field(default_factory=RoleContext)
    goal: RoleContext = field(default_factory=RoleContext)
    agent: RoleContext = field(default_factory=RoleContext)


@dataclass(slots=True)
class DecisionResult:
    """Agent decision output for the current barebones runtime."""

    final_action: ActionSpec
    trace: AgentTrace


@dataclass(slots=True)
class RuntimeConfig:
    """Minimal runtime configuration for one multi-game call."""

    run_id: str
    database_path: str | Path | None = None
    game_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class MStateRecord:
    """One complete persistent M state row for a frame turn."""

    id: int
    game_id: str
    run_id: str
    step: int | None
    frame_index: int
    frame_count: int
    current_observation: dict[str, Any]
    chosen_action: dict[str, Any] | None
    world_context: RoleContext
    goal_context: RoleContext
    agent_context: RoleContext
    agent_trace: dict[str, Any] | None
    world_prediction: dict[str, Any] | None
    goal_prediction: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: str
    turn_metrics: TurnMetrics = field(
        default_factory=TurnMetrics
    )


@dataclass(slots=True)
class EExperimentRecord:
    """One experimental tool output stored in rolling memory E."""

    id: int
    game_id: str
    run_id: str
    turn_id: int
    tool_name: ToolName
    source_state_id: int
    tool_call: dict[str, Any]
    output_description: dict[str, Any]
    tool_result: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class ExperimentToolInvocationResult:
    """Result of an orchestration-mediated tool call persisted to E."""

    tool_result: ToolResult
    experiment_record: EExperimentRecord


@dataclass(slots=True)
class GameRunResult:
    """Result of a runtime boundary for one game."""

    run_id: str
    game_id: str
    initial_observation_ref: ObservationRef | None = None
    decision: DecisionResult | None = None
    state_record_ids: tuple[int, ...] = ()
    stop_reason: str | None = None
    step_count: int = 0
    completed_levels: int = 0
    last_state: GameState | None = None
