"""Shared architecture contracts for the ARC-AGI-3 framework.

These contracts are intentionally small. They name the boundaries from the
architecture docs without choosing a model backend, frame representation, or
final database schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from arcengine import FrameDataRaw, GameAction, GameState

MemoryDomain = Literal["state", "experimental"]
ToolName = str
ActionId: TypeAlias = GameAction | str
FrameControlReason = Literal["animation_unroll", "real_environment_turn"]
VisualCoordinateSpace = Literal["pixel", "normalized_1000"]
VisualBBoxOrder = Literal["xyxy", "yxyx"]
VisualAxisFrame = Literal["top_left_x_right_y_down"]
CANONICAL_VISUAL_BBOX_ORDER: VisualBBoxOrder = "xyxy"
CANONICAL_VISUAL_AXIS_FRAME: VisualAxisFrame = "top_left_x_right_y_down"

NONE_ACTION_ID = "NONE"


@dataclass(slots=True)
class ActionSpec:
    """Game action selected from the active ARC-AGI action space.

    The starter shell keeps the ARC `GameAction` enum intact instead of
    redefining actions locally.
    """

    action_id: ActionId
    data: dict[str, Any] | None = None
    target: str | None = None

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
    def animation_unroll(
        cls,
        allowed_actions: tuple[ActionSpec, ...],
    ) -> "FrameControlMode":
        """Return the control mode for non-final unrolled frames."""

        return cls(
            controllable=False,
            allowed_actions=allowed_actions,
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
    previous_observation_ref: ObservationRef | None = None
    recent_action_history: tuple["ActionHistoryItem", ...] = ()


@dataclass(slots=True)
class FrameSourceRef:
    """Model-facing label for a callable source frame."""

    label: str
    source_state_id: int | None


@dataclass(slots=True)
class UpdaterFrameTransitionInput:
    """Updater input for an observed frame transition."""

    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    decision_trace: "AgentTrace"
    actual_next_observation: Observation | None = None
    turn_metrics: "TurnMetrics" = field(
        default_factory=lambda: TurnMetrics()
    )
    submitted_action: ActionSpec | None = None
    synthetic_none_action: ActionSpec | None = None
    action_history_entry: "ActionHistoryEntry | None" = None
    action_history_entries: tuple["ActionHistoryEntry", ...] = ()
    frame_observations: tuple[Observation, ...] = ()
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
class ChangeSummaryElement:
    """One visual element tracked by the change-summary role."""

    element_name: str
    element_description: str
    element_mutation: str = ""


@dataclass(slots=True)
class ActionHistoryEntry:
    """Persisted raw frame-turn action and compact outcome signal."""

    action: ActionSpec
    controllable: bool
    changed_pixel_count: float
    change_summary: str
    change_elements: tuple[ChangeSummaryElement, ...] = ()
    completed_levels: int | None = None
    action_count: int | None = None
    action_mode: Literal["probing", "policy"] | None = None
    skipped_intermediate_animation_frame_count: int = 0
    animation_frame_count: int | None = None
    avg_changed_pixel_count: float | None = None


@dataclass(slots=True)
class ActionHistoryResetMarker:
    """Prompt-facing marker for an environment reset between action rows."""

    reason: str
    restart_count: int


ActionHistoryItem: TypeAlias = ActionHistoryEntry | ActionHistoryResetMarker


@dataclass(slots=True)
class SamePastStateDetection:
    """Prior same-run strategy fields stored for an exact matching frame."""

    probing_strategy: str
    policy_strategy: str
    probing_evolution: str = ""
    policy_evolution: str = ""


@dataclass(slots=True)
class ToolCall:
    """Agent X tool-call shape used by the provider loop and E schema."""

    tool: ToolName
    source_state_id: int
    action: ActionSpec | None = None


@dataclass(slots=True)
class ToolResult:
    """Generic Agent X tool output."""

    id: str
    tool: ToolName
    output: Any
    source_observation_ref: ObservationRef
    source_state_id: int | None = None
    action: ActionSpec | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    cumulative_score: float | None = None


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
    """Context documents for the agent role."""

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
    deadline_monotonic: float | None = None


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
    agent_context: RoleContext
    agent_trace: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: str
    turn_metrics: TurnMetrics = field(
        default_factory=TurnMetrics
    )


@dataclass(slots=True)
class LevelSolutionSummaryRecord:
    """Persisted method summary for one completed level."""

    id: int
    run_id: str
    game_id: str
    completed_level: int
    source_state_ids: tuple[int, ...]
    solution_method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


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


@dataclass(slots=True)
class ParallelGameRunFailure:
    """Captured failure for one game inside a parallel runtime batch."""

    game_index: int
    game_id: str
    run_id: str
    database_path: str
    exception_type: str
    message: str
    attempt_count: int = 1


@dataclass(slots=True)
class ParallelGameRunSuccess:
    """Captured success metadata for one game inside a parallel runtime batch."""

    game_index: int
    game_id: str
    run_id: str
    database_path: str
    result: GameRunResult
    attempt_count: int = 1


@dataclass(slots=True)
class ParallelGameRunResult:
    """Aggregate result for a parallel multi-game runtime batch."""

    batch_run_id: str
    successes: tuple[ParallelGameRunSuccess, ...] = ()
    failures: tuple[ParallelGameRunFailure, ...] = ()
