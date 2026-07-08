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
    action_history_score_advance_marker: (
        ActionHistoryScoreAdvanceMarker | None
    ) = None
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
    """Persisted frame-turn action and compact normalized outcome signal."""

    action: ActionSpec
    controllable: bool
    changed_pixel_percent: float
    change_summary: str
    reward: TurnReward | None = None
    reward_judge_notes: str = ""
    reward_error_tags: tuple[str, ...] = ()
    retained_animation_frame_count: int = 0
    skipped_intermediate_animation_frame_count: int = 0
    animation_avg_changed_pixel_percent: float | None = None


@dataclass(slots=True)
class ActionHistoryResetMarker:
    """Prompt-facing marker for an environment reset between action rows."""

    reason: str
    restart_count: int


@dataclass(slots=True)
class ActionHistoryScoreAdvanceMarker:
    """Prompt-facing marker for a score/progress advance between action rows."""

    previous_score: float | None
    new_score: float
    delta: float | None


ActionHistoryItem: TypeAlias = (
    ActionHistoryEntry
    | ActionHistoryResetMarker
    | ActionHistoryScoreAdvanceMarker
)


@dataclass(slots=True)
class ActionOutcomeEvidence:
    """Deterministic low-information action evidence for model prompts."""

    suppression_threshold: int = 0
    # Prompt-facing action-choice labels, not necessarily whole action classes.
    suppressed_actions: tuple[str, ...] = ()
    suppression_reason: str = ""
    suppression_disabled_reason: str = ""
    latest_repeated_action: str = ""
    latest_repeated_action_count: int = 0
    latest_same_action_zero_changed_pixel_turn_count: int = 0
    stagnation_warning_threshold: int = 0
    stagnation_warning: bool = False


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
    model_prompt_tokens: int = 0
    model_completion_tokens: int = 0
    model_total_tokens: int = 0


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


@dataclass(frozen=True, slots=True)
class AgentCandidateAction:
    """One candidate action considered by the two-stage agent loop."""

    action: ActionSpec
    source: Literal["runtime_simple_action", "agent_coordinate_proposal"]
    rank: int
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class MemoryDocument:
    """Current model-facing memory document regenerated from the turn ledger."""

    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GoalPrediction:
    """Structured goal estimate used by the agent and reward shaping."""

    goal: str
    subgoals: tuple[str, ...]
    steps_remaining: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorldPrediction:
    """Predicted visible transition for one candidate action."""

    candidate_index: int
    action: ActionSpec
    predicted_change: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateValuePrediction:
    """Interest/value estimate for one candidate action."""

    candidate_index: int
    action: ActionSpec
    expected_learning_progress: float
    expected_goal_delta: float
    confidence: float
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InterestPrediction:
    """Batch Interest output for the current candidate set."""

    candidate_values: tuple[CandidateValuePrediction, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardJudgeScore:
    """Scalar text/VLM judge score for a world prediction."""

    score: float
    notes: str
    error_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnReward:
    """Computed reward components for one executed environment action."""

    prediction_accuracy: float
    learning_progress: float | None
    goal_delta: float
    progress_bonus: float
    resource_cost: float
    lp_weight: float
    goal_weight: float
    total: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnLedgerEntry:
    """Text-and-JSON durable ledger entry used by Memory and debug records."""

    turn_id: int
    action: ActionSpec
    change_summary: str
    reward: TurnReward | None = None
    candidate_predictions: tuple[WorldPrediction, ...] = ()
    judge_scores: tuple[RewardJudgeScore, ...] = ()
    goal_before: GoalPrediction | None = None
    goal_after: GoalPrediction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
class TurnLedgerRecord:
    """Durable turn-ledger row for the new memory/world/goal runtime."""

    id: int
    m_state_id: int | None
    game_id: str
    run_id: str
    turn_id: int
    action: dict[str, Any]
    change_summary: str
    memory_document: str
    goal_prediction: dict[str, Any] | None
    reward: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class CandidatePredictionRecord:
    """Durable world prediction row for one candidate action."""

    id: int
    game_id: str
    run_id: str
    turn_id: int
    candidate_index: int
    action: dict[str, Any]
    prediction: str
    source: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class JudgeScoreRecord:
    """Durable reward-judge row for one candidate prediction."""

    id: int
    game_id: str
    run_id: str
    turn_id: int
    candidate_prediction_id: int | None
    score: float
    notes: str
    error_tags: tuple[str, ...]
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class GoalPredictionRecord:
    """Durable goal prediction row."""

    id: int
    game_id: str
    run_id: str
    turn_id: int
    goal_prediction: dict[str, Any]
    memory_document: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class RewardRecord:
    """Durable reward row for one turn."""

    id: int
    game_id: str
    run_id: str
    turn_id: int
    reward: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class RunMetadataRecord:
    """One durable run-level metadata row stored with memory."""

    id: int
    game_id: str
    run_id: str
    kind: str
    metadata: dict[str, Any]
    created_at: str


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
