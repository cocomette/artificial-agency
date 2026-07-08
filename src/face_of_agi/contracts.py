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
    transition_summary: str
    skipped_intermediate_animation_frame_count: int = 0


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
class AgentTrace:
    """Trace produced by the online learner for one decision step."""

    step: int
    first_observation_ref: ObservationRef
    current_observation_ref: ObservationRef
    final_action: ActionSpec
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnMetrics:
    """Frame-turn cost, timing, and score/progress metrics."""

    time_cost: float | None = None
    trace_cost: float | None = None
    cumulative_score: float | None = None


@dataclass(slots=True)
class DecisionResult:
    """Online learner decision output for the current runtime."""

    final_action: ActionSpec
    trace: AgentTrace


@dataclass(slots=True)
class TransitionRecord:
    """Action-conditioned observed transition used for online learning."""

    previous_observation_ref: ObservationRef
    next_observation_ref: ObservationRef
    action: ActionSpec
    controllable: bool
    changed_pixel_percent: float
    score_delta: float | None = None
    completed_levels: int | None = None
    prediction_error: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplayStats:
    """Bounded replay/update work completed after one observed transition."""

    real_update_count: int = 0
    replay_update_count: int = 0
    elapsed_seconds: float = 0.0
    sampled_transition_ids: tuple[str, ...] = ()
    mean_prediction_error: float | None = None


@dataclass(slots=True)
class PlannerCandidate:
    """One action candidate scored by the local planner."""

    action: ActionSpec
    score: float
    predicted_value: float = 0.0
    uncertainty: float = 0.0
    information_gain: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LearnerTurnTrace:
    """Structured learner trace persisted for dashboard inspection."""

    decision: DecisionResult
    transition: TransitionRecord | None = None
    replay: ReplayStats = field(default_factory=ReplayStats)
    planner_candidates: tuple[PlannerCandidate, ...] = ()
    backbone_metadata: dict[str, Any] = field(default_factory=dict)
    learner_metadata: dict[str, Any] = field(default_factory=dict)


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
    learner_snapshot: dict[str, Any]
    learner_trace: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: str
    turn_metrics: TurnMetrics = field(
        default_factory=TurnMetrics
    )


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
