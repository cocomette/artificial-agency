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
ToolName = Literal["world", "goal"]
ActionId: TypeAlias = GameAction | str
FrameControlReason = Literal["animation_unroll", "real_environment_turn"]

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


@dataclass(slots=True)
class PostDecisionPredictions:
    """Committed S/G predictions produced after X chooses a real action."""

    world_prediction: ToolResult | None = None
    goal_prediction: ToolResult | None = None


@dataclass(slots=True)
class UpdaterFrameTransitionInput:
    """Updater input for a real observed frame transition."""

    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    decision_trace: "AgentTrace"
    post_decision_predictions: PostDecisionPredictions = field(
        default_factory=PostDecisionPredictions
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
class ToolCall:
    """Agent request to call the world or goal model as a tool.

    World calls require `action`; goal calls use only `observation_ref`.
    """

    tool: ToolName
    observation_ref: ObservationRef
    action: ActionSpec | None = None


@dataclass(slots=True)
class ToolResult:
    """World or goal model output from a tool call.

    World results include the candidate action. Goal results leave it unset.
    """

    id: str
    tool: ToolName
    predicted_observation: Any
    source_observation_ref: ObservationRef
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
class RewardUpdateQuantities:
    """Structured update signals for the agent updater."""

    prediction_error: float | None = None
    prediction_error_delta: float | None = None
    goal_distance: float | None = None
    time_cost: float | None = None
    trace_cost: float | None = None
    score_delta: float | None = None
    notes: str | None = None


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
class MemoryRecord:
    """Generic record persisted in state memory M or experimental memory E."""

    id: int
    domain: MemoryDomain
    run_id: str
    game_id: str
    step: int | None
    kind: str
    payload: dict[str, Any]
    created_at: str


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
    chosen_action: dict[str, Any]
    world_context: RoleContext
    goal_context: RoleContext
    agent_context: RoleContext
    agent_trace: dict[str, Any]
    world_prediction: dict[str, Any] | None
    goal_prediction: dict[str, Any] | None
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
    source_observation_ref: ObservationRef
    tool_call: dict[str, Any]
    output_observation: dict[str, Any]
    tool_result: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class ExperimentToolInvocationResult:
    """Result of an orchestration-mediated tool call persisted to E."""

    tool_result: ToolResult
    observation_ref: ObservationRef
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
