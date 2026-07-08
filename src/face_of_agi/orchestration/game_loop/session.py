"""In-memory run cursor and frame-turn snapshots for the game loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionSpec,
    DecisionResult,
    EnvironmentInfo,
    FrameControlMode,
    FrameTurnContext,
    GameRunResult,
    Observation,
    ObservationRef,
    RuntimeConfig,
    TurnMetrics,
    UpdaterFrameTransitionInput,
)
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.models.orchestrator_agent import AgentToolRuntime


@dataclass(frozen=True, slots=True)
class FrameTurnSnapshot:
    """Immutable input snapshot for one frame-turn pass through orchestration."""

    run_id: str
    game_id: str
    turn_id: int
    observation: Observation
    observation_ref: ObservationRef
    source_state_id: int | None
    frame_index: int
    frame_count: int
    control_mode: FrameControlMode | None
    first_observation_ref: ObservationRef
    previous_observation_ref: ObservationRef | None = None
    recent_action_history: tuple[ActionHistoryItem, ...] = ()

    def to_frame_context(self) -> FrameTurnContext:
        """Return the existing shared contract used by model/update boundaries."""

        if self.control_mode is None:
            raise RuntimeError("frame-turn snapshot is missing a control mode")
        return FrameTurnContext(
            run_id=self.run_id,
            game_id=self.game_id,
            first_observation_ref=self.first_observation_ref,
            current_observation_ref=self.observation_ref,
            current_observation=self.observation,
            current_source_state_id=self.source_state_id,
            frame_index=self.frame_index,
            frame_count=self.frame_count,
            control_mode=self.control_mode,
            previous_observation_ref=self.previous_observation_ref,
            recent_action_history=self.recent_action_history,
        )


@dataclass(slots=True)
class GameLoopSession:
    """Mutable in-memory working state for one game-loop run."""

    config: RuntimeConfig
    environment: EnvironmentAdapter
    environment_config: EnvironmentConfig
    game_id: str
    latest_environment_observation: Observation
    remaining_actions: int
    current_info: EnvironmentInfo | None = None
    real_actions: tuple[ActionSpec, ...] = ()
    frame_buffer: tuple[Observation, ...] = ()
    frame_index: int = 0
    current: FrameTurnSnapshot | None = None
    next: FrameTurnSnapshot | None = None
    previous_observation: Observation | None = None
    tool_runtime: AgentToolRuntime | None = None
    decision: DecisionResult | None = None
    decision_duration_seconds: float | None = None
    trace_cost_seconds: float | None = None
    turn_metrics: TurnMetrics | None = None
    update_input: UpdaterFrameTransitionInput | None = None
    next_environment_observation: Observation | None = None
    next_frame_buffer: tuple[Observation, ...] = ()
    last_transition_frame_observations: tuple[Observation, ...] = ()
    real_step_count: int = 0
    frame_turn_count: int = 0
    game_start_turn_id: int = 1
    game_start_reason: str = "initial_start"
    game_restart_count: int = 0
    completed_levels: int = 0
    last_completed_levels: int = 0
    first_observation: Observation | None = None
    first_observation_ref: ObservationRef | None = None
    previous_observation_ref: ObservationRef | None = None
    last_decision: DecisionResult | None = None
    queued_updater_actions: tuple[ActionSpec, ...] = ()
    queued_updater_mode: str | None = None
    pending_game_over_reset: bool = False
    world_model_context: dict[str, Any] | None = None
    agent_context_strategy_snapshot: dict[str, Any] | None = None
    action_history: list[ActionHistoryItem] = field(default_factory=list)
    state_record_ids: list[int] = field(default_factory=list)
    running: bool = True
    process_turn: bool = True
    terminal_result: GameRunResult | None = None

    def current_ref_for(self, observation: Observation) -> ObservationRef:
        """Return the stable state-memory observation ref for an observed frame."""

        return ObservationRef(memory="state", id=observation.id)
