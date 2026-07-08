"""Frame-unrolled game-loop state machine for the online learner."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryScoreAdvanceMarker,
    ActionSpec,
    FrameControlMode,
    FrameTurnContext,
    GameRunResult,
    Observation,
    ObservationRef,
    RuntimeConfig,
    TransitionRecord,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    EnvironmentStepRecorded,
    FrameDecisionRecorded,
    FrameTurnCompleted,
    FrameTurnStarted,
    MStatePersisted,
)
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import StateMemory
from face_of_agi.online.agent import OnlineLearnerAgent
from face_of_agi.orchestration.game_loop.actions.metrics import turn_metrics
from face_of_agi.orchestration.game_loop.frame_turns import (
    bounded_action_history,
    changed_pixel_count,
    frame_control_mode,
    unroll_observation,
    validate_decision,
)
from face_of_agi.orchestration.game_loop.lifecycle import (
    check_lifecycle,
    check_runtime_deadline,
    finish_run,
    start_run,
)
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)
from face_of_agi.runtime import timing as runtime_timing


class GameLoopStateMachine:
    """Run one ARC game through the online learner frame-turn loop."""

    def __init__(
        self,
        *,
        state_memory: StateMemory | None,
        agent: OnlineLearnerAgent,
        debug: DebugBus,
    ) -> None:
        self.state_memory = state_memory
        self.agent = agent
        self.debug = debug

    def run(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
    ) -> GameRunResult:
        """Run one selected ARC game until a terminal loop condition."""

        session = start_run(
            config=config,
            environment=environment,
            environment_config=environment_config,
            state_memory=self.state_memory,
            debug=self.debug,
        )
        while session.running:
            session.process_turn = True
            if check_runtime_deadline(session):
                continue
            check_lifecycle(session)
            if not session.process_turn:
                continue
            self._process_turn(session)
        return finish_run(session, debug=self.debug)

    def _process_turn(self, session: GameLoopSession) -> None:
        turn_started_at = perf_counter()
        self._load_frame_buffer_if_needed(session)
        current = self._enter_frame_turn(session)
        frame_context = current.to_frame_context()
        if check_runtime_deadline(session):
            return

        if current.control_mode is None:
            raise RuntimeError("current frame snapshot is missing control mode")
        if current.control_mode.controllable:
            decision, planner_candidates, backbone_metadata = self.agent.decide(
                frame_context
            )
        else:
            decision = self.agent.synthetic_none_decision(frame_context)
            planner_candidates = ()
            backbone_metadata = {}
        _validate_action_payload(decision.final_action)
        validate_decision(decision.final_action, control_mode=current.control_mode)
        session.decision = decision
        session.last_decision = decision
        session.frame_turn_count = current.turn_id
        self.debug.emit(
            FrameDecisionRecorded(
                frame_turn=current.turn_id,
                frame_context=frame_context,
                action=decision.final_action,
                trace=decision.trace,
            )
        )
        if check_runtime_deadline(session):
            return

        next_frame = self._resolve_next_frame(session, decision.final_action)
        transition, score_marker = self._build_transition(
            session,
            current=current,
            next_frame=next_frame,
            action=decision.final_action,
        )
        completed_level = score_marker is not None
        learner_trace, learner_snapshot = self.agent.observe_transition(
            frame_context=frame_context,
            decision=decision,
            transition=transition,
            next_observation=next_frame,
            planner_candidates=planner_candidates,
            completed_level=completed_level,
        )
        session.turn_metrics = turn_metrics(
            actual_next_observation=next_frame,
            trace_cost_seconds=None,
            cumulative_time_cost=float(session.real_step_count),
        )
        self._persist_turn(
            session,
            current=current,
            chosen_action=decision.final_action,
            learner_snapshot=learner_snapshot,
            learner_trace=learner_trace,
        )
        self._append_action_history(
            session,
            current=current,
            transition=transition,
            score_marker=score_marker,
            next_frame=next_frame,
        )
        self.debug.emit(
            FrameTurnCompleted(
                run_id=session.config.run_id,
                game_id=session.game_id,
                game_index=session.environment_config.game_index,
                turn_id=current.turn_id,
                env_step=current.observation.step,
                frame_index=current.frame_index,
                frame_count=current.frame_count,
                controllable=current.control_mode.controllable,
                action=decision.final_action,
                turn_duration_seconds=perf_counter() - turn_started_at,
                completed_levels=_completed_levels_after_turn(session),
                remaining_actions=session.remaining_actions,
            )
        )
        self._advance(session, current=current)

    def _load_frame_buffer_if_needed(self, session: GameLoopSession) -> None:
        if session.frame_buffer and session.frame_index < len(session.frame_buffer):
            return
        session.frame_buffer = unroll_observation(
            session.latest_environment_observation,
            animation_keyframe_pixel_threshold=(
                session.environment_config.animation_keyframe_pixel_threshold
            ),
        )
        session.frame_index = 0

    def _enter_frame_turn(self, session: GameLoopSession) -> FrameTurnSnapshot:
        current_observation = session.frame_buffer[session.frame_index]
        frame_count = len(session.frame_buffer)
        control_mode = frame_control_mode(
            frame_index=session.frame_index,
            frame_count=frame_count,
            real_actions=session.real_actions,
        )
        current_ref = session.current_ref_for(current_observation)
        turn_id = session.frame_turn_count + 1
        source_state = (
            self.state_memory.prewrite_frame_turn_source(
                run_id=session.config.run_id,
                game_id=session.game_id,
                turn_id=turn_id,
                current_observation=current_observation,
                frame_index=session.frame_index,
                frame_count=frame_count,
                control_mode=control_mode,
                learner_snapshot=self.agent.snapshot(),
            )
            if self.state_memory is not None
            else None
        )
        if session.first_observation is None:
            session.first_observation = current_observation
            session.first_observation_ref = current_ref
        if session.first_observation_ref is None:
            raise RuntimeError("frame turn is missing the first observation ref")
        recent_history = bounded_action_history(
            session.action_history,
            window=session.environment_config.action_history_window,
            key="action_history_window",
        )
        session.current = FrameTurnSnapshot(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=turn_id,
            observation=current_observation,
            observation_ref=current_ref,
            source_state_id=source_state.id if source_state is not None else None,
            frame_index=session.frame_index,
            frame_count=frame_count,
            control_mode=control_mode,
            first_observation_ref=session.first_observation_ref,
            previous_observation_ref=session.previous_observation_ref,
            recent_action_history=recent_history,
        )
        self.debug.emit(
            FrameTurnStarted(
                frame_turn=turn_id,
                frame_context=session.current.to_frame_context(),
                lifecycle_state=(
                    session.current_info.state
                    if session.current_info is not None
                else None
                ),
                completed_levels=session.completed_levels,
                remaining_actions=session.remaining_actions,
            )
        )
        return session.current

    def _resolve_next_frame(
        self,
        session: GameLoopSession,
        action: ActionSpec,
    ) -> Observation:
        current = _require_current(session)
        if current.control_mode is None:
            raise RuntimeError("current frame snapshot is missing control mode")
        if current.control_mode.controllable:
            session.real_step_count += 1
            with runtime_timing.span(
                "game_loop.environment_step",
                turn_id=current.turn_id,
                step=current.observation.step,
            ):
                next_observation = session.environment.step(action)
            session.remaining_actions -= 1
            session.next_environment_observation = next_observation
            self.debug.emit(
                EnvironmentStepRecorded(
                    action=action,
                    next_observation=next_observation,
                    remaining_actions=session.remaining_actions,
                )
            )
            session.next_frame_buffer = unroll_observation(
                next_observation,
                animation_keyframe_pixel_threshold=(
                    session.environment_config.animation_keyframe_pixel_threshold
                ),
                anchor_frame=current.observation.frame,
            )
            if not session.next_frame_buffer:
                raise RuntimeError("environment response produced no frames")
            return session.next_frame_buffer[0]
        next_index = session.frame_index + 1
        if next_index >= len(session.frame_buffer):
            raise RuntimeError("animation frame turn has no next frame")
        return session.frame_buffer[next_index]

    def _build_transition(
        self,
        session: GameLoopSession,
        *,
        current: FrameTurnSnapshot,
        next_frame: Observation,
        action: ActionSpec,
    ) -> tuple[TransitionRecord, ActionHistoryScoreAdvanceMarker | None]:
        next_ref = session.current_ref_for(next_frame)
        changed_percent = changed_pixel_percent(
            current.observation.frame,
            next_frame.frame,
        )
        metrics = turn_metrics(
            actual_next_observation=next_frame,
            trace_cost_seconds=None,
            cumulative_time_cost=float(session.real_step_count),
        )
        marker = _score_advance_marker(
            previous_score=session.last_observed_cumulative_score,
            new_score=metrics.cumulative_score,
        )
        transition = TransitionRecord(
            previous_observation_ref=current.observation_ref,
            next_observation_ref=next_ref,
            action=action,
            controllable=(
                current.control_mode.controllable
                if current.control_mode is not None
                else False
            ),
            changed_pixel_percent=changed_percent,
            score_delta=(marker.delta if marker is not None else None),
            completed_levels=(
                int(metrics.cumulative_score)
                if metrics.cumulative_score is not None
                else None
            ),
            metadata={
                "next_observation_id": next_frame.id,
                "next_step": next_frame.step,
            },
        )
        return transition, marker

    def _persist_turn(
        self,
        session: GameLoopSession,
        *,
        current: FrameTurnSnapshot,
        chosen_action: ActionSpec,
        learner_snapshot: dict[str, Any],
        learner_trace: Any,
    ) -> None:
        if self.state_memory is None or current.source_state_id is None:
            return
        if current.control_mode is None:
            raise RuntimeError("current frame snapshot is missing control mode")
        state = self.state_memory.complete_frame_turn_state(
            state_id=current.source_state_id,
            turn_id=current.turn_id,
            control_mode=current.control_mode,
            previous_observation_ref=current.previous_observation_ref,
            recent_action_history=current.recent_action_history,
            chosen_action=chosen_action,
            learner_snapshot=learner_snapshot,
            learner_trace=learner_trace,
            turn_metrics=session.turn_metrics,
        )
        session.state_record_ids.append(state.id)
        self.debug.emit(MStatePersisted(record_id=state.id, turn_id=current.turn_id))

    def _append_action_history(
        self,
        session: GameLoopSession,
        *,
        current: FrameTurnSnapshot,
        transition: TransitionRecord,
        score_marker: ActionHistoryScoreAdvanceMarker | None,
        next_frame: Observation,
    ) -> None:
        if current.control_mode is None:
            raise RuntimeError("current frame snapshot is missing control mode")
        session.action_history.append(
            ActionHistoryEntry(
                action=transition.action,
                controllable=current.control_mode.controllable,
                changed_pixel_percent=transition.changed_pixel_percent,
                transition_summary=(
                    f"changed_pixel_percent={transition.changed_pixel_percent:.6f}"
                ),
                skipped_intermediate_animation_frame_count=(
                    _skipped_intermediate_animation_frame_count(next_frame)
                ),
            )
        )
        if score_marker is not None:
            session.action_history.append(score_marker)
            session.last_score_advance_turn_id = current.turn_id
        if session.turn_metrics is not None and session.turn_metrics.cumulative_score is not None:
            session.last_observed_cumulative_score = float(
                session.turn_metrics.cumulative_score
            )

    def _advance(
        self,
        session: GameLoopSession,
        *,
        current: FrameTurnSnapshot,
    ) -> None:
        session.previous_observation_ref = current.observation_ref
        if current.control_mode is not None and current.control_mode.controllable:
            if session.next_environment_observation is None:
                raise RuntimeError("controllable turn is missing next observation")
            if not session.next_frame_buffer:
                raise RuntimeError("controllable turn is missing next frame buffer")
            session.latest_environment_observation = session.next_environment_observation
            session.frame_buffer = session.next_frame_buffer
            session.frame_index = 0
        else:
            session.frame_index += 1
        session.current = None
        session.next = None
        session.decision = None
        session.turn_metrics = None
        session.next_environment_observation = None
        session.next_frame_buffer = ()


def changed_pixel_percent(left: Any, right: Any) -> float:
    """Return changed raw cells/pixels as a percentage."""

    import numpy as np

    if left is None or right is None:
        return 0.0
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    surface = max(_frame_surface_size(left_array), _frame_surface_size(right_array))
    if surface <= 0:
        return 0.0
    return min(100.0, changed_pixel_count(left, right) * 100.0 / surface)


def _frame_surface_size(array: Any) -> int:
    if array.shape == ():
        return 1
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        return int(array.shape[0] * array.shape[1])
    return int(array.size)


def _score_advance_marker(
    *,
    previous_score: float | None,
    new_score: float | None,
) -> ActionHistoryScoreAdvanceMarker | None:
    if new_score is None:
        return None
    current = float(new_score)
    if previous_score is None:
        if current <= 0.0:
            return None
        return ActionHistoryScoreAdvanceMarker(
            previous_score=None,
            new_score=current,
            delta=current,
        )
    previous = float(previous_score)
    if current <= previous:
        return None
    return ActionHistoryScoreAdvanceMarker(
        previous_score=previous,
        new_score=current,
        delta=current - previous,
    )


def _skipped_intermediate_animation_frame_count(observation: Observation) -> int:
    value = observation.metadata.get("skipped_intermediate_animation_frame_count", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _validate_action_payload(action: ActionSpec) -> None:
    if action.name != "ACTION6":
        return
    data = action.data or {}
    for key in ("x", "y"):
        if key not in data:
            raise RuntimeError(f"ACTION6 data missing {key!r}")
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"ACTION6 data {key!r} must be numeric")
        numeric = float(value)
        if not numeric.is_integer():
            raise RuntimeError(f"ACTION6 data {key!r} must be an ARC grid integer")
        if not 0 <= numeric <= 63:
            raise RuntimeError(f"ACTION6 data {key!r} must be in ARC grid 0..63")


def _completed_levels_after_turn(session: GameLoopSession) -> int:
    metrics = session.turn_metrics
    if metrics is not None and metrics.cumulative_score is not None:
        return int(metrics.cumulative_score)
    return int(session.completed_levels)


def _require_current(session: GameLoopSession) -> FrameTurnSnapshot:
    if session.current is None:
        raise RuntimeError("game-loop session is missing the current turn")
    return session.current
