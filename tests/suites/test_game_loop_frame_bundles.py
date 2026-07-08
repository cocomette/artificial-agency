"""Tests for game-loop frame bundle transition handling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryScoreAdvanceMarker,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    DecisionResult,
    EnvironmentInfo,
    FrameControlMode,
    Observation,
    ObservationRef,
    RuntimeConfig,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.orchestration.game_loop.actions import steps
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)
from face_of_agi.debug.bus import DebugBus


class BundleStepEnvironment:
    """Fake environment that returns retained animation frames after one action."""

    def __init__(self, frames: tuple[Image.Image, ...]) -> None:
        self.frames = frames
        self.step_actions: list[ActionSpec] = []

    def step(self, action: ActionSpec) -> Observation:
        self.step_actions.append(action)
        return Observation(id="after-action", step=1, frames=self.frames)

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (ActionSpec("ACTION1"),)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class RecordingChangeModel:
    """Change-summary test double that records bundled frame observations."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def summarize(
        self,
        previous_observation: Observation,
        current_observation: Observation,
        action: ActionSpec,
        *,
        glossary_actions: Sequence[ActionSpec],
        frame_observations: Sequence[Observation] | None = None,
        previous_change_elements: Sequence[ChangeSummaryElement] = (),
    ) -> ChangeSummaryResult:
        self.calls.append(
            {
                "previous_observation": previous_observation,
                "current_observation": current_observation,
                "action": action,
                "glossary_actions": tuple(glossary_actions),
                "frame_observations": tuple(frame_observations or ()),
                "previous_change_elements": tuple(previous_change_elements),
            }
        )
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="bundle",
                    element_description="transition bundle",
                    element_mutation="summarized bundle",
                ),
            ),
            changed_pixel_count=1,
            change_detected=True,
            metadata={},
            changed_pixel_percent=25.0,
        )


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (2, 2), color=color)


def _session_for_controllable_step(
    *,
    current_observation: Observation,
    next_frames: tuple[Image.Image, ...],
) -> GameLoopSession:
    action = ActionSpec("ACTION1")
    current_ref = ObservationRef(memory="state", id=current_observation.id)
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=BundleStepEnvironment(next_frames),
        environment_config=EnvironmentConfig(
            game_index=0,
            max_actions_per_level=10,
            animation_keyframe_pixel_threshold=1,
        ),
        game_id="game-1",
        latest_environment_observation=current_observation,
        remaining_actions=10,
        real_actions=(action,),
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=current_observation,
        observation_ref=current_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=current_ref,
    )
    session.decision = DecisionResult(
        final_action=action,
        trace=AgentTrace(
            step=0,
            first_observation_ref=current_ref,
            current_observation_ref=current_ref,
            final_action=action,
        ),
    )
    session.trace_cost_seconds = 0.0
    return session


def test_controllable_step_summarizes_bundle_and_advances_to_final_frame() -> None:
    current = Observation(id="current", step=0, frame=_image((0, 0, 0)))
    middle_frame = _image((127, 127, 127))
    final_frame = _image((255, 255, 255))
    session = _session_for_controllable_step(
        current_observation=current,
        next_frames=(middle_frame, final_frame),
    )

    steps.resolve_next_snapshot(session, debug=DebugBus.disabled())

    assert len(session.transition_frame_observations) == 3
    assert session.transition_frame_observations[0].id == "current"
    assert session.transition_frame_observations[1].frame is middle_frame
    assert session.transition_frame_observations[2].frame is final_frame
    assert session.next is not None
    assert session.next.observation.frame is final_frame
    assert session.next.frame_count == 1
    assert session.next_frame_buffer == (session.next.observation,)

    change_model = RecordingChangeModel()
    result = steps.summarize_change_model(
        session,
        change_model=change_model,
        debug=DebugBus.disabled(),
    )

    assert result.elements[0].element_mutation == "summarized bundle"
    assert len(change_model.calls) == 1
    call = change_model.calls[0]
    assert call["previous_observation"] is session.transition_frame_observations[0]
    assert call["current_observation"] is session.transition_frame_observations[-1]
    assert call["frame_observations"] == session.transition_frame_observations


def test_change_summary_previous_elements_clear_after_score_advance() -> None:
    current = Observation(id="current", step=0, frame=_image((0, 0, 0)))
    final_frame = _image((255, 255, 255))
    session = _session_for_controllable_step(
        current_observation=current,
        next_frames=(final_frame,),
    )
    steps.resolve_next_snapshot(session, debug=DebugBus.disabled())
    element = ChangeSummaryElement(
        element_name="cursor",
        element_description="small white square",
        element_mutation="moved right",
    )
    session.action_history.append(
        ActionHistoryEntry(
            action=ActionSpec("ACTION1"),
            controllable=True,
            changed_pixel_count=1,
            change_summary="cursor moved",
            change_elements=(element,),
        )
    )
    change_model = RecordingChangeModel()

    steps.summarize_change_model(
        session,
        change_model=change_model,
        debug=DebugBus.disabled(),
    )

    assert change_model.calls[-1]["previous_change_elements"] == (element,)

    session.action_history.append(
        ActionHistoryScoreAdvanceMarker(
            previous_score=0,
            new_score=1,
            delta=1,
        )
    )
    steps.summarize_change_model(
        session,
        change_model=change_model,
        debug=DebugBus.disabled(),
    )

    assert change_model.calls[-1]["previous_change_elements"] == ()
