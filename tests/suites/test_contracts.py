"""Smoke tests for public online-learner contracts."""

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    DecisionResult,
    FrameControlMode,
    LearnerTurnTrace,
    Observation,
    ObservationRef,
    PlannerCandidate,
    ReplayStats,
    RuntimeConfig,
    TransitionRecord,
)


def test_contracts_import_and_compose_learner_trace() -> None:
    observation = Observation(id="obs-0", step=0, frame=object())
    action = ActionSpec(action_id="ACTION1")
    ref = ObservationRef(memory="state", id=observation.id)
    decision = DecisionResult(
        final_action=action,
        trace=AgentTrace(
            step=0,
            first_observation_ref=ref,
            current_observation_ref=ref,
            final_action=action,
        ),
    )
    trace = LearnerTurnTrace(
        decision=decision,
        transition=TransitionRecord(
            previous_observation_ref=ref,
            next_observation_ref=ObservationRef(memory="state", id="obs-1"),
            action=action,
            controllable=True,
            changed_pixel_percent=1.0,
            prediction_error=0.5,
        ),
        replay=ReplayStats(real_update_count=1),
        planner_candidates=(PlannerCandidate(action=action, score=0.75),),
    )
    config = RuntimeConfig(run_id="run-1", game_ids=("game-1",))

    assert trace.decision.final_action.action_id == "ACTION1"
    assert trace.transition is not None
    assert trace.transition.prediction_error == 0.5
    assert trace.planner_candidates[0].score == 0.75
    assert config.game_ids == ("game-1",)


def test_animation_unroll_keeps_real_allowed_actions() -> None:
    action = ActionSpec(action_id="ACTION1")
    control_mode = FrameControlMode.animation_unroll((action,))

    assert control_mode.controllable is False
    assert control_mode.allowed_actions == (action,)
