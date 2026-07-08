"""Tests for threshold-selected animation keyframes."""

from __future__ import annotations

from collections.abc import Sequence
import json
import sqlite3

from face_of_agi.contracts import (
    ActionSpec,
    AgentCandidateAction,
    AgentTrace,
    CandidateValuePrediction,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    Observation,
    ObservationRef,
    RewardJudgeScore,
    RuntimeConfig,
    WorldPrediction,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import DebugEvent, FrameTurnCompleted
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.memory import MemoryLedgerEntry
from face_of_agi.orchestration.game_loop.helpers import unroll_observation
from face_of_agi.orchestration.game_loop.state_machine import GameLoopStateMachine

def test_animation_keyframes_continue_from_last_retained_frame() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(1), _frame(2), _frame(3), _frame(4)),
    )

    frames = unroll_observation(
        observation,
        animation_keyframe_pixel_threshold=2,
        anchor_frame=_frame(0),
    )

    assert [item.frame for item in frames] == [_frame(2), _frame(4)]
    assert [
        item.metadata["skipped_intermediate_animation_frame_count"]
        for item in frames
    ] == [1, 1]


def test_animation_keyframes_fall_back_to_final_frame_when_threshold_never_hits() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_single_pixel_frame(0), _single_pixel_frame(1)),
    )

    frames = unroll_observation(
        observation,
        animation_keyframe_pixel_threshold=2,
        anchor_frame=_frame(0),
    )

    assert [item.frame for item in frames] == [_single_pixel_frame(1)]
    assert frames[0].metadata["skipped_intermediate_animation_frame_count"] == 1


def test_animation_keyframes_keep_final_frame_even_below_threshold_after_hit() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(2), _frame(3)),
    )

    frames = unroll_observation(
        observation,
        animation_keyframe_pixel_threshold=2,
        anchor_frame=_frame(0),
    )

    assert [item.frame for item in frames] == [_frame(2), _frame(3)]
    assert frames[1].metadata["skipped_intermediate_animation_frame_count"] == 0


def test_animation_keyframes_collapse_exact_consecutive_duplicates() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(1), _frame(1), _frame(2)),
    )

    frames = unroll_observation(
        observation,
        animation_keyframe_pixel_threshold=0,
        anchor_frame=_frame(0),
    )

    assert [item.frame for item in frames] == [_frame(1), _frame(2)]
    assert [item.metadata["bundle_frame_index"] for item in frames] == [1, 2]
    assert frames[0].metadata["skipped_intermediate_animation_frame_count"] == 1


def test_animation_keyframes_threshold_zero_keeps_every_non_duplicate_frame() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(1), _frame(2), _frame(3)),
    )

    frames = unroll_observation(
        observation,
        animation_keyframe_pixel_threshold=0,
        anchor_frame=_frame(0),
    )

    assert [item.frame for item in frames] == [_frame(1), _frame(2), _frame(3)]


def test_state_machine_summarizes_retained_animation_bundle_on_real_action() -> None:
    environment = _FakeEnvironment()
    agent = _FakeAgent()
    change_model = _FakeChangeSummary()
    memory = _FakeMemory()
    sink = _EventSink()

    result = GameLoopStateMachine(
        state_memory=None,
        contexts=ContextDocuments(),
        agent=agent,
        change_summary_model=change_model,
        memory_model=memory,
        world_model=_FakeWorld(),
        goal_model=_FakeGoal(),
        interest_model=_FakeInterest(),
        reward_judge_model=_FakeRewardJudge(),
        debug=DebugBus(sink=sink),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=2,
            animation_keyframe_pixel_threshold=2,
        ),
    )

    assert result.stop_reason == "action_limit_reached"
    assert agent.call_count == 2
    assert [action.name for action in change_model.actions] == ["ACTION1", "ACTION1"]
    assert [len(frames) for frames in change_model.frame_observations] == [3, 2]
    entry = memory.ledger_snapshots[-1][-1]
    assert isinstance(entry, MemoryLedgerEntry)
    assert entry.action == "ACTION1"
    assert entry.change_summary == "changed"
    assert [
        event.controllable
        for event in sink.events
        if isinstance(event, FrameTurnCompleted)
    ] == [True, True]


def test_state_machine_skips_change_model_when_visible_frames_match() -> None:
    environment = _OneStepEnvironment(step_frame=_frame(0))
    agent = _FakeAgent()
    change_model = _FakeChangeSummary()
    memory = _FakeMemory()

    GameLoopStateMachine(
        state_memory=None,
        contexts=ContextDocuments(),
        agent=agent,
        change_summary_model=change_model,
        memory_model=memory,
        world_model=_FakeWorld(),
        goal_model=_FakeGoal(),
        interest_model=_FakeInterest(),
        reward_judge_model=_FakeRewardJudge(),
        debug=DebugBus(),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert change_model.actions == []
    entry = memory.ledger_snapshots[-1][-1]
    assert entry.change_summary == "no changes"
    assert entry.action == "ACTION1"


def test_state_machine_overrides_change_summary_when_pixels_changed_but_model_says_no() -> None:
    environment = _OneStepEnvironment(step_frame=_frame(1))
    agent = _FakeAgent()
    change_model = _FakeChangeSummary(
        summary="No visible change.",
        change_detected=False,
    )
    memory = _FakeMemory()

    GameLoopStateMachine(
        state_memory=None,
        contexts=ContextDocuments(),
        agent=agent,
        change_summary_model=change_model,
        memory_model=memory,
        world_model=_FakeWorld(),
        goal_model=_FakeGoal(),
        interest_model=_FakeInterest(),
        reward_judge_model=_FakeRewardJudge(),
        debug=DebugBus(),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert len(change_model.actions) == 1
    entry = memory.ledger_snapshots[-1][-1]
    assert (
        entry.change_summary
        == "visible pixels changed, but the specific change is uncertain."
    )


def test_real_turn_persists_reward_and_sanitizes_memory_ledger(
    tmp_path,
) -> None:
    environment = _OneStepEnvironment(step_frame=_frame(1))
    agent = _FakeAgent()
    change_model = _FakeChangeSummary()
    memory_model = _FakeMemory()
    goal_model = _FakeGoal()
    state_memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))

    GameLoopStateMachine(
        state_memory=state_memory,
        contexts=ContextDocuments(),
        agent=agent,
        change_summary_model=change_model,
        memory_model=memory_model,
        world_model=_FakeWorld(),
        goal_model=goal_model,
        interest_model=_FakeInterest(),
        reward_judge_model=_FakeRewardJudge(),
        debug=DebugBus(),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert len(memory_model.ledger_snapshots) == 2
    assert goal_model.call_count == 3
    memory_entry = memory_model.ledger_snapshots[-1][-1]
    assert isinstance(memory_entry, MemoryLedgerEntry)
    assert memory_entry.action == "ACTION1"
    assert memory_entry.change_summary == "changed"
    assert memory_entry.reward_feedback is not None
    assert memory_entry.reward_feedback["prediction_accuracy"] == 1.0
    assert memory_entry.reward_feedback["learning_progress"] == 1.0
    assert memory_entry.reward_feedback["judge_notes"] == "match"

    reward = _latest_reward(state_memory.database.path)
    assert reward["prediction_accuracy"] == 1.0
    assert reward["learning_progress"] == 1.0
    assert reward["metadata"]["learning_progress_proxy"] == (
        "reward_judge_prediction_accuracy"
    )
    assert reward["goal_delta"] > 0.0
    assert reward["total"] > reward["prediction_accuracy"] * reward["lp_weight"]


class _FakeEnvironment:
    def __init__(self) -> None:
        self.step_count = 0

    def list_available_games(self):
        return ()

    def list_local_games(self):
        return ()

    def resolve_game_id(self, game_index: int) -> str:
        del game_index
        return "game-1"

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        return Observation(id="reset", step=0, frame=_frame(0), frames=(_frame(0),))

    def step(self, action: ActionSpec, reasoning=None) -> Observation:
        del action, reasoning
        self.step_count += 1
        if self.step_count == 1:
            return Observation(
                id="step-1",
                step=1,
                frames=(_frame(1), _frame(2), _frame(3)),
            )
        return Observation(id="step-2", step=2, frame=_frame(4), frames=(_frame(4),))

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (ActionSpec(action_id="ACTION1"),)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class _OneStepEnvironment:
    def __init__(self, *, step_frame) -> None:
        self.step_frame = step_frame

    def list_available_games(self):
        return ()

    def list_local_games(self):
        return ()

    def resolve_game_id(self, game_index: int) -> str:
        del game_index
        return "game-1"

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        return Observation(id="reset", step=0, frame=_frame(0), frames=(_frame(0),))

    def step(self, action: ActionSpec, reasoning=None) -> Observation:
        del action, reasoning
        return Observation(
            id="step-1",
            step=1,
            frame=self.step_frame,
            frames=(self.step_frame,),
        )

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (ActionSpec(action_id="ACTION1"),)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class _FakeAgent:
    def __init__(self) -> None:
        self.call_count = 0

    def propose_candidate_actions(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        glossary_actions: Sequence[ActionSpec],
        max_candidates: int,
        recent_action_history=(),
    ) -> tuple[AgentCandidateAction, ...]:
        del memory, goal, current_observation, action_space, glossary_actions
        del max_candidates, recent_action_history
        return ()

    def select_action(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        candidates: Sequence[AgentCandidateAction],
        world_predictions: Sequence[WorldPrediction],
        interest_prediction: InterestPrediction | None = None,
        first_observation_ref: ObservationRef | None = None,
        recent_action_history=(),
        glossary_actions: Sequence[ActionSpec],
    ) -> DecisionResult:
        del memory, goal, world_predictions, interest_prediction
        del recent_action_history, glossary_actions
        self.call_count += 1
        action = candidates[0].action
        current_ref = ObservationRef(memory="state", id=current_observation.id)
        trace = AgentTrace(
            step=current_observation.step,
            first_observation_ref=first_observation_ref or current_ref,
            current_observation_ref=current_ref,
            final_action=action,
        )
        return DecisionResult(final_action=action, trace=trace)


class _FakeChangeSummary:
    def __init__(
        self,
        *,
        summary: str = "changed",
        change_detected: bool = True,
    ) -> None:
        self.summary = summary
        self.change_detected = change_detected
        self.actions: list[ActionSpec] = []
        self.frame_observations: list[tuple[Observation, ...]] = []

    def summarize(
        self,
        previous_observation: Observation,
        current_observation: Observation,
        action: ActionSpec,
        *,
        glossary_actions: Sequence[ActionSpec],
        changed_pixel_percent: float,
        frame_observations: Sequence[Observation] | None = None,
        max_transition_changed_pixel_percent: float | None = None,
    ) -> ChangeSummaryResult:
        del (
            previous_observation,
            current_observation,
            glossary_actions,
            changed_pixel_percent,
            max_transition_changed_pixel_percent,
        )
        self.actions.append(action)
        self.frame_observations.append(tuple(frame_observations or ()))
        return ChangeSummaryResult(
            summary=self.summary,
            changed_pixel_percent=1,
            change_detected=self.change_detected,
            metadata={},
        )


class _FakeMemory:
    def __init__(self) -> None:
        self.ledger_snapshots: list[tuple[MemoryLedgerEntry, ...]] = []

    def build_memory(
        self,
        build_input,
    ) -> MemoryDocument:
        self.ledger_snapshots.append(tuple(build_input.ledger))
        return MemoryDocument(document=f"memory {len(build_input.ledger)}")


class _FakeGoal:
    def __init__(self) -> None:
        self.call_count = 0

    def predict_goal(self, prediction_input) -> GoalPrediction:
        del prediction_input
        self.call_count += 1
        return GoalPrediction(
            goal="solve",
            subgoals=("advance",),
            steps_remaining=max(0, 10 - self.call_count),
            confidence=0.5,
        )


class _FakeWorld:
    def predict_transition(self, prediction_input) -> WorldPrediction:
        return WorldPrediction(
            candidate_index=prediction_input.candidate_index,
            action=prediction_input.action,
            predicted_change="changed",
        )


class _FakeInterest:
    def score_candidates(self, prediction_input) -> InterestPrediction:
        values = tuple(
            CandidateValuePrediction(
                candidate_index=candidate.rank,
                action=candidate.action,
                expected_learning_progress=0.4,
                expected_goal_delta=0.2,
                confidence=0.5,
                notes="interesting",
            )
            for candidate in prediction_input.candidates
        )
        return InterestPrediction(
            candidate_values=values,
        )


class _FakeRewardJudge:
    def judge_prediction(self, judge_input) -> RewardJudgeScore:
        del judge_input
        return RewardJudgeScore(score=1.0, notes="match")


class _EventSink:
    def __init__(self) -> None:
        self.events: list[DebugEvent] = []

    def emit(self, event: DebugEvent) -> None:
        self.events.append(event)


def _frame(changed_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(1 if row * 3 + column < changed_count else 0 for column in range(3))
        for row in range(3)
    )


def _single_pixel_frame(index: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(1 if row * 3 + column == index else 0 for column in range(3))
        for row in range(3)
    )


def _latest_reward(path) -> dict:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT reward_json FROM rewards ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise AssertionError("expected one persisted reward")
    return json.loads(str(row[0]))
