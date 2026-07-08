"""Tests for threshold-selected animation keyframes in the learner loop."""

from __future__ import annotations

from collections.abc import Sequence

from face_of_agi.contracts import (
    ActionSpec,
    EnvironmentInfo,
    Observation,
    RuntimeConfig,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import DebugEvent, FrameTurnCompleted
from face_of_agi.environment.config import (
    AgentRuntimeConfig,
    BackboneRuntimeConfig,
    EnvironmentConfig,
    OnlineRuntimeConfig,
    ReplayRuntimeConfig,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.online.agent import OnlineLearnerAgent
from face_of_agi.online.backbone import DeterministicBackbone
from face_of_agi.orchestration.game_loop.frame_turns import unroll_observation
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


def test_state_machine_persists_learner_trace_on_retained_animation_keyframes(
    tmp_path,
) -> None:
    environment = _FakeEnvironment()
    agent = OnlineLearnerAgent(
        config=_agent_config(),
        backbone=DeterministicBackbone(feature_dim=8),
    )
    sink = _EventSink()
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))

    result = GameLoopStateMachine(
        state_memory=memory,
        agent=agent,
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

    rows = memory.list_states(game_id="game-1")
    assert result.stop_reason == "action_limit_reached"
    assert result.state_record_ids == tuple(row.id for row in rows)
    assert agent.real_transition_count == 2
    assert agent.frame_turn_count == 3
    assert [row.learner_trace["transition"]["controllable"] for row in rows] == [
        True,
        False,
        True,
    ]
    assert [
        row.learner_trace["decision"]["final_action"]["action_id"]
        for row in rows
    ] == ["ACTION1", "NONE", "ACTION1"]
    assert [
        event.controllable
        for event in sink.events
        if isinstance(event, FrameTurnCompleted)
    ] == [True, False, True]


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


class _EventSink:
    def __init__(self) -> None:
        self.events: list[DebugEvent] = []

    def emit(self, event: DebugEvent) -> None:
        self.events.append(event)


def _agent_config() -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        backbone=BackboneRuntimeConfig(
            backend="deterministic",
            model_path="unused-test-backbone",
        ),
        online=OnlineRuntimeConfig(hidden_dim=8, ensemble_size=2, batch_size=2),
        replay=ReplayRuntimeConfig(max_updates_per_turn=1, max_seconds_per_turn=1.0),
    )


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
