"""Tests for bundled animation frame handling."""

from __future__ import annotations

from collections.abc import Sequence
import json

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    Observation,
    RuntimeConfig,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import DebugEvent, FrameTurnCompleted
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.historizer import (
    AgentContextHistoryInput,
    AgentContextHistorySummary,
)
from face_of_agi.models.world import AgentContextWorldSummary, AgentWorldModelInput
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
    UpdaterTaskRegistry,
)
from face_of_agi.orchestration.game_loop.helpers import (
    bundle_frame_observations,
    change_summary_transition_frame_observations,
    unroll_observation,
)
from face_of_agi.orchestration.game_loop.state_machine import GameLoopStateMachine


def test_unroll_observation_collapse_exact_consecutive_duplicates() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(1), _frame(1), _frame(2)),
    )

    frames = unroll_observation(observation)

    assert [item.frame for item in frames] == [_frame(1), _frame(2)]
    assert [item.metadata["bundle_frame_index"] for item in frames] == [1, 2]
    assert frames[0].metadata["skipped_intermediate_animation_frame_count"] == 1


def test_bundle_frame_observations_collapse_exact_consecutive_duplicates() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(1), _frame(1), _frame(2), _frame(2)),
    )

    frames = bundle_frame_observations(observation)

    assert [item.frame for item in frames] == [_frame(1), _frame(2)]
    assert [item.metadata["frame_count"] for item in frames] == [2, 2]
    assert [item.metadata["bundle_frame_index"] for item in frames] == [1, 3]
    assert [
        item.metadata["skipped_intermediate_animation_frame_count"]
        for item in frames
    ] == [1, 1]


def test_unroll_observation_keeps_every_non_duplicate_frame() -> None:
    observation = Observation(
        id="bundle",
        step=1,
        frames=(_frame(1), _frame(2), _frame(3)),
    )

    frames = unroll_observation(observation)

    assert [item.frame for item in frames] == [_frame(1), _frame(2), _frame(3)]


def test_change_summary_transition_frames_snapshot_previous_observation() -> None:
    from PIL import Image

    previous_frame = Image.new("RGB", (64, 64), (10, 0, 0))
    previous = Observation(id="previous", step=0, frame=previous_frame)
    next_observation = Observation(
        id="next",
        step=1,
        frames=(
            Image.new("RGB", (64, 64), (20, 0, 0)),
            Image.new("RGB", (64, 64), (20, 0, 0)),
            Image.new("RGB", (64, 64), (30, 0, 0)),
            Image.new("RGB", (64, 64), (10, 0, 0)),
            Image.new("RGB", (64, 64), (10, 0, 0)),
        ),
    )

    frames = change_summary_transition_frame_observations(
        previous_observation=previous,
        next_observation=next_observation,
        crop_edges=None,
    )
    previous_frame.paste((99, 0, 0), (0, 0, 64, 64))

    assert [item.frame.getpixel((0, 0)) for item in frames] == [
        (10, 0, 0),
        (20, 0, 0),
        (30, 0, 0),
        (10, 0, 0),
        (10, 0, 0),
    ]


def test_change_summary_transition_frames_dedupe_on_cropped_playfield() -> None:
    from PIL import Image

    previous_frame = Image.new("RGB", (64, 64), (0, 0, 0))
    edge_only = previous_frame.copy()
    edge_only.putpixel((0, 10), (255, 0, 0))
    visible_change = previous_frame.copy()
    visible_change.putpixel((10, 10), (255, 0, 0))

    frames = change_summary_transition_frame_observations(
        previous_observation=Observation(id="previous", step=0, frame=previous_frame),
        next_observation=Observation(
            id="next",
            step=1,
            frames=(edge_only, visible_change),
        ),
        crop_edges=(1, 0, 0, 0),
    )

    assert [item.id for item in frames] == ["previous", "next-frame-1"]
    assert [item.frame.size for item in frames] == [(63, 64), (63, 64)]


def test_change_summary_transition_frames_keep_final_duplicate() -> None:
    from PIL import Image

    previous_frame = Image.new("RGB", (64, 64), (1, 0, 0))
    animation_frame = Image.new("RGB", (64, 64), (2, 0, 0))
    final_frame = Image.new("RGB", (64, 64), (2, 0, 0))

    frames = change_summary_transition_frame_observations(
        previous_observation=Observation(id="previous", step=0, frame=previous_frame),
        next_observation=Observation(
            id="next",
            step=1,
            frames=(animation_frame, final_frame),
        ),
        crop_edges=None,
    )

    assert [item.id for item in frames] == [
        "previous",
        "next-frame-0",
        "next-frame-1",
    ]


def test_state_machine_updates_action_at_controllable_animation_boundary(tmp_path) -> None:
    environment = _FakeEnvironment()
    change_model = _FakeChangeSummary()
    world_model = _FakeWorldModel()
    historizer = _FakeHistorizer()
    updater = _FakeAgentGameUpdater()
    sink = _EventSink()

    result = GameLoopStateMachine(
        state_memory=StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite")),
        contexts=ContextDocuments(),
        change_summary_model=change_model,
        world_model=world_model,
        agent_context_historizer=historizer,
        updater_tasks=UpdaterTaskRegistry(
            agent_probing_updater=updater,
            agent_policy_updater=updater,
        ),
        debug=DebugBus(sink=sink),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=2,
            agent_context_history_window=10,
        ),
    )

    assert result.stop_reason == "action_limit_reached"
    assert [action.name for action in change_model.actions] == ["ACTION1"]
    assert change_model.frame_observation_counts == [4]
    assert [
        item.current_observation.id
        for item in world_model.inputs
    ] == [
        "step-1-frame-1",
    ]
    assert [entry.action.name for entry in world_model.inputs[0].action_history] == [
        "ACTION1",
        "NONE",
    ]
    animation_entry = world_model.inputs[0].action_history[-1]
    assert (
        animation_entry.change_summary
        == "- element: visible object; mutations: changed"
    )
    assert animation_entry.animation_frame_count == 3
    assert animation_entry.avg_changed_pixel_count == 7.4074
    assert not animation_entry.controllable
    assert [
        item.current_observation.id if item.current_observation else None
        for item in historizer.history_inputs
    ] == ["step-1-frame-1"]
    assert historizer.history_inputs[0].strategy_history == (
        '{\n  "probing_strategy": "updated 1",\n'
        '  "policy_strategy": ""\n}',
    )
    first_updater_input = updater.update_inputs[0]
    assert first_updater_input.current_observation.id == "reset"
    assert first_updater_input.context_history.world_description == ""
    assert first_updater_input.context_history.action_effects == {}
    assert first_updater_input.context_history.updater_mode == "probing"
    assert updater.update_inputs[1].current_observation.id == "step-1-frame-1"
    assert [
        event.controllable
        for event in sink.events
        if isinstance(event, FrameTurnCompleted)
    ] == [True, True]


def test_state_machine_snapshots_previous_frame_before_environment_step(tmp_path) -> None:
    environment = _MutatingPreviousFrameEnvironment()
    change_model = _FakeChangeSummary()
    updater = _FakeAgentGameUpdater()

    GameLoopStateMachine(
        state_memory=StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite")),
        contexts=ContextDocuments(),
        change_summary_model=change_model,
        world_model=_FakeWorldModel(),
        agent_context_historizer=_FakeHistorizer(),
        updater_tasks=UpdaterTaskRegistry(
            agent_probing_updater=updater,
            agent_policy_updater=updater,
        ),
        debug=DebugBus(),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=2,
            agent_context_history_window=10,
        ),
    )

    assert change_model.frame_pixels == [
        [(10, 0, 0), (20, 0, 0), (30, 0, 0)]
    ]


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
                frames=(_frame(1), _frame(1), _frame(2), _frame(2)),
            )
        return Observation(id="step-2", step=2, frame=_frame(4), frames=(_frame(4),))

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (ActionSpec(action_id="ACTION1"),)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class _MutatingPreviousFrameEnvironment(_FakeEnvironment):
    def reset(self) -> Observation:
        from PIL import Image

        self.current_frame = Image.new("RGB", (64, 64), (10, 0, 0))
        return Observation(
            id="reset",
            step=0,
            frame=self.current_frame,
            frames=(self.current_frame,),
        )

    def step(self, action: ActionSpec, reasoning=None) -> Observation:
        del action, reasoning
        from PIL import Image

        self.current_frame.paste((99, 0, 0), (0, 0, 64, 64))
        return Observation(
            id="step-1",
            step=1,
            frames=(
                Image.new("RGB", (64, 64), (20, 0, 0)),
                Image.new("RGB", (64, 64), (20, 0, 0)),
                Image.new("RGB", (64, 64), (30, 0, 0)),
            ),
        )


class _FakeChangeSummary:
    def __init__(self) -> None:
        self.actions: list[ActionSpec] = []
        self.frame_observation_counts: list[int] = []
        self.frame_pixels: list[list[tuple[int, int, int]]] = []

    def summarize(
        self,
        previous_observation: Observation,
        current_observation: Observation,
        action: ActionSpec,
        *,
        glossary_actions: Sequence[ActionSpec],
        frame_observations: Sequence[Observation] | None = None,
        previous_change_elements: Sequence[ChangeSummaryElement],
    ) -> ChangeSummaryResult:
        del previous_observation, current_observation, glossary_actions
        del previous_change_elements
        self.actions.append(action)
        self.frame_observation_counts.append(len(frame_observations or ()))
        self.frame_pixels.append(
            [
                item.frame.getpixel((0, 0))
                for item in frame_observations or ()
                if hasattr(item.frame, "getpixel")
            ]
        )
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="element",
                    element_description="visible object",
                    element_mutation="changed",
                ),
            ),
            change_detected=True,
            metadata={},
        )


class _FakeHistorizer:
    def __init__(self) -> None:
        self.history_inputs: list[AgentContextHistoryInput] = []

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        self.history_inputs.append(history_input)
        world = history_input.current_world_model
        if world is None:
            raise AssertionError("historizer input is missing current world model")
        return AgentContextHistorySummary(
            world_description=world.world_description,
            action_effects=world.action_effects,
            updater_mode="probing",
            probing_evolution="probing evolved",
            policy_evolution="policy evolved",
        )


class _FakeWorldModel:
    def __init__(self) -> None:
        self.inputs: list[AgentWorldModelInput] = []

    def summarize_agent_world_model(
        self,
        history_input: AgentWorldModelInput,
    ) -> AgentContextWorldSummary:
        self.inputs.append(history_input)
        return AgentContextWorldSummary(
            world_description=f"world {len(self.inputs)}",
            action_effects={
                action.name: f"effect {action.name}"
                for action in history_input.allowed_actions
            },
        )


class _FakeAgentGameUpdater:
    def __init__(self) -> None:
        self.update_inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_probing_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.update_inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context=json.dumps(
                {"probing_strategy": f"updated {len(self.update_inputs)}"}
            ),
            next_actions=(update_input.allowed_actions[0],),
            updater_mode="probing",
        )

    def update_agent_policy_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.update_inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context=json.dumps(
                {"policy_strategy": f"updated {len(self.update_inputs)}"}
            ),
            next_actions=(update_input.allowed_actions[0],),
            updater_mode="policy",
        )


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
