"""Tests for multi-action updater chain execution."""

from __future__ import annotations

from collections.abc import Sequence
import json

from arcengine import GameState

from face_of_agi.contracts import (
    ActionSpec,
    ChangeSummaryElement,
    ContextDocuments,
    EnvironmentInfo,
    Observation,
    RoleContext,
    RuntimeConfig,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.historizer import (
    AgentContextHistoryInput,
    AgentContextHistorySummary,
)
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
    GeneralKnowledgeUpdateInput,
    UpdaterTaskRegistry,
)
from face_of_agi.models.world import AgentContextWorldSummary, AgentWorldModelInput
from face_of_agi.orchestration.game_loop.state_machine import GameLoopStateMachine

Frame = tuple[tuple[int, ...], ...]


def test_action_chain_skips_updaters_until_final_transition() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1", "ACTION2", "ACTION3"),
            _actions("ACTION1", "ACTION2", "ACTION3"),
            _actions("ACTION1", "ACTION2", "ACTION3"),
            _actions("ACTION1"),
        )
    )
    change = _ChangeSummary()
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1", "ACTION2", "ACTION3"),
            _actions("ACTION1", "ACTION1", "ACTION1"),
        )
    )

    result = _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=3,
        max_actions_per_level=4,
    )

    assert result.stop_reason == "action_limit_reached"
    assert [action.name for action in environment.actions] == [
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION1",
    ]
    assert [action.name for action in change.actions] == [
        "ACTION1",
        "ACTION2",
        "ACTION3",
    ]
    assert [item.actions_window for item in updater.inputs] == [3, 3]
    assert [item.current_observation.id for item in updater.inputs] == [
        "reset-1",
        "step-3",
    ]
    assert [item.current_observation.id for item in world.inputs] == ["step-3"]
    assert [item.current_observation.id for item in historizer.inputs] == ["step-3"]


def test_level_completion_clears_chain_and_refreshes_context() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION1", "ACTION2"),
        ),
        completed_levels_by_step={1: 1},
        states_by_step={2: GameState.WIN},
    )
    change = _ChangeSummary()
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION1", "ACTION1"),
        )
    )

    _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=2,
        max_actions_per_level=2,
    )

    assert [action.name for action in environment.actions] == ["ACTION1", "ACTION1"]
    assert [item.current_observation.id for item in updater.inputs] == [
        "reset-1",
        "step-1",
    ]
    assert [item.current_observation.id for item in world.inputs] == ["step-1"]
    assert [item.current_observation.id for item in historizer.inputs] == ["step-1"]


def test_game_over_transition_refreshes_context_before_reset() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION1",),
        ),
        states_by_step={1: GameState.GAME_OVER},
        state_after_second_reset=GameState.WIN,
    )
    change = _ChangeSummary()
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION1", "ACTION1"),
        )
    )

    result = _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=2,
        max_actions_per_level=3,
    )

    assert result.stop_reason == "game_end"
    assert environment.reset_count == 2
    assert [action.name for action in environment.actions] == ["ACTION1"]
    assert [item.current_observation.id for item in updater.inputs] == [
        "reset-1",
        "step-1",
    ]
    assert [item.current_observation.id for item in world.inputs] == ["step-1"]
    assert [item.current_observation.id for item in historizer.inputs] == ["step-1"]


def test_invalid_queued_action_refreshes_instead_of_submitting() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION1"),
        )
    )
    change = _ChangeSummary()
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION2", "ACTION2"),
            _actions("ACTION1", "ACTION1"),
        )
    )

    _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=2,
        max_actions_per_level=2,
    )

    assert [action.name for action in environment.actions] == ["ACTION2", "ACTION1"]
    assert [item.current_observation.id for item in updater.inputs] == [
        "reset-1",
        "step-1",
    ]
    assert [item.current_observation.id for item in world.inputs] == ["step-1"]


def test_net_noop_transition_clears_queued_actions() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1", "ACTION2", "ACTION3"),
            _actions("ACTION1", "ACTION2", "ACTION3"),
        ),
        frames_by_step={1: (_frame(0),)},
    )
    change = _ChangeSummary()
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION3", "ACTION3"),
        )
    )

    _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=2,
        max_actions_per_level=2,
    )

    assert [action.name for action in environment.actions] == ["ACTION1", "ACTION3"]
    assert change.actions == []
    assert [item.current_observation.id for item in updater.inputs] == [
        "reset-1",
        "step-1",
    ]
    assert [item.current_observation.id for item in world.inputs] == ["step-1"]
    assert [item.current_observation.id for item in historizer.inputs] == ["step-1"]


def test_animation_with_net_noop_clears_queued_actions_after_change_summary() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1", "ACTION2", "ACTION3"),
            _actions("ACTION1", "ACTION2", "ACTION3"),
        ),
        frames_by_step={1: (_frame(0), _frame(1), _frame(0))},
    )
    change = _ChangeSummary()
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1", "ACTION2"),
            _actions("ACTION3", "ACTION3"),
        )
    )

    _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=2,
        max_actions_per_level=2,
    )

    assert [action.name for action in environment.actions] == ["ACTION1", "ACTION3"]
    assert [action.name for action in change.actions] == ["ACTION1"]
    assert [item.current_observation.id for item in updater.inputs] == [
        "reset-1",
        "step-1-frame-2",
    ]
    assert [item.current_observation.id for item in world.inputs] == [
        "step-1-frame-2"
    ]
    assert [item.current_observation.id for item in historizer.inputs] == [
        "step-1-frame-2"
    ]


def test_change_detected_false_replaces_uncertain_direct_change_summary() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1"),
            _actions("ACTION1"),
        ),
        states_by_step={2: GameState.WIN},
    )
    change = _ChangeSummary(change_detected=False)
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1",),
            _actions("ACTION1",),
        )
    )

    _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=1,
        max_actions_per_level=2,
    )

    assert world.inputs[0].action_history[-1].change_summary == (
        "This action produced changes but it is uncertain what changed exactly."
    )


def test_change_detected_false_replaces_uncertain_animation_change_summary() -> None:
    environment = _ChainEnvironment(
        action_spaces=(
            _actions("ACTION1"),
            _actions("ACTION1"),
        ),
        frames_by_step={1: (_frame(0), _frame(1), _frame(0))},
        states_by_step={2: GameState.WIN},
    )
    change = _ChangeSummary(change_detected=False)
    world = _WorldModel()
    historizer = _Historizer()
    updater = _SequenceUpdater(
        (
            _actions("ACTION1",),
            _actions("ACTION1",),
        )
    )

    _run(
        environment=environment,
        change=change,
        world=world,
        historizer=historizer,
        updater=updater,
        probing_actions_window=1,
        max_actions_per_level=2,
    )

    assert world.inputs[0].action_history[-1].change_summary == (
        "animation produced changes but it is uncertain what changed exactly."
    )


def _run(
    *,
    environment: "_ChainEnvironment",
    change: "_ChangeSummary",
    world: "_WorldModel",
    historizer: "_Historizer",
    updater: "_SequenceUpdater",
    probing_actions_window: int,
    max_actions_per_level: int,
):
    return GameLoopStateMachine(
        state_memory=None,
        contexts=ContextDocuments(),
        change_summary_model=change,
        world_model=world,
        agent_context_historizer=historizer,
        updater_tasks=UpdaterTaskRegistry(
            agent_probing_updater=updater,
            agent_policy_updater=updater,
            general_updater=_GeneralUpdater(),
        ),
        debug=DebugBus.disabled(),
    ).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=max_actions_per_level,
            probing_actions_window=probing_actions_window,
            policy_actions_window=1,
            probing_mode_cap_ratio=1.0,
        ),
    )


def _actions(*names: str) -> tuple[ActionSpec, ...]:
    return tuple(ActionSpec(action_id=name) for name in names)


def _frame(value: int) -> Frame:
    return ((value,),)


class _ChainEnvironment:
    def __init__(
        self,
        *,
        action_spaces: Sequence[tuple[ActionSpec, ...]],
        completed_levels_by_step: dict[int, int] | None = None,
        states_by_step: dict[int, GameState] | None = None,
        state_after_second_reset: GameState | None = None,
        frames_by_step: dict[int, tuple[Frame, ...]] | None = None,
    ) -> None:
        self.action_spaces = tuple(action_spaces)
        self.completed_levels_by_step = completed_levels_by_step or {}
        self.states_by_step = states_by_step or {}
        self.state_after_second_reset = state_after_second_reset
        self.frames_by_step = frames_by_step or {}
        self.actions: list[ActionSpec] = []
        self.reset_count = 0
        self.step_count = 0
        self.state: GameState | None = None
        self.completed_levels = 0

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
        self.reset_count += 1
        if self.reset_count > 1:
            self.state = self.state_after_second_reset
        else:
            self.state = None
        return Observation(
            id=f"reset-{self.reset_count}",
            step=0,
            frame=_frame(0),
            frames=(_frame(0),),
        )

    def step(self, action: ActionSpec, reasoning=None) -> Observation:
        del reasoning
        self.actions.append(action)
        self.step_count += 1
        self.state = self.states_by_step.get(self.step_count)
        self.completed_levels = max(
            self.completed_levels,
            self.completed_levels_by_step.get(self.step_count, 0),
        )
        frames = self.frames_by_step.get(self.step_count, (_frame(self.step_count),))
        return Observation(
            id=f"step-{self.step_count}",
            step=self.step_count,
            frame=frames[0],
            frames=frames,
        )

    def get_action_space(self) -> Sequence[ActionSpec]:
        index = min(self.step_count, len(self.action_spaces) - 1)
        return self.action_spaces[index]

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            state=self.state,
            available_actions=tuple(self.get_action_space()),
            levels_completed=self.completed_levels,
        )


class _ChangeSummary:
    config = None

    def __init__(self, *, change_detected: bool = True) -> None:
        self.actions: list[ActionSpec] = []
        self.change_detected = change_detected

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
        del frame_observations, previous_change_elements
        self.actions.append(action)
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="element",
                    element_description="visible object",
                    element_mutation=f"changed by {action.name}",
                ),
            ),
            change_detected=self.change_detected,
            metadata={},
        )


class _WorldModel:
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
                action.name: "effect" for action in history_input.allowed_actions
            },
        )


class _Historizer:
    def __init__(self) -> None:
        self.inputs: list[AgentContextHistoryInput] = []

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        self.inputs.append(history_input)
        world = history_input.current_world_model
        return AgentContextHistorySummary(
            world_description=world.world_description if world is not None else "",
            action_effects=world.action_effects if world is not None else {},
            updater_mode="probing",
            probing_evolution="probing evolved",
            policy_evolution="policy evolved",
        )


class _SequenceUpdater:
    def __init__(self, action_batches: Sequence[tuple[ActionSpec, ...]]) -> None:
        self.action_batches = list(action_batches)
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_probing_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context=json.dumps(
                {"probing_strategy": f"updated {len(self.inputs)}"}
            ),
            next_actions=self.action_batches.pop(0),
            updater_mode="probing",
        )

    def update_agent_policy_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context=json.dumps({"policy_strategy": f"updated {len(self.inputs)}"}),
            next_actions=self.action_batches.pop(0),
            updater_mode="policy",
        )


class _GeneralUpdater:
    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context
