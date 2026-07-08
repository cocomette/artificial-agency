"""Tests for known-state simulation in the game loop."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from PIL import Image

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    FrameControlMode,
    Observation,
    ObservationRef,
    RoleContext,
    RuntimeConfig,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models.adapters import ModelRegistry
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.memory import GameMemoryDocument, GameMemoryInput
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    UpdaterTaskRegistry,
)
from face_of_agi.orchestration import Orchestrator
from face_of_agi.orchestration.game_loop.simulation import (
    SIMULATION_CATCHUP_KEY,
    SIMULATED_ROW_KEY,
    _edge_for_action,
    _known_state_transition_edges,
)
from face_of_agi.runtime.loop import RuntimeLoop


def test_known_state_edges_skip_none_successor_boundary(tmp_path) -> None:
    state = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action = ActionSpec("ACTION1")
    source = _prewrite_source(state, turn_id=1, observation=_observation("a", 0))
    state.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=_trace(source.current_observation, action),
        action_history_entry=_history_entry(action, "opened"),
    )
    reset = _prewrite_source(state, turn_id=2, observation=_observation("b", 1))
    state.complete_frame_turn_state(
        state_id=reset.id,
        turn_id=2,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=ActionSpec.none(),
        contexts=ContextDocuments(),
        agent_trace=_trace(reset.current_observation, ActionSpec.none()),
        action_history_entry=_history_entry(ActionSpec.none(), "reset"),
    )
    current = _prewrite_source(state, turn_id=3, observation=_observation("a2", 2))

    edges = _known_state_transition_edges(
        state,
        game_id="game-1",
        run_id="run-1",
        before_state_id=current.id,
    )

    assert edges == ()


def test_runtime_simulates_known_state_and_skips_change_summary(tmp_path) -> None:
    state = StateMemory(SQLiteDatabase(tmp_path / "runtime.sqlite"))
    agent = ScriptedAgent(("ACTION1", "ACTION2", "ACTION1", "ACTION4"))
    change = RecordingChangeSummary()
    memory = RecordingMemory()
    updater = RecordingUpdater()
    environment = LoopingEnvironment()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(
            orchestrator_agent=agent,
            change_summary_model=change,
            game_memory_model=memory,
            updater_tasks=UpdaterTaskRegistry(
                agent_game_updater=updater,
                general_updater=NoopGeneralUpdater(),
            ),
        ),
        contexts=ContextDocuments(),
    )

    result = RuntimeLoop(orchestrator).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=4,
            agent_context_history_window=0,
            debug_keep_all_m_states=True,
            debug_trace="off",
        ),
    )

    states = state.list_states(game_id="game-1")
    simulated = [row for row in states if row.metadata.get(SIMULATED_ROW_KEY)]
    assert result.stop_reason == "action_limit_reached"
    assert [action.name for action in environment.step_actions] == [
        "ACTION1",
        "ACTION2",
        "ACTION1",
        "ACTION4",
    ]
    assert [action.name for action in change.actions] == [
        "ACTION1",
        "ACTION2",
        "ACTION4",
    ]
    assert len(memory.inputs) == 4
    assert len(updater.inputs) == 4
    assert len(states) == 4
    assert len(simulated) == 1
    assert simulated[0].chosen_action["action_id"] == "ACTION1"
    catchup = states[-1].metadata[SIMULATION_CATCHUP_KEY]
    assert catchup["successful"] is True
    assert catchup["simulated_action_count"] == 1
    assert catchup["catchup_action_count"] == 1
    assert catchup["saved_environment_action_count"] == 0


def test_known_state_hash_uses_change_summary_crop(tmp_path) -> None:
    state = StateMemory(SQLiteDatabase(tmp_path / "runtime.sqlite"))
    agent = ScriptedAgent(("ACTION1", "ACTION2", "ACTION1", "ACTION4"))
    change = RecordingChangeSummary(
        crop_box_normalized=(0.25, 0.0, 1.0, 1.0),
    )
    memory = RecordingMemory()
    updater = RecordingUpdater()
    environment = CroppedHashEnvironment()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(
            orchestrator_agent=agent,
            change_summary_model=change,
            game_memory_model=memory,
            updater_tasks=UpdaterTaskRegistry(
                agent_game_updater=updater,
                general_updater=NoopGeneralUpdater(),
            ),
        ),
        contexts=ContextDocuments(),
    )

    result = RuntimeLoop(orchestrator).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=4,
            agent_context_history_window=0,
            debug_keep_all_m_states=True,
            debug_trace="off",
        ),
    )

    states = state.list_states(game_id="game-1")
    simulated = [row for row in states if row.metadata.get(SIMULATED_ROW_KEY)]
    assert result.stop_reason == "action_limit_reached"
    assert len(simulated) == 1
    assert states[0].metadata["current_frame_hash"] == states[2].metadata[
        "current_frame_hash"
    ]
    assert states[0].metadata["current_frame_hash_crop_edges"] == [16, 0, 0, 0]
    catchup = states[-1].metadata[SIMULATION_CATCHUP_KEY]
    assert catchup["successful"] is True
    assert catchup["frame_hash_crop_edges"] == [16, 0, 0, 0]


def test_action6_known_state_edge_matches_current_bbox_with_crop(tmp_path) -> None:
    state = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    historical_action = ActionSpec(
        "ACTION6",
        data={"x": 20, "y": 30},
        target="red pixel",
        target_value=8,
    )
    successor_action = ActionSpec("ACTION1")
    source = _prewrite_source(
        state,
        turn_id=1,
        observation=_observation("a", 0),
        action=historical_action,
    )
    state.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn((historical_action,)),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=historical_action,
        contexts=ContextDocuments(),
        agent_trace=_trace(source.current_observation, historical_action),
        action_history_entry=_history_entry(historical_action, "clicked red"),
    )
    successor = _prewrite_source(
        state,
        turn_id=2,
        observation=_observation("b", 1, color=(255, 0, 0)),
        action=successor_action,
    )
    state.complete_frame_turn_state(
        state_id=successor.id,
        turn_id=2,
        control_mode=FrameControlMode.real_environment_turn((successor_action,)),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=successor_action,
        contexts=ContextDocuments(),
        agent_trace=_trace(successor.current_observation, successor_action),
        action_history_entry=_history_entry(successor_action, "advanced"),
    )
    current = _prewrite_source(state, turn_id=3, observation=_observation("a2", 2))
    edges = _known_state_transition_edges(
        state,
        game_id="game-1",
        run_id="run-1",
        before_state_id=current.id,
    )
    current_action = ActionSpec(
        "ACTION6",
        data={"x": 22, "y": 31},
        target="fresh red target label",
        target_value=8,
        target_bbox=(60, 400, 120, 550),
    )

    edge = _edge_for_action(
        edges,
        frame_hash=current.metadata["current_frame_hash"],
        action=current_action,
        crop_edges=(16, 0, 0, 0),
    )
    mismatched_value = _edge_for_action(
        edges,
        frame_hash=current.metadata["current_frame_hash"],
        action=ActionSpec(
            "ACTION6",
            data={"x": 22, "y": 31},
            target="fresh red target label",
            target_value=9,
            target_bbox=(60, 400, 120, 550),
        ),
        crop_edges=(16, 0, 0, 0),
    )

    assert edge is not None
    assert edge.source_state_id == source.id
    assert mismatched_value is None


class LoopingEnvironment:
    def __init__(self) -> None:
        self.step_actions: list[ActionSpec] = []

    def list_available_games(self) -> Sequence[object]:
        return ()

    def list_local_games(self) -> Sequence[object]:
        return ()

    def resolve_game_id(self, game_index: int) -> str:
        return "game-1"

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        return _observation("obs-0", 0, color=(0, 0, 0))

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del reasoning
        self.step_actions.append(action)
        if action.name == "ACTION1":
            return _observation("obs-1", len(self.step_actions), color=(255, 0, 0))
        if action.name == "ACTION2":
            return _observation("obs-0-repeat", len(self.step_actions), color=(0, 0, 0))
        if action.name == "ACTION4":
            return _observation("obs-3", len(self.step_actions), color=(0, 0, 255))
        return _observation("obs-other", len(self.step_actions), color=(0, 255, 0))

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (
            ActionSpec("ACTION1"),
            ActionSpec("ACTION2"),
            ActionSpec("ACTION4"),
        )

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class CroppedHashEnvironment:
    def __init__(self) -> None:
        self.step_actions: list[ActionSpec] = []

    def list_available_games(self) -> Sequence[object]:
        return ()

    def list_local_games(self) -> Sequence[object]:
        return ()

    def resolve_game_id(self, game_index: int) -> str:
        return "game-1"

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        return _cropped_observation(
            "obs-0",
            0,
            left_color=(255, 0, 0),
            body_color=(0, 0, 0),
        )

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del reasoning
        self.step_actions.append(action)
        if action.name == "ACTION1":
            return _cropped_observation(
                "obs-1",
                len(self.step_actions),
                left_color=(0, 255, 0),
                body_color=(255, 0, 0),
            )
        if action.name == "ACTION2":
            return _cropped_observation(
                "obs-0-repeat",
                len(self.step_actions),
                left_color=(0, 0, 255),
                body_color=(0, 0, 0),
            )
        if action.name == "ACTION4":
            return _cropped_observation(
                "obs-3",
                len(self.step_actions),
                left_color=(255, 255, 0),
                body_color=(0, 0, 255),
            )
        return _cropped_observation(
            "obs-other",
            len(self.step_actions),
            left_color=(255, 0, 255),
            body_color=(0, 255, 0),
        )

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (
            ActionSpec("ACTION1"),
            ActionSpec("ACTION2"),
            ActionSpec("ACTION4"),
        )

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class ScriptedAgent:
    def __init__(self, actions: Sequence[str]) -> None:
        self.actions = list(actions)

    def decide(
        self,
        context: RoleContext,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[object, ...] = (),
        *,
        glossary_actions: Sequence[ActionSpec],
        first_observation_ref: ObservationRef | None = None,
        recent_action_history_available: bool = True,
        action_outcome_evidence: object | None = None,
        game_memory: GameMemoryDocument | None = None,
    ) -> DecisionResult:
        del (
            context,
            tool_runtime,
            recent_action_history,
            glossary_actions,
            recent_action_history_available,
            action_outcome_evidence,
            game_memory,
        )
        name = self.actions.pop(0)
        action = next(item for item in action_space if item.name == name)
        current_ref = ObservationRef(memory="state", id=current_observation.id)
        return DecisionResult(
            final_action=action,
            trace=AgentTrace(
                step=current_observation.step,
                first_observation_ref=first_observation_ref or current_ref,
                current_observation_ref=current_ref,
                final_action=action,
            ),
        )


class RecordingChangeSummary:
    def __init__(
        self,
        crop_box_normalized: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.actions: list[ActionSpec] = []
        self.config = SimpleNamespace(
            input_image_crop_box_normalized=crop_box_normalized,
        )

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
        del (
            previous_observation,
            current_observation,
            glossary_actions,
            frame_observations,
            previous_change_elements,
        )
        self.actions.append(action)
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="frame",
                    element_description="visible frame",
                    element_mutation=f"{action.name} changed the frame",
                ),
            ),
            changed_pixel_count=1,
            change_detected=True,
            metadata={},
            changed_pixel_percent=1.0,
        )


class RecordingMemory:
    def __init__(self) -> None:
        self.inputs: list[GameMemoryInput] = []

    def summarize_game_memory(
        self,
        memory_input: GameMemoryInput,
    ) -> GameMemoryDocument:
        self.inputs.append(memory_input)
        return GameMemoryDocument(
            markdown=f"memory-{len(self.inputs)}",
            metadata={"available": True},
        )


class RecordingUpdater:
    def __init__(self) -> None:
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        self.inputs.append(update_input)
        return update_input.previous_context


class NoopGeneralUpdater:
    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context


def _prewrite_source(
    state: StateMemory,
    *,
    turn_id: int,
    observation: Observation,
    action: ActionSpec | None = None,
):
    action = action or ActionSpec("ACTION1")
    return state.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=turn_id,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=ContextDocuments(),
    )


def _history_entry(action: ActionSpec, summary: str) -> ActionHistoryEntry:
    return ActionHistoryEntry(
        action=action,
        controllable=not action.is_none(),
        changed_pixel_count=1,
        change_summary=summary,
    )


def _trace(observation_payload: object, action: ActionSpec) -> AgentTrace:
    observation = (
        observation_payload
        if isinstance(observation_payload, Observation)
        else Observation(id="obs", step=0, frame=_image((0, 0, 0)))
    )
    observation_ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )


def _observation(
    observation_id: str,
    step: int,
    color: tuple[int, int, int] = (0, 0, 0),
) -> Observation:
    return Observation(id=observation_id, step=step, frame=_image(color))


def _cropped_observation(
    observation_id: str,
    step: int,
    *,
    left_color: tuple[int, int, int],
    body_color: tuple[int, int, int],
) -> Observation:
    return Observation(
        id=observation_id,
        step=step,
        frame=_cropped_hash_image(left_color=left_color, body_color=body_color),
    )


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (4, 4), color=color)


def _cropped_hash_image(
    *,
    left_color: tuple[int, int, int],
    body_color: tuple[int, int, int],
) -> Image.Image:
    image = Image.new("RGB", (64, 64), body_color)
    image.paste(left_color, (0, 0, 16, 64))
    return image
