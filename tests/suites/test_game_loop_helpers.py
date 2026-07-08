"""Tests for small game-loop helper contracts."""

import json

from PIL import Image

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryResetMarker,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    RoleContext,
    RuntimeConfig,
    UpdaterFrameTransitionInput,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.historizer import AgentContextHistorySummary
from face_of_agi.models.level_summary import (
    LevelSolutionSummary,
    LevelSolutionSummaryInput,
)
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
    UpdaterTaskRegistry,
)
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_agent_context_update,
)
from face_of_agi.orchestration.game_loop.actions.steps import (
    bootstrap_agent_updater_decision,
    has_observed_transition,
    summarize_change,
    summarize_change_model,
)
from face_of_agi.orchestration.game_loop.helpers import build_action_history_entry
from face_of_agi.orchestration.game_loop.lifecycle import (
    check_lifecycle,
    reset_after_game_over,
)
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)


def test_action_history_changed_pixel_count_uses_change_summary_crop() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))

    entry = build_action_history_entry(
        frame_context=_frame_context(previous_frame),
        final_action=ActionSpec(action_id="ACTION1"),
        next_observation=Observation(id="current", step=1, frame=current_frame),
        change_summary="Only the cropped-away border changed.",
        change_summary_crop_edges=4,
    )

    assert entry.changed_pixel_count == 0


def test_action_history_changed_pixel_count_is_frame_area_percentage() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))

    entry = build_action_history_entry(
        frame_context=_frame_context(previous_frame),
        final_action=ActionSpec(action_id="ACTION1"),
        next_observation=Observation(id="current", step=1, frame=current_frame),
        change_summary="One pixel changed.",
        change_summary_crop_edges=None,
    )

    assert entry.changed_pixel_count == 0.0244


def test_action_history_changed_pixel_count_uses_cropped_frame_area_percentage() -> None:
    previous_frame = [[0 for _ in range(64)] for _ in range(64)]
    current_frame = [[0 for _ in range(64)] for _ in range(64)]
    current_frame[4][4] = 1

    entry = build_action_history_entry(
        frame_context=_frame_context(previous_frame),
        final_action=ActionSpec(action_id="ACTION1"),
        next_observation=Observation(id="current", step=1, frame=current_frame),
        change_summary="One visible grid cell changed.",
        change_summary_crop_edges=4,
    )

    assert entry.changed_pixel_count == 0.0319


def test_summarize_change_skips_model_when_visible_frames_are_identical() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))
    session = _session(previous_frame, current_frame)
    change_model = _RecordingChangeModel()

    summarize_change(
        session,
        change_model=change_model,
        debug=DebugBus.disabled(),
    )

    assert change_model.calls == 0
    assert session.update_input is not None
    entry = session.update_input.action_history_entry
    assert entry is not None
    assert entry.changed_pixel_count == 0
    assert (
        entry.change_summary
        == "No changes happened for this transition. "
        "The previous and current frames are identical"
    )


def test_game_over_reset_clears_observed_transition_cursor() -> None:
    frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    previous = Observation(id="before-reset", step=3, frame=frame)
    previous_ref = ObservationRef(memory="state", id=previous.id)
    action = ActionSpec(action_id="ACTION1")
    trace = AgentTrace(
        step=previous.step,
        first_observation_ref=previous_ref,
        current_observation_ref=previous_ref,
        final_action=action,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=_ResetEnvironment(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=previous,
        remaining_actions=1,
    )
    session.previous_observation = previous
    session.previous_observation_ref = previous_ref
    session.last_decision = DecisionResult(final_action=action, trace=trace)

    assert has_observed_transition(session)

    reset_after_game_over(session)

    assert session.previous_observation is None
    assert session.previous_observation_ref is None
    assert session.last_decision is None
    assert not has_observed_transition(session)


def test_level_completion_preserves_world_model_and_action_history() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    observation = Observation(id="level-complete", step=1, frame=frame)
    action = ActionSpec(action_id="ACTION1")
    history_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=10,
        change_summary="Completed the level.",
    )
    world_model_context = {
        "world_description": "buttons unlock doors",
        "action_effects": {"ACTION1": "moves"},
    }
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=_CompletedLevelEnvironment(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=1,
    )
    session.queued_updater_actions = (action,)
    session.queued_updater_mode = "probing"
    session.world_model_context = world_model_context
    session.action_history.append(history_entry)

    check_lifecycle(session)

    assert session.queued_updater_actions == ()
    assert session.queued_updater_mode is None
    assert session.world_model_context == world_model_context
    assert session.action_history == [history_entry]


def test_level_completion_summarizes_strategy_history(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="agent L"))
    first = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=contexts,
        agent_trace=_trace(_observation("obs-1")),
        metadata={
            "agent_context_history": {
                "probing_strategy": "probe the switch",
                "policy_strategy": "",
            }
        },
    )
    second = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-2"),
        chosen_action=ActionSpec(action_id="ACTION2"),
        contexts=contexts,
        agent_trace=_trace(_observation("obs-2")),
        metadata={
            "agent_context_history": {
                "probing_strategy": "probe the switch",
                "policy_strategy": "walk to the goal",
            }
        },
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=_CompletedLevelEnvironment(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=_observation("level-complete"),
        remaining_actions=1,
    )
    session.state_record_ids.extend((first.id, second.id))
    summarizer = _RecordingLevelSolutionSummarizer(
        "Probe the switch, then walk to the goal."
    )

    check_lifecycle(
        session,
        state_memory=memory,
        level_solution_summarizer=summarizer,
        debug=DebugBus.disabled(),
    )

    stored = memory.read_latest_level_solution_summary(
        run_id="run-1",
        game_id="game-1",
    )
    assert stored is not None
    assert stored.completed_level == 1
    assert stored.source_state_ids == (first.id, second.id)
    assert stored.solution_method == "Probe the switch, then walk to the goal."
    assert summarizer.inputs[0].strategy_history == (
        '{\n  "probing_strategy": "probe the switch",\n'
        '  "policy_strategy": ""\n}',
        '{\n  "probing_strategy": "probe the switch",\n'
        '  "policy_strategy": "walk to the goal"\n}',
    )


def test_post_reset_bootstrap_ignores_pre_reset_controllable_actions() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    observation = Observation(id="after-reset", step=0, frame=frame)
    observation_ref = ObservationRef(memory="state", id=observation.id)
    action = ActionSpec(action_id="ACTION1")
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=1,
        game_start_turn_id=6,
        game_start_reason="game_over_reset",
    )
    session.action_history.extend(
        (
            ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=0,
                change_summary="pre-reset action",
            ),
            ActionHistoryResetMarker(reason="game_over_reset", restart_count=1),
        )
    )
    session.first_observation_ref = observation_ref
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=6,
        observation=observation,
        observation_ref=observation_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=observation_ref,
    )
    updater = _RecordingAgentProbingUpdater(next_actions=(action,))

    bootstrap_agent_updater_decision(
        session,
        contexts=ContextDocuments(),
        updater_tasks=UpdaterTaskRegistry(agent_probing_updater=updater),
        debug=DebugBus.disabled(),
    )

    assert session.queued_updater_actions == (action,)
    assert session.queued_updater_mode == "probing"
    assert updater.calls == 1
    assert updater.inputs[0].action_history == tuple(session.action_history)


def test_agent_updater_receives_both_previous_game_summaries() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    action = ActionSpec(action_id="ACTION1")
    contexts = ContextDocuments()
    contexts.agent.game = json.dumps(
        {
            "probing_strategy": "probe each action",
            "policy_strategy": "reach the bright area",
        }
    )
    updater = _RecordingAgentProbingUpdater(next_actions=(action,))

    apply_agent_context_update(
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_probing_updater=updater),
        debug=DebugBus.disabled(),
        frame_context=_frame_context(frame),
        current_observation=Observation(id="current", step=1, frame=frame),
        action_history=(),
        allowed_action_source=(action,),
        agent_context_history=AgentContextHistorySummary(
            world_description="mechanics",
            action_effects={},
            updater_mode="probing",
            probing_evolution="probing evolved",
            policy_evolution="policy evolved",
        ),
        turn_id=1,
    )

    assert json.loads(updater.inputs[0].previous_context.game) == {
        "probing_strategy": "probe each action",
        "policy_strategy": "reach the bright area",
    }
    assert json.loads(contexts.agent.game) == {
        "probing_strategy": "fresh post-reset plan",
        "policy_strategy": "reach the bright area",
    }


def test_agent_updater_overrides_probing_when_recent_probe_ratio_is_high() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    action = ActionSpec(action_id="ACTION1")
    contexts = ContextDocuments()
    probing_updater = _RecordingAgentProbingUpdater(next_actions=(action,))
    policy_updater = _RecordingAgentPolicyUpdater(next_actions=(action,))

    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(
            agent_probing_updater=probing_updater,
            agent_policy_updater=policy_updater,
        ),
        debug=DebugBus.disabled(),
        frame_context=_frame_context(frame),
        current_observation=Observation(id="current", step=1, frame=frame),
        action_history=(
            ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=1,
                change_summary="probe",
                action_mode="probing",
            ),
        ),
        allowed_action_source=(action,),
        agent_context_history=AgentContextHistorySummary(
            world_description="mechanics",
            action_effects={},
            updater_mode="probing",
            probing_evolution="probing evolved",
            policy_evolution="policy evolved",
        ),
        turn_id=1,
        probing_actions_window=3,
        policy_actions_window=1,
        action_history_window=10,
    )

    assert result.updater_mode == "policy"
    assert probing_updater.calls == 0
    assert policy_updater.calls == 1
    override = policy_updater.inputs[0].context_history.metadata[
        "updater_mode_override"
    ]
    assert override["from"] == "probing"
    assert override["to"] == "policy"
    assert override["probing_ratio"] == 0.4


def test_agent_updater_keeps_probing_at_probe_ratio_cap_boundary() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    action = ActionSpec(action_id="ACTION1")
    contexts = ContextDocuments()
    probing_updater = _RecordingAgentProbingUpdater(
        next_actions=(action, action, action)
    )
    policy_updater = _RecordingAgentPolicyUpdater(next_actions=(action,))

    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(
            agent_probing_updater=probing_updater,
            agent_policy_updater=policy_updater,
        ),
        debug=DebugBus.disabled(),
        frame_context=_frame_context(frame),
        current_observation=Observation(id="current", step=1, frame=frame),
        action_history=tuple(
            ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=1,
                change_summary="probe",
                action_mode="probing",
            )
            for _ in range(4)
        ),
        allowed_action_source=(action,),
        agent_context_history=AgentContextHistorySummary(
            world_description="mechanics",
            action_effects={},
            updater_mode="probing",
            probing_evolution="probing evolved",
            policy_evolution="policy evolved",
        ),
        turn_id=1,
        probing_actions_window=3,
        policy_actions_window=1,
        action_history_window=20,
    )

    assert result.updater_mode == "probing"
    assert probing_updater.calls == 1
    assert policy_updater.calls == 0
    assert (
        "updater_mode_override"
        not in probing_updater.inputs[0].context_history.metadata
    )


def test_agent_updater_uses_configured_probe_ratio_cap() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    action = ActionSpec(action_id="ACTION1")
    contexts = ContextDocuments()
    probing_updater = _RecordingAgentProbingUpdater(
        next_actions=(action, action, action)
    )
    policy_updater = _RecordingAgentPolicyUpdater(next_actions=(action,))

    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(
            agent_probing_updater=probing_updater,
            agent_policy_updater=policy_updater,
        ),
        debug=DebugBus.disabled(),
        frame_context=_frame_context(frame),
        current_observation=Observation(id="current", step=1, frame=frame),
        action_history=(
            ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=1,
                change_summary="probe",
                action_mode="probing",
            ),
        ),
        allowed_action_source=(action,),
        agent_context_history=AgentContextHistorySummary(
            world_description="mechanics",
            action_effects={},
            updater_mode="probing",
            probing_evolution="probing evolved",
            policy_evolution="policy evolved",
        ),
        turn_id=1,
        probing_actions_window=3,
        policy_actions_window=1,
        action_history_window=10,
        probing_mode_cap_ratio=0.45,
    )

    assert result.updater_mode == "probing"
    assert probing_updater.calls == 1
    assert policy_updater.calls == 0


def test_change_summary_receives_previous_change_elements() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    current_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    session = _session(previous_frame, current_frame)
    previous_element = ChangeSummaryElement(
        element_name="player",
        element_description="red square",
        element_mutation="moved right",
    )
    session.action_history.append(
        ActionHistoryEntry(
            action=ActionSpec(action_id="ACTION1"),
            controllable=True,
            changed_pixel_count=1,
            change_summary="- player: red square; mutations: moved right",
            change_elements=(previous_element,),
        )
    )
    change_model = _RecordingChangeModel()

    summarize_change_model(
        session,
        change_model=change_model,
        debug=DebugBus.disabled(),
    )

    assert change_model.previous_change_elements == [(previous_element,)]


def _frame_context(frame: Image.Image) -> FrameTurnContext:
    observation = Observation(id="previous", step=0, frame=frame)
    ref = ObservationRef(memory="state", id=observation.id)
    action = ActionSpec(action_id="ACTION1")
    return FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )


def _session(
    previous_frame: Image.Image,
    current_frame: Image.Image,
    *,
    source_state_id: int | None = None,
) -> GameLoopSession:
    action = ActionSpec(action_id="ACTION1")
    previous = Observation(id="previous", step=0, frame=previous_frame)
    current = Observation(id="current", step=1, frame=current_frame)
    previous_ref = ObservationRef(memory="state", id=previous.id)
    current_ref = ObservationRef(memory="state", id=current.id)
    control_mode = FrameControlMode.real_environment_turn((action,))
    trace = AgentTrace(
        step=previous.step,
        first_observation_ref=previous_ref,
        current_observation_ref=previous_ref,
        final_action=action,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=previous,
        remaining_actions=1,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=previous,
        observation_ref=previous_ref,
        source_state_id=source_state_id,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=previous_ref,
    )
    session.next = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=current,
        observation_ref=current_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=previous_ref,
    )
    session.decision = DecisionResult(final_action=action, trace=trace)
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=previous_ref,
        actual_next_observation_ref=current_ref,
        decision_trace=trace,
        actual_next_observation=current,
    )
    return session


def _observation(observation_id: str) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=(1, 2, 3)),
    )


def _trace(observation: Observation) -> AgentTrace:
    ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=ref,
        current_observation_ref=ref,
        final_action=ActionSpec(action_id="ACTION1"),
    )


class _RecordingChangeModel:
    def __init__(self) -> None:
        self.config = type(
            "Config",
            (),
            {"input_image_crop_arc_grid_edges": 4},
        )()
        self.calls = 0
        self.previous_change_elements: list[tuple[ChangeSummaryElement, ...]] = []

    def summarize(self, *args: object, **kwargs: object) -> ChangeSummaryResult:
        self.calls += 1
        self.previous_change_elements.append(tuple(kwargs["previous_change_elements"]))
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="element",
                    element_description="visible object",
                    element_mutation="model summary",
                ),
            ),
            change_detected=True,
            metadata={},
        )


class _RecordingAgentProbingUpdater:
    def __init__(self, *, next_actions: tuple[ActionSpec, ...]) -> None:
        self.next_actions = next_actions
        self.calls = 0
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_probing_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.calls += 1
        self.inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context='{"probing_strategy": "fresh post-reset plan"}',
            next_actions=self.next_actions,
            updater_mode="probing",
        )


class _RecordingAgentPolicyUpdater:
    def __init__(self, *, next_actions: tuple[ActionSpec, ...]) -> None:
        self.next_actions = next_actions
        self.calls = 0
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_policy_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.calls += 1
        self.inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context='{"policy_strategy": "fresh policy plan"}',
            next_actions=self.next_actions,
            updater_mode="policy",
        )


class _RecordingLevelSolutionSummarizer:
    def __init__(self, solution_method: str) -> None:
        self.solution_method = solution_method
        self.inputs: list[LevelSolutionSummaryInput] = []

    def summarize_level_solution(
        self,
        summary_input: LevelSolutionSummaryInput,
    ) -> LevelSolutionSummary:
        self.inputs.append(summary_input)
        return LevelSolutionSummary(solution_method=self.solution_method)


class _ResetEnvironment:
    def reset(self) -> Observation:
        frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
        return Observation(id="after-reset", step=0, frame=frame)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(game_id="game-1")


class _CompletedLevelEnvironment:
    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(game_id="game-1", levels_completed=1)
