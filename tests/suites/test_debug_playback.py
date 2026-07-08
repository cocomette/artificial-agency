"""Regression tests for debug-only playback wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from io import StringIO

import pytest

from debug.playback import PlaybackRequest, prepare_playback
from debug.playback.runtime import PlaybackError, load_replay_rows
from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RoleContext,
    RuntimeConfig,
    ToolResult,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models import ModelRegistry, UpdaterTaskRegistry
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    WorldGameContextUpdateInput,
)
from face_of_agi.orchestration import Orchestrator
from face_of_agi.runtime import RuntimeLoop
from face_of_agi.runtime import shell


class PlaybackTestEnvironment:
    """Small deterministic environment for replay and handoff tests."""

    def __init__(self) -> None:
        self.step_actions: list[ActionSpec] = []
        self.reset_calls = 0

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        self.reset_calls += 1
        return Observation(id="obs-0", step=0, frame={"frame": 0})

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del reasoning
        self.step_actions.append(action)
        step = len(self.step_actions)
        return Observation(id=f"obs-{step}", step=step, frame={"frame": step})

    def get_action_space(self) -> Sequence[ActionSpec]:
        return [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")]

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class CountingAgent:
    """Live Agent X fake used after playback handoff."""

    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self,
        context: RoleContext,
        history_anchor_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[object, ...] = (),
    ) -> DecisionResult:
        del context, tool_runtime, recent_action_history
        self.calls += 1
        final_action = action_space[0]
        first_ref = ObservationRef(memory="state", id=history_anchor_observation.id)
        current_ref = ObservationRef(memory="state", id=current_observation.id)
        trace = AgentTrace(
            step=current_observation.step,
            first_observation_ref=first_ref,
            current_observation_ref=current_ref,
            final_action=final_action,
            reasoning_summary="live agent",
        )
        return DecisionResult(final_action=final_action, trace=trace)


class CountingWorldModel:
    """Live world model fake used after playback handoff."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> ToolResult:
        del context
        self.calls += 1
        return ToolResult(
            id=f"live-world-{self.calls}",
            tool="world",
            predicted_description=[{"description": "live world", "bbox_2d": [0, 0, 1, 1]}],
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action,
        )


class CountingGoalModel:
    """Live goal model fake used after playback handoff."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        del context
        self.calls += 1
        return ToolResult(
            id=f"live-goal-{self.calls}",
            tool="goal",
            predicted_description=[{"description": "live goal", "bbox_2d": [0, 0, 1, 1]}],
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
        )


class CountingUpdater:
    """Live updater fake used after playback handoff."""

    def __init__(self) -> None:
        self.world_calls = 0
        self.goal_calls = 0
        self.agent_calls = 0
        self.general_calls = 0

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        del update_input
        self.world_calls += 1
        return RoleContext(game=f"live-world-context-{self.world_calls}")

    def update_goal_game_context(
        self,
        update_input: GoalGameContextUpdateInput,
    ) -> RoleContext:
        del update_input
        self.goal_calls += 1
        return RoleContext(game=f"live-goal-context-{self.goal_calls}")

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        del update_input
        self.agent_calls += 1
        return RoleContext(game=f"live-agent-context-{self.agent_calls}")

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        self.general_calls += 1
        return update_input.previous_context


def test_playback_requires_contiguous_prior_rows(tmp_path) -> None:
    state = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    _write_source_state(state, turn_id=1, action_id="ACTION1")
    _write_source_state(state, turn_id=3, action_id="ACTION1")

    with pytest.raises(PlaybackError, match="missing required prior turn"):
        load_replay_rows(
            state,
            PlaybackRequest(
                source_run_id="source-run",
                game_id="game-1",
                turn_id=3,
            ),
        )


def test_playback_turn_one_requires_no_prior_rows(tmp_path) -> None:
    state = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    _write_source_state(state, turn_id=1, action_id="ACTION1")

    replay_rows = load_replay_rows(
        state,
        PlaybackRequest(
            source_run_id="source-run",
            game_id="game-1",
            turn_id=1,
        ),
    )

    assert replay_rows == ()


def test_playback_replays_recorded_actions_then_hands_off_to_live_models(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    _write_source_state(state, turn_id=1, action_id="ACTION2")
    _write_source_state(state, turn_id=2, action_id="ACTION2")

    agent = CountingAgent()
    world = CountingWorldModel()
    goal = CountingGoalModel()
    updater = CountingUpdater()
    live_models = ModelRegistry(
        orchestrator_agent=agent,
        world_prediction_model=world,
        goal_prediction_model=goal,
        updater_tasks=UpdaterTaskRegistry(
            world_game_updater=updater,
            goal_game_updater=updater,
            agent_game_updater=updater,
            general_updater=updater,
        ),
    )
    playback = prepare_playback(
        state_memory=state,
        request=PlaybackRequest(
            source_run_id="source-run",
            game_id="game-1",
            turn_id=2,
        ),
        live_models=live_models,
    )
    orchestrator = Orchestrator(
        state_memory=state,
        models=playback.models,
        contexts=playback.contexts,
    )
    environment = PlaybackTestEnvironment()

    RuntimeLoop(orchestrator, trace_output=StringIO()).run(
        config=RuntimeConfig(run_id="new-run"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=2,
            use_learned_contexts=False,
            debug_keep_all_m_states=True,
        ),
    )

    assert [action.name for action in environment.step_actions] == [
        "ACTION2",
        "ACTION1",
    ]
    assert agent.calls == 1
    assert world.calls == 1
    assert goal.calls == 0
    assert updater.world_calls == 1
    assert updater.goal_calls == 0
    assert updater.agent_calls == 1

    new_rows = [
        row for row in state.list_states(game_id="game-1") if row.run_id == "new-run"
    ]
    assert [row.chosen_action["action_id"] for row in new_rows] == [
        "ACTION2",
        "ACTION1",
    ]
    assert new_rows[0].world_context.game == "source-world-1"
    assert new_rows[0].goal_context.game == ""
    assert new_rows[0].agent_context.game == "source-agent-1"
    assert new_rows[1].world_context.game == "live-world-context-1"


def test_runtime_shell_rejects_partial_playback_flags() -> None:
    parser = shell._build_parser()
    args = parser.parse_args(["--playback-run-id", "run-1"])

    with pytest.raises(SystemExit):
        shell._playback_request_from_args(parser, args)


def test_runtime_shell_accepts_complete_playback_flags() -> None:
    parser = shell._build_parser()
    args = parser.parse_args(
        [
            "--playback-run-id",
            "run-1",
            "--playback-game-id",
            "game-1",
            "--playback-turn-id",
            "2",
        ]
    )

    request = shell._playback_request_from_args(parser, args)

    assert request == PlaybackRequest(
        source_run_id="run-1",
        game_id="game-1",
        turn_id=2,
    )


def _write_source_state(
    state: StateMemory,
    *,
    turn_id: int,
    action_id: str,
) -> None:
    observation = Observation(
        id=f"source-obs-{turn_id}",
        step=turn_id - 1,
        frame={"frame": turn_id},
    )
    action = ActionSpec(action_id=action_id)
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=observation.step,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
        reasoning_summary="source trace",
    )
    state.write_state(
        run_id="source-run",
        game_id="game-1",
        step=observation.step,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(
            world=RoleContext(game=f"source-world-{turn_id}"),
            goal=RoleContext(game=f"source-goal-{turn_id}"),
            agent=RoleContext(game=f"source-agent-{turn_id}"),
        ),
        agent_trace=trace,
        post_decision_predictions=PostDecisionPredictions(
            world_prediction=ToolResult(
                id=f"source-world-prediction-{turn_id}",
                tool="world",
                predicted_description=[
                    {"description": "source world", "bbox_2d": [0, 0, 1, 1]}
                ],
                source_observation_ref=observation_ref,
                action=action,
            ),
            goal_prediction=ToolResult(
                id=f"source-goal-prediction-{turn_id}",
                tool="goal",
                predicted_description=[
                    {"description": "source goal", "bbox_2d": [0, 0, 1, 1]}
                ],
                source_observation_ref=observation_ref,
            ),
        ),
        metadata={
            "turn_id": turn_id,
            "control_mode": {
                "controllable": True,
                "allowed_actions": [{"action_id": "ACTION1"}, {"action_id": "ACTION2"}],
                "reason": "real_environment_turn",
            },
        },
    )
