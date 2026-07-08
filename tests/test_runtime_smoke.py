"""Smoke test for the reset-only runtime boundary."""

from collections.abc import Sequence
from io import StringIO

import pytest

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RoleContext,
    RuntimeConfig,
    ToolCall,
    ToolResult,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.environment.config import ModelRoleConfig
from face_of_agi.environment.config import load_environment_config
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import ModelRegistry
from face_of_agi.models import (
    AgentContextUpdateInput,
    ToolContextUpdateInput,
)
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.random import (
    RandomOrchestratorAgentAdapter,
)
from face_of_agi.models.tools.goal.providers.openai import OpenAIGoalToolAdapter
from face_of_agi.models.tools.world.providers.openai import OpenAIWorldToolAdapter
from face_of_agi.orchestration import Orchestrator
from face_of_agi.runtime import RuntimeLoop
from face_of_agi.runtime import shell


class FakeEnvironment:
    """Fake adapter that fails if the runtime applies a real action."""

    def __init__(self) -> None:
        self.step_called = False

    def reset(self) -> Observation:
        return Observation(
            id="obs-0",
            step=0,
            frame={"pixels": "fake"},
            metadata={"available_actions": ["ACTION1"]},
        )

    def step(self, action: ActionSpec, reasoning: str | None = None) -> Observation:
        self.step_called = True
        raise AssertionError("reset-only smoke flow must not call step")

    def get_action_space(self) -> Sequence[ActionSpec]:
        return [ActionSpec(action_id="ACTION1")]

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(game_id="game-1")


class FakeAgent:
    """Fake agent that returns one final action and trace."""

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
    ) -> DecisionResult:
        del tool_runtime
        final_action = action_space[0]
        observation_ref = ObservationRef(memory="state", id=current_observation.id)
        trace = AgentTrace(
            step=current_observation.step,
            first_observation_ref=observation_ref,
            current_observation_ref=observation_ref,
            final_action=final_action,
            reasoning_summary="fake trace",
        )
        return DecisionResult(final_action=final_action, trace=trace)


class FrameBundleEnvironment:
    """Fake ARC adapter that returns multiple reset frames before control."""

    def __init__(self) -> None:
        self.step_actions: list[ActionSpec] = []
        self.reset_calls = 0

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        self.reset_calls += 1
        return Observation(
            id="obs-reset",
            step=0,
            frames=({"frame": 0}, {"frame": 1}, {"frame": 2}),
        )

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del reasoning
        if action.is_none():
            raise AssertionError("synthetic NONE must never be sent to ARC")
        self.step_actions.append(action)
        return Observation(
            id="obs-after-action",
            step=1,
            frame={"frame": 3},
        )

    def get_action_space(self) -> Sequence[ActionSpec]:
        return [ActionSpec(action_id="ACTION1")]

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class SingleFrameEnvironment(FrameBundleEnvironment):
    """Fake ARC adapter that exposes one controllable frame per observation."""

    def reset(self) -> Observation:
        self.reset_calls += 1
        return Observation(id="obs-reset", step=0, frame={"frame": 0})


class CapturingAgent(FakeAgent):
    """Fake agent that records the context it receives."""

    def __init__(self) -> None:
        self.contexts: list[RoleContext] = []
        self.tool_runtimes: list[object | None] = []

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
    ) -> DecisionResult:
        self.contexts.append(context)
        self.tool_runtimes.append(tool_runtime)
        return super().decide(
            context=context,
            first_observation=first_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
        )


class FailingStepEnvironment(FrameBundleEnvironment):
    """Fake adapter that fails while applying the real environment step."""

    def reset(self) -> Observation:
        return Observation(id="obs-reset", step=0, frame={"frame": 0})

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del action, reasoning
        raise RuntimeError("boom")


class FakeWorldTool:
    """Fake world tool that records the resolved source observation."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.contexts: list[RoleContext] = []
        self.actions: list[ActionSpec] = []

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> ToolResult:
        self.contexts.append(context)
        self.actions.append(action)
        self.observations.append(observation)
        return ToolResult(
            id="world-out",
            tool="world",
            predicted_observation={"predicted_from": observation.id},
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action,
        )


class FakeGoalTool:
    """Fake goal tool that records the resolved source observation."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.observations: list[Observation] = []
        self.contexts: list[RoleContext] = []

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        if self.fail:
            raise RuntimeError("tool failed")
        self.contexts.append(context)
        self.observations.append(observation)
        return ToolResult(
            id="goal-out",
            tool="goal",
            predicted_observation={"goal_from": observation.id},
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
        )


class ToolCallingAgent(FakeAgent):
    """Fake X shell that exercises the orchestration-owned tool runtime."""

    def __init__(self) -> None:
        self.experiment_refs: list[ObservationRef] = []

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
    ) -> DecisionResult:
        if tool_runtime is None:
            raise AssertionError("tool runtime should be available during game loop")

        action = action_space[0]
        call = ToolCall(
            tool="world",
            observation_ref=tool_runtime.current_observation_ref,
            action=action,
        )
        invocation = tool_runtime.invoke(call)
        self.experiment_refs.append(invocation.observation_ref)

        decision = super().decide(
            context=context,
            first_observation=first_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
        )
        decision.trace.tool_calls.append(call)
        decision.trace.tool_results.append(invocation.tool_result)
        return decision


class DualToolCallingAgent(FakeAgent):
    """Fake X shell that calls both configured tools on controllable frames."""

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
    ) -> DecisionResult:
        decision = super().decide(
            context=context,
            first_observation=first_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
        )
        if tool_runtime is None or not tool_runtime.available_tools():
            return decision

        world_call = ToolCall(
            tool="world",
            observation_ref=tool_runtime.current_observation_ref,
            action=decision.final_action,
        )
        goal_call = ToolCall(
            tool="goal",
            observation_ref=tool_runtime.current_observation_ref,
        )
        for call in (world_call, goal_call):
            invocation = tool_runtime.invoke(call)
            decision.trace.tool_calls.append(call)
            decision.trace.tool_results.append(invocation.tool_result)

        return decision


class MutatingFakeUpdater:
    """Fake updater that proves orchestration injects returned contexts."""

    def __init__(self) -> None:
        self.world_calls = 0
        self.goal_calls = 0
        self.agent_calls = 0
        self.world_inputs: list[ToolContextUpdateInput] = []
        self.goal_inputs: list[ToolContextUpdateInput] = []
        self.agent_inputs: list[AgentContextUpdateInput] = []

    def update_tool_context(
        self,
        update_input: ToolContextUpdateInput,
    ) -> RoleContext:
        if update_input.role == "world":
            self.world_calls += 1
            self.world_inputs.append(update_input)
            return RoleContext(
                general=update_input.previous_context.general,
                game=f"world-{self.world_calls}",
            )

        self.goal_calls += 1
        self.goal_inputs.append(update_input)
        return RoleContext(
            general=update_input.previous_context.general,
            game=f"goal-{self.goal_calls}",
        )

    def update_agent_context(
        self,
        update_input: AgentContextUpdateInput,
    ) -> RoleContext:
        self.agent_calls += 1
        self.agent_inputs.append(update_input)
        return RoleContext(
            general=update_input.previous_context.general,
            game=f"agent-{self.agent_calls}",
        )


def test_runtime_reset_only_flow_persists_initial_trace(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    registry = ModelRegistry(orchestrator_agent=FakeAgent())
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=registry,
    )
    runtime = RuntimeLoop(orchestrator)
    environment = FakeEnvironment()

    results = runtime.run(
        config=RuntimeConfig(run_id="run-1", game_ids=("game-1",)),
        environments={"game-1": environment},
    )

    states = state.list_states(game_id="game-1")
    assert len(results) == 1
    assert results[0].game_id == "game-1"
    assert results[0].decision.final_action.action_id == "ACTION1"
    assert len(states) == 1
    assert states[0].current_observation["id"] == "obs-0"
    assert states[0].agent_trace["reasoning_summary"] == "fake trace"
    assert state.list_records(run_id="run-1", game_id="game-1") == []
    assert experimental.list_records(run_id="run-1", game_id="game-1") == []
    assert environment.step_called is False


def test_environment_shell_unrolls_frames_and_steps_only_on_final_frame(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)
    environment = FrameBundleEnvironment()

    result = runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert isinstance(result, object)
    assert result.stop_reason == "action_limit_reached"
    assert result.step_count == 1
    assert len(environment.step_actions) == 1
    assert not environment.step_actions[0].is_none()
    assert output.getvalue().count("X returned NONE") == 2
    assert "X selected ACTION1" in output.getvalue()
    assert len(state.list_states(game_id="game-1")) == 1
    assert state.list_records(run_id="run-1", game_id="game-1") == []
    assert experimental.list_records(run_id="run-1", game_id="game-1") == []


def test_orchestration_writes_m_state_for_each_frame_turn_without_cleanup(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    orchestrator = Orchestrator(state_memory=state)

    result = orchestrator.run_environment_shell(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
        trace_output=StringIO(),
    )

    states = state.list_states(game_id="game-1")
    assert result.stop_reason == "action_limit_reached"
    assert len(states) == 3
    assert states[-1].agent_trace["tool_calls"] == []
    assert states[-1].agent_trace["tool_results"] == []


def test_environment_shell_hydrates_contexts_from_latest_m_state(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    observation = Observation(id="previous", step=0, frame={"frame": "old"})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=0,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )
    state.write_state(
        run_id="old-run",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="learned context")),
        agent_trace=trace,
    )
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(orchestrator_agent=agent),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-2"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert agent.contexts
    assert {context.game for context in agent.contexts} == {"learned context"}


def test_environment_shell_uses_default_contexts_when_m_state_is_empty(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(orchestrator_agent=agent),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert agent.contexts
    assert {context.game for context in agent.contexts} == {""}


def test_game_loop_injects_updater_agent_context_on_next_x_call(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(
            orchestrator_agent=agent,
            updater=updater,
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert [context.game for context in agent.contexts] == [
        "",
        "agent-1",
        "agent-2",
    ]
    assert updater.agent_calls == 3


def test_game_loop_injects_updater_tool_contexts_and_persists_them(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    world_tool = FakeWorldTool()
    goal_tool = FakeGoalTool()
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=ModelRegistry(
            world_tool=world_tool,
            goal_tool=goal_tool,
            orchestrator_agent=DualToolCallingAgent(),
            updater=updater,
        ),
    )

    result = orchestrator.run_environment_shell(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
        trace_output=StringIO(),
    )

    states = state.list_states(game_id="game-1")
    assert result.stop_reason == "action_limit_reached"
    assert [context.game for context in world_tool.contexts] == ["world-2"]
    assert [context.game for context in goal_tool.contexts] == ["goal-2"]
    assert states[-1].world_context.game == "world-3"
    assert states[-1].goal_context.game == "goal-3"
    assert states[-1].agent_context.game == "agent-3"
    assert updater.world_calls == 3
    assert updater.goal_calls == 3
    assert updater.agent_calls == 3


def test_mock_post_decision_predictions_persist_to_m_and_reach_updater(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(
            orchestrator_agent=FakeAgent(),
            updater=updater,
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    states = state.list_states(game_id="game-1")
    assert states[0].world_prediction is not None
    assert states[0].goal_prediction is not None
    assert states[0].world_prediction["predicted_observation"] == {"frame": 0}
    assert states[0].goal_prediction["predicted_observation"] == {"frame": 0}
    assert states[0].world_prediction["metadata"][
        "prompt_model_calls_enabled"
    ] is False
    assert states[0].agent_trace["tool_calls"] == []
    assert states[0].agent_trace["tool_results"] == []
    assert (
        updater.world_inputs[0].post_decision_predictions.world_prediction
        is not None
    )
    assert (
        updater.goal_inputs[0].post_decision_predictions.goal_prediction
        is not None
    )
    assert isinstance(
        updater.agent_inputs[0].post_decision_predictions,
        PostDecisionPredictions,
    )


def test_real_post_decision_predictions_call_tools_and_persist_to_m(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    world_tool = FakeWorldTool()
    goal_tool = FakeGoalTool()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(
            world_tool=world_tool,
            goal_tool=goal_tool,
            orchestrator_agent=FakeAgent(),
        ),
        prompt_model_calls_enabled=True,
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    states = state.list_states(game_id="game-1")
    assert [observation.id for observation in world_tool.observations] == ["obs-reset"]
    assert [observation.id for observation in goal_tool.observations] == ["obs-reset"]
    assert [action.name for action in world_tool.actions] == ["ACTION1"]
    assert states[0].world_prediction["predicted_observation"] == {
        "predicted_from": "obs-reset"
    }
    assert states[0].goal_prediction["predicted_observation"] == {
        "goal_from": "obs-reset"
    }
    assert states[0].world_prediction["metadata"][
        "prompt_model_calls_enabled"
    ] is True
    assert states[0].agent_trace["tool_results"] == []


def test_real_post_decision_predictions_require_configured_tools(tmp_path) -> None:
    orchestrator = Orchestrator(
        state_memory=StateMemory(SQLiteDatabase(tmp_path / "runtime.sqlite")),
        models=ModelRegistry(orchestrator_agent=FakeAgent()),
        prompt_model_calls_enabled=True,
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    with pytest.raises(RuntimeError, match="world model is not registered"):
        runtime.run(
            config=RuntimeConfig(run_id="run-1"),
            environment=SingleFrameEnvironment(),
            environment_config=EnvironmentConfig(
                game_index=0,
                game_id="game-1",
                max_actions_per_level=1,
            ),
        )


def test_non_controllable_frames_skip_post_decision_predictions(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(orchestrator_agent=FakeAgent()),
    )

    orchestrator.run_environment_shell(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
        trace_output=StringIO(),
    )

    states = state.list_states(game_id="game-1")
    assert states[0].world_prediction is None
    assert states[0].goal_prediction is None
    assert states[1].world_prediction is None
    assert states[1].goal_prediction is None
    assert states[2].world_prediction is not None
    assert states[2].goal_prediction is not None


def test_orchestration_tool_invocation_persists_world_output_to_e(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    source_observation = Observation(id="obs-0", step=0, frame={"frame": "real"})
    source_ref = ObservationRef(memory="state", id=source_observation.id)
    action = ActionSpec(action_id="ACTION1")
    trace = AgentTrace(
        step=0,
        first_observation_ref=source_ref,
        current_observation_ref=source_ref,
        final_action=action,
    )
    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=source_observation,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=trace,
    )
    world_tool = FakeWorldTool()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=ModelRegistry(world_tool=world_tool),
        experimental_memory_turn_buffer=2,
    )
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=source_ref,
        current_observation_ref=source_ref,
        current_observation=source_observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )

    result = orchestrator.invoke_tool_for_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        frame_context=frame_context,
        call=ToolCall(
            tool="world",
            observation_ref=source_ref,
            action=action,
        ),
    )

    experiments = experimental.list_experiments(run_id="run-1", game_id="game-1")
    assert result.observation_ref.memory == "experimental"
    assert result.observation_ref.id == str(experiments[0].id)
    assert world_tool.observations[0].frame == {"frame": "real"}
    assert experiments[0].source_observation_ref == source_ref
    assert experiments[0].output_observation["frame"] == {"predicted_from": "obs-0"}


def test_game_loop_passes_tool_runtime_to_x_and_persists_e_output(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    world_tool = FakeWorldTool()
    agent = ToolCallingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=ModelRegistry(
            world_tool=world_tool,
            orchestrator_agent=agent,
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    result = runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    experiments = experimental.list_experiments(run_id="run-1", game_id="game-1")
    states = state.list_states(game_id="game-1")
    assert result.stop_reason == "action_limit_reached"
    assert len(experiments) == 1
    assert agent.experiment_refs == [
        ObservationRef(memory="experimental", id=str(experiments[0].id))
    ]
    assert world_tool.observations[0].id == "obs-reset"
    assert experiments[0].source_observation_ref == ObservationRef(
        memory="state",
        id="obs-reset",
    )
    assert experiments[0].metadata["requested_by"] == "agent_x"
    assert states[0].agent_trace["tool_calls"]
    assert states[0].agent_trace["tool_results"]


def test_game_loop_exposes_tools_only_on_controllable_frames(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=ModelRegistry(
            world_tool=FakeWorldTool(),
            goal_tool=FakeGoalTool(),
            orchestrator_agent=agent,
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert len(agent.tool_runtimes) == 3
    assert agent.tool_runtimes[0].available_tools() == ()
    assert agent.tool_runtimes[1].available_tools() == ()
    assert agent.tool_runtimes[2].available_tools() == ("world", "goal")
    with pytest.raises(RuntimeError, match="disabled"):
        agent.tool_runtimes[0].invoke(
            ToolCall(
                tool="world",
                observation_ref=agent.tool_runtimes[0].current_observation_ref,
                action=ActionSpec(action_id="ACTION1"),
            )
        )


def test_orchestration_tool_invocation_can_use_e_output_as_source(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    source_observation = Observation(id="obs-0", step=0, frame={"frame": "real"})
    source_ref = ObservationRef(memory="state", id=source_observation.id)
    action = ActionSpec(action_id="ACTION1")
    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=source_observation,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=AgentTrace(
            step=0,
            first_observation_ref=source_ref,
            current_observation_ref=source_ref,
            final_action=action,
        ),
    )
    goal_tool = FakeGoalTool()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=ModelRegistry(
            world_tool=FakeWorldTool(),
            goal_tool=goal_tool,
        ),
    )
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=source_ref,
        current_observation_ref=source_ref,
        current_observation=source_observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )

    first = orchestrator.invoke_tool_for_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        frame_context=frame_context,
        call=ToolCall(tool="world", observation_ref=source_ref, action=action),
    )
    second = orchestrator.invoke_tool_for_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=2,
        frame_context=frame_context,
        call=ToolCall(tool="goal", observation_ref=first.observation_ref),
    )

    assert second.observation_ref.memory == "experimental"
    assert goal_tool.observations[0].frame == {"predicted_from": "obs-0"}


def test_orchestration_tool_invocation_does_not_persist_failed_calls(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    source_observation = Observation(id="obs-0", step=0, frame={"frame": "real"})
    source_ref = ObservationRef(memory="state", id=source_observation.id)
    action = ActionSpec(action_id="ACTION1")
    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=source_observation,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=AgentTrace(
            step=0,
            first_observation_ref=source_ref,
            current_observation_ref=source_ref,
            final_action=action,
        ),
    )
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=ModelRegistry(goal_tool=FakeGoalTool(fail=True)),
    )
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=source_ref,
        current_observation_ref=source_ref,
        current_observation=source_observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )

    with pytest.raises(RuntimeError, match="tool failed"):
        orchestrator.invoke_tool_for_experiment(
            run_id="run-1",
            game_id="game-1",
            turn_id=1,
            frame_context=frame_context,
            call=ToolCall(tool="goal", observation_ref=source_ref),
        )

    assert experimental.list_experiments() == []


def test_failed_environment_shell_does_not_cleanup_m_states(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    observation = Observation(id="previous", step=0, frame={"frame": "old"})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=0,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )
    for run_id in ("old-1", "old-2"):
        state.write_state(
            run_id=run_id,
            game_id="game-1",
            step=0,
            frame_index=0,
            frame_count=1,
            current_observation=observation,
            chosen_action=action,
            contexts=ContextDocuments(),
            agent_trace=trace,
        )
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(orchestrator_agent=FakeAgent()),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    with pytest.raises(RuntimeError, match="boom"):
        runtime.run(
            config=RuntimeConfig(run_id="run-1"),
            environment=FailingStepEnvironment(),
            environment_config=EnvironmentConfig(
                game_index=0,
                game_id="game-1",
                max_actions_per_level=1,
            ),
        )

    assert len(state.list_states(game_id="game-1")) == 2


def test_clean_db_cli_clears_memory_rows_without_starting_arc(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    database_path = tmp_path / "runtime.sqlite"
    database = SQLiteDatabase(database_path)
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=0,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )
    state.write_record(
        run_id="run-1",
        game_id="game-1",
        step=0,
        kind="legacy",
        payload={"keep": True},
    )
    experimental.write_record(
        run_id="run-1",
        game_id="game-1",
        step=0,
        kind="legacy",
        payload={"keep": True},
    )
    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=trace,
    )
    experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="goal", observation_ref=observation_ref),
        output_observation=Observation(id="goal-0", step=0, frame={"goal": True}),
        tool_result=ToolResult(
            id="goal-0",
            tool="goal",
            predicted_observation={"goal": True},
            source_observation_ref=observation_ref,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "shell",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--database",
            str(database_path),
            "--clean-db",
        ],
    )

    shell.main()

    assert state.list_states() == []
    assert state.list_records(game_id="game-1") == []
    assert experimental.list_records(game_id="game-1") == []
    assert experimental.list_experiments(game_id="game-1") == []
    assert "cleared memory database rows" in capsys.readouterr().out


def test_shell_model_registry_keeps_random_agent_default() -> None:
    registry = shell._build_model_registry(
        agent_config=ModelRoleConfig(),
        world_config=ModelRoleConfig(),
        goal_config=ModelRoleConfig(),
    )

    assert isinstance(registry.orchestrator_agent, RandomOrchestratorAgentAdapter)
    assert registry.world_tool is None
    assert registry.goal_tool is None


def test_environment_config_loads_prompt_model_calls_flag(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
                "  prompt_model_calls_enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.models.prompt_model_calls_enabled is True


def test_environment_config_loads_cheat_action_context_flag(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    game_dir = tmp_path / "game"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "cheat_action_context: true",
                f"cheat_action_context_game_dir: {game_dir}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.cheat_action_context is True
    assert config.cheat_action_context_game_dir == str(game_dir)


def test_shell_builds_cheat_context_documents_from_game_source(tmp_path) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    (game_dir / "game.py").write_text(
        """
from arcengine import GameAction


class Game:
    def step(self, action):
        dx = 0
        dy = 0
        if action == GameAction.ACTION1:
            dy = -1
        if action == GameAction.ACTION2:
            dy = 1
        if action == GameAction.ACTION3:
            dx = -1
        if action == GameAction.ACTION4:
            dx = 1
        x_pos, y_pos = (self.gisrhqpee * dx, self.tbwnoxqgc * dy)
        position = (x_pos, y_pos)
        return position
""".lstrip(),
        encoding="utf-8",
    )
    config = EnvironmentConfig(
        game_index=0,
        game_id="ls20-9607627b",
        max_actions_per_level=1,
        cheat_action_context=True,
        cheat_action_context_game_dir=str(game_dir),
    )

    contexts = shell._build_context_documents(config)

    assert "Cheat action context from the local game source:" in contexts.agent.game
    assert "ACTION1: up arrow" in contexts.agent.game
    assert "ACTION4: right arrow" in contexts.world.game
    assert "ACTION4: right arrow" in contexts.goal.game


def test_shell_orchestrator_defaults_prompt_model_calls_disabled(tmp_path) -> None:
    orchestrator = shell._build_orchestrator(tmp_path / "runtime.sqlite")

    assert orchestrator.prompt_model_calls_enabled is False


def test_shell_orchestrator_wires_prompt_model_calls_flag(tmp_path) -> None:
    orchestrator = shell._build_orchestrator(
        tmp_path / "runtime.sqlite",
        prompt_model_calls_enabled=True,
    )

    assert orchestrator.prompt_model_calls_enabled is True


def test_shell_model_registry_wires_openai_roles_explicitly() -> None:
    registry = shell._build_model_registry(
        agent_config=ModelRoleConfig(
            backend="openai",
            model="gpt-5-nano",
            max_tool_calls=2,
            repair_attempts=1,
        ),
        world_config=ModelRoleConfig(backend="openai"),
        goal_config=ModelRoleConfig(backend="openai"),
    )

    assert isinstance(registry.orchestrator_agent, OpenAIOrchestratorAgentAdapter)
    assert isinstance(registry.world_tool, OpenAIWorldToolAdapter)
    assert isinstance(registry.goal_tool, OpenAIGoalToolAdapter)


def test_shell_model_registry_wires_ollama_agent_without_auto_tools() -> None:
    registry = shell._build_model_registry(
        agent_config=ModelRoleConfig(
            backend="ollama",
            model="gemma4:e4b",
            max_tool_calls=2,
            repair_attempts=1,
        ),
        world_config=ModelRoleConfig(backend="none"),
        goal_config=ModelRoleConfig(backend="none"),
    )

    assert isinstance(registry.orchestrator_agent, OllamaOrchestratorAgentAdapter)
    assert registry.world_tool is None
    assert registry.goal_tool is None
