"""Smoke tests for the runtime game-loop boundary."""

import base64
from collections.abc import Sequence
from io import BytesIO, StringIO

from arcengine import GameState
import numpy as np
from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionHistoryEntry,
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
    TurnMetrics,
    RoleContext,
    RuntimeConfig,
    ToolCall,
    ToolResult,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.environment.config import ModelRoleConfig
from face_of_agi.environment.config import UpdaterRuntimeConfig
from face_of_agi.environment.config import load_environment_config
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models import ModelRegistry
from face_of_agi.models import (
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    UpdaterTaskRegistry,
    WorldPredictionAdapter,
    WorldGameContextUpdateInput,
)
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.world import WorldPredictionAdapter
from face_of_agi.orchestration import Orchestrator
from face_of_agi.debug.sanitize import sanitize_for_debug
from face_of_agi.orchestration.game_loop.helpers import unroll_observation
from face_of_agi.orchestration.game_loop.actions.metrics import (
    effective_trace_cost_seconds,
)
from face_of_agi.runtime import RuntimeLoop
from face_of_agi.runtime import shell


class FakeAgent:
    """Fake agent that returns one final action and trace."""

    def decide(
        self,
        context: RoleContext,
        history_anchor_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        del tool_runtime, recent_action_history
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


class LongReasoningAgent(FakeAgent):
    """Fake agent that emits a long reasoning field for wrapping tests."""

    def decide(
        self,
        context: RoleContext,
        history_anchor_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        decision = super().decide(
            context=context,
            history_anchor_observation=history_anchor_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
            recent_action_history=recent_action_history,
        )
        decision.trace.reasoning_summary = (
            "long reasoning field " * 30
            + "terminal wrapping should preserve this final sentence"
        )
        return decision


class LoadingTraceAgent(FakeAgent):
    """Fake agent that reports provider/model load durations in its trace."""

    def decide(
        self,
        context: RoleContext,
        history_anchor_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        decision = super().decide(
            context=context,
            history_anchor_observation=history_anchor_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
            recent_action_history=recent_action_history,
        )
        observation_ref = ObservationRef(memory="state", id=current_observation.id)
        decision.trace.metadata["usage"] = [
            {"load_duration": 2_000_000_000},
        ]
        decision.trace.tool_results.append(
            ToolResult(
                id="world-load",
                tool="world",
                predicted_description=[],
                source_observation_ref=observation_ref,
                metadata={"usage": {"load_duration": 2_000_000_000}},
            )
        )
        return decision


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


class CountingSingleFrameEnvironment(FrameBundleEnvironment):
    """Fake ARC adapter that exposes unique frames across real actions."""

    def __init__(self) -> None:
        super().__init__()
        self.step_count = 0

    def reset(self) -> Observation:
        self.reset_calls += 1
        self.step_count = 0
        return Observation(id="obs-0", step=0, frame={"frame": 0})

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del reasoning
        if action.is_none():
            raise AssertionError("synthetic NONE must never be sent to ARC")
        self.step_actions.append(action)
        self.step_count += 1
        return Observation(
            id=f"obs-{self.step_count}",
            step=self.step_count,
            frame={"frame": self.step_count},
        )


class DuplicateFrameBundleEnvironment(FrameBundleEnvironment):
    """Fake ARC adapter that returns duplicate animation frames."""

    def reset(self) -> Observation:
        self.reset_calls += 1
        return Observation(
            id="obs-reset",
            step=0,
            frames=(
                {"frame": 0},
                {"frame": 0},
                {"frame": 1},
                {"frame": 1},
                {"frame": 2},
                {"frame": 2},
            ),
        )


class ImageScoreEnvironment(FrameBundleEnvironment):
    """Fake ARC adapter with visual frames and level progress metadata."""

    def reset(self) -> Observation:
        self.reset_calls += 1
        return Observation(
            id="obs-reset",
            step=0,
            frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
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
            frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
            metadata={"levels_completed": 1},
        )

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
            levels_completed=0,
        )


class GameOverAfterOneStepEnvironment(SingleFrameEnvironment):
    """Fake ARC adapter that resets once after a completed environment step."""

    def __init__(self) -> None:
        super().__init__()
        self._game_over_pending = False
        self._real_steps = 0

    def reset(self) -> Observation:
        self.reset_calls += 1
        return Observation(
            id=f"obs-reset-{self.reset_calls}",
            step=0,
            frame={"frame": f"reset-{self.reset_calls}"},
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
        self._real_steps += 1
        if self._real_steps == 1:
            self._game_over_pending = True
        return Observation(
            id=f"obs-after-action-{self._real_steps}",
            step=self._real_steps,
            frame={"frame": self._real_steps},
        )

    def get_info(self) -> EnvironmentInfo:
        if self._game_over_pending:
            self._game_over_pending = False
            return EnvironmentInfo(
                game_id="game-1",
                state=GameState.GAME_OVER,
                available_actions=tuple(self.get_action_space()),
            )
        if self._real_steps >= 2:
            return EnvironmentInfo(
                game_id="game-1",
                state=GameState.WIN,
                available_actions=tuple(self.get_action_space()),
            )
        return super().get_info()


class CapturingAgent(FakeAgent):
    """Fake agent that records the context it receives."""

    def __init__(self) -> None:
        self.contexts: list[RoleContext] = []
        self.history_anchor_observations: list[Observation] = []
        self.current_observations: list[Observation] = []
        self.recent_action_histories: list[tuple[ActionHistoryEntry, ...]] = []
        self.tool_runtimes: list[object | None] = []

    def decide(
        self,
        context: RoleContext,
        history_anchor_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        self.contexts.append(context)
        self.history_anchor_observations.append(history_anchor_observation)
        self.current_observations.append(current_observation)
        self.recent_action_histories.append(recent_action_history)
        self.tool_runtimes.append(tool_runtime)
        return super().decide(
            context=context,
            history_anchor_observation=history_anchor_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
            recent_action_history=recent_action_history,
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


def test_unroll_observation_keeps_second_frame_from_duplicate_pair() -> None:
    first = np.array([[1]], dtype=np.uint8)
    second = np.array([[1]], dtype=np.uint8)

    frames = unroll_observation(Observation(id="obs", step=0, frames=(first, second)))

    assert len(frames) == 1
    assert frames[0].id == "obs-frame-0"
    assert frames[0].frame is second
    assert frames[0].metadata["frame_index"] == 0
    assert frames[0].metadata["frame_count"] == 1


def test_unroll_observation_keeps_rightmost_frame_from_each_duplicate_run() -> None:
    frames_in = tuple(
        np.array([[value]], dtype=np.uint8)
        for value in (0, 0, 1, 1, 2, 2)
    )

    frames = unroll_observation(Observation(id="obs", step=0, frames=frames_in))

    assert [frame.id for frame in frames] == [
        "obs-frame-0",
        "obs-frame-1",
        "obs-frame-2",
    ]
    assert frames[0].frame is frames_in[1]
    assert frames[1].frame is frames_in[3]
    assert frames[2].frame is frames_in[5]
    assert [frame.metadata["frame_index"] for frame in frames] == [0, 1, 2]
    assert [frame.metadata["frame_count"] for frame in frames] == [3, 3, 3]


def test_unroll_observation_keeps_distinct_frames_and_shape_mismatches() -> None:
    frames_in = (
        np.array([[0]], dtype=np.uint8),
        np.array([[1]], dtype=np.uint8),
        np.array([[1, 1]], dtype=np.uint8),
    )

    frames = unroll_observation(Observation(id="obs", step=0, frames=frames_in))

    assert tuple(frame.frame for frame in frames) == frames_in


def test_unroll_observation_keeps_last_frame_when_all_frames_are_identical() -> None:
    frames_in = tuple(np.array([[7]], dtype=np.uint8) for _ in range(3))

    frames = unroll_observation(Observation(id="obs", step=0, frames=frames_in))

    assert len(frames) == 1
    assert frames[0].frame is frames_in[-1]
    assert frames[0].metadata["frame_count"] == 1


class FakeWorldPredictionModel:
    """Fake world prediction model that records the source observation."""

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
            predicted_description={"predicted_from": observation.id},
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action,
        )


class InspectableWorldPredictionModel(FakeWorldPredictionModel):
    """Fake world prediction model with captured provider input."""

    def __init__(self) -> None:
        super().__init__()
        self.last_prompt: str | None = None
        self.last_request: dict[str, object] | None = None

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> ToolResult:
        self.last_prompt = "WORLD PROVIDER PROMPT"
        self.last_request = {
            "api_key": "secret",
            "input": [
                {
                    "content": [
                        {"type": "input_text", "text": self.last_prompt},
                        {
                            "type": "input_image",
                            "image_url": _tiny_png_data_url(),
                            "detail": "auto",
                        },
                    ]
                }
            ],
        }
        return super().predict(context, action, observation)


class FakeGoalPredictionModel:
    """Fake goal prediction model that records the source observation."""

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
            predicted_description={"goal_from": observation.id},
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
        )


class InspectableGoalPredictionModel(FakeGoalPredictionModel):
    """Fake goal prediction model with captured provider input."""

    def __init__(self) -> None:
        super().__init__()
        self.last_prompt: str | None = None
        self.last_request: dict[str, object] | None = None

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        self.last_prompt = "GOAL PROVIDER PROMPT"
        self.last_request = {
            "authorization": "Bearer secret",
            "prompt": self.last_prompt,
        }
        return super().predict(context, observation)


def _tiny_png_data_url() -> str:
    image = Image.new("RGB", (2, 3), color=(1, 2, 3))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class DualToolCallingAgent(FakeAgent):
    """Fake X shell that would call tools if any were exposed."""

    def decide(
        self,
        context: RoleContext,
        history_anchor_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        decision = super().decide(
            context=context,
            history_anchor_observation=history_anchor_observation,
            current_observation=current_observation,
            action_space=action_space,
            tool_runtime=tool_runtime,
            recent_action_history=recent_action_history,
        )
        if tool_runtime is None or not tool_runtime.available_tools():
            return decision

        world_call = ToolCall(
            tool="world",
            source_state_id=tool_runtime.current_source_state_id,
            action=decision.final_action,
        )
        goal_call = ToolCall(
            tool="goal",
            source_state_id=tool_runtime.current_source_state_id,
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
        self.world_inputs: list[WorldGameContextUpdateInput] = []
        self.goal_inputs: list[GoalGameContextUpdateInput] = []
        self.agent_inputs: list[AgentGameContextUpdateInput] = []
        self.general_inputs: list[GeneralKnowledgeUpdateInput] = []

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        self.world_calls += 1
        self.world_inputs.append(update_input)
        return RoleContext(
            general=update_input.previous_context.general,
            game=f"world-{self.world_calls}",
        )

    def update_goal_game_context(
        self,
        update_input: GoalGameContextUpdateInput,
    ) -> RoleContext:
        self.goal_calls += 1
        self.goal_inputs.append(update_input)
        return RoleContext(
            general=update_input.previous_context.general,
            game=f"goal-{self.goal_calls}",
        )

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        self.agent_calls += 1
        self.agent_inputs.append(update_input)
        return RoleContext(
            general=update_input.previous_context.general,
            game=f"agent-{self.agent_calls}",
        )

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        self.general_inputs.append(update_input)
        return RoleContext(
            general=f"{update_input.role}-general-{len(self.general_inputs)}",
            game=update_input.previous_context.game,
        )


class InspectableUpdater(MutatingFakeUpdater):
    """Fake updater with provider-like debug capture fields."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: dict[str, object] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, object] | None = None

    def _capture_provider_io(self, role: str) -> None:
        self.last_request = {
            "api_key": "secret",
            "role": role,
            "image_url": _tiny_png_data_url(),
        }
        self.last_response_text = f"UPDATER RAW OUTPUT {role}"
        self.last_response_metadata = {
            "backend": "fake",
            "usage": {"load_duration": 1_000_000_000},
        }

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        self._capture_provider_io("world")
        return super().update_world_game_context(update_input)

    def update_goal_game_context(
        self,
        update_input: GoalGameContextUpdateInput,
    ) -> RoleContext:
        self._capture_provider_io("goal")
        return super().update_goal_game_context(update_input)

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        self._capture_provider_io("agent")
        return super().update_agent_game_context(update_input)

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        self._capture_provider_io(update_input.role)
        return super().update_general_knowledge(update_input)


class FailingInspectableUpdater(MutatingFakeUpdater):
    """Fake updater that captures provider output before failing."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: dict[str, object] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, object] | None = None

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        del update_input
        self.last_request = {"role": "world"}
        self.last_response_text = "UPDATER RAW OUTPUT BEFORE FAILURE"
        self.last_response_metadata = {"backend": "fake"}
        raise RuntimeError("updater parse failed")


class PassiveTestUpdater:
    """Test-only updater that leaves contexts unchanged."""

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context

    def update_goal_game_context(
        self,
        update_input: GoalGameContextUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context


def _updater_tasks(updater: MutatingFakeUpdater) -> UpdaterTaskRegistry:
    return UpdaterTaskRegistry(
        world_game_updater=updater,
        goal_game_updater=updater,
        agent_game_updater=updater,
        general_updater=updater,
    )


def _passive_updater_tasks() -> UpdaterTaskRegistry:
    return UpdaterTaskRegistry(
        world_game_updater=PassiveTestUpdater(),
        goal_game_updater=PassiveTestUpdater(),
        agent_game_updater=PassiveTestUpdater(),
        general_updater=PassiveTestUpdater(),
    )


def _openai_updater_config() -> UpdaterRuntimeConfig:
    return UpdaterRuntimeConfig(
        world=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        agent=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
    )


def _openai_agent_config() -> ModelRoleConfig:
    return ModelRoleConfig(
        backend="openai",
        model="gpt-5-nano",
        max_tool_calls=0,
        repair_attempts=1,
    )


def _openai_tool_config() -> ModelRoleConfig:
    return ModelRoleConfig(backend="openai", model="gpt-5-nano")


def _runtime_models(
    *,
    world_prediction_model: object | None = None,
    goal_prediction_model: object | None = None,
    orchestrator_agent: object | None = None,
    updater_tasks: UpdaterTaskRegistry | None = None,
) -> ModelRegistry:
    return ModelRegistry(
        world_prediction_model=world_prediction_model or FakeWorldPredictionModel(),
        goal_prediction_model=goal_prediction_model,
        orchestrator_agent=orchestrator_agent or FakeAgent(),
        updater_tasks=updater_tasks or _passive_updater_tasks(),
    )


def test_environment_shell_unrolls_frames_and_steps_only_on_final_frame(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=_runtime_models(updater_tasks=_passive_updater_tasks()),
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
    assert output.getvalue().count("orchestration synthesized NONE") == 2
    assert "X selected ACTION1" in output.getvalue()
    assert len(state.list_states(game_id="game-1")) == 1
    assert experimental.list_experiments(run_id="run-1", game_id="game-1") == []


def test_environment_shell_skips_duplicate_unrolled_frames(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    world_model = FakeWorldPredictionModel()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            world_prediction_model=world_model,
            orchestrator_agent=FakeAgent(),
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)
    environment = DuplicateFrameBundleEnvironment()

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    assert len(environment.step_actions) == 1
    assert output.getvalue().count("orchestration synthesized NONE") == 2
    assert "X selected ACTION1" in output.getvalue()
    assert [observation.id for observation in world_model.observations] == [
        "obs-reset-frame-0",
        "obs-reset-frame-1",
        "obs-reset-frame-2",
    ]
    assert [observation.frame for observation in world_model.observations] == [
        {"frame": 0},
        {"frame": 1},
        {"frame": 2},
    ]


def test_environment_shell_debug_trace_off_suppresses_output(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    orchestrator = Orchestrator(
        state_memory=StateMemory(database),
        experimental_memory=ExperimentalMemory(database),
        models=_runtime_models(updater_tasks=_passive_updater_tasks()),
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
            debug_trace="off",
        ),
    )

    assert output.getvalue() == ""


def test_environment_shell_verbose_debug_trace_includes_loop_details(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    orchestrator = Orchestrator(
        state_memory=StateMemory(database),
        experimental_memory=ExperimentalMemory(database),
        models=_runtime_models(
            orchestrator_agent=FakeAgent(),
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
            debug_trace="verbose",
            debug_color="never",
            debug_keep_all_m_states=True,
        ),
    )

    trace = output.getvalue()
    assert "Run start" in trace
    assert "Frame turn" in trace
    assert "Agent X decision" in trace
    assert "fake trace" in trace
    assert "Post-decision predictions" in trace
    assert "Environment step" in trace
    assert "Persisted M state" in trace
    assert "Run stop" in trace
    assert "action_limit_reached" in trace


def test_environment_shell_agent_decision_trace_only_shows_x_decision(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    orchestrator = Orchestrator(
        state_memory=StateMemory(database),
        experimental_memory=ExperimentalMemory(database),
        models=_runtime_models(
            orchestrator_agent=LongReasoningAgent(),
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
            debug_trace="agent_decision",
            debug_color="never",
            debug_keep_all_m_states=True,
        ),
    )

    trace = output.getvalue()
    assert "Agent X decision" in trace
    assert "terminal wrapping should" in trace
    assert "preserve this final" in trace
    assert "sentence" in trace
    assert max(len(line) for line in trace.splitlines()) <= 100


def test_environment_shell_model_inputs_debug_trace_prints_sanitized_inputs(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    updater = InspectableUpdater()
    orchestrator = Orchestrator(
        state_memory=StateMemory(database),
        experimental_memory=ExperimentalMemory(database),
        models=_runtime_models(
            world_prediction_model=InspectableWorldPredictionModel(),
            orchestrator_agent=DualToolCallingAgent(),
            updater_tasks=_updater_tasks(updater),
        ),
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
            debug_trace="model_inputs",
            debug_color="never",
            debug_keep_all_m_states=True,
        ),
    )

    trace = output.getvalue()
    assert "Agent X framework input" in trace
    assert "world model input" in trace
    assert "world provider input" in trace
    assert "WORLD PROVIDER PROMPT" in trace
    assert "[redacted]" in trace
    assert "omitted_image_data_url" in trace
    assert "Updater P agent input" in trace
    assert "Updater P agent provider output" in trace
    assert "UPDATER RAW OUTPUT agent" in trace


def test_model_inputs_trace_logs_updater_provider_output_on_failure(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    updater = FailingInspectableUpdater()
    orchestrator = Orchestrator(
        state_memory=StateMemory(database),
        models=_runtime_models(
            world_prediction_model=FakeWorldPredictionModel(),
            goal_prediction_model=FakeGoalPredictionModel(),
            orchestrator_agent=FakeAgent(),
            updater_tasks=UpdaterTaskRegistry(
                world_game_updater=updater,
                goal_game_updater=PassiveTestUpdater(),
                agent_game_updater=PassiveTestUpdater(),
                general_updater=PassiveTestUpdater(),
            ),
        ),
    )
    output = StringIO()
    runtime = RuntimeLoop(orchestrator, trace_output=output)

    with pytest.raises(RuntimeError, match="updater parse failed"):
        runtime.run(
            config=RuntimeConfig(run_id="run-1"),
            environment=SingleFrameEnvironment(),
            environment_config=EnvironmentConfig(
                game_index=0,
                game_id="game-1",
                max_actions_per_level=1,
                debug_trace="model_inputs",
                debug_color="never",
            ),
        )

    trace = output.getvalue()
    assert "Updater P world provider output" in trace
    assert "UPDATER RAW OUTPUT BEFORE FAILURE" in trace


def test_debug_trace_sanitizes_secrets_and_image_payloads() -> None:
    sanitized = sanitize_for_debug(
        {
            "api_key": "secret",
            "headers": {"authorization": "Bearer secret"},
            "image_url": _tiny_png_data_url(),
            "max_output_tokens": 256,
            "session_token": "secret",
            "plain_text": "keep me",
        }
    )

    assert sanitized["api_key"] == "[redacted]"
    assert sanitized["headers"]["authorization"] == "[redacted]"
    assert sanitized["plain_text"] == "keep me"
    assert sanitized["max_output_tokens"] == 256
    assert sanitized["session_token"] == "[redacted]"
    assert sanitized["image_url"]["kind"] == "omitted_image_data_url"
    assert sanitized["image_url"]["mime_type"] == "image/png"
    assert sanitized["image_url"]["image_size"] == [2, 3]


def test_orchestration_writes_m_state_for_each_frame_turn_without_cleanup(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(updater_tasks=_passive_updater_tasks()),
    )

    result = orchestrator.run_environment_shell(
        config=RuntimeConfig(run_id="run-1"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
            debug_keep_all_m_states=True,
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
        contexts=ContextDocuments(
            agent=RoleContext(
                general="learned general context",
                game="learned game context",
            )
        ),
        agent_trace=trace,
    )
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
        ),
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
    assert {context.general for context in agent.contexts} == {
        "learned general context"
    }
    assert {context.game for context in agent.contexts} == {"learned game context"}


def test_environment_shell_can_skip_learned_context_hydration(tmp_path) -> None:
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
        contexts=ContextDocuments(
            agent=RoleContext(
                general="learned general context",
                game="learned game context",
            )
        ),
        agent_trace=trace,
    )
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-2"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
            use_learned_contexts=False,
        ),
    )

    assert agent.contexts
    assert {context.general for context in agent.contexts} == {""}
    assert {context.game for context in agent.contexts} == {""}


def test_environment_shell_carries_global_k_without_cross_game_l_leakage(
    tmp_path,
) -> None:
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
        contexts=ContextDocuments(
            agent=RoleContext(
                general="cross-game agent knowledge",
                game="game-1-only agent knowledge",
            )
        ),
        agent_trace=trace,
    )
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-2"),
        environment=FrameBundleEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-2",
            max_actions_per_level=1,
        ),
    )

    assert agent.contexts
    assert {context.general for context in agent.contexts} == {
        "cross-game agent knowledge"
    }
    assert {context.game for context in agent.contexts} == {""}


def test_environment_shell_uses_default_contexts_when_m_state_is_empty(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
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

    assert agent.contexts
    assert {context.game for context in agent.contexts} == {""}


def test_game_loop_injects_updater_agent_context_on_next_x_call(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_updater_tasks(updater),
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

    assert [context.game for context in agent.contexts] == ["agent-2"]
    assert updater.agent_calls == 3
    assert updater.general_inputs == []


def test_game_loop_exposes_synthetic_none_turns_in_action_history(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
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
            debug_keep_all_m_states=True,
        ),
    )

    states = state.list_states(game_id="game-1")
    assert len(agent.recent_action_histories) == 1
    final_history = agent.recent_action_histories[0]

    assert len(states) == 3
    assert [entry.action.name for entry in final_history] == ["NONE", "NONE"]
    assert [entry.controllable for entry in final_history] == [False, False]
    assert states[0].chosen_action["action_id"] == "NONE"
    assert states[0].agent_trace["tool_calls"] == []
    assert states[0].agent_trace["tool_results"] == []
    assert states[0].agent_trace["metadata"] == {
        "decision_source": "orchestration_synthetic_none",
        "agent_x_called": False,
    }


def test_game_loop_exposes_prior_real_actions_in_action_history(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=SingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=2,
        ),
    )

    assert len(agent.recent_action_histories) == 2
    first_history = agent.recent_action_histories[0]
    second_history = agent.recent_action_histories[1]

    assert first_history == ()
    assert len(second_history) == 1
    assert second_history[0].action.action_id == "ACTION1"
    assert second_history[0].controllable is True
def test_game_loop_runs_agent_after_game_over_reset(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_updater_tasks(updater),
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    result = runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=GameOverAfterOneStepEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=5,
        ),
    )

    assert result.stop_reason == "game_end"
    assert len(agent.contexts) == 2
    assert [update.role for update in updater.general_inputs] == [
        "world",
        "agent",
    ]


def test_game_loop_trims_action_history_window_and_persists_snapshot(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
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
            action_history_window=1,
            debug_keep_all_m_states=True,
        ),
    )

    final_history = agent.recent_action_histories[0]
    states = state.list_states(game_id="game-1")

    assert len(final_history) == 1
    assert states[-1].metadata["recent_action_history"] == [
        {
            "action": {"action_id": "NONE", "data": None},
            "controllable": False,
        }
    ]


def test_game_loop_uses_oldest_visible_action_source_as_history_anchor(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=CountingSingleFrameEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=7,
            action_history_window=5,
        ),
    )

    final_history = agent.recent_action_histories[-1]

    assert len(agent.history_anchor_observations) == 7
    assert len(final_history) == 5
    assert [entry.action.name for entry in final_history] == ["ACTION1"] * 5
    assert agent.history_anchor_observations[-1].id == "obs-1"
    assert agent.history_anchor_observations[-1].frame == {"frame": 1}
    assert agent.current_observations[-1].id == "obs-6"
    assert agent.current_observations[-1].frame == {"frame": 6}


def test_game_loop_persists_updater_contexts_from_post_decision_predictions(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    world_model = FakeWorldPredictionModel()
    goal_model = FakeGoalPredictionModel()
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=_runtime_models(
            world_prediction_model=world_model,
            goal_prediction_model=goal_model,
            orchestrator_agent=DualToolCallingAgent(),
            updater_tasks=_updater_tasks(updater),
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
    experiments = experimental.list_experiments(run_id="run-1", game_id="game-1")
    assert result.stop_reason == "action_limit_reached"
    assert len(world_model.contexts) == 3
    assert goal_model.contexts == []
    assert experiments == []
    assert states[-1].world_context.game == "world-3"
    assert states[-1].goal_context.game == ""
    assert states[-1].agent_context.game == "agent-3"
    assert states[-1].world_context.general == ""
    assert states[-1].goal_context.general == ""
    assert states[-1].agent_context.general == ""
    assert updater.world_calls == 3
    assert updater.goal_calls == 0
    assert updater.agent_calls == 3
    assert updater.general_inputs == []


def test_game_loop_uses_distinct_updater_task_slots(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    world_updater = MutatingFakeUpdater()
    goal_updater = MutatingFakeUpdater()
    agent_updater = MutatingFakeUpdater()
    general_updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=FakeAgent(),
            updater_tasks=UpdaterTaskRegistry(
                world_game_updater=world_updater,
                goal_game_updater=goal_updater,
                agent_game_updater=agent_updater,
                general_updater=general_updater,
            ),
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

    assert world_updater.world_calls == 3
    assert world_updater.goal_calls == 0
    assert world_updater.agent_calls == 0
    assert goal_updater.world_calls == 0
    assert goal_updater.goal_calls == 0
    assert goal_updater.agent_calls == 0
    assert agent_updater.world_calls == 0
    assert agent_updater.goal_calls == 0
    assert agent_updater.agent_calls == 3
    assert general_updater.general_inputs == []


def test_real_step_turn_metrics_reaches_updater_and_persists_to_m(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=FakeAgent(),
            updater_tasks=_updater_tasks(updater),
        ),
    )
    runtime = RuntimeLoop(orchestrator, trace_output=StringIO())

    runtime.run(
        config=RuntimeConfig(run_id="run-1"),
        environment=ImageScoreEnvironment(),
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-1",
            max_actions_per_level=1,
        ),
    )

    metrics = updater.agent_inputs[0].turn_metrics
    assert metrics.time_cost == 1.0
    assert metrics.cumulative_score == 1.0
    assert metrics.agent_context_word_count == 0
    assert [entry.action.name for entry in updater.agent_inputs[0].action_history] == [
        "ACTION1"
    ]
    assert [entry.controllable for entry in updater.agent_inputs[0].action_history] == [
        True
    ]
    states = state.list_states(game_id="game-1")
    assert states[0].turn_metrics.time_cost == 1.0
    assert states[0].turn_metrics.cumulative_score == 1.0


def test_effective_trace_cost_subtracts_agent_and_tool_load_duration() -> None:
    observation_ref = ObservationRef(memory="state", id="obs-1")
    decision = DecisionResult(
        final_action=ActionSpec(action_id="ACTION1"),
        trace=AgentTrace(
            step=1,
            first_observation_ref=observation_ref,
            current_observation_ref=observation_ref,
            final_action=ActionSpec(action_id="ACTION1"),
            tool_results=[
                ToolResult(
                    id="world-load",
                    tool="world",
                    predicted_description=[],
                    source_observation_ref=observation_ref,
                    metadata={"usage": {"load_duration": 2_000_000_000}},
                )
            ],
            metadata={"usage": [{"load_duration": 1_000_000_000}]},
        ),
    )

    assert (
        effective_trace_cost_seconds(
            decision=decision,
            wall_clock_seconds=10.0,
        )
        == 7.0
    )
    assert (
        effective_trace_cost_seconds(
            decision=decision,
            wall_clock_seconds=1.0,
        )
        == 0.0
    )


def test_runtime_trace_cost_uses_load_adjusted_decision_time(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=LoadingTraceAgent(),
            updater_tasks=_updater_tasks(updater),
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
    assert states[0].turn_metrics.trace_cost == 0.0


def test_animation_frame_predictions_and_turn_metrics_reach_updaters(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    updater = MutatingFakeUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            orchestrator_agent=FakeAgent(),
            updater_tasks=_updater_tasks(updater),
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

    assert updater.world_calls == 3
    assert updater.goal_calls == 0
    assert updater.agent_calls == 3
    assert updater.world_inputs[0].synthetic_none_action is not None
    assert updater.world_inputs[0].synthetic_none_action.is_none()
    assert updater.world_inputs[0].current_observation.id == "obs-reset-frame-1"
    assert (
        updater.world_inputs[0].post_decision_predictions.world_prediction
        is not None
    )
    assert updater.agent_inputs[0].turn_metrics.time_cost == 0.0
    assert updater.agent_inputs[0].previous_observation.id == "obs-reset-frame-0"
    assert updater.agent_inputs[0].current_observation.id == "obs-reset-frame-1"
    assert updater.agent_inputs[1].turn_metrics.time_cost == 0.0
    assert updater.agent_inputs[1].previous_observation.id == "obs-reset-frame-1"
    assert updater.agent_inputs[1].current_observation.id == "obs-reset-frame-2"
    assert updater.agent_inputs[2].turn_metrics.time_cost == 1.0
    assert updater.agent_inputs[2].previous_observation.id == "obs-reset-frame-2"
    assert updater.agent_inputs[2].current_observation.id == "obs-after-action"
    assert [
        input_.turn_metrics.agent_context_word_count
        for input_ in updater.agent_inputs
    ] == [0, 1, 1]
    assert [
        input_.current_turn_world_game_context
        for input_ in updater.agent_inputs
    ] == ["", "world-1", "world-2"]
    assert [
        input_.previous_turn_world_game_context
        for input_ in updater.agent_inputs
    ] == [None, None, "world-1"]
    assert [
        [entry.action.name for entry in input_.action_history]
        for input_ in updater.agent_inputs
    ] == [["NONE"], ["NONE", "NONE"], ["NONE", "NONE", "ACTION1"]]
    assert [
        [entry.controllable for entry in input_.action_history]
        for input_ in updater.agent_inputs
    ] == [[False], [False, False], [False, False, True]]

def test_real_post_decision_predictions_call_world_model_and_persist_to_m(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    world_model = FakeWorldPredictionModel()
    goal_model = FakeGoalPredictionModel()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            world_prediction_model=world_model,
            goal_prediction_model=goal_model,
            orchestrator_agent=FakeAgent(),
            updater_tasks=_passive_updater_tasks(),
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
    assert [observation.id for observation in world_model.observations] == ["obs-reset"]
    assert goal_model.observations == []
    assert [action.name for action in world_model.actions] == ["ACTION1"]
    assert states[0].world_prediction["predicted_description"] == {
        "predicted_from": "obs-reset"
    }
    assert states[0].goal_prediction is None
    assert states[0].world_prediction["metadata"]["purpose"] == (
        "post_decision_update_prediction"
    )
    assert states[0].agent_trace["tool_results"] == []


def test_real_post_decision_predictions_require_configured_models(tmp_path) -> None:
    orchestrator = Orchestrator(
        state_memory=StateMemory(SQLiteDatabase(tmp_path / "runtime.sqlite")),
        models=ModelRegistry(
            orchestrator_agent=FakeAgent(),
            updater_tasks=_passive_updater_tasks(),
        ),
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


def test_non_controllable_frames_run_post_decision_predictions(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    world_model = FakeWorldPredictionModel()
    goal_model = FakeGoalPredictionModel()
    orchestrator = Orchestrator(
        state_memory=state,
        models=_runtime_models(
            world_prediction_model=world_model,
            goal_prediction_model=goal_model,
            orchestrator_agent=FakeAgent(),
            updater_tasks=_passive_updater_tasks(),
        ),
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
    assert [observation.id for observation in world_model.observations] == [
        "obs-reset-frame-0",
        "obs-reset-frame-1",
        "obs-reset-frame-2",
    ]
    assert goal_model.observations == []
    assert [action.name for action in world_model.actions] == [
        "NONE",
        "NONE",
        "ACTION1",
    ]
    assert [state.world_prediction is not None for state in states] == [
        True,
        True,
        True,
    ]
    assert [state.goal_prediction is not None for state in states] == [
        False,
        False,
        False,
    ]


def test_game_loop_runs_world_model_after_agent_decision(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    world_model = FakeWorldPredictionModel()
    goal_model = FakeGoalPredictionModel()
    agent = CapturingAgent()
    orchestrator = Orchestrator(
        state_memory=state,
        experimental_memory=experimental,
        models=_runtime_models(
            world_prediction_model=world_model,
            goal_prediction_model=goal_model,
            orchestrator_agent=agent,
            updater_tasks=_passive_updater_tasks(),
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
    assert experiments == []
    assert [observation.id for observation in world_model.observations] == [
        "obs-reset"
    ]
    assert goal_model.observations == []
    assert states[0].world_prediction is not None
    assert states[0].goal_prediction is None


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
        models=_runtime_models(
            orchestrator_agent=FakeAgent(),
            updater_tasks=_passive_updater_tasks(),
        ),
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
    source_state = state.write_state(
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
        tool_call=ToolCall(tool="goal", source_state_id=source_state.id),
        output_description=Observation(id="goal-0", step=0, frame={"goal": True}),
        tool_result=ToolResult(
            id="goal-0",
            tool="goal",
            predicted_description={"goal": True},
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
    assert experimental.list_experiments(game_id="game-1") == []
    assert "cleared memory database rows" in capsys.readouterr().out


def test_shell_model_registry_requires_agent_backend() -> None:
    with pytest.raises(ValueError, match="models.agent.backend is required"):
        shell._build_model_registry(
            agent_config=ModelRoleConfig(),
            world_config=ModelRoleConfig(backend="openai"),
            goal_config=ModelRoleConfig(backend="openai"),
            updater_config=_openai_updater_config(),
        )


def test_environment_config_loads_shared_vlm_config(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
                "  shared_vlm:",
                "    backend: ollama",
                "    model: qwen3.6",
                "    keep_alive: 30m",
                "    temperature: 0",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.models.shared_vlm.backend == "ollama"
    assert config.models.shared_vlm.model == "qwen3.6"
    assert config.models.shared_vlm.options["keep_alive"] == "30m"
    assert config.models.shared_vlm.options["temperature"] == 0


def test_environment_config_defaults_action_history_window(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.action_history_window == 8
    assert config.use_learned_contexts is True
    assert config.debug_trace == "minimal"
    assert config.debug_color == "auto"


def test_environment_config_allows_disabling_action_history(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "action_history_window: 0",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.action_history_window == 0


def test_environment_config_allows_disabling_learned_contexts(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "use_learned_contexts: false",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.use_learned_contexts is False


def test_environment_config_rejects_negative_action_history_window(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "action_history_window: -1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_history_window must be non-negative"):
        load_environment_config(config_path)


@pytest.mark.parametrize(
    "mode",
    ["off", "minimal", "agent_decision", "verbose", "model_inputs"],
)
def test_environment_config_loads_debug_trace_modes(tmp_path, mode: str) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                f"debug_trace: {mode}",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.debug_trace == mode


@pytest.mark.parametrize("color", ["auto", "always", "never"])
def test_environment_config_loads_debug_color_modes(tmp_path, color: str) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                f"debug_color: {color}",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.debug_color == color


def test_environment_config_rejects_invalid_debug_trace(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "debug_trace: noisy",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="debug_trace must be one of"):
        load_environment_config(config_path)


def test_environment_config_rejects_invalid_debug_color(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "debug_color: rainbow",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="debug_color must be one of"):
        load_environment_config(config_path)


def test_environment_config_requires_updater_config(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.updater config is required"):
        load_environment_config(config_path)


def test_environment_config_requires_each_updater_task(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.updater.general config"):
        load_environment_config(config_path)


def test_environment_config_allows_missing_goal_updater_config(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.models.updater.goal.backend is None


def test_environment_config_loads_updater_role_config(tmp_path) -> None:
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 1",
                "models:",
                "  updater:",
                "    world:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "      instruction_dir: custom/world",
                "    goal:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    agent:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "    general:",
                "      backend: openai",
                "      model: gpt-5-nano",
                "      instruction_dir: custom/general",
            ]
        ),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.models.updater.world.backend == "openai"
    assert config.models.updater.world.options["instruction_dir"] == "custom/world"
    assert config.models.updater.general.backend == "openai"
    assert (
        config.models.updater.general.options["instruction_dir"] == "custom/general"
    )


def test_openai_nano_config_uses_description_options() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/openai/openai_all_gpt5_nano_test.yaml"
    )

    assert config.models.world.backend == "openai"
    assert config.models.world.model == "gpt-5-nano"
    assert config.models.world.options["input_image_size"] == "1024x1024"
    assert config.models.goal.backend == "openai"
    assert config.models.goal.model == "gpt-5-nano"
    assert config.models.goal.options["input_image_size"] == "1024x1024"


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
        updater_config=_openai_updater_config(),
    )

    assert isinstance(registry.orchestrator_agent, OpenAIOrchestratorAgentAdapter)
    assert isinstance(registry.world_prediction_model, WorldPredictionAdapter)
    assert registry.goal_prediction_model is None
    assert registry.updater_tasks.world_game_updater.provider.backend == "openai"
    assert registry.updater_tasks.goal_game_updater is None
    assert registry.updater_tasks.agent_game_updater.provider.backend == "openai"
    assert registry.updater_tasks.general_updater.provider.backend == "openai"


def test_shell_model_registry_wires_updater_task_configs_independently() -> None:
    registry = shell._build_model_registry(
        agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
        updater_config=UpdaterRuntimeConfig(
            world=ModelRoleConfig(backend="openai", model="gpt-5-nano", options={"instruction_dir": "w"}),
            goal=ModelRoleConfig(backend="openai", model="gpt-5-nano", options={"instruction_dir": "g"}),
            agent=ModelRoleConfig(backend="openai", model="gpt-5-nano", options={"instruction_dir": "x"}),
            general=ModelRoleConfig(
                backend="openai", model="gpt-5-nano",
                options={"instruction_dir": "general"},
            ),
        ),
    )

    assert registry.updater_tasks.world_game_updater.config.instruction_dir == "w"
    assert registry.updater_tasks.goal_game_updater is None
    assert registry.updater_tasks.agent_game_updater.config.instruction_dir == "x"
    assert (
        registry.updater_tasks.general_updater.config.instruction_dir == "general"
    )


def test_shell_model_registry_rejects_unknown_updater_backend() -> None:
    with pytest.raises(ValueError, match="unknown updater backend"):
        shell._build_model_registry(
            agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
            updater_config=UpdaterRuntimeConfig(
                world=ModelRoleConfig(backend="mystery"),
                goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                agent=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            ),
        )


def test_shell_model_registry_rejects_real_updater_without_model() -> None:
    with pytest.raises(ValueError, match="models.updater.world.model"):
        shell._build_model_registry(
            agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
            updater_config=UpdaterRuntimeConfig(
                world=ModelRoleConfig(backend="openai"),
                goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                agent=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            ),
        )


def test_shell_model_registry_rejects_real_agent_updater_without_model() -> None:
    with pytest.raises(ValueError, match="models.updater.agent.model"):
        shell._build_model_registry(
            agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
            updater_config=UpdaterRuntimeConfig(
                world=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                agent=ModelRoleConfig(backend="openai"),
                general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            ),
        )


def test_shell_model_registry_ignores_real_goal_updater_config() -> None:
    registry = shell._build_model_registry(
        agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
        updater_config=UpdaterRuntimeConfig(
            world=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            goal=ModelRoleConfig(backend="ollama", model="gemma4:e4b"),
            agent=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        ),
    )

    assert registry.updater_tasks.world_game_updater.provider.backend == "openai"
    assert registry.updater_tasks.goal_game_updater is None
    assert registry.updater_tasks.agent_game_updater.provider.backend == "openai"
    assert registry.updater_tasks.general_updater.provider.backend == "openai"


def test_shell_model_registry_wires_real_agent_updater() -> None:
    registry = shell._build_model_registry(
        agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
        updater_config=UpdaterRuntimeConfig(
            world=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            agent=ModelRoleConfig(backend="ollama", model="gemma4:e4b"),
            general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        ),
    )

    assert registry.updater_tasks.agent_game_updater.provider.backend == "ollama"


def test_shell_model_registry_wires_real_general_updater() -> None:
    registry = shell._build_model_registry(
        agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
        updater_config=UpdaterRuntimeConfig(
            world=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            agent=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            general=ModelRoleConfig(backend="ollama", model="gemma4:e4b"),
        ),
    )

    assert registry.updater_tasks.general_updater.provider.backend == "ollama"


def test_shell_model_registry_applies_shared_vlm_to_local_roles() -> None:
    registry = shell._build_model_registry(
        shared_vlm_config=ModelRoleConfig(
            backend="ollama",
            model="qwen3.6",
            options={
                "keep_alive": "30m",
                "temperature": 0,
                "num_ctx": 8192,
            },
        ),
        agent_config=ModelRoleConfig(
            backend="ollama",
            max_tool_calls=1,
            options={
                "input_image_size": "1024x1024",
                "options": {"num_predict": 800},
            },
        ),
        world_config=ModelRoleConfig(
            backend="ollama",
            options={
                "input_image_size": "1024x1024",
                "num_predict": 500,
            },
        ),
        goal_config=ModelRoleConfig(
            backend="ollama",
            options={
                "num_predict": 500,
            },
        ),
        updater_config=UpdaterRuntimeConfig(
            world=ModelRoleConfig(
                backend="ollama",
                options={
                    "input_image_size": "1024x1024",
                    "num_predict": 1200,
                },
            ),
            goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            agent=ModelRoleConfig(
                backend="ollama",
                options={
                    "options": {"num_predict": 1200},
                },
            ),
            general=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        ),
    )

    assert isinstance(registry.orchestrator_agent, OllamaOrchestratorAgentAdapter)
    assert registry.orchestrator_agent.provider.model == "qwen3.6"
    assert registry.orchestrator_agent.provider.config.input_image_size == "1024x1024"
    assert registry.orchestrator_agent.provider.config.options == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 800,
    }
    assert isinstance(registry.world_prediction_model, WorldPredictionAdapter)
    assert registry.world_prediction_model.config.model == "qwen3.6"
    assert registry.world_prediction_model.config.keep_alive == "30m"
    assert registry.world_prediction_model.config.options == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 500,
    }
    assert registry.world_prediction_model.config.input_image_size == "1024x1024"
    assert registry.goal_prediction_model is None
    assert registry.updater_tasks.world_game_updater.provider.model == "qwen3.6"
    assert (
        registry.updater_tasks.world_game_updater.provider.config.input_image_size
        == "1024x1024"
    )
    assert registry.updater_tasks.world_game_updater.provider.config.options == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 1200,
    }
    assert registry.updater_tasks.agent_game_updater.provider.model == "qwen3.6"
    assert registry.updater_tasks.agent_game_updater.provider.config.options == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 1200,
    }


def test_shell_model_registry_applies_shared_vlm_to_vllm_roles() -> None:
    registry = shell._build_model_registry(
        shared_vlm_config=ModelRoleConfig(
            backend="vllm",
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            options={
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key": "EMPTY",
                "input_image_size": "64x64",
                "temperature": 0,
                "max_tokens": 900,
                "server": {"port": 8000},
            },
        ),
        agent_config=ModelRoleConfig(
            backend="vllm",
            max_tool_calls=0,
            options={"temperature": 0.2, "max_tokens": 256},
        ),
        world_config=ModelRoleConfig(backend="vllm"),
        goal_config=ModelRoleConfig(backend="vllm"),
        updater_config=UpdaterRuntimeConfig(
            world=ModelRoleConfig(backend="vllm"),
            goal=ModelRoleConfig(backend="vllm"),
            agent=ModelRoleConfig(backend="vllm", options={"max_tokens": 1200}),
            general=ModelRoleConfig(backend="vllm"),
        ),
    )

    assert isinstance(registry.orchestrator_agent, VLLMOrchestratorAgentAdapter)
    assert registry.orchestrator_agent.provider.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert registry.orchestrator_agent.provider.config.base_url == (
        "http://127.0.0.1:8000/v1"
    )
    assert registry.orchestrator_agent.provider.config.input_image_size == "64x64"
    assert registry.orchestrator_agent.provider.config.temperature == 0.2
    assert registry.orchestrator_agent.provider.config.max_tokens == 256
    assert "server" not in registry.orchestrator_agent.provider.config.options
    assert registry.world_prediction_model.config.model == (
        "Qwen/Qwen3.6-35B-A3B-FP8"
    )
    assert registry.world_prediction_model.config.temperature == 0
    assert registry.world_prediction_model.config.input_image_size == "64x64"
    assert registry.goal_prediction_model is None
    assert registry.updater_tasks.world_game_updater.provider.backend == "vllm"
    assert registry.updater_tasks.world_game_updater.provider.model == (
        "Qwen/Qwen3.6-35B-A3B-FP8"
    )
    assert registry.updater_tasks.goal_game_updater is None
    assert registry.updater_tasks.agent_game_updater.provider.config.max_tokens == 1200


def test_shell_model_registry_rejects_vllm_roles_without_model() -> None:
    with pytest.raises(ValueError, match="models.agent.model"):
        shell._build_model_registry(
            agent_config=ModelRoleConfig(backend="vllm"),
            world_config=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            goal_config=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
            updater_config=_openai_updater_config(),
        )


def test_shell_model_registry_rejects_real_general_updater_without_model() -> None:
    with pytest.raises(ValueError, match="models.updater.general.model"):
        shell._build_model_registry(
            agent_config=_openai_agent_config(),
            world_config=_openai_tool_config(),
            goal_config=_openai_tool_config(),
            updater_config=UpdaterRuntimeConfig(
                world=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                goal=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                agent=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
                general=ModelRoleConfig(backend="ollama"),
            ),
        )


def test_shell_model_registry_wires_ollama_agent_with_explicit_tools() -> None:
    registry = shell._build_model_registry(
        agent_config=ModelRoleConfig(
            backend="ollama",
            model="gemma4:e4b",
            max_tool_calls=2,
            repair_attempts=1,
        ),
        world_config=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        goal_config=ModelRoleConfig(backend="openai", model="gpt-5-nano"),
        updater_config=_openai_updater_config(),
    )

    assert isinstance(registry.orchestrator_agent, OllamaOrchestratorAgentAdapter)
    assert isinstance(registry.world_prediction_model, WorldPredictionAdapter)
    assert registry.goal_prediction_model is None
