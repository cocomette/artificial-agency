"""Central orchestration boundary."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, TextIO

from face_of_agi.contracts import (
    ActionSpec,
    ContextDocuments,
    ExperimentToolInvocationResult,
    FrameTurnContext,
    GameRunResult,
    Observation,
    ObservationRef,
    RuntimeConfig,
    ToolCall,
    ToolName,
    ToolResult,
)
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.frames import normalize_frame_for_memory
from face_of_agi.memory import ExperimentalMemory, StateMemory
from face_of_agi.models.adapters import ModelRegistry, OrchestratorAgentModel
from face_of_agi.models.orchestrator_agent.providers.random import (
    RandomOrchestratorAgentAdapter,
)
from face_of_agi.models.updater import UpdaterAdapter
from face_of_agi.orchestration.game_loop import (
    GameLoopStateMachine,
    PostDecisionPredictionRunner,
)
from face_of_agi.orchestration.tool_runtime import OrchestrationAgentToolRuntime
from face_of_agi.tools import ToolRouter


class Orchestrator:
    """Coordinate environment, memory, and model boundaries.

    Sub-orchestration components own concrete workflows such as the game-loop
    state machine. This class wires dependencies and keeps those workflows
    behind a single orchestration boundary.
    """

    def __init__(
        self,
        *,
        state_memory: StateMemory | None = None,
        experimental_memory: ExperimentalMemory | None = None,
        models: ModelRegistry | None = None,
        contexts: ContextDocuments | None = None,
        rng: random.Random | None = None,
        experimental_memory_turn_buffer: int = 2,
        prompt_model_calls_enabled: bool = False,
    ) -> None:
        self.state_memory = state_memory
        self.experimental_memory = experimental_memory
        self.rng = rng or random.Random()
        self.models = self._ensure_models(models)
        self.contexts = contexts or ContextDocuments()
        if experimental_memory_turn_buffer < 1:
            raise ValueError("experimental memory turn buffer must be at least 1")
        self.experimental_memory_turn_buffer = experimental_memory_turn_buffer
        self.prompt_model_calls_enabled = prompt_model_calls_enabled

    def run_environment_shell(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
        trace_output: TextIO | None = None,
    ) -> GameRunResult:
        """Run one ARC game through the dedicated game-loop component."""

        return GameLoopStateMachine(
            state_memory=self.state_memory,
            contexts=self.contexts,
            agent=self._require_orchestrator_agent(),
            updater=self.models.require_updater(),
            post_decision_prediction_runner=self._build_post_decision_prediction_runner(),
            tool_runtime_factory=self._build_agent_tool_runtime,
            trace_output=trace_output,
        ).run(
            config=config,
            environment=environment,
            environment_config=environment_config,
        )

    def cleanup_state_memory_keep_latest(self) -> None:
        """Prune dedicated M state rows after a normal run finishes."""

        if self.state_memory is None:
            return
        self.state_memory.cleanup_keep_latest_per_game()

    def invoke_tool_for_experiment(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        frame_context: FrameTurnContext,
        call: ToolCall,
        metadata: dict[str, Any] | None = None,
        state_observations: Mapping[str, Observation] | None = None,
    ) -> ExperimentToolInvocationResult:
        """Route one tool call and persist its output frame in E.

        This is the orchestration-owned interface future Agent X reasoning will
        use. It deliberately does not implement that reasoning loop yet.
        """

        if self.experimental_memory is None:
            raise RuntimeError("experimental memory is not configured")

        source_observation = self._resolve_observation_ref(
            game_id=game_id,
            ref=call.observation_ref,
            state_observations=state_observations,
        )
        result = self._route_tool_call(call=call, observation=source_observation)
        result.predicted_observation = normalize_frame_for_memory(
            result.predicted_observation
        )
        output_observation = self._observation_from_tool_result(
            result,
            step=frame_context.current_observation.step,
        )
        record = self.experimental_memory.write_experiment(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            tool_call=call,
            output_observation=output_observation,
            tool_result=result,
            metadata=metadata,
        )
        self.experimental_memory.cleanup_keep_latest_turns_per_game(
            run_id=run_id,
            game_id=game_id,
            max_turns=self.experimental_memory_turn_buffer,
        )
        return ExperimentToolInvocationResult(
            tool_result=result,
            observation_ref=ObservationRef(memory="experimental", id=str(record.id)),
            experiment_record=record,
        )

    def choose_random_action(
        self,
        *,
        observation: Observation,
        available_actions: list[ActionSpec] | tuple[ActionSpec, ...],
        lifecycle_signal: Any | None = None,
    ) -> ActionSpec:
        """Compatibility wrapper that delegates random selection to `X`."""

        if not available_actions:
            raise RuntimeError("environment did not provide any valid actions")

        del lifecycle_signal

        decision = self._require_orchestrator_agent().decide(
            context=self.contexts.agent,
            first_observation=observation,
            current_observation=observation,
            action_space=available_actions,
        )
        return decision.final_action

    def run_reset_only(
        self,
        *,
        run_id: str,
        game_id: str,
        environment: EnvironmentAdapter,
    ) -> GameRunResult:
        """Run the current barebones flow for one game without stepping it."""

        if (
            self.state_memory is None
            or self.experimental_memory is None
            or self.models is None
        ):
            raise RuntimeError("reset-only scaffolding requires memory and model wiring")

        observation = environment.reset()
        observation_ref = ObservationRef(memory="state", id=observation.id)

        agent = self.models.require_orchestrator_agent()
        decision = agent.decide(
            context=self.contexts.agent,
            first_observation=observation,
            current_observation=observation,
            action_space=environment.get_action_space(),
        )
        state_record = self.state_memory.write_state(
            run_id=run_id,
            game_id=game_id,
            step=observation.step,
            frame_index=0,
            frame_count=observation.frame_count(),
            current_observation=observation,
            chosen_action=decision.final_action,
            contexts=self.contexts,
            agent_trace=decision.trace,
            metadata={"shell": "reset_only"},
        )

        return GameRunResult(
            run_id=run_id,
            game_id=game_id,
            initial_observation_ref=observation_ref,
            decision=decision,
            state_record_ids=(state_record.id,),
        )

    def _ensure_models(self, models: ModelRegistry | None) -> ModelRegistry:
        registry = models or ModelRegistry()
        if registry.orchestrator_agent is None:
            registry.orchestrator_agent = RandomOrchestratorAgentAdapter(rng=self.rng)
        if registry.updater is None:
            registry.updater = UpdaterAdapter()
        return registry

    def _require_orchestrator_agent(self) -> OrchestratorAgentModel:
        if self.models is None:
            raise RuntimeError("orchestrator models were not configured")
        return self.models.require_orchestrator_agent()

    def _build_agent_tool_runtime(
        self,
        run_id: str,
        game_id: str,
        turn_id: int,
        frame_context: FrameTurnContext,
    ) -> OrchestrationAgentToolRuntime:
        """Build the controlled tool interface for one X decision turn."""

        return OrchestrationAgentToolRuntime(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            frame_context=frame_context,
            invoke_tool=self.invoke_tool_for_experiment,
            state_observations={
                frame_context.current_observation_ref.id: (
                    frame_context.current_observation
                ),
            },
            available_tool_names=self._available_tool_names(frame_context),
            tools_enabled=frame_context.control_mode.controllable,
        )

    def _build_post_decision_prediction_runner(
        self,
    ) -> PostDecisionPredictionRunner:
        """Build the orchestration-owned committed prediction runner."""

        return PostDecisionPredictionRunner(
            world_tool=self.models.world_tool,
            goal_tool=self.models.goal_tool,
            prompt_model_calls_enabled=self.prompt_model_calls_enabled,
        )

    def _available_tool_names(
        self,
        frame_context: FrameTurnContext,
    ) -> tuple[ToolName, ...]:
        """Return configured tools exposed to X on this frame."""

        if not frame_context.control_mode.controllable:
            return ()

        tools: list[ToolName] = []
        if self.models.world_tool is not None:
            tools.append("world")
        if self.models.goal_tool is not None:
            tools.append("goal")
        return tuple(tools)

    def _route_tool_call(
        self,
        *,
        call: ToolCall,
        observation: Observation,
    ) -> ToolResult:
        router = ToolRouter(
            world_tool=self.models.world_tool,
            goal_tool=self.models.goal_tool,
        )
        context = self.contexts.world if call.tool == "world" else self.contexts.goal
        return router.route(call=call, context=context, observation=observation)

    def _resolve_observation_ref(
        self,
        *,
        game_id: str,
        ref: ObservationRef,
        state_observations: Mapping[str, Observation] | None = None,
    ) -> Observation:
        if ref.memory == "experimental":
            if self.experimental_memory is None:
                raise RuntimeError("experimental memory is not configured")
            observation = self.experimental_memory.resolve_observation(ref)
            if observation is None:
                raise RuntimeError(f"unknown experimental observation ref: {ref.id}")
            return observation

        if self.state_memory is None:
            raise RuntimeError("state memory is not configured")

        if state_observations is not None and ref.id in state_observations:
            return state_observations[ref.id]

        for state in reversed(self.state_memory.list_states(game_id=game_id)):
            if str(state.current_observation.get("id", "")) == ref.id:
                return _observation_from_payload(state.current_observation)

        raise RuntimeError(f"unknown state observation ref: {ref.id}")

    def _observation_from_tool_result(
        self,
        result: ToolResult,
        *,
        step: int,
    ) -> Observation:
        return Observation(
            id=result.id,
            step=step,
            frame=result.predicted_observation,
            frames=(result.predicted_observation,),
            metadata={
                "tool": result.tool,
                "source_observation_ref": {
                    "memory": result.source_observation_ref.memory,
                    "id": result.source_observation_ref.id,
                },
                **result.metadata,
            },
        )


def _observation_from_payload(payload: dict[str, Any]) -> Observation:
    """Rehydrate the minimal observation shape stored in memory JSON."""

    frames = payload.get("frames") or ()
    return Observation(
        id=str(payload.get("id", "")),
        step=int(payload.get("step", 0)),
        frame=payload.get("frame"),
        frames=tuple(frames),
        metadata=dict(payload.get("metadata") or {}),
    )
