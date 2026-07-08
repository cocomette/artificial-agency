"""Frame-unrolled game-loop state machine owned by orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import sys
from typing import Any, TextIO

from arcengine import GameState

from face_of_agi.contracts import (
    ActionSpec,
    ContextDocuments,
    FrameControlMode,
    FrameTurnContext,
    GameRunResult,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RewardUpdateQuantities,
    RuntimeConfig,
    ToolResult,
    UpdaterFrameTransitionInput,
)
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import StateMemory
from face_of_agi.models.adapters import OrchestratorAgentModel
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import (
    AgentContextUpdateInput,
    ToolContextUpdateInput,
    UpdaterModel,
)
from face_of_agi.orchestration.game_loop.post_decision_predictions import (
    PostDecisionPredictionRunner,
)

AgentToolRuntimeFactory = Callable[
    [str, str, int, FrameTurnContext],
    AgentToolRuntime,
]


class GameLoopStateMachine:
    """Run one ARC game through the target frame-turn state machine.

    This component owns the game-loop mechanics. The top-level `Orchestrator`
    remains the coordinator that wires dependencies and invokes this component.
    """

    def __init__(
        self,
        *,
        state_memory: StateMemory | None,
        contexts: ContextDocuments,
        agent: OrchestratorAgentModel,
        updater: UpdaterModel,
        post_decision_prediction_runner: PostDecisionPredictionRunner,
        tool_runtime_factory: AgentToolRuntimeFactory | None = None,
        trace_output: TextIO | None = None,
    ) -> None:
        self.state_memory = state_memory
        self.contexts = contexts
        self.agent = agent
        self.updater = updater
        self.post_decision_prediction_runner = post_decision_prediction_runner
        self.tool_runtime_factory = tool_runtime_factory
        self.trace_output = trace_output or sys.stdout

    def run(
        self,
        *,
        config: RuntimeConfig,
        environment: EnvironmentAdapter,
        environment_config: EnvironmentConfig,
    ) -> GameRunResult:
        """Run one selected ARC game until a terminal loop condition."""

        if environment_config.game_id is None:
            raise RuntimeError("environment config is missing the resolved game_id")

        selected_game_id = environment.select_game_by_id(environment_config.game_id)
        self._hydrate_contexts_from_latest_state(selected_game_id)
        observation = environment.reset()
        remaining_actions = environment_config.max_actions_per_level
        real_step_count = 0
        frame_turn_count = 0
        completed_levels = 0
        last_completed_levels = 0
        first_observation: Observation | None = None
        first_observation_ref: ObservationRef | None = None
        last_decision = None
        state_record_ids: list[int] = []
        persisted_observation_ids: set[str] = set()

        while True:
            info = environment.get_info()
            state = info.state

            if state == GameState.WIN:
                return GameRunResult(
                    run_id=config.run_id,
                    game_id=selected_game_id,
                    initial_observation_ref=first_observation_ref,
                    decision=last_decision,
                    state_record_ids=tuple(state_record_ids),
                    stop_reason="game_end",
                    step_count=real_step_count,
                    completed_levels=info.levels_completed,
                    last_state=state,
                )

            if state == GameState.GAME_OVER:
                observation = environment.reset()
                remaining_actions = environment_config.max_actions_per_level
                reset_info = environment.get_info()
                last_completed_levels = reset_info.levels_completed
                continue

            if info.levels_completed > last_completed_levels:
                completed_levels = info.levels_completed
                last_completed_levels = info.levels_completed
                remaining_actions = environment_config.max_actions_per_level

            if remaining_actions <= 0:
                return GameRunResult(
                    run_id=config.run_id,
                    game_id=selected_game_id,
                    initial_observation_ref=first_observation_ref,
                    decision=last_decision,
                    state_record_ids=tuple(state_record_ids),
                    stop_reason="action_limit_reached",
                    step_count=real_step_count,
                    completed_levels=completed_levels,
                    last_state=state,
                )

            real_actions = tuple(info.available_actions) or tuple(
                environment.get_action_space()
            )
            frame_buffer = self._unroll_observation(observation)

            for frame_index, current_observation in enumerate(frame_buffer):
                is_final_frame = frame_index == len(frame_buffer) - 1
                control_mode = (
                    FrameControlMode.real_environment_turn(real_actions)
                    if is_final_frame
                    else FrameControlMode.animation_unroll()
                )
                current_ref = self._persist_observation_once(
                    observation=current_observation,
                    persisted_observation_ids=persisted_observation_ids,
                )

                if first_observation is None:
                    first_observation = current_observation
                    first_observation_ref = current_ref

                frame_context = FrameTurnContext(
                    run_id=config.run_id,
                    game_id=selected_game_id,
                    first_observation_ref=first_observation_ref,
                    current_observation_ref=current_ref,
                    current_observation=current_observation,
                    frame_index=frame_index,
                    frame_count=len(frame_buffer),
                    control_mode=control_mode,
                )
                turn_id = frame_turn_count + 1
                tool_runtime = self._build_tool_runtime(
                    run_id=config.run_id,
                    game_id=selected_game_id,
                    turn_id=turn_id,
                    frame_context=frame_context,
                )
                decision = self.agent.decide(
                    context=self.contexts.agent,
                    first_observation=first_observation,
                    current_observation=current_observation,
                    action_space=control_mode.allowed_actions,
                    tool_runtime=tool_runtime,
                )
                last_decision = decision
                frame_turn_count = turn_id

                self._validate_decision(
                    decision.final_action,
                    control_mode=control_mode,
                )
                self._write_frame_trace(
                    frame_turn=frame_turn_count,
                    frame_context=frame_context,
                    action=decision.final_action,
                )

                if control_mode.controllable:
                    post_decision_predictions = self._run_post_decision_predictions(
                        current_ref=current_ref,
                        current_observation=current_observation,
                        final_action=decision.final_action,
                    )
                    real_step_count += 1
                    next_observation = environment.step(decision.final_action)
                    remaining_actions -= 1
                    next_frame = self._unroll_observation(next_observation)[0]
                    next_ref = self._persist_observation_once(
                        observation=next_frame,
                        persisted_observation_ids=persisted_observation_ids,
                    )
                    update_input = UpdaterFrameTransitionInput(
                        current_observation_ref=current_ref,
                        actual_next_observation_ref=next_ref,
                        decision_trace=decision.trace,
                        post_decision_predictions=post_decision_predictions,
                        submitted_action=decision.final_action,
                        metadata={"shell": "noop", "controllable": True},
                    )
                    self._apply_context_updates(update_input)
                    self._persist_turn_shell(
                        run_id=config.run_id,
                        game_id=selected_game_id,
                        frame_context=frame_context,
                        decision=decision,
                        update_input=update_input,
                        state_record_ids=state_record_ids,
                    )
                    observation = next_observation
                    break

                next_frame = frame_buffer[frame_index + 1]
                next_ref = self._persist_observation_once(
                    observation=next_frame,
                    persisted_observation_ids=persisted_observation_ids,
                )
                update_input = UpdaterFrameTransitionInput(
                    current_observation_ref=current_ref,
                    actual_next_observation_ref=next_ref,
                    decision_trace=decision.trace,
                    synthetic_none_action=decision.final_action,
                    metadata={"shell": "noop", "controllable": False},
                )
                self._apply_context_updates(update_input)
                self._persist_turn_shell(
                    run_id=config.run_id,
                    game_id=selected_game_id,
                    frame_context=frame_context,
                    decision=decision,
                    update_input=update_input,
                    state_record_ids=state_record_ids,
                )

    def _run_post_decision_predictions(
        self,
        *,
        current_ref: ObservationRef,
        current_observation: Observation,
        final_action: ActionSpec,
    ) -> PostDecisionPredictions:
        """Run committed S/G predictions after X chooses a real action."""

        return self.post_decision_prediction_runner.predict(
            current_observation_ref=current_ref,
            current_observation=current_observation,
            final_action=final_action,
            world_context=self.contexts.world,
            goal_context=self.contexts.goal,
        )

    def _unroll_observation(self, observation: Observation) -> tuple[Observation, ...]:
        frames = observation.frames
        if not frames:
            frames = (observation.frame,)

        if len(frames) == 1:
            return (
                Observation(
                    id=observation.id,
                    step=observation.step,
                    frame=frames[0],
                    frames=(frames[0],),
                    raw_frame_data=observation.raw_frame_data,
                    metadata={
                        **observation.metadata,
                        "bundle_observation_id": observation.id,
                        "frame_index": 0,
                        "frame_count": 1,
                    },
                ),
            )

        return tuple(
            Observation(
                id=f"{observation.id}-frame-{index}",
                step=observation.step,
                frame=frame,
                frames=(frame,),
                raw_frame_data=observation.raw_frame_data,
                metadata={
                    **observation.metadata,
                    "bundle_observation_id": observation.id,
                    "frame_index": index,
                    "frame_count": len(frames),
                },
            )
            for index, frame in enumerate(frames)
        )

    def _persist_observation_once(
        self,
        *,
        observation: Observation,
        persisted_observation_ids: set[str],
    ) -> ObservationRef:
        ref = ObservationRef(memory="state", id=observation.id)
        if self.state_memory is None or observation.id in persisted_observation_ids:
            return ref

        persisted_observation_ids.add(observation.id)
        return ref

    def _validate_decision(
        self,
        action: ActionSpec,
        *,
        control_mode: FrameControlMode,
    ) -> None:
        if not control_mode.controllable:
            if not action.is_none():
                raise RuntimeError(
                    "non-final unrolled frame requires synthetic NONE action"
                )
            return

        if action.is_none():
            raise RuntimeError("final controllable frame cannot submit synthetic NONE")

        is_allowed = any(
            candidate.action_id == action.action_id
            for candidate in control_mode.allowed_actions
        )
        if not is_allowed:
            raise RuntimeError(
                f"X returned invalid action for current frame: {action.name}"
            )

    def _build_tool_runtime(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        frame_context: FrameTurnContext,
    ) -> AgentToolRuntime | None:
        if self.tool_runtime_factory is None:
            return None
        return self.tool_runtime_factory(
            run_id,
            game_id,
            turn_id,
            frame_context,
        )

    def _persist_turn_shell(
        self,
        *,
        run_id: str,
        game_id: str,
        frame_context: FrameTurnContext,
        decision: Any,
        update_input: UpdaterFrameTransitionInput,
        state_record_ids: list[int],
    ) -> None:
        if self.state_memory is None:
            return

        state = self.state_memory.write_state(
            run_id=run_id,
            game_id=game_id,
            step=frame_context.current_observation.step,
            frame_index=frame_context.frame_index,
            frame_count=frame_context.frame_count,
            current_observation=frame_context.current_observation,
            chosen_action=decision.final_action,
            contexts=self.contexts,
            agent_trace=decision.trace,
            post_decision_predictions=update_input.post_decision_predictions,
            metadata={
                "control_mode": asdict(frame_context.control_mode),
                "update_input": asdict(update_input),
            },
        )
        state_record_ids.append(state.id)

    def _apply_context_updates(
        self,
        update_input: UpdaterFrameTransitionInput,
    ) -> None:
        """Apply updater P to the live working contexts before persistence."""

        quantities = RewardUpdateQuantities()
        common_kwargs = {
            "current_observation_ref": update_input.current_observation_ref,
            "actual_next_observation_ref": update_input.actual_next_observation_ref,
            "post_decision_predictions": update_input.post_decision_predictions,
            "quantities": quantities,
            "submitted_action": update_input.submitted_action,
            "synthetic_none_action": update_input.synthetic_none_action,
            "metadata": dict(update_input.metadata),
        }

        self.contexts.world = self.updater.update_tool_context(
            ToolContextUpdateInput(
                role="world",
                previous_context=self.contexts.world,
                tool_results=self._tool_results_for_role(update_input, "world"),
                **common_kwargs,
            )
        )
        self.contexts.goal = self.updater.update_tool_context(
            ToolContextUpdateInput(
                role="goal",
                previous_context=self.contexts.goal,
                tool_results=self._tool_results_for_role(update_input, "goal"),
                **common_kwargs,
            )
        )
        self.contexts.agent = self.updater.update_agent_context(
            AgentContextUpdateInput(
                previous_context=self.contexts.agent,
                trace=update_input.decision_trace,
                **common_kwargs,
            )
        )

    def _tool_results_for_role(
        self,
        update_input: UpdaterFrameTransitionInput,
        role: str,
    ) -> tuple[ToolResult, ...]:
        """Return live trace tool results for one updater role."""

        return tuple(
            result
            for result in update_input.decision_trace.tool_results
            if result.tool == role
        )

    def _hydrate_contexts_from_latest_state(self, game_id: str) -> None:
        """Use the previous M state contexts when this game has run before."""

        if self.state_memory is None:
            return

        latest_state = self.state_memory.read_latest_state(game_id)
        if latest_state is None:
            return

        self.contexts.world = latest_state.world_context
        self.contexts.goal = latest_state.goal_context
        self.contexts.agent = latest_state.agent_context

    def _write_frame_trace(
        self,
        *,
        frame_turn: int,
        frame_context: FrameTurnContext,
        action: ActionSpec,
    ) -> None:
        controllable = "yes" if frame_context.control_mode.controllable else "no"
        print(
            "frame turn"
            f" {frame_turn}: env_step={frame_context.current_observation.step}"
            f" frame={frame_context.frame_index + 1}/{frame_context.frame_count}"
            f" controllable={controllable}",
            file=self.trace_output,
        )
        if action.is_none():
            print(
                "action: X returned NONE; environment not stepped",
                file=self.trace_output,
            )
        else:
            print(
                f"action: X selected {self._format_action(action)}",
                file=self.trace_output,
            )

    def _format_action(self, action: ActionSpec) -> str:
        if action.data:
            return f"{action.name} {action.data}"
        return action.name
