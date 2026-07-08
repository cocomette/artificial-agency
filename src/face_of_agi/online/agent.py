"""Online learner agent coordinating backbone, replay, and planning."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    DecisionResult,
    FrameTurnContext,
    LearnerTurnTrace,
    ObservationRef,
    ReplayStats,
    TransitionRecord,
)
from face_of_agi.environment.config import AgentRuntimeConfig
from face_of_agi.online.backbone import EncodedObservation, FrozenBackbone
from face_of_agi.online.learning import (
    EncodedTransition,
    OnlineWorldModel,
    ReplayTrainer,
    TransitionBuffer,
    ValueModel,
    transition_priority,
)
from face_of_agi.online.planner import ShortHorizonPlanner


class OnlineLearnerAgent:
    """Frozen-backbone online agent with bounded replay."""

    def __init__(
        self,
        *,
        config: AgentRuntimeConfig,
        backbone: FrozenBackbone,
    ) -> None:
        self.config = config
        self.backbone = backbone
        self.buffer = TransitionBuffer(config.online.buffer_size)
        self.world_model = OnlineWorldModel(config.online)
        self.value_model = ValueModel(config.online.learning_rate)
        self.replay = ReplayTrainer(
            config=config.replay,
            buffer=self.buffer,
            world_model=self.world_model,
            value_model=self.value_model,
        )
        self.planner = ShortHorizonPlanner(
            config=config.planner,
            world_model=self.world_model,
            value_model=self.value_model,
        )
        self.real_transition_count = 0
        self.frame_turn_count = 0
        self._last_encoded: EncodedObservation | None = None
        self._last_decision: DecisionResult | None = None

    def decide(
        self,
        frame_context: FrameTurnContext,
    ) -> tuple[DecisionResult, tuple[Any, ...], dict[str, Any]]:
        """Choose one action for a controllable frame."""

        if not frame_context.control_mode.controllable:
            raise RuntimeError("OnlineLearnerAgent.decide requires a controllable frame")
        encoded = self.backbone.encode(frame_context.current_observation)
        result = self.planner.choose(
            features=encoded.features,
            action_space=frame_context.control_mode.allowed_actions,
            real_turn_index=self.real_transition_count,
        )
        action = result.action
        trace = AgentTrace(
            step=frame_context.current_observation.step,
            first_observation_ref=frame_context.first_observation_ref,
            current_observation_ref=frame_context.current_observation_ref,
            final_action=action,
            reasoning_summary="online learner planner selected highest-scoring action",
            metadata={
                "decision_source": "online_learner",
                "backbone": encoded.metadata,
                "planner_candidate_count": len(result.candidates),
                "selected_score": result.candidates[0].score,
            },
        )
        decision = DecisionResult(final_action=action, trace=trace)
        self._last_encoded = encoded
        self._last_decision = decision
        return decision, result.candidates, encoded.metadata

    def synthetic_none_decision(
        self,
        frame_context: FrameTurnContext,
    ) -> DecisionResult:
        """Return the internal no-control action for animation frames."""

        final_action = ActionSpec.none()
        trace = AgentTrace(
            step=frame_context.current_observation.step,
            first_observation_ref=frame_context.first_observation_ref,
            current_observation_ref=frame_context.current_observation_ref,
            final_action=final_action,
            reasoning_summary="non-controllable animation frame",
            metadata={
                "decision_source": "orchestration_synthetic_none",
                "online_agent_called": False,
            },
        )
        return DecisionResult(final_action=final_action, trace=trace)

    def observe_transition(
        self,
        *,
        frame_context: FrameTurnContext,
        decision: DecisionResult,
        transition: TransitionRecord,
        next_observation: Any,
        planner_candidates: tuple[Any, ...] = (),
        completed_level: bool,
    ) -> tuple[LearnerTurnTrace, dict[str, Any]]:
        """Ground the learner with one observed transition and run replay."""

        previous = self._encoded_for_current(frame_context, decision)
        next_encoded = self.backbone.encode(next_observation)
        prediction_error = self.world_model.prediction_error(
            previous.features,
            decision.final_action,
            next_encoded.features,
        )
        transition.prediction_error = prediction_error
        encoded_transition = EncodedTransition(
            id=_transition_id(frame_context, transition),
            previous=previous.features,
            action=decision.final_action,
            next=next_encoded.features,
            record=transition,
            priority=transition_priority(transition),
            metadata={
                "previous_backbone": previous.metadata,
                "next_backbone": next_encoded.metadata,
            },
        )
        self.buffer.add(encoded_transition)
        replay = self.replay.update_after_real_transition(
            encoded_transition,
            completed_level=completed_level,
        )
        if transition.controllable:
            self.real_transition_count += 1
        self.frame_turn_count += 1
        trace = LearnerTurnTrace(
            decision=decision,
            transition=transition,
            replay=replay,
            planner_candidates=planner_candidates,
            backbone_metadata={
                "previous": previous.metadata,
                "next": next_encoded.metadata,
            },
            learner_metadata={
                "buffer": self.buffer.summary(),
                "world_model": self.world_model.snapshot(),
                "value_model": self.value_model.snapshot(),
            },
        )
        return trace, self.snapshot()

    def complete_turn_trace(
        self,
        *,
        decision: DecisionResult,
        transition: TransitionRecord,
        replay: ReplayStats,
        planner_candidates: tuple[Any, ...],
        backbone_metadata: dict[str, Any],
    ) -> LearnerTurnTrace:
        """Build a trace when replay was handled outside the agent."""

        return LearnerTurnTrace(
            decision=decision,
            transition=transition,
            replay=replay,
            planner_candidates=planner_candidates,
            backbone_metadata=backbone_metadata,
            learner_metadata={
                "buffer": self.buffer.summary(),
                "world_model": self.world_model.snapshot(),
                "value_model": self.value_model.snapshot(),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a compact serializable learner snapshot."""

        return {
            "type": "online_learner_snapshot.v1",
            "real_transition_count": self.real_transition_count,
            "frame_turn_count": self.frame_turn_count,
            "buffer": self.buffer.summary(),
            "world_model": self.world_model.snapshot(),
            "value_model": self.value_model.snapshot(),
            "backbone": self.backbone.metadata(),
            "config": {
                "online": asdict(self.config.online),
                "replay": asdict(self.config.replay),
                "planner": asdict(self.config.planner),
            },
        }

    def _encoded_for_current(
        self,
        frame_context: FrameTurnContext,
        decision: DecisionResult,
    ) -> EncodedObservation:
        if (
            self._last_encoded is not None
            and self._last_decision is decision
            and frame_context.control_mode.controllable
        ):
            return self._last_encoded
        return self.backbone.encode(frame_context.current_observation)


def _transition_id(
    frame_context: FrameTurnContext,
    transition: TransitionRecord,
) -> str:
    return (
        f"{frame_context.run_id}:{frame_context.game_id}:"
        f"{frame_context.current_observation.step}:"
        f"{frame_context.frame_index}:{transition.action.name}"
    )
