"""vLLM adapter for the Goal role."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_of_agi.contracts import GoalPrediction
from face_of_agi.models.goal.config import HFGoalConfig, VLLMGoalConfig
from face_of_agi.models.goal.contracts import (
    GoalPredictionInput,
    goal_output_json_schema,
)
from face_of_agi.models.hf_roles import (
    HFJsonRoleClient,
    bounded_float as hf_bounded_float,
    non_negative_int as hf_non_negative_int,
    observation_image as hf_observation_image,
    parse_json_object as hf_parse_json_object,
)
from face_of_agi.models.vllm_roles import (
    VLLMJsonRoleClient,
    bounded_float,
    non_negative_int,
    observation_image,
    parse_json_object,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class VLLMGoalAdapter:
    """Goal role backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMGoalConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = VLLMJsonRoleClient(
            config=config,
            call_slot="goal",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            client=client,
        )

    def predict_goal(self, prediction_input: GoalPredictionInput) -> GoalPrediction:
        """Predict current goal and remaining steps from memory."""

        text = self.provider.complete_json(
            prompt_text=_goal_prompt(prediction_input),
            output_schema=goal_output_json_schema(),
            schema_name="goal_prediction",
            images=(observation_image(self.config, prediction_input.current_observation),),
        )
        payload = parse_json_object(text, label="goal")
        goal = str(payload.get("goal") or "").strip()
        if not goal:
            raise RuntimeError("goal response requires non-empty goal")
        raw_subgoals = payload.get("subgoals")
        if not isinstance(raw_subgoals, list) or any(
            not isinstance(item, str) for item in raw_subgoals
        ):
            raise RuntimeError("goal response requires string list subgoals")
        return GoalPrediction(
            goal=goal,
            subgoals=tuple(item.strip() for item in raw_subgoals if item.strip()),
            steps_remaining=non_negative_int(
                payload.get("steps_remaining"),
                label="steps_remaining",
            ),
            confidence=bounded_float(
                payload.get("confidence"),
                label="confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            metadata={
                "backend": "vllm",
                "model": self.config.model,
                "usage": self.provider.last_usage,
            },
        )


class HFGoalAdapter:
    """Goal role backed by the shared HF/Transformers VLM."""

    def __init__(
        self,
        config: HFGoalConfig,
        *,
        engine: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = HFJsonRoleClient(
            config=config,
            call_slot="goal",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            engine=engine,
        )

    def predict_goal(self, prediction_input: GoalPredictionInput) -> GoalPrediction:
        """Predict current goal and remaining steps from memory."""

        text = self.provider.complete_json(
            prompt_text=_goal_prompt(prediction_input),
            output_schema=goal_output_json_schema(),
            schema_name="goal_prediction",
            images=(hf_observation_image(self.config, prediction_input.current_observation),),
        )
        payload = hf_parse_json_object(text, label="goal")
        goal = str(payload.get("goal") or "").strip()
        if not goal:
            raise RuntimeError("goal response requires non-empty goal")
        raw_subgoals = payload.get("subgoals")
        if not isinstance(raw_subgoals, list) or any(
            not isinstance(item, str) for item in raw_subgoals
        ):
            raise RuntimeError("goal response requires string list subgoals")
        return GoalPrediction(
            goal=goal,
            subgoals=tuple(item.strip() for item in raw_subgoals if item.strip()),
            steps_remaining=hf_non_negative_int(
                payload.get("steps_remaining"),
                label="steps_remaining",
            ),
            confidence=hf_bounded_float(
                payload.get("confidence"),
                label="confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            metadata={
                "backend": "hf_transformers",
                "model": self.provider.model,
                "usage": self.provider.last_usage,
            },
        )


def _goal_prompt(prediction_input: GoalPredictionInput) -> str:
    previous = prediction_input.previous_goal
    previous_text = "none"
    if previous is not None:
        previous_text = (
            f"goal={previous.goal}\n"
            f"subgoals={list(previous.subgoals)}\n"
            f"steps_remaining={previous.steps_remaining}\n"
            f"confidence={previous.confidence}"
        )
    return "\n\n".join(
        [
            f"run_id: {prediction_input.run_id}",
            f"game_id: {prediction_input.game_id}",
            "Attached image: current frame.",
            "Current Memory document:",
            prediction_input.memory.document,
            "Previous Goal prediction:",
            previous_text,
        ]
    )
