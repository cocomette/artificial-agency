"""vLLM adapter for the Reward Judge role."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_of_agi.contracts import RewardJudgeScore
from face_of_agi.models.reward_judge.config import (
    VLLMRewardJudgeConfig,
)
from face_of_agi.models.reward_judge.contracts import (
    RewardJudgeInput,
    reward_judge_json_schema,
)
from face_of_agi.models.vllm_roles import (
    VLLMJsonRoleClient,
    action_text,
    bounded_float,
    observation_image,
    parse_json_object,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class VLLMRewardJudgeAdapter:
    """Reward Judge backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMRewardJudgeConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = VLLMJsonRoleClient(
            config=config,
            call_slot="reward_judge",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            client=client,
        )

    def judge_prediction(self, judge_input: RewardJudgeInput) -> RewardJudgeScore:
        """Score a world prediction against the observed change summary."""

        text = self.provider.complete_json(
            prompt_text=_judge_prompt(self.config, judge_input),
            output_schema=reward_judge_json_schema(),
            schema_name="reward_judge_score",
            images=(
                observation_image(self.config, judge_input.previous_observation),
                observation_image(self.config, judge_input.current_observation),
            ),
        )
        payload = parse_json_object(text, label="reward judge")
        raw_tags = payload.get("error_tags")
        if not isinstance(raw_tags, list) or any(
            not isinstance(item, str) for item in raw_tags
        ):
            raise RuntimeError("reward judge response requires string list error_tags")
        return RewardJudgeScore(
            score=bounded_float(
                payload.get("score"),
                label="score",
                minimum=0.0,
                maximum=1.0,
            ),
            notes=str(payload.get("notes") or "").strip(),
            error_tags=tuple(tag.strip() for tag in raw_tags if tag.strip()),
            metadata={
                "backend": "vllm",
                "model": self.config.model,
                "usage": self.provider.last_usage,
            },
        )


def _judge_prompt(
    config: VLLMRewardJudgeConfig,
    judge_input: RewardJudgeInput,
) -> str:
    return "\n\n".join(
        [
            f"run_id: {judge_input.run_id}",
            f"game_id: {judge_input.game_id}",
            f"turn_id: {judge_input.turn_id}",
            "Attached images: previous frame, observed current frame.",
            "Executed action:",
            action_text(
                judge_input.action,
                crop_edges=config.input_image_crop_arc_grid_edges,
            ),
            "World prediction:",
            judge_input.prediction.predicted_change,
            "Ground-truth Change Summary:",
            judge_input.change_summary,
        ]
    )
