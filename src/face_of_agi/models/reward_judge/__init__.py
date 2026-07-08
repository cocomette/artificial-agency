"""Reward Judge model role."""

from face_of_agi.models.reward_judge.adapter import VLLMRewardJudgeAdapter
from face_of_agi.models.reward_judge.config import VLLMRewardJudgeConfig
from face_of_agi.models.reward_judge.contracts import (
    RewardJudgeInput,
    RewardJudgeModel,
    reward_judge_json_schema,
)

__all__ = [
    "RewardJudgeInput",
    "RewardJudgeModel",
    "VLLMRewardJudgeAdapter",
    "VLLMRewardJudgeConfig",
    "reward_judge_json_schema",
]
