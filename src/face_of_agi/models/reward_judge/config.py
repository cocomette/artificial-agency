"""Configuration for the Reward Judge role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.hf_roles import HFRoleConfig
from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMRewardJudgeConfig(VLLMRoleConfig):
    """vLLM-backed Reward Judge role config."""

    pass


@dataclass(slots=True)
class HFRewardJudgeConfig(HFRoleConfig):
    """HF/Transformers-backed Reward Judge role config."""

    pass
