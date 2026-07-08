"""Configuration for the Goal role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.hf_roles import HFRoleConfig
from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMGoalConfig(VLLMRoleConfig):
    """vLLM-backed Goal role config."""

    pass


@dataclass(slots=True)
class HFGoalConfig(HFRoleConfig):
    """HF/Transformers-backed Goal role config."""

    pass
