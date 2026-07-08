"""Configuration for the Goal role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMGoalConfig(VLLMRoleConfig):
    """vLLM-backed Goal role config."""

    pass
