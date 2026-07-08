"""Configuration for the World role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMWorldConfig(VLLMRoleConfig):
    """vLLM-backed World role config."""
