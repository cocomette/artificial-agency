"""Configuration for the Interest role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.hf_roles import HFRoleConfig
from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMInterestConfig(VLLMRoleConfig):
    """vLLM-backed Interest role config."""


@dataclass(slots=True)
class HFInterestConfig(HFRoleConfig):
    """HF/Transformers-backed Interest role config."""
