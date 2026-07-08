"""Configuration for the Memory role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.hf_roles import HFRoleConfig
from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMMemoryConfig(VLLMRoleConfig):
    """vLLM-backed Memory role config."""

    pass


@dataclass(slots=True)
class HFMemoryConfig(HFRoleConfig):
    """HF/Transformers-backed Memory role config."""

    pass
