"""Configuration for the World role."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.hf_roles import HFRoleConfig
from face_of_agi.models.vllm_roles import VLLMRoleConfig


@dataclass(slots=True)
class VLLMWorldConfig(VLLMRoleConfig):
    """vLLM-backed World role config."""

    lora_adapter_name: str | None = None


@dataclass(slots=True)
class HFWorldConfig(HFRoleConfig):
    """HF/Transformers-backed World role config."""

    lora_adapter_name: str | None = None
