"""Provider adapters for the agent context historizer role."""

from face_of_agi.models.historizer.providers.vllm import VLLMHistorizerAdapter

__all__ = [
    "VLLMHistorizerAdapter",
]
