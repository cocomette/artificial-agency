"""HF/Transformers provider adapter for orchestrator agent X."""

from __future__ import annotations

from typing import Any

from face_of_agi.models.orchestrator_agent.adapter import OrchestratorAgentAdapter
from face_of_agi.models.orchestrator_agent.config import HFOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
    VLLMOrchestratorAgentProvider,
)
from face_of_agi.models.providers.hf_transformers import HFChatClient


class HFOrchestratorAgentAdapter(VLLMOrchestratorAgentAdapter):
    """Agent X adapter backed by the shared HF/Transformers VLM."""

    def __init__(
        self,
        config: HFOrchestratorAgentConfig,
        *,
        engine: Any | None = None,
    ) -> None:
        if not config.model and not config.model_path:
            raise ValueError("HF Agent X requires an explicit model or model_path")
        if config.max_tool_calls > 0:
            raise ValueError("HF Agent X does not support tool calls")
        self.provider = HFOrchestratorAgentProvider(config, engine=engine)
        OrchestratorAgentAdapter.__init__(
            self,
            provider=self.provider,
            config=config,
        )


class HFOrchestratorAgentProvider(VLLMOrchestratorAgentProvider):
    """Thin HF transport layer for the shared Agent X loop."""

    backend = "hf_transformers"

    def __init__(
        self,
        config: HFOrchestratorAgentConfig,
        *,
        engine: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model_path or config.model
        self._client = HFChatClient(config, engine=engine)
        self.instructions = ""
        self.messages: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None
        self.last_response_id: str | None = None
        self.last_usage: Any | None = None
        self.active_lora_adapter_name: str | None = None

    def activate_lora_adapter(self, adapter_name: str) -> None:
        """Use a loaded HF LoRA adapter for future provider calls."""

        self.active_lora_adapter_name = adapter_name

    def _request_model(self) -> str | None:
        return self.active_lora_adapter_name or self.model


__all__ = [
    "HFOrchestratorAgentAdapter",
    "HFOrchestratorAgentProvider",
]
