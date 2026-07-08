"""vLLM adapter for the regenerated Memory role."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_of_agi.contracts import MemoryDocument
from face_of_agi.models.hf_roles import (
    HFJsonRoleClient,
    observation_image as hf_observation_image,
    parse_json_object as hf_parse_json_object,
)
from face_of_agi.models.memory.config import HFMemoryConfig, VLLMMemoryConfig
from face_of_agi.models.memory.contracts import (
    MemoryBuildInput,
    MemoryLedgerEntry,
    memory_output_json_schema,
)
from face_of_agi.models.vllm_roles import (
    VLLMJsonRoleClient,
    observation_image,
    parse_json_object,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class VLLMMemoryAdapter:
    """Memory role backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = VLLMJsonRoleClient(
            config=config,
            call_slot="memory",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            client=client,
        )

    def build_memory(self, build_input: MemoryBuildInput) -> MemoryDocument:
        """Regenerate memory from first/current frames and a sanitized ledger."""

        text = self.provider.complete_json(
            prompt_text=_memory_prompt(build_input),
            output_schema=memory_output_json_schema(),
            schema_name="memory_document",
            images=(
                observation_image(self.config, build_input.first_observation),
                observation_image(self.config, build_input.current_observation),
            ),
        )
        payload = parse_json_object(text, label="memory")
        document = str(payload.get("document") or "").strip()
        if not document:
            raise RuntimeError("memory response requires non-empty document")
        return MemoryDocument(
            document=document,
            metadata={
                "backend": "vllm",
                "model": self.config.model,
                "ledger_entry_count": len(build_input.ledger),
                "usage": self.provider.last_usage,
            },
        )


class HFMemoryAdapter:
    """Memory role backed by the shared HF/Transformers VLM."""

    def __init__(
        self,
        config: HFMemoryConfig,
        *,
        engine: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = HFJsonRoleClient(
            config=config,
            call_slot="memory",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            engine=engine,
        )

    def build_memory(self, build_input: MemoryBuildInput) -> MemoryDocument:
        """Regenerate memory from first/current frames and a sanitized ledger."""

        text = self.provider.complete_json(
            prompt_text=_memory_prompt(build_input),
            output_schema=memory_output_json_schema(),
            schema_name="memory_document",
            images=(
                hf_observation_image(self.config, build_input.first_observation),
                hf_observation_image(self.config, build_input.current_observation),
            ),
        )
        payload = hf_parse_json_object(text, label="memory")
        document = str(payload.get("document") or "").strip()
        if not document:
            raise RuntimeError("memory response requires non-empty document")
        return MemoryDocument(
            document=document,
            metadata={
                "backend": "hf_transformers",
                "model": self.provider.model,
                "ledger_entry_count": len(build_input.ledger),
                "usage": self.provider.last_usage,
            },
        )


def _memory_prompt(build_input: MemoryBuildInput) -> str:
    """Return the Memory role prompt."""

    return "\n\n".join(
        [
            f"run_id: {build_input.run_id}",
            f"game_id: {build_input.game_id}",
            "Attached images: first frame, current frame.",
            "Sanitized action/change ledger JSON:",
            _json_text([_ledger_entry_json(entry) for entry in build_input.ledger]),
            "Regenerate the complete Memory document from this ledger. "
            "Preserve knowledge across GAME_RESET ledger entries: old failed "
            "attempts and discovered mechanics remain useful, but distinguish "
            "them from the current post-reset state.",
            "The document must be comprehensive but compressed: explain what "
            "happened so far, your interpretation of game mechanics, current "
            "state, tried actions, dead ends, reset history, visible progress "
            "signals, and open hypotheses.",
        ]
    )


def _ledger_entry_json(entry: MemoryLedgerEntry) -> dict[str, Any]:
    return {
        "turn_id": entry.turn_id,
        "action": entry.action,
        "change_summary": entry.change_summary,
    }


def _json_text(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
