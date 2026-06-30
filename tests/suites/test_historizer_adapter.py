"""Tests for the agent context historizer adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest

from face_of_agi.models.historizer import (
    AGENT_CONTEXT_HISTORY_KEYS,
    HistorizerOutputError,
    PromptHistorizerRequest,
    agent_context_history_json_schema,
    parse_agent_context_history_output,
)
from face_of_agi.models.historizer.config import VLLMHistorizerConfig
from face_of_agi.models.historizer.providers.vllm import VLLMHistorizerProvider


class FakeCompletions:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.chat = type(
            "FakeChat",
            (),
            {"completions": FakeCompletions(responses)},
        )()


def _chat_response(content: str) -> dict[str, Any]:
    return {
        "id": "resp-historizer",
        "model": "fake-vllm",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _valid_history_json(**overrides: str) -> str:
    fields = {key: f"value {key}" for key in AGENT_CONTEXT_HISTORY_KEYS}
    fields.update(overrides)
    return json.dumps({"field_evolution": fields})


def test_historizer_schema_includes_configured_max_lengths() -> None:
    schema = agent_context_history_json_schema(field_evolution_max_chars=123)

    fields = schema["properties"]["field_evolution"]["properties"]
    assert fields["goals"]["maxLength"] == 123
    assert fields["extras"]["maxLength"] == 123


def test_historizer_parser_rejects_oversized_field() -> None:
    text = _valid_history_json(goals="abcdef")

    with pytest.raises(HistorizerOutputError, match="too long"):
        parse_agent_context_history_output(text, field_evolution_max_chars=5)


def test_vllm_historizer_provider_clips_invalid_output_in_repair_prompt() -> None:
    client = FakeClient([_chat_response(_valid_history_json())])
    provider = VLLMHistorizerProvider(
        VLLMHistorizerConfig(
            model="fake-vllm",
            repair_invalid_output_preview_chars=80,
        ),
        client=client,
    )
    request = PromptHistorizerRequest(
        instructions="instructions",
        text="input",
        output_schema=agent_context_history_json_schema(),
    )
    invalid_text = "a" * 120 + "TAIL"

    provider.repair_context_history(
        request,
        invalid_text=invalid_text,
        validation_error="bad",
        attempt=1,
    )

    repair_text = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "Invalid output preview:" in repair_text
    assert "omitted" in repair_text
    assert "TAIL" in repair_text
    assert invalid_text not in repair_text
