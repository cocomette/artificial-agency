"""Tests for the shared Ollama final chat provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.providers.ollama import (
    OLLAMA_JSON_ASSISTANT_PREFILL,
    OllamaChatClient,
    assistant_json_prefill_message,
    message_content,
    response_usage,
)


@dataclass(slots=True)
class FakeOllamaConfig:
    model: str = "gemma4:e4b"
    host: str | None = None
    think: bool | str | None = False
    keep_alive: int | str | None = "5m"
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})


class FakeOllamaClient:
    def __init__(self, responses: list[dict[str, object]] | None = None) -> None:
        self.responses = responses or [
            {
                "message": {"content": "ok"},
                "prompt_eval_count": 4,
                "eval_count": 2,
            }
        ]
        self.calls: list[dict[str, object]] = []

    def chat(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def test_ollama_chat_client_builds_final_provider_request() -> None:
    client = FakeOllamaClient()
    adapter = OllamaChatClient(FakeOllamaConfig(), client=client)

    response = adapter.chat(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "world"}}],
    )

    request = client.calls[0]
    assert message_content(response) == "ok"
    assert request["model"] == "gemma4:e4b"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["tools"] == [
        {"type": "function", "function": {"name": "world"}}
    ]
    assert request["stream"] is False
    assert request["think"] is False
    assert request["keep_alive"] == "5m"
    assert request["options"] == {"temperature": 0}
    assert response_usage(response) == {
        "prompt_eval_count": 4,
        "eval_count": 2,
    }


def test_ollama_chat_client_omits_think_when_unset() -> None:
    client = FakeOllamaClient()
    config = FakeOllamaConfig(think=None)
    adapter = OllamaChatClient(config, client=client)

    adapter.chat(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "hello"}],
    )

    request = client.calls[0]
    assert "think" not in request


def test_ollama_structured_chat_runs_two_pass_when_thinking_enabled() -> None:
    schema = {"type": "object"}
    client = FakeOllamaClient(
        [
            {
                "message": {
                    "content": "I will return ok.",
                    "thinking": "reasoned about the image",
                },
                "prompt_eval_count": 4,
                "eval_count": 8,
            },
            {
                "message": {"content": '{"ok": true}'},
                "prompt_eval_count": 2,
                "eval_count": 3,
            },
        ]
    )
    adapter = OllamaChatClient(FakeOllamaConfig(think=True), client=client)

    result = adapter.structured_chat(
        model="gemma4:e4b",
        messages=[
            {"role": "user", "content": "Return ok."},
            assistant_json_prefill_message(),
        ],
        response_format=schema,
    )

    thinking_request, structured_request = client.calls
    assert [call.kind for call in result.calls] == ["thinking", "structured"]
    assert result.response == client.responses[1]
    assert "format" not in thinking_request
    assert thinking_request["think"] is True
    assert thinking_request["messages"][-1] == {"role": "user", "content": "Return ok."}
    assert structured_request["format"] == schema
    assert structured_request["think"] is False
    assert structured_request["messages"][-1] == {
        "role": "assistant",
        "content": OLLAMA_JSON_ASSISTANT_PREFILL,
    }
    assert "reasoned about the image" in structured_request["messages"][1]["content"]


def test_ollama_structured_chat_skips_conversion_when_thinking_returns_tools() -> None:
    tool_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "inspect", "arguments": "{}"}},
            ],
        }
    }
    client = FakeOllamaClient([tool_response])
    adapter = OllamaChatClient(FakeOllamaConfig(think=True), client=client)

    result = adapter.structured_chat(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "Inspect first."}],
        tools=[{"type": "function", "function": {"name": "inspect"}}],
        response_format={"type": "object"},
    )

    assert len(client.calls) == 1
    assert result.response == tool_response
    assert result.calls[0].kind == "thinking"
    assert client.calls[0]["tools"] == [
        {"type": "function", "function": {"name": "inspect"}}
    ]
    assert "format" not in client.calls[0]
