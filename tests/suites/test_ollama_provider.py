"""Tests for the shared Ollama final chat provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
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
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return {
            "message": {"content": "ok"},
            "prompt_eval_count": 4,
            "eval_count": 2,
        }


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
