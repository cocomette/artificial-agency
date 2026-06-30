"""Tests for shared vLLM provider-call behavior."""

from __future__ import annotations

from typing import Any

import pytest

from face_of_agi.models.providers.vllm import VLLMChatClient, VLLMChatConfig


class FakeContextOverflowError(RuntimeError):
    """Fake OpenAI/vLLM BadRequestError carrying vLLM's overflow wording."""


class FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = type(
            "FakeChat",
            (),
            {"completions": FakeCompletions(responses)},
        )()


def _overflow_error(
    *,
    max_context_tokens: int = 100,
    output_tokens: int = 0,
    input_tokens: int = 120,
) -> FakeContextOverflowError:
    return FakeContextOverflowError(
        "This model's maximum context length is "
        f"{max_context_tokens} tokens. However, you requested "
        f"{output_tokens} output tokens and your prompt contains at least "
        f"{input_tokens} input tokens."
    )


def _chat_response() -> dict[str, Any]:
    return {
        "id": "response-1",
        "choices": [{"message": {"role": "assistant", "content": "{}"}}],
    }


def test_vllm_chat_retries_context_overflow_with_truncated_mutable_text() -> None:
    client = FakeClient([_overflow_error(), _chat_response()])
    chat = VLLMChatClient(
        VLLMChatConfig(
            model="fake-vllm",
            max_context_tokens=100,
            max_completion_tokens=10,
            context_truncation_margin_tokens=5,
        ),
        client=client,
    )
    user_text = "a" * 1000

    response = chat.chat(
        model="fake-vllm",
        messages=[
            {"role": "system", "content": "fixed instructions"},
            {"role": "user", "content": user_text},
        ],
    )

    assert response["id"] == "response-1"
    assert len(client.chat.completions.calls) == 2
    first_request = client.chat.completions.calls[0]
    second_request = client.chat.completions.calls[1]
    assert first_request["messages"][1]["content"] == user_text
    assert second_request["messages"][0]["content"] == "fixed instructions"
    assert len(second_request["messages"][1]["content"]) < len(user_text)
    assert "truncated to fit vLLM context window" in (
        second_request["messages"][1]["content"]
    )
    assert chat.last_truncation is not None
    assert chat.last_truncation["removed_chars"] > 0
    assert chat.last_truncation["prompt_token_budget"] == 85


def test_vllm_chat_truncates_multimodal_text_without_removing_images() -> None:
    client = FakeClient([_overflow_error(), _chat_response()])
    chat = VLLMChatClient(
        VLLMChatConfig(
            model="fake-vllm",
            max_context_tokens=100,
            max_completion_tokens=10,
            context_truncation_margin_tokens=5,
        ),
        client=client,
    )
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"},
    }

    chat.chat(
        model="fake-vllm",
        messages=[
            {"role": "system", "content": "fixed instructions"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a" * 1000},
                    image_part,
                ],
            },
        ],
    )

    second_request = client.chat.completions.calls[1]
    content = second_request["messages"][1]["content"]
    assert content[1] == image_part
    assert len(content[0]["text"]) < 1000
    assert "truncated to fit vLLM context window" in content[0]["text"]
    assert chat.last_truncation is not None
    assert chat.last_truncation["removed_chars"] > 0


def test_vllm_chat_does_not_truncate_system_only_context_overflow() -> None:
    client = FakeClient([_overflow_error()])
    chat = VLLMChatClient(
        VLLMChatConfig(model="fake-vllm", context_overflow_retries=1),
        client=client,
    )

    with pytest.raises(RuntimeError, match="no mutable chat message text"):
        chat.chat(
            model="fake-vllm",
            messages=[{"role": "system", "content": "x" * 1000}],
        )

    assert len(client.chat.completions.calls) == 1
