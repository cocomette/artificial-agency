"""Tests for the agent context historizer model role."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.historizer import (
    AgentContextHistorizerAdapter,
    AgentContextHistoryInput,
    HistorizerConfig,
    HistorizerOutputError,
    OllamaHistorizerConfig,
    OpenAIHistorizerConfig,
    PromptHistorizerProviderResponse,
    PromptHistorizerRequest,
    VLLMHistorizerConfig,
    agent_context_history_json_schema,
    parse_agent_context_history_output,
)
from face_of_agi.models.historizer.providers.ollama import OllamaHistorizerAdapter
from face_of_agi.models.historizer.providers.openai import OpenAIHistorizerAdapter
from face_of_agi.models.historizer.providers.vllm import VLLMHistorizerAdapter


def _field_evolution(prefix: str) -> dict[str, str]:
    return {
        "goals": f"{prefix} goals",
        "game_mechanics": f"{prefix} mechanics",
        "policy": f"{prefix} policy",
        "history": f"{prefix} history",
        "extras": f"{prefix} extras",
    }


def _history_output(prefix: str = "evolved") -> str:
    return json.dumps({"field_evolution": _field_evolution(prefix)})


def _context(index: int) -> str:
    return json.dumps(
        {
            "goals": f"goals {index}",
            "game_mechanics": f"mechanics {index}",
            "policy": f"policy {index}",
            "history": f"history {index}",
            "extras": f"extras {index}",
        },
        indent=2,
    )


class FakeHistorizerProvider:
    backend = "fake"
    model = "fake-historizer"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [_history_output()]
        self.requests: list[PromptHistorizerRequest] = []
        self.repairs: list[dict[str, object]] = []

    def summarize_context_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        self.requests.append(request)
        return PromptHistorizerProviderResponse(text=self.responses[0])

    def repair_context_history(
        self,
        request: PromptHistorizerRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptHistorizerProviderResponse:
        self.repairs.append(
            {
                "request": request,
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            }
        )
        return PromptHistorizerProviderResponse(text=self.responses[attempt])


class FakeOpenAIResponses:
    def __init__(self, output_text: str | list[str]) -> None:
        self.output_texts = (
            output_text if isinstance(output_text, list) else [output_text]
        )
        self.calls: list[dict[str, object]] = []

    def create(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        output_text = self.output_texts[
            min(len(self.calls) - 1, len(self.output_texts) - 1)
        ]
        return {
            "id": f"resp-{len(self.calls)}",
            "model": request["model"],
            "status": "completed",
            "output_text": output_text,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class FakeOpenAIClient:
    def __init__(self, output_text: str | list[str]) -> None:
        self.responses = FakeOpenAIResponses(output_text)


class FakeOllamaClient:
    def __init__(self, content: str | list[Any]) -> None:
        self.contents = [content] if isinstance(content, str) else list(content)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **request: Any) -> Any:
        self.calls.append(request)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return SimpleNamespace(
            message={"content": content},
            prompt_eval_count=1,
            eval_count=1,
        )


class FakeVLLMChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "id": "chatcmpl-1",
            "model": request["model"],
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class FakeVLLMClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=FakeVLLMChatCompletions(content))


def _history_input() -> AgentContextHistoryInput:
    return AgentContextHistoryInput(
        game_id="game-1",
        context_window=8,
        contexts=(_context(1), _context(2)),
    )


def test_parse_agent_context_history_output_requires_exact_fields() -> None:
    summary = parse_agent_context_history_output(_history_output("updated"))

    assert summary.field_evolution == _field_evolution("updated")

    with pytest.raises(HistorizerOutputError, match="missing keys"):
        parse_agent_context_history_output(
            json.dumps({"field_evolution": {"goals": "only"}})
        )
    with pytest.raises(HistorizerOutputError, match="unexpected keys"):
        parse_agent_context_history_output(
            json.dumps(
                {
                    "field_evolution": {
                        **_field_evolution("updated"),
                        "extra": "nope",
                    }
                }
            )
        )


def test_parse_agent_context_history_output_accepts_long_fields() -> None:
    field_evolution = {
        key: f"{key} " + ("expanded detail " * 250)
        for key in _field_evolution("long")
    }

    summary = parse_agent_context_history_output(
        json.dumps({"field_evolution": field_evolution})
    )

    assert summary.field_evolution == field_evolution


def test_prompt_historizer_builds_context_history_prompt(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("historizer instructions", encoding="utf-8")
    provider = FakeHistorizerProvider()
    historizer = AgentContextHistorizerAdapter(
        provider=provider,
        config=HistorizerConfig(instruction_path=str(instruction_path)),
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.field_evolution == _field_evolution("evolved")
    request = provider.requests[0]
    assert request.instructions == "historizer instructions"
    assert request.output_schema == agent_context_history_json_schema()
    assert "## Agent game context history" in request.text
    assert "1. {" in request.text
    assert "goals 1" in request.text
    assert "goals 2" in request.text


def test_prompt_historizer_repairs_invalid_structured_output() -> None:
    provider = FakeHistorizerProvider(
        responses=[
            json.dumps({"field_evolution": {"goals": "only"}}),
            _history_output("repaired"),
        ]
    )
    historizer = AgentContextHistorizerAdapter(
        provider=provider,
        config=HistorizerConfig(repair_attempts=1),
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.field_evolution == _field_evolution("repaired")
    assert summary.metadata["repair_attempts"] == 1
    assert provider.repairs[0]["attempt"] == 1


def test_openai_historizer_uses_structured_response_format(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("openai instructions", encoding="utf-8")
    client = FakeOpenAIClient(_history_output("openai"))
    historizer = OpenAIHistorizerAdapter(
        OpenAIHistorizerConfig(
            backend="openai",
            model="gpt-5-nano",
            instruction_path=str(instruction_path),
        ),
        client=client,
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.field_evolution == _field_evolution("openai")
    request = client.responses.calls[0]
    assert request["instructions"] == "openai instructions"
    assert (
        request["text"]["format"]["schema"] == agent_context_history_json_schema()
    )
    assert "oldest-to-newest" in request["input"][0]["content"][0]["text"]
    records = drain_model_input_debug_records(historizer)
    assert records[0]["call_slot"] == "historizer"
    assert records[0]["provider"] == "openai"


def test_ollama_historizer_uses_structured_chat_schema(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("ollama instructions", encoding="utf-8")
    client = FakeOllamaClient(_history_output("ollama"))
    historizer = OllamaHistorizerAdapter(
        OllamaHistorizerConfig(
            backend="ollama",
            model="gemma4:e4b",
            instruction_path=str(instruction_path),
        ),
        client=client,
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.field_evolution == _field_evolution("ollama")
    request = client.calls[0]
    assert request["format"] == agent_context_history_json_schema()
    assert request["messages"][0]["content"] == "ollama instructions"
    assert request["messages"][1]["content"].startswith("## Game")
    records = drain_model_input_debug_records(historizer)
    assert records[0]["call_slot"] == "historizer"
    assert records[0]["provider"] == "ollama"


def test_vllm_historizer_uses_json_schema_response_format(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("vllm instructions", encoding="utf-8")
    client = FakeVLLMClient(_history_output("vllm"))
    historizer = VLLMHistorizerAdapter(
        VLLMHistorizerConfig(
            backend="vllm",
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            instruction_path=str(instruction_path),
            use_response_format=True,
        ),
        client=client,
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.field_evolution == _field_evolution("vllm")
    request = client.chat.completions.calls[0]
    assert request["messages"][0]["content"] == "vllm instructions"
    assert request["response_format"]["json_schema"]["schema"] == (
        agent_context_history_json_schema()
    )
    records = drain_model_input_debug_records(historizer)
    assert records[0]["call_slot"] == "historizer"
    assert records[0]["provider"] == "vllm"
