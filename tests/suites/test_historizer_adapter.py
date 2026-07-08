"""Tests for the agent context historizer model role."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
)
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
from face_of_agi.models.world import (
    AgentContextWorldSummary,
    AgentWorldModelInput,
    PromptWorldProviderResponse,
    PromptWorldRequest,
    WorldModelConfig,
    WorldModelOutputError,
    agent_world_model_json_schema,
    parse_agent_world_model_output,
)
from face_of_agi.models.world.adapter import AgentWorldModelAdapter


def _summary_output(
    prefix: str = "evolved",
    *,
    updater_mode: str = "probing",
) -> str:
    return json.dumps(
        {
            "probing_evolution": f"{prefix} probing",
            "policy_evolution": f"{prefix} policy",
            "strategy_summary": f"{prefix} strategy",
            "updater_mode": updater_mode,
        }
    )


def _world_output(prefix: str = "evolved") -> str:
    return json.dumps(
        {
            "world_description": f"{prefix} latest mechanics",
            "special_events": f"{prefix} rare feedback",
            "action_effects": {
                "ACTION1": f"{prefix} action1",
                "ACTION6": f"{prefix} action6",
            },
        }
    )


def _summary_snapshot(index: int) -> str:
    return json.dumps(
        {
            "probing_strategy": f"probing summary {index}",
            "policy_strategy": f"policy summary {index}",
        },
        indent=2,
    )


class FakeHistorizerProvider:
    backend = "fake"
    model = "fake-historizer"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [_summary_output()]
        self.requests: list[PromptHistorizerRequest] = []
        self.repairs: list[dict[str, object]] = []

    def summarize_context_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        self.requests.append(request)
        return PromptHistorizerProviderResponse(text=self.responses.pop(0))

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
        return PromptHistorizerProviderResponse(text=self.responses.pop(0))

class FakeWorldProvider:
    backend = "fake"
    model = "fake-world"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [_world_output()]
        self.requests: list[PromptWorldRequest] = []
        self.repairs: list[dict[str, object]] = []

    def summarize_world_model(
        self,
        request: PromptWorldRequest,
    ) -> PromptWorldProviderResponse:
        self.requests.append(request)
        return PromptWorldProviderResponse(text=self.responses.pop(0))

    def repair_world_model(
        self,
        request: PromptWorldRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptWorldProviderResponse:
        self.repairs.append(
            {
                "request": request,
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            }
        )
        return PromptWorldProviderResponse(text=self.responses.pop(0))

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
    def __init__(self, content: str | list[str]) -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return {
            "id": "chatcmpl-1",
            "model": request["model"],
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class FakeVLLMClient:
    def __init__(self, content: str | list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeVLLMChatCompletions(content))


def _history_input() -> AgentContextHistoryInput:
    return AgentContextHistoryInput(
        game_id="game-1",
        context_window=8,
        strategy_history=(_summary_snapshot(1), _summary_snapshot(2)),
        current_world_model=_world_summary(),
        previous_world_model=_world_output("previous"),
        previous_observation=Observation(
            id="obs-0",
            step=2,
            frame=Image.new("RGB", (64, 64), color=(5, 6, 7)),
        ),
        current_observation=Observation(
            id="obs-1",
            step=3,
            frame=Image.new("RGB", (64, 64), color=(1, 2, 3)),
        ),
        action_history=(
            ActionHistoryEntry(
                action=ActionSpec(action_id="ACTION1"),
                controllable=True,
                changed_pixel_count=12,
                change_summary="opened a path",
                action_mode="probing",
            ),
        ),
        allowed_actions=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION6"),
        ),
    )


def _world_input() -> AgentWorldModelInput:
    history_input = _history_input()
    return AgentWorldModelInput(
        game_id=history_input.game_id,
        previous_world_model=history_input.previous_world_model,
        current_observation=history_input.current_observation,
        action_history=history_input.action_history,
        allowed_actions=history_input.allowed_actions,
    )


def _world_summary(prefix: str = "evolved") -> AgentContextWorldSummary:
    return AgentContextWorldSummary(
        world_description=f"{prefix} latest mechanics",
        action_effects={
            "ACTION1": f"{prefix} action1",
            "ACTION6": f"{prefix} action6",
        },
        special_events=f"{prefix} rare feedback",
        metadata={"source": "test"},
    )


def test_parse_agent_context_history_output_reads_fields_and_mode() -> None:
    summary = parse_agent_context_history_output(
        _summary_output("updated", updater_mode="policy")
    )

    assert summary.probing_evolution == "updated probing"
    assert summary.policy_evolution == "updated policy"
    assert summary.strategy_summary == "updated strategy"
    assert summary.updater_mode == "policy"

    with pytest.raises(HistorizerOutputError, match="probing_evolution"):
        parse_agent_context_history_output(
            json.dumps(
                {
                    "updater_mode": "probing",
                }
            )
        )
    with pytest.raises(HistorizerOutputError, match="updater_mode"):
        parse_agent_context_history_output(
            json.dumps(
                {
                    "probing_evolution": "updated probing",
                    "policy_evolution": "updated policy",
                    "strategy_summary": "updated strategy",
                }
            )
        )
    with pytest.raises(HistorizerOutputError, match="strategy_summary"):
        parse_agent_context_history_output(
            json.dumps(
                {
                    "probing_evolution": "updated probing",
                    "policy_evolution": "updated policy",
                    "updater_mode": "probing",
                }
            )
        )
    with pytest.raises(HistorizerOutputError, match="updater_mode"):
        parse_agent_context_history_output(
            json.dumps(
                {
                    "probing_evolution": "updated probing",
                    "policy_evolution": "updated policy",
                    "strategy_summary": "updated strategy",
                    "updater_mode": "other",
                }
            )
        )


def test_parse_agent_context_history_output_accepts_long_fields() -> None:
    probing_evolution = "probing " + ("compressed detail " * 250)
    policy_evolution = "policy " + ("compressed detail " * 250)
    strategy_summary = "strategy " + ("compressed detail " * 250)

    summary = parse_agent_context_history_output(
        json.dumps(
            {
                "probing_evolution": probing_evolution,
                "policy_evolution": policy_evolution,
                "strategy_summary": strategy_summary,
                "updater_mode": "probing",
            }
        )
    )

    assert summary.probing_evolution == probing_evolution
    assert summary.policy_evolution == policy_evolution
    assert summary.strategy_summary == strategy_summary


def test_parse_agent_world_model_output_reads_mechanics_and_actions() -> None:
    summary = parse_agent_world_model_output(
        _world_output("updated"),
        allowed_actions=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION6"),
        ),
    )

    assert summary.world_description == "updated latest mechanics"
    assert summary.special_events == "updated rare feedback"
    assert summary.action_effects == {
        "ACTION1": "updated action1",
        "ACTION6": "updated action6",
    }

    with pytest.raises(WorldModelOutputError, match="action_effects"):
        parse_agent_world_model_output(
            json.dumps(
                {
                    "world_description": "updated latest mechanics",
                    "special_events": "updated rare feedback",
                }
            )
        )
    with pytest.raises(WorldModelOutputError, match="missing keys"):
        parse_agent_world_model_output(
            json.dumps(
                {
                    "world_description": "updated latest mechanics",
                    "special_events": "updated rare feedback",
                    "action_effects": {"ACTION1": "effect"},
                }
            ),
            allowed_actions=(
                ActionSpec(action_id="ACTION1"),
                ActionSpec(action_id="ACTION6"),
            ),
        )


def test_prompt_historizer_builds_context_history_prompt(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("historizer instructions", encoding="utf-8")
    provider = FakeHistorizerProvider()
    historizer = AgentContextHistorizerAdapter(
        provider=provider,
        config=HistorizerConfig(instruction_path=str(instruction_path)),
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.probing_evolution == "evolved probing"
    assert summary.policy_evolution == "evolved policy"
    assert summary.strategy_summary == "evolved strategy"
    assert summary.action_effects["ACTION1"] == "evolved action1"
    assert summary.special_events == "evolved rare feedback"
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.instructions.startswith("historizer instructions")
    assert "Output JSON must match this schema exactly." in request.instructions
    assert '"probing_evolution"' in request.instructions
    assert '"policy_evolution"' in request.instructions
    assert '"strategy_summary"' in request.instructions
    assert request.output_schema == agent_context_history_json_schema()
    assert "## World model" in request.text
    assert "Special events:" in request.text
    assert "evolved rare feedback" in request.text
    assert "## Probing/policy history" in request.text
    assert "## Allowed actions" in request.text
    assert "ACTION1" in request.text
    assert "ACTION6" in request.text
    assert "## Action history" in request.text
    assert "[mode=probing]" in request.text
    assert "opened a path" in request.text
    assert "1. {" in request.text
    assert "probing_strategy" in request.text
    assert "policy_strategy" in request.text
    assert "probing summary 1" in request.text
    assert "policy summary 2" in request.text


def test_prompt_world_model_builds_prompt_and_crops_images() -> None:
    provider = FakeWorldProvider(responses=[_world_output()])
    world_model = AgentWorldModelAdapter(
        provider=provider,
        config=WorldModelConfig(),
    )
    world_input = _world_input()
    assert world_input.current_observation is not None
    world_input.current_observation.frame = Image.new(
        "RGB",
        (64, 64),
        color=(1, 2, 3),
    )

    world_model.summarize_agent_world_model(world_input)

    world_request = provider.requests[0]
    assert '"world_description"' in world_request.instructions
    assert '"special_events"' in world_request.instructions
    assert "## Action glossary" in world_request.instructions
    assert "- `ACTION1`: up." in world_request.instructions
    assert "- `ACTION6`:" in world_request.instructions
    assert world_request.output_schema == agent_world_model_json_schema(
        world_input.allowed_actions
    )
    assert "## Previous world model" in world_request.text
    assert [image.label for image in world_request.images] == [
        "current_observation_frame",
    ]
    assert [image.image.size for image in world_request.images] == [(56, 56)]


def test_prompt_world_model_ignores_animation_bundle_frames() -> None:
    provider = FakeWorldProvider(responses=[_world_output()])
    world_model = AgentWorldModelAdapter(
        provider=provider,
        config=WorldModelConfig(input_image_size=(100, 80)),
    )
    frame_observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=1,
            frame=Image.new("RGB", (64, 64), color=(index, index, index)),
        )
        for index in range(20)
    )
    world_input = _world_input()
    world_input.current_observation = frame_observations[-1]

    world_model.summarize_agent_world_model(world_input)

    world_request = provider.requests[0]
    assert [image.label for image in world_request.images] == [
        "current_observation_frame",
    ]
    assert [image.image.size for image in world_request.images] == [(100, 80)]


def test_prompt_world_model_falls_back_to_previous_world_after_repair_exhaustion(
    caplog,
) -> None:
    provider = FakeWorldProvider(responses=["{}"])
    world_model = AgentWorldModelAdapter(
        provider=provider,
        config=WorldModelConfig(repair_attempts=0),
    )
    world_input = _world_input()

    with caplog.at_level("ERROR"):
        summary = world_model.summarize_agent_world_model(world_input)

    assert summary.world_description == "previous latest mechanics"
    assert summary.special_events == "previous rare feedback"
    assert summary.action_effects == {
        "ACTION1": "previous action1",
        "ACTION6": "previous action6",
    }
    assert summary.metadata["fallback"] == "repair_exhausted"
    assert "world model structured output repair exhausted" in caplog.text


def test_prompt_world_model_fallback_uses_empty_schema_without_previous(
    caplog,
) -> None:
    provider = FakeWorldProvider(responses=["{}"])
    world_model = AgentWorldModelAdapter(
        provider=provider,
        config=WorldModelConfig(repair_attempts=0),
    )
    world_input = _world_input()
    world_input.previous_world_model = ""

    with caplog.at_level("ERROR"):
        summary = world_model.summarize_agent_world_model(world_input)

    assert summary.world_description == ""
    assert summary.special_events == ""
    assert summary.action_effects == {"ACTION1": "", "ACTION6": ""}
    assert summary.metadata["fallback"] == "repair_exhausted"


def test_prompt_historizer_repairs_invalid_structured_output() -> None:
    provider = FakeHistorizerProvider(
        responses=[
            json.dumps(
                {
                    "updater_mode": "probing",
                }
            ),
            _summary_output("repaired"),
        ]
    )
    historizer = AgentContextHistorizerAdapter(
        provider=provider,
        config=HistorizerConfig(repair_attempts=1),
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.probing_evolution == "repaired probing"
    assert summary.policy_evolution == "repaired policy"
    assert summary.strategy_summary == "repaired strategy"
    assert summary.metadata["historizer"]["repair_attempts"] == 1
    assert provider.repairs[0]["attempt"] == 1


def test_prompt_historizer_falls_back_to_policy_after_repair_exhaustion(
    caplog,
) -> None:
    provider = FakeHistorizerProvider(responses=["{}"])
    historizer = AgentContextHistorizerAdapter(
        provider=provider,
        config=HistorizerConfig(repair_attempts=0),
    )

    with caplog.at_level("ERROR"):
        summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.probing_evolution == ""
    assert summary.policy_evolution == ""
    assert summary.strategy_summary == ""
    assert summary.updater_mode == "policy"
    assert summary.metadata["historizer"]["fallback"] == "repair_exhausted"
    assert "historizer structured output repair exhausted" in caplog.text


def test_openai_historizer_uses_structured_response_format(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("openai instructions", encoding="utf-8")
    client = FakeOpenAIClient([_summary_output("openai")])
    historizer = OpenAIHistorizerAdapter(
        OpenAIHistorizerConfig(
            backend="openai",
            model="gpt-5-nano",
            instruction_path=str(instruction_path),
        ),
        client=client,
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.probing_evolution == "openai probing"
    assert summary.policy_evolution == "openai policy"
    summary_request = client.responses.calls[0]
    assert summary_request["instructions"].startswith("openai instructions")
    assert "Output JSON must match this schema exactly." in summary_request["instructions"]
    assert (
        summary_request["text"]["format"]["schema"]
        == agent_context_history_json_schema()
    )
    assert summary_request["text"]["format"]["name"] == "agent_context_history"
    assert (
        '1. {\n  "probing_strategy": "probing summary 1",'
        in summary_request["input"][0]["content"][0]["text"]
    )
    assert len(summary_request["input"][0]["content"]) == 1
    assert summary_request["input"][0]["content"][0]["type"] == "input_text"
    records = drain_model_input_debug_records(historizer)
    assert records[0]["call_slot"] == "historizer"
    assert {record["provider"] for record in records} == {"openai"}


def test_ollama_historizer_uses_structured_chat_schema(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("ollama instructions", encoding="utf-8")
    client = FakeOllamaClient([_summary_output("ollama")])
    historizer = OllamaHistorizerAdapter(
        OllamaHistorizerConfig(
            backend="ollama",
            model="gemma4:e4b",
            instruction_path=str(instruction_path),
        ),
        client=client,
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.probing_evolution == "ollama probing"
    assert summary.policy_evolution == "ollama policy"
    summary_request = client.calls[0]
    assert summary_request["format"] == agent_context_history_json_schema()
    assert summary_request["messages"][0]["content"].startswith("ollama instructions")
    assert (
        "Output JSON must match this schema exactly."
        in summary_request["messages"][0]["content"]
    )
    assert summary_request["messages"][1]["content"].startswith(
        "## World model"
    )
    assert "images" not in summary_request["messages"][1]
    records = drain_model_input_debug_records(historizer)
    assert records[0]["call_slot"] == "historizer"
    assert {record["provider"] for record in records} == {"ollama"}


def test_vllm_historizer_uses_json_schema_response_format(tmp_path) -> None:
    instruction_path = tmp_path / "historizer.md"
    instruction_path.write_text("vllm instructions", encoding="utf-8")
    client = FakeVLLMClient([_summary_output("vllm")])
    historizer = VLLMHistorizerAdapter(
        VLLMHistorizerConfig(
            backend="vllm",
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            instruction_path=str(instruction_path),
        ),
        client=client,
    )

    summary = historizer.summarize_agent_context_history(_history_input())

    assert summary.probing_evolution == "vllm probing"
    assert summary.policy_evolution == "vllm policy"
    summary_request = client.chat.completions.calls[0]
    assert summary_request["messages"][0]["content"].startswith("vllm instructions")
    assert (
        "Output JSON must match this schema exactly."
        in summary_request["messages"][0]["content"]
    )
    assert summary_request["response_format"]["json_schema"]["schema"] == (
        agent_context_history_json_schema()
    )
    assert summary_request["response_format"]["json_schema"]["name"] == (
        "agent_context_history"
    )
    assert len(summary_request["messages"][1]["content"]) == 1
    assert summary_request["messages"][1]["content"][0]["type"] == "text"
    records = drain_model_input_debug_records(historizer)
    assert records[0]["call_slot"] == "historizer"
    assert {record["provider"] for record in records} == {"vllm"}
