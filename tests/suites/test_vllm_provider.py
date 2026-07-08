"""Tests for active vLLM Chat Completions model providers."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any

from PIL import Image

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.orchestrator_agent.config import VLLMOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.updater.config import VLLMUpdaterConfig
from face_of_agi.models.updater.contracts import (
    agent_game_updated_context_json_schema,
)
from face_of_agi.models.updater.providers.vllm import VLLMUpdaterAdapter
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
)

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


class FakeChatCompletions:
    """Captures vLLM-compatible Chat Completions calls."""

    def __init__(self, contents: str | dict[str, Any] | list[Any]) -> None:
        self.contents = list(contents) if isinstance(contents, list) else [contents]
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        content = self.contents[index]
        message = (
            {"role": "assistant", "content": content}
            if isinstance(content, str)
            else {"role": "assistant", **content}
        )
        return {
            "id": f"chatcmpl-{index}",
            "model": request["model"],
            "object": "chat.completion",
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        }


class FakeOpenAIChatClient:
    """Tiny OpenAI Chat client stand-in."""

    def __init__(self, contents: str | list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(contents))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def test_vllm_agent_uses_chat_completions_images_and_structured_output() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(json.dumps({"action": {"action_id": "ACTION1"}}))
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            input_image_size="10x12",
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(game="choose directly"),
        current,
        [ActionSpec(action_id="ACTION1")],
    )

    request = client.calls[0]
    images = _input_images(request)
    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["backend"] == "vllm"
    assert request["messages"][0]["role"] == "system"
    assert len(images) == 1
    assert [_decode_data_url_image(image["image_url"]["url"]).size for image in images] == [
        (10, 12),
    ]
    assert request["response_format"]["json_schema"]["name"] == "agent_final_action"
    records = drain_model_input_debug_records(adapter)
    assert records[0]["provider"] == "vllm"
    assert records[0]["phase"] == "final_action"
    assert records[0]["usage"]["total_tokens"] == 6


def test_vllm_agent_game_updater_uses_object_schema_and_image(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    payload = {
        "current_strategy": "try action one",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    client = FakeOpenAIChatClient(json.dumps(payload))
    updater = VLLMUpdaterAdapter(
        VLLMUpdaterConfig(model=MODEL, instruction_dir=str(tmp_path)),
        client=client,
    )

    result = updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="L"),
            current_observation=_observation("obs-agent"),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            world_model_context="world_description: world updated",
        )
    )

    request = client.calls[0]
    assert json.loads(result.context) == {
        "current_strategy": "try action one",
    }
    assert result.next_actions[0].name == "ACTION1"
    assert request["response_format"]["json_schema"]["schema"] == (
        agent_game_updated_context_json_schema(
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )
    assert request["messages"][1]["content"][1]["type"] == "image_url"
    records = drain_model_input_debug_records(updater)
    assert [record["phase"] for record in records] == ["update_prompt"]


def _input_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    content = request["messages"][1]["content"]
    return [item for item in content if item.get("type") == "image_url"]


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _write_instruction_files(path) -> None:
    (path / "agent_context_updater_prompt.md").write_text(
        "agent instructions",
        encoding="utf-8",
    )


def _observation(observation_id: str) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )
