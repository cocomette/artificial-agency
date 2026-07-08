"""Tests for vLLM Chat Completions model providers."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RoleContext,
    ToolResult,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.config import VLLMOrchestratorAgentConfig
from face_of_agi.models.updater.config import VLLMUpdaterConfig
from face_of_agi.models.updater.contracts import updated_context_json_schema
from face_of_agi.models.updater.providers.vllm import VLLMUpdaterAdapter
from face_of_agi.models.world import VLLMDescriptionConfig, WorldPredictionAdapter
from face_of_agi.models.updater import GoalGameContextUpdateInput

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
            "choices": [
                {
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
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


def _observations() -> tuple[Observation, Observation]:
    return (
        Observation(
            id="obs-first",
            step=0,
            frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        ),
        Observation(
            id="obs-current",
            step=2,
            frame=Image.new("RGB", (8, 8), color=(255, 255, 255)),
        ),
    )


def _submit_arguments(action_id: str = "ACTION1") -> str:
    return json.dumps({"action": {"action_id": action_id}})


def _input_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    content = request["messages"][1]["content"]
    return [item for item in content if item.get("type") == "image_url"]


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _description_prediction(description: str = "predicted change") -> list[dict]:
    return [{"bbox_2d": [0.0, 0.0, 4.0, 4.0], "description": description}]


def test_vllm_agent_uses_chat_completions_images_and_structured_output() -> None:
    first, current = _observations()
    client = FakeOpenAIChatClient(_submit_arguments())
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
        first,
        current,
        [ActionSpec(action_id="ACTION1")],
    )

    request = client.calls[0]
    images = _input_images(request)
    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["backend"] == "vllm"
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["content"][0]["type"] == "text"
    assert len(images) == 2
    assert [_decode_data_url_image(image["image_url"]["url"]).size for image in images] == [
        (10, 12),
        (10, 12),
    ]
    assert request["response_format"]["json_schema"]["name"] == "agent_final_action"
    assert request["response_format"]["json_schema"]["schema"]["properties"]["action"][
        "properties"
    ]["action_id"]["enum"] == ["ACTION1"]
    records = drain_model_input_debug_records(adapter)
    assert records[0]["provider"] == "vllm"
    assert records[0]["phase"] == "final_action"
    assert records[0]["usage"]["total_tokens"] == 6


def test_vllm_world_prediction_composes_chat_request_and_returns_result() -> None:
    client = FakeOpenAIChatClient(
        [
            {"reasoning": "reasoning without final content"},
            json.dumps(
                {
                    "items": [
                        {
                            "bbox_2d": [0, 0, 7, 7],
                            "description": "black source frame area",
                        }
                    ]
                }
            ),
        ]
    )
    adapter = WorldPredictionAdapter(
        config=VLLMDescriptionConfig(model=MODEL, repair_attempts=1),
        client=client,
    )
    observation = Observation(
        id="obs-vllm-world",
        step=4,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(general="General world facts.", game="Game dynamics."),
        action=ActionSpec(action_id="ACTION1"),
        observation=observation,
    )

    request = client.calls[1]
    records = drain_model_input_debug_records(adapter)
    assert request["model"] == MODEL
    assert request["messages"][0]["role"] == "system"
    assert request["response_format"]["json_schema"]["name"] == (
        "description_prediction"
    )
    assert request["response_format"]["json_schema"]["schema"]["type"] == "object"
    assert result.predicted_description[0]["description"] == "black source frame area"
    assert result.metadata["backend"] == "vllm"
    assert result.metadata["model"] == MODEL
    assert [record["phase"] for record in records] == ["complete", "repair_complete"]


def test_vllm_updater_updates_goal_game_context(tmp_path) -> None:
    (tmp_path / "goal_game_context_updater_prompt.md").write_text(
        "goal game instructions",
        encoding="utf-8",
    )
    client = FakeOpenAIChatClient(
        json.dumps({"updated_context": "fixed goal context"})
    )
    updater = VLLMUpdaterAdapter(
        VLLMUpdaterConfig(
            model=MODEL,
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )

    result = updater.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=RoleContext(general="K^G", game="L^G"),
            current_observation=Observation(
                id="obs-1",
                step=1,
                frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
            ),
            post_decision_predictions=PostDecisionPredictions(
                goal_prediction=ToolResult(
                    id="goal-out",
                    tool="goal",
                    predicted_description=_description_prediction(),
                    source_observation_ref=ObservationRef(memory="state", id="obs-0"),
                )
            ),
        )
    )

    assert result.game == "fixed goal context"
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"]["json_schema"]["schema"] == (
        updated_context_json_schema()
    )
    assert client.calls[0]["messages"][1]["content"][1]["type"] == "image_url"
    records = drain_model_input_debug_records(updater)
    assert [record["phase"] for record in records] == ["update_prompt"]
