"""Tests for the concrete world-model description adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.world import WorldPredictionAdapter, OllamaDescriptionConfig


class FakeOllamaClient:
    """Tiny Ollama client stand-in."""

    def __init__(self, contents: list[str] | None = None) -> None:
        self.contents = contents
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = (
            self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
            if self.contents is not None
            else json.dumps(
                [
                    {
                        "bbox_2d": [0, 0, 1000, 1000],
                        "description": "predicted next black frame",
                    }
                ]
            )
        )
        return SimpleNamespace(
            message={"content": content},
            done_reason="stop",
        )


def test_world_prediction_returns_description_result() -> None:
    client = FakeOllamaClient()
    adapter = WorldPredictionAdapter(
        config=OllamaDescriptionConfig(model="gemma4:e4b", input_image_size="64x64"),
        client=client,
    )
    action = ActionSpec(action_id="ACTION6", data={"x": 2, "y": 3})
    observation = Observation(
        id="obs-1",
        step=4,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(general="General world facts.", game="Game dynamics."),
        action=action,
        observation=observation,
    )

    prompt = adapter.last_prompt
    request = client.calls[0]
    assert prompt is not None
    user_prompt = request["messages"][1]["content"]
    assert adapter.last_instructions is not None
    assert prompt == user_prompt
    assert request["model"] == "gemma4:e4b"
    assert request["format"]
    assert "bbox_2d" in request["format"]["items"]["properties"]
    assert request["format"]["items"]["properties"]["bbox_2d"]["type"] == "array"
    assert request["format"]["items"]["properties"]["description"]["description"] == (
        "Concise expected next-frame change for the currently bounded visible "
        "image area."
    )
    assert request["messages"][1]["images"]
    assert request["messages"][2] == {"role": "assistant", "content": "```json\n"}
    assert result.id.startswith("world-")
    assert result.tool == "world"
    assert result.action == action
    assert result.source_observation_ref.id == "obs-1"
    assert result.predicted_description[0]["bbox_2d"] == [0.0, 0.0, 8.0, 8.0]
    assert (
        result.predicted_description[0]["description"]
        == "predicted next black frame"
    )
    assert result.metadata["backend"] == "ollama"
    assert result.metadata["input_source"] == "image"
    assert result.metadata["visual_coordinate_space"] == "normalized_1000"
    assert result.metadata["visual_coordinate_space_source"] == "model_profile"
    records = drain_model_input_debug_records(adapter)
    assert records[0]["call_slot"] == "world"
    assert records[0]["provider"] == "ollama"
    assert records[0]["phase"] == "complete"
    assert records[0]["request"]["messages"][1]["images"]
    assert (
        json.loads(records[0]["metadata"]["response_output_text"])[0]["description"]
        == "predicted next black frame"
    )
    assert records[0]["metadata"]["response_payload"]["message"]["content"] == (
        records[0]["metadata"]["response_output_text"]
    )


def test_world_prediction_can_include_output_schema_in_instructions() -> None:
    client = FakeOllamaClient()
    adapter = WorldPredictionAdapter(
        config=OllamaDescriptionConfig(
            model="gemma4:e4b",
            input_image_size="64x64",
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )

    adapter.predict(
        context=RoleContext(game="Game dynamics."),
        action=ActionSpec(action_id="ACTION1"),
        observation=Observation(
            id="obs-schema",
            step=1,
            frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        ),
    )

    instructions = str(client.calls[0]["messages"][0]["content"])
    assert "Output JSON must match this schema exactly." in instructions
    assert '"bbox_2d"' in instructions
    assert '"description"' in instructions


def test_world_prediction_repairs_invalid_description_json() -> None:
    client = FakeOllamaClient(
        [
            '{"description": "not an array"}',
            json.dumps(
                [
                    {
                        "bbox_2d": [0, 0, 1000, 1000],
                        "description": "repaired prediction",
                    }
                ]
            ),
        ]
    )
    adapter = WorldPredictionAdapter(
        config=OllamaDescriptionConfig(model="gemma4:e4b", input_image_size="64x64"),
        client=client,
    )

    result = adapter.predict(
        context=RoleContext(game="Objects move after actions."),
        action=ActionSpec(action_id="ACTION1"),
        observation=Observation(
            id="obs-repair",
            step=1,
            frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        ),
    )

    assert len(client.calls) == 2
    repair_request = client.calls[1]
    assert "Repair attempt 1" in repair_request["messages"][1]["content"]
    assert "description prediction must be a JSON array" in repair_request[
        "messages"
    ][1]["content"]
    assert result.predicted_description[0]["description"] == "repaired prediction"
    assert result.metadata["repair_attempts"] == 1


def test_world_prediction_requires_model_vision_profile() -> None:
    client = FakeOllamaClient()

    with pytest.raises(ValueError, match="vision_profiles.json"):
        WorldPredictionAdapter(
            config=OllamaDescriptionConfig(model="unknown-vlm"),
            client=client,
        )
