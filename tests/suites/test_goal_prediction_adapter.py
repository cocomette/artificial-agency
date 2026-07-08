"""Tests for the concrete goal-model description adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

from face_of_agi.contracts import (
    Observation,
    ObservationRef,
    PredictionCall,
    RoleContext,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.goal import GoalPredictionAdapter, OllamaDescriptionConfig
from face_of_agi.orchestration.prediction_router import PredictionRouter


class FakeOllamaClient:
    """Tiny Ollama client stand-in."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            message={
                "content": json.dumps(
                    [
                        {
                            "bbox_2d": [0, 0, 1000, 1000],
                            "description": "goal-relevant target area",
                        }
                    ]
                )
            },
            done_reason="stop",
        )


def test_goal_prediction_returns_description_result() -> None:
    client = FakeOllamaClient()
    adapter = GoalPredictionAdapter(
        config=OllamaDescriptionConfig(model="gemma4:e4b", input_image_size="64x64"),
        client=client,
    )
    observation = Observation(
        id="obs-goal",
        step=4,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(general="General goal facts.", game="Reach the exit."),
        observation=observation,
    )

    prompt = adapter.last_prompt
    request = client.calls[0]
    assert prompt is not None
    system_prompt = request["messages"][0]["content"]
    user_prompt = request["messages"][1]["content"]
    assert adapter.last_instructions is not None
    assert prompt == user_prompt
    assert "Your task is to understand the current goal hypothesis" in system_prompt
    assert "ROLE_CONTEXT:" in prompt
    assert "General goal facts." in prompt
    assert "Reach the exit." in prompt
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
    assert result.id.startswith("goal-")
    assert result.tool == "goal"
    assert result.action is None
    assert result.source_observation_ref.id == "obs-goal"
    assert result.predicted_description[0]["bbox_2d"] == [0.0, 0.0, 8.0, 8.0]
    assert result.predicted_description[0]["description"] == "goal-relevant target area"
    assert result.metadata["backend"] == "ollama"
    assert result.metadata["input_source"] == "image"
    assert result.metadata["visual_coordinate_space"] == "normalized_1000"
    assert result.metadata["visual_coordinate_space_source"] == "model_profile"
    records = drain_model_input_debug_records(adapter)
    assert records[0]["call_slot"] == "goal"
    assert records[0]["provider"] == "ollama"
    assert records[0]["phase"] == "complete"
    assert records[0]["request"]["messages"][1]["images"]
    assert (
        json.loads(records[0]["metadata"]["response_output_text"])[0]["description"]
        == "goal-relevant target area"
    )
    assert records[0]["metadata"]["response_payload"]["message"]["content"] == (
        records[0]["metadata"]["response_output_text"]
    )


def test_goal_prediction_empty_context_user_message_is_only_context_section() -> None:
    client = FakeOllamaClient()
    adapter = GoalPredictionAdapter(
        config=OllamaDescriptionConfig(model="gemma4:e4b", input_image_size="64x64"),
        client=client,
    )

    adapter.predict(
        context=RoleContext(),
        observation=Observation(
            id="obs-empty-goal-context",
            step=1,
            frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        ),
    )

    user_prompt = client.calls[0]["messages"][1]["content"]
    assert user_prompt == "ROLE_CONTEXT:"


def test_prediction_router_routes_goal_without_action() -> None:
    client = FakeOllamaClient()
    goal_model = GoalPredictionAdapter(
        config=OllamaDescriptionConfig(model="gemma4:e4b"),
        client=client,
    )
    router = PredictionRouter(goal_model=goal_model)
    observation = Observation(
        id="obs-router",
        step=1,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )
    call = PredictionCall(
        tool="goal",
        source_state_id=7,
    )

    result = router.route(
        call=call,
        context=RoleContext(game="Goal model doc."),
        observation=observation,
    )

    assert result.tool == "goal"
    assert result.action is None
    assert result.source_state_id == 7
    assert result.source_observation_ref.id == observation.id
