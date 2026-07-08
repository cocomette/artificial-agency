"""Tests for OpenAI-backed world and goal prediction adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.goal import OpenAIDescriptionConfig
from face_of_agi.models.goal import GoalPredictionAdapter
from face_of_agi.models.world import OpenAIDescriptionConfig
from face_of_agi.models.world import WorldPredictionAdapter


class FakeResponses:
    """Captures Responses calls and returns one description response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-description",
            model=kwargs["model"],
            status="completed",
            output=[],
            output_text=json.dumps(
                {
                    "items": [
                        {
                            "bbox_2d": [0, 0, 7, 7],
                            "description": "black source frame area",
                        }
                    ]
                }
            ),
            usage=None,
            incomplete_details=None,
        )


class FakeClient:
    """Tiny OpenAI client stand-in."""

    def __init__(self, responses: object | None = None) -> None:
        self.responses = responses or FakeResponses()


def test_openai_world_prediction_composes_prompt_and_returns_result() -> None:
    client = FakeClient()
    adapter = WorldPredictionAdapter(
        config=OpenAIDescriptionConfig(metadata={"role": "world"}),
        client=client,
    )
    action = ActionSpec(action_id="ACTION6", data={"x": 2, "y": 3})
    observation = Observation(
        id="obs-openai-world",
        step=4,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(general="General world facts.", game="Game dynamics."),
        action=action,
        observation=observation,
    )

    prompt = adapter.last_prompt
    request = client.responses.calls[0]
    assert prompt is not None
    assert adapter.last_instructions is not None
    assert request["model"] == "gpt-5-nano"
    assert request["text"]["format"]["schema"]["type"] == "object"
    assert "items" in request["text"]["format"]["schema"]["properties"]
    item_properties = request["text"]["format"]["schema"]["properties"]["items"][
        "items"
    ]["properties"]
    assert "bbox_2d" in item_properties
    assert item_properties["bbox_2d"]["type"] == "array"
    assert result.id.startswith("world-")
    assert result.tool == "world"
    assert result.action == action
    assert result.source_observation_ref.id == "obs-openai-world"
    assert result.predicted_description[0]["description"] == "black source frame area"
    assert result.explanation == "Predicted next state as a structured description."
    assert result.metadata["backend"] == "openai"
    assert result.metadata["model"] == "gpt-5-nano"


def test_openai_goal_prediction_composes_prompt_and_returns_result() -> None:
    client = FakeClient()
    adapter = GoalPredictionAdapter(
        config=OpenAIDescriptionConfig(metadata={"role": "goal"}),
        client=client,
    )
    observation = Observation(
        id="obs-openai-goal",
        step=5,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(
            general="General goal facts.",
            game="Reach the green exit.",
        ),
        observation=observation,
    )

    prompt = adapter.last_prompt
    request = client.responses.calls[0]
    assert prompt is not None
    assert adapter.last_instructions is not None
    assert request["model"] == "gpt-5-nano"
    assert request["text"]["format"]["schema"]["type"] == "object"
    assert "items" in request["text"]["format"]["schema"]["properties"]
    item_properties = request["text"]["format"]["schema"]["properties"]["items"][
        "items"
    ]["properties"]
    assert "bbox_2d" in item_properties
    assert item_properties["bbox_2d"]["type"] == "array"
    assert result.id.startswith("goal-")
    assert result.tool == "goal"
    assert result.action is None
    assert result.source_observation_ref.id == "obs-openai-goal"
    assert result.predicted_description[0]["description"] == "black source frame area"
    assert (
        result.explanation
        == "Predicted goal-relevant state as a structured description."
    )
    assert result.metadata["backend"] == "openai"
    assert result.metadata["model"] == "gpt-5-nano"
