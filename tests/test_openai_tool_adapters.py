"""Tests for OpenAI-backed world and goal model tool adapters."""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.tools.goal import OpenAIGoalToolConfig
from face_of_agi.models.tools.goal.providers.openai import OpenAIGoalToolAdapter
from face_of_agi.models.tools.world import OpenAIWorldToolConfig
from face_of_agi.models.tools.world.providers.openai import OpenAIWorldToolAdapter


class FakeResponses:
    """Captures Responses calls and returns one generated image response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-tool",
            model=kwargs["model"],
            status="completed",
            output=[
                SimpleNamespace(
                    type="image_generation_call",
                    id="ig-tool",
                    result=_encoded_png(),
                    status="completed",
                )
            ],
            output_text="Model-generated tool explanation.",
            usage=None,
            incomplete_details=None,
        )


class FakeClient:
    """Tiny OpenAI client stand-in."""

    def __init__(self) -> None:
        self.responses = FakeResponses()


def _encoded_png() -> str:
    image = Image.new("RGB", (7, 9), color=(90, 10, 20))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_openai_world_tool_predict_composes_prompt_and_returns_tool_result() -> None:
    client = FakeClient()
    adapter = OpenAIWorldToolAdapter(
        config=OpenAIWorldToolConfig(metadata={"role": "world"}),
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
    assert "World Model Instruction" in prompt
    assert "WORLD MODEL DOC (K^S + L^S):" in prompt
    assert "General world facts." in prompt
    assert "Game dynamics." in prompt
    assert "SOURCE OBSERVATION:" in prompt
    assert "id: obs-openai-world" in prompt
    assert "PROPOSED ACTION:" in prompt
    assert "action_id: ACTION6" in prompt
    assert '"x": 2' in prompt
    assert request["model"] == "gpt-5-nano"
    assert request["tools"][0]["model"] == "gpt-image-1-mini"
    assert result.id.startswith("world-")
    assert result.tool == "world"
    assert result.action == action
    assert result.source_observation_ref.id == "obs-openai-world"
    assert isinstance(result.predicted_observation, Image.Image)
    assert result.predicted_observation.size == (7, 9)
    assert result.explanation == "Model-generated tool explanation."
    assert result.metadata["backend"] == "openai"
    assert result.metadata["model"] == "gpt-5-nano"
    assert result.metadata["image_model"] == "gpt-image-1-mini"
    assert result.metadata["image_size"] == (7, 9)
    assert result.metadata["input_image_size"] == "1024x1024"
    assert result.metadata["input_image_resample"] == "nearest"
    assert result.metadata["tool_choice"] == "image_generation"


def test_openai_goal_tool_predict_excludes_action_from_prompt() -> None:
    client = FakeClient()
    adapter = OpenAIGoalToolAdapter(
        config=OpenAIGoalToolConfig(metadata={"role": "goal"}),
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
    assert "Goal Model Instruction" in prompt
    assert "best goal-directed action" in prompt
    assert "Do not merely reproduce the source frame" in prompt
    assert "GOAL MODEL DOC (K^G + L^G):" in prompt
    assert "General goal facts." in prompt
    assert "Reach the green exit." in prompt
    assert "SOURCE OBSERVATION:" in prompt
    assert "id: obs-openai-goal" in prompt
    assert "PROPOSED ACTION" not in prompt
    assert "action_id:" not in prompt
    assert request["model"] == "gpt-5-nano"
    assert request["tools"][0]["model"] == "gpt-image-1-mini"
    assert result.id.startswith("goal-")
    assert result.tool == "goal"
    assert result.action is None
    assert result.source_observation_ref.id == "obs-openai-goal"
    assert isinstance(result.predicted_observation, Image.Image)
    assert result.predicted_observation.size == (7, 9)
    assert result.explanation == "Model-generated tool explanation."
    assert result.metadata["backend"] == "openai"
    assert result.metadata["model"] == "gpt-5-nano"
    assert result.metadata["image_model"] == "gpt-image-1-mini"
    assert result.metadata["input_image_size"] == "1024x1024"
    assert result.metadata["input_image_resample"] == "nearest"
