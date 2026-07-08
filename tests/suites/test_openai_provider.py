"""Tests for the shared OpenAI Responses image provider."""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
import pytest

from face_of_agi.contracts import Observation
from face_of_agi.models.providers import (
    OpenAIImageGenerationClient,
    OpenAIResponsesClient,
    OpenAIResponsesImageConfig,
)


class FakeUsage:
    """Small SDK-model stand-in with the model_dump API."""

    def model_dump(self, **kwargs: object) -> dict[str, int]:
        return {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}


class FakeResponses:
    """Captures Responses API calls and returns a configured response."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    """Tiny OpenAI client stand-in."""

    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


def _encoded_png(size: tuple[int, int] = (3, 4)) -> str:
    image = Image.new("RGB", size, color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def test_openai_provider_builds_responses_image_generation_request() -> None:
    response = SimpleNamespace(
        id="resp-123",
        model="gpt-5-nano",
        status="completed",
        output=[
            SimpleNamespace(
                type="image_generation_call",
                id="ig-123",
                result=_encoded_png(),
                status="completed",
            )
        ],
        output_text="Generated a predicted ARC frame.",
        usage=FakeUsage(),
        incomplete_details=None,
    )
    client = FakeClient(response)
    adapter = OpenAIImageGenerationClient(
        OpenAIResponsesImageConfig(
            reasoning={"effort": "medium"},
            max_output_tokens=256,
            max_tool_calls=1,
            temperature=0.2,
            top_p=0.9,
            text={"verbosity": "low"},
            metadata={"role": "world"},
            store=False,
            service_tier="default",
            prompt_cache_key="arc-world",
            prompt_cache_retention="24h",
            safety_identifier="safe-user",
            truncation="disabled",
            parallel_tool_calls=False,
            include=["message.input_image.image_url"],
            image_quality="high",
            image_size="1024x1024",
            image_output_format="png",
            image_output_compression=80,
            image_background="opaque",
            image_input_fidelity="high",
            image_moderation="low",
            image_partial_images=0,
            image_tool_options={"custom_option": "value"},
            extra_request_options={"extra_body_key": "extra"},
        ),
        client=client,
    )
    observation = Observation(
        id="obs-openai",
        step=2,
        frame=Image.new("RGB", (5, 6), color=(0, 0, 0)),
    )

    result = adapter.generate_image(prompt="Predict the next frame.", observation=observation)

    request = client.responses.calls[0]
    content = request["input"][0]["content"]
    image_tool = request["tools"][0]

    assert request["model"] == "gpt-5-nano"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["tool_choice"] == {"type": "image_generation"}
    assert request["max_tool_calls"] == 1
    assert request["max_output_tokens"] == 256
    assert request["temperature"] == 0.2
    assert request["top_p"] == 0.9
    assert request["text"] == {"verbosity": "low"}
    assert request["metadata"] == {"role": "world"}
    assert request["store"] is False
    assert request["service_tier"] == "default"
    assert request["prompt_cache_key"] == "arc-world"
    assert request["prompt_cache_retention"] == "24h"
    assert request["safety_identifier"] == "safe-user"
    assert request["truncation"] == "disabled"
    assert request["parallel_tool_calls"] is False
    assert request["include"] == ["message.input_image.image_url"]
    assert request["extra_body_key"] == "extra"
    assert content[0] == {"type": "input_text", "text": "Predict the next frame."}
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "auto"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert image_tool["type"] == "image_generation"
    assert image_tool["model"] == "gpt-image-1-mini"
    assert image_tool["action"] == "edit"
    assert image_tool["quality"] == "high"
    assert image_tool["size"] == "1024x1024"
    assert image_tool["output_format"] == "png"
    assert image_tool["output_compression"] == 80
    assert image_tool["background"] == "opaque"
    assert image_tool["input_fidelity"] == "high"
    assert image_tool["moderation"] == "low"
    assert image_tool["partial_images"] == 0
    assert image_tool["custom_option"] == "value"
    assert result.image.size == (3, 4)
    assert result.output_text == "Generated a predicted ARC frame."
    assert result.response_id == "resp-123"
    assert result.metadata["response_id"] == "resp-123"
    assert result.metadata["image_generation_call_id"] == "ig-123"
    assert result.metadata["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }


def test_openai_responses_client_builds_final_provider_request() -> None:
    response = SimpleNamespace(
        id="resp-final",
        model="gpt-5-nano",
        status="completed",
        output_text="ok",
    )
    client = FakeClient(response)
    adapter = OpenAIResponsesClient(
        OpenAIResponsesImageConfig(
            reasoning={"effort": "medium"},
            max_tool_calls=3,
            metadata={"role": "provider-test"},
            extra_request_options={"custom": "value"},
        ),
        client=client,
    )

    result = adapter.create_response(
        model="gpt-5-nano",
        instructions="system",
        input_items=[{"role": "user", "content": "payload"}],
        tools=[{"type": "function", "name": "world"}],
        tool_choice={"type": "function", "name": "world"},
    )

    request = client.responses.calls[0]
    assert result is response
    assert request["model"] == "gpt-5-nano"
    assert request["instructions"] == "system"
    assert request["input"] == [{"role": "user", "content": "payload"}]
    assert request["tools"] == [{"type": "function", "name": "world"}]
    assert request["tool_choice"] == {
        "type": "function",
        "name": "world",
    }
    assert request["reasoning"] == {"effort": "medium"}
    assert request["max_tool_calls"] == 3
    assert request["metadata"] == {"role": "provider-test"}
    assert request["custom"] == "value"


def test_openai_provider_can_resize_input_image_before_upload() -> None:
    response = SimpleNamespace(
        id="resp-123",
        model="gpt-5-nano",
        status="completed",
        output=[
            SimpleNamespace(
                type="image_generation_call",
                id="ig-123",
                result=_encoded_png(),
                status="completed",
            )
        ],
        output_text=None,
    )
    client = FakeClient(response)
    adapter = OpenAIImageGenerationClient(
        OpenAIResponsesImageConfig(
            input_image_size="10x12",
            input_image_resample="nearest",
        ),
        client=client,
    )

    adapter.generate_image(
        prompt="Predict the next frame.",
        observation=Observation(
            id="obs-openai",
            step=2,
            frame=Image.new("RGB", (5, 6), color=(0, 0, 0)),
        ),
    )

    content = client.responses.calls[0]["input"][0]["content"]
    uploaded_image = _decode_data_url_image(content[1]["image_url"])

    assert uploaded_image.size == (10, 12)


def test_openai_provider_raises_when_response_has_no_generated_image() -> None:
    client = FakeClient(
        SimpleNamespace(
            id="resp-empty",
            model="gpt-5-nano",
            status="completed",
            output=[],
            output_text="No image was generated.",
        )
    )
    adapter = OpenAIImageGenerationClient(
        OpenAIResponsesImageConfig(),
        client=client,
    )

    with pytest.raises(RuntimeError, match="image_generation_call"):
        adapter.generate_image(
            prompt="Predict the next frame.",
            observation=Observation(
                id="obs-empty",
                step=0,
                frame=Image.new("RGB", (2, 2), color=(0, 0, 0)),
            ),
        )
