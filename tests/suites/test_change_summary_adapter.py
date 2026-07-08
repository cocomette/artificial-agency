"""Tests for the transition change-summary model role."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

from face_of_agi.contracts import ActionSpec, Observation
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.change import (
    CHANGE_SUMMARY_PROMPT,
    ChangeSummaryAdapter,
    ChangeSummaryOutputError,
    ChangeSummaryProviderResponse,
    OllamaChangeSummaryConfig,
    OpenAIChangeSummaryConfig,
    VLLMChangeSummaryConfig,
    build_change_summary_prompt,
    change_summary_json_schema,
    load_change_summary_instructions,
    model_visible_changed_pixel_percent,
    parse_change_summary_output,
)
from face_of_agi.models.change.providers.ollama import OllamaChangeSummaryProvider
from face_of_agi.models.change.providers.openai import OpenAIChangeSummaryProvider
from face_of_agi.models.change.providers.vllm import VLLMChangeSummaryProvider
from face_of_agi.models.image_inputs import frame_bundle_image_size


class FakeOpenAIResponses:
    """Tiny OpenAI Responses stand-in."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "id": "resp-1",
            "model": request["model"],
            "status": "completed",
            "output_text": self.output_text,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class FakeOpenAIClient:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeOpenAIResponses(output_text)


class FakeOllamaClient:
    """Tiny Ollama chat stand-in."""

    def __init__(self, content: str | list[Any]) -> None:
        self.contents = [content] if isinstance(content, str) else list(content)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **request: Any) -> Any:
        self.calls.append(request)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        if isinstance(content, dict):
            return content
        return SimpleNamespace(
            message={"content": content},
            prompt_eval_count=1,
            eval_count=1,
        )


class FakeVLLMChatCompletions:
    """Tiny OpenAI-compatible Chat Completions stand-in."""

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


class RecordingChangeProvider:
    """Fake provider that records adapter prompt and image payloads."""

    backend = "fake"
    model = "fake-change"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def complete(self, **request: Any) -> ChangeSummaryProviderResponse:
        self.calls.append(request)
        return ChangeSummaryProviderResponse(
            text=self.output_text,
            metadata={"backend": self.backend},
            request={"prompt_text": request["prompt_text"]},
        )

    def repair_complete(self, **request: Any) -> ChangeSummaryProviderResponse:
        raise AssertionError("repair should not be called")


def _change_payload(summary: str, *, change_detected: bool = True) -> str:
    return json.dumps(
        {
            "summary": summary,
            "change_detected": change_detected,
        }
    )


def test_change_summary_parser_accepts_json_and_fenced_json() -> None:
    no_change = parse_change_summary_output(
        '{"summary": "No visible change.", "change_detected": false}'
    )
    assert no_change.summary == "No visible change."
    assert no_change.change_detected is False

    moved = parse_change_summary_output(
        '```json\n{"summary": "The avatar moved right.", '
        '"change_detected": true}\n```'
    )
    assert moved.summary == "The avatar moved right."
    assert moved.change_detected is True


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"summary": ""}',
        '{"summary": "No visible change."}',
        '{"summary": "No visible change.", "change_detected": "yes"}',
        (
            '{"summary": "No visible change.", "change_detected": false, '
            '"extra": "field"}'
        ),
    ],
)
def test_change_summary_parser_rejects_invalid_payloads(
    payload: str,
) -> None:
    with pytest.raises(ChangeSummaryOutputError):
        parse_change_summary_output(payload)


def test_change_summary_schema_requires_summary_and_change_detected() -> None:
    schema = change_summary_json_schema()

    assert schema["required"] == ["summary", "change_detected"]
    assert set(schema["properties"]) == {"summary", "change_detected"}


def test_change_adapter_passes_action_prompt_and_two_images() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            frame_scale=1,
            input_image_size=(2, 2),
            input_image_resample="nearest",
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = Observation(
        id="previous",
        step=0,
        frame=Image.new("RGB", (2, 2), color=(10, 20, 30)),
    )
    current = Observation(
        id="current",
        step=1,
        frame=Image.new("RGB", (2, 2), color=(200, 210, 220)),
    )
    action = ActionSpec(action_id="ACTION6", data={"y": 3, "x": 2})

    result = adapter.summarize(
        previous,
        current,
        action,
        glossary_actions=(ActionSpec(action_id="ACTION1"), action),
        changed_pixel_percent=100,
    )

    call = provider.calls[0]
    prompt = build_change_summary_prompt(action, changed_pixel_percent=100)
    assert result.summary == "The cursor moved."
    assert result.changed_pixel_percent == 100
    assert result.change_detected is True
    assert CHANGE_SUMMARY_PROMPT == (
        "Compare the attached observation frames from oldest to newest."
    )
    assert prompt.startswith(
        "Compare the attached observation frames from oldest to newest.\n\n"
        "TRANSITION:"
    )
    assert adapter.last_prompt == prompt
    assert call["prompt_text"] == prompt
    assert "## Action glossary" in adapter.last_instructions
    assert "- `ACTION1`: up." in adapter.last_instructions
    assert "- `ACTION6`: coordinate action mapped to the game grid." in (
        adapter.last_instructions
    )
    assert "- `ACTION2`" not in adapter.last_instructions
    assert "TRANSITION:\nattached_frame_count: 2\nchanged_pixel_percent: 100" in prompt
    assert "ACTION:\naction_id: ACTION6\ndata: {\"x\": 31, \"y\": 47}" in prompt
    assert "coordinate_space: normalized_0_1000" in prompt
    previous_image = call["previous_image"]
    current_image = call["current_image"]
    assert call["images"] == (previous_image, current_image)
    assert previous_image.size == (2, 2)
    assert current_image.size == (2, 2)
    assert previous_image.getpixel((0, 0)) == (10, 20, 30)
    assert current_image.getpixel((0, 0)) == (200, 210, 220)


def test_change_summary_prompt_renders_action6_relative_to_arc_crop() -> None:
    action = ActionSpec(action_id="ACTION6", data={"x": 32, "y": 43})

    prompt = build_change_summary_prompt(
        action,
        changed_pixel_percent=50,
        crop_edges=4,
    )

    assert "ACTION:\naction_id: ACTION6\ndata: {\"x\": 500, \"y\": 696}" in prompt
    assert "coordinate_space: normalized_0_1000" in prompt


def test_change_adapter_passes_multi_frame_transition_evidence() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor animated."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            frame_scale=1,
            input_image_size=(4, 4),
            input_image_resample="nearest",
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = Observation(
        id="previous",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(10, 20, 30)),
    )
    animation = Observation(
        id="animation",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(100, 110, 120)),
    )
    current = Observation(
        id="current",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(200, 210, 220)),
    )

    adapter.summarize(
        previous,
        current,
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        changed_pixel_percent=100,
        frame_observations=(previous, animation, current),
        max_transition_changed_pixel_percent=100,
    )

    call = provider.calls[0]
    assert len(call["images"]) == 3
    assert "attached_frame_count: 3" in call["prompt_text"]
    assert "max_transition_changed_pixel_percent: 100" in call["prompt_text"]
    assert call["previous_image"] is call["images"][0]
    assert call["current_image"] is call["images"][-1]
    assert [image.size for image in call["images"]] == [(3, 3), (3, 3), (3, 3)]


def test_frame_bundle_image_size_keeps_area_budget_near_two_frames() -> None:
    assert frame_bundle_image_size("1024x1024", frame_count=2) == (1024, 1024)
    assert frame_bundle_image_size("1024x1024", frame_count=4) == (724, 724)
    assert frame_bundle_image_size(None, frame_count=4) is None


def test_change_adapter_applies_arc_grid_crop_after_resize() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            frame_scale=1,
            input_image_size=(4, 3),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=[0, 0, 0, 16],
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous_frame = Image.new("RGB", (4, 4), color=(10, 20, 30))
    current_frame = Image.new("RGB", (4, 4), color=(200, 210, 220))
    for x in range(4):
        previous_frame.putpixel((x, 3), (0, 255, 0))
        current_frame.putpixel((x, 3), (255, 0, 0))

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous_frame),
        Observation(id="current", step=1, frame=current_frame),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        changed_pixel_percent=100,
    )

    previous_image = provider.calls[0]["previous_image"]
    current_image = provider.calls[0]["current_image"]
    assert previous_image.size == (4, 2)
    assert current_image.size == (4, 2)
    assert adapter.last_prompt is not None
    assert "changed_pixel_percent: 100" in adapter.last_prompt
    assert all(
        previous_image.getpixel((x, y)) == (10, 20, 30)
        for x in range(4)
        for y in range(2)
    )
    assert all(
        current_image.getpixel((x, y)) == (200, 210, 220)
        for x in range(4)
        for y in range(2)
    )


def test_change_adapter_uses_provider_when_orchestration_reports_no_change() -> None:
    provider = RecordingChangeProvider(
        _change_payload("No visible change.", change_detected=False)
    )
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            frame_scale=1,
            input_image_size=(4, 4),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=[0, 0, 0, 32],
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous_frame = Image.new("RGB", (4, 4), color=(10, 20, 30))
    current_frame = Image.new("RGB", (4, 4), color=(10, 20, 30))
    for x in range(4):
        current_frame.putpixel((x, 3), (200, 210, 220))

    result = adapter.summarize(
        Observation(id="previous", step=0, frame=previous_frame),
        Observation(id="current", step=1, frame=current_frame),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        changed_pixel_percent=0,
    )

    assert result.summary == "No visible change."
    assert result.changed_pixel_percent == 0
    assert result.change_detected is False
    assert len(provider.calls) == 1
    assert adapter.last_prompt is not None
    assert "changed_pixel_percent: 0" in adapter.last_prompt
    assert adapter.last_instructions is not None
    assert adapter.last_request is not None


def test_model_visible_changed_pixel_percent_returns_zero_for_matching_images() -> None:
    image = Image.new("RGB", (2, 2), color=(1, 2, 3))

    assert model_visible_changed_pixel_percent(image, image) == 0.0


def test_model_visible_changed_pixel_percent_scales_partial_change() -> None:
    previous = Image.new("RGB", (2, 2), color=(1, 2, 3))
    current = Image.new("RGB", (2, 2), color=(1, 2, 3))
    current.putpixel((0, 0), (1, 2, 4))
    current.putpixel((1, 1), (5, 6, 7))

    assert model_visible_changed_pixel_percent(previous, current) == 50.0


def test_model_visible_changed_pixel_percent_returns_100_for_full_change() -> None:
    previous = Image.new("RGB", (2, 2), color=(1, 2, 3))
    current = Image.new("RGB", (2, 2), color=(5, 6, 7))

    assert model_visible_changed_pixel_percent(previous, current) == 100.0


def test_model_visible_changed_pixel_percent_returns_100_for_shape_mismatch() -> None:
    previous = Image.new("RGB", (2, 2), color=(1, 2, 3))
    current = Image.new("RGB", (3, 2), color=(1, 2, 3))

    assert model_visible_changed_pixel_percent(previous, current) == 100.0


def test_change_adapter_rejects_invalid_arc_grid_crop() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            frame_scale=1,
            input_image_size=(4, 4),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=[0, 32, 0, 32],
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )

    with pytest.raises(ValueError, match="leaves no visible frame"):
        adapter.summarize(
            Observation(id="previous", step=0, frame=Image.new("RGB", (4, 4))),
            Observation(id="current", step=1, frame=Image.new("RGB", (4, 4))),
            ActionSpec(action_id="ACTION1"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            changed_pixel_percent=100,
        )
    assert provider.calls == []


def test_openai_change_provider_sends_two_images() -> None:
    payload = _change_payload("The screen inverted.")
    client = FakeOpenAIClient(payload)
    provider = OpenAIChangeSummaryProvider(
        OpenAIChangeSummaryConfig(
            model="gpt-5-nano",
            input_image_size="2x2",
        ),
        client=client,
    )
    previous_image = _previous_image()
    current_image = _current_image()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=CHANGE_SUMMARY_PROMPT,
        previous_image=previous_image,
        current_image=current_image,
        output_schema=change_summary_json_schema(),
    )

    request = client.responses.calls[0]
    content = request["input"][0]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert request["text"]["format"]["schema"]["required"] == [
        "summary",
        "change_detected",
    ]
    assert result.text == payload
    assert content[0] == {"type": "input_text", "text": CHANGE_SUMMARY_PROMPT}
    assert len(images) == 2
    previous_payload = _image_from_data_url(images[0]["image_url"])
    current_payload = _image_from_data_url(images[1]["image_url"])
    assert previous_payload.size == (2, 2)
    assert current_payload.size == (2, 2)
    assert previous_payload.getpixel((0, 0)) == (10, 20, 30)
    assert current_payload.getpixel((0, 0)) == (200, 210, 220)
    records = drain_model_input_debug_records(provider)
    assert records[0]["call_slot"] == "change"


def test_ollama_change_provider_sends_two_images() -> None:
    payload = _change_payload("The screen inverted.")
    client = FakeOllamaClient(payload)
    provider = OllamaChangeSummaryProvider(
        OllamaChangeSummaryConfig(
            model="gemma4:e4b",
            input_image_size="2x2",
        ),
        client=client,
    )
    previous_image = _previous_image()
    current_image = _current_image()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=CHANGE_SUMMARY_PROMPT,
        previous_image=previous_image,
        current_image=current_image,
        output_schema=change_summary_json_schema(),
    )

    request = client.calls[0]
    user_message = request["messages"][1]
    assert request["format"]["required"] == ["summary", "change_detected"]
    assert result.text == payload
    assert user_message["content"] == CHANGE_SUMMARY_PROMPT
    assert len(user_message["images"]) == 2
    previous_payload = _image_from_base64_png(user_message["images"][0])
    current_payload = _image_from_base64_png(user_message["images"][1])
    assert previous_payload.size == (2, 2)
    assert current_payload.size == (2, 2)
    assert previous_payload.getpixel((0, 0)) == (10, 20, 30)
    assert current_payload.getpixel((0, 0)) == (200, 210, 220)
    records = drain_model_input_debug_records(provider)
    assert records[0]["call_slot"] == "change"


def test_vllm_change_provider_sends_two_images() -> None:
    payload = _change_payload("The screen inverted.")
    client = FakeVLLMClient(payload)
    provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            input_image_size="2x2",
            use_response_format=True,
        ),
        client=client,
    )
    previous_image = _previous_image()
    current_image = _current_image()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=CHANGE_SUMMARY_PROMPT,
        previous_image=previous_image,
        current_image=current_image,
        output_schema=change_summary_json_schema(),
    )

    request = client.chat.completions.calls[0]
    content = request["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert request["response_format"]["json_schema"]["schema"]["required"] == [
        "summary",
        "change_detected",
    ]
    assert result.text == payload
    assert request["response_format"]["type"] == "json_schema"
    assert content[0] == {"type": "text", "text": CHANGE_SUMMARY_PROMPT}
    assert len(images) == 2
    previous_payload = _image_from_data_url(images[0]["image_url"]["url"])
    current_payload = _image_from_data_url(images[1]["image_url"]["url"])
    assert previous_payload.size == (2, 2)
    assert current_payload.size == (2, 2)
    assert previous_payload.getpixel((0, 0)) == (10, 20, 30)
    assert current_payload.getpixel((0, 0)) == (200, 210, 220)
    records = drain_model_input_debug_records(provider)
    assert records[0]["call_slot"] == "change"


def test_vllm_change_provider_preserves_preprocessed_image_size() -> None:
    payload = _change_payload("The screen inverted.")
    client = FakeVLLMClient(payload)
    provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            input_image_size="4x4",
        ),
        client=client,
    )

    provider.complete(
        instructions_text="Return JSON.",
        prompt_text=CHANGE_SUMMARY_PROMPT,
        previous_image=_previous_image(),
        current_image=_current_image(),
        output_schema=change_summary_json_schema(),
    )

    content = client.chat.completions.calls[0]["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    previous_payload = _image_from_data_url(images[0]["image_url"]["url"])
    current_payload = _image_from_data_url(images[1]["image_url"]["url"])
    assert previous_payload.size == (2, 2)
    assert current_payload.size == (2, 2)


def test_ollama_change_provider_two_passes_when_thinking_enabled() -> None:
    client = FakeOllamaClient(
        [
            {
                "message": {
                    "content": "The screen inverted.",
                    "thinking": "compared both images",
                },
                "prompt_eval_count": 2,
                "eval_count": 3,
            },
            _change_payload("The screen inverted."),
        ]
    )
    provider = OllamaChangeSummaryProvider(
        OllamaChangeSummaryConfig(
            model="gemma4:e4b",
            input_image_size="2x2",
            think=True,
        ),
        client=client,
    )

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=CHANGE_SUMMARY_PROMPT,
        previous_image=_previous_image(),
        current_image=_current_image(),
        output_schema=change_summary_json_schema(),
    )

    assert result.text == _change_payload("The screen inverted.")
    assert len(client.calls) == 2
    assert "format" not in client.calls[0]
    assert client.calls[0]["think"] is True
    assert client.calls[0]["messages"][-1]["role"] == "user"
    assert client.calls[1]["format"] == change_summary_json_schema()
    assert client.calls[1]["think"] is False
    assert client.calls[1]["messages"][-1] == {
        "role": "assistant",
        "content": "```json\n",
    }
    records = drain_model_input_debug_records(provider)
    assert [record["phase"] for record in records] == [
        "complete_thinking",
        "complete",
    ]
    assert records[0]["metadata"]["response_payload"]["message"]["thinking"] == (
        "compared both images"
    )
    assert records[1]["request"] == client.calls[1]


def _previous_image() -> Image.Image:
    return Image.new("RGB", (2, 2), color=(10, 20, 30))


def _current_image() -> Image.Image:
    return Image.new("RGB", (2, 2), color=(200, 210, 220))


def _image_from_data_url(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return _image_from_base64_png(encoded)


def _image_from_base64_png(encoded: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
