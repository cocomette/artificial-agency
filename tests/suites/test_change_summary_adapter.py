"""Tests for the transition change-summary model role."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

from face_of_agi.contracts import ActionSpec, ChangeSummaryElement, Observation
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.change import (
    ChangeSummaryAdapter,
    ChangeSummaryOutputError,
    ChangeSummaryProviderResponse,
    OllamaChangeSummaryConfig,
    OpenAIChangeSummaryConfig,
    VLLMChangeSummaryConfig,
    build_change_summary_prompt,
    change_summary_json_schema,
    load_change_summary_instructions,
    parse_change_summary_output,
    change_summary_elements_text,
)
from face_of_agi.models.change.providers.ollama import OllamaChangeSummaryProvider
from face_of_agi.models.change.providers.openai import OpenAIChangeSummaryProvider
from face_of_agi.models.change.providers.vllm import VLLMChangeSummaryProvider


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

    def __init__(self, output_text: str | Sequence[str]) -> None:
        self.output_texts = [output_text] if isinstance(output_text, str) else list(
            output_text
        )
        self.calls: list[dict[str, Any]] = []

    def complete(self, **request: Any) -> ChangeSummaryProviderResponse:
        output_text = self.output_texts[
            min(len(self.calls), len(self.output_texts) - 1)
        ]
        self.calls.append(request)
        return ChangeSummaryProviderResponse(
            text=output_text,
            metadata={"backend": self.backend},
            request={"prompt_text": request["prompt_text"]},
        )

    def repair_complete(self, **request: Any) -> ChangeSummaryProviderResponse:
        raise AssertionError("repair should not be called")


def _change_payload(mutation: str, *, change_detected: bool = True) -> str:
    return json.dumps(
        {
            "elements": [
                {
                    "element_name": "cursor",
                    "element_description": "small bright cursor",
                    "element_mutation": mutation,
                }
            ],
            "change_detected": change_detected,
        }
    )


def _action_prompt() -> str:
    return (
        "## Previous change elements\n\n[]\n\n"
        "ACTION:\naction_id: ACTION1\ndata: {}"
    )


def _arc_grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]


def prompt_frame_text(prompt: str, *, frame_index: int) -> str:
    start_marker = f"frame {frame_index}:"
    start = prompt.index(start_marker)
    next_markers = (f"\n\nframe {frame_index + 1}:", "\n\nACTION:")
    end_candidates = [
        prompt.index(marker, start)
        for marker in next_markers
        if marker in prompt[start:]
    ]
    if not end_candidates:
        return prompt[start:]
    end = min(end_candidates)
    return prompt[start:end]


def prompt_previous_change_elements(prompt: str) -> list[dict[str, str]]:
    start_marker = "## Previous change elements\n\n"
    start = prompt.index(start_marker) + len(start_marker)
    end_candidates = [
        prompt.index(marker, start)
        for marker in ("\n\n## Frame components", "\n\nACTION:")
        if marker in prompt[start:]
    ]
    return json.loads(prompt[start : min(end_candidates)])


def test_change_summary_parser_accepts_json_and_fenced_json() -> None:
    no_change = parse_change_summary_output(
        '{"elements": [{"element_name": "wall", '
        '"element_description": "blue vertical wall", '
        '"element_mutation": ""}], "change_detected": false}'
    )
    assert no_change.elements[0].element_name == "wall"
    assert no_change.elements[0].element_description == "blue vertical wall"
    assert no_change.elements[0].element_mutation == ""
    assert no_change.change_detected is False

    moved = parse_change_summary_output(
        '```json\n{"elements": [{"element_name": "avatar", '
        '"element_description": "red square", '
        '"element_mutation": "moved right"}], '
        '"change_detected": true}\n```'
    )
    assert moved.elements[0].element_name == "avatar"
    assert moved.elements[0].element_mutation == "moved right"
    assert moved.change_detected is True


def test_change_summary_parser_renames_duplicate_element_names() -> None:
    result = parse_change_summary_output(
        json.dumps(
            {
                "elements": [
                    {
                        "element_name": "block",
                        "element_description": "left blue block",
                        "element_mutation": "moved left",
                    },
                    {
                        "element_name": "block",
                        "element_description": "right blue block",
                        "element_mutation": "moved right",
                    },
                    {
                        "element_name": "cursor",
                        "element_description": "white cursor",
                        "element_mutation": "",
                    },
                ],
                "change_detected": True,
            }
        )
    )

    assert [element.element_name for element in result.elements] == [
        "block_0",
        "block_1",
        "cursor",
    ]


def test_change_summary_elements_text_renders_readable_bullets() -> None:
    assert (
        change_summary_elements_text(
            (
                ChangeSummaryElement(
                    element_name="black L-shaped object",
                    element_description="A black L-shaped object with white dots.",
                    element_mutation="Moved to the right.",
                ),
                ChangeSummaryElement(
                    element_name="yellow L-shaped object",
                    element_description="A yellow L-shaped object.",
                    element_mutation="",
                ),
            )
        )
        == "- black L-shaped object: A black L-shaped object with white dots.; "
        "mutations: Moved to the right.\n"
        "- yellow L-shaped object: A yellow L-shaped object.; "
        "mutations: no detected changes for this element"
    )


@pytest.mark.parametrize(
    "payload",
        [
            "{}",
            '{"elements": "Moved.", "change_detected": true}',
            '{"elements": [{}], "change_detected": true}',
            '{"elements": [], "change_detected": "true"}',
        ],
    )
def test_change_summary_parser_rejects_invalid_payloads(
    payload: str,
) -> None:
    with pytest.raises(ChangeSummaryOutputError):
        parse_change_summary_output(payload)


def test_change_summary_falls_back_after_repair_exhaustion(caplog) -> None:
    provider = RecordingChangeProvider("{}")
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            backend="ollama",
            model="fake-model",
            repair_attempts=0,
        ),
        provider=provider,
    )

    with caplog.at_level("ERROR"):
        result = adapter.summarize(
            Observation(
                id="obs-1",
                step=1,
                frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            ),
            Observation(
                id="obs-2",
                step=2,
                frame=Image.new("RGB", (8, 8), color=(1, 1, 1)),
            ),
            ActionSpec(action_id="ACTION1"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            previous_change_elements=(),
        )

    assert result.elements == ()
    assert result.change_detected is False
    assert result.metadata["fallback"] == "repair_exhausted"
    assert "change summary structured output repair exhausted" in caplog.text


def test_change_summary_schema_requires_elements() -> None:
    schema = change_summary_json_schema()

    assert "elements" in schema["required"]
    assert "change_detected" in schema["required"]
    element_schema = schema["properties"]["elements"]["items"]
    assert element_schema["required"] == [
        "element_name",
        "element_description",
        "element_mutation",
    ]
    assert schema["properties"]["change_detected"]["type"] == "boolean"


def test_change_summary_configs_default_bounding_boxes_off() -> None:
    configs = (
        OllamaChangeSummaryConfig(),
        OpenAIChangeSummaryConfig(),
        VLLMChangeSummaryConfig(),
    )

    assert [config.activate_bounding_boxes for config in configs] == [False] * 3
    assert [config.dilation_bounding_boxes for config in configs] == [3] * 3
    assert [config.width_bounding_boxes for config in configs] == [3] * 3
    assert [config.animation_frame_budget_coefficient for config in configs] == (
        [2] * 3
    )
    assert [config.gaussian_blur_kernel_size for config in configs] == [0] * 3
    assert [config.gaussian_noise_deviation for config in configs] == [0.0] * 3
    assert [config.activate_diff_mask for config in configs] == [False] * 3
    assert [config.activate_components for config in configs] == [False] * 3
    assert [config.max_nb_components for config in configs] == [50] * 3
    assert [config.max_frames_per_call for config in configs] == [10] * 3
    assert VLLMChangeSummaryConfig().frame_input_mode == "image"
    assert VLLMChangeSummaryConfig().video_fps == 1.0
    assert VLLMChangeSummaryConfig().video_mime_type == "video/jpeg"


def test_load_change_summary_instructions_merges_bounding_box_guidance_on_request() -> None:
    base = load_change_summary_instructions()
    with_boxes = load_change_summary_instructions(include_bounding_boxes=True)

    assert "bright pink" not in base
    assert "pink outlines" not in base
    assert "bright pink" in with_boxes
    assert "pink outlines" in with_boxes


def test_load_change_summary_instructions_merges_diff_mask_guidance_on_request() -> None:
    base = load_change_summary_instructions()
    with_diff_masks = load_change_summary_instructions(include_diff_masks=True)

    assert "Diff Mask Inputs" not in base
    assert "binary changed-pixel masks" not in base
    assert "Diff Mask Inputs" in with_diff_masks
    assert "binary changed-pixel masks" in with_diff_masks


def test_load_change_summary_instructions_merges_component_guidance_on_request() -> None:
    base = load_change_summary_instructions()
    with_components = load_change_summary_instructions(include_components=True)

    assert "Component Guidance" not in base
    assert "ARC rendered color legend" not in base
    assert "Component Guidance" in with_components
    assert "0 to 1000" in with_components
    assert "ARC rendered color legend" in with_components
    assert "- symbol 0: rgb(" in with_components


def test_change_adapter_passes_action_prompt_and_two_images() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
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
    action = ActionSpec(
        action_id="ACTION6",
        data={"y": 3, "x": 2},
        target="the cursor target",
    )

    result = adapter.summarize(
        previous,
        current,
        action,
        glossary_actions=(ActionSpec(action_id="ACTION1"), action),
        previous_change_elements=(),
    )

    call = provider.calls[0]
    prompt = build_change_summary_prompt(
        action,
        previous_change_elements=(),
    )
    assert result.elements[0].element_mutation == "The cursor moved."
    assert result.change_detected is True
    assert "ACTION:" in prompt
    assert adapter.last_prompt == prompt
    assert call["prompt_text"] == prompt
    assert "## Action glossary" in adapter.last_instructions
    assert "- `ACTION1`: up." in adapter.last_instructions
    assert "- `ACTION6`: coordinate action mapped to the game grid, shown by target." in (
        adapter.last_instructions
    )
    assert "- `ACTION2`" not in adapter.last_instructions
    assert "bright pink" not in adapter.last_instructions
    assert "Frame components" not in adapter.last_prompt
    assert "ACTION:\naction_id: ACTION6\ndata: {\"x\": 31, \"y\": 47}" in prompt
    assert "target: the cursor target" in prompt
    assert "coordinate_space: normalized_0_1000" in prompt
    previous_image = call["previous_image"]
    current_image = call["current_image"]
    assert len(call["images"]) == 2
    assert previous_image.size == (2, 2)
    assert current_image.size == (2, 2)
    assert previous_image.getpixel((0, 0)) == (10, 20, 30)
    assert current_image.getpixel((0, 0)) == (200, 210, 220)


def test_change_adapter_adds_compact_components_when_enabled() -> None:
    provider = RecordingChangeProvider(_change_payload("The block appeared."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_components=True,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = _arc_grid()
    current = _arc_grid()
    for y in range(20, 22):
        for x in range(10, 12):
            current[y][x] = 4

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous),
        Observation(id="current", step=1, frame=current),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    prompt = provider.calls[0]["prompt_text"]
    assert "Component Guidance" in adapter.last_instructions
    assert "ARC rendered color legend" in adapter.last_instructions
    assert "## Frame components" in prompt
    assert "frame 0:" in prompt
    assert "frame 1:" in prompt
    assert "visible_arc_crop_edges" not in prompt
    assert "visible_image_size" not in prompt
    assert "id=" not in prompt
    assert "color=rgb(" not in prompt
    assert "area=" not in prompt
    assert "centroid" not in prompt
    assert "runs=" not in prompt
    assert "- symbol=4 nb=1 box=[(156,312,188,344)]" in prompt


def test_change_adapter_scales_components_after_change_summary_crop() -> None:
    provider = RecordingChangeProvider(_change_payload("The block changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            activate_components=True,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = _arc_grid()
    current = _arc_grid()
    current[4][4] = 8

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous),
        Observation(id="current", step=1, frame=current),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    prompt = provider.calls[0]["prompt_text"]
    assert "- symbol=8 nb=1 box=[(0,0,18,18)]" in prompt


def test_change_adapter_groups_matching_components_by_shape_and_symbol() -> None:
    provider = RecordingChangeProvider(_change_payload("The blocks changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_components=True,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = _arc_grid()
    current = _arc_grid()
    for origin_x, origin_y in ((10, 20), (30, 40)):
        for y in range(origin_y, origin_y + 2):
            for x in range(origin_x, origin_x + 2):
                current[y][x] = 4

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous),
        Observation(id="current", step=1, frame=current),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    prompt = provider.calls[0]["prompt_text"]
    assert (
        "- symbol=4 nb=2 box=[(156,312,188,344), (469,625,500,656)]"
        in prompt
    )


def test_change_adapter_sorts_and_prunes_component_groups() -> None:
    provider = RecordingChangeProvider(_change_payload("The blocks changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_components=True,
            max_nb_components=3,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = _arc_grid()
    current = _arc_grid()
    for y in range(10, 12):
        for x in range(10, 12):
            current[y][x] = 5
    current[20][20] = 7
    current[30][30] = 6
    current[40][40] = 6

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous),
        Observation(id="current", step=1, frame=current),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    frame_1_text = prompt_frame_text(provider.calls[0]["prompt_text"], frame_index=1)
    symbol_0_index = frame_1_text.index("- symbol=0")
    symbol_5_index = frame_1_text.index("- symbol=5 nb=1")
    symbol_7_index = frame_1_text.index("- symbol=7 nb=1")
    assert symbol_0_index < symbol_5_index < symbol_7_index
    assert "- symbol=6 nb=2" not in frame_1_text


def test_change_adapter_builds_components_for_each_animation_frame() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_components=True,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    observations = []
    for index, symbol in enumerate((1, 2, 3)):
        frame = _arc_grid()
        frame[10][10] = symbol
        observations.append(Observation(id=f"frame-{index}", step=index, frame=frame))

    adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=tuple(observations),
        previous_change_elements=(),
    )

    prompt = provider.calls[0]["prompt_text"]
    assert prompt.count("frame 0:") == 1
    assert prompt.count("frame 1:") == 1
    assert prompt.count("frame 2:") == 1
    assert "- symbol=1 nb=1 box=[(156,156,172,172)]" in prompt
    assert "- symbol=2 nb=1 box=[(156,156,172,172)]" in prompt
    assert "- symbol=3 nb=1 box=[(156,156,172,172)]" in prompt


def test_change_adapter_chunks_long_frame_bundles_and_merges_elements() -> None:
    provider = RecordingChangeProvider(
        (
            json.dumps(
                {
                    "elements": [
                        {
                            "element_name": "door",
                            "element_description": "closed red door",
                            "element_mutation": "opened",
                        }
                    ],
                    "change_detected": True,
                }
            ),
            json.dumps(
                {
                    "elements": [
                        {
                            "element_name": "door",
                            "element_description": "open red doorway",
                            "element_mutation": "camera followed it",
                        },
                        {
                            "element_name": "key",
                            "element_description": "yellow key",
                            "element_mutation": "",
                        },
                    ],
                    "change_detected": True,
                }
            ),
            json.dumps(
                {
                    "elements": [
                        {
                            "element_name": "door",
                            "element_description": "red exit doorway",
                            "element_mutation": "",
                        },
                        {
                            "element_name": "key",
                            "element_description": "yellow key near the exit",
                            "element_mutation": "moved near the exit",
                        },
                    ],
                    "change_detected": False,
                }
            ),
        )
    )
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            max_frames_per_call=4,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=index,
            frame=Image.new("RGB", (2, 2), color=(index, index, index)),
        )
        for index in range(9)
    )

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=observations,
        previous_change_elements=(),
    )

    assert [len(call["images"]) for call in provider.calls] == [4, 4, 3]
    assert provider.calls[0]["previous_image"].getpixel((0, 0)) == (0, 0, 0)
    assert provider.calls[1]["previous_image"].getpixel((0, 0)) == (3, 3, 3)
    assert provider.calls[2]["previous_image"].getpixel((0, 0)) == (6, 6, 6)
    assert prompt_previous_change_elements(provider.calls[0]["prompt_text"]) == []
    assert prompt_previous_change_elements(provider.calls[1]["prompt_text"]) == [
        {
            "element_name": "door",
            "element_description": "closed red door",
            "element_mutation": "opened",
        }
    ]
    assert prompt_previous_change_elements(provider.calls[2]["prompt_text"]) == [
        {
            "element_name": "door",
            "element_description": "open red doorway",
            "element_mutation": "opened; camera followed it",
        },
        {
            "element_name": "key",
            "element_description": "yellow key",
            "element_mutation": "",
        },
    ]
    assert result.change_detected is True
    assert result.metadata["chunk_count"] == 3
    assert result.elements == (
        ChangeSummaryElement(
            element_name="door",
            element_description="red exit doorway",
            element_mutation="opened; camera followed it",
        ),
        ChangeSummaryElement(
            element_name="key",
            element_description="yellow key near the exit",
            element_mutation="moved near the exit",
        ),
    )


def test_change_adapter_balances_overlapping_frame_chunks() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            max_frames_per_call=10,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=index,
            frame=Image.new("RGB", (2, 2), color=(index, index, index)),
        )
        for index in range(22)
    )

    adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=observations,
        previous_change_elements=(),
    )

    assert [len(call["images"]) for call in provider.calls] == [8, 8, 8]
    assert provider.calls[0]["previous_image"].getpixel((0, 0)) == (0, 0, 0)
    assert provider.calls[1]["previous_image"].getpixel((0, 0)) == (7, 7, 7)
    assert provider.calls[2]["previous_image"].getpixel((0, 0)) == (14, 14, 14)
    assert provider.calls[-1]["current_image"].getpixel((0, 0)) == (21, 21, 21)


def test_change_adapter_inserts_diff_mask_between_two_frames() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_diff_mask=True,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous = Image.new("RGB", (2, 2), color=(0, 0, 0))
    current = previous.copy()
    current.putpixel((1, 0), (255, 0, 0))

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous),
        Observation(id="current", step=1, frame=current),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    previous_image, diff_mask, current_image = provider.calls[0]["images"]
    assert "Diff Mask Inputs" in adapter.last_instructions
    assert len(provider.calls[0]["images"]) == 3
    assert previous_image.getpixel((1, 0)) == (0, 0, 0)
    assert current_image.getpixel((1, 0)) == (255, 0, 0)
    assert diff_mask.getpixel((0, 0)) == (0, 0, 0)
    assert diff_mask.getpixel((1, 0)) == (255, 255, 255)
    assert provider.calls[0]["previous_image"] is previous_image
    assert provider.calls[0]["current_image"] is current_image


def test_change_adapter_includes_previous_change_elements_context() -> None:
    provider = RecordingChangeProvider(_change_payload("moved down"))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(2, 2),
            input_image_resample="nearest",
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )

    adapter.summarize(
        Observation(
            id="previous",
            step=0,
            frame=Image.new("RGB", (2, 2), color=(10, 20, 30)),
        ),
        Observation(
            id="current",
            step=1,
            frame=Image.new("RGB", (2, 2), color=(200, 210, 220)),
        ),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(
            ChangeSummaryElement(
                element_name="player",
                element_description="red square",
                element_mutation="moved right",
            ),
        ),
    )

    prompt = provider.calls[0]["prompt_text"]
    assert '"element_name": "player"' in prompt
    assert '"element_description": "red square"' in prompt
    assert '"element_mutation": "moved right"' in prompt


def test_change_adapter_applies_arc_grid_crop_before_resize() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(28, 28),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous_frame = Image.new("RGB", (64, 64), color=(10, 20, 30))
    current_frame = Image.new("RGB", (64, 64), color=(200, 210, 220))
    for x in range(64):
        previous_frame.putpixel((x, 0), (0, 255, 0))
        current_frame.putpixel((x, 0), (255, 0, 0))

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous_frame),
        Observation(id="current", step=1, frame=current_frame),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    previous_image = provider.calls[0]["previous_image"]
    current_image = provider.calls[0]["current_image"]
    assert previous_image.size == (28, 28)
    assert current_image.size == (28, 28)
    assert all(
        previous_image.getpixel((x, y)) == (10, 20, 30)
        for x in range(28)
        for y in range(28)
    )
    assert all(
        current_image.getpixel((x, y)) == (200, 210, 220)
        for x in range(28)
        for y in range(28)
    )


def test_change_adapter_does_not_recrop_marked_frame_bundle() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=index,
            frame=Image.new("RGB", (56, 56), color=(index, index, index)),
            metadata={"change_summary_crop_edges": (4, 4, 4, 4)},
        )
        for index in range(2)
    )

    adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=observations,
        previous_change_elements=(),
    )

    assert [image.size for image in provider.calls[0]["images"]] == [(56, 56)] * 2


def test_change_adapter_inserts_diff_masks_between_animation_frames() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_diff_mask=True,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    first = Image.new("RGB", (3, 3), color=(0, 0, 0))
    middle = first.copy()
    middle.putpixel((0, 0), (255, 0, 0))
    final = middle.copy()
    final.putpixel((2, 2), (0, 0, 255))
    observations = (
        Observation(id="first", step=0, frame=first),
        Observation(id="middle", step=1, frame=middle),
        Observation(id="final", step=2, frame=final),
    )

    adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=observations,
        previous_change_elements=(),
    )

    images = provider.calls[0]["images"]
    assert len(images) == 5
    first_image, first_mask, middle_image, second_mask, final_image = images
    assert first_image.getpixel((0, 0)) == (0, 0, 0)
    assert first_mask.getpixel((0, 0)) == (255, 255, 255)
    assert first_mask.getpixel((2, 2)) == (0, 0, 0)
    assert middle_image.getpixel((0, 0)) == (255, 0, 0)
    assert second_mask.getpixel((0, 0)) == (0, 0, 0)
    assert second_mask.getpixel((2, 2)) == (255, 255, 255)
    assert final_image.getpixel((2, 2)) == (0, 0, 255)


def test_change_adapter_normalizes_action6_coordinates_to_cropped_frame() -> None:
    provider = RecordingChangeProvider(_change_payload("The selected area changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(28, 28),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous_frame = Image.new("RGB", (64, 64), color=(10, 20, 30))
    current_frame = Image.new("RGB", (64, 64), color=(200, 210, 220))

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous_frame),
        Observation(id="current", step=1, frame=current_frame),
        ActionSpec(
            action_id="ACTION6",
            data={"x": 32, "y": 43},
            target="the lower middle tile",
        ),
        glossary_actions=(ActionSpec(action_id="ACTION6"),),
        previous_change_elements=(),
    )

    prompt = provider.calls[0]["prompt_text"]
    assert "action_id: ACTION6" in prompt
    assert 'data: {"x": 500, "y": 696}' in prompt
    assert "target: the lower middle tile" in prompt
    assert "coordinate_space: normalized_0_1000" in prompt


def test_change_adapter_resizes_animation_bundle_to_two_frame_budget() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation played."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(100, 80),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            max_frames_per_call=20,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    frame_observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=1,
            frame=Image.new("RGB", (64, 64), color=(index, index, index)),
        )
        for index in range(20)
    )

    adapter.summarize(
        frame_observations[0],
        frame_observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=frame_observations,
        previous_change_elements=(),
    )

    call = provider.calls[0]
    assert len(call["images"]) == 20
    assert [image.size for image in call["images"]] == [(31, 25)] * 20
    assert call["previous_image"].size == (31, 25)
    assert call["current_image"].size == (31, 25)


def test_change_adapter_uses_configured_animation_frame_budget_coefficient() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation played."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(100, 80),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            animation_frame_budget_coefficient=8,
            max_frames_per_call=20,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    frame_observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=1,
            frame=Image.new("RGB", (64, 64), color=(index, index, index)),
        )
        for index in range(20)
    )

    adapter.summarize(
        frame_observations[0],
        frame_observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=frame_observations,
        previous_change_elements=(),
    )

    call = provider.calls[0]
    assert len(call["images"]) == 20
    assert [image.size for image in call["images"]] == [(63, 50)] * 20
    assert call["previous_image"].size == (63, 50)
    assert call["current_image"].size == (63, 50)


def test_change_adapter_clamps_animation_frame_budget_coefficient_to_two() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation played."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(100, 80),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=4,
            animation_frame_budget_coefficient=1,
            max_frames_per_call=20,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    frame_observations = tuple(
        Observation(
            id=f"frame-{index}",
            step=1,
            frame=Image.new("RGB", (64, 64), color=(index, index, index)),
        )
        for index in range(20)
    )

    adapter.summarize(
        frame_observations[0],
        frame_observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=frame_observations,
        previous_change_elements=(),
    )

    call = provider.calls[0]
    assert len(call["images"]) == 20
    assert [image.size for image in call["images"]] == [(31, 25)] * 20
    assert call["previous_image"].size == (31, 25)
    assert call["current_image"].size == (31, 25)


def test_change_adapter_draws_cumulative_mask_edges_on_model_input_images() -> None:
    provider = RecordingChangeProvider(_change_payload("The changed pixel is marked."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(16, 16),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_bounding_boxes=True,
            dilation_bounding_boxes=1,
            width_bounding_boxes=1,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    current_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    current_frame.putpixel((3, 3), (255, 0, 0))

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous_frame),
        Observation(id="current", step=1, frame=current_frame),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    previous_image, current_image = provider.calls[0]["images"]
    assert "bright pink" in adapter.last_instructions
    assert "pink outlines" in adapter.last_instructions
    assert previous_image.size == (16, 16)
    assert current_image.size == (16, 16)
    for image in (previous_image, current_image):
        assert image.getpixel((6, 6)) == (255, 0, 255)
        assert image.getpixel((7, 7)) == (255, 0, 255)
    assert previous_image.getpixel((3, 3)) == (0, 0, 0)
    assert current_image.getpixel((3, 3)) == (0, 0, 0)


def test_change_adapter_draws_growing_cumulative_masks_on_animation_frames() -> None:
    provider = RecordingChangeProvider(_change_payload("The animation changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            activate_bounding_boxes=True,
            dilation_bounding_boxes=1,
            width_bounding_boxes=1,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    first = Image.new("RGB", (8, 8), color=(0, 0, 0))
    middle = first.copy()
    middle.putpixel((1, 1), (255, 0, 0))
    final = middle.copy()
    final.putpixel((6, 6), (0, 0, 255))
    observations = (
        Observation(id="first", step=0, frame=first),
        Observation(id="middle", step=1, frame=middle),
        Observation(id="final", step=2, frame=final),
    )

    adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=observations,
        previous_change_elements=(),
    )

    first_image, middle_image, final_image = provider.calls[0]["images"]
    assert first_image.getpixel((1, 1)) == (255, 0, 255)
    assert middle_image.getpixel((1, 1)) == (255, 0, 255)
    assert middle_image.getpixel((6, 6)) == (0, 0, 0)
    assert final_image.getpixel((1, 1)) == (255, 0, 255)
    assert final_image.getpixel((6, 6)) == (255, 0, 255)


def test_change_adapter_adds_independent_gaussian_noise_to_model_images() -> None:
    provider = RecordingChangeProvider(_change_payload("The frames changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            gaussian_noise_deviation=40.0,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    frame = Image.new("RGB", (8, 8), color=(128, 128, 128))
    observations = (
        Observation(id="first", step=0, frame=frame.copy()),
        Observation(id="middle", step=1, frame=frame.copy()),
        Observation(id="final", step=2, frame=frame.copy()),
    )

    adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        frame_observations=observations,
        previous_change_elements=(),
    )

    noisy_images = provider.calls[0]["images"]
    assert len(noisy_images) == 3
    assert all(image.size == (8, 8) for image in noisy_images)
    assert all(image.tobytes() != frame.tobytes() for image in noisy_images)
    assert len({image.tobytes() for image in noisy_images}) == 3


def test_change_adapter_blurs_model_images_before_noise() -> None:
    provider = RecordingChangeProvider(_change_payload("The frames changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            gaussian_blur_kernel_size=7,
            gaussian_noise_deviation=0.0,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    frame = Image.new("RGB", (9, 9), color=(0, 0, 0))
    frame.putpixel((4, 4), (255, 255, 255))
    before = frame.tobytes()

    adapter.summarize(
        Observation(id="previous", step=0, frame=frame),
        Observation(id="current", step=1, frame=frame.copy()),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    blurred = provider.calls[0]["previous_image"]
    assert frame.tobytes() == before
    assert blurred.tobytes() != before
    assert blurred.getpixel((4, 4))[0] < 255
    assert blurred.getpixel((4, 3))[0] > 0


def test_change_adapter_gaussian_noise_does_not_mutate_observation_frames() -> None:
    provider = RecordingChangeProvider(_change_payload("The frames changed."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            gaussian_noise_deviation=40.0,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )
    previous_frame = Image.new("RGB", (8, 8), color=(10, 20, 30))
    current_frame = Image.new("RGB", (8, 8), color=(200, 210, 220))
    previous_before = previous_frame.tobytes()
    current_before = current_frame.tobytes()

    adapter.summarize(
        Observation(id="previous", step=0, frame=previous_frame),
        Observation(id="current", step=1, frame=current_frame),
        ActionSpec(action_id="ACTION1"),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
        previous_change_elements=(),
    )

    assert previous_frame.tobytes() == previous_before
    assert current_frame.tobytes() == current_before
    assert provider.calls[0]["previous_image"].tobytes() != previous_before
    assert provider.calls[0]["current_image"].tobytes() != current_before


def test_change_adapter_rejects_invalid_gaussian_blur_kernel_size() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            gaussian_blur_kernel_size=4,
            gaussian_noise_deviation=0.0,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )

    with pytest.raises(ValueError, match="gaussian_blur_kernel_size"):
        adapter.summarize(
            Observation(id="previous", step=0, frame=Image.new("RGB", (8, 8))),
            Observation(id="current", step=1, frame=Image.new("RGB", (8, 8))),
            ActionSpec(action_id="ACTION1"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            previous_change_elements=(),
        )
    assert provider.calls == []


def test_change_adapter_rejects_invalid_gaussian_noise_deviation() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=None,
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=None,
            gaussian_noise_deviation=-1.0,
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )

    with pytest.raises(ValueError, match="gaussian_noise_deviation"):
        adapter.summarize(
            Observation(id="previous", step=0, frame=Image.new("RGB", (8, 8))),
            Observation(id="current", step=1, frame=Image.new("RGB", (8, 8))),
            ActionSpec(action_id="ACTION1"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            previous_change_elements=(),
        )
    assert provider.calls == []


def test_change_adapter_rejects_invalid_arc_grid_crop() -> None:
    provider = RecordingChangeProvider(_change_payload("The cursor moved."))
    adapter = ChangeSummaryAdapter(
        config=SimpleNamespace(
            backend="fake",
            input_image_size=(64, 64),
            input_image_resample="nearest",
            input_image_crop_arc_grid_edges=(32, 0, 32, 0),
            include_output_schema_in_instructions=False,
            repair_attempts=0,
        ),
        provider=provider,
    )

    with pytest.raises(ValueError, match="leaves no visible frame"):
        adapter.summarize(
            Observation(id="previous", step=0, frame=Image.new("RGB", (64, 64))),
            Observation(id="current", step=1, frame=Image.new("RGB", (64, 64))),
            ActionSpec(action_id="ACTION1"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            previous_change_elements=(),
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
    prompt = _action_prompt()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=prompt,
        previous_image=previous_image,
        current_image=current_image,
        output_schema=change_summary_json_schema(),
    )

    request = client.responses.calls[0]
    content = request["input"][0]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert "elements" in request["text"]["format"]["schema"]["required"]
    assert "change_detected" in request["text"]["format"]["schema"]["required"]
    assert result.text == payload
    assert content[0] == {"type": "input_text", "text": prompt}
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
    prompt = _action_prompt()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=prompt,
        previous_image=previous_image,
        current_image=current_image,
        output_schema=change_summary_json_schema(),
    )

    request = client.calls[0]
    user_message = request["messages"][1]
    assert "elements" in request["format"]["required"]
    assert "change_detected" in request["format"]["required"]
    assert result.text == payload
    assert user_message["content"] == prompt
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
        ),
        client=client,
    )
    previous_image = _previous_image()
    current_image = _current_image()
    prompt = _action_prompt()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=prompt,
        previous_image=previous_image,
        current_image=current_image,
        output_schema=change_summary_json_schema(),
    )

    request = client.chat.completions.calls[0]
    content = request["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert "elements" in request["response_format"]["json_schema"]["schema"]["required"]
    assert "change_detected" in (
        request["response_format"]["json_schema"]["schema"]["required"]
    )
    assert (
        request["response_format"]["json_schema"]["schema"]["properties"]["elements"][
            "type"
        ]
        == "array"
    )
    assert result.text == payload
    assert request["response_format"]["type"] == "json_schema"
    assert content[0] == {"type": "text", "text": prompt}
    assert len(images) == 2
    previous_payload = _image_from_data_url(images[0]["image_url"]["url"])
    current_payload = _image_from_data_url(images[1]["image_url"]["url"])
    assert previous_payload.size == (2, 2)
    assert current_payload.size == (2, 2)
    assert previous_payload.getpixel((0, 0)) == (10, 20, 30)
    assert current_payload.getpixel((0, 0)) == (200, 210, 220)
    records = drain_model_input_debug_records(provider)
    assert records[0]["call_slot"] == "change"


def test_vllm_change_provider_can_send_frame_sequence_as_video() -> None:
    payload = _change_payload("The animation played.")
    client = FakeVLLMClient(payload)
    provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct",
            input_image_size="2x2",
            frame_input_mode="video",
            video_fps=2.0,
            options={"thinking_token_budget": 1024},
        ),
        client=client,
    )
    previous_image = _previous_image()
    middle_image = Image.new("RGB", (2, 2), color=(100, 110, 120))
    current_image = _current_image()

    result = provider.complete(
        instructions_text="Return JSON.",
        prompt_text=_action_prompt(),
        previous_image=previous_image,
        current_image=current_image,
        images=(previous_image, middle_image, current_image),
        output_schema=change_summary_json_schema(),
    )

    request = client.chat.completions.calls[0]
    content = request["messages"][1]["content"]
    videos = [item for item in content if item["type"] == "video_url"]
    images = [item for item in content if item["type"] == "image_url"]
    video_metadata = request["extra_body"]["media_io_kwargs"]["video"]

    assert result.text == payload
    assert content[0] == {"type": "text", "text": _action_prompt()}
    assert images == []
    assert len(videos) == 1
    assert request["extra_body"]["thinking_token_budget"] == 1024
    assert video_metadata == {
        "fps": 2.0,
        "frames_indices": [0, 1, 2],
        "total_num_frames": 3,
        "duration": 1.5,
        "do_sample_frames": False,
    }

    prefix, encoded_frames_text = videos[0]["video_url"]["url"].split(",", 1)
    encoded_frames = encoded_frames_text.split(",")
    assert prefix == "data:video/jpeg;base64"
    assert len(encoded_frames) == 3
    assert [_image_from_base64_png(frame).size for frame in encoded_frames] == [
        (2, 2),
        (2, 2),
        (2, 2),
    ]


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
        prompt_text=_action_prompt(),
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
        prompt_text=_action_prompt(),
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
