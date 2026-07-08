"""Tests for bundled change-summary image inputs."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image
import pytest

from face_of_agi.contracts import ActionSpec, ChangeSummaryElement, Observation
from face_of_agi.models.change import (
    ChangeSummaryAdapter,
    ChangeSummaryProviderResponse,
    OllamaChangeSummaryConfig,
    OpenAIChangeSummaryConfig,
    build_change_summary_prompt,
    change_summary_elements_text,
    change_summary_observation_images,
    change_summary_json_schema,
    load_change_summary_instructions,
    parse_change_summary_output,
)
from face_of_agi.models.change.providers.ollama import OllamaChangeSummaryProvider
from face_of_agi.models.change.providers.openai import OpenAIChangeSummaryProvider
from face_of_agi.models.change.providers.vllm import VLLMChangeSummaryProvider
from face_of_agi.models.change.config import VLLMChangeSummaryConfig
from face_of_agi.models.change.components import (
    ARC_RENDERED_COLOR_NAMES_BY_RGB,
    arc_rendered_color_map,
    arc_rendered_color_name_map,
)


class FakeChangeSummaryProvider:
    backend = "fake"
    model = "fake-change"

    def __init__(
        self,
        text: str = (
            '{"elements": [{"element_name": "screen", '
            '"element_description": "whole screen", '
            '"element_mutation": "middle frame flashed white"}], '
            '"change_detected": true}'
        ),
        *,
        repair_text: str | None = None,
    ) -> None:
        self.text = text
        self.repair_text = repair_text
        self.calls: list[dict[str, Any]] = []
        self.repairs: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> ChangeSummaryProviderResponse:
        self.calls.append(kwargs)
        return ChangeSummaryProviderResponse(
            text=self.text,
            metadata={"backend": self.backend},
            request={"called": True},
        )

    def repair_complete(self, **kwargs: Any) -> ChangeSummaryProviderResponse:
        self.repairs.append(kwargs)
        if self.repair_text is None:
            raise AssertionError("repair should not be needed")
        return ChangeSummaryProviderResponse(
            text=self.repair_text,
            metadata={"backend": self.backend, "phase": "repair"},
            request={"repaired": True},
        )


def _observation(id_: str, color: tuple[int, int, int]) -> Observation:
    return Observation(
        id=id_,
        step=0,
        frame=Image.new("RGB", (4, 4), color=color),
    )


def test_change_summary_instructions_explain_zero_pixel_bundles() -> None:
    instructions = load_change_summary_instructions()

    assert "net" in instructions
    assert "first-to-final comparison" in instructions
    assert "visible transient animation" in instructions
    assert "transient element mutations" in instructions
    assert "targeted object or area as it appears in the first image" in instructions


def test_change_summary_attaches_full_bundle_without_schema_change() -> None:
    provider = FakeChangeSummaryProvider()
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=0,
        ),
        provider=provider,
    )
    first = _observation("first", (0, 0, 0))
    middle = _observation("middle", (255, 255, 255))
    final = _observation("final", (0, 0, 0))

    result = adapter.summarize(
        first,
        final,
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=(first, middle, final),
    )

    assert result.elements[0].element_mutation == "middle frame flashed white"
    assert result.changed_pixel_count == 0
    assert result.changed_pixel_percent == 0.0
    assert result.change_detected is True
    assert result.metadata["frame_count"] == 3
    assert result.metadata["any_adjacent_frame_changed"] is True
    assert "deterministic_change_detected" not in result.metadata
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert len(call["images"]) == 3
    assert call["previous_image"] is call["images"][0]
    assert call["current_image"] is call["images"][-1]
    assert call["output_schema"] == change_summary_json_schema()
    assert call["output_schema"]["required"] == ["elements", "change_detected"]
    assert "elements" in call["output_schema"]["properties"]
    assert "change_detected" in call["output_schema"]["properties"]
    assert "attached_frame_count: 3" in call["prompt_text"]
    assert "any_adjacent_frame_changed: true" in call["prompt_text"]
    assert "at most 20 visible elements" in call["prompt_text"]


def test_change_summary_schema_limits_element_count() -> None:
    schema = change_summary_json_schema()

    assert schema["properties"]["elements"]["maxItems"] == 20
    assert "adjacent attached frame pair" in (
        schema["properties"]["change_detected"]["description"]
    )


def test_change_summary_skips_two_frame_zero_change_after_normalized_crop() -> None:
    provider = FakeChangeSummaryProvider()
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=0,
            input_image_crop_box_normalized=(0.0, 0.0, 0.5, 0.5),
        ),
        provider=provider,
    )
    first = _observation("first", (0, 0, 0))
    final_image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    final_image.putpixel((3, 3), (255, 255, 255))
    final = Observation(id="final", step=1, frame=final_image)

    result = adapter.summarize(
        first,
        final,
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.elements == ()
    assert result.changed_pixel_count == 0
    assert result.changed_pixel_percent == 0.0
    assert result.change_detected is False
    assert result.metadata["skipped"] is True
    assert result.metadata["frame_count"] == 2
    assert result.metadata["any_adjacent_frame_changed"] is False
    assert "deterministic_change_detected" not in result.metadata
    assert provider.calls == []


def test_change_summary_autocorrects_change_detected_conflict() -> None:
    provider = FakeChangeSummaryProvider(
        text=(
            '{"elements": [{"element_name": "screen", '
            '"element_description": "whole screen", '
            '"element_mutation": "frame became white"}], '
            '"change_detected": false}'
        ),
    )
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=1,
        ),
        provider=provider,
    )

    result = adapter.summarize(
        _observation("first", (0, 0, 0)),
        _observation("final", (255, 255, 255)),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.elements[0].element_mutation == "frame became white"
    assert result.changed_pixel_count == 16
    assert result.changed_pixel_percent == 100.0
    assert result.change_detected is True
    assert result.metadata["repair_attempts"] == 0
    assert result.metadata["any_adjacent_frame_changed"] is True
    assert result.metadata["autocorrected_change_detected"] is True
    assert result.metadata["model_change_detected"] is False
    assert (
        result.metadata["autocorrect_reason"]
        == "boolean_mismatch_elements_consistent_with_change"
    )
    assert provider.repairs == []


def test_change_summary_repairs_direct_no_change_mismatch() -> None:
    provider = FakeChangeSummaryProvider(
        text=(
            '{"elements": [{"element_name": "screen", '
            '"element_description": "whole screen", '
            '"element_mutation": "nothing changed"}], '
            '"change_detected": false}'
        ),
        repair_text=(
            '{"elements": [{"element_name": "screen", '
            '"element_description": "whole screen", '
            '"element_mutation": "frame became white"}], '
            '"change_detected": true}'
        ),
    )
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=1,
        ),
        provider=provider,
    )

    result = adapter.summarize(
        _observation("first", (0, 0, 0)),
        _observation("final", (255, 255, 255)),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.elements[0].element_mutation == "frame became white"
    assert result.changed_pixel_count == 16
    assert result.changed_pixel_percent == 100.0
    assert result.change_detected is True
    assert result.metadata["repair_attempts"] == 1
    assert "autocorrected_change_detected" not in result.metadata
    assert len(provider.repairs) == 1
    assert "change_detected" in provider.repairs[0]["validation_error"]


def test_change_summary_falls_back_after_repair_exhaustion() -> None:
    provider = FakeChangeSummaryProvider(
        text=(
            '{"elements": [{"element_name": "screen", '
            '"element_description": "whole screen", '
            '"element_mutation": "nothing changed"}], '
            '"change_detected": false}'
        ),
        repair_text=(
            '{"elements": [{"element_name": "screen", '
            '"element_description": "whole screen", '
            '"element_mutation": "nothing changed"}], '
            '"change_detected": false}'
        ),
    )
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=1,
        ),
        provider=provider,
    )

    result = adapter.summarize(
        _observation("first", (0, 0, 0)),
        _observation("final", (255, 255, 255)),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.elements == ()
    assert result.changed_pixel_count == 16
    assert result.changed_pixel_percent == 100.0
    assert result.change_detected is True
    assert result.metadata["fallback"] == "repair_exhausted"
    assert result.metadata["any_adjacent_frame_changed"] is True
    assert "deterministic_change_detected" not in result.metadata
    assert len(provider.repairs) == 1


def test_change_summary_parser_accepts_elements_and_renames_duplicates() -> None:
    parsed = parse_change_summary_output(
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
                ],
                "change_detected": True,
            }
        )
    )

    assert [element.element_name for element in parsed.elements] == [
        "block_0",
        "block_1",
    ]


@pytest.mark.parametrize(
    "payload",
    (
        "{}",
        '{"summary": "old schema", "change_detected": true}',
        '{"elements": "not a list", "change_detected": true}',
        '{"elements": [{}], "change_detected": true}',
    ),
)
def test_change_summary_parser_rejects_invalid_payloads(payload: str) -> None:
    with pytest.raises(Exception):
        parse_change_summary_output(payload)


def test_change_summary_parser_rejects_too_many_elements() -> None:
    element = {
        "element_name": "tile",
        "element_description": "colored tile",
        "element_mutation": "changed",
    }
    payload = json.dumps(
        {
            "elements": [element for _index in range(21)],
            "change_detected": True,
        }
    )

    with pytest.raises(Exception, match="too many elements"):
        parse_change_summary_output(payload)


def test_change_summary_elements_text_renders_bullets() -> None:
    assert (
        change_summary_elements_text(
            (
                ChangeSummaryElement(
                    element_name="cursor",
                    element_description="small white square",
                    element_mutation="moved right",
                ),
                ChangeSummaryElement(
                    element_name="wall",
                    element_description="blue vertical wall",
                    element_mutation="",
                ),
            )
        )
        == "- cursor: small white square; mutations: moved right\n"
        "- wall: blue vertical wall; mutations: no detected changes for this element"
    )


def test_change_summary_chunks_long_animation_and_merges_elements() -> None:
    provider = FakeChangeSummaryProvider(
        text=(
            '{"elements": [{"element_name": "cursor", '
            '"element_description": "small bright cursor", '
            '"element_mutation": "moved in chunk"}], '
            '"change_detected": true}'
        )
    )
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=0,
            max_frames_per_call=3,
        ),
        provider=provider,
    )
    observations = tuple(
        _observation(f"frame-{index}", (index * 30, index * 30, index * 30))
        for index in range(5)
    )

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert len(provider.calls) == 2
    assert result.metadata["chunk_count"] == 2
    assert result.elements[0].element_name == "cursor"
    assert result.elements[0].element_mutation == "moved in chunk"


def test_change_summary_can_persist_only_changed_elements() -> None:
    provider = FakeChangeSummaryProvider(
        text=(
            '{"elements": ['
            '{"element_name": "cursor", '
            '"element_description": "small bright cursor", '
            '"element_mutation": "moved right"}, '
            '{"element_name": "wall", '
            '"element_description": "gray wall", '
            '"element_mutation": "no visible changes"}], '
            '"change_detected": true}'
        )
    )
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=0,
            persist_changed_elements_only=True,
        ),
        provider=provider,
    )

    result = adapter.summarize(
        _observation("first", (0, 0, 0)),
        _observation("final", (255, 255, 255)),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert [element.element_name for element in result.elements] == ["cursor"]
    assert result.metadata["persist_changed_elements_only"] is True
    assert result.metadata["element_count_before_persist_filter"] == 2
    assert result.metadata["element_count_after_persist_filter"] == 1


def test_change_summary_component_facts_are_added_for_native_grids() -> None:
    provider = FakeChangeSummaryProvider()
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=0,
            activate_components=True,
        ),
        provider=provider,
    )
    first_grid = [[0 for _x in range(64)] for _y in range(64)]
    final_grid = [[0 for _x in range(64)] for _y in range(64)]
    final_grid[10][10] = 1

    adapter.summarize(
        Observation(id="first", step=0, frame=first_grid),
        Observation(id="final", step=1, frame=final_grid),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    prompt = provider.calls[0]["prompt_text"]
    instructions = provider.calls[0]["instructions_text"]
    assert "## Frame components" in prompt
    assert "frame 0:" in prompt
    assert "color=white" in prompt
    assert "color=light_gray" in prompt
    assert "symbol=" not in prompt
    assert "rgb(" not in prompt
    assert "ARC rendered color legend" not in instructions


def test_change_summary_component_color_names_match_renderer_palette() -> None:
    rendered_colors = set(arc_rendered_color_map().values())

    assert rendered_colors == set(ARC_RENDERED_COLOR_NAMES_BY_RGB)
    assert arc_rendered_color_name_map()[0] == "white"
    assert arc_rendered_color_name_map()[9] == "blue"
    assert arc_rendered_color_name_map()[15] == "purple"


def test_change_summary_component_failure_is_fatal_when_enabled() -> None:
    provider = FakeChangeSummaryProvider()
    adapter = ChangeSummaryAdapter(
        OllamaChangeSummaryConfig(
            input_image_size=None,
            frame_scale=1,
            repair_attempts=0,
            activate_components=True,
        ),
        provider=provider,
    )

    with pytest.raises(ValueError, match="component extraction requires"):
        adapter.summarize(
            _observation("first", (1, 2, 3)),
            _observation("final", (4, 5, 6)),
            ActionSpec("ACTION1"),
            glossary_actions=(ActionSpec("ACTION1"),),
        )


def test_change_summary_prompt_renders_action6_normalized_target_data() -> None:
    prompt = build_change_summary_prompt(
        ActionSpec(
            "ACTION6",
            data={"x": 32, "y": 16},
            target="upper middle tile",
        ),
        changed_pixel_count=4,
        frame_count=2,
    )

    assert "action_id: ACTION6" in prompt
    assert 'data: {"x": 500, "y": 250}' in prompt
    assert 'target: "upper middle tile"' in prompt
    assert "coordinate_space: normalized_0_1000" in prompt


def test_change_summary_bundle_budget_allows_four_input_sized_images() -> None:
    observations = tuple(
        _observation(f"frame-{index}", (index, index, index))
        for index in range(8)
    )

    images = change_summary_observation_images(
        observations,
        frame_scale=1,
        size=(100, 100),
        resample="nearest",
        crop_box_normalized=None,
    )

    assert {image.size for image in images} == {(70, 70)}


def test_change_providers_prefer_full_bundle_payload() -> None:
    images = (
        Image.new("RGB", (1, 1), color=(0, 0, 0)),
        Image.new("RGB", (1, 1), color=(127, 127, 127)),
        Image.new("RGB", (1, 1), color=(255, 255, 255)),
    )
    openai_provider = OpenAIChangeSummaryProvider(
        OpenAIChangeSummaryConfig(model="fake-openai"),
        client=object(),
    )
    ollama_provider = OllamaChangeSummaryProvider(
        OllamaChangeSummaryConfig(model="fake-ollama"),
        client=object(),
    )
    vllm_provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(model="fake-vllm"),
        client=object(),
    )

    openai_item = openai_provider._input_item(
        "prompt",
        previous_image=images[0],
        current_image=images[-1],
        images=images,
    )
    ollama_message = ollama_provider._user_message(
        "prompt",
        previous_image=images[0],
        current_image=images[-1],
        images=images,
    )
    vllm_message = vllm_provider._user_message(
        "prompt",
        previous_image=images[0],
        current_image=images[-1],
        images=images,
    )

    openai_image_items = [
        item for item in openai_item["content"] if item.get("type") == "input_image"
    ]
    vllm_image_items = [
        item for item in vllm_message["content"] if item.get("type") == "image_url"
    ]

    assert len(openai_image_items) == 3
    assert len(ollama_message["images"]) == 3
    assert len(vllm_image_items) == 3
