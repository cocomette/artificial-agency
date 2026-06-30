"""Tests for vLLM change summary prompts."""

from __future__ import annotations

from typing import Any

from PIL import Image
import pytest

from face_of_agi.contracts import ActionSpec, Observation
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.debug.sanitize import sanitize_for_debug
from face_of_agi.models.change import (
    ChangeSummaryAdapter,
    ChangeSummaryOutputError,
    ChangeSummaryProviderResponse,
    change_summary_json_schema,
    parse_change_summary_output,
)
from face_of_agi.models.change.config import VLLMChangeSummaryConfig
from face_of_agi.models.change.providers.vllm import VLLMChangeSummaryProvider


class FakeChangeSummaryProvider:
    backend = "fake"
    model = "fake-change"

    def __init__(
        self,
        responses: list[str] | None = None,
        repair_responses: list[str] | None = None,
        reduce_responses: list[str] | None = None,
        repair_reduce_responses: list[str] | None = None,
    ) -> None:
        self.responses = list(responses) if responses is not None else [
            '{"summary": "symbol 2 appeared", "change_detected": true}'
        ]
        self.repair_responses = (
            list(repair_responses) if repair_responses is not None else []
        )
        self.reduce_responses = (
            list(reduce_responses)
            if reduce_responses is not None
            else ['{"summary": "reduced summary", "change_detected": true}']
        )
        self.repair_reduce_responses = (
            list(repair_reduce_responses)
            if repair_reduce_responses is not None
            else []
        )
        self.calls: list[dict[str, Any]] = []
        self.repair_calls: list[dict[str, Any]] = []
        self.reduce_calls: list[dict[str, Any]] = []
        self.repair_reduce_calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> ChangeSummaryProviderResponse:
        self.calls.append(kwargs)
        return ChangeSummaryProviderResponse(
            text=self.responses.pop(0),
            metadata={"backend": self.backend},
            request={"messages": [{"role": "user", "content": kwargs["prompt_text"]}]},
        )

    def repair_complete(self, **kwargs: Any) -> ChangeSummaryProviderResponse:
        self.repair_calls.append(kwargs)
        if not self.repair_responses:
            raise AssertionError("repair should not be needed")
        return ChangeSummaryProviderResponse(
            text=self.repair_responses.pop(0),
            metadata={"backend": self.backend, "repair": True},
            request={"messages": [{"role": "user", "content": kwargs["prompt_text"]}]},
        )

    def reduce_complete(self, **kwargs: Any) -> ChangeSummaryProviderResponse:
        self.reduce_calls.append(kwargs)
        return ChangeSummaryProviderResponse(
            text=self.reduce_responses.pop(0),
            metadata={"backend": self.backend, "phase": "reduce_complete"},
            request={"messages": [{"role": "user", "content": kwargs["prompt_text"]}]},
        )

    def repair_reduce_complete(self, **kwargs: Any) -> ChangeSummaryProviderResponse:
        self.repair_reduce_calls.append(kwargs)
        if not self.repair_reduce_responses:
            raise AssertionError("reducer repair should not be needed")
        return ChangeSummaryProviderResponse(
            text=self.repair_reduce_responses.pop(0),
            metadata={
                "backend": self.backend,
                "phase": "repair_reduce_complete",
                "repair": True,
            },
            request={"messages": [{"role": "user", "content": kwargs["prompt_text"]}]},
        )


def _grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]


def _observation(id_: str, frame: list[list[int]], step: int = 0) -> Observation:
    return Observation(id=id_, step=step, frame=frame)


def _assert_no_stale_text_prompt_terms(text: str) -> None:
    lower_text = text.lower()
    stale_terms = (
        "attached image",
        "attached images",
        "attached frame",
        "current image frame",
        "0..1000",
        "0 to 1000",
    )
    for term in stale_terms:
        assert term not in lower_text


def _assert_no_media_payload_terms(text: str) -> None:
    lower_text = text.lower()
    media_terms = ("base64", "data:image", "image_url", "image url", "images:")
    for term in media_terms:
        assert term not in lower_text


def _text_part(content: Any) -> str:
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    return content[0]["text"]


def _image_parts(content: Any) -> list[dict[str, Any]]:
    assert isinstance(content, list)
    return [part for part in content if part.get("type") == "image_url"]


def test_change_summary_schema_includes_summary_max_length() -> None:
    schema = change_summary_json_schema(summary_max_chars=123)

    assert schema["properties"]["summary"]["maxLength"] == 123


def test_change_summary_parser_rejects_oversized_summary() -> None:
    text = '{"summary": "' + ("x" * 6) + '", "change_detected": true}'

    with pytest.raises(ChangeSummaryOutputError, match="too long"):
        parse_change_summary_output(text, summary_max_chars=5)


def test_change_summary_sends_observation_text_without_images() -> None:
    provider = FakeChangeSummaryProvider()
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(model="fake-vllm", repair_attempts=0),
        provider=provider,
    )
    first_grid = _grid()
    final_grid = _grid()
    final_grid[10][11] = 2

    result = adapter.summarize(
        _observation("first", first_grid),
        _observation("final", final_grid, step=1),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.summary == "symbol 2 appeared"
    assert result.changed_pixel_count == 1
    assert result.change_detected is True
    assert result.changed_cell_percent is not None
    assert result.changed_cell_percent > 0
    call = provider.calls[0]
    assert "images" in call
    assert len(call["images"]) == 2
    assert [image.size for image in call["images"]] == [(2048, 2048), (2048, 2048)]
    assert "previous_image" not in call
    assert "current_image" not in call
    _assert_no_stale_text_prompt_terms(call["instructions_text"])
    _assert_no_stale_text_prompt_terms(call["prompt_text"])
    assert call["output_schema"] == change_summary_json_schema()
    assert "OBSERVATIONS:" in call["prompt_text"]
    assert "## change_evidence_observations\n\n### frame 0" in call["prompt_text"]
    assert "x_range: 3..60" in call["prompt_text"]
    assert "observation_id:" not in call["prompt_text"]
    assert "crop_bounds_original_xyxy:" not in call["prompt_text"]
    assert "coordinate_system:" not in call["prompt_text"]
    assert "symbols:" not in call["prompt_text"]
    assert "ARC color symbols" not in call["prompt_text"]
    assert "shape, colors" not in call["instructions_text"]
    assert "ARC color glossary" in call["instructions_text"]
    assert "0=white" not in call["instructions_text"]
    assert "A=cyan" not in call["instructions_text"]
    assert "symbol A (cyan)" not in call["instructions_text"]
    assert "canonical glossary colors" not in call["instructions_text"]
    assert "symbol A: light cyan" in call["instructions_text"]
    assert "symbol 0" in call["instructions_text"]
    assert "symbol F" in call["instructions_text"]
    assert "A-cells" in call["instructions_text"]
    assert "no symbol legend is provided" not in call["instructions_text"]
    assert "frame-local labels" in call["instructions_text"]
    assert "persistent object identity" in call["instructions_text"]
    assert "y=10:" in call["prompt_text"]


def test_change_summary_skips_two_frame_zero_change_inside_crop() -> None:
    provider = FakeChangeSummaryProvider()
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(model="fake-vllm", repair_attempts=0),
        provider=provider,
    )
    first_grid = _grid()
    final_grid = _grid()
    final_grid[0][0] = 9

    result = adapter.summarize(
        _observation("first", first_grid),
        _observation("final", final_grid, step=1),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.summary == "no changes"
    assert result.changed_pixel_count == 0
    assert result.change_detected is False
    assert result.changed_cell_percent == 0.0
    assert result.metadata["skip_reason"] == "zero_changed_cells"
    assert provider.calls == []


def test_change_summary_chunks_large_frame_bundle_with_overlap() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "first chunk changed", "change_detected": true}',
            '{"summary": "middle chunk changed", "change_detected": true}',
            '{"summary": "final chunk changed", "change_detected": true}',
        ],
        reduce_responses=[
            '{"summary": "reduced chronological change", "change_detected": true}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=0,
            max_frames_per_call=5,
        ),
        provider=provider,
    )
    observations = []
    for index in range(11):
        frame = _grid()
        frame[10][10] = index
        observations.append(_observation(f"f{index}", frame, step=index))

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.changed_pixel_count == 1
    assert result.change_detected is True
    assert result.metadata["serialized_frame_count"] == 11
    assert result.metadata["source_frame_count"] == 11
    assert result.metadata["chunk_count"] == 3
    assert result.metadata["reducer"] is True
    assert result.metadata["reducer_keyframe_indices"] == (0, 4, 7, 10)
    assert len(provider.calls) == 3
    assert len(provider.reduce_calls) == 1
    assert result.summary == "reduced chronological change"
    prompt = provider.calls[0]["prompt_text"]
    assert "serialized_frame_count: 5" in prompt
    assert "source_filtered_frame_count:" not in prompt
    assert "chunk_index:" not in prompt
    assert "chunk_count:" not in prompt
    assert "any_adjacent_frame_changed: true" in prompt
    assert "observation_id:" not in prompt
    assert "chunk_count" not in provider.calls[0]["instructions_text"]
    assert "overlapping chunk" not in provider.calls[0]["instructions_text"]
    second_prompt = provider.calls[1]["prompt_text"]
    assert "serialized_frame_count: 4" in second_prompt
    assert "chunk_index:" not in second_prompt
    assert "observation_id:" not in prompt
    reducer_prompt = provider.reduce_calls[0]["prompt_text"]
    assert "ORDERED_PARTIAL_SUMMARIES:" in reducer_prompt
    assert "selected_frame_indices: 0, 4, 7, 10" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=0" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=10" in reducer_prompt
    assert "#### components" not in reducer_prompt
    assert "### component deltas" not in reducer_prompt
    assert "### frame deltas" not in reducer_prompt
    _assert_no_media_payload_terms(reducer_prompt)


def test_change_summary_reducer_disabled_preserves_deterministic_merge() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "first chunk changed", "change_detected": true}',
            '{"summary": "final chunk changed", "change_detected": true}',
        ]
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=0,
            max_frames_per_call=3,
            reduce_chunk_summaries=False,
        ),
        provider=provider,
    )
    observations = []
    for index in range(5):
        frame = _grid()
        frame[10][10] = index
        observations.append(_observation(f"f{index}", frame, step=index))

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.summary == "first chunk changed. final chunk changed."
    assert result.metadata["chunk_count"] == 2
    assert "reducer" not in result.metadata
    assert len(provider.calls) == 2
    assert provider.reduce_calls == []


def test_change_summary_single_chunk_does_not_invoke_reducer() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "single chunk changed", "change_detected": true}',
        ]
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=0,
            max_frames_per_call=8,
        ),
        provider=provider,
    )
    observations = []
    for index in range(4):
        frame = _grid()
        frame[10][10] = index
        observations.append(_observation(f"f{index}", frame, step=index))

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.summary == "single chunk changed"
    assert result.metadata["chunk_count"] == 1
    assert len(provider.calls) == 1
    assert provider.reduce_calls == []


def test_change_summary_reducer_keyframes_respect_limit() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "chunk changed", "change_detected": true}'
            for _index in range(8)
        ],
        reduce_responses=[
            '{"summary": "limited keyframes reduced", "change_detected": true}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=0,
            max_frames_per_call=3,
            reducer_keyframe_limit=4,
        ),
        provider=provider,
    )
    observations = []
    for index in range(17):
        frame = _grid()
        frame[10][10] = index % 16
        observations.append(_observation(f"f{index}", frame, step=index))

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.metadata["chunk_count"] == 8
    assert result.metadata["reducer_keyframe_indices"] == (0, 2, 14, 16)
    reducer_prompt = provider.reduce_calls[0]["prompt_text"]
    assert "selected_frame_indices: 0, 2, 14, 16" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=0" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=2" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=14" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=16" in reducer_prompt
    assert "## reducer_keyframe original_frame_index=4" not in reducer_prompt


def test_change_summary_does_not_skip_zero_net_change_with_intermediate_frames() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "symbol appeared", "change_detected": true}',
            '{"summary": "symbol disappeared", "change_detected": true}',
        ]
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=0,
            max_frames_per_call=2,
            reduce_chunk_summaries=False,
        ),
        provider=provider,
    )
    first_grid = _grid()
    transient_grid = _grid()
    transient_grid[10][10] = 1
    final_grid = _grid()
    observations = (
        _observation("first", first_grid),
        _observation("transient", transient_grid),
        _observation("final", final_grid, step=1),
    )

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.changed_pixel_count == 0
    assert result.change_detected is True
    assert result.summary == "symbol appeared. symbol disappeared."
    assert len(provider.calls) == 2
    assert provider.reduce_calls == []
    prompt = provider.calls[0]["prompt_text"]
    assert "serialized_frame_count: 2" in prompt
    assert "source_filtered_frame_count:" not in prompt
    assert "changed_cell_count: 1" in prompt


def test_change_summary_reducer_autocorrects_conflicting_change_detected() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "first chunk changed", "change_detected": true}',
            '{"summary": "final chunk changed", "change_detected": true}',
        ],
        reduce_responses=[
            '{"summary": "reduced chronological change", "change_detected": false}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=1,
            max_frames_per_call=3,
        ),
        provider=provider,
    )
    observations = []
    for index in range(5):
        frame = _grid()
        frame[10][10] = index
        observations.append(_observation(f"f{index}", frame, step=index))

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.summary == "reduced chronological change"
    assert result.change_detected is True
    assert result.metadata["reducer_repair_attempts"] == 0
    assert result.metadata["autocorrected_change_detected"] is True
    assert result.metadata["model_change_detected"] is False
    assert (
        result.metadata["autocorrect_reason"]
        == "boolean_mismatch_summary_consistent_with_change"
    )
    assert len(provider.reduce_calls) == 1
    assert len(provider.repair_reduce_calls) == 0


def test_change_summary_reducer_repair_exhaustion_uses_deterministic_merge() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "first chunk changed", "change_detected": true}',
            '{"summary": "final chunk changed", "change_detected": true}',
        ],
        reduce_responses=[
            '{"summary": "No visible playfield change occurred.", '
            '"change_detected": false}',
        ],
        repair_reduce_responses=[
            '{"summary": "No visible playfield change occurred.", '
            '"change_detected": false}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=1,
            max_frames_per_call=3,
        ),
        provider=provider,
    )
    observations = []
    for index in range(5):
        frame = _grid()
        frame[10][10] = index
        observations.append(_observation(f"f{index}", frame, step=index))

    result = adapter.summarize(
        observations[0],
        observations[-1],
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
        frame_observations=observations,
    )

    assert result.summary == "first chunk changed. final chunk changed."
    assert result.change_detected is True
    assert result.metadata["reducer_fallback"] == "repair_exhausted"
    assert len(provider.repair_reduce_calls) == 1


def test_change_summary_autocorrects_conflicting_change_detected() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "symbol 2 temporarily appeared then reverted", '
            '"change_detected": false}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(model="fake-vllm", repair_attempts=1),
        provider=provider,
    )
    first_grid = _grid()
    final_grid = _grid()
    final_grid[10][11] = 2

    result = adapter.summarize(
        _observation("first", first_grid),
        _observation("final", final_grid, step=1),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.summary == "symbol 2 temporarily appeared then reverted"
    assert result.change_detected is True
    assert result.metadata["repair_attempts"] == 0
    assert result.metadata["autocorrected_change_detected"] is True
    assert result.metadata["model_change_detected"] is False
    assert (
        result.metadata["autocorrect_reason"]
        == "boolean_mismatch_summary_consistent_with_change"
    )
    assert len(provider.repair_calls) == 0


def test_change_summary_repairs_direct_no_change_mismatch() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "No visible playfield change occurred.", '
            '"change_detected": false}',
        ],
        repair_responses=[
            '{"summary": "symbol 2 appeared", "change_detected": true}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(model="fake-vllm", repair_attempts=1),
        provider=provider,
    )
    first_grid = _grid()
    final_grid = _grid()
    final_grid[10][11] = 2

    result = adapter.summarize(
        _observation("first", first_grid),
        _observation("final", final_grid, step=1),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.summary == "symbol 2 appeared"
    assert result.change_detected is True
    assert result.metadata["repair_attempts"] == 1
    assert "autocorrected_change_detected" not in result.metadata
    assert len(provider.repair_calls) == 1


def test_change_summary_repair_exhaustion_uses_deterministic_fallback() -> None:
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "No visible playfield change occurred.", '
            '"change_detected": false}',
        ],
        repair_responses=[
            '{"summary": "No visible playfield change occurred.", '
            '"change_detected": false}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(model="fake-vllm", repair_attempts=1),
        provider=provider,
    )
    first_grid = _grid()
    final_grid = _grid()
    final_grid[10][11] = 2

    result = adapter.summarize(
        _observation("first", first_grid),
        _observation("final", final_grid, step=1),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.summary == "Visible changes occurred, but summary unavailable."
    assert result.change_detected is True
    assert result.metadata["fallback"] == "repair_exhausted"
    assert len(provider.repair_calls) == 1


def test_change_summary_oversized_output_repairs_then_falls_back() -> None:
    oversized_summary = "x" * 12
    provider = FakeChangeSummaryProvider(
        responses=[
            '{"summary": "' + oversized_summary + '", "change_detected": true}',
        ],
        repair_responses=[
            '{"summary": "' + oversized_summary + '", "change_detected": true}',
        ],
    )
    adapter = ChangeSummaryAdapter(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_attempts=1,
            summary_max_chars=5,
        ),
        provider=provider,
    )
    first_grid = _grid()
    final_grid = _grid()
    final_grid[10][11] = 2

    result = adapter.summarize(
        _observation("first", first_grid),
        _observation("final", final_grid, step=1),
        ActionSpec("ACTION1"),
        glossary_actions=(ActionSpec("ACTION1"),),
    )

    assert result.summary == "Visible changes occurred, but summary unavailable."
    assert result.change_detected is True
    assert result.metadata["fallback"] == "repair_exhausted"
    assert len(provider.repair_calls) == 1


def test_vllm_change_provider_uses_multimodal_chat_message() -> None:
    provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(model="fake-vllm"),
        client=object(),
    )

    message = provider._user_message(
        "serialized observations",
        images=(Image.new("RGB", (2, 2)),),
    )

    assert message["role"] == "user"
    assert _text_part(message["content"]) == "serialized observations"
    image_parts = _image_parts(message["content"])
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


class FakeVLLMCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "id": f"response-{len(self.calls)}",
            "model": kwargs["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"summary": "ok", "change_detected": true}',
                    }
                }
            ],
        }


class FakeVLLMClient:
    def __init__(self) -> None:
        self.chat = type(
            "FakeChat",
            (),
            {"completions": FakeVLLMCompletions()},
        )()


def test_vllm_change_provider_clips_invalid_output_in_repair_prompt() -> None:
    provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(
            model="fake-vllm",
            repair_invalid_output_preview_chars=80,
        ),
        client=FakeVLLMClient(),
    )
    invalid_text = "a" * 120 + "TAIL"

    provider.repair_complete(
        instructions_text="instructions",
        prompt_text="prompt",
        images=(),
        output_schema=change_summary_json_schema(),
        invalid_text=invalid_text,
        validation_error="bad",
        attempt=1,
    )

    request = provider._client._client.chat.completions.calls[0]
    repair_text = _text_part(request["messages"][1]["content"])
    assert "Invalid output preview:" in repair_text
    assert "omitted" in repair_text
    assert "TAIL" in repair_text
    assert invalid_text not in repair_text


def test_vllm_change_provider_captures_reducer_phases() -> None:
    provider = VLLMChangeSummaryProvider(
        VLLMChangeSummaryConfig(model="fake-vllm"),
        client=FakeVLLMClient(),
    )

    provider.reduce_complete(
        instructions_text="instructions",
        prompt_text="prompt",
        images=(Image.new("RGB", (2, 2)),),
        output_schema=change_summary_json_schema(),
    )
    provider.repair_reduce_complete(
        instructions_text="instructions",
        prompt_text="prompt",
        images=(Image.new("RGB", (2, 2)),),
        output_schema=change_summary_json_schema(),
        invalid_text="{}",
        validation_error="bad",
        attempt=1,
    )

    records = drain_model_input_debug_records(provider)
    assert [record["phase"] for record in records] == [
        "reduce_complete",
        "repair_reduce_complete",
    ]
    assert records[1]["attempt"] == 1
    raw_url = records[0]["request"]["messages"][1]["content"][1]["image_url"]["url"]
    assert raw_url.startswith("data:image/png;base64,")

    sanitized = sanitize_for_debug(records[0])
    sanitized_url = sanitized["request"]["messages"][1]["content"][1]["image_url"]["url"]
    assert sanitized_url["kind"] == "omitted_image_data_url"
    assert sanitized_url["mime_type"] == "image/png"
