"""Provider-neutral transition change summary adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Sequence

from face_of_agi.contracts import ActionSpec, Observation
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.color_glossary import append_arc_color_glossary
from face_of_agi.models.change.config import VLLMChangeSummaryConfig
from face_of_agi.models.change.contracts import (
    ChangeSummaryProvider,
    ChangeSummaryResult,
    DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    change_summary_json_schema,
)
from face_of_agi.models.observation_text import (
    ObservationTextConfig,
    cropped_changed_cell_count,
    serialize_observation,
    serialize_observations,
)
from face_of_agi.models.image_inputs import observations_to_cropped_images
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "instruction_prompt.md"
)
DEFAULT_REDUCER_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "reducer_instruction_prompt.md"
)
CHANGE_SUMMARY_PROMPT = "Compare the serialized observation frames from oldest to newest."
CHANGE_SUMMARY_REDUCER_PROMPT = (
    "Reconcile the ordered partial transition summaries into one final summary."
)
LOGGER = logging.getLogger(__name__)


class ChangeSummaryOutputError(RuntimeError):
    """Raised when a change summary backend returns invalid output."""


@dataclass(frozen=True, slots=True)
class ParsedChangeSummary:
    """Validated structured change-summary payload."""

    summary: str
    change_detected: bool
    model_change_detected: bool | None = None
    autocorrected_change_detected: bool = False
    autocorrect_reason: str | None = None


class ChangeSummaryAdapter:
    """Model role that summarizes visual change between consecutive frames."""

    def __init__(
        self,
        config: Any | None = None,
        *,
        provider: ChangeSummaryProvider | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config or VLLMChangeSummaryConfig()
        self.provider = provider or self._default_provider(client=client)
        self.last_instructions: str | None = None
        self.last_prompt: str | None = None
        self.last_request: dict[str, Any] | None = None

    def summarize(
        self,
        previous_observation: Observation,
        current_observation: Observation,
        action: ActionSpec,
        *,
        glossary_actions: Sequence[ActionSpec],
        frame_observations: Sequence[Observation] | None = None,
    ) -> ChangeSummaryResult:
        """Return a compact description of visible change between observations."""

        self.last_instructions = None
        self.last_prompt = None
        self.last_request = None
        evidence_observations = tuple(
            frame_observations
            if frame_observations is not None
            else (previous_observation, current_observation)
        )
        if len(evidence_observations) < 2:
            raise ValueError("change summary requires at least two evidence frames")
        source_frame_count = len(evidence_observations)
        changed_pixel_count = cropped_changed_cell_count(
            evidence_observations[0].frame,
            evidence_observations[-1].frame,
            config=self.config.observation_text,
        )
        changed_cell_percent = cropped_changed_cell_percent(
            evidence_observations[0].frame,
            evidence_observations[-1].frame,
            changed_cell_count=changed_pixel_count,
            config=self.config.observation_text,
        )
        deterministic_change_detected = any_cropped_change_detected(
            evidence_observations,
            config=self.config.observation_text,
        )
        if not deterministic_change_detected:
            return ChangeSummaryResult(
                summary="no changes",
                changed_pixel_count=0,
                change_detected=False,
                metadata={
                    "skipped": True,
                    "skip_reason": "zero_changed_cells",
                    "frame_count": len(evidence_observations),
                    "serialized_frame_count": len(evidence_observations),
                    "source_frame_count": source_frame_count,
                    "chunk_count": 0,
                    "deterministic_change_detected": False,
                    "any_adjacent_frame_changed": False,
                },
                changed_cell_percent=0.0,
            )

        summary_max_chars = getattr(
            self.config,
            "summary_max_chars",
            DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
        )
        output_schema = change_summary_json_schema(
            summary_max_chars=summary_max_chars,
        )
        instructions = append_output_schema_to_instructions(
            append_arc_color_glossary(
                append_action_glossary(
                    load_change_summary_instructions(),
                    glossary_actions,
                    mode="committed_action",
                    observation_text_config=self.config.observation_text,
                )
            ),
            output_schema,
            include=bool(
                getattr(self.config, "include_output_schema_in_instructions", False)
            ),
        )
        self.last_instructions = instructions
        chunks = _overlapping_observation_chunks(
            evidence_observations,
            max_frames_per_call=getattr(self.config, "max_frames_per_call", None),
        )
        chunk_results = tuple(
            self._summarize_observation_chunk(
                observations=chunk,
                action=action,
                output_schema=output_schema,
                instructions=instructions,
                chunk_index=index,
                chunk_count=len(chunks),
                source_frame_count=source_frame_count,
            )
            for index, chunk in enumerate(chunks)
        )
        if len(chunk_results) == 1:
            result = chunk_results[0]
            return ChangeSummaryResult(
                summary=result.summary,
                changed_pixel_count=changed_pixel_count,
                change_detected=deterministic_change_detected,
                metadata={
                    **result.metadata,
                    "frame_count": len(evidence_observations),
                    "serialized_frame_count": len(evidence_observations),
                    "source_frame_count": source_frame_count,
                    "chunk_count": 1,
                    "deterministic_change_detected": deterministic_change_detected,
                },
                changed_cell_percent=changed_cell_percent,
            )
        merged_result = _merged_change_summary_result(
            chunk_results,
            changed_pixel_count=changed_pixel_count,
            changed_cell_percent=changed_cell_percent,
            change_detected=deterministic_change_detected,
            frame_count=len(evidence_observations),
            source_frame_count=source_frame_count,
        )
        if not getattr(self.config, "reduce_chunk_summaries", True):
            return merged_result
        return self._reduce_chunk_summaries(
            chunk_results=chunk_results,
            chunks=chunks,
            evidence_observations=evidence_observations,
            action=action,
            glossary_actions=glossary_actions,
            output_schema=output_schema,
            changed_pixel_count=changed_pixel_count,
            changed_cell_percent=changed_cell_percent,
            deterministic_change_detected=deterministic_change_detected,
            source_frame_count=source_frame_count,
            fallback_result=merged_result,
        )

    def _summarize_observation_chunk(
        self,
        *,
        observations: tuple[Observation, ...],
        action: ActionSpec,
        output_schema: dict[str, Any],
        instructions: str,
        chunk_index: int,
        chunk_count: int,
        source_frame_count: int,
    ) -> ChangeSummaryResult:
        """Return a change summary for one overlapping text-observation chunk."""

        observation_text = serialize_observations(
            observations,
            config=self.config.observation_text,
            label="change_evidence_observations",
            include_header_metadata=False,
        )
        changed_pixel_count = cropped_changed_cell_count(
            observations[0].frame,
            observations[-1].frame,
            config=self.config.observation_text,
        )
        changed_cell_percent = cropped_changed_cell_percent(
            observations[0].frame,
            observations[-1].frame,
            changed_cell_count=changed_pixel_count,
            config=self.config.observation_text,
        )
        deterministic_change_detected = any_cropped_change_detected(
            observations,
            config=self.config.observation_text,
        )
        prompt = build_change_summary_prompt(
            action,
            observation_text=observation_text.text,
            changed_pixel_count=changed_pixel_count,
            frame_count=len(observations),
            change_detected=deterministic_change_detected,
        )
        images = observations_to_cropped_images(
            observations,
            observation_text_config=self.config.observation_text,
            frame_scale=self.config.frame_scale,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
        )
        self.last_prompt = prompt
        response = self.provider.complete(
            instructions_text=instructions,
            prompt_text=prompt,
            images=images,
            output_schema=output_schema,
        )
        self.last_request = response.request
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} change summary",
                response=response,
                text_of=lambda item: item.text,
                validate=lambda text: validate_change_summary_output(
                    text,
                    deterministic_change_detected=deterministic_change_detected,
                    summary_max_chars=getattr(
                        self.config,
                        "summary_max_chars",
                        DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
                    ),
                ),
                repair=provider_repair_callback(
                    self.provider,
                    "repair_complete",
                    kwargs={
                        "instructions_text": instructions,
                        "prompt_text": prompt,
                        "images": images,
                        "output_schema": output_schema,
                    },
                ),
                max_repair_attempts=getattr(self.config, "repair_attempts", 0),
                error_factory=ChangeSummaryOutputError,
            )
        except ChangeSummaryOutputError as exc:
            LOGGER.warning("falling back after change-summary repair failure: %s", exc)
            return ChangeSummaryResult(
                summary=_fallback_summary(deterministic_change_detected),
                changed_pixel_count=changed_pixel_count,
                change_detected=deterministic_change_detected,
                metadata={
                    **response.metadata,
                    "repair_attempts": getattr(self.config, "repair_attempts", 0),
                    "frame_count": len(observations),
                    "serialized_frame_count": len(observations),
                    "source_frame_count": source_frame_count,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "deterministic_change_detected": deterministic_change_detected,
                    "any_adjacent_frame_changed": deterministic_change_detected,
                    "fallback": "repair_exhausted",
                    "fallback_reason": str(exc),
                },
                changed_cell_percent=changed_cell_percent,
            )
        response = validated.response
        self.last_request = response.request
        return ChangeSummaryResult(
            summary=validated.value.summary,
            changed_pixel_count=changed_pixel_count,
            change_detected=validated.value.change_detected,
            metadata={
                **response.metadata,
                "repair_attempts": validated.repair_attempts,
                "frame_count": len(observations),
                "serialized_frame_count": len(observations),
                "source_frame_count": source_frame_count,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "deterministic_change_detected": deterministic_change_detected,
                "any_adjacent_frame_changed": deterministic_change_detected,
                **_change_summary_autocorrect_metadata(validated.value),
            },
            changed_cell_percent=changed_cell_percent,
        )

    def _reduce_chunk_summaries(
        self,
        *,
        chunk_results: tuple[ChangeSummaryResult, ...],
        chunks: tuple[tuple[Observation, ...], ...],
        evidence_observations: tuple[Observation, ...],
        action: ActionSpec,
        glossary_actions: Sequence[ActionSpec],
        output_schema: dict[str, Any],
        changed_pixel_count: int,
        changed_cell_percent: float,
        deterministic_change_detected: bool,
        source_frame_count: int,
        fallback_result: ChangeSummaryResult,
    ) -> ChangeSummaryResult:
        """Reduce multiple chunk summaries into one final transition summary."""

        keyframe_indices = _reducer_keyframe_indices(
            evidence_observations,
            chunks=chunks,
            limit=getattr(self.config, "reducer_keyframe_limit", 6),
        )
        keyframe_text = _reducer_keyframe_text(
            evidence_observations,
            indices=keyframe_indices,
            config=self.config.observation_text,
        )
        keyframe_images = observations_to_cropped_images(
            tuple(evidence_observations[index] for index in keyframe_indices),
            observation_text_config=self.config.observation_text,
            frame_scale=self.config.frame_scale,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
        )
        instructions = append_output_schema_to_instructions(
            append_arc_color_glossary(
                append_action_glossary(
                    load_change_summary_reducer_instructions(),
                    glossary_actions,
                    mode="committed_action",
                    observation_text_config=self.config.observation_text,
                )
            ),
            output_schema,
            include=bool(
                getattr(self.config, "include_output_schema_in_instructions", False)
            ),
        )
        prompt = build_change_summary_reducer_prompt(
            action,
            chunk_results=chunk_results,
            keyframe_text=keyframe_text,
            keyframe_indices=keyframe_indices,
            changed_pixel_count=changed_pixel_count,
            changed_cell_percent=changed_cell_percent,
            frame_count=len(evidence_observations),
            change_detected=deterministic_change_detected,
        )
        self.last_instructions = instructions
        self.last_prompt = prompt
        response = self.provider.reduce_complete(
            instructions_text=instructions,
            prompt_text=prompt,
            images=keyframe_images,
            output_schema=output_schema,
        )
        self.last_request = response.request
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} reduced change summary",
                response=response,
                text_of=lambda item: item.text,
                validate=lambda text: validate_change_summary_output(
                    text,
                    deterministic_change_detected=deterministic_change_detected,
                    summary_max_chars=getattr(
                        self.config,
                        "summary_max_chars",
                        DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
                    ),
                ),
                repair=provider_repair_callback(
                    self.provider,
                    "repair_reduce_complete",
                    kwargs={
                        "instructions_text": instructions,
                        "prompt_text": prompt,
                        "images": keyframe_images,
                        "output_schema": output_schema,
                    },
                ),
                max_repair_attempts=getattr(self.config, "repair_attempts", 0),
                error_factory=ChangeSummaryOutputError,
            )
        except ChangeSummaryOutputError as exc:
            LOGGER.warning(
                "falling back after change-summary reducer repair failure: %s", exc
            )
            return ChangeSummaryResult(
                summary=fallback_result.summary,
                changed_pixel_count=fallback_result.changed_pixel_count,
                change_detected=fallback_result.change_detected,
                metadata={
                    **fallback_result.metadata,
                    **response.metadata,
                    "frame_count": len(evidence_observations),
                    "serialized_frame_count": len(evidence_observations),
                    "source_frame_count": source_frame_count,
                    "chunk_count": len(chunk_results),
                    "deterministic_change_detected": deterministic_change_detected,
                    "any_adjacent_frame_changed": deterministic_change_detected,
                    "reducer": True,
                    "reducer_keyframe_indices": keyframe_indices,
                    "reducer_repair_attempts": getattr(
                        self.config, "repair_attempts", 0
                    ),
                    "reducer_fallback": "repair_exhausted",
                    "reducer_fallback_reason": str(exc),
                },
                changed_cell_percent=fallback_result.changed_cell_percent,
            )

        response = validated.response
        self.last_request = response.request
        return ChangeSummaryResult(
            summary=validated.value.summary,
            changed_pixel_count=changed_pixel_count,
            change_detected=validated.value.change_detected,
            metadata={
                **response.metadata,
                "reducer": True,
                "reducer_repair_attempts": validated.repair_attempts,
                "reducer_keyframe_indices": keyframe_indices,
                "frame_count": len(evidence_observations),
                "serialized_frame_count": len(evidence_observations),
                "source_frame_count": source_frame_count,
                "chunk_count": len(chunk_results),
                "chunk_repair_attempts": tuple(
                    result.metadata.get("repair_attempts", 0)
                    for result in chunk_results
                ),
                "chunk_fallbacks": tuple(
                    result.metadata.get("fallback") for result in chunk_results
                ),
                "deterministic_change_detected": deterministic_change_detected,
                "any_adjacent_frame_changed": deterministic_change_detected,
                **_change_summary_autocorrect_metadata(validated.value),
            },
            changed_cell_percent=changed_cell_percent,
        )

    def _default_provider(self, *, client: Any | None) -> ChangeSummaryProvider:
        backend = (getattr(self.config, "backend", None) or "").lower()
        if backend == "vllm":
            from face_of_agi.models.change.providers.vllm import (
                VLLMChangeSummaryProvider,
            )

            return VLLMChangeSummaryProvider(self.config, client=client)
        raise ValueError(f"unknown change summary backend: {self.config.backend}")


def load_change_summary_instructions(
    path: str | Path | None = None,
) -> str:
    """Load the human-editable change summary instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def load_change_summary_reducer_instructions(
    path: str | Path | None = None,
) -> str:
    """Load the human-editable change summary reducer instruction prompt."""

    instruction_path = (
        Path(path) if path is not None else DEFAULT_REDUCER_INSTRUCTION_PATH
    )
    return instruction_path.read_text(encoding="utf-8").strip()


def build_change_summary_prompt(
    action: ActionSpec,
    *,
    observation_text: str,
    changed_pixel_count: int,
    frame_count: int = 2,
    change_detected: bool | None = None,
) -> str:
    """Return the change-summary user prompt with action context."""

    if changed_pixel_count < 0:
        raise ValueError("changed_pixel_count must be non-negative")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    transition_lines = [
        "TRANSITION:",
        f"serialized_frame_count: {frame_count}",
        f"changed_cell_count: {changed_pixel_count}",
    ]
    if change_detected is not None:
        transition_lines.append(
            f"any_adjacent_frame_changed: {str(change_detected).lower()}"
        )
    return "\n\n".join(
        [
            CHANGE_SUMMARY_PROMPT,
            "\n".join(transition_lines),
            "OBSERVATIONS:\n" + observation_text,
            "ACTION:\n"
            f"action_id: {action.name}\n"
            f"data: {_action_data_text(action)}"
            + _action_target_text(action),
        ]
    )


def build_change_summary_reducer_prompt(
    action: ActionSpec,
    *,
    chunk_results: Sequence[ChangeSummaryResult],
    keyframe_text: str,
    keyframe_indices: Sequence[int],
    changed_pixel_count: int,
    changed_cell_percent: float,
    frame_count: int,
    change_detected: bool,
) -> str:
    """Return the final reducer user prompt for multiple chunk summaries."""

    if changed_pixel_count < 0:
        raise ValueError("changed_pixel_count must be non-negative")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not chunk_results:
        raise ValueError("chunk_results must not be empty")
    transition_lines = [
        "TRANSITION:",
        f"serialized_frame_count: {frame_count}",
        f"changed_cell_count: {changed_pixel_count}",
        f"changed_cell_percent: {_percent_text(changed_cell_percent)}",
        f"any_adjacent_frame_changed: {str(change_detected).lower()}",
    ]
    partial_lines: list[str] = []
    for index, result in enumerate(chunk_results, start=1):
        partial_lines.extend(
            [
                f"{index}. summary: {json.dumps(result.summary)}",
                f"   change_detected: {str(result.change_detected).lower()}",
            ]
        )
    selected_indices = ", ".join(str(index) for index in keyframe_indices)
    return "\n\n".join(
        [
            CHANGE_SUMMARY_REDUCER_PROMPT,
            "\n".join(transition_lines),
            "ORDERED_PARTIAL_SUMMARIES:\n" + "\n".join(partial_lines),
            "SELECTED_KEYFRAMES:\n"
            f"selected_frame_indices: {selected_indices}\n\n"
            + keyframe_text,
            "ACTION:\n"
            f"action_id: {action.name}\n"
            f"data: {_action_data_text(action)}"
            + _action_target_text(action),
        ]
    )


def cropped_changed_cell_percent(
    left: Any,
    right: Any,
    *,
    changed_cell_count: int | None = None,
    config: Any | None = None,
) -> float:
    """Return first-to-final changed percentage over the visible ARC crop."""

    from face_of_agi.models.observation_text import ObservationTextConfig

    resolved_config = (
        config
        if isinstance(config, ObservationTextConfig)
        else ObservationTextConfig(**config)
        if isinstance(config, dict)
        else ObservationTextConfig()
    )
    visible_axis = 64 - (2 * resolved_config.crop_cells)
    if visible_axis <= 0:
        raise ValueError("observation_text.crop_cells leaves an empty crop")
    count = (
        cropped_changed_cell_count(left, right, config=resolved_config)
        if changed_cell_count is None
        else changed_cell_count
    )
    return min(
        100.0,
        max(0.0, float(count) * 100.0 / float(visible_axis * visible_axis)),
    )


def any_cropped_change_detected(
    observations: Sequence[Observation],
    *,
    config: Any | None = None,
) -> bool:
    """Return whether any adjacent retained observations differ in the crop."""

    return any(
        cropped_changed_cell_count(left.frame, right.frame, config=config) > 0
        for left, right in zip(observations, observations[1:], strict=False)
    )


def _overlapping_observation_chunks(
    observations: Sequence[Observation],
    *,
    max_frames_per_call: int | None,
) -> tuple[tuple[Observation, ...], ...]:
    evidence = tuple(observations)
    if max_frames_per_call is None or len(evidence) <= max_frames_per_call:
        return (evidence,)
    limit = _normalized_max_frames_per_call(max_frames_per_call)
    chunk_count = math.ceil((len(evidence) - 1) / (limit - 1))
    total_chunk_frames = len(evidence) + chunk_count - 1
    base_chunk_size = total_chunk_frames // chunk_count
    extra_frames = total_chunk_frames % chunk_count
    chunk_sizes = tuple(
        base_chunk_size + (1 if index < extra_frames else 0)
        for index in range(chunk_count)
    )
    chunks: list[tuple[Observation, ...]] = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(tuple(evidence[start:end]))
        start = end - 1
    return tuple(chunks)


def _normalized_max_frames_per_call(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("max_frames_per_call must be an integer at least 2 or null")
    return value


def _reducer_keyframe_indices(
    observations: Sequence[Observation],
    *,
    chunks: Sequence[Sequence[Observation]],
    limit: int,
) -> tuple[int, ...]:
    evidence = tuple(observations)
    if len(evidence) < 2:
        raise ValueError("reducer keyframes require at least two evidence frames")
    normalized_limit = _normalized_reducer_keyframe_limit(limit)
    first_index = 0
    final_index = len(evidence) - 1
    index_by_identity = {
        id(observation): index for index, observation in enumerate(evidence)
    }
    boundary_indices: list[int] = []
    for chunk in chunks[:-1]:
        if not chunk:
            continue
        boundary_index = index_by_identity.get(id(chunk[-1]))
        if boundary_index is None or boundary_index in {first_index, final_index}:
            continue
        if boundary_index not in boundary_indices:
            boundary_indices.append(boundary_index)
    allowed_boundary_count = max(0, normalized_limit - 2)
    selected_boundaries = _evenly_sample_indices(
        tuple(sorted(boundary_indices)),
        count=allowed_boundary_count,
    )
    return tuple(sorted({first_index, *selected_boundaries, final_index}))


def _normalized_reducer_keyframe_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("reducer_keyframe_limit must be an integer at least 2")
    return value


def _evenly_sample_indices(values: tuple[int, ...], *, count: int) -> tuple[int, ...]:
    if count <= 0 or not values:
        return ()
    if count >= len(values):
        return values
    if count == 1:
        return (values[len(values) // 2],)
    sampled_positions = [
        round(index * (len(values) - 1) / (count - 1)) for index in range(count)
    ]
    sampled: list[int] = []
    for position in sampled_positions:
        value = values[position]
        if value not in sampled:
            sampled.append(value)
    for value in values:
        if len(sampled) >= count:
            break
        if value not in sampled:
            sampled.append(value)
    return tuple(sorted(sampled[:count]))


def _reducer_keyframe_text(
    observations: Sequence[Observation],
    *,
    indices: Sequence[int],
    config: ObservationTextConfig | dict[str, Any],
) -> str:
    base_config = (
        config
        if isinstance(config, ObservationTextConfig)
        else ObservationTextConfig(**config)
    )
    rows_config = ObservationTextConfig(
        crop_cells=base_config.crop_cells,
        overflow_chars_per_frame=base_config.overflow_chars_per_frame,
        include_rows=True,
        include_components=False,
        include_component_runs=False,
        compact_components=False,
    )
    frames: list[str] = []
    for index in indices:
        serialized = serialize_observation(
            observations[index],
            config=rows_config,
            label=f"reducer_keyframe original_frame_index={index}",
            include_header_metadata=False,
        )
        frames.append(serialized.text)
    return "\n\n".join(frames)


def _merged_change_summary_result(
    results: Sequence[ChangeSummaryResult],
    *,
    changed_pixel_count: int,
    changed_cell_percent: float,
    change_detected: bool,
    frame_count: int,
    source_frame_count: int,
) -> ChangeSummaryResult:
    return ChangeSummaryResult(
        summary=_merged_summary_text(results, change_detected=change_detected),
        changed_pixel_count=changed_pixel_count,
        change_detected=change_detected,
        metadata={
            **(results[-1].metadata if results else {}),
            "frame_count": frame_count,
            "serialized_frame_count": frame_count,
            "source_frame_count": source_frame_count,
            "chunk_count": len(results),
            "chunk_repair_attempts": tuple(
                result.metadata.get("repair_attempts", 0) for result in results
            ),
            "chunk_fallbacks": tuple(
                result.metadata.get("fallback") for result in results
            ),
            "deterministic_change_detected": change_detected,
            "any_adjacent_frame_changed": change_detected,
        },
        changed_cell_percent=changed_cell_percent,
    )


def _merged_summary_text(
    results: Sequence[ChangeSummaryResult],
    *,
    change_detected: bool,
) -> str:
    summaries: list[str] = []
    seen: set[str] = set()
    for result in results:
        summary = result.summary.strip()
        if not summary or _generic_zero_change_summary(summary):
            continue
        normalized = summary.lower()
        if normalized in seen:
            continue
        summaries.append(summary)
        seen.add(normalized)
    if not summaries:
        return _fallback_summary(change_detected)
    return " ".join(_sentence_text(summary) for summary in summaries)


def parse_change_summary_output(
    text: str,
    *,
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
) -> ParsedChangeSummary:
    """Parse the required JSON change summary output contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise ChangeSummaryOutputError(
            "change summary response must be JSON with non-empty 'summary' "
            "and boolean 'change_detected' fields; raw response preview: "
            f"{preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise ChangeSummaryOutputError("change summary response must be a JSON object")
    if set(loaded) != {"summary", "change_detected"}:
        raise ChangeSummaryOutputError(
            "change summary response must contain only 'summary' and "
            "'change_detected' fields"
        )
    summary = loaded.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ChangeSummaryOutputError(
            "change summary response is missing non-empty string field 'summary'"
        )
    summary = summary.strip()
    if summary_max_chars is not None and len(summary) > int(summary_max_chars):
        raise ChangeSummaryOutputError(
            "change summary response field 'summary' is too long: "
            f"{len(summary)} characters exceeds the "
            f"{int(summary_max_chars)} character cap"
        )
    change_detected = loaded.get("change_detected")
    if not isinstance(change_detected, bool):
        raise ChangeSummaryOutputError(
            "change summary response is missing boolean field 'change_detected'"
        )
    return ParsedChangeSummary(
        summary=summary,
        change_detected=change_detected,
    )


def validate_change_summary_output(
    text: str,
    *,
    deterministic_change_detected: bool,
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
) -> ParsedChangeSummary:
    """Parse output and enforce deterministic visible-change evidence."""

    parsed = parse_change_summary_output(
        text,
        summary_max_chars=summary_max_chars,
    )
    if parsed.change_detected != deterministic_change_detected:
        if _safe_change_detected_autocorrect(
            parsed,
            deterministic_change_detected=deterministic_change_detected,
        ):
            return ParsedChangeSummary(
                summary=parsed.summary,
                change_detected=deterministic_change_detected,
                model_change_detected=parsed.change_detected,
                autocorrected_change_detected=True,
                autocorrect_reason="boolean_mismatch_summary_consistent_with_change",
            )
        expected = str(deterministic_change_detected).lower()
        actual = str(parsed.change_detected).lower()
        raise ChangeSummaryOutputError(
            "change summary response field 'change_detected' conflicts with "
            f"deterministic visible change evidence: expected {expected}, got {actual}"
        )
    return parsed


def _safe_change_detected_autocorrect(
    parsed: ParsedChangeSummary,
    *,
    deterministic_change_detected: bool,
) -> bool:
    """Return whether a wrong model boolean can be corrected without repair."""

    return (
        deterministic_change_detected is True
        and parsed.change_detected is False
        and not _obvious_no_change_summary(parsed.summary)
    )


def _change_summary_autocorrect_metadata(
    parsed: ParsedChangeSummary,
) -> dict[str, Any]:
    """Return result metadata for adapter-owned boolean corrections."""

    if not parsed.autocorrected_change_detected:
        return {}
    return {
        "autocorrected_change_detected": True,
        "model_change_detected": parsed.model_change_detected,
        "autocorrect_reason": parsed.autocorrect_reason,
    }


def _action_data_text(action: ActionSpec) -> str:
    if action.data is None:
        return "{}"
    return json.dumps(action.data, sort_keys=True)


def _action_target_text(action: ActionSpec) -> str:
    if action.name != "ACTION6" or action.target is None or not action.target.strip():
        return ""
    return f"\ntarget: {json.dumps(action.target.strip())}"


def _fallback_summary(change_detected: bool) -> str:
    if change_detected:
        return "Visible changes occurred, but summary unavailable."
    return "no changes"


def _percent_text(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _sentence_text(summary: str) -> str:
    stripped = summary.strip()
    if not stripped:
        return stripped
    if stripped[-1] in ".!?":
        return stripped
    return stripped + "."


def _generic_zero_change_summary(summary: str) -> bool:
    normalized = summary.strip().lower().rstrip(".!")
    return normalized in {
        "no change",
        "no changes",
        "nothing changed",
        "no visible change",
        "no visible changes",
        "no visible playfield change",
        "no visible playfield changes",
        "no visible playfield change occurred",
        "first and final frames are identical",
    }


def _obvious_no_change_summary(summary: str) -> bool:
    normalized = re.sub(r"\s+", " ", summary.strip().lower()).strip(" .!")
    no_change_starts = (
        "no changes",
        "no change",
        "no visible change",
        "no visible changes",
        "no visible playfield change",
        "no visible playfield changes",
        "no meaningful visible change",
        "nothing changed",
        "no visual change occurred",
        "no visual change occurs",
    )
    no_change_contains = (
        " no visible change occurred",
        " no visible changes occurred",
        " no visible playfield change occurred",
        " no meaningful visible change occurred",
        " no visual change occurred",
        " no visual change occurs",
        " remains static across all frames",
        " remain static across all frames",
    )
    return normalized.startswith(no_change_starts) or any(
        phrase in f" {normalized}" for phrase in no_change_contains
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return stripped
    return match.group(1).strip()
