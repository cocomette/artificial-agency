"""Provider-neutral transition change summary adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from face_of_agi.contracts import ActionSpec, ChangeSummaryElement, Observation
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_coordinates import action6_data_to_normalized_1000
from face_of_agi.models.change.components import (
    component_instruction_text,
    frame_components_prompt_text,
)
from face_of_agi.models.change.config import OllamaChangeSummaryConfig
from face_of_agi.models.change.contracts import (
    ChangeSummaryProvider,
    ChangeSummaryResult,
    DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
    DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    change_summary_json_schema,
)
from face_of_agi.models.image_inputs import (
    crop_image_normalized,
    frame_bundle_image_size,
    resize_image,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"
DEFAULT_COMPONENT_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "component_instruction_prompt.md"
)
CHANGE_SUMMARY_PROMPT = "Compare the attached observation frames from oldest to newest."
CHANGE_SUMMARY_IMAGE_BUDGET_FRAME_COUNT = 4
LOGGER = logging.getLogger(__name__)


class ChangeSummaryOutputError(RuntimeError):
    """Raised when a change summary backend returns invalid output."""


@dataclass(frozen=True, slots=True)
class ParsedChangeSummary:
    """Validated structured change-summary payload."""

    elements: tuple[ChangeSummaryElement, ...]
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
        self.config = config or OllamaChangeSummaryConfig()
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
        previous_change_elements: Sequence[ChangeSummaryElement] = (),
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
        crop_box_normalized = getattr(
            self.config,
            "input_image_crop_box_normalized",
            None,
        )
        whole_images = change_summary_observation_images(
            evidence_observations,
            frame_scale=getattr(self.config, "frame_scale", 4),
            size=getattr(self.config, "input_image_size", None),
            resample=getattr(self.config, "input_image_resample", "nearest"),
            crop_box_normalized=crop_box_normalized,
        )
        changed_pixel_count = model_visible_changed_pixel_count(
            whole_images[0],
            whole_images[-1],
        )
        changed_pixel_percent = model_visible_changed_pixel_percent(
            whole_images[0],
            whole_images[-1],
            changed_pixel_count=changed_pixel_count,
        )
        any_adjacent_frame_changed = model_visible_any_change_detected(whole_images)
        if changed_pixel_count == 0 and len(whole_images) <= 2:
            return ChangeSummaryResult(
                elements=(),
                changed_pixel_count=0,
                change_detected=False,
                metadata={
                    "skipped": True,
                    "skip_reason": "zero_changed_pixels",
                    "frame_count": len(whole_images),
                    "any_adjacent_frame_changed": False,
                },
                changed_pixel_percent=0.0,
            )

        summary_max_chars = getattr(
            self.config,
            "summary_max_chars",
            DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
        )
        summary_max_elements = getattr(
            self.config,
            "summary_max_elements",
            DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
        )
        output_schema = change_summary_json_schema(
            summary_max_chars=summary_max_chars,
            summary_max_elements=summary_max_elements,
        )
        instructions = append_output_schema_to_instructions(
            append_action_glossary(
                load_change_summary_instructions(
                    include_components=bool(
                        getattr(self.config, "activate_components", False)
                    )
                ),
                glossary_actions,
                mode="committed_action",
            ),
            output_schema,
            include=bool(
                getattr(self.config, "include_output_schema_in_instructions", False)
            ),
        )
        self.last_instructions = instructions
        chunks = _overlapping_observation_chunks(
            evidence_observations,
            max_frames_per_call=getattr(self.config, "max_frames_per_call", 10),
        )
        chunk_results: list[ChangeSummaryResult] = []
        persist_changed_elements_only = bool(
            getattr(self.config, "persist_changed_elements_only", False)
        )
        prompt_change_elements = _prompt_change_elements(
            _renamed_duplicate_change_elements(previous_change_elements),
            persist_changed_elements_only=persist_changed_elements_only,
        )
        for chunk in chunks:
            result = self._summarize_observation_chunk(
                observations=chunk,
                action=action,
                output_schema=output_schema,
                instructions=instructions,
                crop_box_normalized=crop_box_normalized,
                previous_change_elements=prompt_change_elements,
            )
            chunk_results.append(result)
            prompt_change_elements = _prompt_change_elements(
                _merged_change_summary_elements(chunk_results),
                persist_changed_elements_only=persist_changed_elements_only,
            )

        unfiltered_elements = _merged_change_summary_elements(chunk_results)
        elements = _prompt_change_elements(
            unfiltered_elements,
            persist_changed_elements_only=persist_changed_elements_only,
        )
        response_metadata = chunk_results[-1].metadata if chunk_results else {}
        return ChangeSummaryResult(
            elements=elements,
            changed_pixel_count=changed_pixel_count,
            change_detected=any_adjacent_frame_changed,
            metadata={
                **response_metadata,
                "frame_count": len(whole_images),
                "any_adjacent_frame_changed": any_adjacent_frame_changed,
                "chunk_count": len(chunk_results),
                "chunk_repair_attempts": tuple(
                    result.metadata.get("repair_attempts", 0)
                    for result in chunk_results
                ),
                **_merged_change_summary_autocorrect_metadata(chunk_results),
                "persist_changed_elements_only": persist_changed_elements_only,
                "element_count_before_persist_filter": len(unfiltered_elements),
                "element_count_after_persist_filter": len(elements),
            },
            changed_pixel_percent=changed_pixel_percent,
        )

    def _summarize_observation_chunk(
        self,
        *,
        observations: tuple[Observation, ...],
        action: ActionSpec,
        output_schema: dict[str, Any],
        instructions: str,
        crop_box_normalized: Any | None,
        previous_change_elements: Sequence[ChangeSummaryElement],
    ) -> ChangeSummaryResult:
        images = change_summary_observation_images(
            observations,
            frame_scale=getattr(self.config, "frame_scale", 4),
            size=getattr(self.config, "input_image_size", None),
            resample=getattr(self.config, "input_image_resample", "nearest"),
            crop_box_normalized=crop_box_normalized,
        )
        previous_image = images[0]
        current_image = images[-1]
        changed_pixel_count = model_visible_changed_pixel_count(
            previous_image,
            current_image,
        )
        changed_pixel_percent = model_visible_changed_pixel_percent(
            previous_image,
            current_image,
            changed_pixel_count=changed_pixel_count,
        )
        any_adjacent_frame_changed = model_visible_any_change_detected(images)
        frame_components_text = (
            frame_components_prompt_text(
                observations,
                crop_box_normalized=crop_box_normalized,
                max_nb_components=getattr(self.config, "max_nb_components", 50),
            )
            if bool(getattr(self.config, "activate_components", False))
            else None
        )
        prompt = build_change_summary_prompt(
            action,
            changed_pixel_count=changed_pixel_count,
            frame_count=len(images),
            crop_box_normalized=crop_box_normalized,
            previous_change_elements=previous_change_elements,
            frame_components_text=frame_components_text,
            any_adjacent_frame_changed=any_adjacent_frame_changed,
            summary_max_elements=getattr(
                self.config,
                "summary_max_elements",
                DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
            ),
        )
        self.last_prompt = prompt
        provider_images = images if len(images) > 2 else None
        response = self.provider.complete(
            instructions_text=instructions,
            prompt_text=prompt,
            previous_image=previous_image,
            current_image=current_image,
            output_schema=output_schema,
            images=provider_images,
        )
        self.last_request = response.request
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} change summary",
                response=response,
                text_of=lambda item: item.text,
                validate=lambda text: validate_change_summary_output(
                    text,
                    any_adjacent_frame_changed=any_adjacent_frame_changed,
                    summary_max_chars=getattr(
                        self.config,
                        "summary_max_chars",
                        DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
                    ),
                    summary_max_elements=getattr(
                        self.config,
                        "summary_max_elements",
                        DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
                    ),
                ),
                repair=provider_repair_callback(
                    self.provider,
                    "repair_complete",
                    kwargs={
                        "instructions_text": instructions,
                        "prompt_text": prompt,
                        "previous_image": previous_image,
                        "current_image": current_image,
                        "output_schema": output_schema,
                        "images": provider_images,
                    },
                ),
                max_repair_attempts=getattr(self.config, "repair_attempts", 0),
                error_factory=ChangeSummaryOutputError,
            )
        except ChangeSummaryOutputError as exc:
            LOGGER.warning("falling back after change-summary repair failure: %s", exc)
            return ChangeSummaryResult(
                elements=(),
                changed_pixel_count=changed_pixel_count,
                change_detected=any_adjacent_frame_changed,
                metadata={
                    **response.metadata,
                    "repair_attempts": getattr(self.config, "repair_attempts", 0),
                    "frame_count": len(images),
                    "any_adjacent_frame_changed": any_adjacent_frame_changed,
                    "fallback": "repair_exhausted",
                    "fallback_reason": str(exc),
                },
                changed_pixel_percent=changed_pixel_percent,
            )
        response = validated.response
        self.last_request = response.request
        return ChangeSummaryResult(
            elements=validated.value.elements,
            changed_pixel_count=changed_pixel_count,
            change_detected=validated.value.change_detected,
            metadata={
                **response.metadata,
                "repair_attempts": validated.repair_attempts,
                "frame_count": len(images),
                "any_adjacent_frame_changed": any_adjacent_frame_changed,
                **_change_summary_autocorrect_metadata(validated.value),
            },
            changed_pixel_percent=changed_pixel_percent,
        )

    def _default_provider(self, *, client: Any | None) -> ChangeSummaryProvider:
        backend = (getattr(self.config, "backend", None) or "").lower()
        if backend == "openai":
            from face_of_agi.models.change.providers.openai import (
                OpenAIChangeSummaryProvider,
            )

            return OpenAIChangeSummaryProvider(self.config, client=client)
        if backend == "ollama":
            from face_of_agi.models.change.providers.ollama import (
                OllamaChangeSummaryProvider,
            )

            return OllamaChangeSummaryProvider(self.config, client=client)
        if backend == "vllm":
            from face_of_agi.models.change.providers.vllm import (
                VLLMChangeSummaryProvider,
            )

            return VLLMChangeSummaryProvider(self.config, client=client)
        raise ValueError(f"unknown change summary backend: {self.config.backend}")


def _overlapping_observation_chunks(
    observations: Sequence[Observation],
    *,
    max_frames_per_call: int,
) -> tuple[tuple[Observation, ...], ...]:
    limit = _normalized_max_frames_per_call(max_frames_per_call)
    if len(observations) <= limit:
        return (tuple(observations),)
    chunk_count = math.ceil((len(observations) - 1) / (limit - 1))
    total_chunk_frames = len(observations) + chunk_count - 1
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
        chunks.append(tuple(observations[start:end]))
        start = end - 1
    return tuple(chunks)


def _normalized_max_frames_per_call(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("max_frames_per_call must be an int greater than 1")
    return value


def _merged_change_summary_elements(
    results: Sequence[ChangeSummaryResult],
) -> tuple[ChangeSummaryElement, ...]:
    return _merged_change_summary_element_sequence(
        element for result in results for element in result.elements
    )


def _merged_change_summary_element_sequence(
    elements: Iterable[ChangeSummaryElement],
) -> tuple[ChangeSummaryElement, ...]:
    order: list[str] = []
    descriptions: dict[str, str] = {}
    mutations: dict[str, list[str]] = {}
    seen_mutations: dict[str, set[str]] = {}
    for element in elements:
        name = element.element_name.strip()
        if not name:
            continue
        if name not in descriptions:
            order.append(name)
            descriptions[name] = ""
            mutations[name] = []
            seen_mutations[name] = set()
        if element.element_description.strip():
            descriptions[name] = element.element_description.strip()
        mutation = element.element_mutation.strip()
        if mutation and mutation not in seen_mutations[name]:
            mutations[name].append(mutation)
            seen_mutations[name].add(mutation)
    return tuple(
        ChangeSummaryElement(
            element_name=name,
            element_description=descriptions[name],
            element_mutation="; ".join(mutations[name]),
        )
        for name in order
    )


def _prompt_change_elements(
    elements: Sequence[ChangeSummaryElement],
    *,
    persist_changed_elements_only: bool,
) -> tuple[ChangeSummaryElement, ...]:
    if not persist_changed_elements_only:
        return tuple(elements)
    return tuple(element for element in elements if _element_has_visible_change(element))


def _elements_have_visible_change(
    elements: Sequence[ChangeSummaryElement],
) -> bool:
    return any(_element_has_visible_change(element) for element in elements)


def _element_has_visible_change(element: ChangeSummaryElement) -> bool:
    mutation = _normalized_no_change_text(element.element_mutation)
    if not mutation:
        return False
    return not any(
        re.fullmatch(pattern, mutation)
        for pattern in _NO_CHANGE_MUTATION_PATTERNS
    )


_NO_CHANGE_MUTATION_PATTERNS = (
    r"none",
    r"no (?:visible |detected )?changes?(?: for this element)?"
    r"(?: across (?:the )?(?:attached )?frames?)?",
    r"nothing changed",
    r"did not change",
    r"unchanged",
    r"(?:stayed|remained|remains|was|is) "
    r"(?:still|static|stationary|unchanged|the same)",
)


def _normalized_no_change_text(value: str) -> str:
    return re.sub(r"[\s.;:,-]+", " ", value.strip().lower()).strip()


def _renamed_duplicate_change_elements(
    elements: Sequence[ChangeSummaryElement],
) -> tuple[ChangeSummaryElement, ...]:
    name_counts: dict[str, int] = {}
    for element in elements:
        name = element.element_name.strip()
        name_counts[name] = name_counts.get(name, 0) + 1
    seen_names: dict[str, int] = {}
    renamed: list[ChangeSummaryElement] = []
    for element in elements:
        name = element.element_name.strip()
        if name_counts.get(name, 0) == 1:
            renamed.append(element)
            continue
        occurrence = seen_names.get(name, 0)
        seen_names[name] = occurrence + 1
        renamed.append(
            ChangeSummaryElement(
                element_name=f"{name}_{occurrence}",
                element_description=element.element_description,
                element_mutation=element.element_mutation,
            )
        )
    return tuple(renamed)


def load_change_summary_instructions(
    path: str | Path | None = None,
    *,
    include_components: bool = False,
    component_path: str | Path | None = None,
) -> str:
    """Load the human-editable change summary instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    instructions = instruction_path.read_text(encoding="utf-8").strip()
    if include_components:
        component_instruction_path = (
            Path(component_path)
            if component_path is not None
            else DEFAULT_COMPONENT_INSTRUCTION_PATH
        )
        instructions += "\n\n" + component_instruction_text(
            component_instruction_path.read_text(encoding="utf-8")
        )
    return instructions


def build_change_summary_prompt(
    action: ActionSpec,
    *,
    changed_pixel_count: int,
    frame_count: int = 2,
    crop_box_normalized: Any | None = None,
    previous_change_elements: Sequence[ChangeSummaryElement] = (),
    frame_components_text: str | None = None,
    any_adjacent_frame_changed: bool | None = None,
    summary_max_elements: int | None = DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
) -> str:
    """Return the change-summary user prompt with action context."""

    if changed_pixel_count < 0:
        raise ValueError("changed_pixel_count must be non-negative")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    blocks = [
        CHANGE_SUMMARY_PROMPT,
        _change_summary_output_limit_text(summary_max_elements),
        "## Previous change elements\n\n"
        + _change_elements_json(previous_change_elements),
        "TRANSITION:\n"
        f"attached_frame_count: {frame_count}\n"
        f"changed_pixel_count: {changed_pixel_count}",
    ]
    if any_adjacent_frame_changed is not None:
        blocks[-1] += (
            "\nany_adjacent_frame_changed: "
            f"{str(any_adjacent_frame_changed).lower()}"
        )
    if frame_components_text:
        blocks.append(frame_components_text)
    blocks.append(
        "ACTION:\n"
        + "\n".join(
            _action_context_lines(action, crop_box_normalized=crop_box_normalized)
        )
    )
    return "\n\n".join(blocks)


def _change_summary_output_limit_text(summary_max_elements: int | None) -> str:
    if summary_max_elements is None:
        return "Output only the most important visible elements."
    return (
        "Output at most "
        f"{summary_max_elements} visible elements in the `elements` array."
    )


def resized_change_summary_images(
    *images: Any,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
) -> tuple[Any, ...]:
    """Return model-visible images after configured resizing."""

    return tuple(
        resize_image(image, size=size, resample=resample)
        for image in images
    )


def cropped_change_summary_images(
    *images: Any,
    crop_box_normalized: Any | None,
) -> tuple[Any, ...]:
    """Return images after an optional normalized crop."""

    return tuple(
        crop_image_normalized(image, crop_box_normalized)
        for image in images
    )


def change_summary_observation_images(
    observations: Sequence[Observation],
    *,
    frame_scale: int,
    size: str | tuple[int, int] | None,
    resample: str,
    crop_box_normalized: Any | None,
) -> tuple[Any, ...]:
    """Return model-visible images for a change-summary frame bundle."""

    bundle_size = frame_bundle_image_size(
        size,
        frame_count=len(observations),
        budget_frame_count=CHANGE_SUMMARY_IMAGE_BUDGET_FRAME_COUNT,
    )
    return tuple(
        crop_image_normalized(
            resize_image(
                observation_to_pil_image(observation, frame_scale=frame_scale),
                size=bundle_size,
                resample=resample,
            ),
            crop_box_normalized,
        )
        for observation in observations
    )


def model_visible_changed_pixel_count(previous_image: Any, current_image: Any) -> int:
    """Return changed pixels between the exact model-visible RGB images."""

    import numpy as np

    previous_array = np.asarray(previous_image.convert("RGB"))
    current_array = np.asarray(current_image.convert("RGB"))
    if previous_array.shape != current_array.shape:
        return max(
            _image_surface_size(previous_array),
            _image_surface_size(current_array),
        )
    changed = previous_array != current_array
    if changed.ndim == 3:
        changed = np.any(changed, axis=-1)
    return int(np.count_nonzero(changed))


def model_visible_changed_pixel_percent(
    previous_image: Any,
    current_image: Any,
    *,
    changed_pixel_count: int | None = None,
) -> float:
    """Return first-to-final changed area percentage for model-visible images."""

    import numpy as np

    previous_array = np.asarray(previous_image.convert("RGB"))
    current_array = np.asarray(current_image.convert("RGB"))
    denominator = max(
        1,
        _image_surface_size(previous_array),
        _image_surface_size(current_array),
    )
    count = (
        model_visible_changed_pixel_count(previous_image, current_image)
        if changed_pixel_count is None
        else changed_pixel_count
    )
    return min(100.0, max(0.0, float(count) * 100.0 / denominator))


def model_visible_any_change_detected(images: Sequence[Any]) -> bool:
    """Return whether any adjacent model-visible frame pair differs."""

    return any(
        model_visible_changed_pixel_count(left, right) > 0
        for left, right in zip(images, images[1:], strict=False)
    )


def parse_change_summary_output(
    text: str,
    *,
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    summary_max_elements: int | None = DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
) -> ParsedChangeSummary:
    """Parse the required JSON change summary output contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise ChangeSummaryOutputError(
            "change summary response must be JSON with 'elements' and boolean "
            f"'change_detected' fields; raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise ChangeSummaryOutputError("change summary response must be a JSON object")
    if set(loaded) != {"elements", "change_detected"}:
        raise ChangeSummaryOutputError(
            "change summary response must contain only 'elements' and "
            "'change_detected' fields"
        )
    elements = _parse_change_elements(
        loaded.get("elements"),
        summary_max_chars=summary_max_chars,
        summary_max_elements=summary_max_elements,
    )
    change_detected = loaded.get("change_detected")
    if not isinstance(change_detected, bool):
        raise ChangeSummaryOutputError(
            "change summary response is missing boolean field 'change_detected'"
        )
    return ParsedChangeSummary(
        elements=elements,
        change_detected=change_detected,
    )


def change_summary_elements_text(
    elements: Sequence[ChangeSummaryElement],
) -> str:
    """Return prompt-facing change text reconstructed from structured elements."""

    return "\n".join(_change_summary_element_line(element) for element in elements)


def _change_summary_element_line(element: ChangeSummaryElement) -> str:
    name = element.element_name.strip()
    description = element.element_description.strip()
    mutation = element.element_mutation.strip()
    if not mutation:
        mutation = "no detected changes for this element"
    return f"- {name}: {description}; mutations: {mutation}"


def _parse_change_elements(
    value: Any,
    *,
    summary_max_chars: int | None,
    summary_max_elements: int | None,
) -> tuple[ChangeSummaryElement, ...]:
    if not isinstance(value, list):
        raise ChangeSummaryOutputError(
            "change summary response is missing array field 'elements'"
        )
    if summary_max_elements is not None and len(value) > summary_max_elements:
        raise ChangeSummaryOutputError(
            "change summary response has too many elements: "
            f"{len(value)} exceeds the {summary_max_elements} element cap"
        )
    elements: list[ChangeSummaryElement] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ChangeSummaryOutputError(
                f"change summary element {index} must be an object"
            )
        unexpected = set(item) - {
            "element_name",
            "element_description",
            "element_mutation",
        }
        if unexpected:
            raise ChangeSummaryOutputError(
                "change summary element has unexpected keys: "
                + ", ".join(sorted(unexpected))
            )
        element_name = item.get("element_name")
        element_description = item.get("element_description")
        element_mutation = item.get("element_mutation")
        if not isinstance(element_name, str) or not element_name.strip():
            raise ChangeSummaryOutputError(
                f"change summary element {index} is missing element_name"
            )
        if not isinstance(element_description, str) or not element_description.strip():
            raise ChangeSummaryOutputError(
                f"change summary element {index} is missing element_description"
            )
        if not isinstance(element_mutation, str):
            raise ChangeSummaryOutputError(
                f"change summary element {index} is missing element_mutation"
            )
        for field_name, field_value in (
            ("element_name", element_name),
            ("element_description", element_description),
            ("element_mutation", element_mutation),
        ):
            _validate_change_summary_field_length(
                field_name,
                field_value,
                index=index,
                summary_max_chars=summary_max_chars,
            )
        elements.append(
            ChangeSummaryElement(
                element_name=element_name.strip(),
                element_description=element_description.strip(),
                element_mutation=element_mutation.strip(),
            )
        )
    return _renamed_duplicate_change_elements(elements)


def validate_change_summary_output(
    text: str,
    *,
    any_adjacent_frame_changed: bool,
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    summary_max_elements: int | None = DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
) -> ParsedChangeSummary:
    """Parse output and enforce deterministic visible-change evidence."""

    parsed = parse_change_summary_output(
        text,
        summary_max_chars=summary_max_chars,
        summary_max_elements=summary_max_elements,
    )
    if parsed.change_detected != any_adjacent_frame_changed:
        if _safe_change_detected_autocorrect(
            parsed,
            any_adjacent_frame_changed=any_adjacent_frame_changed,
        ):
            return ParsedChangeSummary(
                elements=parsed.elements,
                change_detected=any_adjacent_frame_changed,
                model_change_detected=parsed.change_detected,
                autocorrected_change_detected=True,
                autocorrect_reason="boolean_mismatch_elements_consistent_with_change",
            )
        expected = str(any_adjacent_frame_changed).lower()
        actual = str(parsed.change_detected).lower()
        raise ChangeSummaryOutputError(
            "change summary response field 'change_detected' conflicts with "
            f"adjacent-frame visible change evidence: expected {expected}, "
            f"got {actual}"
        )
    return parsed


def _safe_change_detected_autocorrect(
    parsed: ParsedChangeSummary,
    *,
    any_adjacent_frame_changed: bool,
) -> bool:
    """Return whether a false model boolean can be corrected without repair."""

    return (
        any_adjacent_frame_changed is True
        and parsed.change_detected is False
        and _elements_have_visible_change(parsed.elements)
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


def _merged_change_summary_autocorrect_metadata(
    results: Sequence[ChangeSummaryResult],
) -> dict[str, Any]:
    if not any(
        result.metadata.get("autocorrected_change_detected") for result in results
    ):
        return {}
    return {
        "autocorrected_change_detected": True,
        "model_change_detected": False,
        "autocorrect_reason": "boolean_mismatch_elements_consistent_with_change",
    }


def _validate_change_summary_field_length(
    field_name: str,
    value: str,
    *,
    index: int,
    summary_max_chars: int | None,
) -> None:
    if summary_max_chars is None or len(value) <= summary_max_chars:
        return
    raise ChangeSummaryOutputError(
        f"change summary element {index} field {field_name!r} is too long: "
        f"{len(value)} characters exceeds the {summary_max_chars} character cap"
    )


def _image_surface_size(array: Any) -> int:
    if array.shape == ():
        return 1
    if array.ndim >= 2:
        return int(array.shape[0] * array.shape[1])
    return int(array.size)


def _change_elements_json(elements: Sequence[ChangeSummaryElement]) -> str:
    return json.dumps(
        [
            {
                "element_name": element.element_name,
                "element_description": element.element_description,
                "element_mutation": element.element_mutation,
            }
            for element in elements
        ],
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def _action_context_lines(
    action: ActionSpec,
    *,
    crop_box_normalized: Any | None,
) -> list[str]:
    lines = [
        f"action_id: {action.name}",
        "data: "
        + _action_data_text(action, crop_box_normalized=crop_box_normalized),
    ]
    if action.name == "ACTION6" and action.data is not None:
        target = _action_target_text(action)
        lines.append(f"target: {json.dumps(target)}")
        lines.append("coordinate_space: normalized_0_1000")
    return lines


def _action_data_text(
    action: ActionSpec,
    *,
    crop_box_normalized: Any | None,
) -> str:
    if action.data is None:
        return "{}"
    data = action.data
    if action.name == "ACTION6":
        data = action6_data_to_normalized_1000(
            action.data,
            crop_box_normalized=crop_box_normalized,
        )
    return json.dumps(data, sort_keys=True)


def _action_target_text(action: ActionSpec) -> str:
    if action.target is None or not action.target.strip():
        raise ValueError("ACTION6 change-summary prompt requires non-empty target")
    return action.target.strip()




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
