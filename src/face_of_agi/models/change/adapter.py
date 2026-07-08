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
from face_of_agi.models.arc_grid_crop import (
    arc_grid_to_normalized_1000,
    crop_image_arc_grid_edges,
    normalize_arc_grid_crop_edges,
)
from face_of_agi.models.change.config import OllamaChangeSummaryConfig
from face_of_agi.models.change.components import (
    component_instruction_text,
    frame_components_prompt_text,
)
from face_of_agi.models.change.contracts import (
    ChangeSummaryProvider,
    ChangeSummaryResult,
    change_summary_json_schema,
)
from face_of_agi.models.image_inputs import (
    cumulative_changed_pixel_masks,
    draw_scaled_cumulative_mask_edges,
    frame_bundle_image_size,
    resize_image,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"
DEFAULT_BOUNDING_BOX_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "bounding_box_instruction_prompt.md"
)
DEFAULT_DIFF_MASK_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "diff_mask_instruction_prompt.md"
)
DEFAULT_COMPONENT_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "component_instruction_prompt.md"
)
LOGGER = logging.getLogger(__name__)


class ChangeSummaryOutputError(RuntimeError):
    """Raised when a change summary backend returns invalid output."""


@dataclass(frozen=True, slots=True)
class ParsedChangeSummary:
    """Validated structured change-summary payload."""

    elements: tuple[ChangeSummaryElement, ...]
    change_detected: bool


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
        previous_change_elements: Sequence[ChangeSummaryElement],
    ) -> ChangeSummaryResult:
        """Return a compact description of visible change between observations."""

        output_schema = change_summary_json_schema()
        instructions = append_output_schema_to_instructions(
            append_action_glossary(
                load_change_summary_instructions(
                    include_bounding_boxes=bool(
                        getattr(self.config, "activate_bounding_boxes", False)
                    ),
                    include_diff_masks=bool(
                        getattr(self.config, "activate_diff_mask", False)
                    ),
                    include_components=bool(
                        getattr(self.config, "activate_components", False)
                    ),
                ),
                glossary_actions,
                mode="committed_action",
            ),
            output_schema,
            include=bool(
                getattr(self.config, "include_output_schema_in_instructions", False)
            ),
        )
        observations = tuple(frame_observations or ())
        if not observations:
            observations = (previous_observation, current_observation)
        crop_edges = getattr(
            self.config,
            "input_image_crop_arc_grid_edges",
            None,
        )
        chunks = _overlapping_observation_chunks(
            observations,
            max_frames_per_call=getattr(self.config, "max_frames_per_call", 10),
        )
        chunk_results: list[ChangeSummaryResult] = []
        prompt_change_elements = _renamed_duplicate_change_elements(
            previous_change_elements
        )
        for chunk in chunks:
            result = self._summarize_observation_chunk(
                observations=chunk,
                action=action,
                output_schema=output_schema,
                instructions=instructions,
                crop_edges=crop_edges,
                previous_change_elements=prompt_change_elements,
            )
            chunk_results.append(result)
            prompt_change_elements = _merged_change_summary_elements(chunk_results)
        if len(chunk_results) == 1:
            return chunk_results[0]
        return _merged_change_summary_result(chunk_results)

    def _summarize_observation_chunk(
        self,
        *,
        observations: tuple[Observation, ...],
        action: ActionSpec,
        output_schema: dict[str, Any],
        instructions: str,
        crop_edges: Any | None,
        previous_change_elements: Sequence[ChangeSummaryElement],
    ) -> ChangeSummaryResult:
        """Return a change summary for one overlapping observation chunk."""

        frame_components_text = (
            frame_components_prompt_text(
                observations,
                crop_edges=crop_edges,
                max_nb_components=getattr(self.config, "max_nb_components", 50),
            )
            if bool(getattr(self.config, "activate_components", False))
            else None
        )
        images = tuple(
            change_summary_observation_image(
                observation,
                crop_edges=crop_edges,
            )
            for observation in observations
        )
        raw_images = images
        final_frame_count = _diff_mask_sequence_frame_count(
            len(raw_images),
            enabled=bool(getattr(self.config, "activate_diff_mask", False)),
        )
        target_size = frame_bundle_image_size(
            getattr(self.config, "input_image_size", None),
            frame_count=final_frame_count,
            budget_frame_count=_animation_frame_budget_coefficient(self.config),
        )
        changed_region_masks = (
            cumulative_changed_pixel_masks(
                *raw_images,
            )
            if bool(getattr(self.config, "activate_bounding_boxes", False))
            else ()
        )
        images = resized_change_summary_images(
            *raw_images,
            size=target_size,
            resample=getattr(self.config, "input_image_resample", "nearest"),
        )
        if bool(getattr(self.config, "activate_bounding_boxes", False)):
            images = draw_scaled_cumulative_mask_edges(
                source_images=raw_images,
                target_images=images,
                frame_masks=changed_region_masks,
                dilation_kernel_size=getattr(
                    self.config,
                    "dilation_bounding_boxes",
                    3,
                ),
                line_width=getattr(self.config, "width_bounding_boxes", 3),
            )
        images = blur_change_summary_images(
            *images,
            kernel_size=getattr(self.config, "gaussian_blur_kernel_size", 0),
        )
        images = add_change_summary_gaussian_noise(
            *images,
            deviation=getattr(self.config, "gaussian_noise_deviation", 0.0),
        )
        if bool(getattr(self.config, "activate_diff_mask", False)):
            images = interleave_change_summary_diff_masks(
                source_images=raw_images,
                target_images=images,
                size=target_size,
                resample=getattr(self.config, "input_image_resample", "nearest"),
            )
        previous_image = images[0]
        current_image = images[-1]
        prompt = build_change_summary_prompt(
            action,
            crop_edges=crop_edges,
            previous_change_elements=previous_change_elements,
            frame_components_text=frame_components_text,
        )
        self.last_instructions = instructions
        self.last_prompt = prompt
        response = self.provider.complete(
            instructions_text=instructions,
            prompt_text=prompt,
            previous_image=previous_image,
            current_image=current_image,
            images=images,
            output_schema=output_schema,
        )
        self.last_request = response.request
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} change summary",
                response=response,
                text_of=lambda item: item.text,
                validate=parse_change_summary_output,
                repair=provider_repair_callback(
                    self.provider,
                    "repair_complete",
                    kwargs={
                        "instructions_text": instructions,
                        "prompt_text": prompt,
                        "previous_image": previous_image,
                        "current_image": current_image,
                        "images": images,
                        "output_schema": output_schema,
                    },
                ),
                max_repair_attempts=getattr(self.config, "repair_attempts", 0),
                error_factory=ChangeSummaryOutputError,
            )
        except ChangeSummaryOutputError as exc:
            LOGGER.error(
                "change summary structured output repair exhausted; "
                "using empty no-change fallback "
                "backend=%s model=%s repair_attempts=%s action=%s",
                self.provider.backend,
                self.provider.model,
                getattr(self.config, "repair_attempts", 0),
                action.name,
                exc_info=True,
            )
            return ChangeSummaryResult(
                elements=(),
                change_detected=False,
                metadata={
                    **response.metadata,
                    "repair_attempts": getattr(self.config, "repair_attempts", 0),
                    "fallback": "repair_exhausted",
                    "fallback_reason": str(exc),
                },
            )
        response = validated.response
        self.last_request = response.request
        return ChangeSummaryResult(
            elements=validated.value.elements,
            change_detected=validated.value.change_detected,
            metadata={
                **response.metadata,
                "repair_attempts": validated.repair_attempts,
            },
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


def _merged_change_summary_result(
    results: Sequence[ChangeSummaryResult],
) -> ChangeSummaryResult:
    return ChangeSummaryResult(
        elements=_merged_change_summary_elements(results),
        change_detected=any(result.change_detected for result in results),
        metadata={
            **(results[-1].metadata if results else {}),
            "chunk_count": len(results),
            "chunk_repair_attempts": tuple(
                result.metadata.get("repair_attempts", 0) for result in results
            ),
        },
    )


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
        if name_counts[name] == 1:
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


def change_summary_observation_image(
    observation: Observation,
    *,
    crop_edges: Any | None,
) -> Any:
    """Return the change-summary image, avoiding a second crop when pre-cropped."""

    image = observation_to_pil_image(observation)
    if _observation_already_cropped_for_change_summary(observation, crop_edges):
        return image
    return crop_image_arc_grid_edges(image, crop_edges)


def _observation_already_cropped_for_change_summary(
    observation: Observation,
    crop_edges: Any | None,
) -> bool:
    metadata_edges = observation.metadata.get("change_summary_crop_edges")
    if metadata_edges is None:
        return False
    try:
        return tuple(metadata_edges) == normalize_arc_grid_crop_edges(crop_edges)
    except TypeError:
        return False


def load_change_summary_instructions(
    path: str | Path | None = None,
    *,
    include_bounding_boxes: bool = False,
    bounding_box_path: str | Path | None = None,
    include_diff_masks: bool = False,
    diff_mask_path: str | Path | None = None,
    include_components: bool = False,
    component_path: str | Path | None = None,
) -> str:
    """Load the human-editable change summary instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    instructions = instruction_path.read_text(encoding="utf-8").strip()
    if include_bounding_boxes:
        bounding_instruction_path = (
            Path(bounding_box_path)
            if bounding_box_path is not None
            else DEFAULT_BOUNDING_BOX_INSTRUCTION_PATH
        )
        instructions += "\n\n" + bounding_instruction_path.read_text(
            encoding="utf-8"
        ).strip()
    if include_diff_masks:
        diff_mask_instruction_path = (
            Path(diff_mask_path)
            if diff_mask_path is not None
            else DEFAULT_DIFF_MASK_INSTRUCTION_PATH
        )
        instructions += "\n\n" + diff_mask_instruction_path.read_text(
            encoding="utf-8"
        ).strip()
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
    crop_edges: object | None = None,
    previous_change_elements: Sequence[ChangeSummaryElement],
    frame_components_text: str | None = None,
) -> str:
    """Return the per-call action context for the change-summary model."""

    action_lines = [
        "ACTION:",
        f"action_id: {action.name}",
        f"data: {_action_data_text(action, crop_edges=crop_edges)}",
    ]
    if action.name == "ACTION6" and action.target is not None:
        action_lines.append(f"target: {action.target.strip()}")
    if action.name == "ACTION6":
        action_lines.append("coordinate_space: normalized_0_1000")
    blocks = [
        "## Previous change elements\n\n"
        + _change_elements_json(previous_change_elements),
    ]
    if frame_components_text:
        blocks.append(frame_components_text)
    blocks.append("\n".join(action_lines))
    return "\n\n".join(blocks)


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


def interleave_change_summary_diff_masks(
    *,
    source_images: Sequence[Any],
    target_images: Sequence[Any],
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
) -> tuple[Any, ...]:
    """Insert binary consecutive-frame diff masks between model-visible frames."""

    if len(source_images) != len(target_images):
        raise ValueError("source_images and target_images must have the same length")
    if len(target_images) <= 1:
        return tuple(image.convert("RGB") for image in target_images)
    diff_masks = resized_change_summary_images(
        *consecutive_changed_pixel_masks(*source_images),
        size=size,
        resample=resample,
    )
    interleaved: list[Any] = []
    for index, image in enumerate(target_images):
        interleaved.append(image.convert("RGB"))
        if index < len(diff_masks):
            interleaved.append(diff_masks[index].convert("RGB"))
    return tuple(interleaved)


def consecutive_changed_pixel_masks(*images: Any) -> tuple[Any, ...]:
    """Return black/white RGB masks for changes between consecutive images."""

    if len(images) <= 1:
        return ()
    from PIL import Image, ImageChops

    masks: list[Any] = []
    for previous_image, current_image in zip(images, images[1:], strict=False):
        previous = previous_image.convert("RGB")
        current = current_image.convert("RGB")
        if previous.size != current.size:
            raise ValueError("diff masks require same-sized images")
        difference = ImageChops.difference(previous, current)
        if difference.getbbox() is None:
            masks.append(Image.new("RGB", previous.size, (0, 0, 0)))
            continue
        mask = difference.convert("L").point([0, *([255] * 255)])
        masks.append(Image.merge("RGB", (mask, mask, mask)))
    return tuple(masks)


def _diff_mask_sequence_frame_count(frame_count: int, *, enabled: bool) -> int:
    if not enabled or frame_count <= 1:
        return frame_count
    return frame_count + frame_count - 1


def _animation_frame_budget_coefficient(config: Any) -> int:
    value = getattr(config, "animation_frame_budget_coefficient", 2)
    return max(2, int(value))


def blur_change_summary_images(
    *images: Any,
    kernel_size: int = 0,
) -> tuple[Any, ...]:
    """Return model-visible images after an optional Gaussian blur pass."""

    normalized_kernel_size = _normalized_gaussian_blur_kernel_size(kernel_size)
    if normalized_kernel_size <= 1:
        return tuple(image.convert("RGB") for image in images)

    from PIL import ImageFilter

    radius = (normalized_kernel_size - 1) / 6.0
    return tuple(
        image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
        for image in images
    )


def _normalized_gaussian_blur_kernel_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("gaussian_blur_kernel_size must be a non-negative odd int")
    if value < 0 or (value > 1 and value % 2 == 0):
        raise ValueError("gaussian_blur_kernel_size must be a non-negative odd int")
    return value


def add_change_summary_gaussian_noise(
    *images: Any,
    deviation: float = 0.0,
) -> tuple[Any, ...]:
    """Return model-visible images with independent zero-centered Gaussian noise."""

    normalized_deviation = _normalized_gaussian_noise_deviation(deviation)
    if normalized_deviation == 0:
        return tuple(image.convert("RGB") for image in images)

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng()
    noisy_images: list[Any] = []
    for image in images:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        noise = rng.normal(
            loc=0.0,
            scale=normalized_deviation,
            size=pixels.shape,
        )
        noisy_pixels = np.clip(pixels + noise, 0, 255).astype("uint8")
        noisy_images.append(Image.fromarray(noisy_pixels))
    return tuple(noisy_images)


def _normalized_gaussian_noise_deviation(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gaussian_noise_deviation must be a non-negative number")
    normalized = float(value)
    if normalized < 0:
        raise ValueError("gaussian_noise_deviation must be a non-negative number")
    return normalized


def cropped_change_summary_images(
    *images: Any,
    crop_edges: Any | None,
) -> tuple[Any, ...]:
    """Return images after an optional ARC-grid crop."""

    return tuple(
        crop_image_arc_grid_edges(image, crop_edges)
        for image in images
    )


def parse_change_summary_output(text: str) -> ParsedChangeSummary:
    """Parse the required JSON change summary output contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise ChangeSummaryOutputError(
            "change summary response must be JSON with 'elements' and boolean "
            "'change_detected' fields; raw response preview: "
            f"{preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise ChangeSummaryOutputError("change summary response must be a JSON object")
    elements = _parse_change_elements(loaded.get("elements"))
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
    """Return the prompt-facing action-history summary reconstructed from elements."""

    return "\n".join(_change_summary_element_line(element) for element in elements)


def _change_summary_element_line(element: ChangeSummaryElement) -> str:
    name = element.element_name.strip()
    description = element.element_description.strip()
    mutation = element.element_mutation.strip()
    if not mutation:
        mutation = "no detected changes for this element"
    return f"- {name}: {description}; mutations: {mutation}"


def _parse_change_elements(value: Any) -> tuple[ChangeSummaryElement, ...]:
    if not isinstance(value, list):
        raise ChangeSummaryOutputError(
            "change summary response is missing array field 'elements'"
        )
    elements: list[ChangeSummaryElement] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ChangeSummaryOutputError(
                f"change summary element {index} must be an object"
            )
        element_name = item.get("element_name")
        element_description = item.get("element_description")
        element_mutation = item.get("element_mutation")
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
        elements.append(
            ChangeSummaryElement(
                element_name=element_name.strip(),
                element_description=element_description.strip(),
                element_mutation=element_mutation.strip(),
            )
        )
    return _renamed_duplicate_change_elements(elements)


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


def _action_data_text(
    action: ActionSpec,
    *,
    crop_edges: object | None,
) -> str:
    if action.data is None:
        return "{}"
    if action.name == "ACTION6":
        return json.dumps(
            {
                "x": arc_grid_to_normalized_1000(
                    action.data,
                    "x",
                    crop_edges=crop_edges,
                ),
                "y": arc_grid_to_normalized_1000(
                    action.data,
                    "y",
                    crop_edges=crop_edges,
                ),
            },
            sort_keys=True,
        )
    return json.dumps(action.data, sort_keys=True)


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
