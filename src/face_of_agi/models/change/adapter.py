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
    frame_components_prompt_text,
)
from face_of_agi.models.change.contracts import (
    ChangeSummaryProvider,
    ChangeSummaryResult,
    change_summary_json_schema,
)
from face_of_agi.models.image_inputs import (
    frame_bundle_image_size,
    resize_image,
)
from face_of_agi.models.structured_output import (
    MODEL_FALLBACK_WARNING,
    append_output_schema_to_instructions,
    provider_repair_callback,
    readable_model_error,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"
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
                load_change_summary_instructions(),
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

        frame_components_text = frame_components_prompt_text(
            observations,
            crop_edges=crop_edges,
            max_nb_components=getattr(self.config, "max_nb_components", 50),
        )
        images = tuple(
            change_summary_observation_image(
                observation,
                crop_edges=crop_edges,
            )
            for observation in observations
        )
        target_size = frame_bundle_image_size(
            getattr(self.config, "input_image_size", None),
            frame_count=len(images),
            budget_frame_count=_animation_frame_budget_coefficient(self.config),
        )
        images = resized_change_summary_images(
            *images,
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
        response = None
        try:
            response = self.provider.complete(
                instructions_text=instructions,
                prompt_text=prompt,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
                output_schema=output_schema,
            )
            self.last_request = response.request
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
        except Exception as exc:
            LOGGER.warning(
                MODEL_FALLBACK_WARNING + " action=%s",
                "empty no-change",
                self.provider.backend,
                self.provider.model,
                getattr(self.config, "repair_attempts", 0),
                readable_model_error(exc),
                action.name,
            )
            return ChangeSummaryResult(
                elements=(),
                change_detected=False,
                metadata={
                    **(response.metadata if response is not None else {}),
                    "repair_attempts": getattr(self.config, "repair_attempts", 0),
                    "fallback": "model_call_or_repair_failed",
                    "fallback_reason": readable_model_error(exc),
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


def load_change_summary_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable change summary instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


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
        "## Previous elements\n\n"
        + _change_element_references_json(previous_change_elements),
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


def _animation_frame_budget_coefficient(config: Any) -> int:
    value = getattr(config, "animation_frame_budget_coefficient", 2)
    return max(2, int(value))


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


def _change_element_references_json(
    elements: Sequence[ChangeSummaryElement],
) -> str:
    return json.dumps(
        [
            {
                "element_name": element.element_name,
                "element_description": element.element_description,
            }
            for element in elements
        ],
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
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
