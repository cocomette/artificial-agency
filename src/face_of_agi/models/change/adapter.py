"""Provider-neutral transition change summary adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import re
from pathlib import Path
from typing import Any, Sequence

from face_of_agi.contracts import ActionSpec, Observation
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.change.config import OllamaChangeSummaryConfig
from face_of_agi.models.change.contracts import (
    ChangeSummaryProvider,
    ChangeSummaryResult,
    change_summary_json_schema,
)
from face_of_agi.models.arc_grid_crop import (
    arc_grid_to_normalized_1000,
    crop_image_arc_grid_edges,
)
from face_of_agi.models.image_inputs import frame_bundle_image_size, resize_image
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"
CHANGE_SUMMARY_PROMPT = "Compare the attached observation frames from oldest to newest."


class ChangeSummaryOutputError(RuntimeError):
    """Raised when a change summary backend returns invalid output."""


@dataclass(frozen=True, slots=True)
class ParsedChangeSummary:
    """Validated structured change-summary payload."""

    summary: str
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
        changed_pixel_percent: float,
        frame_observations: Sequence[Observation] | None = None,
        max_transition_changed_pixel_percent: float | None = None,
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
        crop_edges = getattr(
            self.config,
            "input_image_crop_arc_grid_edges",
            None,
        )
        images = change_summary_observation_images(
            evidence_observations,
            frame_scale=getattr(self.config, "frame_scale", 4),
            size=getattr(self.config, "input_image_size", None),
            resample=getattr(self.config, "input_image_resample", "nearest"),
            crop_edges=crop_edges,
        )
        prompt = build_change_summary_prompt(
            action,
            changed_pixel_percent=changed_pixel_percent,
            frame_count=len(images),
            max_transition_changed_pixel_percent=(
                max_transition_changed_pixel_percent
            ),
            crop_edges=crop_edges,
        )
        self.last_prompt = prompt
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
        self.last_instructions = instructions
        response = self.provider.complete(
            instructions_text=instructions,
            prompt_text=prompt,
            previous_image=images[0],
            current_image=images[-1],
            output_schema=output_schema,
            images=images,
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
                    "previous_image": images[0],
                    "current_image": images[-1],
                    "output_schema": output_schema,
                    "images": images,
                },
            ),
            max_repair_attempts=getattr(self.config, "repair_attempts", 0),
            error_factory=ChangeSummaryOutputError,
        )
        response = validated.response
        self.last_request = response.request
        return ChangeSummaryResult(
            summary=validated.value.summary,
            changed_pixel_percent=changed_pixel_percent,
            change_detected=validated.value.change_detected,
            metadata={
                **response.metadata,
                "repair_attempts": validated.repair_attempts,
                "frame_count": len(images),
                "max_transition_changed_pixel_percent": (
                    max_transition_changed_pixel_percent
                ),
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


def load_change_summary_instructions(
    path: str | Path | None = None,
) -> str:
    """Load the human-editable change summary instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def build_change_summary_prompt(
    action: ActionSpec,
    *,
    changed_pixel_percent: float,
    frame_count: int = 2,
    max_transition_changed_pixel_percent: float | None = None,
    crop_edges: Any | None = None,
) -> str:
    """Return the change-summary user prompt with action context."""

    _validate_changed_pixel_percent(
        changed_pixel_percent,
        name="changed_pixel_percent",
    )
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    transition_lines = [
        f"attached_frame_count: {frame_count}",
        "changed_pixel_percent: "
        f"{format_changed_pixel_percent(changed_pixel_percent)}",
    ]
    if max_transition_changed_pixel_percent is not None:
        _validate_changed_pixel_percent(
            max_transition_changed_pixel_percent,
            name="max_transition_changed_pixel_percent",
        )
        transition_lines.append(
            "max_transition_changed_pixel_percent: "
            f"{format_changed_pixel_percent(max_transition_changed_pixel_percent)}"
        )
    action_lines = [
        "ACTION:",
        f"action_id: {action.name}",
        f"data: {_action_data_text(action, crop_edges=crop_edges)}",
    ]
    if action.name == "ACTION6":
        action_lines.append("coordinate_space: normalized_0_1000")
    return "\n\n".join(
        [
            CHANGE_SUMMARY_PROMPT,
            "TRANSITION:\n" + "\n".join(transition_lines),
            "\n".join(action_lines),
        ]
    )


def resized_change_summary_images(
    previous_image: Any,
    current_image: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
) -> tuple[Any, Any]:
    """Return model-visible previous/current images after configured resizing."""

    return (
        resize_image(previous_image, size=size, resample=resample),
        resize_image(current_image, size=size, resample=resample),
    )


def cropped_change_summary_images(
    previous_image: Any,
    current_image: Any,
    *,
    crop_edges: Any | None,
) -> tuple[Any, Any]:
    """Return previous/current images after an optional ARC-grid crop."""

    return (
        crop_image_arc_grid_edges(previous_image, crop_edges),
        crop_image_arc_grid_edges(current_image, crop_edges),
    )


def change_summary_observation_images(
    observations: Sequence[Observation],
    *,
    frame_scale: int,
    size: str | tuple[int, int] | None,
    resample: str,
    crop_edges: Any | None,
) -> tuple[Any, ...]:
    """Return model-visible images for a change-summary frame bundle."""

    bundle_size = frame_bundle_image_size(size, frame_count=len(observations))
    return tuple(
        crop_image_arc_grid_edges(
            resize_image(
                observation_to_pil_image(observation, frame_scale=frame_scale),
                size=bundle_size,
                resample=resample,
            ),
            crop_edges,
        )
        for observation in observations
    )


def model_visible_changed_pixel_percent(previous_image: Any, current_image: Any) -> float:
    """Return changed-pixel percentage between exact model-visible RGB images."""

    import numpy as np

    previous_array = np.asarray(previous_image.convert("RGB"))
    current_array = np.asarray(current_image.convert("RGB"))
    if previous_array.shape != current_array.shape:
        return 100.0
    surface_size = _image_surface_size(previous_array)
    if surface_size <= 0:
        return 0.0
    changed = previous_array != current_array
    if changed.ndim == 3:
        changed = np.any(changed, axis=-1)
    return float(np.count_nonzero(changed) * 100.0 / surface_size)


def format_changed_pixel_percent(value: float) -> str:
    """Return a compact prompt/debug representation for a 0-100 percent value."""

    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _validate_changed_pixel_percent(value: float, *, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be finite and within 0..100")


def parse_change_summary_output(text: str) -> ParsedChangeSummary:
    """Parse the required JSON change summary output contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise ChangeSummaryOutputError(
            "change summary response must be JSON with a non-empty 'summary' "
            f"field; raw response preview: {preview!r}"
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
    change_detected = loaded.get("change_detected")
    if not isinstance(change_detected, bool):
        raise ChangeSummaryOutputError(
            "change summary response is missing boolean field 'change_detected'"
        )
    return ParsedChangeSummary(
        summary=summary.strip(),
        change_detected=change_detected,
    )


def _image_surface_size(array: Any) -> int:
    if array.shape == ():
        return 1
    if array.ndim >= 2:
        return int(array.shape[0] * array.shape[1])
    return int(array.size)


def _action_data_text(action: ActionSpec, *, crop_edges: Any | None = None) -> str:
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
