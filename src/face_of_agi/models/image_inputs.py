"""Shared model image normalization and vLLM content helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Sequence

from face_of_agi.contracts import Observation
from face_of_agi.frames import (
    frame_to_pil_image,
    parse_image_size,
    resize_image_if_needed,
)
from face_of_agi.models.observation_text import (
    ARC_GRID_SIZE,
    ObservationTextConfig,
)


def observation_to_cropped_image(
    observation: Observation,
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
    frame_scale: int = 4,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
) -> Any:
    """Return the image crop that matches serialized ObservationText rows."""

    frame = observation.frame
    if frame is None and observation.frames:
        frame = observation.frames[-1]
    if frame is None:
        raise ValueError(f"observation '{observation.id}' does not contain a frame")
    return frame_to_cropped_image(
        frame,
        step=observation.step,
        observation_text_config=observation_text_config,
        frame_scale=frame_scale,
        size=size,
        resample=resample,
        label=observation.id,
    )


def observations_to_cropped_images(
    observations: Sequence[Observation],
    *,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
    frame_scale: int = 4,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
) -> tuple[Any, ...]:
    """Return one model-visible cropped image for each observation."""

    return tuple(
        observation_to_cropped_image(
            observation,
            observation_text_config=observation_text_config,
            frame_scale=frame_scale,
            size=size,
            resample=resample,
        )
        for observation in observations
    )


def frame_to_cropped_image(
    frame: Any,
    *,
    step: int = 0,
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
    frame_scale: int = 4,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
    label: str = "frame",
) -> Any:
    """Render a frame and crop to the configured ObservationText bounds."""

    if frame_scale <= 0:
        raise ValueError("frame_scale must be positive")
    image = frame_to_pil_image(
        frame,
        step=step,
        grid_scale=frame_scale,
        label=label,
    )
    cropped = image.crop(_crop_box_for_image(image.size, observation_text_config))
    return resize_image_if_needed(cropped, size=size, resample=resample)


def vllm_text_image_content(
    text: str,
    images: Sequence[Any],
    *,
    detail: str | None = "auto",
    mime_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Return OpenAI Chat-compatible text plus image content parts."""

    return [
        {"type": "text", "text": text},
        *vllm_image_content(images, detail=detail, mime_type=mime_type),
    ]


def vllm_image_content(
    images: Sequence[Any],
    *,
    detail: str | None = "auto",
    mime_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Return OpenAI Chat-compatible image content items for vLLM."""

    content: list[dict[str, Any]] = []
    for image in images:
        image_url: dict[str, Any] = {
            "url": image_to_data_url(image, mime_type=mime_type),
        }
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return content


def image_to_data_url(image: Any, *, mime_type: str = "image/png") -> str:
    """Encode a PIL-compatible image as a provider data URL."""

    image = image.convert("RGB")
    if mime_type == "image/png":
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    elif mime_type == "image/jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
    elif mime_type == "image/webp":
        buffer = BytesIO()
        image.save(buffer, format="WEBP")
    else:
        raise ValueError(f"unsupported image_mime_type: {mime_type}")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_crop_bounds(
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    """Return inclusive ARC-grid crop bounds matching ObservationText."""

    config = _observation_text_config(observation_text_config)
    crop_cells = config.crop_cells
    if crop_cells < 0:
        raise ValueError("observation_text.crop_cells must be non-negative")
    x0 = crop_cells
    y0 = crop_cells
    x1 = ARC_GRID_SIZE - crop_cells - 1
    y1 = ARC_GRID_SIZE - crop_cells - 1
    if x0 > x1 or y0 > y1:
        raise ValueError("observation_text.crop_cells leaves an empty image crop")
    return (x0, y0, x1, y1)


def image_crop_size(
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
    *,
    frame_scale: int = 4,
    input_image_size: str | tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return final image size for the configured ObservationText crop."""

    parsed_size = parse_image_size(input_image_size, field_name="input_image_size")
    if parsed_size is not None:
        return parsed_size
    if frame_scale <= 0:
        raise ValueError("frame_scale must be positive")
    x0, y0, x1, y1 = image_crop_bounds(observation_text_config)
    return ((x1 - x0 + 1) * frame_scale, (y1 - y0 + 1) * frame_scale)


def _crop_box_for_image(
    image_size: tuple[int, int],
    observation_text_config: ObservationTextConfig | dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = image_crop_bounds(observation_text_config)
    width, height = image_size
    left = round(x0 * width / ARC_GRID_SIZE)
    top = round(y0 * height / ARC_GRID_SIZE)
    right = round((x1 + 1) * width / ARC_GRID_SIZE)
    bottom = round((y1 + 1) * height / ARC_GRID_SIZE)
    if left >= right or top >= bottom:
        raise ValueError(
            "observation_text.crop_cells resolves to an empty image crop "
            f"for image size {image_size}: {(left, top, right, bottom)!r}"
        )
    return (left, top, right, bottom)


def _observation_text_config(
    value: ObservationTextConfig | dict[str, Any] | None,
) -> ObservationTextConfig:
    if value is None:
        return ObservationTextConfig()
    if isinstance(value, ObservationTextConfig):
        return value
    if isinstance(value, dict):
        return ObservationTextConfig(**value)
    raise TypeError("observation_text_config must be an ObservationTextConfig")
