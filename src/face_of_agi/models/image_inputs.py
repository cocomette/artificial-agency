"""Shared model image normalization and provider input helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from math import sqrt
from typing import Any, Sequence

from PIL import Image

from face_of_agi.contracts import Observation
from face_of_agi.frames import (
    frame_to_pil_image,
    image_to_base64_png,
    observation_to_pil_image,
)


def parse_image_size(size: str | tuple[int, int] | None) -> tuple[int, int] | None:
    """Return an optional image size tuple from common config forms."""

    if size is None:
        return None
    if isinstance(size, tuple) and len(size) == 2:
        width, height = size
    elif isinstance(size, str) and "x" in size:
        width_text, height_text = size.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    else:
        raise ValueError(
            "input_image_size must be None, a (width, height) tuple, "
            "or a string like '1024x1024'"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"input_image_size must be positive, got {size!r}")
    return (width, height)


def resize_image(
    image: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
) -> Any:
    """Resize a PIL-compatible image when a provider config requests it."""

    target_size = parse_image_size(size)
    image = image.convert("RGB")
    if target_size is None or image.size == target_size:
        return image
    return image.resize(target_size, image_resampling_filter(resample))


def frame_bundle_image_size(
    size: str | tuple[int, int] | None,
    *,
    frame_count: int,
    budget_frame_count: int = 2,
) -> tuple[int, int] | None:
    """Return per-frame size that fits a bundle in a fixed frame-area budget."""

    target_size = parse_image_size(size)
    if target_size is None or frame_count <= budget_frame_count:
        return target_size
    width, height = target_size
    scale = sqrt(budget_frame_count / frame_count)
    return (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )


def image_resampling_filter(resample: str) -> Any:
    """Return the PIL resampling filter for a configured resize mode."""

    from PIL import Image

    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if resample not in filters:
        raise ValueError(f"unsupported input_image_resample: {resample}")
    return filters[resample]


def pil_format_for_mime(mime_type: str) -> str:
    """Return the PIL save format matching an image MIME type."""

    formats = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }
    if mime_type not in formats:
        raise ValueError(f"unsupported image_mime_type: {mime_type}")
    return formats[mime_type]


def image_to_provider_data_url(
    image: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
    mime_type: str = "image/png",
) -> str:
    """Encode a PIL-compatible image as a provider data URL."""

    encoded = image_to_provider_base64(
        image,
        size=size,
        resample=resample,
        mime_type=mime_type,
    )
    return f"data:{mime_type};base64,{encoded}"


def image_to_provider_base64(
    image: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
    mime_type: str = "image/png",
) -> str:
    """Encode a PIL-compatible image for provider payloads."""

    image = resize_image(image, size=size, resample=resample)

    buffer = BytesIO()
    image.save(buffer, format=pil_format_for_mime(mime_type))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def frame_to_provider_data_url(
    frame: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
    mime_type: str = "image/png",
) -> str:
    """Normalize a framework frame and encode it as a provider data URL."""

    image = frame_to_pil_image(frame)
    return image_to_provider_data_url(
        image,
        size=size,
        resample=resample,
        mime_type=mime_type,
    )


def observation_to_provider_data_url(
    observation: Observation,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
    mime_type: str = "image/png",
) -> str:
    """Normalize an observation frame and encode it as a provider data URL."""

    image = observation_to_pil_image(observation)
    return image_to_provider_data_url(
        image,
        size=size,
        resample=resample,
        mime_type=mime_type,
    )


def image_to_ollama_base64_png(
    image: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
) -> str:
    """Encode a PIL-compatible image for Ollama chat vision messages."""

    return image_to_base64_png(
        image,
        size=parse_image_size(size),
        resample=resample,
    )


def frame_to_ollama_base64_png(
    frame: Any,
    *,
    size: str | tuple[int, int] | None,
    resample: str = "nearest",
) -> str:
    """Normalize a framework frame and encode it for Ollama chat."""

    image = frame_to_pil_image(frame)
    return image_to_ollama_base64_png(
        image,
        size=size,
        resample=resample,
    )


def openai_image_content(
    images: Sequence[Any],
    *,
    detail: str,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
    mime_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Return OpenAI Responses image content items."""

    return [
        {
            "type": "input_image",
            "image_url": image_to_provider_data_url(
                image,
                size=size,
                resample=resample,
                mime_type=mime_type,
            ),
            "detail": detail,
        }
        for image in images
    ]


def vllm_image_content(
    images: Sequence[Any],
    *,
    detail: str | None = None,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
    mime_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Return OpenAI Chat-compatible image content items for vLLM."""

    content: list[dict[str, Any]] = []
    for image in images:
        image_url: dict[str, Any] = {
            "url": image_to_provider_data_url(
                image,
                size=size,
                resample=resample,
                mime_type=mime_type,
            )
        }
        if detail:
            image_url["detail"] = detail
        content.append(
            {
                "type": "image_url",
                "image_url": image_url,
            }
        )
    return content


def ollama_image_payloads(
    images: Sequence[Any],
    *,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
) -> list[str]:
    """Return Ollama chat image payloads."""

    return [
        image_to_ollama_base64_png(image, size=size, resample=resample)
        for image in images
    ]
