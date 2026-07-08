"""Shared model image normalization and provider input helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from math import sqrt
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageFilter

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


def cumulative_changed_pixel_masks(
    *images: Any,
) -> tuple[Any, ...]:
    """Return per-frame cumulative raw changed-pixel masks."""

    normalized_images = tuple(image.convert("RGB").copy() for image in images)
    if not normalized_images:
        return ()
    if len(normalized_images) == 1:
        return (Image.new("L", normalized_images[0].size, 0),)

    cumulative = Image.new("L", normalized_images[0].size, 0)
    masks: list[Any] = []
    for previous_image, current_image in zip(
        normalized_images,
        normalized_images[1:],
        strict=False,
    ):
        cumulative = ImageChops.lighter(
            cumulative,
            _changed_pixel_mask(previous_image, current_image),
        )
        masks.append(cumulative.copy())
    return (masks[0], *masks)


def _changed_pixel_mask(previous_image: Any, current_image: Any) -> Any:
    """Return a binary Pillow mask for raw pixels changed between two frames."""

    if previous_image.size != current_image.size:
        raise ValueError("changed-pixel masks require same-sized images")

    difference = ImageChops.difference(
        previous_image.convert("RGB"),
        current_image.convert("RGB"),
    )
    bounds = difference.getbbox()
    if bounds is None:
        return Image.new("L", previous_image.size, 0)
    return difference.convert("L").point([0, *([255] * 255)])


def draw_scaled_cumulative_mask_edges(
    *,
    source_images: Sequence[Any],
    target_images: Sequence[Any],
    frame_masks: Sequence[Any],
    dilation_kernel_size: int = 3,
    color: tuple[int, int, int] = (255, 0, 255),
    line_width: int = 3,
) -> tuple[Any, ...]:
    """Draw scaled magenta edges for per-frame cumulative changed-pixel masks."""

    normalized_dilation = _normalize_bounding_box_dilation(dilation_kernel_size)
    normalized_line_width = _normalize_bounding_box_line_width(line_width)
    annotated = tuple(image.convert("RGB").copy() for image in target_images)
    if len(source_images) != len(annotated):
        raise ValueError("source_images and target_images must have the same length")
    if len(frame_masks) != len(annotated):
        raise ValueError("frame_masks and target_images must have the same length")

    for source_image, image, mask in zip(source_images, annotated, frame_masks, strict=True):
        if source_image.size != mask.size:
            raise ValueError("frame mask size must match its source image size")
        edge_mask = _scaled_cumulative_mask_edge(
            mask,
            target_size=image.size,
            dilation_kernel_size=normalized_dilation,
            line_width=normalized_line_width,
        )
        if edge_mask.getbbox() is None:
            continue
        _paste_mask_color(
            image=image,
            mask=edge_mask,
            color=color,
        )
    return annotated


def _scaled_cumulative_mask_edge(
    mask: Any,
    *,
    target_size: tuple[int, int],
    dilation_kernel_size: int,
    line_width: int,
) -> Any:
    """Return the resized, final-pixel-width edge mask for a cumulative mask."""

    drawing_mask = mask
    if dilation_kernel_size > 1:
        drawing_mask = drawing_mask.filter(ImageFilter.MaxFilter(dilation_kernel_size))
    if drawing_mask.size != target_size:
        drawing_mask = drawing_mask.resize(target_size, Image.Resampling.NEAREST)
    edge_mask = ImageChops.subtract(drawing_mask, _erode_4_connected(drawing_mask))
    if line_width > 1:
        edge_mask = edge_mask.filter(ImageFilter.MaxFilter(_odd_filter_size(line_width)))
    return edge_mask


def _erode_4_connected(mask: Any) -> Any:
    """Return a binary erosion using only the four direct neighbors."""

    return ImageChops.darker(
        ImageChops.darker(_shift_mask(mask, -1, 0), _shift_mask(mask, 1, 0)),
        ImageChops.darker(_shift_mask(mask, 0, -1), _shift_mask(mask, 0, 1)),
    )


def _shift_mask(mask: Any, dx: int, dy: int) -> Any:
    shifted = Image.new("L", mask.size, 0)
    width, height = mask.size
    source_left = max(0, -dx)
    source_top = max(0, -dy)
    source_right = min(width, width - dx)
    source_bottom = min(height, height - dy)
    if source_left >= source_right or source_top >= source_bottom:
        return shifted
    shifted.paste(
        mask.crop((source_left, source_top, source_right, source_bottom)),
        (source_left + dx, source_top + dy),
    )
    return shifted


def _paste_mask_color(
    *,
    image: Any,
    mask: Any,
    color: tuple[int, int, int],
) -> None:
    image.paste(Image.new("RGB", image.size, color), mask=mask)


def _odd_filter_size(value: int) -> int:
    return value if value % 2 == 1 else value + 1


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


def _normalize_bounding_box_line_width(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("bounding box line width must be a positive int")
    if value <= 0:
        raise ValueError("bounding box line width must be a positive int")
    return value


def _normalize_bounding_box_dilation(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("bounding box dilation kernel size must be a positive odd int")
    if value <= 0 or value % 2 == 0:
        raise ValueError("bounding box dilation kernel size must be a positive odd int")
    return value


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


def vllm_video_content(
    images: Sequence[Any],
    *,
    size: str | tuple[int, int] | None = None,
    resample: str = "nearest",
    mime_type: str = "video/jpeg",
) -> list[dict[str, Any]]:
    """Return one vLLM OpenAI-compatible pre-extracted video frame item."""

    if mime_type != "video/jpeg":
        raise ValueError(
            "vLLM pre-extracted video frames require video_mime_type=video/jpeg"
        )
    frames = [
        image_to_provider_base64(
            image,
            size=size,
            resample=resample,
            mime_type="image/jpeg",
        )
        for image in images
    ]
    return [
        {
            "type": "video_url",
            "video_url": {
                "url": f"data:{mime_type};base64,{','.join(frames)}",
            },
        }
    ]


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
