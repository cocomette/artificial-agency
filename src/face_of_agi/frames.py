"""Shared frame and image helpers.

The runtime stores visual observations by reference. These helpers keep that
storage path small and provider-neutral while still allowing refs to rehydrate
to real images for later model calls.
"""

from __future__ import annotations

import base64
from dataclasses import fields, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from face_of_agi.contracts import Observation, ToolResult

FRAME_PAYLOAD_TYPE = "face_of_agi.frame.png_base64.v1"
DEFAULT_MEMORY_IMAGE_SIZE = (64, 64)


def observation_to_pil_image(
    observation: Observation,
    *,
    frame_scale: int = 4,
) -> Any:
    """Return the visible observation frame as a PIL RGB image."""

    frame = observation.frame
    if frame is None and observation.frames:
        frame = observation.frames[-1]
    if frame is None:
        raise ValueError(f"observation '{observation.id}' does not contain a frame")
    return frame_to_pil_image(
        frame,
        step=observation.step,
        frame_scale=frame_scale,
        label=observation.id,
    )


def frame_to_pil_image(
    frame: Any,
    *,
    step: int = 0,
    frame_scale: int = 4,
    label: str = "frame",
) -> Any:
    """Normalize a PIL/numpy/grid frame into a PIL RGB image."""

    from PIL import Image
    import numpy as np

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    array = np.asarray(frame)
    if array.ndim == 2:
        from arc_agi.rendering import frame_to_rgb_array

        rgb_array = frame_to_rgb_array(
            steps=step,
            frame=array,
            scale=frame_scale,
        )
        return Image.fromarray(rgb_array).convert("RGB")

    if array.ndim == 3 and array.shape[2] in {3, 4}:
        return Image.fromarray(array.astype("uint8")).convert("RGB")

    raise ValueError(f"{label!r} cannot be converted to an RGB image")


def image_to_base64_png(
    image: Any,
    *,
    size: tuple[int, int] | None = None,
    resample: str = "nearest",
) -> str:
    """Encode a PIL-compatible image as base64 PNG."""

    image = _resize_image_if_needed(image.convert("RGB"), size=size, resample=resample)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_data_url(
    image: Any,
    *,
    size: tuple[int, int] | None = None,
    resample: str = "nearest",
) -> str:
    """Encode an image as a PNG data URL for provider APIs."""

    encoded = image_to_base64_png(image, size=size, resample=resample)
    return f"data:image/png;base64,{encoded}"


def image_from_base64_png(encoded: str) -> Any:
    """Decode base64 or data-URL PNG data into a PIL RGB image."""

    from PIL import Image

    if encoded.startswith("data:"):
        _, encoded = encoded.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def normalize_frame_for_memory(
    frame: Any,
    *,
    size: tuple[int, int] = DEFAULT_MEMORY_IMAGE_SIZE,
    frame_scale: int = 4,
) -> Any:
    """Return the 64x64 image that memory will persist, or the value unchanged."""

    try:
        image = frame_to_pil_image(frame, frame_scale=frame_scale)
    except Exception:
        return frame
    return _resize_image_if_needed(image, size=size, resample="nearest")


def to_memory_jsonable(value: Any) -> Any:
    """Convert runtime objects into JSON-safe payloads for SQLite."""

    image_payload = _try_serialize_frame(value)
    if image_payload is not None:
        return image_payload

    if isinstance(value, Observation):
        return {
            "id": value.id,
            "step": value.step,
            "frame": to_memory_jsonable(value.frame),
            "frames": [to_memory_jsonable(frame) for frame in value.frames],
            "raw_frame_data": _json_fallback(value.raw_frame_data),
            "metadata": to_memory_jsonable(value.metadata),
        }

    if isinstance(value, ToolResult):
        return {
            "id": value.id,
            "tool": value.tool,
            "predicted_observation": to_memory_jsonable(value.predicted_observation),
            "source_observation_ref": to_memory_jsonable(value.source_observation_ref),
            "action": to_memory_jsonable(value.action),
            "explanation": value.explanation,
            "metadata": to_memory_jsonable(value.metadata),
        }

    if is_dataclass(value):
        return {
            field.name: to_memory_jsonable(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, dict):
        return {str(key): to_memory_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_memory_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return _json_fallback(value)


def from_memory_jsonable(value: Any) -> Any:
    """Rehydrate JSON payloads produced by `to_memory_jsonable`."""

    if isinstance(value, dict):
        if value.get("__type__") == FRAME_PAYLOAD_TYPE:
            return image_from_base64_png(str(value["data"]))
        return {key: from_memory_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [from_memory_jsonable(item) for item in value]

    return value


def _try_serialize_frame(value: Any) -> dict[str, Any] | None:
    """Return a compact image payload when a value is visually serializable."""

    try:
        image = normalize_frame_for_memory(value)
        if image is value:
            return None
    except Exception:
        return None

    encoded = image_to_base64_png(image)
    return {
        "__type__": FRAME_PAYLOAD_TYPE,
        "mime_type": "image/png",
        "encoding": "base64",
        "width": image.width,
        "height": image.height,
        "data": encoded,
    }


def _resize_image_if_needed(
    image: Any,
    *,
    size: tuple[int, int] | None,
    resample: str,
) -> Any:
    """Resize a PIL image when requested."""

    if size is None or image.size == size:
        return image

    from PIL import Image

    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return image.resize(size, filters[resample])


def _json_fallback(value: Any) -> Any:
    """Keep JSON-native values and stringify framework-specific objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        import json

        json.dumps(value)
    except TypeError:
        return repr(value)
    return value
