"""Helpers for dashboard model-input debug records."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
import math
from typing import Any

MODEL_INPUT_SLOTS: tuple[tuple[str, str], ...] = (
    ("agent", "Agent X"),
    ("world", "World S"),
    ("goal", "Goal G"),
    ("updater_world", "World Updater"),
    ("updater_goal", "Goal Updater"),
    ("updater_agent", "Agent Updater"),
)


@dataclass(frozen=True, slots=True)
class SentImage:
    """One provider-normalized image payload extracted for display."""

    label: str
    image: Any | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    """Provider response fields normalized for dashboard display."""

    text: str | None
    parsed_json: Any | None
    metadata: dict[str, Any]
    raw_response: Any | None

    @property
    def available(self) -> bool:
        """Return whether any provider output data was captured."""

        return (
            self.text is not None
            or self.raw_response is not None
            or bool(self.metadata)
        )


@dataclass(frozen=True, slots=True)
class PredictionOverlay:
    """Annotated provider output preview for structured description predictions."""

    image: Any | None
    source_label: str | None
    source_size: tuple[int, int] | None
    drawn_count: int
    area_count: int
    warnings: tuple[str, ...] = ()


def token_counts(value: Any) -> dict[str, int | None]:
    """Return provider-reported input/output/total token counts."""

    usage = value.get("usage") if isinstance(value, dict) else value
    totals = _token_counts_from_usage(usage)
    return {
        "input": totals.get("input"),
        "output": totals.get("output"),
        "total": totals.get("total"),
    }


def records_for_slot(
    records: list[dict[str, Any]],
    slot: str,
) -> list[dict[str, Any]]:
    """Return records for one model-call slot in persisted order."""

    return [record for record in records if record.get("call_slot") == slot]


def sent_images(record: dict[str, Any]) -> list[SentImage]:
    """Return provider image payloads decoded from one debug record."""

    request = _dict(record.get("request"))
    images: list[SentImage] = []

    for item_index, item in enumerate(_list(request.get("input")), start=1):
        item_payload = _dict(item)
        for content_index, part in enumerate(
            _list(item_payload.get("content")),
            start=1,
        ):
            part_payload = _dict(part)
            if part_payload.get("type") != "input_image":
                continue
            images.append(
                _decode_openai_image(
                    part_payload.get("image_url"),
                    label=f"Input item {item_index} image {content_index}",
                )
            )

    for message_index, message in enumerate(_list(request.get("messages")), start=1):
        message_payload = _dict(message)
        for content_index, part in enumerate(
            _list(message_payload.get("content")),
            start=1,
        ):
            part_payload = _dict(part)
            if part_payload.get("type") != "image_url":
                continue
            images.append(
                _decode_vllm_image_url(
                    part_payload.get("image_url"),
                    label=f"Message {message_index} content {content_index}",
                )
            )

        for image_index, image_payload in enumerate(
            _list(message_payload.get("images")),
            start=1,
        ):
            images.append(
                _decode_ollama_image(
                    image_payload,
                    label=f"Message {message_index} image {image_index}",
                )
            )

    return images


def provider_output(record: dict[str, Any]) -> ProviderOutput:
    """Return provider output data captured in a model-input debug record."""

    metadata = _dict(record.get("metadata"))
    text = _provider_output_text(metadata)
    return ProviderOutput(
        text=text,
        parsed_json=_parse_json_text(text),
        metadata=_provider_response_metadata(metadata),
        raw_response=metadata.get("response_payload"),
    )


def prediction_overlay(
    record: dict[str, Any],
    *,
    display_size: tuple[int, int] | None = None,
) -> PredictionOverlay:
    """Return predicted-description boxes drawn over the first sent image."""

    output = provider_output(record)
    areas, area_warning = _description_areas(output.parsed_json)
    if areas is None:
        return PredictionOverlay(
            image=None,
            source_label=None,
            source_size=None,
            drawn_count=0,
            area_count=0,
            warnings=(area_warning,),
        )

    base_image = _first_available_image(sent_images(record))
    if base_image is None:
        return PredictionOverlay(
            image=None,
            source_label=None,
            source_size=None,
            drawn_count=0,
            area_count=len(areas),
            warnings=("No decoded input image available for bbox overlay.",),
        )

    source = base_image.image.convert("RGB")
    source_size = source.size
    annotated = _display_image(source, display_size=display_size)
    coordinate_space = _visual_coordinate_space(record)
    warnings: list[str] = []
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, area in enumerate(areas, start=1):
        bbox, warning = _overlay_bbox(
            area.get("bbox_2d"),
            source_image_size=source_size,
            draw_image_size=annotated.size,
            coordinate_space=coordinate_space,
            label=f"item {index} bbox_2d",
        )
        if warning is not None:
            warnings.append(warning)
            continue
        if bbox is None:
            continue
        boxes.append((str(index), bbox))

    if boxes:
        _draw_labeled_boxes(annotated, boxes)
    if not boxes and areas:
        warnings.append("No valid bounding boxes found in provider output.")

    return PredictionOverlay(
        image=annotated,
        source_label=base_image.label,
        source_size=source_size,
        drawn_count=len(boxes),
        area_count=len(areas),
        warnings=tuple(warnings),
    )


def _token_counts_from_usage(value: Any) -> dict[str, int | None]:
    if isinstance(value, list):
        merged = {"input": 0, "output": 0, "total": 0}
        seen = {"input": False, "output": False, "total": False}
        for item in value:
            counts = _token_counts_from_usage(item)
            for key in merged:
                if counts.get(key) is None:
                    continue
                seen[key] = True
                merged[key] += int(counts[key])
        return {
            key: merged[key] if seen[key] else None
            for key in merged
        }

    if not isinstance(value, dict):
        return {"input": None, "output": None, "total": None}

    input_tokens = _optional_int(
        value.get(
            "input_tokens",
            value.get("prompt_tokens", value.get("prompt_eval_count")),
        )
    )
    output_tokens = _optional_int(
        value.get(
            "output_tokens",
            value.get("completion_tokens", value.get("eval_count")),
        )
    )
    total_tokens = _optional_int(value.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _decode_openai_image(value: Any, *, label: str) -> SentImage:
    if not isinstance(value, str) or not value:
        return SentImage(label=label, image=None, error="No image_url string found.")
    if not value.startswith("data:"):
        return SentImage(
            label=label,
            image=None,
            error="Only inline data URL image payloads can be rendered.",
        )
    return _decode_base64_image(_data_url_base64(value), label=label)


def _decode_ollama_image(value: Any, *, label: str) -> SentImage:
    if not isinstance(value, str) or not value:
        return SentImage(label=label, image=None, error="No base64 image string found.")
    if value.startswith("data:"):
        value = _data_url_base64(value)
    return _decode_base64_image(value, label=label)


def _decode_vllm_image_url(value: Any, *, label: str) -> SentImage:
    payload = _dict(value)
    image_url = payload.get("url") if payload else value
    if not isinstance(image_url, str) or not image_url:
        return SentImage(label=label, image=None, error="No image_url URL found.")
    if not image_url.startswith("data:"):
        return SentImage(
            label=label,
            image=None,
            error="Only inline data URL image payloads can be rendered.",
        )
    return _decode_base64_image(_data_url_base64(image_url), label=label)


def _decode_base64_image(value: str, *, label: str) -> SentImage:
    if not value:
        return SentImage(label=label, image=None, error="No base64 image data found.")
    try:
        from PIL import Image

        encoded = "".join(value.split())
        image = Image.open(BytesIO(base64.b64decode(encoded, validate=True)))
        return SentImage(label=label, image=image.convert("RGB"))
    except Exception as exc:
        return SentImage(
            label=label,
            image=None,
            error=f"Could not decode image payload: {exc}",
        )


def _data_url_base64(value: str) -> str:
    prefix, separator, encoded = value.partition(",")
    if not separator or ";base64" not in prefix:
        return ""
    return encoded


def _provider_output_text(metadata: dict[str, Any]) -> str | None:
    for key in ("response_output_text", "response_text"):
        value = metadata.get(key)
        if isinstance(value, str):
            return value

    payload = _dict(metadata.get("response_payload"))
    message = _dict(payload.get("message"))
    content = message.get("content")
    if isinstance(content, str):
        return content
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    return None


def _parse_json_text(value: str | None) -> Any | None:
    if value is None:
        return None
    text = _strip_json_fence(value.strip())
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _strip_json_fence(value: str) -> str:
    if value.startswith("```json"):
        value = value.removeprefix("```json").strip()
    if value.startswith("```"):
        value = value.removeprefix("```").strip()
    if value.endswith("```"):
        value = value.removesuffix("```").strip()
    return value


def _provider_response_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    nested = _compact_dict(_dict(metadata.get("response_metadata")))
    if nested:
        return nested

    keys = (
        "response_id",
        "response_model",
        "response_status",
        "usage",
        "incomplete_details",
        "done_reason",
    )
    return _compact_dict({key: metadata.get(key) for key in keys})


def _description_areas(value: Any) -> tuple[list[dict[str, Any]] | None, str]:
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        value = value["items"]
    if isinstance(value, dict) and isinstance(
        value.get("predicted_description"),
        list,
    ):
        value = value["predicted_description"]
    if not isinstance(value, list):
        return None, "Provider output is not a JSON array; no bbox overlay available."
    return [_dict(item) for item in value], ""


def _first_available_image(images: list[SentImage]) -> SentImage | None:
    for image in images:
        if image.image is not None:
            return image
    return None


def _display_image(image: Any, *, display_size: tuple[int, int] | None) -> Any:
    if display_size is None or image.size == display_size:
        return image.copy()

    from PIL import Image

    return image.resize(display_size, Image.Resampling.NEAREST)


def _visual_coordinate_space(record: dict[str, Any]) -> str | None:
    metadata = _dict(record.get("metadata"))
    value = metadata.get("visual_coordinate_space")
    if isinstance(value, str) and value:
        return value

    nested = _dict(metadata.get("response_metadata"))
    value = nested.get("visual_coordinate_space")
    if isinstance(value, str) and value:
        return value
    return None


def _overlay_bbox(
    value: Any,
    *,
    source_image_size: tuple[int, int],
    draw_image_size: tuple[int, int],
    coordinate_space: str | None,
    label: str,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    coordinates, error = _bbox_coordinates(value, label=label)
    if error is not None:
        return None, error

    scaled_normalized = coordinate_space == "normalized_1000" or (
        coordinate_space is None
        and _looks_normalized_1000(coordinates, image_size=source_image_size)
    )
    if scaled_normalized:
        coordinates = _scale_normalized_1000(
            coordinates,
            image_size=draw_image_size,
        )
    else:
        coordinates, warning = _scale_pixel_bbox(
            coordinates,
            source_image_size=source_image_size,
            draw_image_size=draw_image_size,
            label=label,
        )
        if warning is not None:
            return None, warning

    x0, y0, x1, y1 = (
        coordinates["x0"],
        coordinates["y0"],
        coordinates["x1"],
        coordinates["y1"],
    )
    if x1 < x0 or y1 < y0:
        return None, f"{label}: bottom-right must be greater than top-left"
    return (x0, y0, x1, y1), None


def _bbox_coordinates(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, float], str | None]:
    if isinstance(value, list):
        return _bbox_array_coordinates(value, label=label)
    return {}, f"{label}: expected array [x0, y0, x1, y1]"


def _bbox_array_coordinates(
    value: list[Any],
    *,
    label: str,
) -> tuple[dict[str, float], str | None]:
    if len(value) != 4:
        return {}, f"{label}: expected 4 coordinates [x0, y0, x1, y1]"
    coordinates: dict[str, float] = {}
    for index, key in enumerate(("x0", "y0", "x1", "y1")):
        number, error = _finite_number(value[index], label=f"{label}[{index}]")
        if error is not None:
            return {}, error
        coordinates[key] = number
    return coordinates, None


def _finite_number(value: Any, *, label: str) -> tuple[float, str | None]:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return 0.0, f"{label}: expected finite number"
    return float(value), None


def _looks_normalized_1000(
    coordinates: dict[str, float],
    *,
    image_size: tuple[int, int],
) -> bool:
    width, height = image_size
    exceeds_image = (
        coordinates["x0"] >= width
        or coordinates["x1"] >= width
        or coordinates["y0"] >= height
        or coordinates["y1"] >= height
    )
    return exceeds_image and all(0 <= value <= 1000 for value in coordinates.values())


def _scale_normalized_1000(
    coordinates: dict[str, float],
    *,
    image_size: tuple[int, int],
) -> dict[str, float]:
    width, height = image_size
    return {
        "x0": _clamp(coordinates["x0"] * width / 1000, width),
        "y0": _clamp(coordinates["y0"] * height / 1000, height),
        "x1": _clamp(coordinates["x1"] * width / 1000, width),
        "y1": _clamp(coordinates["y1"] * height / 1000, height),
    }


def _scale_pixel_bbox(
    coordinates: dict[str, float],
    *,
    source_image_size: tuple[int, int],
    draw_image_size: tuple[int, int],
    label: str,
) -> tuple[dict[str, float], str | None]:
    source_width, source_height = source_image_size
    if (
        not 0 <= coordinates["x0"] <= source_width
        or not 0 <= coordinates["x1"] <= source_width
        or not 0 <= coordinates["y0"] <= source_height
        or not 0 <= coordinates["y1"] <= source_height
    ):
        return (
            coordinates,
            f"{label}: coordinates outside image bounds "
            f"{source_width}x{source_height}",
        )

    draw_width, draw_height = draw_image_size
    scale_x = draw_width / source_width
    scale_y = draw_height / source_height
    return (
        {
            "x0": _clamp(coordinates["x0"] * scale_x, draw_width),
            "y0": _clamp(coordinates["y0"] * scale_y, draw_height),
            "x1": _clamp(coordinates["x1"] * scale_x, draw_width),
            "y1": _clamp(coordinates["y1"] * scale_y, draw_height),
        },
        None,
    )


def _clamp(value: float, size: int) -> float:
    return max(0.0, min(float(round(value)), float(size)))


def _draw_labeled_boxes(
    image: Any,
    boxes: list[tuple[str, tuple[float, float, float, float]]],
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for label, bbox in boxes:
        rectangle = tuple(int(value) for value in bbox)
        draw.rectangle(rectangle, outline=(0, 255, 0), width=3)
        label_position = (rectangle[0] + 1, rectangle[1] + 1)
        text_bbox = draw.textbbox(label_position, label)
        draw.rectangle(text_bbox, fill=(0, 255, 0))
        draw.text(label_position, label, fill=(0, 0, 0))


def _compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != {} and item != []
    }


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []
