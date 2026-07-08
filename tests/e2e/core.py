"""Shared helpers for manual E2E scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageDraw

from face_of_agi.frames import image_to_base64_png
from face_of_agi.models.providers.ollama import OllamaChatClient, object_get

ASSISTANT_JSON_PREFILL = "```json\n"

BBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x0": {"type": "number", "description": "Top-left pixel x coordinate."},
        "y0": {"type": "number", "description": "Top-left pixel y coordinate."},
        "x1": {"type": "number", "description": "Bottom-right pixel x coordinate."},
        "y1": {"type": "number", "description": "Bottom-right pixel y coordinate."},
    },
    "required": ["x0", "y0", "x1", "y1"],
    "additionalProperties": False,
}

DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "bbox": BBOX_SCHEMA,
            "description": {
                "type": "string",
                "description": "Concise description of this image area.",
            },
        },
        "required": ["bbox", "description"],
        "additionalProperties": False,
    },
}


@dataclass(slots=True)
class ImageDescriptionConfig:
    """Small config surface consumed by the shared Ollama chat client."""

    host: str | None = None
    think: bool | str | None = False
    keep_alive: int | str | None = "5m"
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})


@dataclass(slots=True)
class ImageDescriptionCallResult:
    """Structured result from the shared single-image description call."""

    prompt: str
    raw_text: str
    parsed_response: Any | None
    described_areas: list[dict[str, Any]]
    validation_errors: list[str]
    response_object: Any


def resolve_output_dir(output_dir: str, *, root: Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: str, *, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def prepare_input_image(
    image: Image.Image,
    *,
    size: str | None,
    resample: str,
) -> Image.Image:
    target_size = input_image_size(size)
    if target_size is None or image.size == target_size:
        return image.convert("RGB")
    return image.convert("RGB").resize(target_size, resampling_filter(resample))


def input_image_size(size: str | None) -> tuple[int, int] | None:
    if not size:
        return None
    width, height = size.lower().split("x", 1)
    parsed = (int(width), int(height))
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError(f"input image size must be positive, got {size!r}")
    return parsed


def resampling_filter(resample: str) -> Image.Resampling:
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return filters[resample]


def describe_image_with_ollama(
    *,
    client: OllamaChatClient,
    model: str,
    image: Image.Image,
    image_path: str,
) -> ImageDescriptionCallResult:
    """Run the shared single-image description request."""

    prompt = image_description_prompt(
        image_path,
        image_size=image.size,
    )
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_payload(image)],
            },
            {"role": "assistant", "content": ASSISTANT_JSON_PREFILL},
        ],
    )
    raw_text = message_content_or_empty(response).strip()
    parsed_response = load_json_value(raw_text)
    described_areas, validation_errors = validated_areas(
        parsed_response,
        image_size=image.size,
    )
    return ImageDescriptionCallResult(
        prompt=prompt,
        raw_text=raw_text,
        parsed_response=parsed_response,
        described_areas=described_areas,
        validation_errors=validation_errors,
        response_object=response,
    )


def image_description_prompt(
    image_path: str,
    *,
    image_size: tuple[int, int],
) -> str:
    del image_path, image_size
    return (
        "Describe this frame accurately and concisely.\n"
        "Return an array of identifiable areas with their bounding box on the image.\n"
        "Areas can contain visible objects, colors, positions, layout, background, shapes and other conceptually identifiable things.\n"
        "Bounding boxes can overlap." 
        "Overlap shall be considered only when areas are significantly different concept example: object within background.\n"
        "Area descriptions shall be exhaustive: "
        "precise colors, shape patterns, orientation and other identifiable concept within those areas."
        "Your response must validate against this exact JSON schema:\n"
        f"{json.dumps(DESCRIPTION_SCHEMA, indent=2, sort_keys=True)}"
    )


def image_payload(image: Image.Image) -> str:
    return image_to_base64_png(image)


def message_content_or_empty(response: Any) -> str:
    message = object_get(response, "message") or {}
    content = object_get(message, "content")
    if isinstance(content, str):
        return content
    return ""


def validated_areas(
    parsed_response: Any | None,
    *,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    if parsed_response is None:
        return [], ["response did not contain parseable JSON"]
    errors: list[str] = []
    if isinstance(parsed_response, dict) and isinstance(parsed_response.get("items"), list):
        parsed_response = parsed_response["items"]
        errors.append("response JSON used object wrapper with items array")
    if not isinstance(parsed_response, list):
        return [], ["response JSON was not an array"]

    areas: list[dict[str, Any]] = []
    for index, item in enumerate(parsed_response, start=1):
        if not isinstance(item, dict):
            errors.append(f"item {index}: expected object")
            continue
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"item {index}: missing non-empty description")
            continue
        item, coercion_error = coerce_common_area_response(item, image_size=image_size)
        if coercion_error is not None:
            errors.append(f"item {index}: {coercion_error}")
        bbox, bbox_error = validated_bbox(
            item.get("bbox"),
            label=f"item {index} bbox",
            image_size=image_size,
            scale_normalized_1000=True,
        )
        if bbox_error is not None:
            errors.append(bbox_error)
        if bbox is None:
            continue
        areas.append(
            {
                "id": len(areas),
                "bbox": bbox,
                "description": description.strip(),
            }
        )
    return areas, errors


def coerce_common_area_response(
    item: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> tuple[dict[str, Any], str | None]:
    if "bbox" in item:
        return item, None

    box_2d = item.get("box_2d")
    if not isinstance(box_2d, list) or len(box_2d) != 4:
        return item, None
    bbox = box_2d_to_bbox(box_2d, image_size)
    if bbox is None:
        return item, "box_2d could not be converted to pixel bbox"
    return {**item, "bbox": bbox}, "coerced box_2d to bbox"


def box_2d_to_bbox(
    value: list[Any],
    image_size: tuple[int, int],
) -> dict[str, int] | None:
    if any(isinstance(coordinate, bool) for coordinate in value):
        return None
    if not all(isinstance(coordinate, (int, float)) for coordinate in value):
        return None
    if not all(math.isfinite(float(coordinate)) for coordinate in value):
        return None

    y0, x0, y1, x1 = [float(coordinate) for coordinate in value]
    width, height = image_size
    if max(x0, y0, x1, y1) > max(width, height):
        scale_x = width / 1000
        scale_y = height / 1000
        x0 *= scale_x
        x1 *= scale_x
        y0 *= scale_y
        y1 *= scale_y

    return {
        "x0": clamp_pixel(round(x0), width),
        "y0": clamp_pixel(round(y0), height),
        "x1": clamp_pixel(round(x1), width),
        "y1": clamp_pixel(round(y1), height),
    }


def clamp_pixel(value: int, size: int) -> int:
    return max(0, min(value, size - 1))


def validated_bbox(
    value: Any,
    *,
    label: str,
    image_size: tuple[int, int],
    scale_normalized_1000: bool,
) -> tuple[dict[str, int] | None, str | None]:
    if not isinstance(value, dict):
        return None, f"{label}: expected object"
    coordinates: dict[str, int] = {}
    for key in ("x0", "y0", "x1", "y1"):
        coordinate = value.get(key)
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            return None, f"{label}.{key}: expected numeric pixel coordinate"
        coordinates[key] = round(float(coordinate))

    width, height = image_size
    if scale_normalized_1000 and looks_normalized_1000(coordinates, image_size):
        coordinates = scale_normalized_bbox(coordinates, image_size)
        coercion = f"{label}: scaled 0..1000 normalized bbox to input image pixels"
    else:
        coercion = None

    if not (0 <= coordinates["x0"] < width and 0 <= coordinates["x1"] < width):
        return None, f"{label}: x coordinates must be in 0..{width - 1}"
    if not (0 <= coordinates["y0"] < height and 0 <= coordinates["y1"] < height):
        return None, f"{label}: y coordinates must be in 0..{height - 1}"
    if coordinates["x0"] > coordinates["x1"] or coordinates["y0"] > coordinates["y1"]:
        return None, f"{label}: x0/y0 must be above and left of x1/y1"
    return coordinates, coercion


def looks_normalized_1000(
    coordinates: dict[str, int],
    image_size: tuple[int, int],
) -> bool:
    width, height = image_size
    values = tuple(coordinates.values())
    exceeds_image = (
        coordinates["x0"] >= width
        or coordinates["x1"] >= width
        or coordinates["y0"] >= height
        or coordinates["y1"] >= height
    )
    return exceeds_image and all(0 <= value <= 1000 for value in values)


def scale_normalized_bbox(
    coordinates: dict[str, int],
    image_size: tuple[int, int],
) -> dict[str, int]:
    width, height = image_size
    return {
        "x0": clamp_pixel(round(coordinates["x0"] * width / 1000), width),
        "y0": clamp_pixel(round(coordinates["y0"] * height / 1000), height),
        "x1": clamp_pixel(round(coordinates["x1"] * width / 1000), width),
        "y1": clamp_pixel(round(coordinates["y1"] * height / 1000), height),
    }


def annotated_area_image(image: Image.Image, areas: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for index, area in enumerate(areas, start=1):
        label = str(area.get("id", index - 1))
        bbox = area["bbox"]
        rectangle = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
        draw.rectangle(rectangle, outline=(0, 255, 0), width=3)
        label_position = (bbox["x0"] + 1, bbox["y0"] + 1)
        text_bbox = draw.textbbox(label_position, label)
        draw.rectangle(text_bbox, fill=(0, 255, 0))
        draw.text(label_position, label, fill=(0, 0, 0))
    return annotated


def load_json_value(text: str) -> Any | None:
    for candidate in json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def json_candidates(text: str) -> tuple[str, ...]:
    candidates = [text]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL):
        candidates.append(match.group(1).strip())
    for opening, closing in (("[", "]"), ("{", "}")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump())
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return repr(value)
