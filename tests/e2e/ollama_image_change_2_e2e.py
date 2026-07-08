"""Manual E2E check for image-2 change understanding from image-1 description.

Start Ollama and pull the model before running:

    ollama serve
    ollama pull gemma4:e4b
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from core import (
    ASSISTANT_JSON_PREFILL,
    BBOX_SCHEMA,
    DESCRIPTION_SCHEMA,
    ImageDescriptionCallResult,
    ImageDescriptionConfig,
    annotated_area_image,
    describe_image_with_ollama,
    display_path,
    image_payload,
    load_json_value,
    message_content_or_empty,
    prepare_input_image,
    resolve_output_dir,
    validated_bbox,
)
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    response_usage,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
TARGET_PATH = FIXTURE_DIR / "ls20_seed0_action1_target.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ollama_image_change_2_e2e"

IMAGE_2_BBOX_SCHEMA: dict[str, Any] = {
    **BBOX_SCHEMA,
    "type": ["object", "null"],
    "description": "Current pixel bounding box in Image 2, or null if the area disappeared.",
}

CHANGE_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "image_1_area_id": {
                "type": ["integer", "null"],
                "description": (
                    "ID of the related Image 1 description item, or null for "
                    "a newly appearing element."
                ),
            },
            "image_2_bbox": IMAGE_2_BBOX_SCHEMA,
            "description": {
                "type": "string",
                "description": "Concise description of what changed.",
            },
        },
        "required": ["image_1_area_id", "image_2_bbox", "description"],
        "additionalProperties": False,
    },
}


@dataclass(slots=True)
class ImageChangeConfig:
    """Small config surface consumed by the shared Ollama chat client."""

    host: str | None = None
    think: bool | str | None = False
    keep_alive: int | str | None = "5m"
    format: dict[str, Any] | None = field(
        default_factory=lambda: CHANGE_DESCRIPTION_SCHEMA
    )
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})


def main() -> None:
    args = _parse_args()
    output_dir = resolve_output_dir(args.output_dir, root=ROOT)
    source = Image.open(args.source).convert("RGB")
    target = Image.open(args.target).convert("RGB")
    source_input = prepare_input_image(
        source,
        size=args.input_image_size,
        resample=args.input_image_resample,
    )
    target_input = prepare_input_image(
        target,
        size=args.input_image_size,
        resample=args.input_image_resample,
    )

    description_config = ImageDescriptionConfig(
        host=args.host,
        think=args.think,
        keep_alive=args.keep_alive,
        format=DESCRIPTION_SCHEMA,
        options={"temperature": args.temperature},
    )
    description_client = OllamaChatClient(description_config)
    source_description = describe_image_with_ollama(
        client=description_client,
        model=args.model,
        image=source_input,
        image_path=args.source,
    )

    prompt = _prompt_text(
        args.source,
        args.target,
        image_size=target_input.size,
        image_1_description=source_description.described_areas,
        schema=CHANGE_DESCRIPTION_SCHEMA,
    )
    config = ImageChangeConfig(
        host=args.host,
        think=args.think,
        keep_alive=args.keep_alive,
        format=None if args.plain_text else CHANGE_DESCRIPTION_SCHEMA,
        options={"temperature": args.temperature},
    )
    client = OllamaChatClient(config)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": [image_payload(target_input)],
        }
    ]
    if not args.plain_text:
        messages.append({"role": "assistant", "content": ASSISTANT_JSON_PREFILL})
    response = client.chat(
        model=args.model,
        messages=messages,
    )
    raw_text = message_content_or_empty(response)
    parsed_response = None if args.plain_text else load_json_value(raw_text.strip())
    changes, validation_errors = _validated_changes(
        parsed_response,
        image_1_ids={area["id"] for area in source_description.described_areas},
        image_2_size=target_input.size,
    )
    response_shape = _response_shape(
        raw_text,
        parsed_response=parsed_response,
        prefer_json=not args.plain_text,
    )
    artifacts = _write_artifacts(
        output_dir=output_dir,
        source=source_input,
        target=target_input,
        source_description=source_description,
        changes=changes,
    )

    summary = {
        "model": {
            "backend": "ollama",
            "model": args.model,
            "host": args.host,
            "think": args.think,
            "keep_alive": args.keep_alive,
            "temperature": args.temperature,
        },
        "fixture": {
            "before_image": display_path(args.source, root=ROOT),
            "after_image": display_path(args.target, root=ROOT),
            "input_image_size": args.input_image_size,
            "input_image_resample": args.input_image_resample,
            "model_input_image_size": list(source_input.size),
        },
        "artifacts": artifacts,
        "image_1_description": {
            "prompt": source_description.prompt,
            "assistant_prefill": ASSISTANT_JSON_PREFILL,
            "described_areas": source_description.described_areas,
            "validation_errors": source_description.validation_errors,
            "usage": response_usage(source_description.response_object),
        },
        "prompt": prompt,
        "assistant_prefill": None if args.plain_text else ASSISTANT_JSON_PREFILL,
        "changes": changes,
        "description_request_format": DESCRIPTION_SCHEMA,
        "request_format": None if args.plain_text else CHANGE_DESCRIPTION_SCHEMA,
        "raw_response": raw_text,
        "response_shape": response_shape,
        "validation_errors": validation_errors,
        "usage": {
            "image_1_description": response_usage(source_description.response_object),
            "difference_description": response_usage(response),
        },
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"saved Ollama image-change-2 E2E result to {output_path}")
    if changes:
        for index, change in enumerate(changes, start=1):
            print(f"{index}. {change['description']}")
    else:
        print(f"no valid changes; response_shape={response_shape}")
        for error in validation_errors:
            print(f"- {error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--source", default=str(SOURCE_PATH))
    parser.add_argument("--target", default=str(TARGET_PATH))
    parser.add_argument("--input-image-size", default="256x256")
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help=(
            "Disable the JSON schema response format for the final difference "
            "call and save raw text as the description."
        ),
    )
    return parser.parse_args()


def _prompt_text(
    source: str,
    target: str,
    *,
    image_size: tuple[int, int],
    image_1_description: list[dict[str, Any]],
    schema: dict[str, Any],
) -> str:
    before_name = Path(source).name
    after_name = Path(target).name
    width, height = image_size
    return (
        "You are comparing a before-frame description to an after-frame image.\n"
        "Image 2 is supplied as an image.\n"
        "Image 1 is supplied only as JSON description items. Each item has a stable id, pixel bbox, and text description.\n"
        "Identify areas in Image 2 that changed relative to Image 1 or appearing/disappearing elements.\n"
        "Return one array element per change. For each item, reference the most relevant Image 1 description id, or null for a newly appearing element.\n"
        "Return the current Image 2 pixel bounding box for the changed area, or null if the Image 1 element disappeared.\n"
        "We are looking at relevant changes such as but not limited to: orientation, movement, color, shape, appearance, disappearance.\n"
        "Image 1 description JSON:\n"
        f"{json.dumps(image_1_description, indent=2, sort_keys=True)}\n"
        "Your response must validate against this exact JSON schema:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}"
    )


def _response_shape(
    raw_text: str,
    *,
    parsed_response: Any | None,
    prefer_json: bool,
) -> str:
    text = raw_text.strip()
    if not prefer_json:
        return "plain_text"

    if isinstance(parsed_response, list):
        return "json_array"
    if isinstance(parsed_response, dict) and isinstance(parsed_response.get("items"), list):
        return "json_items_wrapper"
    if parsed_response is not None:
        return "unexpected_json"
    if text:
        return "prose_fallback"
    return "empty"


def _validated_changes(
    parsed_response: Any | None,
    *,
    image_1_ids: set[int],
    image_2_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    if parsed_response is None:
        return [], ["response did not contain parseable JSON"]
    errors: list[str] = []
    if isinstance(parsed_response, dict) and isinstance(parsed_response.get("items"), list):
        parsed_response = parsed_response["items"]
        errors.append("response JSON used object wrapper with items array")
    if not isinstance(parsed_response, list):
        return [], ["response JSON was not an array"]

    changes: list[dict[str, Any]] = []
    for index, item in enumerate(parsed_response, start=1):
        if not isinstance(item, dict):
            errors.append(f"item {index}: expected object")
            continue
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"item {index}: missing non-empty description")
            continue
        image_1_area_id, image_1_error = _validated_area_id(
            item.get("image_1_area_id"),
            label=f"item {index} image_1_area_id",
            valid_ids=image_1_ids,
        )
        image_2_bbox, image_2_error = _validated_optional_bbox(
            item.get("image_2_bbox"),
            label=f"item {index} image_2_bbox",
            image_size=image_2_size,
        )
        if image_1_error is not None:
            errors.append(image_1_error)
        if image_2_error is not None:
            errors.append(image_2_error)
        if image_1_error is not None or image_2_error is not None:
            continue
        if image_1_area_id is None and image_2_bbox is None:
            errors.append(f"item {index}: area id and image_2_bbox cannot both be null")
            continue
        changes.append(
            {
                "image_1_area_id": image_1_area_id,
                "image_2_bbox": image_2_bbox,
                "description": description.strip(),
            }
        )
    return changes, errors


def _validated_area_id(
    value: Any,
    *,
    label: str,
    valid_ids: set[int],
) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"{label}: expected integer id or null"
    if value not in valid_ids:
        return None, f"{label}: id {value} was not present in image description"
    return value, None


def _validated_optional_bbox(
    value: Any,
    *,
    label: str,
    image_size: tuple[int, int],
) -> tuple[dict[str, int] | None, str | None]:
    if value is None:
        return None, None
    return validated_bbox(
        value,
        label=label,
        image_size=image_size,
        scale_normalized_1000=True,
    )


def _write_artifacts(
    *,
    output_dir: Path,
    source: Image.Image,
    target: Image.Image,
    source_description: ImageDescriptionCallResult,
    changes: list[dict[str, Any]],
) -> dict[str, str]:
    source_path = output_dir / "image_1.png"
    target_path = output_dir / "image_2.png"
    source.convert("RGB").save(source_path)
    target.convert("RGB").save(target_path)

    annotated_source_path = output_dir / "image_1_annotated.png"
    annotated_target_path = output_dir / "image_2_change_annotated.png"
    annotated_area_image(source, source_description.described_areas).save(
        annotated_source_path
    )
    _annotated_change_image(target, changes).save(annotated_target_path)
    return {
        "image_1": str(source_path.relative_to(output_dir)),
        "image_2": str(target_path.relative_to(output_dir)),
        "image_1_annotated": str(annotated_source_path.relative_to(output_dir)),
        "image_2_change_annotated": str(annotated_target_path.relative_to(output_dir)),
    }


def _annotated_change_image(
    image: Image.Image,
    changes: list[dict[str, Any]],
) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for index, change in enumerate(changes, start=1):
        bbox = change.get("image_2_bbox")
        if bbox is None:
            continue
        rectangle = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
        draw.rectangle(rectangle, outline=(0, 255, 0), width=3)
        label_position = (bbox["x0"] + 1, bbox["y0"] + 1)
        label = str(index)
        text_bbox = draw.textbbox(label_position, label)
        draw.rectangle(text_bbox, fill=(0, 255, 0))
        draw.text(label_position, label, fill=(0, 0, 0))
    return annotated


if __name__ == "__main__":
    main()
