"""Manual E2E check for structured Ollama image-area description.

Start Ollama and pull the model before running:

    ollama serve
    ollama pull gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from core import (
    ASSISTANT_JSON_PREFILL,
    DESCRIPTION_SCHEMA,
    ImageDescriptionConfig,
    annotated_area_image,
    describe_image_with_ollama,
    display_path,
    jsonable,
    prepare_input_image,
    resolve_output_dir,
)
from face_of_agi.models.providers.ollama import OllamaChatClient, response_usage

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ollama_image_description_e2e"


def main() -> None:
    args = _parse_args()
    output_dir = resolve_output_dir(args.output_dir, root=ROOT)
    image = Image.open(args.image).convert("RGB")
    input_image = prepare_input_image(
        image,
        size=args.input_image_size,
        resample=args.input_image_resample,
    )

    config = ImageDescriptionConfig(
        host=args.host,
        think=args.think,
        keep_alive=args.keep_alive,
        format=DESCRIPTION_SCHEMA,
        options={"temperature": args.temperature},
    )
    client = OllamaChatClient(config)
    description_result = describe_image_with_ollama(
        client=client,
        model=args.model,
        image=input_image,
        image_path=args.image,
    )
    described_areas = description_result.described_areas

    input_path = output_dir / "input_image.png"
    annotated_path = output_dir / "input_image_annotated.png"
    input_image.save(input_path)
    annotated_area_image(input_image, described_areas).save(annotated_path)
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
            "image": display_path(args.image, root=ROOT),
            "input_image_size": args.input_image_size,
            "input_image_resample": args.input_image_resample,
            "model_input_image_size": list(input_image.size),
        },
        "artifacts": {
            "input_image": input_path.name,
            "input_image_annotated": annotated_path.name,
        },
        "prompt": description_result.prompt,
        "assistant_prefill": ASSISTANT_JSON_PREFILL,
        "description": description_result.raw_text,
        "described_areas": described_areas,
        "request_format": DESCRIPTION_SCHEMA,
        "raw_response_object": jsonable(description_result.response_object),
        "validation_errors": description_result.validation_errors,
        "usage": response_usage(description_result.response_object),
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"saved Ollama image-description E2E result to {output_path}")
    if described_areas:
        for index, area in enumerate(described_areas, start=1):
            print(f"{index}. {area['description']}")
    else:
        print(description_result.raw_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--image", default=str(SOURCE_PATH))
    parser.add_argument("--input-image-size", default="256x256")
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
