"""Manual E2E check for Ollama goal prediction description predictions.

Start Ollama and pull the model before running:

    ollama serve
    ollama pull gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from core import annotated_area_image, prepare_input_image, resolve_output_dir
from face_of_agi.contracts import Observation, RoleContext
from face_of_agi.frames import to_memory_jsonable
from face_of_agi.models.goal import GoalPredictionAdapter, OllamaDescriptionConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ollama_goal_description_e2e"


def main() -> None:
    args = _parse_args()
    output_dir = resolve_output_dir(args.output_dir, root=ROOT)
    image = prepare_input_image(
        Image.open(args.image).convert("RGB"),
        size=args.input_image_size,
        resample=args.input_image_resample,
    )
    adapter = GoalPredictionAdapter(
        OllamaDescriptionConfig(
            model=args.model,
            host=args.host,
            think=args.think,
            keep_alive=args.keep_alive,
            options={"temperature": args.temperature},
            input_image_size=args.input_image_size,
            input_image_resample=args.input_image_resample,
        )
    )
    result = adapter.predict(
        context=RoleContext(game=args.context),
        observation=Observation(id="manual-goal-source", step=0, frame=image),
    )
    annotated_path = output_dir / "prediction_annotated.png"
    annotated_area_image(image, result.predicted_description).save(annotated_path)
    summary = {"prediction_result": to_memory_jsonable(result)}
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"saved Ollama goal-description E2E result to {output_path}")
    print(json.dumps(summary["prediction_result"], indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--image", default=str(SOURCE_PATH))
    parser.add_argument("--context", default="Unknown objective.")
    parser.add_argument("--input-image-size", default="256x256")
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
