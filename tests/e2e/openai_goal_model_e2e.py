"""Manual E2E check for OpenAI goal prediction description predictions.

This script calls the real OpenAI API. Set OPENAI_API_KEY before running it.
It asks G for goal-relevant area descriptions from one ARC source frame and
writes the resulting prediction result under runs/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import annotated_area_image, prepare_input_image, resolve_output_dir
from face_of_agi.contracts import Observation, RoleContext
from face_of_agi.frames import to_memory_jsonable
from face_of_agi.models.goal import (
    GoalPredictionAdapter,
    OpenAIDescriptionConfig,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "openai_goal_model_e2e"


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    output_dir = resolve_output_dir(args.output_dir, root=ROOT)

    image = prepare_input_image(
        Image.open(args.image).convert("RGB"),
        size=args.input_image_size,
        resample=args.input_image_resample,
    )
    observation = Observation(id="ls20-seed0-step0", step=0, frame=image)
    context = RoleContext(
        general=(
            "Infer goal-relevant visual evidence from the current ARC frame. "
            "Focus on progress signals, not a specific transition action."
        ),
        game=(
            "This E2E check uses game ls20-9607627b, seed 0, source step 0. "
            "Return only structured descriptions of goal-relevant areas."
        ),
    )
    adapter = GoalPredictionAdapter(_config_from_args(args))

    result = adapter.predict(context=context, observation=observation)
    _validate_no_action_prompt(adapter.last_prompt or "")
    _validate_description(result.predicted_description)
    annotated_path = output_dir / "prediction_annotated.png"
    annotated_area_image(image, result.predicted_description).save(annotated_path)

    summary = {"prediction_result": to_memory_jsonable(result)}
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"saved OpenAI goal-description E2E result to {output_path}")
    print(json.dumps(summary["prediction_result"], indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Dotenv file to load before calling OpenAI. Use an empty value to disable.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--input-image-detail", default="auto")
    parser.add_argument("--input-image-size", default="1024x1024")
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--image", default=str(SOURCE_PATH))
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> OpenAIDescriptionConfig:
    return OpenAIDescriptionConfig(
        api_key_env=args.api_key_env,
        model=args.model,
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        input_image_detail=args.input_image_detail,
        input_image_size=args.input_image_size,
        input_image_resample=args.input_image_resample,
        timeout=args.timeout,
        max_retries=args.max_retries,
        metadata={"role": "goal", "script": "openai_goal_model_e2e"},
    )


def _load_env_file(env_file: str) -> None:
    """Load a dotenv file when present without printing secret values."""

    if not env_file:
        return
    path = Path(env_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _validate_no_action_prompt(prompt: str) -> None:
    if "PROPOSED ACTION" in prompt or "action_id:" in prompt:
        raise RuntimeError("goal prompt unexpectedly includes action text")


def _validate_description(description: object) -> None:
    if not isinstance(description, list) or not description:
        raise RuntimeError("prediction must be a non-empty description array")
    for index, area in enumerate(description, start=1):
        if not isinstance(area, dict):
            raise RuntimeError(f"description item {index} must be an object")
        if not str(area.get("description", "")).strip():
            raise RuntimeError(f"description item {index} is missing text")


if __name__ == "__main__":
    main()
