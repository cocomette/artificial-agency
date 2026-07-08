"""Manual E2E check for the OpenAI-backed goal model tool.

This script calls the real OpenAI API. Set OPENAI_API_KEY before running it.
It uses the committed ARC world fixture as the source observation and saves the
generated goal-relevant prediction image plus metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from face_of_agi.contracts import Observation, RoleContext
from face_of_agi.models.tools.goal import (
    OpenAIGoalToolAdapter,
    OpenAIGoalToolConfig,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
REFERENCE_PATH = FIXTURE_DIR / "ls20_seed0_action1_target.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "openai_goal_model_e2e"


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    output_dir = _resolve_output_dir(args.output_dir)

    source = Image.open(SOURCE_PATH).convert("RGB")
    reference = Image.open(REFERENCE_PATH).convert("RGB")
    observation = Observation(id="ls20-seed0-step0", step=0, frame=source)
    context = RoleContext(
        general=(
            "Infer a goal-relevant visual observation from the current ARC "
            "frame. Focus on evidence of progress rather than a specific "
            "transition action."
        ),
        game=(
            "This E2E check uses game ls20-9607627b, seed 0, source step 0. "
            "Treat the next observed frame only as a diagnostic reference."
        ),
    )
    adapter = OpenAIGoalToolAdapter(_config_from_args(args))

    prompt = adapter._compose_prompt(context, observation)
    _print_prompt(prompt)
    _validate_no_action_prompt(prompt)

    result = adapter.predict(context=context, observation=observation)
    if adapter.last_prompt != prompt:
        raise RuntimeError("printed prompt does not match prompt sent to model")

    prediction = result.predicted_observation.convert("RGB")
    _validate_prediction(prediction)

    prediction_path = output_dir / "prediction.png"
    source_copy_path = output_dir / "source.png"
    reference_copy_path = output_dir / "reference.png"
    metrics_path = output_dir / "metrics.json"

    prediction.save(prediction_path)
    source.save(source_copy_path)
    reference.save(reference_copy_path)

    prediction_for_metrics, resized = _fit_to_target(prediction, reference)
    metrics = {
        "fixture": {
            "game_id": "ls20-9607627b",
            "seed": 0,
            "source": str(SOURCE_PATH.relative_to(ROOT)),
            "diagnostic_reference": str(REFERENCE_PATH.relative_to(ROOT)),
        },
        "output": {
            "prediction": str(prediction_path.relative_to(ROOT)),
            "source_copy": str(source_copy_path.relative_to(ROOT)),
            "reference_copy": str(reference_copy_path.relative_to(ROOT)),
            "prediction_resized_for_metrics": resized,
        },
        "model": result.metadata,
        "prompt": {
            "full_prompt": prompt,
            "contains_action_section": False,
        },
        "distance": {
            "prediction_to_diagnostic_reference": _image_distance(
                prediction_for_metrics,
                reference,
            ),
            "source_to_diagnostic_reference_baseline": _image_distance(
                source,
                reference,
            ),
        },
        "tool_result": {
            "id": result.id,
            "tool": result.tool,
            "source_observation_ref": asdict(result.source_observation_ref),
            "action": result.action,
            "explanation": result.explanation,
        },
    }
    _validate_metrics(metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _validate_saved_files(
        prediction_path,
        source_copy_path,
        reference_copy_path,
        metrics_path,
    )

    print(f"saved OpenAI goal prediction and metrics to {output_dir}")
    print(json.dumps(metrics["distance"], indent=2, sort_keys=True))


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
    parser.add_argument("--image-model", default="gpt-image-1-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-quality", default="low")
    parser.add_argument("--image-output-format", default="png")
    parser.add_argument("--input-image-detail", default="auto")
    parser.add_argument(
        "--input-image-size",
        default="1024x1024",
        help="Optional WxH resize applied to the input image before API upload.",
    )
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> OpenAIGoalToolConfig:
    return OpenAIGoalToolConfig(
        api_key_env=args.api_key_env,
        model=args.model,
        image_model=args.image_model,
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        image_size=args.image_size,
        image_quality=args.image_quality,
        image_output_format=args.image_output_format,
        input_image_detail=args.input_image_detail,
        input_image_size=args.input_image_size,
        input_image_resample=args.input_image_resample,
        timeout=args.timeout,
        max_retries=args.max_retries,
        metadata={"role": "goal", "script": "openai_goal_model_e2e"},
    )


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _print_prompt(prompt: str) -> None:
    print("--- PROMPT SENT TO OPENAI GOAL MODEL ---")
    print(prompt)


def _validate_no_action_prompt(prompt: str) -> None:
    if "PROPOSED ACTION" in prompt or "action_id:" in prompt:
        raise RuntimeError("goal prompt unexpectedly includes action text")


def _fit_to_target(
    prediction: Image.Image,
    target: Image.Image,
) -> tuple[Image.Image, bool]:
    if prediction.size == target.size:
        return prediction, False
    return prediction.resize(target.size, Image.Resampling.BICUBIC), True


def _image_distance(left: Image.Image, right: Image.Image) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    diff = left_array - right_array
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = 100.0 if mse == 0 else 20 * math.log10(255.0 / rmse)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
    }


def _validate_prediction(prediction: Image.Image) -> None:
    if prediction.mode != "RGB":
        raise RuntimeError(f"prediction must be RGB, got {prediction.mode}")
    if prediction.width <= 0 or prediction.height <= 0:
        raise RuntimeError(f"prediction has invalid size: {prediction.size}")

    pixels = np.asarray(prediction)
    if not np.any(pixels):
        raise RuntimeError("prediction image is blank")


def _validate_metrics(metrics: dict[str, Any]) -> None:
    for distance in metrics["distance"].values():
        for key, value in distance.items():
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite metric {key}: {value}")


def _validate_saved_files(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"expected output files were not saved: {missing}")


if __name__ == "__main__":
    main()
