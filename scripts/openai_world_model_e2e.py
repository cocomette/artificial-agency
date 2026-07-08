"""Manual E2E check for the OpenAI-backed world model tool.

This script calls the real OpenAI API. Set OPENAI_API_KEY before running it.
It uses a committed ARC source/target frame pair, saves the generated world
prediction image, and records diagnostic distances to the observed next frame.
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
from arcengine import GameAction

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.environment.cheat_context import load_cheat_action_context
from face_of_agi.models.tools.world import OpenAIWorldToolConfig
from face_of_agi.models.tools.world.providers.openai import OpenAIWorldToolAdapter

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
TARGET_PATH = FIXTURE_DIR / "ls20_seed0_action1_target.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "openai_world_model_e2e"
DEFAULT_GAME_DIR = ROOT / "environment_files" / "ls20" / "9607627b"


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    output_dir = _resolve_output_dir(args.output_dir)

    source = Image.open(SOURCE_PATH).convert("RGB")
    target = Image.open(TARGET_PATH).convert("RGB")
    observation = Observation(id="ls20-seed0-step0", step=0, frame=source)
    action = ActionSpec(action_id=GameAction.ACTION1)
    cheat_action_context = (
        load_cheat_action_context(_resolve_game_dir(args.game_dir))
        if args.cheat_action_context
        else None
    )
    context = RoleContext(
        general=(
            "Predict the next ARC frame from the supplied source frame and "
            "action. Use the observed game transition evidence when known."
        ),
        game=_compose_game_context(
            cheat_action_context=cheat_action_context,
        ),
    )
    adapter = OpenAIWorldToolAdapter(_config_from_args(args))

    prompt = adapter._compose_prompt(context, action, observation)
    _print_prompt(prompt)
    _validate_action_in_prompt(prompt, action)

    result = adapter.predict(context=context, action=action, observation=observation)
    if adapter.last_prompt != prompt:
        raise RuntimeError("printed prompt does not match prompt sent to model")

    prediction = result.predicted_observation.convert("RGB")
    _validate_prediction(prediction)

    prediction_path = output_dir / "prediction.png"
    source_copy_path = output_dir / "source.png"
    target_copy_path = output_dir / "target.png"
    metrics_path = output_dir / "metrics.json"

    prediction.save(prediction_path)
    source.save(source_copy_path)
    target.save(target_copy_path)

    prediction_for_metrics, resized = _fit_to_target(prediction, target)
    metrics = {
        "fixture": {
            "game_id": "ls20-9607627b",
            "seed": 0,
            "action": "ACTION1",
            "source": str(SOURCE_PATH.relative_to(ROOT)),
            "target": str(TARGET_PATH.relative_to(ROOT)),
        },
        "output": {
            "prediction": str(prediction_path.relative_to(ROOT)),
            "source_copy": str(source_copy_path.relative_to(ROOT)),
            "target_copy": str(target_copy_path.relative_to(ROOT)),
            "prediction_resized_for_metrics": resized,
        },
        "model": result.metadata,
        "prompt": {
            "cheat_action_context": cheat_action_context,
            "cheat_action_context_enabled": args.cheat_action_context,
            "full_prompt": prompt,
            "action_in_prompt": True,
        },
        "distance": {
            "prediction_to_target": _image_distance(prediction_for_metrics, target),
            "source_to_target_baseline": _image_distance(source, target),
        },
        "tool_result": {
            "id": result.id,
            "tool": result.tool,
            "source_observation_ref": asdict(result.source_observation_ref),
            "action": {
                "action_id": _action_id_text(result.action),
                "data": result.action.data if result.action else None,
            },
            "explanation": result.explanation,
        },
    }
    _validate_metrics(metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _validate_saved_files(prediction_path, source_copy_path, target_copy_path, metrics_path)

    print(f"saved OpenAI world prediction and metrics to {output_dir}")
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
    parser.add_argument(
        "--cheat-action-context",
        action="store_true",
        help="Append action semantics parsed from the local game files.",
    )
    parser.add_argument(
        "--game-dir",
        default=str(DEFAULT_GAME_DIR),
        help="Local ARC game directory used for --cheat-action-context.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> OpenAIWorldToolConfig:
    return OpenAIWorldToolConfig(
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
        metadata={"role": "world", "script": "openai_world_model_e2e"},
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


def _compose_game_context(*, cheat_action_context: str | None = None) -> str:
    """Return fixture-specific world context, optionally with cheat details."""

    context = (
        "This E2E check uses game ls20-9607627b, seed 0, source step 0. "
        "The proposed action is ACTION1."
    )
    if not cheat_action_context:
        return context
    return "\n\n".join([context, cheat_action_context])


def _resolve_game_dir(game_dir: str | Path) -> Path:
    path = Path(game_dir).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _print_prompt(prompt: str) -> None:
    print("--- PROMPT SENT TO OPENAI WORLD MODEL ---")
    print(prompt)


def _validate_action_in_prompt(prompt: str, action: ActionSpec) -> None:
    action_id = _action_id_text(action)
    if _normalized(action_id) not in _normalized(prompt):
        raise RuntimeError(f"prompt does not include action id: {action_id}")

    if action.data is not None:
        action_data = json.dumps(action.data, sort_keys=True)
        if action_data not in prompt:
            raise RuntimeError(f"prompt does not include action data: {action_data}")


def _action_id_text(action: ActionSpec | None) -> str | None:
    if action is None:
        return None
    return str(getattr(action.action_id, "name", action.action_id))


def _normalized(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


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
