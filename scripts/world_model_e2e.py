"""Manual E2E check for the Diffusers-backed world model tool.

This script loads a committed ARC source/target frame pair, calls the real
world model backend, and records diagnostic image distances to the observed
next frame. The distances are not pass/fail thresholds yet; they are evidence
for prompt/model iteration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from arcengine import GameAction

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.tools.world import WorldToolAdapter, WorldToolConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
TARGET_PATH = FIXTURE_DIR / "ls20_seed0_action1_target.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "world_model_e2e"


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(SOURCE_PATH).convert("RGB")
    target = Image.open(TARGET_PATH).convert("RGB")
    observation = Observation(id="ls20-seed0-step0", step=0, frame=source)
    action = ActionSpec(action_id=GameAction.ACTION1)
    context = RoleContext(
        general=(
            "Predict the next ARC frame from the supplied source frame and "
            "action. Use the observed game transition evidence when known."
        ),
        game=(
            "This E2E check uses game ls20-9607627b, seed 0, source step 0. "
            "The proposed action is ACTION1."
        ),
    )
    adapter = WorldToolAdapter(
        WorldToolConfig(
            model=args.model,
            pipeline_type=args.pipeline_type,
            quantized_model=args.quantized_model,
            quantized_subdir=args.quantized_subdir,
            quantize_text_encoder=not args.no_quantize_text_encoder,
            device=args.device,
            torch_dtype=args.torch_dtype,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            max_sequence_length=args.max_sequence_length,
            max_area=args.max_area,
        )
    )
    prompt = adapter._compose_prompt(context, action, observation)
    _print_prompt(prompt)
    _validate_action_in_prompt(prompt, action)

    result = adapter.predict(
        context=context,
        action=action,
        observation=observation,
    )
    if adapter.last_prompt != prompt:
        raise RuntimeError("printed prompt does not match prompt sent to model")
    prompt_check = _inspect_prompt_token_window(adapter, prompt, action)
    print("--- PROMPT CHECK ---")
    print(json.dumps(prompt_check, indent=2, sort_keys=True))

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
        "prompt": prompt_check,
        "distance": {
            "prediction_to_target": _image_distance(prediction_for_metrics, target),
            "source_to_target_baseline": _image_distance(source, target),
        },
        "tool_result": {
            "id": result.id,
            "tool": result.tool,
            "source_observation_ref": asdict(result.source_observation_ref),
            "action": {
                "action_id": result.action.action_id.name,
                "data": result.action.data,
            },
            "explanation": result.explanation,
        },
    }
    _validate_metrics(metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    _validate_saved_files(
        prediction_path,
        source_copy_path,
        target_copy_path,
        metrics_path,
    )
    print(f"saved prediction and metrics to {output_dir}")
    print(json.dumps(metrics["distance"], indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit")
    parser.add_argument(
        "--pipeline-type",
        choices=("qwen_image_edit", "instruct_pix2pix", "flux_kontext_qint8"),
        default="qwen_image_edit",
    )
    parser.add_argument("--quantized-model", default=None)
    parser.add_argument("--quantized-subdir", default="flux-1-kontext-dev")
    parser.add_argument(
        "--no-quantize-text-encoder",
        action="store_true",
        help="Keep the base T5 text encoder instead of replacing it with qint8.",
    )
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-area", type=int, default=1_048_576)
    return parser.parse_args()


def _print_prompt(prompt: str) -> None:
    print("--- PROMPT SENT TO MODEL ---")
    print(prompt)


def _validate_action_in_prompt(prompt: str, action: ActionSpec) -> None:
    action_id = _action_id_text(action)
    if _normalized(action_id) not in _normalized(prompt):
        raise RuntimeError(f"prompt does not include action id: {action_id}")

    if action.data is not None:
        action_data = json.dumps(action.data, sort_keys=True)
        if action_data not in prompt:
            raise RuntimeError(f"prompt does not include action data: {action_data}")


def _inspect_prompt_token_window(
    adapter: WorldToolAdapter,
    prompt: str,
    action: ActionSpec,
) -> dict[str, Any]:
    tokenizer_checks = [
        _inspect_tokenizer_prompt_window(name, tokenizer, prompt, action)
        for name, tokenizer in (
            ("tokenizer", getattr(adapter._pipeline, "tokenizer", None)),
            ("tokenizer_2", getattr(adapter._pipeline, "tokenizer_2", None)),
        )
        if tokenizer is not None
    ]

    if not tokenizer_checks:
        return {
            "full_prompt": prompt,
            "action_in_prompt": True,
            "action_in_retained_prompt": None,
            "tokenizers": [],
        }

    return {
        "full_prompt": prompt,
        "action_in_prompt": True,
        "action_in_retained_prompt": all(
            check["action_in_retained_prompt"] for check in tokenizer_checks
        ),
        "tokenizers": tokenizer_checks,
    }


def _inspect_tokenizer_prompt_window(
    name: str,
    tokenizer: Any,
    prompt: str,
    action: ActionSpec,
) -> dict[str, Any]:
    tokenized = tokenizer(prompt, truncation=False, add_special_tokens=True)
    input_ids = tokenized.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    model_max_length = int(getattr(tokenizer, "model_max_length", len(input_ids)))
    kept_ids = input_ids[:model_max_length]
    dropped_ids = input_ids[model_max_length:]
    kept_prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)
    dropped_prompt = tokenizer.decode(dropped_ids, skip_special_tokens=True)
    action_id = _action_id_text(action)
    action_in_kept_prompt = _normalized(action_id) in _normalized(kept_prompt)

    if not action_in_kept_prompt:
        raise RuntimeError(
            f"prompt action id {action_id} is not retained in {name} window"
        )

    return {
        "name": name,
        "tokenizer": tokenizer.__class__.__name__,
        "model_max_length": model_max_length,
        "total_tokens": len(input_ids),
        "kept_tokens": len(kept_ids),
        "dropped_tokens": len(dropped_ids),
        "retained_prompt": kept_prompt,
        "dropped_prompt": dropped_prompt,
        "action_in_retained_prompt": action_in_kept_prompt,
    }


def _action_id_text(action: ActionSpec) -> str:
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
