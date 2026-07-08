"""Measure whether frozen VLM latents move for ARC frame changes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch

from single_vlm_arc.config import load_config
from single_vlm_arc.env import build_session
from single_vlm_arc.history import decision_frame
from single_vlm_arc.model import build_model
from single_vlm_arc.online_update import frame_to_palette_tensor


DEFAULT_PROMPT = "Return a compact visual representation of this ARC-AGI frame."


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.model.device = args.device
    if args.image_size is not None:
        config.model.image_size = _parse_size(args.image_size)
    if args.disable_lora:
        config.model.lora.enabled = False
    config.environment.max_turns = max(int(args.samples), 1)

    output_dir = Path(args.output_dir or config.logging.output_dir) / (
        "latent_sensitivity_" + time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "pairs.jsonl"
    summary_path = output_dir / "summary.json"

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    session = build_session(config.environment, dry_run=bool(config.dry_run))
    model = build_model(config.model, palette_size=config.palette_size)
    model.eval()

    observation = session.reset()
    records: list[dict[str, Any]] = []
    frame_latents: list[dict[str, Any]] = []

    with rows_path.open("w", encoding="utf-8") as handle:
        for turn in range(max(int(args.samples), 1)):
            current_frame = decision_frame(observation)
            action_space = tuple(session.get_action_space())
            if current_frame is None or not action_space:
                break

            current_latents = _frame_latents(
                model,
                current_frame,
                prompt=str(args.prompt),
                include_vision_latents=bool(args.vision_latents),
            )
            frame_latents.append(
                {
                    "turn": turn,
                    "observation_id": getattr(observation, "id", None),
                    "latents": current_latents,
                }
            )

            action = random.choice(action_space)
            next_observation = session.step(action)
            next_frame = decision_frame(next_observation)
            if next_frame is None:
                break
            next_latents = _frame_latents(
                model,
                next_frame,
                prompt=str(args.prompt),
                include_vision_latents=bool(args.vision_latents),
            )
            frame_latents.append(
                {
                    "turn": turn,
                    "observation_id": getattr(next_observation, "id", None),
                    "latents": next_latents,
                }
            )

            pair_records = [
                _pair_record(
                    pair_type="actual_next",
                    turn=turn,
                    action_name=getattr(action, "name", str(action)),
                    left_frame=current_frame,
                    right_frame=next_frame,
                    left_latents=current_latents,
                    right_latents=next_latents,
                    config=config,
                )
            ]
            for variant_name, variant_frame in _synthetic_variants(
                current_frame,
                palette_size=int(config.palette_size),
                frame_size=tuple(config.frame_size),
                rng=random,
            ):
                pair_records.append(
                    _pair_record(
                        pair_type=variant_name,
                        turn=turn,
                        action_name=None,
                        left_frame=current_frame,
                        right_frame=variant_frame,
                        left_latents=current_latents,
                        right_latents=_frame_latents(
                            model,
                            variant_frame,
                            prompt=str(args.prompt),
                            include_vision_latents=bool(args.vision_latents),
                        ),
                        config=config,
                    )
                )

            for record in pair_records:
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")

            observation = next_observation

    records.extend(_random_pair_records(frame_latents, config=config))
    with rows_path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["pair_type"] == "random_pair":
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = _summary(
        records,
        config=config,
        model_id=str(config.model.model_id),
        prompt=str(args.prompt),
        vision_latents=bool(args.vision_latents),
        output_dir=output_dir,
        rows_path=rows_path,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _frame_latents(
    model: Any,
    frame: Any,
    *,
    prompt: str,
    include_vision_latents: bool,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        inputs = model._prepare_inputs(prompt, [frame])
        hidden_sequence = model._forward_hidden_sequence(inputs).detach().float()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_sequence.device)
        pooled = model._pool_hidden_sequence(hidden_sequence, attention_mask)
        masked_mean = _masked_mean(hidden_sequence, attention_mask)
        latents = {
            "pooled": _tensor_to_array(pooled.squeeze(0)),
            "masked_mean": _tensor_to_array(masked_mean.squeeze(0)),
        }
        if include_vision_latents:
            latents.update(_vision_latents(model, inputs))
        return latents


def _masked_mean(
    hidden_sequence: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None or attention_mask.shape != hidden_sequence.shape[:2]:
        return hidden_sequence.mean(dim=1)
    weights = attention_mask.to(
        device=hidden_sequence.device,
        dtype=hidden_sequence.dtype,
    )
    total = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden_sequence * weights.unsqueeze(-1)).sum(dim=1) / total


def _tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _vision_latents(model: Any, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return direct vision-tower token latents when the model exposes them."""

    base_model = getattr(model, "base_model", None)
    inner_model = getattr(base_model, "model", None)
    vision_tower = getattr(inner_model, "vision_tower", None)
    pixel_values = inputs.get("pixel_values")
    position_ids = _image_position_ids(inputs)
    if vision_tower is None or pixel_values is None or position_ids is None:
        return {}

    latents: dict[str, np.ndarray] = {}
    padding_positions = (position_ids == -1).all(dim=-1)

    if hasattr(vision_tower, "patch_embedder") and hasattr(vision_tower, "encoder"):
        patch_embeds = vision_tower.patch_embedder(
            pixel_values,
            position_ids,
            padding_positions,
        )
        encoder_output = vision_tower.encoder(
            inputs_embeds=patch_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=position_ids,
            return_dict=True,
        )
        valid_tokens = encoder_output.last_hidden_state[~padding_positions]
        latents["vision_patch_mean"] = _tensor_to_array(valid_tokens.mean(dim=0))
        latents["vision_patch_flat"] = _tensor_to_array(valid_tokens.reshape(-1))

    vision_output = vision_tower(
        pixel_values=pixel_values,
        pixel_position_ids=position_ids,
        return_dict=True,
    )
    vision_tokens = vision_output.last_hidden_state
    vision_tokens = vision_tokens.reshape(-1, vision_tokens.shape[-1])
    latents["vision_soft_token_mean"] = _tensor_to_array(vision_tokens.mean(dim=0))
    latents["vision_soft_token_flat"] = _tensor_to_array(vision_tokens.reshape(-1))

    if hasattr(inner_model, "get_image_features"):
        image_features = inner_model.get_image_features(
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            return_dict=True,
        )
        projected_tokens = getattr(image_features, "pooler_output", None)
        if projected_tokens is not None:
            projected_tokens = projected_tokens.reshape(-1, projected_tokens.shape[-1])
            latents["vision_projected_token_mean"] = _tensor_to_array(
                projected_tokens.mean(dim=0)
            )
            latents["vision_projected_token_flat"] = _tensor_to_array(
                projected_tokens.reshape(-1)
            )
    return latents


def _image_position_ids(inputs: dict[str, Any]) -> torch.Tensor | None:
    if "image_position_ids" in inputs:
        return inputs["image_position_ids"]
    if "pixel_position_ids" in inputs:
        return inputs["pixel_position_ids"]
    return None


def _pair_record(
    *,
    pair_type: str,
    turn: int | None,
    action_name: str | None,
    left_frame: Any,
    right_frame: Any,
    left_latents: dict[str, np.ndarray],
    right_latents: dict[str, np.ndarray],
    config: Any,
) -> dict[str, Any]:
    pixel_metrics = _pixel_metrics(
        left_frame,
        right_frame,
        palette_size=int(config.palette_size),
        frame_size=tuple(config.frame_size),
    )
    distances = {
        key: _latent_distances(left_latents[key], right_latents[key])
        for key in left_latents
        if key in right_latents
    }
    return {
        "pair_type": pair_type,
        "turn": turn,
        "action_name": action_name,
        "pixel": pixel_metrics,
        "latent": distances,
    }


def _synthetic_variants(
    frame: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int],
    rng: random.Random | Any,
) -> list[tuple[str, np.ndarray]]:
    base = frame_to_palette_tensor(
        frame,
        palette_size=palette_size,
        frame_size=frame_size,
    ).cpu().numpy().astype(np.uint8)
    height, width = base.shape
    variants: list[tuple[str, np.ndarray]] = []
    for pair_type, side in (
        ("synthetic_1px", 1),
        ("synthetic_2x2", 2),
        ("synthetic_4x4", 4),
    ):
        edited = base.copy()
        x = rng.randrange(max(width - side + 1, 1))
        y = rng.randrange(max(height - side + 1, 1))
        patch = edited[y : y + side, x : x + side]
        patch[:] = (patch.astype(np.int64) + 1) % max(palette_size, 1)
        variants.append((pair_type, edited))
    return variants


def _random_pair_records(
    frame_latents: Sequence[dict[str, Any]],
    *,
    config: Any,
) -> list[dict[str, Any]]:
    if len(frame_latents) < 2:
        return []
    records: list[dict[str, Any]] = []
    for index, left in enumerate(frame_latents):
        offset = max(len(frame_latents) // 2, 1)
        right = frame_latents[(index + offset) % len(frame_latents)]
        records.append(
            {
                "pair_type": "random_pair",
                "turn": left["turn"],
                "action_name": None,
                "pixel": {
                    "changed_pixels": None,
                    "changed_fraction": None,
                },
                "latent": {
                    key: _latent_distances(left["latents"][key], right["latents"][key])
                    for key in left["latents"]
                    if key in right["latents"]
                },
            }
        )
    return records


def _pixel_metrics(
    left_frame: Any,
    right_frame: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int],
) -> dict[str, Any]:
    left = frame_to_palette_tensor(
        left_frame,
        palette_size=palette_size,
        frame_size=frame_size,
    )
    right = frame_to_palette_tensor(
        right_frame,
        palette_size=palette_size,
        frame_size=frame_size,
    )
    changed_pixels = int((left != right).sum().item())
    total_pixels = max(int(left.numel()), 1)
    return {
        "changed_pixels": changed_pixels,
        "changed_fraction": float(changed_pixels / total_pixels),
    }


def _latent_distances(left: Any, right: Any) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.size != right_array.size:
        common_size = min(left_array.size, right_array.size)
        left_array = left_array[:common_size]
        right_array = right_array[:common_size]
    delta = left_array - right_array
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    denominator = max(left_norm * right_norm, 1e-12)
    cosine_similarity = float(np.dot(left_array, right_array) / denominator)
    return {
        "cosine_distance": float(1.0 - cosine_similarity),
        "l2": float(np.linalg.norm(delta)),
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def _summary(
    records: Sequence[dict[str, Any]],
    *,
    config: Any,
    model_id: str,
    prompt: str,
    vision_latents: bool,
    output_dir: Path,
    rows_path: Path,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        pair_type = str(record["pair_type"])
        for latent_name, distances in record["latent"].items():
            for metric_name, value in distances.items():
                grouped[f"{pair_type}.{latent_name}"][metric_name].append(float(value))

    metrics = {
        group_name: {
            metric_name: _stats(values)
            for metric_name, values in metric_values.items()
        }
        for group_name, metric_values in grouped.items()
    }
    separability = _separability(metrics)
    return {
        "model_id": model_id,
        "game_id": config.environment.game_id,
        "samples": len([row for row in records if row["pair_type"] == "actual_next"]),
        "prompt": prompt,
        "vision_latents": vision_latents,
        "output_dir": str(output_dir),
        "rows_path": str(rows_path),
        "metrics": metrics,
        "separability": separability,
        "interpretation": _interpretation(separability),
    }


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _separability(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for latent_name in ("pooled", "masked_mean"):
        result.update(_separability_for_latent(metrics, latent_name))
    for latent_name in (
        "vision_patch_mean",
        "vision_patch_flat",
        "vision_soft_token_mean",
        "vision_soft_token_flat",
        "vision_projected_token_mean",
        "vision_projected_token_flat",
    ):
        result.update(_separability_for_latent(metrics, latent_name))
    return result


def _separability_for_latent(
    metrics: dict[str, Any],
    latent_name: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    random_key = f"random_pair.{latent_name}"
    random_mean = (
        metrics.get(random_key, {})
        .get("cosine_distance", {})
        .get("mean")
    )
    if random_mean is None or random_mean <= 0:
        return result
    for pair_type in (
        "actual_next",
        "synthetic_1px",
        "synthetic_2x2",
        "synthetic_4x4",
    ):
        key = f"{pair_type}.{latent_name}"
        mean = (
            metrics.get(key, {})
            .get("cosine_distance", {})
            .get("mean")
        )
        if mean is None:
            continue
        result[f"{key}_vs_random_cosine_ratio"] = float(mean / random_mean)
    return result


def _interpretation(separability: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for key, value in separability.items():
        if "synthetic_1px" in key and float(value) < 0.05:
            notes.append(
                f"{key} is {value:.4f}; one-pixel changes are nearly invisible "
                "relative to random-frame latent distances."
            )
        if "synthetic_4x4" in key and float(value) < 0.2:
            notes.append(
                f"{key} is {value:.4f}; even small patch changes are weakly "
                "represented."
            )
    if not notes:
        notes.append("Latent distances moved measurably for the tested perturbations.")
    return notes


def _parse_size(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise ValueError("size must be formatted like 128x128")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure frozen VLM latent sensitivity to ARC frame changes.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", default=None)
    parser.add_argument(
        "--no-vision-latents",
        dest="vision_latents",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--with-lora",
        dest="disable_lora",
        action="store_false",
        default=True,
    )
    return parser


if __name__ == "__main__":
    main()
