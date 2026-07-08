"""Run artifact logging for the single-VLM experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from single_vlm_arc.config import ExperimentConfig, config_to_dict


class ExperimentLogger:
    """Write config, turn metrics, summaries, and checkpoints."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.output_dir = Path(config.logging.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.turns_path = self.output_dir / "turns.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.config_path = self.output_dir / "config.resolved.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_to_dict(config), handle, sort_keys=True)

    def append_turn(self, payload: dict[str, Any]) -> None:
        with self.turns_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")

    def write_summary(self, payload: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def step_checkpoint_path(self, turn: int) -> Path:
        return self.checkpoint_dir / f"adapter_step_{turn:06d}.safetensors"

    def final_adapter_dir(self) -> Path:
        return self.checkpoint_dir / "final_adapter"

    def video_recorder(self, config: ExperimentConfig) -> "ObservationVideoRecorder":
        return ObservationVideoRecorder(
            self.output_dir,
            enabled=bool(config.logging.save_video),
            fps=int(config.logging.video_fps),
            frame_scale=int(config.logging.video_frame_scale),
        )

    def frame_prediction_logger(
        self,
        config: ExperimentConfig,
    ) -> "FramePredictionLogger":
        return FramePredictionLogger(
            self.output_dir,
            enabled=(
                bool(config.logging.save_frame_predictions)
                and not bool(config.dry_run)
            ),
            save_every=int(config.logging.frame_prediction_save_every),
            frame_scale=int(config.logging.frame_prediction_frame_scale),
            palette_size=int(config.palette_size),
            frame_size=tuple(config.frame_size),
        )

    def latent_prediction_logger(
        self,
        config: ExperimentConfig,
    ) -> "LatentPredictionLogger":
        return LatentPredictionLogger(
            self.output_dir,
            enabled=(
                bool(config.logging.save_latent_predictions)
                and str(getattr(config.update, "world_loss_mode", "pixel_ce"))
                in {"latent_grid", "hybrid"}
            ),
            save_every=int(config.logging.latent_prediction_save_every),
            frame_scale=int(config.logging.latent_prediction_frame_scale),
        )


class ObservationVideoRecorder:
    """Write visual observations to one replayable MP4 plus a frame manifest."""

    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool,
        fps: int,
        frame_scale: int,
    ) -> None:
        self.enabled = enabled
        self.video_path = output_dir / "frames.mp4"
        self.manifest_path = output_dir / "frame_manifest.jsonl"
        self.fps = max(int(fps), 1)
        self.frame_scale = max(int(frame_scale), 1)
        self.frame_count = 0
        self._writer: Any | None = None
        self._manifest_handle: Any | None = None
        self._video_size: tuple[int, int] | None = None
        if not self.enabled:
            return

        import imageio.v2 as imageio

        self._writer = imageio.get_writer(
            str(self.video_path),
            fps=self.fps,
            codec="libx264",
            macro_block_size=1,
        )
        self._manifest_handle = self.manifest_path.open("w", encoding="utf-8")

    def append_observation(
        self,
        observation: Any,
        *,
        turn: int | None,
        action_name: str | None,
        phase: str,
    ) -> None:
        if not self.enabled:
            return
        if self._writer is None or self._manifest_handle is None:
            raise RuntimeError("video recorder is enabled but not open")

        frames = _observation_frames(observation)
        for observation_frame_index, frame in enumerate(frames):
            image = _frame_to_video_image(
                frame,
                step=int(getattr(observation, "step", 0) or 0),
                frame_scale=self.frame_scale,
            )
            if self._video_size is None:
                self._video_size = image.size
            elif image.size != self._video_size:
                from PIL import Image

                image = image.resize(self._video_size, Image.Resampling.NEAREST)

            self._writer.append_data(_image_to_uint8_rgb_array(image))
            self._manifest_handle.write(
                json.dumps(
                    {
                        "video_frame_index": self.frame_count,
                        "turn": turn,
                        "action_name": action_name,
                        "phase": phase,
                        "observation_id": getattr(observation, "id", None),
                        "observation_step": getattr(observation, "step", None),
                        "observation_frame_index": observation_frame_index,
                        "observation_frame_count": len(frames),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            self.frame_count += 1

    def close(self) -> None:
        if self._manifest_handle is not None:
            self._manifest_handle.close()
            self._manifest_handle = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class FramePredictionLogger:
    """Write next-frame prediction composites and a JSONL metrics manifest."""

    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool,
        save_every: int,
        frame_scale: int,
        palette_size: int,
        frame_size: tuple[int, int],
    ) -> None:
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.save_every = max(int(save_every), 1)
        self.frame_scale = max(int(frame_scale), 1)
        self.palette_size = max(int(palette_size), 1)
        self.frame_size = frame_size
        self.prediction_dir = output_dir / "frame_predictions"
        self.manifest_path = output_dir / "frame_prediction_manifest.jsonl"
        self.count = 0
        if self.enabled:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def should_log(self, turn: int) -> bool:
        return self.enabled and int(turn) % self.save_every == 0

    def append_prediction(
        self,
        *,
        turn: int,
        action_name: str,
        selected_action_index: int,
        observation_id: str | None,
        next_observation_id: str | None,
        current_frame: Any,
        target_frame: Any,
        pre_update_prediction: Any,
        post_update_prediction: Any,
        pre_update_loss: float | None,
        post_update_loss: float | None,
    ) -> dict[str, Any] | None:
        if not self.should_log(turn):
            return None

        current = _palette_array(
            current_frame,
            palette_size=self.palette_size,
            frame_size=self.frame_size,
        )
        target = _palette_array(
            target_frame,
            palette_size=self.palette_size,
            frame_size=self.frame_size,
        )
        pre_update = _palette_array(
            pre_update_prediction,
            palette_size=self.palette_size,
            frame_size=self.frame_size,
        )
        post_update = _palette_array(
            post_update_prediction,
            palette_size=self.palette_size,
            frame_size=self.frame_size,
        )

        stem = f"turn_{int(turn):06d}_{_safe_name(action_name)}"
        image_path = self.prediction_dir / f"{stem}.png"
        raw_path = self.prediction_dir / f"{stem}.npz"
        composite = _prediction_composite(
            current=current,
            target=target,
            pre_update=pre_update,
            post_update=post_update,
            frame_scale=self.frame_scale,
            palette_size=self.palette_size,
        )
        composite.save(image_path)
        _save_prediction_arrays(
            raw_path,
            current=current,
            target=target,
            pre_update=pre_update,
            post_update=post_update,
        )

        copy_metrics = _prediction_metrics(current, target, current)
        pre_metrics = _prediction_metrics(current, target, pre_update)
        post_metrics = _prediction_metrics(current, target, post_update)
        row = {
            "turn": int(turn),
            "action_name": action_name,
            "selected_action_index": int(selected_action_index),
            "observation_id": observation_id,
            "next_observation_id": next_observation_id,
            "image_path": _relative_path(image_path, self.output_dir),
            "raw_arrays_path": _relative_path(raw_path, self.output_dir),
            "pre_update_loss": pre_update_loss,
            "post_update_loss": post_update_loss,
            "copy_baseline": copy_metrics,
            "pre_update": pre_metrics,
            "post_update": post_metrics,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        self.count += 1
        return row


class LatentPredictionLogger:
    """Write latent-grid prediction heatmaps and a JSONL metrics manifest."""

    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool,
        save_every: int,
        frame_scale: int,
    ) -> None:
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.save_every = max(int(save_every), 1)
        self.frame_scale = max(int(frame_scale), 1)
        self.prediction_dir = output_dir / "latent_predictions"
        self.manifest_path = output_dir / "latent_prediction_manifest.jsonl"
        self.count = 0
        if self.enabled:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def should_log(self, turn: int) -> bool:
        return self.enabled and int(turn) % self.save_every == 0

    def append_prediction(
        self,
        *,
        turn: int,
        action_name: str,
        selected_action_index: int,
        observation_id: str | None,
        next_observation_id: str | None,
        current_latent_grid: Any,
        target_latent_grid: Any,
        changed_patch_mask: Any,
        pre_update_prediction: Any,
        post_update_prediction: Any,
        pre_update_loss: float | None,
        post_update_loss: float | None,
    ) -> dict[str, Any] | None:
        if not self.should_log(turn):
            return None

        maps = _latent_prediction_maps(
            current_latent_grid=current_latent_grid,
            target_latent_grid=target_latent_grid,
            changed_patch_mask=changed_patch_mask,
            pre_update_prediction=pre_update_prediction,
            post_update_prediction=post_update_prediction,
        )
        stem = f"turn_{int(turn):06d}_{_safe_name(action_name)}"
        image_path = self.prediction_dir / f"{stem}.png"
        raw_path = self.prediction_dir / f"{stem}.npz"
        composite = _latent_prediction_composite(
            maps,
            frame_scale=self.frame_scale,
        )
        composite.save(image_path)
        _save_latent_prediction_arrays(raw_path, maps)

        row = {
            "turn": int(turn),
            "action_name": action_name,
            "selected_action_index": int(selected_action_index),
            "observation_id": observation_id,
            "next_observation_id": next_observation_id,
            "image_path": _relative_path(image_path, self.output_dir),
            "raw_arrays_path": _relative_path(raw_path, self.output_dir),
            "grid_shape": list(maps["changed_patch_mask"].shape),
            "changed_patch_count": int(maps["changed_patch_mask"].sum()),
            "changed_patch_fraction": _fraction(
                int(maps["changed_patch_mask"].sum()),
                int(maps["changed_patch_mask"].size),
            ),
            "pre_update_loss": pre_update_loss,
            "post_update_loss": post_update_loss,
            "pre_error_norm_mean": float(maps["pre_error_norm"].mean()),
            "post_error_norm_mean": float(maps["post_error_norm"].mean()),
            "improvement_norm_mean": float(maps["improvement"].mean()),
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        self.count += 1
        return row


def _observation_frames(observation: Any) -> list[Any]:
    frames = list(getattr(observation, "frames", ()) or ())
    if frames:
        return [frame for frame in frames if frame is not None]
    frame = getattr(observation, "frame", None)
    if frame is None:
        return []
    return [frame]


def _latent_prediction_maps(
    *,
    current_latent_grid: Any,
    target_latent_grid: Any,
    changed_patch_mask: Any,
    pre_update_prediction: Any,
    post_update_prediction: Any,
) -> dict[str, Any]:
    import numpy as np

    current = _to_numpy(current_latent_grid).astype("float32", copy=False)
    target = _to_numpy(target_latent_grid).astype("float32", copy=False)
    pre = _to_numpy(pre_update_prediction).astype("float32", copy=False)
    post = _to_numpy(post_update_prediction).astype("float32", copy=False)
    mask = _to_numpy(changed_patch_mask).astype(bool, copy=False)
    target_delta = target - current
    pre_error = np.linalg.norm(pre - target_delta, axis=-1)
    post_error = np.linalg.norm(post - target_delta, axis=-1)
    return {
        "changed_patch_mask": mask,
        "target_delta_norm": np.linalg.norm(target_delta, axis=-1),
        "pre_error_norm": pre_error,
        "post_error_norm": post_error,
        "improvement": pre_error - post_error,
    }


def _latent_prediction_composite(
    maps: dict[str, Any],
    *,
    frame_scale: int,
) -> Any:
    panels = [
        (
            "changed",
            _mask_image(
                maps["changed_patch_mask"],
                frame_scale=frame_scale,
                on_color=(255, 255, 255),
                off_color=(0, 0, 0),
            ),
        ),
        (
            "target_delta",
            _heatmap_image(maps["target_delta_norm"], frame_scale=frame_scale),
        ),
        (
            "pre_error",
            _heatmap_image(maps["pre_error_norm"], frame_scale=frame_scale),
        ),
        (
            "post_error",
            _heatmap_image(maps["post_error_norm"], frame_scale=frame_scale),
        ),
        (
            "improvement",
            _signed_heatmap_image(maps["improvement"], frame_scale=frame_scale),
        ),
    ]
    return _tile_labeled_images(panels, columns=5)


def _save_latent_prediction_arrays(path: Path, maps: dict[str, Any]) -> None:
    import numpy as np

    np.savez_compressed(path, **maps)


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    import numpy as np

    return np.asarray(value)


def _frame_to_video_image(frame: Any, *, step: int, frame_scale: int) -> Any:
    from face_of_agi.frames import frame_to_pil_image

    return frame_to_pil_image(
        frame,
        step=step,
        frame_scale=frame_scale,
        label="video_frame",
    ).convert("RGB")


def _image_to_uint8_rgb_array(image: Any) -> Any:
    import numpy as np

    array = np.asarray(image.convert("RGB"))
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def _palette_array(
    value: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int],
) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        from single_vlm_arc.online_update import frame_to_palette_tensor

        array = frame_to_palette_tensor(
            value,
            palette_size=palette_size,
            frame_size=frame_size,
        ).cpu().numpy()
    if array.ndim != 2:
        raise ValueError(f"prediction frame must be 2D, got shape {array.shape}")
    return array.astype(np.uint8, copy=False)


def _prediction_composite(
    *,
    current: Any,
    target: Any,
    pre_update: Any,
    post_update: Any,
    frame_scale: int,
    palette_size: int,
) -> Any:
    panels = [
        (
            "current",
            _render_palette_image(
                current,
                frame_scale=frame_scale,
                palette_size=palette_size,
            ),
        ),
        (
            "target",
            _render_palette_image(
                target,
                frame_scale=frame_scale,
                palette_size=palette_size,
            ),
        ),
        (
            "pre_update",
            _render_palette_image(
                pre_update,
                frame_scale=frame_scale,
                palette_size=palette_size,
            ),
        ),
        (
            "post_update",
            _render_palette_image(
                post_update,
                frame_scale=frame_scale,
                palette_size=palette_size,
            ),
        ),
        (
            "target_delta",
            _mask_image(
                current != target,
                frame_scale=frame_scale,
                on_color=(255, 255, 255),
                off_color=(0, 0, 0),
            ),
        ),
        (
            "pre_errors",
            _mask_image(
                pre_update != target,
                frame_scale=frame_scale,
                on_color=(255, 0, 0),
                off_color=(24, 24, 24),
            ),
        ),
        (
            "post_errors",
            _mask_image(
                post_update != target,
                frame_scale=frame_scale,
                on_color=(255, 0, 0),
                off_color=(24, 24, 24),
            ),
        ),
        (
            "post_delta",
            _mask_image(
                post_update != current,
                frame_scale=frame_scale,
                on_color=(255, 255, 255),
                off_color=(0, 0, 0),
            ),
        ),
    ]
    return _tile_labeled_images(panels, columns=4)


def _render_palette_image(
    array: Any,
    *,
    frame_scale: int,
    palette_size: int,
) -> Any:
    from PIL import Image
    import numpy as np

    try:
        from arc_agi.rendering import frame_to_rgb_array

        return Image.fromarray(
            frame_to_rgb_array(
                steps=0,
                frame=np.asarray(array, dtype=np.uint8),
                scale=frame_scale,
            )
        ).convert("RGB")
    except Exception:
        colors = _fallback_palette(max(palette_size, int(np.max(array, initial=0)) + 1))
        clipped = np.asarray(array, dtype=np.int64).clip(0, len(colors) - 1)
        image = Image.fromarray(colors[clipped].astype(np.uint8), "RGB")
        if frame_scale == 1:
            return image
        return image.resize(
            (image.width * frame_scale, image.height * frame_scale),
            Image.Resampling.NEAREST,
        )


def _mask_image(
    mask: Any,
    *,
    frame_scale: int,
    on_color: tuple[int, int, int],
    off_color: tuple[int, int, int],
) -> Any:
    from PIL import Image
    import numpy as np

    mask_array = np.asarray(mask, dtype=bool)
    rgb = np.empty((*mask_array.shape, 3), dtype=np.uint8)
    rgb[mask_array] = on_color
    rgb[~mask_array] = off_color
    image = Image.fromarray(rgb, "RGB")
    if frame_scale == 1:
        return image
    return image.resize(
        (image.width * frame_scale, image.height * frame_scale),
        Image.Resampling.NEAREST,
    )


def _heatmap_image(values: Any, *, frame_scale: int) -> Any:
    from PIL import Image
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if finite.any():
        max_value = float(np.max(array[finite]))
    else:
        max_value = 0.0
    normalized = np.zeros_like(array, dtype=np.float32)
    if max_value > 1e-12:
        normalized = np.clip(array / max_value, 0.0, 1.0)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255.0 * normalized).astype(np.uint8)
    rgb[..., 1] = (180.0 * np.sqrt(normalized)).astype(np.uint8)
    rgb[..., 2] = (40.0 * (1.0 - normalized)).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    if frame_scale == 1:
        return image
    return image.resize(
        (image.width * frame_scale, image.height * frame_scale),
        Image.Resampling.NEAREST,
    )


def _signed_heatmap_image(values: Any, *, frame_scale: int) -> Any:
    from PIL import Image
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    magnitude = float(np.max(np.abs(array))) if array.size else 0.0
    normalized = np.zeros_like(array, dtype=np.float32)
    if magnitude > 1e-12:
        normalized = np.clip(array / magnitude, -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    base = np.full((*array.shape, 3), 32, dtype=np.uint8)
    base[..., 1] = np.maximum(base[..., 1], (255.0 * positive).astype(np.uint8))
    base[..., 0] = np.maximum(base[..., 0], (255.0 * negative).astype(np.uint8))
    image = Image.fromarray(base, "RGB")
    if frame_scale == 1:
        return image
    return image.resize(
        (image.width * frame_scale, image.height * frame_scale),
        Image.Resampling.NEAREST,
    )


def _tile_labeled_images(
    panels: list[tuple[str, Any]],
    *,
    columns: int,
) -> Any:
    from PIL import Image

    labeled = [_with_label(image, label) for label, image in panels]
    tile_width = max(image.width for image in labeled)
    tile_height = max(image.height for image in labeled)
    rows = (len(labeled) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        (255, 255, 255),
    )
    for index, image in enumerate(labeled):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(image, (x, y))
    return canvas


def _with_label(image: Any, label: str) -> Any:
    from PIL import Image, ImageDraw

    label_height = 18
    labeled = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, image.width, label_height), fill=(240, 240, 240))
    draw.text((4, 3), label, fill=(0, 0, 0))
    labeled.paste(image, (0, label_height))
    return labeled


def _fallback_palette(size: int) -> Any:
    import numpy as np

    base = np.asarray(
        [
            (0, 0, 0),
            (0, 116, 217),
            (255, 65, 54),
            (46, 204, 64),
            (255, 220, 0),
            (170, 170, 170),
            (240, 18, 190),
            (255, 133, 27),
            (127, 219, 255),
            (135, 12, 37),
            (255, 255, 255),
            (148, 0, 211),
            (0, 128, 128),
            (128, 64, 0),
            (255, 192, 203),
            (64, 64, 64),
        ],
        dtype=np.uint8,
    )
    if size <= len(base):
        return base[:size]
    repeats = (size + len(base) - 1) // len(base)
    return np.tile(base, (repeats, 1))[:size]


def _prediction_metrics(current: Any, target: Any, prediction: Any) -> dict[str, Any]:
    import numpy as np

    current = np.asarray(current)
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    correct = prediction == target
    changed = current != target
    unchanged = ~changed
    predicted_changed = prediction != current
    error = ~correct
    total_pixels = int(target.size)
    changed_pixels = int(changed.sum())
    unchanged_pixels = int(unchanged.sum())
    predicted_changed_pixels = int(predicted_changed.sum())
    return {
        "accuracy": _fraction(int(correct.sum()), total_pixels),
        "error_pixels": int(error.sum()),
        "target_changed_pixels": changed_pixels,
        "target_changed_fraction": _fraction(changed_pixels, total_pixels),
        "predicted_changed_pixels": predicted_changed_pixels,
        "predicted_changed_fraction": _fraction(predicted_changed_pixels, total_pixels),
        "changed_accuracy": _fraction(
            int((correct & changed).sum()),
            changed_pixels,
        ),
        "unchanged_accuracy": _fraction(
            int((correct & unchanged).sum()),
            unchanged_pixels,
        ),
        "false_changed_pixels": int((predicted_changed & unchanged).sum()),
        "missed_changed_pixels": int((changed & (prediction == current)).sum()),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _save_prediction_arrays(
    path: Path,
    *,
    current: Any,
    target: Any,
    pre_update: Any,
    post_update: Any,
) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        current=current,
        target=target,
        pre_update=pre_update,
        post_update=post_update,
    )


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _safe_name(value: str) -> str:
    safe = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")
    return safe or "action"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
