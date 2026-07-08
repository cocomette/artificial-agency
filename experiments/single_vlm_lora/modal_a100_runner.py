"""Modal A100 40GB runner for the isolated single-VLM LoRA experiment.

Run from the repo root:

    uv run --with modal modal run experiments/single_vlm_lora/modal_a100_runner.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import threading
import time
from typing import Any, TextIO

try:
    import modal
except ImportError:  # pragma: no cover - local tests need not install Modal.
    modal = None  # type: ignore[assignment]


DEFAULT_GPU_TYPE = "A100-40GB"
GPU_TYPE = os.environ.get("SINGLE_VLM_MODAL_GPU", DEFAULT_GPU_TYPE)
APP_NAME = os.environ.get("SINGLE_VLM_MODAL_APP_NAME", "single-vlm-lora-a100-40gb")
MODEL_VOLUME_NAME = "face-of-agi-local-models"
RUN_VOLUME_NAME = "face-of-agi-runs"
MODEL_VOLUME_PATH = PurePosixPath("/vol/models")
RUN_VOLUME_PATH = PurePosixPath("/vol/runs")
REMOTE_REPO_PATH = PurePosixPath("/root/repo")
REMOTE_EXPERIMENT_PATH = REMOTE_REPO_PATH / "experiments" / "single_vlm_lora"
REMOTE_CONFIG_DIR = RUN_VOLUME_PATH / "single_vlm_lora" / "configs"
DEFAULT_CONFIG = "experiments/single_vlm_lora/configs/qwen3_vl_4b_a100_40gb.yaml"
LIVE_RUN_COMMIT_SECONDS = 60
TIMEOUT_SECONDS = 60 * 60 * 8


@dataclass(frozen=True)
class StreamedProcessResult:
    """Completed process data captured while output was streamed live."""

    returncode: int
    stdout: str
    stderr: str


def _modal_env() -> dict[str, str]:
    """Return env vars that place all model/cache state on the model Volume."""

    hf_home = MODEL_VOLUME_PATH / "huggingface"
    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "TORCH_HOME": str(MODEL_VOLUME_PATH / "torch"),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        ),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": ":".join(
            (
                str(REMOTE_REPO_PATH / "src"),
                str(REMOTE_EXPERIMENT_PATH),
            )
        ),
    }


def run_streamed_subprocess(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> StreamedProcessResult:
    """Run a subprocess, streaming stdout/stderr while preserving text output."""

    stdout_stream = stdout or sys.stdout
    stderr_stream = stderr or sys.stderr
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    process = subprocess.Popen(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
        cwd=str(cwd) if cwd is not None else None,
        bufsize=1,
    )
    stdout_thread = _forward_stream(
        process.stdout,
        stdout_stream,
        stdout_parts,
    )
    stderr_thread = _forward_stream(
        process.stderr,
        stderr_stream,
        stderr_parts,
    )
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return StreamedProcessResult(
        returncode=returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
    )


def _forward_stream(
    pipe: Any,
    target: TextIO,
    parts: list[str],
) -> threading.Thread:
    thread = threading.Thread(
        target=_forward_stream_lines,
        args=(pipe, target, parts),
        daemon=True,
    )
    thread.start()
    return thread


def _forward_stream_lines(pipe: Any, target: TextIO, parts: list[str]) -> None:
    if pipe is None:
        return
    try:
        for line in iter(pipe.readline, ""):
            parts.append(line)
            target.write(line)
            target.flush()
    finally:
        pipe.close()


def remote_output_dir(
    *,
    config_name: str,
    output_dir: str | None = None,
    run_name: str | None = None,
    timestamp: str | None = None,
) -> PurePosixPath:
    """Return the remote output directory for one Modal experiment run."""

    if output_dir:
        requested = PurePosixPath(output_dir)
        if requested.is_absolute():
            return requested
        return RUN_VOLUME_PATH / requested

    run_id = run_name or Path(config_name).stem
    return (
        RUN_VOLUME_PATH
        / "single_vlm_lora"
        / f"{run_id}-{timestamp or _timestamp()}"
    )


def build_runner_command(
    *,
    config_path: str | PurePosixPath,
    output_dir: str | PurePosixPath,
    max_turns: int | None = None,
    game_id: str | None = None,
    game_index: int | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    save_video: bool = False,
    video_fps: int | None = None,
    video_frame_scale: int | None = None,
    world_loss_mode: str | None = None,
    latent_loss_weight: float | None = None,
    latent_changed_patch_weight: float | None = None,
    latent_huber_beta: float | None = None,
    latent_cosine_loss_weight: float | None = None,
    latent_cosine_min_delta_norm: float | None = None,
    latent_learning_progress_normalization: bool | None = None,
    latent_learning_progress_normalization_floor: float | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build the remote subprocess command for the isolated runner."""

    command = [
        python_executable or sys.executable,
        str(REMOTE_EXPERIMENT_PATH / "run_single_vlm_lora.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if max_turns is not None:
        command.extend(("--max-turns", str(max_turns)))
    if game_id is not None:
        command.extend(("--game-id", game_id))
    if game_index is not None:
        command.extend(("--game-index", str(game_index)))
    if seed is not None:
        command.extend(("--seed", str(seed)))
    if dry_run:
        command.append("--dry-run")
    if save_video:
        command.append("--save-video")
    if video_fps is not None:
        command.extend(("--video-fps", str(video_fps)))
    if video_frame_scale is not None:
        command.extend(("--video-frame-scale", str(video_frame_scale)))
    if world_loss_mode is not None:
        command.extend(("--world-loss-mode", world_loss_mode))
    if latent_loss_weight is not None:
        command.extend(("--latent-loss-weight", str(latent_loss_weight)))
    if latent_changed_patch_weight is not None:
        command.extend(
            ("--latent-changed-patch-weight", str(latent_changed_patch_weight))
        )
    if latent_huber_beta is not None:
        command.extend(("--latent-huber-beta", str(latent_huber_beta)))
    if latent_cosine_loss_weight is not None:
        command.extend(("--latent-cosine-loss-weight", str(latent_cosine_loss_weight)))
    if latent_cosine_min_delta_norm is not None:
        command.extend(
            ("--latent-cosine-min-delta-norm", str(latent_cosine_min_delta_norm))
        )
    if latent_learning_progress_normalization is True:
        command.append("--latent-learning-progress-normalization")
    elif latent_learning_progress_normalization is False:
        command.append("--no-latent-learning-progress-normalization")
    if latent_learning_progress_normalization_floor is not None:
        command.extend(
            (
                "--latent-learning-progress-normalization-floor",
                str(latent_learning_progress_normalization_floor),
            )
        )
    return command


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


if modal is not None:
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
    run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ca-certificates", "git", "libgl1", "libglib2.0-0")
        .pip_install(
            "accelerate>=1.12.0",
            "arc-agi>=0.9.8",
            "hf-xet>=1.4.3",
            "imageio>=2.37.0",
            "imageio-ffmpeg>=0.6.0",
            "matplotlib>=3.10.9",
            "num2words>=0.5.14",
            "peft>=0.18.0",
            "pillow>=12.0.0",
            "protobuf>=6.33.0",
            "pyyaml>=6.0.3",
            "rich>=13.9.4",
            "safetensors>=0.7.0",
            "sentencepiece>=0.2.1",
            "torch>=2.9.0",
            "torchvision>=0.24.0",
            "transformers>=5.5.2,<6",
        )
        .env(_modal_env())
        .add_local_file(
            "pyproject.toml",
            remote_path=str(REMOTE_REPO_PATH / "pyproject.toml"),
        )
        .add_local_dir("src", remote_path=str(REMOTE_REPO_PATH / "src"))
        .add_local_dir(
            "environment_files",
            remote_path=str(REMOTE_REPO_PATH / "environment_files"),
        )
        .add_local_file(
            "experiments/single_vlm_lora/run_single_vlm_lora.py",
            remote_path=str(REMOTE_EXPERIMENT_PATH / "run_single_vlm_lora.py"),
        )
        .add_local_dir(
            "experiments/single_vlm_lora/single_vlm_arc",
            remote_path=str(REMOTE_EXPERIMENT_PATH / "single_vlm_arc"),
        )
        .add_local_dir(
            "experiments/single_vlm_lora/configs",
            remote_path=str(REMOTE_EXPERIMENT_PATH / "configs"),
        )
    )

    app = modal.App(APP_NAME)

    def _modal_secrets() -> list[Any]:
        secrets = []
        if hf_token := os.environ.get("HF_TOKEN"):
            secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token}))
        return secrets

    def _start_live_run_committer(
        stop_event: threading.Event,
        *,
        interval_seconds: int,
    ) -> threading.Thread | None:
        """Commit the run volume periodically so results are visible mid-run."""

        if interval_seconds <= 0:
            return None
        thread = threading.Thread(
            target=_commit_run_volume_until_stopped,
            args=(stop_event, interval_seconds),
            daemon=True,
        )
        thread.start()
        return thread

    def _commit_run_volume_until_stopped(
        stop_event: threading.Event,
        interval_seconds: int,
    ) -> None:
        while not stop_event.wait(interval_seconds):
            try:
                run_volume.commit()
            except Exception as exc:  # pragma: no cover - Modal runtime behavior.
                print(f"live run volume commit failed: {exc}", file=sys.stderr)

    @app.function(
        image=image,
        gpu=GPU_TYPE,
        volumes={
            str(MODEL_VOLUME_PATH): model_volume,
            str(RUN_VOLUME_PATH): run_volume,
        },
        secrets=_modal_secrets(),
        timeout=TIMEOUT_SECONDS,
        scaledown_window=300,
    )
    def run_config(
        *,
        config_text: str,
        config_name: str,
        max_turns: int | None = None,
        game_id: str | None = None,
        game_index: int | None = None,
        seed: int | None = None,
        output_dir: str | None = None,
        run_name: str | None = None,
        dry_run: bool = False,
        save_video: bool = False,
        video_fps: int | None = None,
        video_frame_scale: int | None = None,
        world_loss_mode: str | None = None,
        latent_loss_weight: float | None = None,
        latent_changed_patch_weight: float | None = None,
        latent_huber_beta: float | None = None,
        latent_cosine_loss_weight: float | None = None,
        latent_cosine_min_delta_norm: float | None = None,
        latent_learning_progress_normalization: bool | None = None,
        latent_learning_progress_normalization_floor: float | None = None,
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
    ) -> dict[str, Any]:
        """Run the isolated single-VLM LoRA experiment remotely."""

        config_path = REMOTE_CONFIG_DIR / Path(config_name).name
        resolved_output_dir = remote_output_dir(
            config_name=config_name,
            output_dir=output_dir,
            run_name=run_name,
        )
        Path(str(config_path)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(config_path)).write_text(config_text, encoding="utf-8")
        Path(str(resolved_output_dir)).mkdir(parents=True, exist_ok=True)

        command = build_runner_command(
            config_path=config_path,
            output_dir=resolved_output_dir,
            max_turns=max_turns,
            game_id=game_id,
            game_index=game_index,
            seed=seed,
            dry_run=dry_run,
            save_video=save_video,
            video_fps=video_fps,
            video_frame_scale=video_frame_scale,
            world_loss_mode=world_loss_mode,
            latent_loss_weight=latent_loss_weight,
            latent_changed_patch_weight=latent_changed_patch_weight,
            latent_huber_beta=latent_huber_beta,
            latent_cosine_loss_weight=latent_cosine_loss_weight,
            latent_cosine_min_delta_norm=latent_cosine_min_delta_norm,
            latent_learning_progress_normalization=(
                latent_learning_progress_normalization
            ),
            latent_learning_progress_normalization_floor=(
                latent_learning_progress_normalization_floor
            ),
        )

        commit_stop = threading.Event()
        commit_thread = _start_live_run_committer(
            commit_stop,
            interval_seconds=live_commit_seconds,
        )
        try:
            completed = run_streamed_subprocess(
                command,
                env={**os.environ, **_modal_env()},
                cwd=str(REMOTE_REPO_PATH),
            )
        finally:
            commit_stop.set()
            if commit_thread is not None:
                commit_thread.join()
            try:
                model_volume.commit()
            finally:
                run_volume.commit()

        return {
            "returncode": completed.returncode,
            "command": command,
            "config_path": str(config_path),
            "output_dir": str(resolved_output_dir),
            "turns_path": str(resolved_output_dir / "turns.jsonl"),
            "summary_path": str(resolved_output_dir / "summary.json"),
            "video_path": str(resolved_output_dir / "frames.mp4"),
            "frame_manifest_path": str(resolved_output_dir / "frame_manifest.jsonl"),
            "frame_prediction_dir": str(resolved_output_dir / "frame_predictions"),
            "frame_prediction_manifest_path": str(
                resolved_output_dir / "frame_prediction_manifest.jsonl"
            ),
            "latent_prediction_dir": str(resolved_output_dir / "latent_predictions"),
            "latent_prediction_manifest_path": str(
                resolved_output_dir / "latent_prediction_manifest.jsonl"
            ),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    @app.local_entrypoint()
    def main(
        config: str = DEFAULT_CONFIG,
        max_turns: int | None = None,
        game_id: str | None = None,
        game_index: int | None = None,
        seed: int | None = None,
        output_dir: str | None = None,
        run_name: str | None = None,
        dry_run: bool = False,
        save_video: bool = False,
        video_fps: int | None = None,
        video_frame_scale: int | None = None,
        world_loss_mode: str | None = None,
        latent_loss_weight: float | None = None,
        latent_changed_patch_weight: float | None = None,
        latent_huber_beta: float | None = None,
        latent_cosine_loss_weight: float | None = None,
        latent_cosine_min_delta_norm: float | None = None,
        latent_learning_progress_normalization: bool | None = None,
        latent_learning_progress_normalization_floor: float | None = None,
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
    ) -> None:
        """Launch one Modal A100 40GB single-VLM LoRA experiment run."""

        config_path = Path(config)
        config_text = config_path.read_text(encoding="utf-8")
        result = run_config.remote(
            config_text=config_text,
            config_name=config_path.name,
            max_turns=max_turns,
            game_id=game_id,
            game_index=game_index,
            seed=seed,
            output_dir=output_dir,
            run_name=run_name,
            dry_run=dry_run,
            save_video=save_video,
            video_fps=video_fps,
            video_frame_scale=video_frame_scale,
            world_loss_mode=world_loss_mode,
            latent_loss_weight=latent_loss_weight,
            latent_changed_patch_weight=latent_changed_patch_weight,
            latent_huber_beta=latent_huber_beta,
            latent_cosine_loss_weight=latent_cosine_loss_weight,
            latent_cosine_min_delta_norm=latent_cosine_min_delta_norm,
            latent_learning_progress_normalization=(
                latent_learning_progress_normalization
            ),
            latent_learning_progress_normalization_floor=(
                latent_learning_progress_normalization_floor
            ),
            live_commit_seconds=live_commit_seconds,
        )
        print(f"remote output dir: {result['output_dir']}")
        print(f"remote summary: {result['summary_path']}")
        if save_video:
            print(f"remote video: {result['video_path']}")
            print(f"remote frame manifest: {result['frame_manifest_path']}")
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])

else:
    app = None
