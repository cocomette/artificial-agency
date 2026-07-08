"""Modal launcher for remote FACE-OF-AGI game-loop runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, TextIO
from urllib.error import URLError
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    local_path = Path(__file__).resolve()
    candidates = [Path("/root/src")]
    if len(local_path.parents) > 2:
        candidates.append(local_path.parents[2])
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break

from face_of_agi.runtime import timing as runtime_timing
from face_of_agi.runtime.vllm_server import (
    VLLMServerConfig,
    vllm_server_command,
    vllm_server_config_from_config_text,
)

try:
    import modal
except ImportError:  # pragma: no cover - local tests need not install Modal.
    modal = None  # type: ignore[assignment]

APP_NAME = "face-of-agi-game-loop"
MODAL_GPU = "H100"
MODEL_VOLUME_NAME = "face-of-agi-local-models"
RUN_VOLUME_NAME = "face-of-agi-runs"
MODEL_VOLUME_PATH = Path("/vol/models")
RUN_VOLUME_PATH = Path("/vol/runs")
REMOTE_CONFIG_DIR = RUN_VOLUME_PATH / "configs"
LIVE_RUN_COMMIT_SECONDS = 30
VLLM_READY_TIMEOUT_SECONDS = 60 * 45
DEFAULT_MODAL_CONFIG = (
    "src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml"
)
DEFAULT_MODAL_PARALLEL_CONFIG = (
    "src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8_parallel.yaml"
)
LOCAL_ENVIRONMENTS_DIR = Path("environment_files")
REMOTE_ENVIRONMENTS_DIR = Path("/root/environment_files")


@dataclass(frozen=True)
class StreamedProcessResult:
    """Completed process data captured while output was streamed live."""

    returncode: int
    stdout: str
    stderr: str


def _modal_env() -> dict[str, str]:
    """Return env vars that place vLLM model state on the model Volume."""

    model_cache = MODEL_VOLUME_PATH / "model-cache"
    return {
        "HF_HOME": str(model_cache),
        "HF_HUB_CACHE": str(model_cache / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(model_cache / "hub"),
        "TRANSFORMERS_CACHE": str(model_cache / "transformers"),
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "MPLBACKEND": "Agg",
    }


def _add_local_environment_files(image: Any) -> Any:
    """Add local ARC files when present."""

    if LOCAL_ENVIRONMENTS_DIR.exists():
        return image.add_local_dir(
            str(LOCAL_ENVIRONMENTS_DIR),
            remote_path=str(REMOTE_ENVIRONMENTS_DIR),
        )
    return image


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


if modal is not None:
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
    run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)

    image = _add_local_environment_files(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ca-certificates", "git", "zstd")
        .uv_sync()
        .uv_pip_install("vllm>=0.19.0")
        .env(_modal_env())
        .run_commands(f"mkdir -p {REMOTE_ENVIRONMENTS_DIR}")
        .add_local_dir("src/face_of_agi", remote_path="/root/face_of_agi")
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("debug", remote_path="/root/debug")
    )

    app = modal.App(APP_NAME)

    def _start_live_run_committer(
        stop_event: threading.Event,
        *,
        interval_seconds: int,
    ) -> threading.Thread | None:
        """Commit the run volume periodically so remote dashboards can pull it."""

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
                with runtime_timing.span("modal.run_volume.live_commit"):
                    run_volume.commit()
            except Exception as exc:  # pragma: no cover - Modal runtime behavior.
                print(f"live run volume commit failed: {exc}", file=sys.stderr)

    @app.cls(
        image=image,
        gpu=MODAL_GPU,
        volumes={
            str(MODEL_VOLUME_PATH): model_volume,
            str(RUN_VOLUME_PATH): run_volume,
        },
        timeout=60 * 60 * 6,
    )
    class ModalGameRunner:
        """Single-container H100 runner for real game-loop executions."""

        @modal.enter()
        def start_model_state(self) -> None:
            MODEL_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            RUN_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            (MODEL_VOLUME_PATH / "model-cache").mkdir(parents=True, exist_ok=True)
            self._vllm_process = None

        @modal.method()
        def run_config(
            self,
            *,
            config_text: str,
            config_name: str,
            database_name: str = "memory.sqlite",
            live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
            vllm_ready_timeout_seconds: int = VLLM_READY_TIMEOUT_SECONDS,
            timing: bool = False,
            playback_run_id: str | None = None,
            playback_game_id: str | None = None,
            playback_turn_id: int | None = None,
        ) -> dict[str, Any]:
            """Run the existing runtime shell remotely and persist artifacts."""

            timing_path = RUN_VOLUME_PATH / "timing" / f"{database_name}.jsonl"
            if timing:
                os.environ["FACE_OF_AGI_TIMING"] = "1"
                os.environ["FACE_OF_AGI_TIMING_JSONL"] = str(timing_path)

            REMOTE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            config_path = REMOTE_CONFIG_DIR / Path(config_name).name
            config_path.write_text(config_text, encoding="utf-8")

            with runtime_timing.span("modal.model_volume.commit"):
                model_volume.commit()

            vllm_config = vllm_server_config_from_config_text(config_text)
            if vllm_config is not None:
                self._start_vllm(
                    vllm_config,
                    ready_timeout_seconds=vllm_ready_timeout_seconds,
                )

            database_path = RUN_VOLUME_PATH / database_name
            command = [
                sys.executable,
                "-m",
                "face_of_agi.runtime.shell",
                "--config",
                str(config_path),
                "--database",
                str(database_path),
                *_playback_command_args(
                    playback_run_id=playback_run_id,
                    playback_game_id=playback_game_id,
                    playback_turn_id=playback_turn_id,
                ),
            ]
            commit_stop = threading.Event()
            commit_thread = _start_live_run_committer(
                commit_stop,
                interval_seconds=live_commit_seconds,
            )
            try:
                subprocess_env = {**os.environ, **_modal_env()}
                if timing:
                    subprocess_env["FACE_OF_AGI_TIMING"] = "1"
                    subprocess_env["FACE_OF_AGI_TIMING_JSONL"] = str(timing_path)
                with runtime_timing.span("modal.runtime_subprocess"):
                    completed = run_streamed_subprocess(
                        command,
                        env=subprocess_env,
                        cwd="/root",
                    )
            finally:
                commit_stop.set()
                if commit_thread is not None:
                    commit_thread.join()
                with runtime_timing.span("modal.run_volume.final_commit"):
                    run_volume.commit()
            return {
                "returncode": completed.returncode,
                "command": command,
                "database_path": str(database_path),
                "config_path": str(config_path),
                "timing_path": str(timing_path) if timing else None,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        def _start_vllm(
            self,
            config: VLLMServerConfig,
            *,
            ready_timeout_seconds: int,
        ) -> None:
            process = getattr(self, "_vllm_process", None)
            if process is not None and process.poll() is None:
                return

            command = vllm_server_command(config)
            with runtime_timing.span("modal.vllm_serve", model=config.model):
                self._vllm_process = subprocess.Popen(
                    list(command),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    env={**os.environ, **_modal_env()},
                )
                self._vllm_stdout_thread = _forward_stream(
                    self._vllm_process.stdout,
                    sys.stdout,
                    [],
                )
                self._vllm_stderr_thread = _forward_stream(
                    self._vllm_process.stderr,
                    sys.stderr,
                    [],
                )
                _wait_for_vllm(
                    config,
                    process=self._vllm_process,
                    timeout_seconds=ready_timeout_seconds,
                )

    @app.local_entrypoint()
    def main(
        config: str = DEFAULT_MODAL_CONFIG,
        database_name: str = "memory.sqlite",
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
        vllm_ready_timeout_seconds: int = VLLM_READY_TIMEOUT_SECONDS,
        timing: bool = False,
        playback_run_id: str | None = None,
        playback_game_id: str | None = None,
        playback_turn_id: int | None = None,
    ) -> None:
        """Launch one Modal H100 game run from a local config file."""

        config_path = Path(config)
        config_text = config_path.read_text(encoding="utf-8")
        result = ModalGameRunner().run_config.remote(
            config_text=config_text,
            config_name=config_path.name,
            database_name=database_name,
            live_commit_seconds=live_commit_seconds,
            vllm_ready_timeout_seconds=vllm_ready_timeout_seconds,
            timing=timing,
            playback_run_id=playback_run_id,
            playback_game_id=playback_game_id,
            playback_turn_id=playback_turn_id,
        )
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])

    @app.local_entrypoint()
    def submit_detached(
        config: str = DEFAULT_MODAL_PARALLEL_CONFIG,
        database_name: str = "memory.sqlite",
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
        vllm_ready_timeout_seconds: int = VLLM_READY_TIMEOUT_SECONDS,
        timing: bool = False,
        playback_run_id: str | None = None,
        playback_game_id: str | None = None,
        playback_turn_id: int | None = None,
    ) -> None:
        """Submit a run_config call and return without waiting for completion."""

        config_path = Path(config)
        call = ModalGameRunner().run_config.spawn(
            config_text=config_path.read_text(encoding="utf-8"),
            config_name=config_path.name,
            database_name=database_name,
            live_commit_seconds=live_commit_seconds,
            vllm_ready_timeout_seconds=vllm_ready_timeout_seconds,
            timing=timing,
            playback_run_id=playback_run_id,
            playback_game_id=playback_game_id,
            playback_turn_id=playback_turn_id,
        )
        print(f"spawned run_config call: {call.object_id}")

else:
    app = None


def _playback_command_args(
    *,
    playback_run_id: str | None,
    playback_game_id: str | None,
    playback_turn_id: int | None,
) -> list[str]:
    """Return runtime-shell playback args for a Modal subprocess."""

    values = (playback_run_id, playback_game_id, playback_turn_id)
    if all(value in {None, ""} for value in values):
        return []
    if any(value in {None, ""} for value in values):
        raise ValueError(
            "playback_run_id, playback_game_id, and playback_turn_id "
            "must be provided together"
        )
    return [
        "--playback-run-id",
        str(playback_run_id),
        "--playback-game-id",
        str(playback_game_id),
        "--playback-turn-id",
        str(playback_turn_id),
    ]


def _wait_for_vllm(
    config: VLLMServerConfig,
    *,
    process: subprocess.Popen[str] | None = None,
    timeout_seconds: float = VLLM_READY_TIMEOUT_SECONDS,
) -> None:
    """Wait until the vLLM OpenAI-compatible server accepts API requests."""

    deadline = time.monotonic() + timeout_seconds
    request = Request(f"{config.base_url}/models")
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "vLLM exited before becoming ready inside the Modal container "
                f"with return code {process.returncode}"
            )
        try:
            with urlopen(request, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except URLError:
            pass
        except TimeoutError:
            pass
        time.sleep(2)
    raise RuntimeError(
        "vLLM did not become ready inside the Modal container "
        f"at {config.base_url}"
    )
