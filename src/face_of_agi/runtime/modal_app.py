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

import yaml

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

try:
    import modal
except ImportError:  # pragma: no cover - local tests need not install Modal.
    modal = None  # type: ignore[assignment]

APP_NAME = "face-of-agi-game-loop"
MODEL_VOLUME_NAME = "face-of-agi-local-models"
RUN_VOLUME_NAME = "face-of-agi-runs"
MODEL_VOLUME_PATH = Path("/vol/models")
RUN_VOLUME_PATH = Path("/vol/runs")
REMOTE_CONFIG_DIR = RUN_VOLUME_PATH / "configs"
LIVE_RUN_COMMIT_SECONDS = 30


@dataclass(frozen=True)
class StreamedProcessResult:
    """Completed process data captured while output was streamed live."""

    returncode: int
    stdout: str
    stderr: str


def ollama_models_from_config_text(config_text: str) -> tuple[str, ...]:
    """Return Ollama model ids referenced by one runtime YAML config."""

    raw = yaml.safe_load(config_text) or {}
    models = raw.get("models") or {}
    if not isinstance(models, dict):
        return ()

    found: list[str] = []
    shared = models.get("shared_vlm") or {}
    if _backend(shared) == "ollama":
        _append_model(found, shared.get("model"))

    for role_name in ("agent",):
        role = models.get(role_name) or {}
        if _backend(role) == "ollama":
            _append_model(found, role.get("model") or shared.get("model"))

    for role_name in ("world", "goal"):
        role = models.get(role_name) or {}
        if _backend(role) == "ollama":
            _append_model(found, role.get("model") or shared.get("model"))

    updater = models.get("updater") or {}
    if isinstance(updater, dict):
        for task in ("world", "goal", "agent", "general"):
            role = updater.get(task) or {}
            if _backend(role) == "ollama":
                _append_model(found, role.get("model") or shared.get("model"))

    return tuple(found)


def _append_model(target: list[str], model: Any) -> None:
    if isinstance(model, str) and model and model not in target:
        target.append(model)


def _backend(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("backend") or "").lower()


def _modal_env() -> dict[str, str]:
    """Return env vars that place all local model state on the model Volume."""

    hf_home = MODEL_VOLUME_PATH / "huggingface"
    return {
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_MODELS": str(MODEL_VOLUME_PATH / "ollama"),
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "DIFFUSERS_CACHE": str(hf_home / "diffusers"),
        "MPLBACKEND": "Agg",
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


if modal is not None:
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
    run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ca-certificates", "curl", "git", "zstd")
        .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
        .uv_sync(extras=["ml"])
        .env(_modal_env())
        .add_local_dir("src/face_of_agi", remote_path="/root/face_of_agi")
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("environment_files", remote_path="/root/environment_files")
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
        gpu="H100!",
        volumes={
            str(MODEL_VOLUME_PATH): model_volume,
            str(RUN_VOLUME_PATH): run_volume,
        },
        timeout=60 * 60 * 6,
        scaledown_window=300,
    )
    class ModalGameRunner:
        """Single-container H100 runner for real game-loop executions."""

        @modal.enter()
        def start_ollama(self) -> None:
            MODEL_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            RUN_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            (MODEL_VOLUME_PATH / "ollama").mkdir(parents=True, exist_ok=True)
            (MODEL_VOLUME_PATH / "huggingface").mkdir(parents=True, exist_ok=True)
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env={**os.environ, **_modal_env()},
            )
            _wait_for_ollama()

        @modal.method()
        def run_config(
            self,
            *,
            config_text: str,
            config_name: str,
            database_name: str = "memory.sqlite",
            live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
            timing: bool = False,
        ) -> dict[str, Any]:
            """Run the existing runtime shell remotely and persist artifacts."""

            timing_path = RUN_VOLUME_PATH / "timing" / f"{database_name}.jsonl"
            if timing:
                os.environ["FACE_OF_AGI_TIMING"] = "1"
                os.environ["FACE_OF_AGI_TIMING_JSONL"] = str(timing_path)

            REMOTE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            config_path = REMOTE_CONFIG_DIR / Path(config_name).name
            config_path.write_text(config_text, encoding="utf-8")

            for model in ollama_models_from_config_text(config_text):
                with runtime_timing.span("modal.ollama_pull", model=model):
                    subprocess.run(
                        ["ollama", "pull", model],
                        check=True,
                        env={**os.environ, **_modal_env()},
                    )
            with runtime_timing.span("modal.model_volume.commit"):
                model_volume.commit()

            database_path = RUN_VOLUME_PATH / database_name
            command = [
                sys.executable,
                "-m",
                "face_of_agi.runtime.shell",
                "--config",
                str(config_path),
                "--database",
                str(database_path),
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

    @app.local_entrypoint()
    def main(
        config: str = "src/face_of_agi/runtime/configs/ollama_all_gemma4_26b.yaml",
        database_name: str = "memory.sqlite",
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
        timing: bool = False,
    ) -> None:
        """Launch one Modal H100 game run from a local config file."""

        config_path = Path(config)
        config_text = config_path.read_text(encoding="utf-8")
        result = ModalGameRunner().run_config.remote(
            config_text=config_text,
            config_name=config_path.name,
            database_name=database_name,
            live_commit_seconds=live_commit_seconds,
            timing=timing,
        )
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])

else:
    app = None


def _wait_for_ollama(timeout_seconds: float = 60.0) -> None:
    """Wait until the local Ollama server accepts CLI requests."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            env={**os.environ, **_modal_env()},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("Ollama did not become ready inside the Modal container")
