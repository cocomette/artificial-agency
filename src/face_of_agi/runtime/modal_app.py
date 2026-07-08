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
DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_VLLM_PORT = 8000


@dataclass(frozen=True)
class StreamedProcessResult:
    """Completed process data captured while output was streamed live."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class VLLMServerConfig:
    """Modal-managed vLLM server settings parsed from one runtime config."""

    model: str
    host: str = DEFAULT_VLLM_HOST
    port: int = DEFAULT_VLLM_PORT
    max_model_len: int | None = None
    reasoning_parser: str | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        """Return the local OpenAI-compatible base URL for the server."""

        return f"http://{self.host}:{self.port}/v1"


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

    for role_name in ("world",):
        role = models.get(role_name) or {}
        if _backend(role) == "ollama":
            _append_model(found, role.get("model") or shared.get("model"))

    updater = models.get("updater") or {}
    if isinstance(updater, dict):
        for task in ("world", "agent", "general"):
            role = updater.get(task) or {}
            if _backend(role) == "ollama":
                _append_model(found, role.get("model") or shared.get("model"))

    return tuple(found)


def vllm_server_config_from_config_text(config_text: str) -> VLLMServerConfig | None:
    """Return Modal-managed vLLM server settings referenced by a YAML config."""

    raw = yaml.safe_load(config_text) or {}
    models = raw.get("models") or {}
    if not isinstance(models, dict):
        return None

    shared = models.get("shared_vlm") or {}
    shared_backend = _backend(shared)
    found: list[str] = []
    if shared_backend == "vllm":
        _append_model(found, shared.get("model"))
    for role_name in ("agent", "world"):
        role = models.get(role_name) or {}
        if _backend(role) == "vllm":
            _append_model(found, role.get("model") or shared.get("model"))
    updater = models.get("updater") or {}
    if isinstance(updater, dict):
        for task in ("world", "agent", "general"):
            role = updater.get(task) or {}
            if _backend(role) == "vllm":
                _append_model(found, role.get("model") or shared.get("model"))
    if not found:
        return None

    server = shared.get("server") if isinstance(shared, dict) else None
    if server is None:
        server = {}
    if not isinstance(server, dict):
        raise ValueError("models.shared_vlm.server must be a mapping")
    extra_args = server.get("extra_args") or ()
    if isinstance(extra_args, str):
        extra_args = (extra_args,)
    if not isinstance(extra_args, (list, tuple)):
        raise ValueError("models.shared_vlm.server.extra_args must be a list")

    return VLLMServerConfig(
        model=found[0],
        host=str(server.get("host", DEFAULT_VLLM_HOST)),
        port=int(server.get("port", DEFAULT_VLLM_PORT)),
        max_model_len=_optional_server_int(server.get("max_model_len")),
        reasoning_parser=_optional_server_string(server.get("reasoning_parser")),
        extra_args=tuple(str(arg) for arg in extra_args),
    )


def vllm_server_command(config: VLLMServerConfig) -> tuple[str, ...]:
    """Return the vLLM serve command for one parsed server config."""

    command = [
        "vllm",
        "serve",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    if config.max_model_len is not None:
        command.extend(["--max-model-len", str(config.max_model_len)])
    if config.reasoning_parser:
        command.extend(["--reasoning-parser", config.reasoning_parser])
    command.extend(config.extra_args)
    return tuple(command)


def _append_model(target: list[str], model: Any) -> None:
    if isinstance(model, str) and model and model not in target:
        target.append(model)


def _backend(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("backend") or "").lower()


def _optional_server_string(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _optional_server_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


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
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
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
        .uv_pip_install("vllm>=0.19.0")
        .env(_modal_env())
        .add_local_dir("src/face_of_agi", remote_path="/root/face_of_agi")
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("debug", remote_path="/root/debug")
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
        def start_model_state(self) -> None:
            MODEL_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            RUN_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            (MODEL_VOLUME_PATH / "ollama").mkdir(parents=True, exist_ok=True)
            (MODEL_VOLUME_PATH / "huggingface").mkdir(parents=True, exist_ok=True)
            self._vllm_process = None
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

            for model in ollama_models_from_config_text(config_text):
                with runtime_timing.span("modal.ollama_pull", model=model):
                    subprocess.run(
                        ["ollama", "pull", model],
                        check=True,
                        env={**os.environ, **_modal_env()},
                    )
            with runtime_timing.span("modal.model_volume.commit"):
                model_volume.commit()

            vllm_config = vllm_server_config_from_config_text(config_text)
            if vllm_config is not None:
                self._start_vllm(vllm_config)

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

        def _start_vllm(self, config: VLLMServerConfig) -> None:
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
                _wait_for_vllm(config, process=self._vllm_process)

    @app.local_entrypoint()
    def main(
        config: str = "src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml",
        database_name: str = "memory.sqlite",
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
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
            timing=timing,
            playback_run_id=playback_run_id,
            playback_game_id=playback_game_id,
            playback_turn_id=playback_turn_id,
        )
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])

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


def _wait_for_vllm(
    config: VLLMServerConfig,
    *,
    process: subprocess.Popen[str] | None = None,
    timeout_seconds: float = 900.0,
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
