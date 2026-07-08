"""Modal launcher for remote FACE-OF-AGI game-loop runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

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
PUBLIC_GAMES_DIR = RUN_VOLUME_PATH / "public-games"
PUBLIC_GAME_CATALOG_PATH = PUBLIC_GAMES_DIR / "local_games.json"
PUBLIC_ENVIRONMENTS_DIR = PUBLIC_GAMES_DIR / "environment_files"
PUBLIC_GAME_RECORDINGS_DIR = PUBLIC_GAMES_DIR / "recordings"
MODAL_BASE_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
LIVE_RUN_COMMIT_SECONDS = 30
DEFAULT_GAME_CATALOG_PATH = "src/face_of_agi/environment/local_games.json"
DEFAULT_ENVIRONMENTS_DIR = "environment_files"
MODAL_TORCH_STACK_PACKAGES = (
    "torch==2.11.0",
    "torchaudio==2.11.0",
    "torchvision==0.26.0",
)
MODAL_HF_STACK_PACKAGES = (
    "huggingface-hub<2.0,>=1.5.0",
    "transformers==5.12.1",
)
MODAL_VLLM_VERSION = "0.19.1"
MODAL_VLLM_STACK_PACKAGES = (
    "compressed-tensors==0.15.0.1",
    "flashinfer-python==0.6.6",
    "quack-kernels==0.4.1",
    "torch-c-dlpack-ext",
    f"vllm=={MODAL_VLLM_VERSION}",
    "xgrammar==0.2.1",
)
MODAL_VLLM_DEPENDENCY_PACKAGES = (
    "apache-tvm-ffi>=0.1.2",
    "click",
    "cloudpickle",
    "cuda-tile",
    "einops",
    "loguru",
    "ml-dtypes",
    "ninja",
    "numpy>=1.23.5",
    "nvidia-cudnn-frontend>=1.13.0",
    "nvidia-cutlass-dsl>=4.4.2",
    "nvidia-ml-py",
    "packaging>=24.2",
    "psutil",
    "pycountry",
    "pydantic>=2.0",
    "pydantic-extra-types",
    "requests",
    "setuptools",
    "tabulate",
    "tqdm>=4.62.3",
    "transformers>=4.45.0",
    "typing-extensions>=4.10.0",
    "z3-solver<4.15.5,>=4.13.0",
)
MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES = (
    "regex",
    "cachetools",
    "psutil",
    "sentencepiece",
    "numpy",
    "requests>=2.26.0",
    "tqdm",
    "blake3",
    "py-cpuinfo",
    "huggingface-hub<2.0,>=1.5.0",
    "transformers==5.12.1",
    "tokenizers>=0.21.1",
    "protobuf!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*,>=5.29.6",
    "fastapi[standard]>=0.115.0",
    "annotated-doc",
    "starlette<1.0.0,>=0.40.0",
    "uvicorn>=0.30.0",
    "uvloop",
    "httptools",
    "websockets",
    "python-multipart",
    "email-validator",
    "aiohttp>=3.13.3",
    "openai>=2.0.0",
    "pydantic>=2.12.0",
    "prometheus_client>=0.18.0",
    "pillow",
    "prometheus-fastapi-instrumentator>=7.0.0",
    "tiktoken>=0.6.0",
    "lm-format-enforcer==0.11.3",
    "llguidance<1.4.0,>=1.3.0",
    "outlines_core==0.2.11",
    "diskcache==5.6.3",
    "lark==1.2.2",
    "typing_extensions>=4.10",
    "filelock>=3.16.1",
    "partial-json-parser",
    "pyzmq>=25.0.0",
    "msgspec",
    "gguf>=0.17.0",
    "mistral_common[image]>=1.10.0",
    "opencv-python-headless>=4.13.0",
    "pyyaml",
    "six>=1.16.0",
    "setuptools<81.0.0,>=77.0.3",
    "einops",
    "depyf==0.20.0",
    "cloudpickle",
    "watchfiles",
    "python-json-logger",
    "ninja",
    "pybase64",
    "cbor2",
    "ijson",
    "jmespath",
    "setproctitle",
    "openai-harmony>=0.0.3",
    "anthropic>=0.71.0",
    "model-hosting-container-standards<1.0.0,>=0.1.13",
    "mcp",
    "opentelemetry-sdk>=1.27.0",
    "opentelemetry-api>=1.27.0",
    "opentelemetry-exporter-otlp>=1.27.0",
    "opentelemetry-semantic-conventions-ai>=0.4.1",
    "numba==0.61.2",
    "flashinfer-cubin==0.6.6",
    "nvidia-cudnn-frontend<1.19.0,>=1.13.0",
    "nvidia-cutlass-dsl>=4.4.0.dev1",
)
DEFAULT_MODAL_CONFIG = (
    "src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml"
)
DEFAULT_MODAL_PARALLEL_CONFIG = (
    "src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8_parallel.yaml"
)
DEFAULT_MODAL_HF_PROBE_CONFIG = (
    "src/face_of_agi/runtime/configs/hf/hf_h100_qwen36_35b_bnb4_debug.yaml"
)
DEFAULT_HF_PROBE_DB_GLOB = "runs/kaggle-debug/runs/memory-game-index-*.sqlite"


@dataclass(frozen=True)
class StreamedProcessResult:
    """Completed process data captured while output was streamed live."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class VLLMProbeResult:
    """One raw OpenAI-compatible vLLM probe response."""

    name: str
    status: int | None
    ok: bool
    body: str
    error: str | None = None


def _modal_env() -> dict[str, str]:
    """Return env vars that place all local model state on the model Volume."""

    hf_home = MODEL_VOLUME_PATH / "huggingface"
    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "DIFFUSERS_CACHE": str(hf_home / "diffusers"),
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
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

    ml_base_image = (
        modal.Image.from_registry(MODAL_BASE_IMAGE, add_python="3.12")
        .apt_install("ca-certificates", "curl", "git", "zstd")
        .uv_sync(extras=["ml"])
        .uv_pip_install(*MODAL_TORCH_STACK_PACKAGES)
        .uv_pip_install(
            *MODAL_HF_STACK_PACKAGES,
            pre=True,
            extra_options="--no-deps",
        )
        .env(_modal_env())
    )
    ml_image = (
        ml_base_image
        .add_local_dir("src/face_of_agi", remote_path="/root/face_of_agi")
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("debug", remote_path="/root/debug")
    )
    image = (
        ml_base_image
        .uv_pip_install(
            *MODAL_VLLM_DEPENDENCY_PACKAGES,
            pre=True,
            extra_options="--no-deps",
        )
        .uv_pip_install(
            *MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES,
            pre=True,
            extra_options="--no-deps",
        )
        .uv_pip_install(*MODAL_VLLM_STACK_PACKAGES, extra_options="--no-deps")
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
            (MODEL_VOLUME_PATH / "huggingface").mkdir(parents=True, exist_ok=True)
            self._vllm_process = None
            self._vllm_command = None

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

            database_path = RUN_VOLUME_PATH / database_name
            config_text = modal_runtime_config_text(
                config_text,
                adapter_root=_modal_lora_adapter_root(database_name),
            )
            REMOTE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            config_path = REMOTE_CONFIG_DIR / Path(config_name).name
            config_path.write_text(config_text, encoding="utf-8")

            vllm_config = vllm_server_config_from_config_text(config_text)
            if vllm_config is not None:
                self._start_vllm(vllm_config)

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
                subprocess_env.update(
                    _vllm_restart_env(
                        getattr(self, "_vllm_process", None),
                        getattr(self, "_vllm_command", None),
                        cwd="/root",
                    )
                )
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

        @modal.method()
        def probe_config(self, *, config_text: str) -> list[dict[str, Any]]:
            """Start vLLM and run raw request probes against the chat endpoint."""

            config_text = modal_runtime_config_text(config_text)
            vllm_config = vllm_server_config_from_config_text(config_text)
            if vllm_config is None:
                raise ValueError("Modal probe requires a config with vLLM roles")
            self._start_vllm(vllm_config)
            return [
                {
                    "name": result.name,
                    "status": result.status,
                    "ok": result.ok,
                    "body": result.body,
                    "error": result.error,
                }
                for result in probe_vllm_chat_endpoint(vllm_config)
            ]

        @modal.method()
        def hf_probe_config(
            self,
            *,
            config_text: str,
            samples: list[dict[str, Any]],
        ) -> dict[str, Any]:
            """Run the single-HF H100 feasibility probe."""
            return _run_hf_probe_config(config_text=config_text, samples=samples)

        def _start_vllm(self, config: VLLMServerConfig) -> None:
            command = vllm_server_command(config)
            self._vllm_command = list(command)
            process = getattr(self, "_vllm_process", None)
            if process is not None and process.poll() is None:
                return

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

    @app.cls(
        image=ml_image,
        gpu=MODAL_GPU,
        volumes={
            str(MODEL_VOLUME_PATH): model_volume,
            str(RUN_VOLUME_PATH): run_volume,
        },
        timeout=60 * 60 * 6,
    )
    class ModalHFRunner:
        """HF-only H100 runner that avoids vLLM dependency overrides."""

        @modal.enter()
        def start_model_state(self) -> None:
            MODEL_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            RUN_VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            (MODEL_VOLUME_PATH / "huggingface").mkdir(parents=True, exist_ok=True)

        @modal.method()
        def hf_probe_config(
            self,
            *,
            config_text: str,
            samples: list[dict[str, Any]],
        ) -> dict[str, Any]:
            """Run the single-HF H100 feasibility probe."""

            return _run_hf_probe_config(config_text=config_text, samples=samples)

    @app.local_entrypoint()
    def main(
        config: str = DEFAULT_MODAL_CONFIG,
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

    @app.local_entrypoint()
    def probe(config: str = DEFAULT_MODAL_CONFIG) -> None:
        """Launch vLLM on Modal and print small raw chat request probes."""

        config_path = Path(config)
        results = ModalGameRunner().probe_config.remote(
            config_text=config_path.read_text(encoding="utf-8"),
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        if not any(result["ok"] for result in results):
            raise SystemExit(1)

    @app.local_entrypoint()
    def hf_probe(
        config: str = DEFAULT_MODAL_HF_PROBE_CONFIG,
        replay_db_glob: str = DEFAULT_HF_PROBE_DB_GLOB,
        max_samples: int = 8,
    ) -> None:
        """Run the single-HF H100 feasibility probe on Modal."""

        config_path = Path(config)
        samples = hf_probe_samples_from_local_dbs(
            replay_db_glob,
            max_samples=max_samples,
        )
        result = ModalHFRunner().hf_probe_config.remote(
            config_text=config_path.read_text(encoding="utf-8"),
            samples=samples,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("ok"):
            raise SystemExit(1)

    @app.local_entrypoint()
    def submit_detached(
        config: str = DEFAULT_MODAL_PARALLEL_CONFIG,
        database_name: str = "memory.sqlite",
        live_commit_seconds: int = LIVE_RUN_COMMIT_SECONDS,
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
            timing=timing,
            playback_run_id=playback_run_id,
            playback_game_id=playback_game_id,
            playback_turn_id=playback_turn_id,
        )
        print(f"spawned run_config call: {call.object_id}")

else:
    app = None


def _vllm_restart_env(
    process: Any,
    command: Sequence[str] | None,
    *,
    cwd: str,
) -> dict[str, str]:
    """Return subprocess env vars allowing runtime to restart local vLLM."""

    if process is None or process.poll() is not None or not command:
        return {}
    return {
        "FACE_OF_AGI_VLLM_PID": str(process.pid),
        "FACE_OF_AGI_VLLM_RESTART_COMMAND_JSON": json.dumps(list(command)),
        "FACE_OF_AGI_VLLM_RESTART_CWD": cwd,
    }


def _run_hf_probe_config(
    *,
    config_text: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the single-HF H100 feasibility probe in the active Modal container."""

    from face_of_agi.environment import load_environment_config
    from face_of_agi.orchestration import online_lora
    from face_of_agi.runtime.shell import (
        _build_hf_engine_for_environment_config,
    )

    config_text = modal_runtime_config_text(
        config_text,
        adapter_root=_modal_lora_adapter_root("hf-probe.sqlite"),
    )
    REMOTE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = REMOTE_CONFIG_DIR / "hf-probe.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    environment_config = load_environment_config(config_path)
    engine = _build_hf_engine_for_environment_config(environment_config)
    if engine is None:
        raise RuntimeError("HF probe config did not build an HF engine")
    records = [_replay_sample_from_jsonable(item) for item in samples]
    by_role = _probe_samples_by_role(records)
    probe_root = RUN_VOLUME_PATH / "hf-probes" / str(int(time.time()))
    metrics: list[dict[str, Any]] = []

    def mark(phase: str) -> None:
        metrics.append(_cuda_metric(phase))

    try:
        mark("start")
        world_request = dict(by_role["world"][0].prompt_json["request"])
        world_request["model"] = engine.model_name
        world_request["max_completion_tokens"] = 16
        world_request["max_tokens"] = 16
        engine.chat(world_request)
        mark("post_world_generate")
        online_lora._train_world_sft_adapter_hf(
            engine=engine,
            adapter_path=probe_root / "world",
            samples=by_role["world"],
            config=environment_config.online_lora,
            previous_adapter_path=None,
            adapter_name="hf_probe_world",
        )
        mark("post_world_sft")
        world_request["model"] = "hf_probe_world"
        engine.chat(world_request)
        mark("post_world_adapter_generate")
        online_lora._train_grpo_adapter_hf(
            engine=engine,
            role="interest",
            adapter_path=probe_root / "interest",
            samples=by_role["interest"],
            config=environment_config.online_lora,
            previous_adapter_path=None,
            adapter_name="hf_probe_interest",
        )
        mark("post_interest_grpo")
        online_lora._train_grpo_adapter_hf(
            engine=engine,
            role="agent",
            adapter_path=probe_root / "agent",
            samples=by_role["agent"],
            config=environment_config.online_lora,
            previous_adapter_path=None,
            adapter_name="hf_probe_agent",
        )
        mark("post_agent_grpo")
    except BaseException as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(exc)),
            "metrics": metrics,
            "sample_count_by_role": {
                role: len(role_samples)
                for role, role_samples in by_role.items()
            },
            "probe_root": str(probe_root),
        }
    return {
        "ok": True,
        "metrics": metrics,
        "sample_count_by_role": {
            role: len(role_samples)
            for role, role_samples in by_role.items()
        },
        "probe_root": str(probe_root),
    }


def hf_probe_samples_from_local_dbs(
    replay_db_glob: str,
    *,
    max_samples: int,
) -> list[dict[str, Any]]:
    """Return complete World/Interest/Agent probe samples from local DBs."""

    from face_of_agi.memory import SQLiteDatabase

    db_paths = sorted(
        (Path(path) for path in glob.glob(replay_db_glob)),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    selected: dict[str, list[Any]] = {"world": [], "interest": [], "agent": []}
    for db_path in db_paths:
        database = SQLiteDatabase(db_path)
        with database.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM replay_samples
                WHERE held_out = 0
                ORDER BY id ASC
                """
            ).fetchall()
        samples = [database._row_to_replay_sample(row) for row in rows]
        by_role = {
            role: {
                sample.turn_id: sample
                for sample in samples
                if sample.role == role
            }
            for role in selected
        }
        turn_ids = set(by_role["world"])
        turn_ids &= set(by_role["interest"])
        turn_ids &= set(by_role["agent"])
        for turn_id in sorted(turn_ids):
            if len(selected["world"]) >= max_samples:
                break
            for role in selected:
                selected[role].append(by_role[role][turn_id])
        if len(selected["world"]) >= max_samples:
            break
    if selected["world"]:
        return [
            _replay_sample_to_jsonable(sample)
            for role in ("world", "interest", "agent")
            for sample in selected[role]
        ]
    return _synthetic_hf_probe_samples(max_samples=max_samples)


def _probe_samples_by_role(
    samples: Sequence[Any],
) -> dict[str, tuple[Any, ...]]:
    by_role = {
        role: tuple(sample for sample in samples if sample.role == role)
        for role in ("world", "interest", "agent")
    }
    missing = [role for role, role_samples in by_role.items() if not role_samples]
    if missing:
        raise RuntimeError(
            "HF probe samples are missing roles: " + ", ".join(missing)
        )
    count = min(len(role_samples) for role_samples in by_role.values())
    return {
        role: tuple(role_samples[:count])
        for role, role_samples in by_role.items()
    }


def _replay_sample_to_jsonable(sample: Any) -> dict[str, Any]:
    return {
        "id": sample.id,
        "game_id": sample.game_id,
        "run_id": sample.run_id,
        "turn_id": sample.turn_id,
        "role": sample.role,
        "prompt_json": sample.prompt_json,
        "completion_json": sample.completion_json,
        "reward": sample.reward,
        "held_out": sample.held_out,
        "metadata": sample.metadata,
        "created_at": sample.created_at,
    }


def _replay_sample_from_jsonable(value: Mapping[str, Any]) -> Any:
    from face_of_agi.contracts import ReplaySampleRecord

    return ReplaySampleRecord(
        id=int(value["id"]),
        game_id=str(value["game_id"]),
        run_id=str(value["run_id"]),
        turn_id=int(value["turn_id"]),
        role=value["role"],
        prompt_json=dict(value["prompt_json"]),
        completion_json=dict(value["completion_json"]),
        reward=float(value["reward"]),
        held_out=bool(value["held_out"]),
        metadata=dict(value["metadata"]),
        created_at=str(value["created_at"]),
    )


def _synthetic_hf_probe_samples(*, max_samples: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    roles = ("world", "interest", "agent")
    for turn_id in range(1, max(1, max_samples) + 1):
        for role in roles:
            samples.append(
                {
                    "id": len(samples) + 1,
                    "game_id": "hf-probe-synthetic",
                    "run_id": "hf-probe",
                    "turn_id": turn_id,
                    "role": role,
                    "prompt_json": {
                        "request": _synthetic_probe_request(role=role),
                    },
                    "completion_json": {
                        "target": _synthetic_probe_target(role=role),
                    },
                    "reward": 0.0,
                    "held_out": False,
                    "metadata": {},
                    "created_at": "synthetic",
                }
            )
    return samples


def _synthetic_probe_request(*, role: str) -> dict[str, Any]:
    return {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"synthetic {role} probe",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _synthetic_png_data_url()},
                    },
                ],
            },
        ],
        "max_completion_tokens": 64,
        "temperature": 0.0,
    }


def _synthetic_probe_target(*, role: str) -> dict[str, Any]:
    if role == "world":
        return {"predicted_change": "nothing visible changed"}
    if role == "interest":
        return {
            "candidate_values": [
                {
                    "candidate_index": 0,
                    "expected_learning_progress": 0.0,
                    "expected_goal_delta": 0.0,
                    "confidence": 0.5,
                    "notes": "synthetic",
                }
            ]
        }
    return {"action": {"action_id": "ACTION1"}}


def _synthetic_png_data_url() -> str:
    from PIL import Image

    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    import base64
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _cuda_metric(phase: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"phase": phase, "cuda_available": False}
    if not torch.cuda.is_available():
        return {"phase": phase, "cuda_available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "phase": phase,
        "cuda_available": True,
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
        "free_gib": free_bytes / (1024**3),
        "total_gib": total_bytes / (1024**3),
    }


def modal_runtime_config_text(
    config_text: str,
    *,
    adapter_root: str | Path | None = None,
) -> str:
    """Return config text with Modal-local public game asset paths resolved."""

    raw = _yaml_mapping(config_text)
    changed = False

    if _uses_default_public_games(raw):
        prepare_public_games_on_run_volume()
        raw["game_catalog_path"] = str(PUBLIC_GAME_CATALOG_PATH)
        raw["environments_dir"] = str(PUBLIC_ENVIRONMENTS_DIR)
        changed = True

    if adapter_root is not None and _modal_online_lora_enabled(raw):
        online_lora = raw.setdefault("online_lora", {})
        if not isinstance(online_lora, dict):
            raise ValueError("online_lora config must be a mapping")
        online_lora["adapter_root"] = str(adapter_root)
        changed = True

    return yaml.safe_dump(raw, sort_keys=False) if changed else config_text


def _modal_lora_adapter_root(database_name: str) -> Path:
    stem = _safe_modal_path_component(Path(database_name).stem or "memory")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = uuid.uuid4().hex[:8]
    return RUN_VOLUME_PATH / "lora" / f"{stem}-{timestamp}-{suffix}"


def prepare_public_games_on_run_volume() -> None:
    """Materialize public ARC games on the Modal run volume."""

    if _public_games_ready():
        return

    PUBLIC_ENVIRONMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_GAME_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = _download_public_games(
        environments_dir=PUBLIC_ENVIRONMENTS_DIR,
        recordings_dir=PUBLIC_GAME_RECORDINGS_DIR,
    )
    PUBLIC_GAME_CATALOG_PATH.write_text(
        json.dumps(
            {str(index): game_id for index, game_id in enumerate(game_ids)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _yaml_mapping(config_text: str) -> dict[str, Any]:
    raw = yaml.safe_load(config_text) or {}
    if not isinstance(raw, dict):
        raise ValueError("Modal runtime config must be a YAML mapping")
    return raw


def _uses_default_public_games(raw: Mapping[str, Any]) -> bool:
    return (
        str(raw.get("game_catalog_path", DEFAULT_GAME_CATALOG_PATH))
        == DEFAULT_GAME_CATALOG_PATH
        and str(raw.get("environments_dir", DEFAULT_ENVIRONMENTS_DIR))
        == DEFAULT_ENVIRONMENTS_DIR
    )


def _modal_online_lora_enabled(raw: Mapping[str, Any]) -> bool:
    online_lora = raw.get("online_lora")
    if online_lora is None:
        return False
    if not isinstance(online_lora, dict):
        raise ValueError("online_lora config must be a mapping")
    return bool(online_lora.get("enabled", True))


def _safe_modal_path_component(value: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "memory"


def _public_games_ready() -> bool:
    return PUBLIC_GAME_CATALOG_PATH.exists() and any(
        PUBLIC_ENVIRONMENTS_DIR.rglob("metadata.json")
    )


def _download_public_games(
    *,
    environments_dir: Path,
    recordings_dir: Path,
) -> tuple[str, ...]:
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(environments_dir),
        recordings_dir=str(recordings_dir),
    )
    game_ids = tuple(
        sorted({str(game.game_id).strip() for game in arcade.get_environments()})
    )
    if not game_ids or any(not game_id for game_id in game_ids):
        raise RuntimeError("ARC returned no public game ids to prepare on Modal")

    for game_id in game_ids:
        environment = arcade.make(game_id, seed=0, save_recording=False)
        if environment is None:
            raise RuntimeError(f"unable to prepare public ARC game '{game_id}'")
    return game_ids


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


def probe_vllm_chat_endpoint(config: VLLMServerConfig) -> list[VLLMProbeResult]:
    """Send minimal raw probes that isolate vLLM request feature failures."""

    return [
        _post_vllm_chat_probe(config, name=name, payload=payload)
        for name, payload in vllm_chat_probe_payloads(config.model)
    ]


def vllm_chat_probe_payloads(model: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return raw chat payloads from least to most like real role calls."""

    schema = {
        "type": "object",
        "properties": {
            "document": {
                "type": "string",
            },
        },
        "required": ["document"],
        "additionalProperties": False,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "probe_document",
            "strict": True,
            "schema": schema,
        },
    }
    image_content = [
        {"type": "text", "text": "Describe the attached image in one short sentence."},
        {
            "type": "image_url",
            "image_url": {
                "url": _probe_image_data_url(),
                "detail": "auto",
            },
        },
    ]
    common: dict[str, Any] = {
        "model": model,
        "stream": False,
        "max_tokens": 24,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    text_messages = [
        {"role": "system", "content": "Answer tersely."},
        {"role": "user", "content": "Return the word OK."},
    ]
    json_messages = [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": "Return a document field with value OK."},
    ]
    return (
        ("text", {**common, "messages": text_messages}),
        (
            "text_json_schema",
            {**common, "messages": json_messages, "response_format": response_format},
        ),
        ("image", {**common, "messages": [{"role": "user", "content": image_content}]}),
        (
            "image_json_schema",
            {
                **common,
                "messages": [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": image_content},
                ],
                "response_format": response_format,
            },
        ),
    )


def _post_vllm_chat_probe(
    config: VLLMServerConfig,
    *,
    name: str,
    payload: dict[str, Any],
) -> VLLMProbeResult:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{config.base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return VLLMProbeResult(
                name=name,
                status=response.status,
                ok=200 <= response.status < 300,
                body=response_body[:4000],
            )
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return VLLMProbeResult(
            name=name,
            status=exc.code,
            ok=False,
            body=response_body[:4000],
            error=str(exc),
        )
    except (TimeoutError, URLError) as exc:
        return VLLMProbeResult(
            name=name,
            status=None,
            ok=False,
            body="",
            error=str(exc),
        )


def _probe_image_data_url() -> str:
    """Return a simple PNG data URL large enough for the Qwen vision processor."""

    import base64
    from io import BytesIO

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    for offset in range(0, 512, 64):
        draw.line((offset, 0, offset, 511), fill="black")
        draw.line((0, offset, 511, offset), fill="black")
    draw.rectangle((192, 192, 320, 320), fill="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
