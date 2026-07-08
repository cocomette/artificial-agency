"""Tests for Modal runtime helper behavior that does not call Modal."""

from __future__ import annotations

from io import StringIO
import sys

from face_of_agi.runtime.modal_app import (
    DEFAULT_MODAL_GPU,
    MODAL_GPU,
    OLLAMA_INSTALL_COMMAND,
    OLLAMA_VERSION,
    _modal_env,
    _modal_gpu_from_launch_context,
    modal_run_database_path,
    modal_run_folder_path,
    ollama_models_from_config_text,
    modal_gpu_from_config_text,
    run_streamed_subprocess,
    vllm_server_command,
    vllm_server_config_from_config_text,
)
from face_of_agi.runtime.vllm_server import (
    vllm_server_command as shared_vllm_server_command,
    vllm_server_config_from_config_text as shared_vllm_server_config_from_config_text,
)


def test_modal_gpu_default_requests_h100() -> None:
    assert MODAL_GPU == "H100"


def test_modal_gpu_helper_reads_configured_gpu() -> None:
    config_text = """
modal:
  gpu: RTX-PRO-6000
game_index: 0
max_actions_per_level: 1
models: {}
"""

    assert modal_gpu_from_config_text(config_text) == "RTX-PRO-6000"


def test_modal_gpu_launch_context_reads_config_arg(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "modal:\n"
        "  gpu: RTX-PRO-6000\n"
        "game_index: 0\n"
        "max_actions_per_level: 1\n"
        "models: {}\n",
        encoding="utf-8",
    )

    assert _modal_gpu_from_launch_context(
        default=DEFAULT_MODAL_GPU,
        argv=["modal", "run", "modal_app.py::main", "--config", str(path)],
    ) == "RTX-PRO-6000"


def test_modal_run_paths_are_grouped_by_run_folder() -> None:
    assert str(modal_run_folder_path("abc123def456")) == "/vol/runs/abc123def456"
    assert str(modal_run_database_path("abc123def456", "memory.sqlite")) == (
        "/vol/runs/abc123def456/memory.sqlite"
    )


def test_modal_image_pins_ollama_version() -> None:
    assert OLLAMA_VERSION == "0.24.0"
    assert f"OLLAMA_VERSION={OLLAMA_VERSION} sh" in OLLAMA_INSTALL_COMMAND
    assert "ollama --version" in OLLAMA_INSTALL_COMMAND


def test_modal_helper_collects_ollama_models_from_shared_vlm_config() -> None:
    config_text = """
game_index: 0
max_actions_per_level: 1
models:
  shared_vlm:
    backend: ollama
    model: qwen3.6
  agent:
    backend: ollama
  change:
    backend: ollama
  compacter:
    backend: ollama
  updater:
    agent:
      backend: ollama
"""

    assert ollama_models_from_config_text(config_text) == ("qwen3.6",)


def test_modal_helper_collects_role_specific_ollama_models() -> None:
    config_text = """
game_index: 0
max_actions_per_level: 1
models:
  agent:
    backend: ollama
    model: qwen-agent
  change:
    backend: ollama
    model: qwen-change
  compacter:
    backend: ollama
    model: qwen-compacter
  updater:
    agent:
      backend: ollama
      model: qwen-agent-updater
"""

    assert ollama_models_from_config_text(config_text) == (
        "qwen-agent",
        "qwen-change",
        "qwen-compacter",
        "qwen-agent-updater",
    )


def test_modal_helper_extracts_vllm_server_config_from_shared_vlm() -> None:
    config_text = """
game_index: 0
max_actions_per_level: 1
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
    server:
      host: 127.0.0.1
      port: 8000
      max_model_len: 262144
      reasoning_parser: qwen3
      extra_args:
        - --disable-log-stats
  agent:
    backend: vllm
  change:
    backend: vllm
  compacter:
    backend: vllm
  updater:
    agent:
      backend: vllm
"""

    server_config = vllm_server_config_from_config_text(config_text)

    assert server_config is not None
    assert server_config.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert server_config.base_url == "http://127.0.0.1:8000/v1"
    assert vllm_server_command(server_config) == (
        "vllm",
        "serve",
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "262144",
        "--reasoning-parser",
        "qwen3",
        "--disable-log-stats",
    )


def test_modal_helper_extracts_vllm_server_config_with_game_indices() -> None:
    config_text = """
game_indices: [0, 1]
max_parallel_games: 2
max_actions_per_level: 1
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
    server:
      port: 8000
  agent:
    backend: vllm
  change:
    backend: vllm
  compacter:
    backend: vllm
  updater:
    agent:
      backend: vllm
"""

    server_config = vllm_server_config_from_config_text(config_text)

    assert server_config is not None
    assert server_config.model == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_vllm_server_command_uses_model_path_and_served_model_name() -> None:
    config_text = """
game_selection: all_available
max_actions_per_level: 1
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
    server:
      model_path: /kaggle/input/face-of-agi-qwen36-35b-fp8/pytorch/default/1
      port: 8000
  agent:
    backend: vllm
  change:
    backend: vllm
  compacter:
    backend: vllm
  updater:
    agent:
      backend: vllm
"""

    server_config = shared_vllm_server_config_from_config_text(config_text)

    assert server_config is not None
    assert server_config.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert server_config.model_path == (
        "/kaggle/input/face-of-agi-qwen36-35b-fp8/pytorch/default/1"
    )
    assert shared_vllm_server_command(server_config) == (
        "vllm",
        "serve",
        "/kaggle/input/face-of-agi-qwen36-35b-fp8/pytorch/default/1",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "Qwen/Qwen3.6-35B-A3B-FP8",
    )


def test_modal_streamed_subprocess_tees_and_captures_output() -> None:
    stdout = StringIO()
    stderr = StringIO()

    result = run_streamed_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('stdout line'); "
                "print('stderr line', file=sys.stderr)"
            ),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert result.returncode == 0
    assert result.stdout == "stdout line\n"
    assert result.stderr == "stderr line\n"
    assert stdout.getvalue() == result.stdout
    assert stderr.getvalue() == result.stderr
