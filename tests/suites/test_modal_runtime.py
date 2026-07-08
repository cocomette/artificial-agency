"""Tests for Modal runtime helper behavior that does not call Modal."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import yaml

from face_of_agi.environment.config import load_environment_config
from face_of_agi.runtime.modal_app import (
    MODAL_BASE_IMAGE,
    MODAL_GPU,
    MODAL_HF_STACK_PACKAGES,
    MODAL_TORCH_STACK_PACKAGES,
    MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES,
    MODAL_VLLM_STACK_PACKAGES,
    MODAL_VLLM_VERSION,
    PUBLIC_ENVIRONMENTS_DIR,
    PUBLIC_GAME_CATALOG_PATH,
    _modal_env,
    modal_runtime_config_text,
    run_streamed_subprocess,
    vllm_chat_probe_payloads,
    vllm_server_command,
    vllm_server_config_from_config_text,
)
from face_of_agi.runtime.vllm_server import (
    vllm_server_command as shared_vllm_server_command,
    vllm_server_config_from_config_text as shared_vllm_server_config_from_config_text,
)


def test_modal_gpu_default_requests_h100() -> None:
    assert MODAL_GPU == "H100"


def test_modal_env_is_vllm_only() -> None:
    env = _modal_env()

    assert "OLLAMA_HOST" not in env
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env
    assert env["HF_HOME"] == "/vol/models/huggingface"


def test_modal_vllm_stack_pins_compatible_torch_build() -> None:
    assert MODAL_BASE_IMAGE == "nvidia/cuda:12.4.1-devel-ubuntu22.04"
    assert "torch==2.10.0" in MODAL_TORCH_STACK_PACKAGES
    assert "torchvision==0.25.0" in MODAL_TORCH_STACK_PACKAGES
    assert "torchaudio==2.10.0" in MODAL_TORCH_STACK_PACKAGES
    assert "huggingface-hub<2.0,>=1.5.0" in MODAL_HF_STACK_PACKAGES
    assert "transformers==5.12.1" in MODAL_HF_STACK_PACKAGES
    assert MODAL_VLLM_VERSION == "0.19.1"
    assert f"vllm=={MODAL_VLLM_VERSION}" in MODAL_VLLM_STACK_PACKAGES
    assert "flashinfer-python==0.6.6" in MODAL_VLLM_STACK_PACKAGES
    assert "gguf>=0.17.0" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES
    assert "flashinfer-cubin==0.6.6" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES
    assert "huggingface-hub<2.0,>=1.5.0" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES
    assert "tokenizers==0.22.2" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES
    assert "annotated-doc" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES
    assert "uvloop" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES
    assert "jmespath" in MODAL_VLLM_RUNTIME_DEPENDENCY_PACKAGES


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
  memory:
    backend: vllm
  world:
    backend: vllm
  goal:
    backend: vllm
  interest:
    backend: vllm
  reward_judge:
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
  memory:
    backend: vllm
  world:
    backend: vllm
  goal:
    backend: vllm
  interest:
    backend: vllm
  reward_judge:
    backend: vllm
"""

    server_config = vllm_server_config_from_config_text(config_text)

    assert server_config is not None
    assert server_config.model == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_modal_h100_debug_config_mirrors_kaggle_debug_shape() -> None:
    config_path = Path(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_h100_qwen36_35b_fp8_debug.yaml"
    )

    config = load_environment_config(config_path)
    server_config = shared_vllm_server_config_from_config_text(
        config_path.read_text(encoding="utf-8")
    )

    assert config.game_selection == "all_available"
    assert config.max_parallel_games == 25
    assert config.models.shared_vlm.options["input_image_size"] == "2048x2048"
    assert config.models.shared_vlm.options["input_image_crop_arc_grid_edges"] == 3
    assert not hasattr(config, "online_lora")
    assert config.models.world.options["max_completion_tokens"] == 3072
    assert server_config is not None
    assert server_config.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert server_config.max_model_len == 32768
    assert _extra_arg_value(server_config.extra_args, "--gpu-memory-utilization") == "0.90"
    assert _extra_arg_value(server_config.extra_args, "--max-num-seqs") == "64"
    assert (
        _extra_arg_value(server_config.extra_args, "--max-num-batched-tokens")
        == "65536"
    )
    assert "--gdn-prefill-backend" in server_config.extra_args
    assert "--chat-template-content-format" in server_config.extra_args
    assert "openai" in server_config.extra_args
    assert "--enable-prefix-caching" in server_config.extra_args
    assert shared_vllm_server_command(server_config)[:3] == (
        "vllm",
        "serve",
        "Qwen/Qwen3.6-35B-A3B-FP8",
    )


def test_vllm_probe_payloads_cover_text_schema_and_image_shapes() -> None:
    payloads = dict(vllm_chat_probe_payloads("model-id"))

    assert tuple(payloads) == (
        "text",
        "text_json_schema",
        "image",
        "image_json_schema",
    )
    assert payloads["text"]["model"] == "model-id"
    assert "response_format" not in payloads["text"]
    assert payloads["text_json_schema"]["response_format"]["type"] == "json_schema"
    image_content = payloads["image"]["messages"][0]["content"]
    assert [part["type"] for part in image_content] == ["text", "image_url"]
    assert image_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (
        payloads["image_json_schema"]["response_format"]["json_schema"]["name"]
        == "probe_document"
    )


def test_modal_runtime_config_rewrites_default_public_game_paths(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "face_of_agi.runtime.modal_app.prepare_public_games_on_run_volume",
        lambda: calls.append("prepared"),
    )
    config_text = """
game_selection: all_available
max_actions_per_level: 1
models:
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
  agent:
    backend: vllm
  change:
    backend: vllm
  memory:
    backend: vllm
  world:
    backend: vllm
  goal:
    backend: vllm
  interest:
    backend: vllm
  reward_judge:
    backend: vllm
"""

    rewritten = yaml.safe_load(modal_runtime_config_text(config_text))

    assert calls == ["prepared"]
    assert rewritten["game_catalog_path"] == str(PUBLIC_GAME_CATALOG_PATH)
    assert rewritten["environments_dir"] == str(PUBLIC_ENVIRONMENTS_DIR)


def test_modal_runtime_config_leaves_explicit_public_game_paths_unchanged(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        "face_of_agi.runtime.modal_app.prepare_public_games_on_run_volume",
        lambda: calls.append("prepared"),
    )
    config_text = """
game_catalog_path: /vol/runs/public-games/local_games.json
environments_dir: /vol/runs/public-games/environment_files
max_actions_per_level: 1
models:
  agent:
    backend: vllm
  change:
    backend: vllm
  memory:
    backend: vllm
  world:
    backend: vllm
  goal:
    backend: vllm
  interest:
    backend: vllm
  reward_judge:
    backend: vllm
"""

    rewritten = yaml.safe_load(modal_runtime_config_text(config_text))

    assert calls == []
    assert rewritten["game_catalog_path"] == str(PUBLIC_GAME_CATALOG_PATH)
    assert rewritten["environments_dir"] == str(PUBLIC_ENVIRONMENTS_DIR)


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
  memory:
    backend: vllm
  world:
    backend: vllm
  goal:
    backend: vllm
  interest:
    backend: vllm
  reward_judge:
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


def _extra_arg_value(extra_args: tuple[str, ...], key: str) -> str:
    index = extra_args.index(key)
    return extra_args[index + 1]
