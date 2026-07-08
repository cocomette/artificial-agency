"""Tests for the single-VLM LoRA Modal runner helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "experiments" / "single_vlm_lora" / "modal_a100_runner.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "single_vlm_lora_modal_a100_runner",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_runner_imports_without_modal_installed() -> None:
    module = _load_runner_module()

    assert module.APP_NAME == "single-vlm-lora-a100-40gb"
    assert module.GPU_TYPE == "A100-40GB"


def test_modal_runner_gpu_type_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("SINGLE_VLM_MODAL_GPU", "A100-80GB")

    module = _load_runner_module()

    assert module.GPU_TYPE == "A100-80GB"


def test_modal_runner_env_places_hf_cache_on_model_volume() -> None:
    module = _load_runner_module()

    env = module._modal_env()

    assert env["HF_HOME"] == "/vol/models/huggingface"
    assert env["HF_HUB_CACHE"] == "/vol/models/huggingface/hub"
    assert env["TORCH_HOME"] == "/vol/models/torch"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "/root/repo/src" in env["PYTHONPATH"]
    assert "/root/repo/experiments/single_vlm_lora" in env["PYTHONPATH"]


def test_modal_runner_remote_output_dir_resolution() -> None:
    module = _load_runner_module()

    default_dir = module.remote_output_dir(
        config_name="qwen3_vl_4b_a100_40gb.yaml",
        timestamp="20260519-120000",
    )
    relative_dir = module.remote_output_dir(
        config_name="ignored.yaml",
        output_dir="single_vlm_lora/custom",
    )
    absolute_dir = module.remote_output_dir(
        config_name="ignored.yaml",
        output_dir="/vol/runs/absolute/custom",
    )

    assert str(default_dir) == (
        "/vol/runs/single_vlm_lora/qwen3_vl_4b_a100_40gb-20260519-120000"
    )
    assert str(relative_dir) == "/vol/runs/single_vlm_lora/custom"
    assert str(absolute_dir) == "/vol/runs/absolute/custom"


def test_modal_runner_builds_experiment_command() -> None:
    module = _load_runner_module()

    command = module.build_runner_command(
        python_executable="/usr/bin/python",
        config_path="/vol/runs/single_vlm_lora/configs/qwen.yaml",
        output_dir="/vol/runs/single_vlm_lora/out",
        max_turns=20,
        game_id="ls20-9607627b",
        game_index=3,
        seed=7,
        dry_run=True,
    )

    assert command == [
        "/usr/bin/python",
        "/root/repo/experiments/single_vlm_lora/run_single_vlm_lora.py",
        "--config",
        "/vol/runs/single_vlm_lora/configs/qwen.yaml",
        "--output-dir",
        "/vol/runs/single_vlm_lora/out",
        "--max-turns",
        "20",
        "--game-id",
        "ls20-9607627b",
        "--game-index",
        "3",
        "--seed",
        "7",
        "--dry-run",
    ]
