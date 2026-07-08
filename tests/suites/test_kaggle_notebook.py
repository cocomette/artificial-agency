"""Tests for Kaggle notebook generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(relative_path: str):
    path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_env_file_declares_owner_and_token_file() -> None:
    helper = _load_script("kaggle/scripts/kaggle_env.py")
    env_values = {
        key.strip(): value.strip()
        for key, value in (
            line.split("=", 1)
            for line in Path("kaggle/.env").read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.strip().startswith("#")
        )
    }

    assert helper.KAGGLE_OWNER_ENV in env_values
    assert helper.KAGGLE_TOKEN_FILE_ENV in env_values
    assert helper.kaggle_owner() == env_values[helper.KAGGLE_OWNER_ENV]
    assert helper.kaggle_token_path() == (
        Path.cwd() / env_values[helper.KAGGLE_TOKEN_FILE_ENV]
    )


def test_submission_notebook_uses_transformers_runtime_without_vllm() -> None:
    builder = _load_script("kaggle/scripts/build_notebook.py")

    notebook = builder.build(source_archive_b64="dGVzdA==")

    kaggle_meta = notebook["metadata"]["kaggle"]
    assert kaggle_meta["accelerator"] == "NvidiaRtxPro6000"
    assert kaggle_meta["isGpuEnabled"] is True
    assert kaggle_meta["isInternetEnabled"] is False

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "kaggle_dataset_input('face-of-agi-transformers-wheelhouse')" in sources
    assert "kaggle_dataset_input('face-of-agi-qwen36-35b-fp8-weights')" in sources
    assert "yaml.safe_load" in sources
    assert '"model_path"' in sources
    assert "src/face_of_agi/runtime/configs/kaggle_transformers.yaml" in sources
    assert "face_of_agi.runtime.kaggle" in sources
    assert "transformers" in sources
    assert "safetensors" in sources
    assert "KAGGLE_IS_COMPETITION_RERUN" in sources
    assert "--deadline-epoch-seconds" in sources
    assert "submission.parquet" in sources
    assert "vllm" not in sources.lower()
    assert "VLLM_" not in sources


def test_debug_notebook_uses_same_transformers_shell_path() -> None:
    builder = _load_script("kaggle/scripts/build_debug_notebook.py")

    notebook = builder.build(source_archive_b64="dGVzdA==")

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "src/face_of_agi/runtime/configs/kaggle_debug_transformers.yaml" in sources
    assert "kaggle_dataset_input('face-of-agi-public-games')" in sources
    assert "kaggle_dataset_input('face-of-agi-qwen36-35b-fp8-weights')" in sources
    assert "yaml.safe_load" in sources
    assert '"model_path"' in sources
    assert "face_of_agi.runtime.shell" in sources
    assert "--debug-keep-all-m-states" in sources
    assert "transformers" in sources
    assert "vllm" not in sources.lower()


def test_kaggle_makefile_defaults_target_transformers_configs() -> None:
    makefile = Path("kaggle/Makefile").read_text(encoding="utf-8")

    assert "KAGGLE_SUBMISSION_CONFIG ?= src/face_of_agi/runtime/configs/kaggle_transformers.yaml" in makefile
    assert "KAGGLE_DEBUG_CONFIG ?= src/face_of_agi/runtime/configs/kaggle_debug_transformers.yaml" in makefile
    assert (
        "KAGGLE_SUBMISSION_MODEL_DATASET_SLUG ?= "
        "face-of-agi-qwen36-35b-fp8-weights"
    ) in makefile
    assert "configs/vllm" not in makefile


def test_kaggle_requirements_exclude_prompt_role_backends() -> None:
    requirements = Path("kaggle/requirements-kaggle.txt").read_text(encoding="utf-8")

    assert "transformers" in requirements
    assert "safetensors" in requirements
    assert "vllm" not in requirements.lower()
    assert "openai" not in requirements.lower()
