"""Tests for Kaggle notebook generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from face_of_agi.environment import load_environment_config
from face_of_agi.runtime.vllm_server import vllm_server_config_from_config_text


def _load_script(relative_path: str):
    path = Path(relative_path)
    spec = importlib.util.spec_from_file_location("kaggle_build_notebook", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configured_kaggle_owner() -> str:
    return _load_script("kaggle/scripts/kaggle_env.py").kaggle_owner()


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


def test_kaggle_notebook_uses_rtx6000_and_offline_inputs() -> None:
    builder = _load_script("kaggle/scripts/build_notebook.py")

    notebook = builder.build(source_archive_b64="dGVzdA==")

    kaggle_meta = notebook["metadata"]["kaggle"]
    assert kaggle_meta["accelerator"] == "NvidiaRtxPro6000"
    assert kaggle_meta["isGpuEnabled"] is True
    assert kaggle_meta["isInternetEnabled"] is False

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "kaggle_dataset_input('face-of-agi-wheelhouse-new')" in sources
    assert "kaggle_dataset_input('face-of-agi-qwen36-35b-fp8-weights')" in sources
    assert "/kaggle/input/datasets" in sources
    assert "runtime_config.yaml" in sources
    assert "arc_agi_3_wheels" in sources
    assert "no_deps=True" in sources
    assert "compressed-tensors==0.15.0.1" in sources
    assert "flashinfer-python==0.6.6" in sources
    assert "quack-kernels==0.4.1" in sources
    assert "torch-c-dlpack-ext" in sources
    assert "vllm==0.19.1" in sources
    assert "xgrammar==0.2.1" in sources
    assert "vllm_server_command" in sources
    assert '"Authorization": f"Bearer {api_key}"' in sources
    assert 'os.environ["PYTHONPATH"]' in sources
    assert "source_dir + os.pathsep + existing_pythonpath" in sources
    assert "face_of_agi.runtime.kaggle" in sources
    assert "KAGGLE_IS_COMPETITION_RERUN" in sources
    assert "clean_deadline_epoch_seconds" in sources
    assert "+ 32400" in sources
    assert "- 600" in sources
    assert "--deadline-epoch-seconds" in sources
    assert '"OPERATION_MODE": "competition"' in sources
    assert "submission.parquet" in sources


def test_kaggle_debug_notebook_uses_shell_and_public_game_input() -> None:
    builder = _load_script("kaggle/scripts/build_debug_notebook.py")

    notebook = builder.build(source_archive_b64="dGVzdA==")

    kaggle_meta = notebook["metadata"]["kaggle"]
    assert kaggle_meta["accelerator"] == "NvidiaRtxPro6000"
    assert kaggle_meta["isGpuEnabled"] is True
    assert kaggle_meta["isInternetEnabled"] is False

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "kaggle_dataset_input('face-of-agi-wheelhouse-new')" in sources
    assert "kaggle_dataset_input('face-of-agi-public-games')" in sources
    assert "kaggle_dataset_input('face-of-agi-qwen36-35b-fp8-weights')" in sources
    assert "/kaggle/input/datasets" in sources
    assert "environment_files.zip" in sources
    assert "zipfile.ZipFile" in sources
    assert "/kaggle/working/public-games" in sources
    assert "runtime_config.yaml" in sources
    assert "no_deps=True" in sources
    assert "compressed-tensors==0.15.0.1" in sources
    assert "flashinfer-python==0.6.6" in sources
    assert "quack-kernels==0.4.1" in sources
    assert "torch-c-dlpack-ext" in sources
    assert "vllm==0.19.1" in sources
    assert "xgrammar==0.2.1" in sources
    assert "/kaggle/working/runs" in sources
    assert '"Authorization": f"Bearer {api_key}"' in sources
    assert 'os.environ["PYTHONPATH"]' in sources
    assert "source_dir + os.pathsep + existing_pythonpath" in sources
    assert "face_of_agi.runtime.shell" in sources
    assert "face_of_agi.runtime.kaggle" not in sources
    assert "KAGGLE_IS_COMPETITION_RERUN" not in sources
    assert "submission.parquet" not in sources


def test_kaggle_dataset_helper_resolves_current_and_versioned_input_layouts() -> None:
    builders = [
        _load_script("kaggle/scripts/build_notebook.py"),
        _load_script("kaggle/scripts/build_debug_notebook.py"),
    ]

    for builder in builders:
        sources = "\n".join(cell["source"] for cell in builder.build("dGVzdA==")["cells"])
        assert "datasets_root / slug" in sources
        assert 'datasets_root.glob(f"*/{slug}")' in sources
        assert 'datasets_root.glob(f"*/{slug}/versions/*")' in sources
        assert "datasets_root.rglob(slug)" in sources


def test_kaggle_model_bootstrap_notebook_imports_hf_to_private_model() -> None:
    builder = _load_script("kaggle/scripts/build_model_bootstrap_notebook.py")

    notebook = builder.build()

    kaggle_meta = notebook["metadata"]["kaggle"]
    assert kaggle_meta["isGpuEnabled"] is False
    assert kaggle_meta["isInternetEnabled"] is True

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "huggingface_hub" in sources
    assert "kaggle>=2.2.0" in sources
    assert "HfApi" in sources
    assert "snapshot_download" in sources
    assert "Qwen/Qwen3.6-35B-A3B-FP8" in sources
    assert '"isPrivate": true' in sources
    assert "KAGGLE_API_TOKEN" in sources
    assert "KAGGLE_TOKEN" in sources
    assert "kaggle_api_token" in sources
    assert "HF_TOKEN" in sources
    assert "EMBEDDED_KAGGLE_API_TOKEN = ''" in sources
    assert "FACE_OF_AGI_KAGGLE_TOKEN_FILE path" in sources
    assert "Add-ons" in sources
    assert "> Secrets" in sources
    assert "Secret lookup details" in sources
    assert "/kaggle/temp" in sources
    assert "HF_HOME" in sources
    assert "HF snapshot size" in sources
    assert "HF download progress" in sources
    assert "HF download complete" in sources
    assert "Not enough scratch disk" in sources
    assert "kaggle-model-parent" in sources
    assert "kaggle-model-default" in sources
    assert "kaggle\", \"models\", \"create" in sources
    assert "Running Kaggle CLI" in sources
    assert "Kaggle CLI command failed with exit code" in sources
    assert '"variations"' in sources
    assert "face-of-agi-qwen36-35b-fp8/pytorch/default" in sources


def test_kaggle_model_bootstrap_can_embed_private_token() -> None:
    builder = _load_script("kaggle/scripts/build_model_bootstrap_notebook.py")

    notebook = builder.build(embedded_kaggle_token="KGAT_fake_token_for_test")

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "EMBEDDED_KAGGLE_API_TOKEN = 'KGAT_fake_token_for_test'" in sources
    assert "Using embedded Kaggle API token" in sources


def test_modal_model_upload_resolves_hf_cache_snapshot(tmp_path) -> None:
    uploader = _load_script("kaggle/scripts/upload_modal_model_to_kaggle.py")
    hf_home = tmp_path / "huggingface"
    repo_dir = (
        hf_home
        / "hub"
        / "models--Qwen--Qwen3.6-35B-A3B-FP8"
    )
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir()
    (refs_dir / "main").write_text("abc123\n", encoding="utf-8")

    assert uploader._hf_cache_repo_dir(
        hf_home,
        "Qwen/Qwen3.6-35B-A3B-FP8",
    ) == repo_dir
    assert uploader._find_hf_snapshot(
        hf_home=hf_home,
        repo_id="Qwen/Qwen3.6-35B-A3B-FP8",
        revision="main",
    ) == snapshot


def test_modal_model_upload_builds_kaggle_refs() -> None:
    uploader = _load_script("kaggle/scripts/upload_modal_model_to_kaggle.py")
    metadata = {
        "ownerSlug": "kaggle-owner",
        "modelSlug": "face-of-agi-qwen36-35b-fp8",
        "framework": "pytorch",
        "instanceSlug": "default",
    }

    assert uploader._parent_model_ref(metadata) == (
        "kaggle-owner/face-of-agi-qwen36-35b-fp8"
    )
    assert uploader._model_instance_ref(metadata) == (
        "kaggle-owner/face-of-agi-qwen36-35b-fp8/pytorch/default"
    )


def test_modal_model_dataset_metadata_builds_kaggle_ref() -> None:
    uploader = _load_script("kaggle/scripts/upload_modal_model_to_kaggle.py")
    metadata = json.loads(
        uploader._metadata_text(
            Path("kaggle/upload/model-dataset/dataset-metadata.json"),
            artifact_kind="dataset",
        )
    )
    owner = _configured_kaggle_owner()

    assert uploader._dataset_ref(metadata) == (
        f"{owner}/face-of-agi-qwen36-35b-fp8-weights"
    )


def test_kaggle_kernel_metadata_declares_competition_and_offline_sources() -> None:
    metadata = json.loads(Path("kaggle/notebooks/kernel-metadata.json").read_text())

    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert _kaggle_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert [source.split("/", 1)[1] for source in metadata["dataset_sources"]] == [
        "face-of-agi-wheelhouse-new",
        "face-of-agi-qwen36-35b-fp8-weights",
    ]
    assert metadata["model_sources"] == []


def test_kaggle_debug_kernel_metadata_declares_offline_sources() -> None:
    metadata = json.loads(
        Path("kaggle/debug-notebooks/kernel-metadata.json").read_text()
    )

    assert metadata["code_file"] == "debug.ipynb"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert _kaggle_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert [source.split("/", 1)[1] for source in metadata["dataset_sources"]] == [
        "face-of-agi-wheelhouse-new",
        "face-of-agi-public-games",
        "face-of-agi-qwen36-35b-fp8-weights",
    ]
    assert metadata["model_sources"] == []


def test_kaggle_debug_kernel_identity_uses_branch_and_commit() -> None:
    builder = _load_script("kaggle/scripts/build_debug_notebook.py")

    kernel_id, title = builder._debug_kernel_identity(
        "kaggle-owner",
        "feat/2-model-setup",
        "5109ce7b",
    )

    assert kernel_id == "kaggle-owner/foa-feat-2-model-setup-5109ce7b"
    assert title == "FOA feat-2-model-setup 5109ce7b"
    assert _kaggle_slug(title) == kernel_id.split("/", 1)[1]


def test_kaggle_debug_metadata_sync_generates_kernel_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_script("kaggle/scripts/build_debug_notebook.py")
    metadata_path = tmp_path / "kernel-metadata.json"
    metadata = json.loads(
        Path("kaggle/debug-notebooks/kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    metadata["id"] = "kaggle-owner/old-debug-kernel"
    metadata["title"] = "Old Debug Kernel"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(builder, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(builder, "kaggle_owner", lambda: "kaggle-owner")
    monkeypatch.setattr(
        builder,
        "_current_debug_kernel_identity",
        lambda owner: builder._debug_kernel_identity(
            owner,
            "feat/2-model-setup",
            "5109ce7b",
        ),
    )

    synced = builder._sync_metadata()

    assert synced["id"] == "kaggle-owner/foa-feat-2-model-setup-5109ce7b"
    assert synced["title"] == "FOA feat-2-model-setup 5109ce7b"
    assert synced["enable_gpu"] is True
    assert synced["enable_internet"] is False
    assert [source.split("/", 1)[1] for source in synced["dataset_sources"]] == [
        "face-of-agi-wheelhouse-new",
        "face-of-agi-public-games",
        "face-of-agi-qwen36-35b-fp8-weights",
    ]


def test_kaggle_submit_targets_request_rtx_pro6000_accelerator() -> None:
    makefile = Path("kaggle/Makefile").read_text(encoding="utf-8")

    assert "-include .env" in makefile
    assert "MODEL_REF ?= $(FACE_OF_AGI_KAGGLE_OWNER)/" in makefile
    assert "KAGGLE_ACCELERATOR ?= NvidiaRtxPro6000" in makefile
    assert "debug-submit: debug-notebook _check-kaggle" in makefile
    assert (
        "$(KAGGLE) kernels push -p notebooks/ $(KAGGLE_ACCELERATOR_ARGS)"
        in makefile
    )
    assert (
        "$(KAGGLE) kernels push -p debug-notebooks/ $(KAGGLE_ACCELERATOR_ARGS)"
        in makefile
    )
    assert (
        "$(KAGGLE) datasets create -p $(PUBLIC_GAMES_DIR) --dir-mode zip"
        in makefile
    )
    assert (
        "$(KAGGLE) datasets version -p $(PUBLIC_GAMES_DIR) "
        '-m "$(VERSION_NOTES)" --delete-old-versions --dir-mode zip'
        in makefile
    )


def test_kaggle_model_bootstrap_metadata_is_private_internet_kernel() -> None:
    metadata = json.loads(
        Path("kaggle/model-bootstrap/kernel-metadata.json").read_text()
    )

    assert metadata["code_file"] == "model_bootstrap.ipynb"
    assert _kaggle_slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == []
    assert metadata["model_sources"] == []


def test_public_games_dataset_metadata_uses_single_other_license() -> None:
    metadata = json.loads(
        Path("kaggle/upload/public-games/dataset-metadata.json").read_text()
    )

    assert metadata["id"].endswith("/face-of-agi-public-games")
    assert metadata["licenses"] == [{"name": "other"}]


def test_kaggle_rtx6000_debug_config_loads_public_debug_runtime() -> None:
    config_path = Path(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_debug.yaml"
    )

    config = load_environment_config(config_path)
    vllm_config = vllm_server_config_from_config_text(
        config_path.read_text(encoding="utf-8")
    )

    assert config.game_ids or config.game_selection == "all_available"
    assert config.max_parallel_games is not None
    assert config.max_parallel_games > 0
    assert config.max_actions_per_level > 0
    assert config.debug_keep_all_m_states is True
    assert config.debug_trace == "off"
    assert config.live_turn_monitor is True
    assert config.environments_dir == (
        "/kaggle/input/face-of-agi-public-games/environment_files"
    )
    assert vllm_config is not None
    assert vllm_config.model_path == (
        "/kaggle/input/face-of-agi-qwen36-35b-fp8-weights"
    )
    assert "--disable-uvicorn-access-log" in vllm_config.extra_args
    assert _extra_arg_value(vllm_config.extra_args, "--reasoning-config") == "{}"


def test_kaggle_rtx6000_submission_config_sets_reasoning_config() -> None:
    config_path = Path(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_parallel.yaml"
    )

    vllm_config = vllm_server_config_from_config_text(
        config_path.read_text(encoding="utf-8")
    )

    assert vllm_config is not None
    assert _extra_arg_value(vllm_config.extra_args, "--reasoning-config") == "{}"


def test_public_games_prepare_sorts_all_public_game_ids() -> None:
    prepare = _load_script("kaggle/scripts/prepare_public_games.py")

    assert prepare._public_game_ids(
        (
            SimpleNamespace(game_id="vc33-5430563c"),
            SimpleNamespace(game_id="ls20-9607627b"),
        )
    ) == (
        "ls20-9607627b",
        "vc33-5430563c",
    )


def test_public_games_prepare_requires_available_games() -> None:
    prepare = _load_script("kaggle/scripts/prepare_public_games.py")

    with pytest.raises(RuntimeError, match="no public game ids"):
        prepare._public_game_ids(())


def test_public_games_prepare_builds_dataset_shape(monkeypatch, tmp_path) -> None:
    prepare = _load_script("kaggle/scripts/prepare_public_games.py")
    metadata_path = tmp_path / "dataset-metadata.json"
    metadata_path.write_text('{"id": "owner/face-of-agi-public-games"}\n')
    captured: dict[str, object] = {}

    def fake_download_all_public_games(*, environments_dir, recordings_dir):
        game_ids = ("ls20-9607627b", "vc33-5430563c")
        captured["environments_dir"] = environments_dir
        captured["recordings_dir"] = recordings_dir
        for game_id in game_ids:
            prefix, suffix = game_id.split("-", 1)
            game_dir = environments_dir / prefix / suffix
            game_dir.mkdir(parents=True)
            (game_dir / "metadata.json").write_text("{}", encoding="utf-8")
        return game_ids

    monkeypatch.setattr(
        prepare,
        "_download_all_public_games",
        fake_download_all_public_games,
    )

    output_dir = prepare.prepare_public_games(
        output_dir=tmp_path / "public-games",
        metadata_path=metadata_path,
    )

    assert captured["environments_dir"] == output_dir / "environment_files"
    assert captured["recordings_dir"] == output_dir / "recordings"
    assert json.loads((output_dir / "dataset-metadata.json").read_text()) == {
        "id": f"{_configured_kaggle_owner()}/face-of-agi-public-games"
    }
    assert json.loads((output_dir / "local_games.json").read_text()) == {
        "0": "ls20-9607627b",
        "1": "vc33-5430563c",
    }
    assert (
        output_dir
        / "environment_files"
        / "ls20"
        / "9607627b"
        / "metadata.json"
    ).exists()


def _kaggle_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _extra_arg_value(extra_args: tuple[str, ...], key: str) -> str:
    index = extra_args.index(key)
    return extra_args[index + 1]
