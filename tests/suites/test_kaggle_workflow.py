"""Tests for local Kaggle workflow helpers."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_generated_kaggle_artifacts_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".kaggle/" in gitignore
    assert "kaggle/build/" in gitignore
    assert "kaggle/notebooks/submission.ipynb" in gitignore
    assert "kaggle/debug-notebooks/debug.ipynb" in gitignore
    assert "kaggle/debug-notebooks/kernel-metadata.json" in gitignore
    assert "kaggle/model-bootstrap/model_bootstrap.ipynb" in gitignore
    assert "kaggle/notebooks/kernel-metadata.json" in gitignore
    assert "kaggle/debug-notebooks/kernel-metadata.template.json" in gitignore
    assert "kaggle/model-bootstrap/kernel-metadata.json" in gitignore
    assert "kaggle/upload/model-dataset/dataset-metadata.json" in gitignore
    assert "kaggle/upload/model/model-instance-metadata.json" in gitignore
    assert "kaggle/upload/model/model-metadata.json" in gitignore
    assert "kaggle/upload/public-games/dataset-metadata.json" in gitignore
    assert (
        "kaggle/upload/wheelhouse-minicpm-v46-thinking/dataset-metadata.json"
        in gitignore
    )
    assert "kaggle/upload/wheelhouse/dataset-metadata.json" in gitignore


def test_sync_metadata_targets_debug_template() -> None:
    script = (ROOT / "kaggle/scripts/sync_kaggle_metadata.py").read_text(
        encoding="utf-8"
    )

    assert 'KAGGLE_ROOT / "debug-notebooks/kernel-metadata.template.json"' in script
    assert 'KAGGLE_ROOT / "debug-notebooks/kernel-metadata.json"' not in script


def test_sync_metadata_uses_defaults_for_missing_generated_files(tmp_path) -> None:
    syncer = _load_script("kaggle/scripts/sync_kaggle_metadata.py", "sync_metadata")
    default = {"id": "local/test", "nested": []}

    metadata = syncer._read_json(tmp_path / "missing.json", default=default)
    metadata["nested"].append("changed")

    assert metadata == {"id": "local/test", "nested": ["changed"]}
    assert default == {"id": "local/test", "nested": []}


def test_resolve_kaggle_kernel_ref_accepts_metadata_and_urls(tmp_path) -> None:
    metadata_path = tmp_path / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps({"id": "owner/from-metadata"}),
        encoding="utf-8",
    )
    script = ROOT / "kaggle/scripts/resolve_kaggle_kernel_ref.py"

    from_metadata = subprocess.run(
        [sys.executable, str(script), "", str(metadata_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    from_url = subprocess.run(
        [
            sys.executable,
            str(script),
            "https://www.kaggle.com/code/other-owner/debug-run",
            str(metadata_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert from_metadata.stdout.strip() == "owner/from-metadata"
    assert from_url.stdout.strip() == "other-owner/debug-run"


def test_kaggle_notebook_install_cells_allow_vllm_prerelease_dependencies() -> None:
    debug_builder = _load_script("kaggle/scripts/build_debug_notebook.py", "debug")
    submission_builder = _load_script("kaggle/scripts/build_notebook.py", "submission")

    for install_source in (
        debug_builder._install_cell()["source"],
        submission_builder._install_cell()["source"],
    ):
        assert (
            "def pip_install(packages, *, no_deps=False, pre=False):"
            in install_source
        )
        assert 'command.append("--pre")' in install_source
        assert "nvidia-cutlass-dsl>=4.4.2" in install_source
        assert install_source.count("pre=True") >= 2
    assert "pip_install(vllm_deps, pre=True)" in (
        debug_builder._install_cell()["source"]
    )
    assert "pip_install(vllm_dependency_requirements(), pre=True)" in (
        submission_builder._install_cell()["source"]
    )


def _load_script(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"test_{name}_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
