"""Sync Kaggle metadata files from ``kaggle/.env``."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import (  # noqa: E402
    KAGGLE_ROOT,
    kaggle_dataset_sources,
    kaggle_owner,
    kaggle_ref,
    with_kaggle_dataset_id,
    with_kaggle_kernel_id,
    with_kaggle_owner_slug,
    write_json_if_changed,
)

WHEELHOUSE_DATASET_SLUG = "face-of-agi-wheelhouse"
MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG = (
    "face-of-agi-minicpm-v46-thinking-wheelhouse"
)
PUBLIC_GAMES_DATASET_SLUG = "face-of-agi-public-games"
MODEL_DATASET_SLUG = "face-of-agi-qwen36-35b-fp8-weights"
SUBMISSION_KERNEL_SLUG = "face-of-agi-arc-agi-3-rtx6000"
DEBUG_KERNEL_SLUG = "face-of-agi-arc-agi-3-rtx6000-debug"
MODEL_BOOTSTRAP_KERNEL_SLUG = "face-of-agi-qwen36-35b-fp8-bootstrap"

SUBMISSION_KERNEL_METADATA = {
    "id": f"local/{SUBMISSION_KERNEL_SLUG}",
    "title": "FACE-OF-AGI ARC-AGI-3 RTX6000",
    "code_file": "submission.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": [],
}
DEBUG_KERNEL_METADATA = {
    "id": f"local/{DEBUG_KERNEL_SLUG}",
    "title": "FACE-OF-AGI ARC-AGI-3 RTX6000 Debug",
    "code_file": "debug.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": [],
}
MODEL_BOOTSTRAP_KERNEL_METADATA = {
    "id": f"local/{MODEL_BOOTSTRAP_KERNEL_SLUG}",
    "title": "FACE OF AGI Qwen36 35B FP8 Bootstrap",
    "code_file": "model_bootstrap.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": ["face-of-agi", "huggingface", "kaggle-model"],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
WHEELHOUSE_DATASET_METADATA = {
    "title": "FACE-OF-AGI Kaggle Wheelhouse",
    "id": f"local/{WHEELHOUSE_DATASET_SLUG}",
    "licenses": [{"name": "apache-2.0"}],
}
MINICPM_V46_THINKING_WHEELHOUSE_DATASET_METADATA = {
    "title": "FACE-OF-AGI MiniCPM-V 4.6 Thinking Wheelhouse",
    "id": f"local/{MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG}",
    "licenses": [{"name": "apache-2.0"}],
}
PUBLIC_GAMES_DATASET_METADATA = {
    "title": "FACE-OF-AGI ARC-AGI-3 Public Games",
    "id": f"local/{PUBLIC_GAMES_DATASET_SLUG}",
    "licenses": [{"name": "other"}],
}
MODEL_DATASET_METADATA = {
    "title": "FACE-OF-AGI Qwen3.6 35B FP8 Weights",
    "id": f"local/{MODEL_DATASET_SLUG}",
    "licenses": [{"name": "apache-2.0"}],
}
MODEL_METADATA = {
    "ownerSlug": "local",
    "title": "FACE-OF-AGI Qwen3.6 35B FP8",
    "slug": "face-of-agi-qwen36-35b-fp8",
    "subtitle": "Qwen3.6 35B FP8 weights for FACE-OF-AGI ARC-AGI-3 submissions",
    "isPrivate": True,
    "licenseName": "Apache 2.0",
    "description": (
        "Private Kaggle model artifact used by the FACE-OF-AGI ARC-AGI-3 "
        "submission notebook."
    ),
    "publishTime": "",
    "provenanceSources": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8",
}
MODEL_INSTANCE_METADATA = {
    "ownerSlug": "local",
    "modelSlug": "face-of-agi-qwen36-35b-fp8",
    "instanceSlug": "default",
    "framework": "pytorch",
    "overview": (
        "Qwen3.6 35B FP8 weights served by vLLM inside the ARC-AGI-3 "
        "Kaggle notebook."
    ),
    "usage": (
        "The generated submission notebook serves this variation from "
        "`/kaggle/input/face-of-agi-qwen36-35b-fp8/pytorch/default/1` "
        "with vLLM."
    ),
    "licenseName": "Apache 2.0",
    "fineTunable": False,
    "trainingData": [],
    "modelInstanceType": "Unspecified",
    "baseModelInstance": "",
    "externalBaseModelUrl": "",
}


def sync_kaggle_metadata() -> None:
    """Apply the configured Kaggle owner to all user-specific metadata."""

    owner = kaggle_owner()
    _sync_kernel_metadata(
        KAGGLE_ROOT / "notebooks/kernel-metadata.json",
        dataset_slugs=(WHEELHOUSE_DATASET_SLUG, MODEL_DATASET_SLUG),
        default=SUBMISSION_KERNEL_METADATA,
    )
    _sync_kernel_metadata(
        KAGGLE_ROOT / "debug-notebooks/kernel-metadata.template.json",
        dataset_slugs=(
            WHEELHOUSE_DATASET_SLUG,
            PUBLIC_GAMES_DATASET_SLUG,
            MODEL_DATASET_SLUG,
        ),
        default=DEBUG_KERNEL_METADATA,
    )
    _sync_kernel_metadata(
        KAGGLE_ROOT / "model-bootstrap/kernel-metadata.json",
        dataset_slugs=(),
        default=MODEL_BOOTSTRAP_KERNEL_METADATA,
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/wheelhouse/dataset-metadata.json",
        WHEELHOUSE_DATASET_SLUG,
        default=WHEELHOUSE_DATASET_METADATA,
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/wheelhouse-minicpm-v46-thinking/dataset-metadata.json",
        MINICPM_V46_THINKING_WHEELHOUSE_DATASET_SLUG,
        default=MINICPM_V46_THINKING_WHEELHOUSE_DATASET_METADATA,
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/public-games/dataset-metadata.json",
        PUBLIC_GAMES_DATASET_SLUG,
        default=PUBLIC_GAMES_DATASET_METADATA,
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/model-dataset/dataset-metadata.json",
        MODEL_DATASET_SLUG,
        default=MODEL_DATASET_METADATA,
    )
    _sync_owner_slug(
        KAGGLE_ROOT / "upload/model/model-metadata.json",
        default=MODEL_METADATA,
    )
    _sync_owner_slug(
        KAGGLE_ROOT / "upload/model/model-instance-metadata.json",
        default=MODEL_INSTANCE_METADATA,
    )
    print(f"[sync_kaggle_metadata] Synced Kaggle metadata for {owner}")


def _sync_kernel_metadata(
    path: Path,
    *,
    dataset_slugs: tuple[str, ...],
    default: dict[str, Any],
) -> None:
    metadata = with_kaggle_kernel_id(_read_json(path, default=default))
    metadata["dataset_sources"] = kaggle_dataset_sources(dataset_slugs)
    metadata["model_sources"] = []
    write_json_if_changed(path, metadata)


def _sync_dataset_metadata(
    path: Path,
    slug: str,
    *,
    default: dict[str, Any],
) -> None:
    metadata = with_kaggle_dataset_id(_read_json(path, default=default), slug)
    write_json_if_changed(path, metadata)


def _sync_owner_slug(path: Path, *, default: dict[str, Any]) -> None:
    metadata = with_kaggle_owner_slug(_read_json(path, default=default))
    if "modelSlug" in metadata:
        metadata["modelSlug"] = str(metadata["modelSlug"])
    if "slug" in metadata:
        metadata["slug"] = str(metadata["slug"])
    write_json_if_changed(path, metadata)


def _read_json(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sync_kaggle_metadata()
