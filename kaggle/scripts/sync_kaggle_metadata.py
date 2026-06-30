"""Sync Kaggle metadata files from ``kaggle/.env``."""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import (  # noqa: E402
    KAGGLE_ROOT,
    kaggle_dataset_sources,
    kaggle_owner,
    with_kaggle_dataset_id,
    with_kaggle_kernel_id,
    write_json_if_changed,
)

WHEELHOUSE_DATASET_SLUG = "face-of-agi-wheelhouse"
PUBLIC_GAMES_DATASET_SLUG = "face-of-agi-public-games"
MODEL_DATASET_SLUG = "face-of-agi-qwen36-35b-fp8-weights"


def sync_kaggle_metadata() -> None:
    """Apply the configured Kaggle owner to all user-specific metadata."""

    owner = kaggle_owner()
    _sync_kernel_metadata(
        KAGGLE_ROOT / "notebooks/kernel-metadata.json",
        dataset_slugs=(WHEELHOUSE_DATASET_SLUG, MODEL_DATASET_SLUG),
    )
    _sync_kernel_metadata(
        KAGGLE_ROOT / "debug-notebooks/kernel-metadata.template.json",
        dataset_slugs=(
            WHEELHOUSE_DATASET_SLUG,
            PUBLIC_GAMES_DATASET_SLUG,
            MODEL_DATASET_SLUG,
        ),
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/wheelhouse/dataset-metadata.json",
        WHEELHOUSE_DATASET_SLUG,
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/public-games/dataset-metadata.json",
        PUBLIC_GAMES_DATASET_SLUG,
    )
    _sync_dataset_metadata(
        KAGGLE_ROOT / "upload/model-dataset/dataset-metadata.json",
        MODEL_DATASET_SLUG,
    )
    print(f"[sync_kaggle_metadata] Synced Kaggle metadata for {owner}")


def _sync_kernel_metadata(path: Path, *, dataset_slugs: tuple[str, ...]) -> None:
    metadata = with_kaggle_kernel_id(_read_json(path))
    metadata["dataset_sources"] = kaggle_dataset_sources(dataset_slugs)
    metadata["model_sources"] = []
    write_json_if_changed(path, metadata)


def _sync_dataset_metadata(path: Path, slug: str) -> None:
    write_json_if_changed(path, with_kaggle_dataset_id(_read_json(path), slug))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sync_kaggle_metadata()
