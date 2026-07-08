"""Shared Kaggle user configuration loaded from ``kaggle/.env``."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = KAGGLE_ROOT / ".env"

KAGGLE_OWNER_ENV = "FACE_OF_AGI_KAGGLE_OWNER"
KAGGLE_TOKEN_FILE_ENV = "FACE_OF_AGI_KAGGLE_TOKEN_FILE"


def kaggle_owner() -> str:
    """Return the configured Kaggle username."""

    owner = _env_value(KAGGLE_OWNER_ENV)
    if not owner:
        raise RuntimeError(f"{KAGGLE_OWNER_ENV} is required in {ENV_PATH}")
    return owner


def kaggle_token_path() -> Path:
    """Return the configured Kaggle token path, resolved from the repo root."""

    token_file = _env_value(KAGGLE_TOKEN_FILE_ENV)
    if not token_file:
        raise RuntimeError(f"{KAGGLE_TOKEN_FILE_ENV} is required in {ENV_PATH}")
    path = Path(token_file).expanduser()
    return path if path.is_absolute() else ROOT / path


def kaggle_ref(slug: str) -> str:
    """Return ``owner/slug`` using the configured Kaggle owner."""

    return f"{kaggle_owner()}/{slug}"


def kaggle_dataset_sources(slugs: tuple[str, ...]) -> list[str]:
    """Return Kaggle dataset source refs for metadata files."""

    return [kaggle_ref(slug) for slug in slugs]


def with_kaggle_dataset_id(metadata: dict, slug: str) -> dict:
    """Return metadata with its Kaggle dataset id owned by the configured user."""

    updated = dict(metadata)
    updated["id"] = kaggle_ref(slug)
    return updated


def with_kaggle_kernel_id(metadata: dict) -> dict:
    """Return kernel metadata with the owner part replaced from ``kaggle/.env``."""

    updated = dict(metadata)
    kernel_id = updated.get("id")
    slug = str(kernel_id).split("/", 1)[-1] if kernel_id else ""
    if not slug:
        raise RuntimeError("Kaggle kernel metadata id must include a slug")
    updated["id"] = kaggle_ref(slug)
    return updated


def with_kaggle_owner_slug(metadata: dict) -> dict:
    """Return model metadata with ``ownerSlug`` set from ``kaggle/.env``."""

    updated = dict(metadata)
    updated["ownerSlug"] = kaggle_owner()
    return updated


def read_json_with_kaggle_dataset_id(path: Path, slug: str) -> dict:
    """Read dataset metadata and apply the configured Kaggle owner."""

    return with_kaggle_dataset_id(_read_json(path), slug)


def write_json_if_changed(path: Path, data: dict) -> None:
    """Write JSON metadata only when the on-disk content differs."""

    text = json.dumps(data, indent=2) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current != text:
        path.write_text(text, encoding="utf-8")


def _env_value(name: str) -> str:
    return os.environ.get(name, _dotenv_values().get(name, "")).strip()


def _dotenv_values() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _unquote_env_value(value.strip())
    return values


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
