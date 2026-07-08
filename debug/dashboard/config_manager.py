"""Config file discovery, validation, and safe writes for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path("src/face_of_agi/runtime/configs")
CONFIG_SUFFIXES = {".yaml", ".yml"}
REQUIRED_TOP_LEVEL_KEYS = {"game_index", "max_actions_per_level", "models"}
REQUIRED_UPDATER_SLOTS = {"world", "agent", "general"}


@dataclass(frozen=True)
class ConfigValidation:
    """Validation result for raw runtime config YAML text."""

    valid: bool
    message: str
    data: dict[str, Any] | None = None


def repo_root() -> Path:
    """Return the repository root for this checkout."""

    return Path(__file__).resolve().parents[2]


def list_config_files(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    """Return YAML config files under the config directory."""

    directory = config_dir_path(config_dir, root=root)
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in CONFIG_SUFFIXES
        ],
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def config_label(
    path: str | Path,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    root: str | Path | None = None,
) -> str:
    """Return the dashboard display label for a config path."""

    config_path = safe_config_path(path, config_dir=config_dir, root=root)
    return config_path.relative_to(config_dir_path(config_dir, root=root)).as_posix()


def read_config(
    path: str | Path,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    root: str | Path | None = None,
) -> str:
    """Read a config file after applying dashboard path safety checks."""

    config_path = safe_config_path(path, config_dir=config_dir, root=root)
    return config_path.read_text(encoding="utf-8")


def save_config(
    path: str | Path,
    text: str,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    root: str | Path | None = None,
) -> Path:
    """Validate and overwrite one existing config file."""

    validation = validate_config_text(text)
    if not validation.valid:
        raise ValueError(validation.message)
    config_path = safe_config_path(path, config_dir=config_dir, root=root)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_with_trailing_newline(text), encoding="utf-8")
    return config_path


def save_config_as(
    filename: str | Path,
    text: str,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and write a new config file under the configs directory."""

    config_path = safe_config_path(filename, config_dir=config_dir, root=root)
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"config already exists: {config_path.name}")
    return save_config(
        config_path,
        text,
        config_dir=config_dir,
        root=root,
    )


def validate_config_text(text: str) -> ConfigValidation:
    """Validate raw YAML enough to catch common runtime config mistakes."""

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return ConfigValidation(False, f"Invalid YAML: {exc}")

    if not isinstance(loaded, dict):
        return ConfigValidation(False, "Config YAML must be a mapping.")

    missing = REQUIRED_TOP_LEVEL_KEYS - loaded.keys()
    if missing:
        return ConfigValidation(
            False,
            f"Missing required config keys: {', '.join(sorted(missing))}.",
        )

    for key in ("game_index", "max_actions_per_level"):
        try:
            int(loaded[key])
        except (TypeError, ValueError):
            return ConfigValidation(False, f"{key} must be an integer.")

    models = loaded.get("models")
    if not isinstance(models, dict):
        return ConfigValidation(False, "models must be a mapping.")

    updater = models.get("updater")
    if not isinstance(updater, dict):
        return ConfigValidation(False, "models.updater must be a mapping.")

    missing_slots = REQUIRED_UPDATER_SLOTS - updater.keys()
    if missing_slots:
        return ConfigValidation(
            False,
            "Missing updater slots: " + ", ".join(sorted(missing_slots)) + ".",
        )

    return ConfigValidation(True, "Config YAML is valid.", loaded)


def safe_config_path(
    value: str | Path,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    root: str | Path | None = None,
) -> Path:
    """Resolve a config path and reject paths outside the config directory."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("config filename is required")

    base_dir = config_dir_path(config_dir, root=root)
    root_path = Path(root).resolve() if root is not None else repo_root()
    path = Path(raw)
    if path.is_absolute():
        candidate = path.resolve()
    elif _starts_with_default_config_dir(path):
        candidate = (root_path / path).resolve()
    else:
        candidate = (base_dir / path).resolve()

    try:
        relative = candidate.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"config path must stay within {base_dir}") from exc

    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("config path must stay within the config directory")
    if candidate.suffix not in CONFIG_SUFFIXES:
        raise ValueError("config filename must end in .yaml or .yml")
    return candidate


def config_dir_path(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve the dashboard config directory."""

    path = Path(config_dir)
    if path.is_absolute():
        return path.resolve()
    root_path = Path(root).resolve() if root is not None else repo_root()
    return (root_path / path).resolve()


def _starts_with_default_config_dir(path: Path) -> bool:
    parts = path.parts
    default_parts = DEFAULT_CONFIG_DIR.parts
    return parts[: len(default_parts)] == default_parts


def _with_trailing_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return text + "\n"
