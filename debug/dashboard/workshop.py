"""E2E command and artifact discovery helpers for the dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

DEFAULT_E2E_DIR = Path("tests/e2e")
DEFAULT_RUNS_DIR = Path("runs")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ImageArtifact:
    """One image artifact ready for display."""

    title: str
    path: Path


@dataclass(frozen=True)
class JsonArtifact:
    """One JSON artifact ready for display."""

    title: str
    path: Path
    content: str
    data: Any | None
    parse_error: str | None = None


@dataclass(frozen=True)
class ResultArtifacts:
    """Generic image and JSON artifacts for one E2E output directory."""

    images: list[ImageArtifact]
    json_files: list[JsonArtifact]


def repo_root() -> Path:
    """Return the repository root for this checkout."""

    return Path(__file__).resolve().parents[2]


def list_e2e_tests(
    test_dir: str | Path = DEFAULT_E2E_DIR,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    """Return direct E2E runner scripts sorted by filename."""

    directory = _resolve_under_root(test_dir, root=root)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix == ".py"
    )


def safe_e2e_path(
    value: str | Path,
    test_dir: str | Path = DEFAULT_E2E_DIR,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve an E2E script path and reject paths outside the E2E directory."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("E2E script filename is required")

    directory = _resolve_under_root(test_dir, root=root)
    root_path = Path(root).resolve() if root is not None else repo_root()
    path = Path(raw)
    if path.is_absolute():
        candidate = path.resolve()
    elif _starts_with_path(path, DEFAULT_E2E_DIR):
        candidate = (root_path / path).resolve()
    else:
        candidate = (directory / path).resolve()

    try:
        relative = candidate.relative_to(directory)
    except ValueError as exc:
        raise ValueError(f"E2E script path must stay within {directory}") from exc

    if len(relative.parts) != 1:
        raise ValueError("E2E scripts must be direct children of the E2E directory")
    if candidate.suffix != ".py":
        raise ValueError("E2E script filename must end in .py")
    return candidate


def build_e2e_command(
    script: str | Path,
    *,
    extra_args: Sequence[str] | None = None,
    test_dir: str | Path = DEFAULT_E2E_DIR,
    root: str | Path | None = None,
) -> list[str]:
    """Build the dashboard command for one manual E2E script."""

    script_path = safe_e2e_path(script, test_dir=test_dir, root=root)
    script_arg = str(_relative_to_root(script_path, root=root))
    command = [*_command_prefix(script_path.name), "python", script_arg]
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command


def list_result_dirs(
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    """Return run output directories containing images or JSON files."""

    directory = _resolve_under_root(runs_dir, root=root)
    if not directory.exists():
        return []
    result_dirs = [
        path
        for path in directory.iterdir()
        if path.is_dir() and any(_is_result_artifact(item) for item in path.rglob("*"))
    ]
    return sorted(
        result_dirs,
        key=lambda path: (-_latest_artifact_mtime(path), path.name),
    )


def collect_result_artifacts(
    result_dir: str | Path,
    *,
    root: str | Path | None = None,
) -> ResultArtifacts:
    """Collect all image and JSON files below one E2E result directory."""

    root_path = Path(root).resolve() if root is not None else repo_root()
    directory = _resolve_under_root(result_dir, root=root_path)
    json_paths = _sorted_json_files(directory)
    title_hints = _image_title_hints(json_paths, directory, root_path)

    images = [
        ImageArtifact(
            title=title_hints.get(
                path.resolve(),
                _default_artifact_title(path, directory),
            ),
            path=path,
        )
        for path in _sorted_artifact_files(directory, IMAGE_SUFFIXES)
    ]
    json_files = [_read_json_artifact(path, directory) for path in json_paths]
    return ResultArtifacts(images=images, json_files=json_files)


def default_result_dir_for_test(
    script: str | Path,
    *,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    root: str | Path | None = None,
) -> Path:
    """Return the conventional runs/<script-stem> output directory."""

    script_path = safe_e2e_path(script, root=root)
    return _resolve_under_root(runs_dir, root=root) / script_path.stem


def _command_prefix(script_name: str) -> list[str]:
    if script_name.startswith("vllm_"):
        return ["uv", "run", "--group", "dev"]
    if script_name.startswith("openai_"):
        return [
            "uv",
            "run",
            "--env-file",
            ".env",
            "--locked",
            "--extra",
            "ml",
            "--no-dev",
        ]
    return ["uv", "run", "--locked", "--extra", "ml", "--no-dev"]


def _image_title_hints(
    json_paths: list[Path],
    result_dir: Path,
    root: Path,
) -> dict[Path, str]:
    hints: dict[Path, str] = {}
    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        _collect_image_title_hints(
            data,
            key_path=(),
            json_path=json_path,
            result_dir=result_dir,
            root=root,
            hints=hints,
        )
    return hints


def _collect_image_title_hints(
    value: Any,
    *,
    key_path: tuple[str, ...],
    json_path: Path,
    result_dir: Path,
    root: Path,
    hints: dict[Path, str],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_image_title_hints(
                item,
                key_path=(*key_path, str(key)),
                json_path=json_path,
                result_dir=result_dir,
                root=root,
                hints=hints,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value, start=1):
            _collect_image_title_hints(
                item,
                key_path=(*key_path, str(index)),
                json_path=json_path,
                result_dir=result_dir,
                root=root,
                hints=hints,
            )
        return
    if not isinstance(value, str) or Path(value).suffix.lower() not in IMAGE_SUFFIXES:
        return

    path = _resolve_artifact_reference(
        value,
        json_path=json_path,
        result_dir=result_dir,
        root=root,
    )
    hints.setdefault(path.resolve(), _title_from_key_path(key_path, path))


def _title_from_key_path(key_path: tuple[str, ...], path: Path) -> str:
    for item in reversed(key_path):
        if item and item not in {"image_path", "path"} and not item.isdigit():
            return item
    return path.stem


def _resolve_artifact_reference(
    value: str,
    *,
    json_path: Path,
    result_dir: Path,
    root: Path,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        result_dir / raw,
        json_path.parent / raw,
        root / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (result_dir / raw).resolve()


def _read_json_artifact(path: Path, result_dir: Path) -> JsonArtifact:
    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return JsonArtifact(
            title=_default_artifact_title(path, result_dir),
            path=path,
            content=content,
            data=None,
            parse_error=str(exc),
        )
    return JsonArtifact(
        title=_json_title(path, result_dir, data),
        path=path,
        content=content,
        data=data,
    )


def _json_title(path: Path, result_dir: Path, data: Any) -> str:
    if isinstance(data, dict):
        for key in ("json_title", "title"):
            title = data.get(key)
            if isinstance(title, str) and title.strip():
                return title.strip()
    return _default_artifact_title(path, result_dir)


def _sorted_json_files(directory: Path) -> list[Path]:
    return sorted(
        _sorted_artifact_files(directory, {".json"}),
        key=lambda path: (
            path.name != "summary.json",
            path.relative_to(directory).as_posix(),
        ),
    )


def _sorted_artifact_files(directory: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _default_artifact_title(path: Path, result_dir: Path) -> str:
    relative = path.relative_to(result_dir)
    if len(relative.parts) == 1:
        return path.stem
    return relative.with_suffix("").as_posix()


def _is_result_artifact(path: Path) -> bool:
    return path.is_file() and (
        path.suffix.lower() in IMAGE_SUFFIXES or path.suffix.lower() == ".json"
    )


def _latest_artifact_mtime(directory: Path) -> float:
    mtimes = [
        path.stat().st_mtime
        for path in directory.rglob("*")
        if _is_result_artifact(path)
    ]
    if mtimes:
        return max(mtimes)
    return directory.stat().st_mtime


def _relative_to_root(path: Path, *, root: str | Path | None = None) -> Path:
    root_path = Path(root).resolve() if root is not None else repo_root()
    try:
        return path.resolve().relative_to(root_path)
    except ValueError:
        return path


def _resolve_under_root(value: str | Path, *, root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root_path = Path(root).resolve() if root is not None else repo_root()
    return (root_path / path).resolve()


def _starts_with_path(path: Path, prefix: Path) -> bool:
    return path.parts[: len(prefix.parts)] == prefix.parts
