"""Runtime source metadata helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

RUNTIME_STARTUP_METADATA_KIND = "runtime_startup"


def build_runtime_source_metadata() -> dict[str, str]:
    """Return source identity metadata for persisted runtime diagnostics."""

    return {
        "source_root": str(_source_root()),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=_source_root(),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return stdout
    error = stderr or stdout or f"git exited with status {completed.returncode}"
    return f"unavailable: {error}"


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]
