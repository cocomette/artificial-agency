"""Optional JSONL timing spans for runtime profiling."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from time import perf_counter, time
from typing import Any, Iterator

_ENABLED_VALUES = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Return whether timing output is enabled for this process."""

    return os.environ.get("FACE_OF_AGI_TIMING", "").lower() in _ENABLED_VALUES


@contextmanager
def span(name: str, **fields: Any) -> Iterator[None]:
    """Emit one timing event when enabled."""

    if not enabled():
        yield
        return

    started_at = perf_counter()
    wall_started_at = time()
    try:
        yield
    finally:
        emit(
            name,
            duration_seconds=perf_counter() - started_at,
            wall_started_at=wall_started_at,
            **fields,
        )


def emit(name: str, **fields: Any) -> None:
    """Print and optionally persist one timing event."""

    if not enabled():
        return

    payload = {
        "event": "timing",
        "name": name,
        **_jsonable_fields(fields),
    }
    line = "FACE_OF_AGI_TIMING " + json.dumps(payload, sort_keys=True)
    print(line, flush=True)

    output_path = os.environ.get("FACE_OF_AGI_TIMING_JSONL")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _jsonable_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in fields.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
