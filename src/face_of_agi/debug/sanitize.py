"""Sanitization helpers for debug output."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
import re
from typing import Any

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "x-api-key",
)
_SENSITIVE_TOKEN_KEYS = (
    "token",
    "access-token",
    "auth-token",
    "bearer-token",
    "refresh-token",
)
_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def sanitize_for_debug(value: Any) -> Any:
    """Convert arbitrary model/debug data to safe JSON-compatible values."""

    return _sanitize(value, key=None)


def _sanitize(value: Any, *, key: str | None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return "[redacted]"

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if _is_image_like(value):
        return _image_object_summary(value)
    if is_dataclass(value):
        return _sanitize(asdict(value), key=key)
    if hasattr(value, "model_dump"):
        try:
            return _sanitize(value.model_dump(mode="json", exclude_none=True), key=key)
        except TypeError:
            return _sanitize(value.model_dump(), key=key)
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(item, key=None) for item in value]
    if hasattr(value, "__dict__"):
        return _sanitize(vars(value), key=key)
    return repr(value)


def _sanitize_string(value: str) -> Any:
    data_url = _DATA_URL_RE.match(value)
    if data_url is not None:
        mime_type = data_url.group(1)
        encoded = data_url.group(2)
        summary: dict[str, Any] = {
            "kind": "omitted_image_data_url",
            "mime_type": mime_type,
            "omitted_chars": len(value),
        }
        size = _image_size_from_base64(encoded)
        if size is not None:
            summary["image_size"] = list(size)
        return summary

    compact = value.strip()
    if len(compact) > 512 and " " not in compact and _BASE64_RE.match(compact):
        return {
            "kind": "omitted_base64",
            "omitted_chars": len(value),
        }

    return value


def _image_size_from_base64(encoded: str) -> tuple[int, int] | None:
    try:
        image_bytes = base64.b64decode(encoded, validate=False)
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            return tuple(image.size)
    except Exception:
        return None


def _is_image_like(value: Any) -> bool:
    return hasattr(value, "size") and hasattr(value, "mode")


def _image_object_summary(value: Any) -> dict[str, Any]:
    size = getattr(value, "size", None)
    return {
        "kind": "image_object",
        "mode": getattr(value, "mode", None),
        "image_size": list(size) if isinstance(size, tuple) else size,
    }


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    dashed = normalized.replace("_", "-")
    compact = dashed.replace("-", "")
    if any(
        part in normalized
        or part in dashed
        or part.replace("_", "").replace("-", "") in compact
        for part in _SENSITIVE_KEY_PARTS
    ):
        return True
    return dashed in _SENSITIVE_TOKEN_KEYS or dashed.endswith("-token")
