"""Resolve a Kaggle kernel ref from a CLI override or metadata file."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from urllib.parse import urlparse


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: resolve_kaggle_kernel_ref.py REF METADATA_PATH")

    ref = sys.argv[1].strip()
    metadata_path = Path(sys.argv[2])
    print(_kernel_id(ref) if ref else _metadata_kernel_id(metadata_path))


def _kernel_id(ref: str) -> str:
    if ref.startswith(("http://", "https://")):
        return _kernel_id_from_url(ref)
    if ref.count("/") != 1:
        raise SystemExit(
            "DEBUG_KERNEL must be owner/slug or a kaggle.com/code/owner/slug URL"
        )
    return ref


def _kernel_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        code_index = parts.index("code")
    except ValueError as exc:
        raise SystemExit(
            "DEBUG_KERNEL URL must look like kaggle.com/code/owner/slug"
        ) from exc

    kernel_parts = parts[code_index + 1 : code_index + 3]
    if len(kernel_parts) != 2 or not all(kernel_parts):
        raise SystemExit(
            "DEBUG_KERNEL URL must include both the owner and kernel slug"
        )
    return "/".join(kernel_parts)


def _metadata_kernel_id(path: Path) -> str:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    kernel_id = metadata.get("id")
    if not isinstance(kernel_id, str) or kernel_id.count("/") != 1:
        raise SystemExit(f"{path} must contain a Kaggle kernel id as owner/slug")
    return kernel_id


if __name__ == "__main__":
    main()
