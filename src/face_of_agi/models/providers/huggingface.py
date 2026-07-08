"""Empty Hugging Face provider shell."""

from __future__ import annotations

from typing import Any


class HuggingFaceProviderShell:
    """Non-functional placeholder for future non-image Hugging Face providers."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError("Hugging Face model providers are not implemented")


__all__ = ["HuggingFaceProviderShell"]
