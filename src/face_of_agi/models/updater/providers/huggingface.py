"""Placeholder Hugging Face provider for updater P."""

from __future__ import annotations


class HuggingFaceUpdaterAdapter:
    """Reserved provider slot for a future Hugging Face updater backend."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError(
            "Hugging Face updater provider is not implemented yet"
        )
