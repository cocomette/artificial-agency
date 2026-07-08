"""Placeholder configurable provider for updater P."""

from __future__ import annotations


class ConfigurableUpdaterAdapter:
    """Reserved provider slot for external/configurable updater backends."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError(
            "Configurable updater provider is not implemented yet"
        )
