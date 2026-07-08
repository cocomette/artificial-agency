"""Placeholder configurable provider for world model tool S."""

from __future__ import annotations


class ConfigurableWorldToolAdapter:
    """Reserved provider slot for external/configurable world tools."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError("Configurable world provider is not implemented yet")
