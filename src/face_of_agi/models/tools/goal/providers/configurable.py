"""Placeholder configurable provider for goal model tool G."""

from __future__ import annotations


class ConfigurableGoalToolAdapter:
    """Reserved provider slot for external/configurable goal tools."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError("Configurable goal provider is not implemented yet")
