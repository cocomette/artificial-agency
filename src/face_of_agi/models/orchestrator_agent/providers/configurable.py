"""Placeholder configurable provider for orchestrator agent X."""

from __future__ import annotations


class ConfigurableOrchestratorAgentAdapter:
    """Reserved provider slot for external/configurable X backends."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError(
            "Configurable Agent X provider is not implemented yet"
        )
