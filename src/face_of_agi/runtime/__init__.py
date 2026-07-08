"""Runtime entry points and loop assembly."""

from typing import Any

__all__ = ["RuntimeLoop"]


def __getattr__(name: str) -> Any:
    """Load heavier runtime entry points only when they are requested."""

    if name == "RuntimeLoop":
        from face_of_agi.runtime.loop import RuntimeLoop

        return RuntimeLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
