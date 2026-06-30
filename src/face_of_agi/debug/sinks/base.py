"""Debug event sink interfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from face_of_agi.debug.events import DebugEvent


class DebugSink(Protocol):
    """Receive typed debug events."""

    def emit(self, event: DebugEvent) -> None:
        """Handle one debug event."""


class NullDebugSink:
    """Debug sink that discards every event."""

    def emit(self, event: DebugEvent) -> None:
        """Discard one debug event."""


class CompositeDebugSink:
    """Fan debug events out to multiple sinks."""

    def __init__(self, sinks: Iterable[DebugSink]) -> None:
        self._sinks = tuple(sinks)

    def emit(self, event: DebugEvent) -> None:
        """Forward one debug event to every configured sink."""

        for sink in self._sinks:
            sink.emit(event)

