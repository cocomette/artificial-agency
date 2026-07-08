"""Debug event sinks."""

from face_of_agi.debug.sinks.base import CompositeDebugSink, DebugSink, NullDebugSink
from face_of_agi.debug.sinks.terminal import DebugTrace

__all__ = [
    "CompositeDebugSink",
    "DebugSink",
    "DebugTrace",
    "NullDebugSink",
]
