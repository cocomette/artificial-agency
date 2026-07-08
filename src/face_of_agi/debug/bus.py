"""Debug bus used by production code to emit debug facts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from face_of_agi.contracts import FrameTurnContext
from face_of_agi.debug.events import (
    DebugEvent,
    EnvironmentStepEventRecorded,
    ModelCallEventRecorded,
)
from face_of_agi.debug.capture.model_inputs import drain_model_input_debug_records
from face_of_agi.debug.sinks import DebugSink, NullDebugSink

if TYPE_CHECKING:
    from face_of_agi.memory.state import StateMemory


class DebugBus:
    """Small production-facing debug boundary."""

    def __init__(
        self,
        *,
        sink: DebugSink | None = None,
        state_memory: StateMemory | None = None,
        persist_model_input_debug_records: bool = True,
    ) -> None:
        self.sink = sink or NullDebugSink()
        self.state_memory = state_memory
        self.persist_model_input_debug_records = persist_model_input_debug_records

    @classmethod
    def disabled(cls) -> "DebugBus":
        """Return a bus that drops all emitted debug facts."""

        return cls()

    def emit(self, event: DebugEvent) -> None:
        """Emit one typed debug event."""

        self.sink.emit(event)

    def capture_model_inputs(
        self,
        frame_context: FrameTurnContext,
        turn_id: int,
        adapter: Any | None,
    ) -> None:
        """Persist and clear provider request captures for one model call slot."""

        if adapter is None:
            return

        records = drain_model_input_debug_records(adapter)
        if not records:
            return

        if not self.persist_model_input_debug_records:
            return

        if self.state_memory is None or frame_context.current_source_state_id is None:
            return

        self.state_memory.write_model_input_debug_records(
            m_state_id=frame_context.current_source_state_id,
            run_id=frame_context.run_id,
            game_id=frame_context.game_id,
            turn_id=turn_id,
            records=records,
        )

    def record_model_call_event(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int | None,
        role: str,
        provider: str,
        model: str | None,
        event: str,
        status: str,
        queue_wait_seconds: float | None = None,
        duration_seconds: float | None = None,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist and emit one model-call lifecycle event."""

        if self.state_memory is not None:
            self.state_memory.write_model_call_event(
                run_id=run_id,
                game_id=game_id,
                turn_id=turn_id,
                role=role,
                provider=provider,
                model=model,
                event=event,
                status=status,
                queue_wait_seconds=queue_wait_seconds,
                duration_seconds=duration_seconds,
                timeout_seconds=timeout_seconds,
                metadata=metadata,
            )
        self.emit(
            ModelCallEventRecorded(
                role=role,
                provider=provider,
                model=model,
                event=event,
                status=status,
                game_id=game_id,
                turn_id=turn_id,
                duration_seconds=duration_seconds,
                queue_wait_seconds=queue_wait_seconds,
                timeout_seconds=timeout_seconds,
                metadata=metadata,
            )
        )

    def record_environment_step_event(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int | None,
        step: int | None,
        action: dict[str, Any],
        status: str,
        duration_seconds: float,
        remaining_actions: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist and emit one environment-step timing event."""

        if self.state_memory is not None:
            self.state_memory.write_environment_step_event(
                run_id=run_id,
                game_id=game_id,
                turn_id=turn_id,
                step=step,
                action=action,
                status=status,
                duration_seconds=duration_seconds,
                remaining_actions=remaining_actions,
                metadata=metadata,
            )
        self.emit(
            EnvironmentStepEventRecorded(
                status=status,
                duration_seconds=duration_seconds,
                remaining_actions=remaining_actions,
            )
        )
