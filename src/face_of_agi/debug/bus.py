"""Debug bus used by production code to emit debug facts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from face_of_agi.contracts import FrameTurnContext
from face_of_agi.debug.events import DebugEvent
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
        self._usage_by_turn: dict[tuple[str, str, int], dict[str, int]] = {}

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

        self._record_usage(
            run_id=frame_context.run_id,
            game_id=frame_context.game_id,
            turn_id=turn_id,
            records=records,
        )

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

    def model_token_usage(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
    ) -> dict[str, int]:
        """Return aggregated provider token usage captured for one turn."""

        return dict(self._usage_by_turn.get((run_id, game_id, turn_id), {}))

    def clear_model_token_usage(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
    ) -> None:
        """Drop retained per-turn token usage after reward persistence."""

        self._usage_by_turn.pop((run_id, game_id, turn_id), None)

    def _record_usage(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        records: list[dict[str, Any]],
    ) -> None:
        target = self._usage_by_turn.setdefault(
            (run_id, game_id, turn_id),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        for record in records:
            usage = record.get("usage")
            prompt, completion, total = _usage_tokens(usage)
            target["prompt_tokens"] += prompt
            target["completion_tokens"] += completion
            target["total_tokens"] += total


def _usage_tokens(value: Any) -> tuple[int, int, int]:
    """Normalize common provider usage payloads to prompt/completion/total."""

    if isinstance(value, dict):
        prompt = _non_negative_token_count(
            value.get("prompt_tokens", value.get("input_tokens", 0))
        )
        completion = _non_negative_token_count(
            value.get("completion_tokens", value.get("output_tokens", 0))
        )
        total = _non_negative_token_count(value.get("total_tokens", 0))
        if total == 0:
            total = prompt + completion
        return prompt, completion, total
    if isinstance(value, (list, tuple)):
        prompt = completion = total = 0
        for item in value:
            item_prompt, item_completion, item_total = _usage_tokens(item)
            prompt += item_prompt
            completion += item_completion
            total += item_total
        return prompt, completion, total
    return 0, 0, 0


def _non_negative_token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))
