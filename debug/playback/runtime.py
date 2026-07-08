"""Debug-only replay wrappers around the normal runtime model roles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionOutcomeEvidence,
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DecisionResult,
    MStateRecord,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.memory import StateMemory
from face_of_agi.models import ModelRegistry, UpdaterTaskRegistry
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
)


class PlaybackError(RuntimeError):
    """Raised when persisted debug playback rows cannot drive a replay."""


@dataclass(frozen=True, slots=True)
class PlaybackRequest:
    """Source run/game/turn selected for debug playback."""

    source_run_id: str
    game_id: str
    turn_id: int


@dataclass(frozen=True, slots=True)
class PlaybackSetup:
    """Prepared model wrappers and initial contexts for one playback run."""

    request: PlaybackRequest
    models: ModelRegistry
    contexts: ContextDocuments
    replay_turn_count: int


def prepare_playback(
    *,
    state_memory: StateMemory,
    request: PlaybackRequest,
    live_models: ModelRegistry,
) -> PlaybackSetup:
    """Load persisted source rows and wrap live models for debug playback."""

    timeline = PlaybackTimeline(load_replay_rows(state_memory, request))
    return PlaybackSetup(
        request=request,
        models=_wrap_models(live_models, timeline),
        contexts=ContextDocuments(),
        replay_turn_count=len(timeline.rows),
    )


def load_replay_rows(
    state_memory: StateMemory,
    request: PlaybackRequest,
) -> tuple[MStateRecord, ...]:
    """Return complete source rows before the selected handoff turn."""

    if request.turn_id < 1:
        raise PlaybackError("playback turn id must be at least 1")

    source_rows = [
        row
        for row in state_memory.list_states(game_id=request.game_id)
        if row.run_id == request.source_run_id
    ]
    rows_by_turn: dict[int, MStateRecord] = {}
    for row in source_rows:
        turn_id = _record_turn_id(row)
        if turn_id in rows_by_turn:
            raise PlaybackError(
                "playback source contains duplicate M rows for "
                f"run={request.source_run_id!r} game={request.game_id!r} "
                f"turn={turn_id}"
            )
        rows_by_turn[turn_id] = row

    if request.turn_id not in rows_by_turn:
        raise PlaybackError(
            "playback target M row was not found or is incomplete: "
            f"run={request.source_run_id!r} game={request.game_id!r} "
            f"turn={request.turn_id}"
        )

    missing = [turn for turn in range(1, request.turn_id) if turn not in rows_by_turn]
    if missing:
        missing_text = ", ".join(str(turn) for turn in missing)
        raise PlaybackError(
            "playback source history is missing required prior turn(s): "
            f"{missing_text}. Enable debug_keep_all_m_states for source runs."
        )

    replay_rows = tuple(rows_by_turn[turn] for turn in range(1, request.turn_id))
    for row in replay_rows:
        _require_replayable_row(row)
    return replay_rows


class PlaybackTimeline:
    """Mutable cursor over recorded rows that should be replayed."""

    def __init__(self, rows: Sequence[MStateRecord]) -> None:
        self.rows = tuple(rows)
        self.index = 0

    def active(self) -> bool:
        """Return whether the current turn should be replayed."""

        return self.index < len(self.rows)

    def current(self) -> MStateRecord:
        """Return the row for the current replay turn."""

        if not self.active():
            raise PlaybackError("playback cursor already reached handoff")
        return self.rows[self.index]

    def advance(self) -> None:
        """Advance after one replay turn has supplied stored contexts."""

        if not self.active():
            raise PlaybackError("playback cursor cannot advance after handoff")
        self.index += 1


class PlaybackAgent:
    """Agent X wrapper that returns stored actions before handoff."""

    def __init__(self, *, timeline: PlaybackTimeline, live_agent: Any) -> None:
        self.timeline = timeline
        self.live_agent = live_agent
        self.provider = _capture_target(live_agent)
        self._provider = live_agent

    def decide(
        self,
        context: RoleContext,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
        *,
        glossary_actions: Sequence[ActionSpec],
        first_observation_ref: ObservationRef | None = None,
        recent_action_history_available: bool = True,
        action_outcome_evidence: ActionOutcomeEvidence | None = None,
    ) -> DecisionResult:
        """Return a stored replay decision or delegate to live Agent X."""

        if not self.timeline.active():
            return self.live_agent.decide(
                context=context,
                current_observation=current_observation,
                action_space=action_space,
                tool_runtime=tool_runtime,
                recent_action_history=recent_action_history,
                glossary_actions=glossary_actions,
                first_observation_ref=first_observation_ref,
                recent_action_history_available=recent_action_history_available,
                action_outcome_evidence=action_outcome_evidence,
            )

        row = self.timeline.current()
        if not _record_controllable(row):
            raise PlaybackError(
                f"playback unexpectedly called Agent X for non-controllable "
                f"turn {_record_turn_id(row)}"
            )

        final_action = _action_from_payload(row.chosen_action, action_space)
        source_state_id = (
            tool_runtime.current_source_state_id if tool_runtime is not None else None
        )
        trace = _agent_trace_from_payload(
            row.agent_trace,
            final_action=final_action,
            current_observation=current_observation,
            first_observation_ref=first_observation_ref,
            source_state_id=source_state_id,
        )
        return DecisionResult(final_action=final_action, trace=trace)


class PlaybackAgentUpdater:
    """Agent updater wrapper that advances replay after stored contexts apply."""

    def __init__(self, *, timeline: PlaybackTimeline, live_updater: Any) -> None:
        self.timeline = timeline
        self.live_updater = live_updater
        self.provider = _capture_target(live_updater)
        self._provider = live_updater

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        """Return stored context during replay, otherwise delegate."""

        if not self.timeline.active():
            return self.live_updater.update_agent_game_context(update_input)
        context = self.timeline.current().agent_context
        self.timeline.advance()
        return context


class PlaybackGeneralUpdater:
    """General updater wrapper; replay never short-circuits end-of-run updates."""

    def __init__(self, *, live_updater: Any) -> None:
        self.live_updater = live_updater
        self.provider = _capture_target(live_updater)
        self._provider = live_updater

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        """Delegate end-of-run updates to the live general updater."""

        return self.live_updater.update_general_knowledge(update_input)


def _wrap_models(
    live_models: ModelRegistry,
    timeline: PlaybackTimeline,
) -> ModelRegistry:
    live_updaters = live_models.require_updater_tasks()
    return ModelRegistry(
        agent_context_historizer_model=(
            live_models.agent_context_historizer_model
        ),
        change_summary_model=live_models.require_change_summary_model(),
        orchestrator_agent=PlaybackAgent(
            timeline=timeline,
            live_agent=live_models.require_orchestrator_agent(),
        ),
        updater_tasks=UpdaterTaskRegistry(
            agent_game_updater=PlaybackAgentUpdater(
                timeline=timeline,
                live_updater=live_updaters.require_agent_game_updater(),
            ),
            general_updater=PlaybackGeneralUpdater(
                live_updater=live_updaters.require_general_updater(),
            ),
        ),
    )


def _capture_target(adapter: Any) -> Any:
    """Return the nested provider object that receives debug capture records."""

    return getattr(adapter, "provider", adapter)


def _require_replayable_row(row: MStateRecord) -> None:
    turn_id = _record_turn_id(row)
    if row.chosen_action is None:
        raise PlaybackError(f"playback turn {turn_id} is missing chosen_action")
    if row.agent_trace is None:
        raise PlaybackError(f"playback turn {turn_id} is missing agent_trace")
    _record_controllable(row)


def _record_turn_id(row: MStateRecord) -> int:
    value = row.metadata.get("turn_id")
    if isinstance(value, bool) or value is None:
        raise PlaybackError(f"M state {row.id} is missing metadata.turn_id")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlaybackError(f"M state {row.id} has invalid metadata.turn_id") from exc


def _record_controllable(row: MStateRecord) -> bool:
    control_mode = row.metadata.get("control_mode")
    if not isinstance(control_mode, dict):
        raise PlaybackError(
            f"playback turn {_record_turn_id(row)} is missing metadata.control_mode"
        )
    return bool(control_mode.get("controllable", False))


def _action_from_payload(
    payload: Any,
    action_space: Sequence[ActionSpec],
) -> ActionSpec:
    if not isinstance(payload, dict):
        raise PlaybackError("recorded action must be an object")

    action_name = _action_name(payload)
    if action_name == "NONE":
        return ActionSpec.none()

    for candidate in action_space:
        if action_name in {candidate.name, str(candidate.action_id)}:
            return ActionSpec(
                action_id=candidate.action_id,
                data=_optional_dict(payload.get("data")),
            )

    allowed = ", ".join(action.name for action in action_space)
    raise PlaybackError(
        f"recorded action {action_name!r} is not available during replay; "
        f"allowed actions: {allowed}"
    )


def _action_name(payload: Any) -> str:
    raw = str(_dict(payload).get("action_id") or "")
    if raw.startswith("<GameAction.") and ":" in raw:
        raw = raw.removeprefix("<GameAction.").split(":", 1)[0]
    if raw.startswith("GameAction."):
        raw = raw.split(".", 1)[1]
    if not raw:
        raise PlaybackError("recorded action is missing action_id")
    return raw


def _agent_trace_from_payload(
    payload: Any,
    *,
    final_action: ActionSpec,
    current_observation: Observation,
    first_observation_ref: ObservationRef | None,
    source_state_id: int | None,
) -> AgentTrace:
    trace = _dict(payload)
    current_ref = ObservationRef(memory="state", id=current_observation.id)
    return AgentTrace(
        step=current_observation.step,
        first_observation_ref=(
            _observation_ref_from_payload(trace.get("first_observation_ref"))
            or first_observation_ref
            or current_ref
        ),
        current_observation_ref=current_ref,
        final_action=final_action,
        tool_calls=[
            _tool_call_from_payload(call, source_state_id=source_state_id)
            for call in _list(trace.get("tool_calls"))
        ],
        tool_results=[
            _tool_result_from_payload(
                result,
                tool=str(_dict(result).get("tool") or "tool"),
                observation=current_observation,
                source_state_id=source_state_id,
            )
            for result in _list(trace.get("tool_results"))
        ],
        reasoning_summary=_optional_string(trace.get("reasoning_summary")),
        metadata=_dict(trace.get("metadata")),
    )


def _observation_ref_from_payload(payload: Any) -> ObservationRef | None:
    value = _dict(payload)
    ref_id = value.get("id")
    if ref_id is None:
        return None
    memory = str(value.get("memory") or "state")
    if memory not in {"state", "experimental"}:
        memory = "state"
    return ObservationRef(memory=memory, id=str(ref_id))  # type: ignore[arg-type]


def _tool_call_from_payload(
    payload: Any,
    *,
    source_state_id: int | None,
) -> ToolCall:
    value = _dict(payload)
    return ToolCall(
        tool=str(value.get("tool") or "tool"),
        source_state_id=_optional_int(source_state_id, value.get("source_state_id")),
        action=_unvalidated_action(value.get("action")),
    )


def _tool_result_from_payload(
    payload: Any,
    *,
    tool: str,
    observation: Observation,
    action: ActionSpec | None = None,
    source_state_id: int | None = None,
) -> ToolResult:
    value = _dict(payload)
    if not value:
        raise PlaybackError(f"recorded {tool} prediction is missing")
    return ToolResult(
        id=str(value.get("id") or f"playback-{tool}-{observation.id}"),
        tool=tool,
        output=value.get("output"),
        source_observation_ref=ObservationRef(memory="state", id=observation.id),
        source_state_id=_optional_int(source_state_id, value.get("source_state_id")),
        action=action or _unvalidated_action(value.get("action")),
        explanation=_optional_string(value.get("explanation")),
        metadata=_dict(value.get("metadata")),
    )


def _unvalidated_action(payload: Any) -> ActionSpec | None:
    if not isinstance(payload, dict):
        return None
    action_name = _action_name(payload)
    if action_name == "NONE":
        return ActionSpec.none()
    return ActionSpec(
        action_id=action_name,
        data=_optional_dict(payload.get("data")),
    )


def _optional_int(primary: Any, fallback: Any = None) -> int:
    value = primary if primary is not None else fallback
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    raise PlaybackError(f"expected recorded action data to be an object, got {value!r}")


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []
