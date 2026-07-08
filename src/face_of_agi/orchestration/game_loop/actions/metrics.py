"""Frame-turn cost and progress metrics."""

from __future__ import annotations

from typing import Any

from face_of_agi.contracts import DecisionResult, Observation, TurnMetrics


def effective_trace_cost_seconds(
    *,
    decision: DecisionResult,
    wall_clock_seconds: float | None,
) -> float | None:
    """Return decision time with explicit provider model-load time removed."""

    if wall_clock_seconds is None:
        return None

    load_seconds = _load_duration_seconds(
        decision.trace.metadata.get("usage")
    ) + sum(
        _load_duration_seconds(result.metadata.get("usage"))
        for result in decision.trace.tool_results
    )
    return max(0.0, wall_clock_seconds - load_seconds)


def turn_metrics(
    *,
    actual_next_observation: Observation | None,
    trace_cost_seconds: float | None,
    cumulative_time_cost: float | None,
) -> TurnMetrics:
    """Build frame-turn metrics for persistence and updater boundaries."""

    return TurnMetrics(
        time_cost=cumulative_time_cost,
        trace_cost=trace_cost_seconds,
        cumulative_score=cumulative_score(actual_next_observation),
    )


def cumulative_score(actual_next_observation: Observation | None) -> float | None:
    """Return completed levels after the frame transition when available."""

    next_levels_completed = levels_completed(actual_next_observation)
    if next_levels_completed is None:
        return None
    return float(next_levels_completed)


def levels_completed(observation: Observation | None) -> int | None:
    """Read ARC levels-completed metadata from an observation."""

    if observation is None:
        return None
    raw_frame_data = observation.raw_frame_data or observation.metadata.get(
        "raw_frame_data"
    )
    if raw_frame_data is not None and hasattr(raw_frame_data, "levels_completed"):
        return int(raw_frame_data.levels_completed)
    metadata_value = observation.metadata.get("levels_completed")
    if metadata_value is None:
        return None
    return int(metadata_value)


def _load_duration_seconds(value: Any) -> float:
    """Sum Ollama-style load_duration nanoseconds from provider usage payloads."""

    if isinstance(value, dict):
        return _nanoseconds_to_seconds(value.get("load_duration"))
    if isinstance(value, (list, tuple)):
        return sum(_load_duration_seconds(item) for item in value)
    return 0.0


def _nanoseconds_to_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value) / 1_000_000_000)
