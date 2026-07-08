"""Shared memory-turn rendering helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from debug.dashboard.memory_reader import (
    action_label,
    observation_image,
    redacted_for_json,
    turn_summary,
)
from debug.dashboard.UI.model_inputs import render_model_inputs

ACTION6_GRID_SIZE = 64


def select_turn(states: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    """Render a selectable turn table and return the selected state."""

    st.subheader("Turns")
    selected_state_key = f"{key}_selected_m_state_id"
    table = pd.DataFrame([turn_summary(state) for state in states])
    selected_index = _selected_index_from_session(
        states,
        selected_state_id=st.session_state.get(selected_state_key),
    )
    if selected_index is None:
        selected_index = len(states) - 1
        st.session_state[selected_state_key] = int(states[selected_index]["id"])
    selection_default = {"selection": {"rows": [selected_index]}}
    _restore_dataframe_selection(key, selection_default)
    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        selection_default=selection_default,
        key=key,
    )
    selected_rows = list(event.selection.rows)
    if selected_rows:
        selected_state = states[selected_rows[0]]
        st.session_state[selected_state_key] = int(selected_state["id"])
        return selected_state

    return states[selected_index]


def render_selected_turn(
    state: dict[str, Any],
    model_input_records: list[dict[str, Any]] | None = None,
    v1_artifacts: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Render detail tabs for one selected M state."""

    _render_turn_heading(state)

    overview, v1, model_inputs, raw = st.tabs(
        ["Overview", "V1 Roles", "Models I/O", "Raw Data"]
    )
    with overview:
        _render_overview(state)
    with v1:
        _render_v1_artifacts(v1_artifacts or {})
    with model_inputs:
        render_model_inputs(model_input_records or [])
    with raw:
        _render_raw(state, model_input_records or [], v1_artifacts or {})


def render_turn_overview(state: dict[str, Any]) -> None:
    """Render the overview panel for one selected M state without extra tabs."""

    _render_turn_heading(state)
    _render_overview(state)


def render_image(image: Any | None, caption: str) -> None:
    """Render an image payload or a compact empty-frame message."""

    if image is None:
        st.info(f"{caption}: no visual frame")
        return
    st.image(image, caption=caption, width="stretch")


def _render_turn_heading(state: dict[str, Any]) -> None:
    st.subheader(
        f"Turn {state['turn_id']} | M state {state['id']} | step {state['step']}"
    )


def _render_overview(state: dict[str, Any]) -> None:
    metadata = _dict(state.get("metadata"))
    control_mode = _dict(metadata.get("control_mode"))

    cols = st.columns(4)
    cols[0].metric("Action", action_label(state.get("chosen_action")))
    cols[1].metric("Control", str(control_mode.get("reason", "-")))
    cols[2].metric("Frame", f"{state['frame_index'] + 1}/{state['frame_count']}")
    cols[3].metric("Controllable", str(bool(control_mode.get("controllable", False))))

    left, right = st.columns([1, 2])
    with left:
        image, marker_error = _action6_marked_image(
            observation_image(state.get("current_observation")),
            state.get("chosen_action"),
        )
        render_image(image, "Current observed frame")
        if marker_error is not None:
            st.warning(marker_error)
    with right:
        st.write("Allowed actions")
        st.json(
            [action_label(action) for action in control_mode.get("allowed_actions") or []]
        )


def _render_raw(
    state: dict[str, Any],
    model_input_records: list[dict[str, Any]],
    v1_artifacts: dict[str, list[dict[str, Any]]],
) -> None:
    st.json(
        redacted_for_json(
            {
                "m_state": state,
                "matching_model_input_debug_records": model_input_records,
                "v1_artifacts": v1_artifacts,
            }
        )
    )


def _render_v1_artifacts(artifacts: dict[str, list[dict[str, Any]]]) -> None:
    if not any(artifacts.values()):
        st.info("No v1 role artifacts are available for this turn.")
        return

    _render_candidate_predictions(artifacts.get("candidate_predictions") or [])
    _render_judge_scores(artifacts.get("judge_scores") or [])
    _render_rewards(artifacts.get("rewards") or [])
    _render_goal_memory(
        artifacts.get("goal_predictions") or [],
        artifacts.get("turn_ledgers") or [],
    )
    _render_lora_updates(artifacts.get("lora_updates") or [])


def _render_candidate_predictions(rows: list[dict[str, Any]]) -> None:
    st.write("Candidate World Predictions")
    if not rows:
        st.caption("No candidate predictions.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "candidate": row["candidate_index"],
                    "action": action_label(row.get("action")),
                    "source": row["source"],
                    "prediction": row["prediction"],
                }
                for row in rows
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def _render_judge_scores(rows: list[dict[str, Any]]) -> None:
    st.write("Reward Judge")
    if not rows:
        st.caption("No judge scores.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "candidate_prediction_id": row["candidate_prediction_id"],
                    "score": row["score"],
                    "error_tags": ", ".join(row.get("error_tags") or []),
                    "notes": row["notes"],
                }
                for row in rows
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def _render_rewards(rows: list[dict[str, Any]]) -> None:
    st.write("Rewards")
    if not rows:
        st.caption("No reward row.")
        return
    reward = _dict(rows[-1].get("reward"))
    cols = st.columns(5)
    cols[0].metric("Total", _metric_value(reward.get("total")))
    cols[1].metric("World", _metric_value(reward.get("world_score")))
    cols[2].metric("LP", _metric_value(reward.get("learning_progress")))
    cols[3].metric("Goal Delta", _metric_value(reward.get("goal_delta")))
    cols[4].metric("Bonus", _metric_value(reward.get("progress_bonus")))
    st.json(redacted_for_json(rows[-1]))


def _render_goal_memory(
    goal_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> None:
    st.write("Goal And Memory")
    if goal_rows:
        goal = _dict(goal_rows[-1].get("goal_prediction"))
        cols = st.columns(3)
        cols[0].metric("Steps", _metric_value(goal.get("steps_remaining")))
        cols[1].metric("Confidence", _metric_value(goal.get("confidence")))
        cols[2].metric("Subgoals", str(len(goal.get("subgoals") or [])))
        st.text_area(
            "Goal",
            value=str(goal.get("goal") or ""),
            height=90,
            disabled=True,
        )
        st.text_area(
            "Memory",
            value=str(goal_rows[-1].get("memory_document") or ""),
            height=240,
            disabled=True,
        )
    elif ledger_rows:
        st.text_area(
            "Memory",
            value=str(ledger_rows[-1].get("memory_document") or ""),
            height=240,
            disabled=True,
        )
    else:
        st.caption("No Goal or Memory rows.")


def _render_lora_updates(rows: list[dict[str, Any]]) -> None:
    st.write("Adapter Updates")
    if not rows:
        st.caption("No LoRA update attempts.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "update": row["update_index"],
                    "role": row["role"],
                    "status": row["status"],
                    "adapter": row["adapter_name"],
                    "path": row["adapter_path"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def _metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "-"
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _action6_marked_image(
    image: Any | None,
    action: Any,
) -> tuple[Any | None, str | None]:
    coordinates, error = _action6_grid_coordinates(action)
    if coordinates is None:
        return image, error
    if image is None:
        return image, None

    from PIL import ImageDraw

    marked = image.copy()
    width, height = marked.size
    x, y = coordinates
    center_x = _grid_coordinate_to_pixel(x, width)
    center_y = _grid_coordinate_to_pixel(y, height)
    radius = _action6_dot_radius(width, height)
    draw = ImageDraw.Draw(marked)

    outline_radius = radius + 1
    draw.ellipse(
        _ellipse_box(center_x, center_y, outline_radius, width, height),
        fill=(255, 255, 255),
    )
    draw.ellipse(
        _ellipse_box(center_x, center_y, radius, width, height),
        fill=(220, 0, 0),
    )
    return marked, None


def _action6_grid_coordinates(
    action: Any,
) -> tuple[tuple[int, int] | None, str | None]:
    payload = _dict(action)
    if _action_name(payload) != "ACTION6":
        return None, None

    data = _dict(payload.get("data"))
    x = _grid_coordinate(data.get("x"))
    y = _grid_coordinate(data.get("y"))
    if x is None or y is None:
        return None, "ACTION6 target marker skipped: missing valid x/y coordinates."
    return (x, y), None


def _action_name(action: dict[str, Any]) -> str:
    raw = str(action.get("action_id") or "")
    if raw.startswith("<GameAction.") and ":" in raw:
        return raw.removeprefix("<GameAction.").split(":", 1)[0]
    if raw.startswith("GameAction."):
        return raw.rsplit(".", 1)[-1]
    return raw


def _grid_coordinate(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        return None
    coordinate = int(numeric)
    if not 0 <= coordinate < ACTION6_GRID_SIZE:
        return None
    return coordinate


def _grid_coordinate_to_pixel(coordinate: int, extent: int) -> int:
    return min(
        extent - 1,
        int((coordinate + 0.5) * extent / ACTION6_GRID_SIZE),
    )


def _action6_dot_radius(width: int, height: int) -> int:
    return max(2, min(8, round(min(width, height) / 32)))


def _ellipse_box(
    center_x: int,
    center_y: int,
    radius: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, center_x - radius),
        max(0, center_y - radius),
        min(width - 1, center_x + radius),
        min(height - 1, center_y + radius),
    )


def _selected_index_from_session(
    states: list[dict[str, Any]],
    *,
    selected_state_id: Any,
) -> int | None:
    if selected_state_id is None:
        return None
    try:
        selected_id = int(selected_state_id)
    except (TypeError, ValueError):
        return None
    for index, state in enumerate(states):
        if int(state["id"]) == selected_id:
            return index
    return None


def _restore_dataframe_selection(
    key: str,
    selection_default: dict[str, dict[str, list[int]]],
) -> None:
    current = st.session_state.get(key)
    if _selected_rows(current):
        return
    st.session_state[key] = selection_default


def _selected_rows(value: Any) -> list[int]:
    if not isinstance(value, dict):
        return []
    selection = value.get("selection")
    if not isinstance(selection, dict):
        return []
    rows = selection.get("rows")
    if not isinstance(rows, list):
        return []
    return [int(row) for row in rows if isinstance(row, int)]
