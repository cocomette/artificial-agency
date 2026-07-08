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
) -> None:
    """Render detail tabs for one selected M state."""

    _render_turn_heading(state)

    overview, model_inputs, raw = st.tabs(["Overview", "Models I/O", "Raw Data"])
    with overview:
        _render_overview(state)
    with model_inputs:
        render_model_inputs(model_input_records or [])
    with raw:
        _render_raw(state, model_input_records or [])


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
        render_image(
            observation_image(state.get("current_observation")),
            "Current observed frame",
        )
    with right:
        st.write("Allowed actions")
        st.json(
            [action_label(action) for action in control_mode.get("allowed_actions") or []]
        )


def _render_raw(
    state: dict[str, Any],
    model_input_records: list[dict[str, Any]],
) -> None:
    st.json(
        redacted_for_json(
            {
                "m_state": state,
                "matching_model_input_debug_records": model_input_records,
            }
        )
    )


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


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
