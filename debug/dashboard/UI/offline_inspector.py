"""Offline memory inspection page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from debug.dashboard.memory_reader import (
    load_m_states,
    load_model_input_debug_records,
    matching_model_input_records,
)
from debug.dashboard.UI.turn_detail import render_selected_turn, select_turn

OFFLINE_RUN_FILTER_KEY = "offline_run_filter"
OFFLINE_GAME_FILTER_KEY = "offline_game_filter"


def render_offline_inspector(database_path: str) -> None:
    """Render the offline persisted-memory inspector."""

    try:
        states, model_input_records = _load_memory(database_path)
    except Exception as exc:
        st.error(str(exc))
        return

    if not states:
        st.info("No M states found in this database.")
        return

    filtered_states = _filter_states(states)
    if not filtered_states:
        st.info("No turns match the selected filters.")
        return

    st.header("Memory Turns")
    selected_state = select_turn(filtered_states, key="offline_turn_table")
    selected_model_inputs = matching_model_input_records(
        model_input_records,
        selected_state,
    )
    render_selected_turn(selected_state, selected_model_inputs)


@st.cache_data(show_spinner=False)
def _load_memory(
    database_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        load_m_states(database_path),
        load_model_input_debug_records(database_path),
    )


def _filter_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_ids = ["All", *sorted({state["run_id"] for state in states})]
    game_ids = ["All", *sorted({state["game_id"] for state in states})]

    filter_cols = st.columns(2)
    run_id = _select_filter_value(
        filter_cols[0],
        "Run",
        run_ids,
        key=OFFLINE_RUN_FILTER_KEY,
    )
    game_id = _select_filter_value(
        filter_cols[1],
        "Game",
        game_ids,
        key=OFFLINE_GAME_FILTER_KEY,
    )

    filtered = states
    if run_id != "All":
        filtered = [state for state in filtered if state["run_id"] == run_id]
    if game_id != "All":
        filtered = [state for state in filtered if state["game_id"] == game_id]
    return filtered


def _select_filter_value(column: Any, label: str, options: list[str], *, key: str) -> str:
    if st.session_state.get(key) not in options:
        st.session_state[key] = "All"
    return str(column.selectbox(label, options, key=key))
