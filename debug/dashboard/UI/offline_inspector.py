"""Offline memory inspection page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from debug.dashboard.memory_reader import (
    load_m_states,
    load_model_input_debug_records,
    matching_model_input_records,
)
from debug.dashboard.runner import (
    RUNNER_PLAYBACK_REQUEST_KEY,
    RUNTIME_RUNNER_KEY,
    RuntimeRunner,
)
from debug.dashboard.UI.sidebar import PAGE_KEY
from debug.dashboard.UI.turn_detail import render_selected_turn, select_turn
from debug.playback import PlaybackRequest

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
    _render_replay_control(states, selected_state)
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


def _render_replay_control(
    states: list[dict[str, Any]],
    selected_state: dict[str, Any],
) -> None:
    available, message = _playback_availability(states, selected_state)
    runner = st.session_state.get(RUNTIME_RUNNER_KEY)
    running = isinstance(runner, RuntimeRunner) and runner.is_running()
    disabled = running or not available

    if st.button("Replay in Runner", disabled=disabled):
        st.session_state[RUNNER_PLAYBACK_REQUEST_KEY] = PlaybackRequest(
            source_run_id=str(selected_state["run_id"]),
            game_id=str(selected_state["game_id"]),
            turn_id=int(selected_state["turn_id"]),
        )
        st.session_state[PAGE_KEY] = "Runner"
        st.rerun()

    if running:
        st.caption("Replay is unavailable while the runner subprocess is active.")
    elif not available:
        st.warning(message)
    else:
        st.caption(message)


def _playback_availability(
    states: list[dict[str, Any]],
    selected_state: dict[str, Any],
) -> tuple[bool, str]:
    run_id = str(selected_state["run_id"])
    game_id = str(selected_state["game_id"])
    target_turn = int(selected_state["turn_id"])
    if target_turn < 1:
        return False, "Replay requires a positive turn id."

    matching_turns = {
        int(state["turn_id"])
        for state in states
        if state["run_id"] == run_id and state["game_id"] == game_id
    }
    if target_turn not in matching_turns:
        return False, "Replay target row is no longer available."

    missing = [turn for turn in range(1, target_turn) if turn not in matching_turns]
    if missing:
        missing_text = ", ".join(str(turn) for turn in missing[:8])
        if len(missing) > 8:
            missing_text += ", ..."
        return (
            False,
            "Replay needs all prior M rows for this run/game; missing turn(s): "
            f"{missing_text}.",
        )

    if target_turn == 1:
        return True, "Replay will hand off at turn 1 without replaying prior turns."
    return True, f"Replay will use {target_turn - 1} prior turn(s), then hand off."
