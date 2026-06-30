"""Live memory-following dashboard page."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from debug.dashboard.runner import RUNTIME_RUNNER_KEY, RuntimeRunner
from debug.dashboard.memory_reader import (
    latest_state,
    load_m_states,
)
from debug.dashboard.UI.turn_detail import render_turn_overview

LIVE_REFRESH_SECONDS = 2


def render_live_play(
    database_path: str,
    *,
    refresh_database: Callable[[], None] | None = None,
    require_running_runtime: bool = True,
) -> None:
    """Render the latest persisted turn for the current run/game."""

    if require_running_runtime and not _runtime_is_running():
        st.info("No runtime config is currently running.")
        return

    if hasattr(st, "fragment"):

        @st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
        def live_fragment() -> None:
            _render_live_body(
                database_path,
                refresh_database=refresh_database,
                require_running_runtime=require_running_runtime,
            )

        live_fragment()
        return

    _render_live_body(
        database_path,
        refresh_database=refresh_database,
        require_running_runtime=require_running_runtime,
    )


def _render_live_body(
    database_path: str,
    *,
    refresh_database: Callable[[], None] | None = None,
    require_running_runtime: bool = True,
) -> None:
    if require_running_runtime and not _runtime_is_running():
        st.info("No runtime config is currently running.")
        return

    try:
        if refresh_database is not None:
            refresh_database()
        states = load_m_states(database_path)
    except Exception as exc:
        st.error(str(exc))
        return

    if not states:
        st.info("No live run is currently persisted in this database.")
        return

    selected_state = latest_state(states)
    if selected_state is None:
        st.info("No live turn is available yet.")
        return

    render_turn_overview(selected_state)


def _runtime_is_running() -> bool:
    runner = st.session_state.get(RUNTIME_RUNNER_KEY)
    if not isinstance(runner, RuntimeRunner):
        return False
    runner.poll()
    return runner.is_running()
