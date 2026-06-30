"""Shared Streamlit sidebar controls for dashboard navigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from debug.dashboard.memory_reader import list_sqlite_database_files
from debug.dashboard.runner import clean_memory_database, format_command

PAGE_KEY = "dashboard_page"
DATABASE_KEY = "dashboard_database_path"
DATABASE_FILE_KEY = "dashboard_database_file"
CLEAR_RESULT_KEY = "dashboard_clear_memory_result"
PAGES = ("Runner", "Test Workshop", "Live Play", "Offline Inspector", "Scoring")


@dataclass(frozen=True)
class PageMetadata:
    """Small display payload for one sidebar destination."""

    title: str
    caption: str


@dataclass(frozen=True)
class SidebarState:
    """Current dashboard navigation and persistence targets."""

    page: str
    inspection_database: str
    local_database: str
    database_folder: str


PAGE_METADATA = {
    "Runner": PageMetadata("Runner", "Launch and edit saved runtime configs."),
    "Test Workshop": PageMetadata(
        "Test Workshop",
        "Run E2E checks and inspect artifacts.",
    ),
    "Live Play": PageMetadata("Live Play", "Follow the latest persisted turn."),
    "Offline Inspector": PageMetadata(
        "Offline Inspector",
        "Inspect historical SQLite turns.",
    ),
    "Scoring": PageMetadata(
        "Scoring",
        "Compare memory runs with human baseline stats.",
    ),
}


def render_sidebar(
    *,
    default_database: str,
) -> SidebarState:
    """Render common sidebar controls and return navigation state."""

    with st.sidebar:
        _ensure_page_state()
        _ensure_database_state(default_database=default_database)
        brand = "Debug Console"
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <span class="brand-kicker">FACE-OF-AGI</span>
                <strong>{brand}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-section-title">Menu</div>',
            unsafe_allow_html=True,
        )
        _render_navigation()

        page = str(st.session_state[PAGE_KEY])
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-section-title">{page} Controls</div>',
            unsafe_allow_html=True,
        )
        targets = _render_dynamic_controls(
            page,
            default_database,
        )
        local_database = targets.local_database
        inspection_database = targets.inspection_database
        database_folder = targets.database_folder

        if page in {"Live Play", "Offline Inspector", "Scoring"} and st.button(
            "Refresh memory view",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()
    return SidebarState(
        page=page,
        inspection_database=inspection_database,
        local_database=local_database,
        database_folder=database_folder,
    )


def _ensure_page_state() -> None:
    page = st.session_state.get(PAGE_KEY)
    if page not in PAGES:
        st.session_state[PAGE_KEY] = "Runner"


def _ensure_database_state(
    *,
    default_database: str,
) -> None:
    st.session_state.setdefault(DATABASE_KEY, default_database)


@dataclass(frozen=True)
class LocalTargets:
    """Resolved local database folder plus concrete SQLite targets."""

    database_folder: str
    inspection_database: str
    local_database: str


def _render_navigation() -> None:
    for page in PAGES:
        metadata = PAGE_METADATA[page]
        active = st.session_state[PAGE_KEY] == page
        if st.button(
            metadata.title,
            key=f"dashboard_nav_{page}",
            disabled=active,
            use_container_width=True,
        ):
            st.session_state[PAGE_KEY] = page
            st.rerun()
        if active:
            st.caption(metadata.caption)


def _render_dynamic_controls(
    page: str,
    default_database: str,
) -> LocalTargets:
    if DATABASE_KEY not in st.session_state:
        st.session_state[DATABASE_KEY] = default_database
    if page == "Test Workshop":
        st.caption("E2E artifacts are read from `runs/`.")
        if st.button("Refresh test results", use_container_width=True):
            st.rerun()
        return _current_targets()

    database_folder = _render_database_folder_input()
    local_database = _local_run_database(database_folder)
    if page == "Runner":
        st.caption("Local runs launched from this dashboard write to `memory.sqlite`.")
        _render_clear_memory_control(local_database)
        return LocalTargets(
            database_folder=database_folder,
            inspection_database=local_database,
            local_database=local_database,
        )
    if page in {"Live Play", "Offline Inspector"}:
        inspection_database = _render_database_file_selector(database_folder)
        return LocalTargets(
            database_folder=database_folder,
            inspection_database=inspection_database,
            local_database=local_database,
        )

    st.caption("Scoring reads all SQLite files directly inside this folder.")
    return LocalTargets(
        database_folder=database_folder,
        inspection_database="",
        local_database=local_database,
    )


def _current_targets() -> LocalTargets:
    database_folder = str(st.session_state[DATABASE_KEY])
    local_database = _local_run_database(database_folder)
    return LocalTargets(
        database_folder=database_folder,
        inspection_database=local_database,
        local_database=local_database,
    )


def _render_database_folder_input() -> str:
    return str(st.text_input("Memory database folder", key=DATABASE_KEY))


def _render_database_file_selector(database_folder: str) -> str:
    try:
        database_files = list_sqlite_database_files(database_folder)
    except (FileNotFoundError, NotADirectoryError) as exc:
        st.warning(str(exc))
        return ""
    if not database_files:
        st.warning(f"No SQLite memory files found in `{database_folder}`.")
        return ""

    options = [path.name for path in database_files]
    if st.session_state.get(DATABASE_FILE_KEY) not in options:
        st.session_state[DATABASE_FILE_KEY] = options[0]
    selected_name = str(
        st.selectbox(
            "Memory database",
            options,
            key=DATABASE_FILE_KEY,
        )
    )
    return str(Path(database_folder) / selected_name)


def _local_run_database(database_folder: str) -> str:
    return str(Path(database_folder) / "memory.sqlite")


def _render_clear_memory_control(database_path: str) -> None:
    if st.button(
        "Clear memory database",
        use_container_width=True,
    ):
        _confirm_clear_memory_dialog(database_path)

    result = st.session_state.get(CLEAR_RESULT_KEY)
    if result is None:
        st.caption(
            "Clears rows with runtime clean-db; resets stale local SQLite files."
        )
        return

    if result.return_code == 0:
        st.success(result.output.strip() or "Memory database cleared.")
    else:
        st.error(result.output.strip() or f"Clear failed with {result.return_code}.")
    st.caption(format_command(result.command))


@st.dialog("Clear Memory Database")
def _confirm_clear_memory_dialog(database_path: str) -> None:
    st.write(f"Clear persisted memory rows from `{database_path}`?")
    st.caption(
        "Stale disposable SQLite files are reset when their schema is obsolete."
    )
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Clear memory database", type="primary"):
        st.session_state[CLEAR_RESULT_KEY] = clean_memory_database(database_path)
        st.cache_data.clear()
        st.rerun()
    if cancel_col.button("Cancel"):
        st.rerun()
