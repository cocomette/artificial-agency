"""Shared Streamlit sidebar controls for dashboard navigation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

from debug.dashboard.modal_snapshot import (
    DEFAULT_MODAL_DATABASE_PATTERN,
    ModalPullError,
    ModalSnapshotConfig,
    pull_modal_sqlite_snapshots,
)
from debug.dashboard.memory_reader import list_sqlite_database_files
from debug.dashboard.runner import clean_memory_database, format_command

PAGE_KEY = "dashboard_page"
DATABASE_KEY = "dashboard_database_path"
DATABASE_FILE_KEY = "dashboard_database_file"
CLEAR_RESULT_KEY = "dashboard_clear_memory_result"
MODAL_LOCAL_DATABASE_KEY = "dashboard_modal_local_database_path"
MODAL_PULL_STATUS_KEY = "dashboard_modal_pull_status"
MODAL_REMOTE_DATABASE_KEY = "dashboard_modal_database_name"
MODAL_DATABASE_PATTERN_KEY = "dashboard_modal_database_pattern"
MODAL_RUN_FOLDER_KEY = "dashboard_modal_run_folder"
MODAL_SNAPSHOT_KEY = "dashboard_modal_snapshot"
MODAL_VOLUME_KEY = "dashboard_modal_volume"
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
    modal_snapshot: ModalSnapshotConfig | None = None


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
    modal_enabled: bool = False,
    default_local_database: str | None = None,
    default_modal_volume: str = "",
    default_modal_database: str = "",
    default_modal_run_folder: str = "",
    default_modal_snapshot: str = "",
) -> SidebarState:
    """Render common sidebar controls and return navigation state."""

    with st.sidebar:
        _ensure_page_state()
        _ensure_database_state(
            default_database=default_database,
            modal_enabled=modal_enabled,
            default_local_database=default_local_database or default_database,
            default_modal_volume=default_modal_volume,
            default_modal_database=default_modal_database,
            default_modal_run_folder=default_modal_run_folder,
            default_modal_snapshot=default_modal_snapshot,
        )
        brand = "Modal Debug Console" if modal_enabled else "Debug Console"
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
            modal_enabled=modal_enabled,
        )
        local_database = targets.local_database
        inspection_database = targets.inspection_database
        database_folder = targets.database_folder
        modal_snapshot = None

        if modal_enabled:
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sidebar-section-title">Modal Snapshot</div>',
                unsafe_allow_html=True,
            )
            modal_snapshot = _render_modal_snapshot_controls(
                database_folder=database_folder,
            )

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
        modal_snapshot=modal_snapshot,
    )


def _ensure_page_state() -> None:
    page = st.session_state.get(PAGE_KEY)
    if page not in PAGES:
        st.session_state[PAGE_KEY] = "Runner"


def _ensure_database_state(
    *,
    default_database: str,
    modal_enabled: bool,
    default_local_database: str,
    default_modal_volume: str,
    default_modal_database: str,
    default_modal_run_folder: str,
    default_modal_snapshot: str,
) -> None:
    if modal_enabled:
        default_modal_folder = _default_modal_database_folder(
            default_modal_snapshot=default_modal_snapshot,
        )
        st.session_state.setdefault(DATABASE_KEY, default_modal_folder)
        st.session_state.setdefault(MODAL_LOCAL_DATABASE_KEY, default_local_database)
        st.session_state.setdefault(MODAL_VOLUME_KEY, default_modal_volume)
        st.session_state.setdefault(MODAL_REMOTE_DATABASE_KEY, default_modal_database)
        st.session_state.setdefault(
            MODAL_DATABASE_PATTERN_KEY,
            DEFAULT_MODAL_DATABASE_PATTERN,
        )
        st.session_state.setdefault(MODAL_RUN_FOLDER_KEY, default_modal_run_folder)
        st.session_state.setdefault(MODAL_SNAPSHOT_KEY, default_modal_snapshot)
    else:
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
    *,
    modal_enabled: bool,
) -> LocalTargets:
    if DATABASE_KEY not in st.session_state:
        st.session_state[DATABASE_KEY] = default_database
    if page == "Test Workshop":
        st.caption("E2E artifacts are read from `runs/`.")
        if st.button("Refresh test results", use_container_width=True):
            st.rerun()
        return _current_targets(modal_enabled)

    if modal_enabled:
        if page == "Runner":
            database_path = str(
                st.text_input(
                    "Local run database",
                    key=MODAL_LOCAL_DATABASE_KEY,
                )
            )
            st.caption("Local runs launched from this dashboard write here.")
            _render_clear_memory_control(database_path)
            return LocalTargets(
                database_folder=str(Path(database_path).parent),
                inspection_database=database_path,
                local_database=database_path,
            )
        if page == "Scoring":
            database_folder = _render_database_folder_input()
            st.caption("Scoring reads local SQLite files from this folder.")
            return LocalTargets(
                database_folder=database_folder,
                inspection_database="",
                local_database=_local_run_database(database_folder),
            )

        database_folder = _render_database_folder_input()
        inspection_database = _render_database_file_selector(database_folder)
        st.caption("Pull Modal snapshots into this folder, then choose a local file.")
        return LocalTargets(
            database_folder=database_folder,
            inspection_database=inspection_database,
            local_database=_local_run_database(database_folder),
        )

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


def _current_targets(modal_enabled: bool) -> LocalTargets:
    current_database = _current_local_database(modal_enabled)
    if modal_enabled:
        return LocalTargets(
            database_folder=str(Path(current_database).parent),
            inspection_database=current_database,
            local_database=current_database,
        )
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


def _current_local_database(modal_enabled: bool) -> str:
    key = MODAL_LOCAL_DATABASE_KEY if modal_enabled else DATABASE_KEY
    return str(st.session_state[key])


def _render_modal_snapshot_controls(*, database_folder: str) -> ModalSnapshotConfig:
    volume_name = str(st.text_input("Modal volume", key=MODAL_VOLUME_KEY))
    run_folder = str(
        st.text_input(
            "Modal run folder / commit id",
            key=MODAL_RUN_FOLDER_KEY,
        )
    )
    database_name, database_pattern = _render_modal_database_control()
    local_snapshot = str(Path(database_folder) / database_name)
    config = ModalSnapshotConfig(
        volume_name=volume_name,
        run_folder=run_folder,
        database_name=database_name,
        database_pattern=database_pattern,
        local_snapshot=local_snapshot,
    )
    st.caption(f"Remote folder: `/vol/runs/{run_folder.strip().strip('/')}`")
    st.caption(f"Local folder: `{database_folder}`")
    if st.button("Pull snapshot now", use_container_width=True):
        try:
            pull_dashboard_modal_snapshot(config)
        except ModalPullError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    _render_pull_status()
    return config


def _render_modal_database_control() -> tuple[str, str]:
    pattern = str(
        st.text_input(
            "Run database regex",
            key=MODAL_DATABASE_PATTERN_KEY,
        )
    )
    return str(st.session_state[MODAL_REMOTE_DATABASE_KEY]), pattern


def pull_dashboard_modal_snapshot(config: ModalSnapshotConfig) -> None:
    """Pull matching Modal SQLite snapshots and update Streamlit state."""

    local_folder = Path(config.local_snapshot).parent
    try:
        pulled = pull_modal_sqlite_snapshots(
            volume_name=config.volume_name,
            run_folder=config.run_folder,
            pattern=config.database_pattern,
            local_folder=local_folder,
        )
    except ModalPullError as exc:
        _record_pull_status("error", str(exc))
        if not Path(config.local_database).exists():
            raise
        st.warning(f"Modal pull failed; showing the last local snapshot. {exc}")
        return

    st.cache_data.clear()
    _record_pull_status(
        "ok",
        f"Pulled {len(pulled)} Modal databases to {local_folder}; "
        "choose one from the local memory database selector.",
    )


def _default_modal_database_folder(
    *,
    default_modal_snapshot: str,
) -> str:
    return str(Path(default_modal_snapshot).parent)


def _record_pull_status(kind: str, message: str) -> None:
    st.session_state[MODAL_PULL_STATUS_KEY] = {
        "kind": kind,
        "message": message,
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def _render_pull_status() -> None:
    status = st.session_state.get(MODAL_PULL_STATUS_KEY)
    if not isinstance(status, dict):
        st.caption("No Modal snapshot has been pulled yet.")
        return

    message = f"{status.get('time', '--:--:--')} - {status.get('message', '')}"
    if status.get("kind") == "ok":
        st.success(message)
    else:
        st.error(message)


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
