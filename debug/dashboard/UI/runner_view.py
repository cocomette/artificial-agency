"""Runtime runner page for launching saved configs from Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from debug.dashboard import config_manager
from debug.dashboard.modal_snapshot import (
    ModalPullError,
    ModalSnapshotConfig,
    volume_relative_path,
)
from debug.dashboard.UI.config_editor import render_config_editor
from debug.dashboard.runner import (
    CommandResult,
    GAME_CATALOG_PATH,
    RUNTIME_RUNNER_KEY,
    RUNNER_PLAYBACK_REQUEST_KEY,
    RuntimeRunner,
    build_modal_run_command,
    build_run_command,
    format_command,
    pull_game_list,
)
from debug.playback import PlaybackRequest

RUNNER_KEY = RUNTIME_RUNNER_KEY
RUNNER_CONFIG_KEY = "runner_config"
RUNNER_SELECTED_CONFIG_KEY = "runner_selected_config"
PENDING_CONFIG_KEY = "runner_pending_config"
GAME_LIST_RESULT_KEY = "runner_game_list_result"
MODAL_LIVE_COMMIT_SECONDS_KEY = "runner_modal_live_commit_seconds"
MODAL_TIMING_KEY = "runner_modal_timing"


def render_runner(
    database_path: str,
    *,
    modal_snapshot: ModalSnapshotConfig | None = None,
) -> None:
    """Render runtime process controls and live output."""

    _render_runner_header()

    config_paths = config_manager.list_config_files()
    if not config_paths:
        st.error(f"No YAML configs found in {config_manager.DEFAULT_CONFIG_DIR}.")
        return

    config_names = [config_manager.config_label(path) for path in config_paths]
    selected_name = _render_config_selector(config_names)
    selected_path = config_manager.safe_config_path(selected_name)
    selected_config_path = _relative_to_repo(selected_path)

    with st.expander("Edit selected config", expanded=False):
        render_config_editor(
            selected_name,
            selected_path,
            saved_config_key=PENDING_CONFIG_KEY,
        )

    keep_all_m_states = bool(st.checkbox("Keep all M states", value=True))
    playback_request = _render_playback_request()
    command = build_run_command(
        selected_config_path,
        database_path,
        keep_all_m_states=keep_all_m_states,
        playback_request=playback_request,
    )
    modal_command = None
    if modal_snapshot is not None:
        try:
            modal_command = _render_modal_run_options(
                selected_config_path,
                modal_snapshot,
                playback_request=playback_request,
            )
        except ModalPullError as exc:
            st.error(str(exc))

    runner = _get_runner()
    if runner is not None:
        runner.poll()

    running = runner is not None and runner.is_running()
    if modal_command is None:
        st.code(format_command(command), language="bash")
        run_col, stop_col, clear_col = st.columns(3)
        if run_col.button("RUN config", disabled=running):
            runner = RuntimeRunner.start(command)
            st.session_state[RUNNER_KEY] = runner
            st.rerun()
    else:
        command_to_start = _render_commands(
            command,
            modal_command,
            running=running,
        )
        if command_to_start is not None:
            runner = RuntimeRunner.start(command_to_start)
            st.session_state[RUNNER_KEY] = runner
            st.rerun()
        stop_col, clear_col = st.columns(2)

    if stop_col.button("Stop", disabled=not running):
        assert runner is not None
        runner.stop()
        st.rerun()

    if clear_col.button("Clear output", disabled=runner is None):
        assert runner is not None
        runner.clear_output()
        st.rerun()

    _render_runner_state(runner)


def _render_config_selector(config_names: list[str]) -> str:
    pending_name = st.session_state.pop(PENDING_CONFIG_KEY, None)
    widget_name = st.session_state.get(RUNNER_CONFIG_KEY)

    if pending_name in config_names:
        st.session_state[RUNNER_CONFIG_KEY] = pending_name
        selected_name = str(st.selectbox("Config", config_names, key=RUNNER_CONFIG_KEY))
        st.session_state[RUNNER_SELECTED_CONFIG_KEY] = selected_name
        return selected_name

    if widget_name in config_names:
        selected_name = str(st.selectbox("Config", config_names, key=RUNNER_CONFIG_KEY))
        st.session_state[RUNNER_SELECTED_CONFIG_KEY] = selected_name
        return selected_name

    st.session_state.pop(RUNNER_CONFIG_KEY, None)
    default_name = _resolve_config_selection(
        config_names,
        stored_name=st.session_state.get(RUNNER_SELECTED_CONFIG_KEY),
        pending_name=None,
    )

    selected_name = str(
        st.selectbox(
            "Config",
            config_names,
            index=config_names.index(default_name),
            key=RUNNER_CONFIG_KEY,
        )
    )
    st.session_state[RUNNER_SELECTED_CONFIG_KEY] = selected_name
    return selected_name


def _resolve_config_selection(
    config_names: list[str],
    *,
    stored_name: object,
    pending_name: object,
) -> str:
    if pending_name in config_names:
        return str(pending_name)
    if stored_name in config_names:
        return str(stored_name)
    return config_names[0]


def _render_modal_run_options(
    selected_config_path: Path,
    modal_snapshot: ModalSnapshotConfig,
    *,
    playback_request: PlaybackRequest | None,
) -> list[str]:
    with st.expander("Modal run options", expanded=False):
        remote_database = volume_relative_path(modal_snapshot.remote_database)
        st.text_input(
            "Remote database name",
            value=remote_database,
            disabled=True,
        )
        live_commit_seconds = int(
            st.number_input(
                "Live commit seconds",
                min_value=0,
                max_value=3600,
                step=5,
                value=30,
                key=MODAL_LIVE_COMMIT_SECONDS_KEY,
            )
        )
        timing = bool(st.checkbox("Write timing JSONL", key=MODAL_TIMING_KEY))
    return build_modal_run_command(
        selected_config_path,
        database_name=remote_database,
        live_commit_seconds=live_commit_seconds,
        timing=timing,
        playback_request=playback_request,
    )


def _render_playback_request() -> PlaybackRequest | None:
    request = st.session_state.get(RUNNER_PLAYBACK_REQUEST_KEY)
    if not isinstance(request, PlaybackRequest):
        return None

    st.info(
        "Playback armed: "
        f"run `{request.source_run_id}`, game `{request.game_id}`, "
        f"handoff turn `{request.turn_id}`. "
        "The next run will replay prior turns before live control resumes."
    )
    if st.button("Clear playback request"):
        st.session_state.pop(RUNNER_PLAYBACK_REQUEST_KEY, None)
        st.rerun()
    return request


def _render_commands(
    local_command: list[str],
    modal_command: list[str],
    *,
    running: bool,
) -> list[str] | None:
    modal_tab, local_tab = st.tabs(["Modal command", "Local command"])
    with modal_tab:
        st.code(format_command(modal_command), language="bash")
        if st.button("RUN config", key="runner_run_modal", disabled=running):
            return modal_command
    with local_tab:
        st.code(format_command(local_command), language="bash")
        if st.button("RUN config", key="runner_run_local", disabled=running):
            return local_command
    return None


def _render_runner_header() -> None:
    title_col, _, action_col = st.columns(
        [0.7, 0.16, 0.14],
        vertical_alignment="center",
    )
    title_col.markdown(
        '<h2 style="margin: 0; line-height: 1.2;">Run Configuration</h2>',
        unsafe_allow_html=True,
    )
    if action_col.button(_game_list_button_label(), use_container_width=True):
        st.session_state[GAME_LIST_RESULT_KEY] = pull_game_list()
        st.rerun()
    _render_game_list_result()


def _game_list_button_label() -> str:
    if _game_catalog_exists():
        return "✅ Pull game list"
    return "❌ Pull game list"


def _game_catalog_exists() -> bool:
    return (config_manager.repo_root() / GAME_CATALOG_PATH).exists()


def _render_game_list_result() -> None:
    result = st.session_state.get(GAME_LIST_RESULT_KEY)
    if not isinstance(result, CommandResult):
        return

    output = result.output.strip()
    if result.return_code == 0:
        st.success("Game list pulled.")
    else:
        st.error(f"Pull game list failed with {result.return_code}.")
    if output:
        with st.expander("Game list output", expanded=result.return_code != 0):
            st.code(output, language="text")


def _render_status(runner: RuntimeRunner | None) -> None:
    status, exit_code, elapsed = _runner_status(runner)
    cols = st.columns(3)
    cols[0].metric("Status", status)
    cols[1].metric("Exit code", exit_code)
    cols[2].metric("Elapsed", elapsed)


def _render_runner_state(runner: RuntimeRunner | None) -> None:
    if runner is not None and runner.is_running() and hasattr(st, "fragment"):

        @st.fragment(run_every="1s")
        def runner_fragment() -> None:
            runner.poll()
            _render_status(runner)
            _output_box(runner)

        runner_fragment()
        return

    if runner is not None:
        runner.poll()
    _render_status(runner)
    _output_box(runner)


def _output_box(runner: RuntimeRunner | None) -> None:
    output = "".join(runner.output) if runner is not None else ""
    st.code(output, language="text")


def _runner_status(runner: RuntimeRunner | None) -> tuple[str, str, str]:
    if runner is None:
        return "Idle", "-", "-"

    return_code = runner.poll()
    elapsed = f"{runner.elapsed_seconds():.1f}s"
    if return_code is None:
        return "Running", "-", elapsed
    return "Exited", str(return_code), elapsed


def _get_runner() -> RuntimeRunner | None:
    runner = st.session_state.get(RUNNER_KEY)
    if isinstance(runner, RuntimeRunner):
        return runner
    return None


def _relative_to_repo(path: Path) -> Path:
    try:
        return path.relative_to(config_manager.repo_root())
    except ValueError:
        return path
