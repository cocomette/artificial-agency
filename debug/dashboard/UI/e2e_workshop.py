"""E2E test workshop page for Streamlit."""

from __future__ import annotations

from pathlib import Path
import shlex

import streamlit as st

from debug.dashboard import workshop
from debug.dashboard.runner import RuntimeRunner, format_command

TEST_RUNNER_KEY = "test_workshop_runner"
TEST_SELECTION_KEY = "test_workshop_selection"
TEST_ARGS_KEY = "test_workshop_extra_args"
ACTIVE_TEST_KEY = "test_workshop_active_test"
ACTIVE_RESULT_LABEL_KEY = "test_workshop_active_result_label"
RUN_WAS_ACTIVE_KEY = "test_workshop_run_was_active"
PENDING_RESULT_SELECTION_KEY = "test_workshop_pending_result_selection"
RESULT_SELECTION_KEY = "test_workshop_result_selection"


def render_test_workshop() -> None:
    """Render E2E runner controls and generic result artifacts."""

    tests = workshop.list_e2e_tests()
    if not tests:
        st.error(f"No E2E scripts found in {workshop.DEFAULT_E2E_DIR}.")
        return

    st.header("Run Configuration")
    selected_name = _render_test_selector(tests)
    command = _build_command(selected_name)
    if command is not None:
        st.code(format_command(command), language="bash")
    _render_runner_controls(command, selected_name)

    st.divider()
    st.header("Output Artifacts")
    _render_result_artifacts(selected_name)


def _render_test_selector(tests: list[Path]) -> str:
    names = [path.name for path in tests]
    selected_name = str(st.selectbox("E2E test", names, key=TEST_SELECTION_KEY))
    st.text_input("Extra arguments", key=TEST_ARGS_KEY)
    return selected_name


def _build_command(selected_name: str) -> list[str] | None:
    raw_args = str(st.session_state.get(TEST_ARGS_KEY) or "").strip()
    try:
        extra_args = shlex.split(raw_args)
    except ValueError as exc:
        st.error(f"Could not parse extra arguments: {exc}")
        return None

    try:
        return workshop.build_e2e_command(selected_name, extra_args=extra_args)
    except ValueError as exc:
        st.error(str(exc))
        return None


def _render_runner_controls(command: list[str] | None, selected_name: str) -> None:
    runner = _get_runner()
    if runner is not None:
        runner.poll()

    running = runner is not None and runner.is_running()

    run_col, stop_col, clear_col = st.columns(3)
    if run_col.button(
        "RUN e2e test",
        disabled=command is None or running,
        use_container_width=True,
    ):
        assert command is not None
        st.session_state[ACTIVE_TEST_KEY] = selected_name
        st.session_state[ACTIVE_RESULT_LABEL_KEY] = _expected_result_label(
            selected_name,
            command,
        )
        st.session_state[RUN_WAS_ACTIVE_KEY] = True
        st.session_state[TEST_RUNNER_KEY] = RuntimeRunner.start(command)
        st.rerun()

    if stop_col.button("Stop", disabled=not running, use_container_width=True):
        assert runner is not None
        runner.stop()
        st.rerun()

    if clear_col.button(
        "Clear output",
        disabled=runner is None,
        use_container_width=True,
    ):
        assert runner is not None
        runner.clear_output()
        st.rerun()

    _render_runner_state(runner, selected_name)


def _render_status(runner: RuntimeRunner | None) -> None:
    status, exit_code, elapsed = _runner_status(runner)
    cols = st.columns(3)
    cols[0].metric("Status", status)
    cols[1].metric("Exit code", exit_code)
    cols[2].metric("Elapsed", elapsed)


def _render_runner_state(runner: RuntimeRunner | None, selected_name: str) -> None:
    if runner is not None and runner.is_running() and hasattr(st, "fragment"):

        @st.fragment(run_every="1s")
        def runner_fragment() -> None:
            _handle_run_completion(runner, selected_name)
            _render_status(runner)
            _output_box(runner)

        runner_fragment()
        return

    _handle_run_completion(runner, selected_name)
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


def _handle_run_completion(runner: RuntimeRunner | None, selected_name: str) -> None:
    if runner is None:
        st.session_state[RUN_WAS_ACTIVE_KEY] = False
        return

    return_code = runner.poll()
    if return_code is None:
        st.session_state[RUN_WAS_ACTIVE_KEY] = True
        return

    if not bool(st.session_state.get(RUN_WAS_ACTIVE_KEY)):
        return

    st.session_state[RUN_WAS_ACTIVE_KEY] = False
    active_label = str(st.session_state.get(ACTIVE_RESULT_LABEL_KEY) or "")
    if active_label:
        st.session_state[PENDING_RESULT_SELECTION_KEY] = active_label
    else:
        active_test = str(st.session_state.get(ACTIVE_TEST_KEY) or selected_name)
        _queue_default_result_selection(active_test)
    _rerun_app()


def _queue_default_result_selection(selected_name: str) -> None:
    default_dir = workshop.default_result_dir_for_test(selected_name)
    st.session_state[PENDING_RESULT_SELECTION_KEY] = _relative_to_repo(default_dir)


def _render_result_artifacts(selected_name: str) -> None:
    result_dirs = workshop.list_result_dirs()
    if not result_dirs:
        st.info("No E2E result folders found.")
        return

    labels = [_relative_to_repo(path) for path in result_dirs]
    default_label = _default_result_label(labels, selected_name)
    pending_label = st.session_state.pop(PENDING_RESULT_SELECTION_KEY, None)
    if pending_label in labels:
        st.session_state[RESULT_SELECTION_KEY] = pending_label
    elif st.session_state.get(RESULT_SELECTION_KEY) not in labels:
        st.session_state[RESULT_SELECTION_KEY] = default_label

    selected_label = str(
        st.selectbox(
            "Run output",
            labels,
            key=RESULT_SELECTION_KEY,
        )
    )
    selected_dir = result_dirs[labels.index(selected_label)]
    artifacts = workshop.collect_result_artifacts(selected_dir)

    cols = st.columns(2)
    cols[0].metric("Images", len(artifacts.images))
    cols[1].metric("JSON files", len(artifacts.json_files))

    for artifact in artifacts.images:
        st.subheader(artifact.title)
        st.image(str(artifact.path), caption=_relative_to_repo(artifact.path))

    for artifact in artifacts.json_files:
        st.subheader(artifact.title)
        if artifact.parse_error:
            st.caption(f"Invalid JSON: {artifact.parse_error}")
            st.code(artifact.content, language="text")
        else:
            st.json(artifact.data)


def _default_result_label(labels: list[str], selected_name: str) -> str:
    default_dir = workshop.default_result_dir_for_test(selected_name)
    default_label = _relative_to_repo(default_dir)
    if default_label in labels:
        return default_label
    return labels[0]


def _expected_result_label(selected_name: str, command: list[str]) -> str:
    output_dir = _output_dir_from_command(command)
    if output_dir is None:
        default_dir = workshop.default_result_dir_for_test(selected_name)
        return _relative_to_repo(default_dir)
    path = Path(output_dir)
    if not path.is_absolute():
        path = workshop.repo_root() / path
    return _relative_to_repo(path)


def _output_dir_from_command(command: list[str]) -> str | None:
    for index, value in enumerate(command):
        if value == "--output-dir" and index + 1 < len(command):
            return command[index + 1]
        if value.startswith("--output-dir="):
            return value.split("=", 1)[1]
    return None


def _get_runner() -> RuntimeRunner | None:
    runner = st.session_state.get(TEST_RUNNER_KEY)
    if isinstance(runner, RuntimeRunner):
        return runner
    return None


def _rerun_app() -> None:
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(workshop.repo_root()).as_posix()
    except ValueError:
        return str(path)
