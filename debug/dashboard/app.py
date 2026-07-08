"""Local Streamlit dashboard for running and inspecting FACE-OF-AGI."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from debug.dashboard.modal_snapshot import (
    DEFAULT_MODAL_DATABASE,
    DEFAULT_MODAL_RUN_FOLDER,
    DEFAULT_MODAL_SNAPSHOT,
    DEFAULT_MODAL_VOLUME,
)
from debug.dashboard.UI.live_play import render_live_play
from debug.dashboard.UI.offline_inspector import render_offline_inspector
from debug.dashboard.UI.runner_view import render_runner
from debug.dashboard.UI.scoring import render_scoring
from debug.dashboard.UI.sidebar import render_sidebar
from debug.dashboard.UI.style import (
    apply_dashboard_style,
    render_dashboard_header,
    resolve_dashboard_theme,
)
from debug.dashboard.UI.e2e_workshop import render_test_workshop

DEFAULT_DATABASE_FOLDER = "runs/kaggle-debug/runs"
DEFAULT_LOCAL_DATABASE = "runs/memory.sqlite"


def main() -> None:
    """Render the local debug dashboard."""

    args = _parse_args()
    page_title = "FACE-OF-AGI Modal Debug" if args.modal else "FACE-OF-AGI Debug"
    st.set_page_config(page_title=page_title, layout="wide")

    sidebar = render_sidebar(
        default_database=args.database,
        modal_enabled=args.modal,
        default_local_database=args.local_database,
        default_modal_volume=args.modal_volume,
        default_modal_database=args.modal_database,
        default_modal_run_folder=args.modal_run_folder,
        default_modal_snapshot=args.modal_snapshot,
    )
    theme = resolve_dashboard_theme()
    apply_dashboard_style()

    page = sidebar.page
    render_dashboard_header(page, theme=theme)
    if page == "Runner":
        render_runner(
            sidebar.local_database,
            modal_snapshot=sidebar.modal_snapshot,
        )
    elif page == "Test Workshop":
        render_test_workshop()
    elif page == "Live Play":
        render_live_play(
            sidebar.inspection_database,
            refresh_database=None,
            require_running_runtime=sidebar.modal_snapshot is None,
        )
    elif page == "Scoring":
        render_scoring(sidebar.database_folder)
    else:
        render_offline_inspector(sidebar.inspection_database)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--database", default=DEFAULT_DATABASE_FOLDER)
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--modal-volume", default=DEFAULT_MODAL_VOLUME)
    parser.add_argument("--modal-database", default=DEFAULT_MODAL_DATABASE)
    parser.add_argument("--modal-run-folder", default="")
    parser.add_argument("--modal-snapshot", default=DEFAULT_MODAL_SNAPSHOT)
    parser.add_argument("--local-database", default=DEFAULT_LOCAL_DATABASE)
    args, _ = parser.parse_known_args(sys.argv[1:])
    args.database = str(Path(args.database))
    args.local_database = str(Path(args.local_database))
    args.modal_snapshot = str(Path(args.modal_snapshot))
    if args.modal and not args.modal_run_folder:
        args.modal_run_folder = _current_git_commit_id()
    return args


def _current_git_commit_id() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return DEFAULT_MODAL_RUN_FOLDER
    return completed.stdout.strip() or DEFAULT_MODAL_RUN_FOLDER


if __name__ == "__main__":
    main()
