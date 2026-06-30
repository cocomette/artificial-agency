"""Local Streamlit dashboard for running and inspecting FACE-OF-AGI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

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
    st.set_page_config(page_title="FACE-OF-AGI Debug", layout="wide")

    sidebar = render_sidebar(
        default_database=args.database,
    )
    theme = resolve_dashboard_theme()
    apply_dashboard_style()

    page = sidebar.page
    render_dashboard_header(page, theme=theme)
    if page == "Runner":
        render_runner(sidebar.local_database)
    elif page == "Test Workshop":
        render_test_workshop()
    elif page == "Live Play":
        render_live_play(
            sidebar.inspection_database,
            refresh_database=None,
            require_running_runtime=True,
        )
    elif page == "Scoring":
        render_scoring(sidebar.database_folder)
    else:
        render_offline_inspector(sidebar.inspection_database)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--database", default=DEFAULT_DATABASE_FOLDER)
    args, _ = parser.parse_known_args(sys.argv[1:])
    args.database = str(Path(args.database))
    return args


if __name__ == "__main__":
    main()
