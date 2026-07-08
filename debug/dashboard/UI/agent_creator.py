"""Compact agent-creator batch inspection UI."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import streamlit as st

from debug.dashboard.agent_creator_reader import load_agent_creator_inspection
from face_of_agi.runtime.agent_creator_paths import latest_agent_creator_database_path

AGENT_CREATOR_DATABASE_KEY = "offline_agent_creator_database"
AGENT_CREATOR_DATABASE_SOURCE_KEY = "offline_agent_creator_database_source"
AGENT_CREATOR_DATABASE_DEFAULT_KEY = "offline_agent_creator_database_default"
AGENT_CREATOR_BATCH_TABLE_KEY = "offline_agent_creator_batch_table"
AGENT_CREATOR_TOOL_TABLE_KEY = "offline_agent_creator_tool_table"
AGENT_CREATOR_REVISION_TABLE_KEY = "offline_agent_creator_revision_table"
AGENT_CREATOR_ROLE_TABLE_KEY = "offline_agent_creator_role_table"
AGENT_CREATOR_ROLE_HISTORY_TABLE_KEY = "offline_agent_creator_role_history_table"
DEFAULT_AGENT_CREATOR_DATABASE = "data/agent_creator_01.sqlite"


def render_agent_creator_inspector(memory_database_path: str) -> None:
    """Render batch-level agent creator changes."""

    database_path = _render_database_input(memory_database_path)
    if not database_path:
        return

    try:
        data = _load_agent_creator(database_path)
    except FileNotFoundError:
        st.info(f"No agent creator database found at `{database_path}`.")
        return
    except sqlite3.OperationalError as exc:
        st.error(f"`{database_path}` does not look like an agent creator database.")
        st.caption(str(exc))
        return
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not read agent creator rows from `{database_path}`.")
        st.caption(str(exc))
        return

    batch_tab, roles_tab = st.tabs(["Batches", "Roles"])
    with batch_tab:
        _render_batches(data, database_path)
    with roles_tab:
        _render_roles(data)


@st.cache_data(show_spinner=False)
def _load_agent_creator(database_path: str) -> dict[str, Any]:
    return load_agent_creator_inspection(database_path)


def _render_database_input(memory_database_path: str) -> str:
    st.subheader("Agent Creator")
    default_path = _default_database_path(memory_database_path)
    source_path = str(Path(memory_database_path))
    if (
        AGENT_CREATOR_DATABASE_KEY not in st.session_state
        or st.session_state.get(AGENT_CREATOR_DATABASE_SOURCE_KEY) != source_path
    ):
        st.session_state[AGENT_CREATOR_DATABASE_KEY] = default_path
        st.session_state[AGENT_CREATOR_DATABASE_SOURCE_KEY] = source_path
    st.session_state[AGENT_CREATOR_DATABASE_DEFAULT_KEY] = default_path

    input_col, button_col = st.columns([0.82, 0.18])
    database_path = str(
        input_col.text_input(
            "Agent creator database",
            key=AGENT_CREATOR_DATABASE_KEY,
        )
    ).strip()
    button_col.button(
        "Use latest",
        on_click=_use_latest_agent_creator_database,
        args=(default_path,),
        width="stretch",
    )
    return database_path


def _use_latest_agent_creator_database(default_path: str) -> None:
    st.session_state[AGENT_CREATOR_DATABASE_KEY] = default_path


def _render_batches(data: dict[str, Any], database_path: str) -> None:
    runs = list(data["runs"])
    if not runs:
        _render_no_batch_runs(data, database_path)
        return

    selected_run = _select_batch(data, key=AGENT_CREATOR_BATCH_TABLE_KEY)
    _render_batch_details(data, selected_run)


def _render_no_batch_runs(data: dict[str, Any], database_path: str) -> None:
    requests = list(data["requests"])
    revisions = list(data["role_revisions"])
    request_statuses = Counter(str(request["status"]) for request in requests)
    active_roles = {
        str(revision["role_name"])
        for revision in revisions
        if revision.get("active") and revision.get("publication_status") == "complete"
    }

    st.info("No agent creator batch runs found in this database.")
    st.caption(f"Selected database: `{database_path}`")
    cols = st.columns(3)
    cols[0].metric("Requests", str(len(requests)))
    cols[1].metric("Queued", str(request_statuses.get("queued", 0)))
    cols[2].metric("Active roles", str(len(active_roles)))
    if requests:
        st.caption(
            "Requests are present, but no batch has been claimed or processed yet."
        )
    elif revisions:
        st.caption(
            "This database has role revisions but no creator batch history. "
            "It is a seed or blessed-role database, not a processed run artifact."
        )
    _render_batch_database_suggestions(database_path)


def _render_batch_database_suggestions(database_path: str) -> None:
    candidates = _nearby_agent_creator_databases_with_runs(database_path)
    if not candidates:
        return

    best = candidates[0]
    st.warning(
        "Found agent creator batch history in another pulled database:"
        f" `{best['path']}`"
    )
    if st.button(
        "Use found batch database",
        on_click=_use_latest_agent_creator_database,
        args=(str(best["path"]),),
    ):
        return
    with st.expander("Other nearby agent creator databases"):
        table = pd.DataFrame(candidates)
        st.dataframe(table, width="stretch", hide_index=True)


def _nearby_agent_creator_databases_with_runs(database_path: str) -> list[dict[str, Any]]:
    path = Path(database_path)
    search_roots = _nearby_agent_creator_search_roots(path)
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("agent_creator*.sqlite")):
            resolved = candidate.resolve()
            if resolved in seen or resolved == path.resolve():
                continue
            seen.add(resolved)
            run_count = _creator_run_count(candidate)
            if run_count <= 0:
                continue
            candidates.append(
                {
                    "path": str(candidate),
                    "batch_runs": run_count,
                    "modified": candidate.stat().st_mtime,
                }
            )
    return sorted(
        candidates,
        key=lambda item: (-int(item["batch_runs"]), -float(item["modified"])),
    )


def _nearby_agent_creator_search_roots(path: Path) -> tuple[Path, ...]:
    directory = path if path.suffix == "" else path.parent
    roots = [directory]
    if directory.name == "runs":
        roots.append(directory.parent)
    if directory.parent.name == "kaggle-debug":
        roots.append(directory.parent)
    return tuple(dict.fromkeys(roots))


def _creator_run_count(database_path: Path) -> int:
    try:
        uri = f"file:{database_path.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            return int(
                connection.execute("SELECT count(*) FROM agent_creator_runs").fetchone()[
                    0
                ]
            )
    except sqlite3.Error:
        return 0


def _default_database_path(memory_database_path: str) -> str:
    path = Path(memory_database_path)
    directory = path if path.suffix == "" else path.parent
    parent_candidate = directory.parent / "agent_creator.sqlite"
    if directory.name == "runs" and parent_candidate.exists():
        return str(parent_candidate)
    latest = latest_agent_creator_database_path(directory)
    if latest is not None:
        return str(latest)
    if directory.parts:
        return str(directory / "agent_creator_01.sqlite")
    return DEFAULT_AGENT_CREATOR_DATABASE


def _select_batch(data: dict[str, Any], *, key: str) -> dict[str, Any]:
    runs = list(data["runs"])
    table = pd.DataFrame([_batch_summary(data, run) for run in runs])
    selected_run_key = f"{key}_selected_run_id"
    selected_index = _selected_index(
        runs,
        id_key="id",
        selected_id=st.session_state.get(selected_run_key),
    )
    if selected_index is None:
        selected_index = len(runs) - 1
        st.session_state[selected_run_key] = int(runs[selected_index]["id"])

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
        selected_run = runs[selected_rows[0]]
        st.session_state[selected_run_key] = int(selected_run["id"])
        return selected_run
    return runs[selected_index]


def _render_batch_details(data: dict[str, Any], run: dict[str, Any]) -> None:
    tool_calls = _tool_calls_for_run(data, run)
    revisions = _revisions_for_run(data, run)
    requests = _requests_for_run(data, run)

    st.subheader(f"Batch {run['id']}")
    cols = st.columns(4)
    cols[0].metric("Status", str(run["status"]))
    cols[1].metric("Tool calls", f"{len(tool_calls)}/{run['max_tool_calls']}")
    cols[2].metric("Role revisions", str(len(revisions)))
    cols[3].metric("Requests", str(len(requests)))
    if run.get("error"):
        st.error(str(run["error"]))

    tool_tab, revisions_tab, requests_tab = st.tabs(
        ["Tool Calls", "Role Revisions", "Requests"]
    )
    with tool_tab:
        _render_tool_calls(
            tool_calls,
            key=f"{AGENT_CREATOR_TOOL_TABLE_KEY}_{run['id']}",
        )
    with revisions_tab:
        _render_role_revisions(
            revisions,
            key=f"{AGENT_CREATOR_REVISION_TABLE_KEY}_{run['id']}",
        )
    with requests_tab:
        _render_requests(requests)


def _render_tool_calls(tool_calls: list[dict[str, Any]], *, key: str) -> None:
    if not tool_calls:
        st.info("No tool calls recorded for this batch.")
        return

    table = pd.DataFrame([_tool_call_summary(call) for call in tool_calls])
    selected = _select_detail_row(
        tool_calls,
        table,
        key=key,
        id_key="id",
    )
    if selected is None:
        return

    st.write("Selected tool call")
    cols = st.columns(4)
    cols[0].metric("Operation", str(selected["tool_name"]))
    cols[1].metric("Role", str(selected["role_name"]))
    cols[2].metric("Status", str(selected["status"]))
    cols[3].metric("Call", str(selected["call_index"]))
    if selected.get("message"):
        st.caption(str(selected["message"]))
    _render_tool_guidance(selected)


def _render_tool_guidance(call: dict[str, Any]) -> None:
    arguments = _dict(call.get("arguments"))
    tool_name = str(call.get("tool_name"))
    fields = {
        "add": ("instruction_guidance", "meta_description"),
        "update": ("identified_failures", "meta_description"),
        "delete": (),
    }.get(tool_name, ())
    for field in fields:
        value = arguments.get(field)
        if isinstance(value, str) and value.strip():
            st.write(field)
            st.code(value, language="text")


def _render_role_revisions(revisions: list[dict[str, Any]], *, key: str) -> None:
    if not revisions:
        st.info("No role revisions were produced by this batch.")
        return

    table = pd.DataFrame([_revision_summary(revision) for revision in revisions])
    selected = _select_detail_row(
        revisions,
        table,
        key=key,
        id_key="id",
    )
    if selected is None:
        return

    st.write("Selected role revision")
    cols = st.columns(4)
    cols[0].metric("Role", str(selected["role_name"]))
    cols[1].metric("Version", str(selected["version"]))
    cols[2].metric("Operation", str(selected["operation"]))
    cols[3].metric("Published", str(selected["publication_status"]))

    st.write("Meta description")
    st.code(str(selected["meta_description"]), language="text")
    st.write("Role instructions")
    st.code(str(selected["role_instructions"]), language="text")


def _render_requests(requests: list[dict[str, Any]]) -> None:
    if not requests:
        st.info("No request rows are linked to this batch.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "request_id": request["id"],
                    "run_id": request["run_id"],
                    "game_id": request["game_id"],
                    "status": request["status"],
                }
                for request in requests
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def _render_roles(data: dict[str, Any]) -> None:
    revisions = list(data["role_revisions"])
    roles = _latest_role_revisions(revisions)
    if not roles:
        st.info("No agent creator roles found in this database.")
        return

    table = pd.DataFrame([_role_summary(role) for role in roles])
    selected = _select_detail_row(
        roles,
        table,
        key=AGENT_CREATOR_ROLE_TABLE_KEY,
        id_key="id",
    )
    if selected is None:
        return

    _render_role_details(selected, revisions)


def _render_role_details(
    role: dict[str, Any],
    revisions: list[dict[str, Any]],
) -> None:
    st.write("Selected role")
    cols = st.columns(5)
    cols[0].metric("Role", str(role["role_name"]))
    cols[1].metric("Version", str(role["version"]))
    cols[2].metric("Status", _role_status(role))
    cols[3].metric("Operation", str(role["operation"]))
    cols[4].metric("Batch", _creator_run_label(role))

    if role.get("error"):
        st.error(str(role["error"]))

    st.write("Meta description")
    st.code(str(role["meta_description"]), language="text")
    st.write("Role instructions")
    st.code(str(role["role_instructions"]), language="text")

    guidance = _dict(role.get("guidance"))
    if guidance:
        st.write("Guidance")
        st.json(guidance)

    role_history = [
        revision
        for revision in revisions
        if revision["role_name"] == role["role_name"]
    ]
    if len(role_history) <= 1:
        return

    st.write("Revision history")
    st.dataframe(
        pd.DataFrame([_revision_summary(revision) for revision in role_history]),
        width="stretch",
        hide_index=True,
        key=f"{AGENT_CREATOR_ROLE_HISTORY_TABLE_KEY}_{role['id']}",
    )


def _select_detail_row(
    rows: list[dict[str, Any]],
    table: pd.DataFrame,
    *,
    key: str,
    id_key: str,
) -> dict[str, Any] | None:
    selected_row_key = f"{key}_selected_id"
    selected_index = _selected_index(
        rows,
        id_key=id_key,
        selected_id=st.session_state.get(selected_row_key),
    )
    if selected_index is None:
        selected_index = 0
        st.session_state[selected_row_key] = int(rows[selected_index][id_key])

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
        selected = rows[selected_rows[0]]
        st.session_state[selected_row_key] = int(selected[id_key])
        return selected
    return rows[selected_index]


def _batch_summary(data: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    requests = _requests_for_run(data, run)
    tool_calls = _tool_calls_for_run(data, run)
    revisions = _revisions_for_run(data, run)
    return {
        "batch_id": run["id"],
        "status": run["status"],
        "requests": len(requests),
        "games": _games_label(requests),
        "tool_calls": _tool_calls_label(tool_calls, run["max_tool_calls"]),
        "changes": _changes_label(tool_calls, revisions),
        "created_at": run["created_at"],
        "completed_at": run["completed_at"] or "",
    }


def _tool_call_summary(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "call": call["call_index"],
        "operation": call["tool_name"],
        "role": call["role_name"],
        "status": call["status"],
    }


def _revision_summary(revision: dict[str, Any]) -> dict[str, Any]:
    return {
        "revision_id": revision["id"],
        "role": revision["role_name"],
        "version": revision["version"],
        "operation": revision["operation"],
        "active": revision["active"],
        "published": revision["publication_status"],
    }


def _role_summary(role: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role["role_name"],
        "status": _role_status(role),
        "version": role["version"],
        "active": role["active"],
        "published": role["publication_status"],
        "operation": role["operation"],
        "batch_id": _creator_run_label(role),
        "updated_at": role["completed_at"] or role["created_at"],
    }


def _requests_for_run(
    data: dict[str, Any],
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    request_ids = set(run["request_ids"])
    return [
        request
        for request in data["requests"]
        if int(request["id"]) in request_ids
    ]


def _tool_calls_for_run(
    data: dict[str, Any],
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        call
        for call in data["tool_calls"]
        if int(call["creator_run_id"]) == int(run["id"])
    ]


def _revisions_for_run(
    data: dict[str, Any],
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        revision
        for revision in data["role_revisions"]
        if revision["creator_run_id"] == int(run["id"])
    ]


def _latest_role_revisions(revisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_role: dict[str, dict[str, Any]] = {}
    for revision in revisions:
        role_name = str(revision["role_name"])
        current = latest_by_role.get(role_name)
        if current is None or int(revision["id"]) > int(current["id"]):
            latest_by_role[role_name] = revision
    return sorted(
        latest_by_role.values(),
        key=lambda role: (
            _role_status_sort_key(role),
            str(role["role_name"]),
        ),
    )


def _role_status(role: dict[str, Any]) -> str:
    publication_status = str(role["publication_status"])
    if publication_status != "complete":
        return publication_status
    if bool(role["active"]):
        return "available"
    return "inactive"


def _role_status_sort_key(role: dict[str, Any]) -> int:
    return {
        "available": 0,
        "staged": 1,
        "failed": 2,
        "inactive": 3,
    }.get(_role_status(role), 4)


def _creator_run_label(role: dict[str, Any]) -> str:
    creator_run_id = role.get("creator_run_id")
    if creator_run_id is None:
        return "-"
    return str(creator_run_id)


def _games_label(requests: list[dict[str, Any]]) -> str:
    labels = [
        f"{request['run_id']}/{request['game_id']}"
        for request in requests
    ]
    if len(labels) <= 2:
        return ", ".join(labels)
    return f"{', '.join(labels[:2])}, +{len(labels) - 2}"


def _tool_calls_label(tool_calls: list[dict[str, Any]], max_tool_calls: int) -> str:
    failed = sum(1 for call in tool_calls if call["status"] != "complete")
    label = f"{len(tool_calls)}/{max_tool_calls}"
    if failed:
        label += f", {failed} failed"
    return label


def _changes_label(
    tool_calls: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
) -> str:
    counts = Counter(str(revision["operation"]) for revision in revisions)
    parts = []
    if counts["add"]:
        parts.append(f"+{counts['add']} add")
    if counts["update"]:
        parts.append(f"~{counts['update']} update")
    if counts["delete"]:
        parts.append(f"-{counts['delete']} delete")
    failed = sum(1 for call in tool_calls if call["status"] != "complete")
    if failed:
        parts.append(f"{failed} failed call")
    return ", ".join(parts) if parts else "no role changes"


def _selected_index(
    rows: list[dict[str, Any]],
    *,
    id_key: str,
    selected_id: Any,
) -> int | None:
    if selected_id is None:
        return None
    try:
        selected = int(selected_id)
    except (TypeError, ValueError):
        return None
    for index, row in enumerate(rows):
        if int(row[id_key]) == selected:
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


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
