"""Read agent-creator SQLite data for the local debug dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote


def load_agent_creator_inspection(database_path: str | Path) -> dict[str, Any]:
    """Return passive dashboard rows from an agent-creator database."""

    return {
        "runs": _load_creator_runs(database_path),
        "requests": _load_game_requests(database_path),
        "tool_calls": _load_tool_calls(database_path),
        "role_revisions": _load_role_revisions(database_path),
    }


def _load_creator_runs(database_path: str | Path) -> list[dict[str, Any]]:
    rows = _query_rows(database_path, "SELECT * FROM agent_creator_runs ORDER BY id")
    return [
        {
            "id": int(row["id"]),
            "status": str(row["status"]),
            "request_ids": tuple(
                int(item) for item in _json_list(row["request_ids_json"])
            ),
            "max_tool_calls": int(row["max_tool_calls"]),
            "created_at": str(row["created_at"]),
            "completed_at": _optional_str(row["completed_at"]),
            "error": _optional_str(row["error"]),
        }
        for row in rows
    ]


def _load_game_requests(database_path: str | Path) -> list[dict[str, Any]]:
    rows = _query_rows(
        database_path,
        "SELECT * FROM agent_creator_game_requests ORDER BY id",
    )
    return [
        {
            "id": int(row["id"]),
            "status": str(row["status"]),
            "run_id": str(row["run_id"]),
            "game_id": str(row["game_id"]),
            "memory_database_path": str(row["memory_database_path"]),
            "claimed_at": _optional_str(row["claimed_at"]),
            "completed_at": _optional_str(row["completed_at"]),
            "error": _optional_str(row["error"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]


def _load_tool_calls(database_path: str | Path) -> list[dict[str, Any]]:
    rows = _query_rows(
        database_path,
        "SELECT * FROM agent_creator_tool_calls ORDER BY run_id, call_index, id",
    )
    tool_calls = []
    for row in rows:
        arguments = _json_dict(row["arguments_json"])
        result = _optional_json_dict(row["result_json"])
        tool_calls.append(
            {
                "id": int(row["id"]),
                "creator_run_id": int(row["run_id"]),
                "call_index": int(row["call_index"]),
                "tool_name": str(row["tool_name"]),
                "role_name": _role_name(arguments),
                "arguments": arguments,
                "status": str(row["status"]),
                "result": result,
                "message": _tool_message(result, row["error"]),
                "error": _optional_str(row["error"]),
                "created_at": str(row["created_at"]),
                "completed_at": _optional_str(row["completed_at"]),
            }
        )
    return tool_calls


def _load_role_revisions(database_path: str | Path) -> list[dict[str, Any]]:
    rows = _query_rows(
        database_path,
        "SELECT * FROM agent_role_revisions ORDER BY id",
    )
    return [
        {
            "id": int(row["id"]),
            "role_name": str(row["role_name"]),
            "version": int(row["version"]),
            "active": bool(row["active"]),
            "publication_status": str(row["publication_status"]),
            "operation": str(row["operation"]),
            "meta_description": str(row["meta_description"]),
            "role_instructions": str(row["role_instructions"]),
            "creator_run_id": _optional_int(row["created_by_run_id"]),
            "guidance": _json_dict(row["guidance_json"]),
            "error": _optional_str(row["error"]),
            "created_at": str(row["created_at"]),
            "completed_at": _optional_str(row["completed_at"]),
        }
        for row in rows
    ]


def _query_rows(database_path: str | Path, query: str) -> list[sqlite3.Row]:
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"agent creator database not found: {path}")

    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return list(connection.execute(query).fetchall())


def _json_list(raw: Any) -> list[Any]:
    loaded = json.loads(str(raw))
    if isinstance(loaded, list):
        return loaded
    return []


def _json_dict(raw: Any) -> dict[str, Any]:
    loaded = json.loads(str(raw))
    if isinstance(loaded, dict):
        return loaded
    return {}


def _optional_json_dict(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return _json_dict(raw)


def _role_name(
    arguments: dict[str, Any],
) -> str:
    value = arguments.get("role_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "-"


def _tool_message(result: dict[str, Any] | None, error: Any) -> str:
    if result is not None:
        reason = result.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
        status = result.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    if error is not None:
        return str(error)
    return ""


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
