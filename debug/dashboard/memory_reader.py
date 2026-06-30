"""Read FACE-OF-AGI SQLite memory for the local debug dashboard.

This module intentionally avoids importing the runtime package. The dashboard is
a sidecar inspector over the persisted SQLite shape, not another runtime entry.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

FRAME_PAYLOAD_TYPE = "face_of_agi.frame.png_base64.v1"
SQLITE_DATABASE_SUFFIXES = (".sqlite", ".sqlite3", ".db")


def list_sqlite_database_files(database_folder: str | Path) -> list[Path]:
    """Return SQLite database files directly inside a dashboard folder."""

    folder = Path(database_folder)
    if not folder.exists():
        raise FileNotFoundError(f"memory database folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"memory database folder is not a directory: {folder}")

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix in SQLITE_DATABASE_SUFFIXES
    )


def load_scoring_memory_rows(database_path: str | Path) -> list[dict[str, Any]]:
    """Load minimal M-state fields needed for dashboard scoring."""

    rows = _query_rows(
        database_path,
        """
        SELECT id, game_id, run_id, turn_metrics_json, metadata_json, created_at
        FROM m_states
        ORDER BY id
        """,
    )
    scoring_rows: list[dict[str, Any]] = []
    turn_counters: dict[tuple[str, str], int] = {}
    for row in rows:
        metadata = _load_json(row["metadata_json"])
        key = (str(row["run_id"]), str(row["game_id"]))
        turn_counters[key] = turn_counters.get(key, 0) + 1
        scoring_rows.append(
            {
                "id": int(row["id"]),
                "game_id": str(row["game_id"]),
                "run_id": str(row["run_id"]),
                "turn_id": int(metadata.get("turn_id") or turn_counters[key]),
                "turn_metrics": _load_optional_json(row["turn_metrics_json"]),
                "created_at": str(row["created_at"]),
            }
        )
    return scoring_rows


def load_m_states(database_path: str | Path) -> list[dict[str, Any]]:
    """Load dedicated M state rows with decoded JSON payloads."""

    rows = _query_rows(database_path, "SELECT * FROM m_states ORDER BY id")
    turn_counters: dict[tuple[str, str], int] = {}
    states: list[dict[str, Any]] = []
    for row in rows:
        state = {
            "id": int(row["id"]),
            "game_id": str(row["game_id"]),
            "run_id": str(row["run_id"]),
            "step": row["step"],
            "frame_index": int(row["frame_index"]),
            "frame_count": int(row["frame_count"]),
            "current_observation": _load_json(row["current_observation_json"]),
            "chosen_action": _load_optional_json(row["chosen_action_json"]),
            "agent_context": _load_json(row["agent_context_json"]),
            "agent_trace": _load_optional_json(row["agent_trace_json"]),
            "turn_metrics": _load_optional_json(
                _row_value(row, "turn_metrics_json")
            ),
            "metadata": _load_json(row["metadata_json"]),
            "created_at": str(row["created_at"]),
        }
        key = (state["run_id"], state["game_id"])
        turn_counters[key] = turn_counters.get(key, 0) + 1
        state["turn_id"] = int(state["metadata"].get("turn_id") or turn_counters[key])
        states.append(state)
    return states


def load_e_experiments(database_path: str | Path) -> list[dict[str, Any]]:
    """Load dedicated E experiment rows with decoded JSON payloads."""

    rows = _query_rows(database_path, "SELECT * FROM e_experiments ORDER BY id")
    experiments: list[dict[str, Any]] = []
    for row in rows:
        tool_call = _load_json(row["tool_call_json"])
        tool_result = _load_json(row["tool_result_json"])
        source_state_id = _row_value(row, "source_state_id")
        source_observation_ref = _dict(tool_result).get("source_observation_ref")
        experiments.append(
            {
                "id": int(row["id"]),
                "game_id": str(row["game_id"]),
                "run_id": str(row["run_id"]),
                "turn_id": int(row["turn_id"]),
                "tool_name": str(row["tool_name"]),
                "source_state_id": (
                    int(source_state_id) if source_state_id is not None else None
                ),
                "source_observation_ref": source_observation_ref,
                "tool_call": tool_call,
                "output_description": _load_json(row["output_description_json"]),
                "tool_result": tool_result,
                "metadata": _load_json(row["metadata_json"]),
                "created_at": str(row["created_at"]),
            }
        )
    return experiments


def load_model_input_debug_records(database_path: str | Path) -> list[dict[str, Any]]:
    """Load raw provider model-input debug records."""

    try:
        rows = _query_rows(
            database_path,
            "SELECT * FROM model_input_debug_records ORDER BY id",
        )
    except sqlite3.OperationalError as exc:
        if "no such table: model_input_debug_records" in str(exc):
            return []
        raise
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "id": int(row["id"]),
                "m_state_id": int(row["m_state_id"]),
                "run_id": str(row["run_id"]),
                "game_id": str(row["game_id"]),
                "turn_id": int(row["turn_id"]),
                "call_slot": str(row["call_slot"]),
                "provider": str(row["provider"]),
                "model": _row_value(row, "model"),
                "phase": str(row["phase"]),
                "attempt": int(row["attempt"]),
                "request": _load_json(row["request_json"]),
                "usage": _load_optional_json(row["usage_json"]),
                "metadata": _load_json(row["metadata_json"]),
                "created_at": str(row["created_at"]),
            }
        )
    return records


def turn_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Return compact table fields for one M state row."""

    metadata = _dict(state.get("metadata"))
    control_mode = _dict(metadata.get("control_mode"))
    trace = _dict(state.get("agent_trace"))
    return {
        "id": state["id"],
        "turn_id": state["turn_id"],
        "run_id": state["run_id"],
        "game_id": state["game_id"],
        "step": state["step"],
        "frame": f"{state['frame_index'] + 1}/{state['frame_count']}",
        "controllable": bool(control_mode.get("controllable", False)),
        "control_reason": control_mode.get("reason", ""),
        "chosen_action": action_label(state.get("chosen_action")),
        "tool_call_count": len(trace.get("tool_calls") or []),
        "created_at": state["created_at"],
    }


def experiment_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    """Return compact table fields for one E experiment row."""

    return {
        "id": experiment["id"],
        "turn_id": experiment["turn_id"],
        "tool": experiment["tool_name"],
        "source": source_label(experiment),
        "action": action_label(_dict(experiment.get("tool_call")).get("action")),
        "created_at": experiment["created_at"],
    }


def matching_experiments(
    experiments: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return E rows associated with the selected M turn."""

    return [
        experiment
        for experiment in experiments
        if experiment["run_id"] == state["run_id"]
        and experiment["game_id"] == state["game_id"]
        and experiment["turn_id"] == state["turn_id"]
    ]


def matching_model_input_records(
    records: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return model-input debug records associated with the selected M turn."""

    state_id = int(state["id"])
    return [
        record
        for record in records
        if int(record["m_state_id"]) == state_id
        or (
            record["run_id"] == state["run_id"]
            and record["game_id"] == state["game_id"]
            and int(record["turn_id"]) == int(state["turn_id"])
        )
    ]


def filter_states(
    states: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    game_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return M states matching the optional run and game filters."""

    filtered = states
    if run_id is not None:
        filtered = [state for state in filtered if state["run_id"] == run_id]
    if game_id is not None:
        filtered = [state for state in filtered if state["game_id"] == game_id]
    return filtered


def latest_state(
    states: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    game_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the newest M state row after optional run/game filtering."""

    filtered = filter_states(states, run_id=run_id, game_id=game_id)
    if not filtered:
        return None
    return max(filtered, key=lambda state: int(state["id"]))


def latest_run_id(states: list[dict[str, Any]]) -> str | None:
    """Return the run id belonging to the newest M state row."""

    state = latest_state(states)
    if state is None:
        return None
    return str(state["run_id"])


def latest_game_id(
    states: list[dict[str, Any]],
    *,
    run_id: str | None = None,
) -> str | None:
    """Return the game id belonging to the newest M state row."""

    state = latest_state(states, run_id=run_id)
    if state is None:
        return None
    return str(state["game_id"])


def states_newer_than(
    states: list[dict[str, Any]],
    state_id: int,
) -> list[dict[str, Any]]:
    """Return M state rows with an id greater than the selected row."""

    return [state for state in states if int(state["id"]) > state_id]


def image_from_payload(value: Any) -> Any | None:
    """Decode a memory PNG payload into a PIL image if possible."""

    if not isinstance(value, dict):
        return None
    if value.get("__type__") != FRAME_PAYLOAD_TYPE:
        return None

    from PIL import Image

    encoded = str(value.get("data", ""))
    if encoded.startswith("data:"):
        _, encoded = encoded.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def observation_image(observation: dict[str, Any] | None) -> Any | None:
    """Return the primary visual frame for an observation payload."""

    payload = _dict(observation)
    image = image_from_payload(payload.get("frame"))
    if image is not None:
        return image
    frames = payload.get("frames") or []
    if frames:
        return image_from_payload(frames[-1])
    return None


def action_label(action: Any) -> str:
    """Return a compact display label for an action payload."""

    payload = _dict(action)
    action_id = str(payload.get("action_id", ""))
    if action_id.startswith("<GameAction.") and ":" in action_id:
        action_id = action_id.removeprefix("<GameAction.").split(":", 1)[0]
    data = payload.get("data")
    if data:
        return f"{action_id} {data}"
    return action_id or "-"


def ref_label(ref: Any) -> str:
    """Return a compact display label for an ObservationRef payload."""

    payload = _dict(ref)
    memory = payload.get("memory")
    ref_id = payload.get("id")
    if memory and ref_id:
        return f"{memory}:{ref_id}"
    return "-"


def source_label(experiment: dict[str, Any]) -> str:
    """Return the current E source label with old ref fallback."""

    source_state_id = experiment.get("source_state_id")
    if source_state_id is not None:
        return f"M:{source_state_id}"
    return ref_label(experiment.get("source_observation_ref"))


def redacted_for_json(value: Any) -> Any:
    """Return JSON-displayable data with image bytes replaced by summaries."""

    copied = deepcopy(value)
    return _redact_images(copied)


def _query_rows(database_path: str | Path, query: str) -> list[sqlite3.Row]:
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"memory database not found: {path}")

    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return list(connection.execute(query).fetchall())


def _load_json(raw: Any) -> Any:
    return json.loads(str(raw))


def _load_optional_json(raw: Any) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return _load_json(raw)


def _row_value(row: sqlite3.Row, key: str) -> Any | None:
    if key not in row.keys():
        return None
    return row[key]


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _redact_images(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__type__") == FRAME_PAYLOAD_TYPE:
            return {
                "__type__": FRAME_PAYLOAD_TYPE,
                "mime_type": value.get("mime_type"),
                "width": value.get("width"),
                "height": value.get("height"),
                "data": "<base64 png redacted>",
            }
        return {key: _redact_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_images(item) for item in value]
    return value
