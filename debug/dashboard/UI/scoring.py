"""Dashboard scoring page for comparing memory runs to human baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

import streamlit as st

from debug.dashboard.memory_reader import (
    list_sqlite_database_files,
    load_scoring_memory_rows,
)

BASELINE_STATS_PATH = Path("debug/scoring/human_baseline_stats.json")
BASELINE_COLUMNS = (
    "avg_number_of_actions",
    "median_number_of_actions",
    "number_of_players",
    "number_of_players_solved_level",
    "number_of_players_below_average_number_of_actions",
    "number_of_players_below_median_number_of_actions",
)
TIME_CURSOR_KEY = "scoring_time_cursor_hours"
TIME_CURSOR_MAX_HOURS = 10.0
TIME_CURSOR_STEP_HOURS = 10.0 / 60.0


class ScoringInputError(RuntimeError):
    """Raised when memory files do not match scoring expectations."""


@dataclass(frozen=True)
class MemoryGameScore:
    """Solved-level action counts extracted from one memory database."""

    game_id: str
    full_game_id: str
    memory_file: str
    run_id: str
    solved_actions_by_level: dict[str, int]


def render_scoring(database_folder: str) -> None:
    """Render per-game, per-level scoring against human baseline stats."""

    cursor_minutes = _render_time_cursor()

    try:
        baseline_stats = _load_baseline_stats(str(BASELINE_STATS_PATH))
        memory_scores, warnings = _load_memory_scores(database_folder, cursor_minutes)
    except Exception as exc:
        st.error(str(exc))
        return

    for warning in warnings:
        st.warning(warning)

    if not memory_scores:
        st.info("No memory games found in this folder.")
        return

    rows = _scoring_rows(baseline_stats, memory_scores)
    with st.expander("Metrics", expanded=True):
        _render_summary(memory_scores, rows)
        st.dataframe(rows, width="stretch", hide_index=True)

    st.header("Scoring")
    scoring = _score_rows(rows)
    _render_scoring(scoring)


@st.cache_data(show_spinner=False)
def _load_baseline_stats(path: str) -> dict[str, Any]:
    baseline_path = Path(path)
    try:
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"human baseline stats not found: {baseline_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"human baseline stats are not valid JSON: {baseline_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"human baseline stats must be a JSON object: {baseline_path}")
    return loaded


def _load_memory_scores(
    database_folder: str,
    cursor_minutes: int,
) -> tuple[list[MemoryGameScore], list[str]]:
    database_files = list_sqlite_database_files(database_folder)
    if not database_files:
        return [], [f"No SQLite memory files found in `{database_folder}`."]

    scores: list[MemoryGameScore] = []
    warnings: list[str] = []
    files_by_game: dict[str, str] = {}
    for database_file in database_files:
        score = _memory_score(database_file, cursor_minutes)
        if score is None:
            warnings.append(f"`{database_file.name}` has no M-state rows.")
            continue

        existing_file = files_by_game.get(score.game_id)
        if existing_file is not None:
            raise ScoringInputError(
                "Memory files are not correct: scoring expects one file per game, "
                f"but `{existing_file}` and `{database_file.name}` both map to "
                f"`{score.game_id}`."
            )
        files_by_game[score.game_id] = database_file.name
        scores.append(score)

    return sorted(scores, key=lambda score: score.game_id), warnings


def _memory_score(database_file: Path, cursor_minutes: int) -> MemoryGameScore | None:
    rows = load_scoring_memory_rows(database_file)
    if not rows:
        return None

    latest_run_id = str(rows[-1]["run_id"])
    latest_run_rows = [row for row in rows if row["run_id"] == latest_run_id]
    cursor_rows = _rows_before_time_cursor(latest_run_rows, cursor_minutes)
    full_game_ids = {str(row["game_id"]) for row in latest_run_rows}
    short_game_ids = {_short_game_id(game_id) for game_id in full_game_ids}
    if len(short_game_ids) != 1:
        raise ScoringInputError(
            f"`{database_file.name}` contains multiple games; scoring expects one "
            "file per game."
        )

    full_game_id = sorted(full_game_ids)[0]
    game_id = sorted(short_game_ids)[0]
    return MemoryGameScore(
        game_id=game_id,
        full_game_id=full_game_id,
        memory_file=database_file.name,
        run_id=latest_run_id,
        solved_actions_by_level=_solved_actions_by_level(cursor_rows),
    )


def _render_time_cursor() -> int:
    cursor_hours = float(
        st.slider(
            "Time cursor (hours)",
            min_value=0.0,
            max_value=TIME_CURSOR_MAX_HOURS,
            value=TIME_CURSOR_MAX_HOURS,
            step=TIME_CURSOR_STEP_HOURS,
            key=TIME_CURSOR_KEY,
            format="%.2f h",
        )
    )
    cursor_minutes = int(round(cursor_hours * 60))
    st.caption(
        f"Scoring includes rows up to {cursor_hours:.2f}h, excluding the newest "
        "in-window row per game."
    )
    return cursor_minutes


def _rows_before_time_cursor(
    rows: list[dict[str, Any]],
    cursor_minutes: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    start = _created_at(rows[0])
    cutoff = start + timedelta(minutes=cursor_minutes)
    in_window = [
        row
        for row in rows
        if _created_at(row) <= cutoff
    ]
    if not in_window:
        return []
    return in_window[:-1]


def _created_at(row: dict[str, Any]) -> datetime:
    raw_created_at = row.get("created_at")
    if not isinstance(raw_created_at, str) or not raw_created_at.strip():
        raise ScoringInputError("memory row is missing created_at")
    try:
        return datetime.fromisoformat(raw_created_at)
    except ValueError as exc:
        raise ScoringInputError(
            f"memory row has invalid created_at: {raw_created_at!r}"
        ) from exc


def _solved_actions_by_level(rows: list[dict[str, Any]]) -> dict[str, int]:
    solved: dict[str, int] = {}
    highest_seen = 0
    for row in rows:
        score = _cumulative_score(row.get("turn_metrics"))
        if score is None or score <= highest_seen:
            continue

        submitted_actions = _submitted_action_count(row)
        for level in range(highest_seen + 1, score + 1):
            solved.setdefault(str(level), submitted_actions)
        highest_seen = score
    return solved


def _submitted_action_count(row: dict[str, Any]) -> int:
    turn_metrics = row.get("turn_metrics")
    if not isinstance(turn_metrics, dict):
        raise ScoringInputError("memory row is missing turn metrics")

    base_count = _nonnegative_integral_number(
        turn_metrics.get("time_cost"),
        field_name="turn metric time_cost",
    )
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not bool(metadata.get("simulated", False)):
        return base_count

    catchup = metadata.get("known_state_simulation_catchup")
    if not isinstance(catchup, dict):
        return base_count

    return base_count + _nonnegative_integral_number(
        catchup.get("catchup_action_count", 0),
        field_name="simulation catchup_action_count",
    )


def _nonnegative_integral_number(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringInputError(f"{field_name} is not numeric: {value!r}")
    numeric = float(value)
    if numeric < 0 or not numeric.is_integer():
        raise ScoringInputError(f"{field_name} is not a non-negative integer: {value!r}")
    return int(numeric)


def _cumulative_score(turn_metrics: Any) -> int | None:
    if not isinstance(turn_metrics, dict):
        return None
    raw_score = turn_metrics.get("cumulative_score")
    if raw_score is None:
        return None
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ScoringInputError(f"turn metric cumulative_score is not numeric: {raw_score!r}")
    return int(raw_score)


def _scoring_rows(
    baseline_stats: dict[str, Any],
    memory_scores: list[MemoryGameScore],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score in memory_scores:
        baseline_levels = _baseline_levels(baseline_stats, score.game_id)
        levels = sorted(
            {*baseline_levels, *score.solved_actions_by_level},
            key=_level_sort_key,
        )
        for level in levels:
            baseline = baseline_levels.get(level) or {}
            row = {
                "game_id": score.game_id,
                "level": level,
                "memory_file": score.memory_file,
                "our_number_of_actions": score.solved_actions_by_level.get(level),
            }
            for column in BASELINE_COLUMNS:
                row[column] = baseline.get(column)
            rows.append(row)
    return rows


def _baseline_levels(baseline_stats: dict[str, Any], game_id: str) -> dict[str, Any]:
    game = baseline_stats.get(game_id)
    if not isinstance(game, dict):
        return {}
    levels = game.get("levels")
    if not isinstance(levels, dict):
        return {}
    return levels


def _render_summary(
    memory_scores: list[MemoryGameScore],
    rows: list[dict[str, Any]],
) -> None:
    solved_level_count = sum(
        1 for row in rows if row["our_number_of_actions"] is not None
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Memory games", len(memory_scores))
    metric_cols[1].metric("Scored levels", len(rows))
    metric_cols[2].metric("Solved levels", solved_level_count)


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_level_rows: list[dict[str, Any]] = []
    level_scores_by_game: dict[str, list[tuple[int, float]]] = {}

    for row in rows:
        agent_actions = row.get("our_number_of_actions")
        human_actions = row.get("median_number_of_actions")
        raw_score = _raw_level_score(
            human_actions=human_actions,
            agent_actions=agent_actions,
        )
        level_score = raw_score**2
        level_index = _level_index(row["level"])
        game_id = str(row["game_id"])
        level_scores_by_game.setdefault(game_id, []).append((level_index, level_score))
        per_level_rows.append(
            {
                "game_id": game_id,
                "level": row["level"],
                "memory_file": row["memory_file"],
                "human_actions_median": human_actions,
                "agent_actions": agent_actions,
                "raw_score": raw_score,
                "per_level_score": level_score,
                "per_level_score_percent": level_score * 100,
                "level_weight": level_index,
            }
        )

    per_game_rows = [
        {
            "game_id": game_id,
            "levels": len(level_scores),
            "completed_levels": sum(1 for _, score in level_scores if score > 0),
            "per_game_score": _weighted_average(level_scores),
            "per_game_score_percent": _weighted_average(level_scores) * 100,
        }
        for game_id, level_scores in sorted(level_scores_by_game.items())
    ]
    total_score = _average(
        [row["per_game_score"] for row in per_game_rows]
    )

    return {
        "per_level_rows": per_level_rows,
        "per_game_rows": per_game_rows,
        "total_score": total_score,
    }


def _render_scoring(scoring: dict[str, Any]) -> None:
    per_level_rows = list(scoring["per_level_rows"])
    per_game_rows = list(scoring["per_game_rows"])
    total_score = scoring["total_score"]

    score_text = "-" if total_score is None else f"{total_score * 100:.2f}%"
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total score", score_text)
    metric_cols[1].metric("Scored games", len(per_game_rows))
    metric_cols[2].metric("Scored levels", len(per_level_rows))

    if not per_level_rows:
        st.info("No levels with human median baselines are available to score.")
        return

    game_tab, level_tab, method_tab = st.tabs(
        ["Per-game scores", "Per-level scores", "Method"]
    )
    with game_tab:
        st.dataframe(per_game_rows, width="stretch", hide_index=True)
    with level_tab:
        st.dataframe(per_level_rows, width="stretch", hide_index=True)
    with method_tab:
        st.markdown(
            """
            For each completed level, the agent action count is compared to the
            human median action count from first-time testers. The agent action
            count is the cumulative number of actions submitted to the ARC
            environment when the level is first observed as solved. Simulated
            memory rows do not add actions. When a simulated row records the
            solved level before catch-up runs, its catch-up submissions are
            included from catch-up metadata. Unsolved levels count as `0%`.

            1. Per-level score: `min(human_actions / agent_actions, 1.0) ** 2`.
            2. Per-game score: weighted average of completed level scores,
               weighted by the 1-indexed level number.
            3. Total score: average of individual per-game scores.
            """
        )


def _weighted_average(level_scores: list[tuple[int, float]]) -> float:
    total_weight = sum(weight for weight, _ in level_scores)
    if total_weight <= 0:
        return 0.0
    return (
        sum(weight * score for weight, score in level_scores)
        / total_weight
    )


def _raw_level_score(
    *,
    human_actions: Any,
    agent_actions: Any,
) -> float:
    if not isinstance(human_actions, (int, float)):
        return 0.0
    if not isinstance(agent_actions, (int, float)) or agent_actions <= 0:
        return 0.0
    return min(float(human_actions) / float(agent_actions), 1.0)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _level_index(level: Any) -> int:
    try:
        return max(1, int(level))
    except (TypeError, ValueError):
        return 1


def _short_game_id(game_id: str) -> str:
    return game_id.split("-", 1)[0]


def _level_sort_key(level: str) -> tuple[int, int | str]:
    try:
        return (0, int(level))
    except ValueError:
        return (1, level)
