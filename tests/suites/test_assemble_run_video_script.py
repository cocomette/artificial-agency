"""Tests for the run-frame video assembly script helpers."""

from __future__ import annotations

import base64
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

from PIL import Image
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "assemble_run_video.py"
SPEC = importlib.util.spec_from_file_location("assemble_run_video", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
assemble_run_video = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assemble_run_video
SPEC.loader.exec_module(assemble_run_video)


def test_load_frame_rows_and_decode_images(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite"
    _create_memory_table(database_path)
    _insert_frame(database_path, run_id="other-run", game_id="game-1", color="white")
    _insert_frame(database_path, run_id="run-1", game_id="game-1", color="red")
    _insert_frame(database_path, run_id="run-1", game_id="game-1", color="blue")

    rows = assemble_run_video.load_frame_rows(
        database_path,
        run_id="run-1",
        game_id=None,
    )
    images = assemble_run_video.extract_images(rows, scale=2)

    assert [row.id for row in rows] == [2, 3]
    assert [image.size for image in images] == [(4, 4), (4, 4)]
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (0, 0, 255)


def test_load_frame_rows_requires_game_filter_for_multi_game_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "memory.sqlite"
    _create_memory_table(database_path)
    _insert_frame(database_path, run_id="run-1", game_id="game-1", color="red")
    _insert_frame(database_path, run_id="run-1", game_id="game-2", color="blue")

    with pytest.raises(RuntimeError, match="multiple games"):
        assemble_run_video.load_frame_rows(
            database_path,
            run_id="run-1",
            game_id=None,
        )


def _create_memory_table(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE m_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step INTEGER,
                frame_index INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                current_observation_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _insert_frame(
    database_path: Path,
    *,
    run_id: str,
    game_id: str,
    color: str,
) -> None:
    observation = {
        "id": f"{run_id}-{game_id}",
        "step": 0,
        "frame": _frame_payload(color),
        "frames": [],
    }
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO m_states (
                game_id,
                run_id,
                step,
                frame_index,
                frame_count,
                current_observation_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, run_id, 0, 0, 1, json.dumps(observation)),
        )


def _frame_payload(color: str) -> dict[str, object]:
    image = Image.new("RGB", (2, 2), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {
        "__type__": assemble_run_video.FRAME_PAYLOAD_TYPE,
        "mime_type": "image/png",
        "encoding": "base64",
        "width": image.width,
        "height": image.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
