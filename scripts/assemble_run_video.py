"""Assemble persisted run observation frames into an MP4 video.

Run from the repo root:

    uv run python scripts/assemble_run_video.py \
        --run-id game-index-3-20260517T200151Z \
        --memory-file resources/shared-memory_modal.sqlite
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Any, Sequence
from urllib.parse import quote

from PIL import Image

FRAME_PAYLOAD_TYPE = "face_of_agi.frame.png_base64.v1"


@dataclass(frozen=True, slots=True)
class FrameRow:
    """One persisted M-state row containing a current observation frame."""

    id: int
    game_id: str
    step: int | None
    frame_index: int
    frame_count: int
    current_observation: dict[str, Any]
    created_at: str


def load_frame_rows(
    memory_file: Path,
    *,
    run_id: str,
    game_id: str | None,
) -> tuple[FrameRow, ...]:
    """Read all M-state frame rows for one run from a memory SQLite file."""

    if not memory_file.exists():
        raise FileNotFoundError(f"memory file not found: {memory_file}")

    uri = f"file:{quote(str(memory_file.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if game_id is not None:
            clauses.append("game_id = ?")
            values.append(game_id)

        rows = connection.execute(
            f"""
            SELECT
                id,
                game_id,
                step,
                frame_index,
                frame_count,
                current_observation_json,
                created_at
            FROM m_states
            WHERE {' AND '.join(clauses)}
            ORDER BY id
            """,
            values,
        ).fetchall()

        if not rows:
            known_runs = _known_runs(connection)
            details = f" Known runs:\n{known_runs}" if known_runs else ""
            raise RuntimeError(
                f"no M-state frames found for run_id={run_id!r}"
                + (f" game_id={game_id!r}" if game_id is not None else "")
                + "."
                + details
            )

    frame_rows = tuple(_frame_row_from_sql(row) for row in rows)
    game_ids = sorted({row.game_id for row in frame_rows})
    if game_id is None and len(game_ids) > 1:
        games = ", ".join(game_ids)
        raise RuntimeError(
            f"run_id={run_id!r} contains multiple games: {games}. "
            "Pass --game-id to select one video timeline."
        )
    return frame_rows


def extract_images(rows: Sequence[FrameRow], *, scale: int) -> tuple[Image.Image, ...]:
    """Decode and scale the observation image from each selected row."""

    if scale < 1:
        raise ValueError("--scale must be at least 1")

    images: list[Image.Image] = []
    for row in rows:
        image = _observation_image(row)
        if scale > 1:
            width, height = image.size
            image = image.resize(
                (width * scale, height * scale),
                Image.Resampling.NEAREST,
            )
        images.append(image)

    if not images:
        raise RuntimeError("no frames were decoded")

    expected_size = images[0].size
    for index, image in enumerate(images, start=1):
        if image.size != expected_size:
            raise RuntimeError(
                "decoded frames do not share one video size: "
                f"frame 1 is {expected_size}, frame {index} is {image.size}"
            )
    return tuple(images)


def write_mp4(
    images: Sequence[Image.Image],
    *,
    output_file: Path,
    fps: float,
    ffmpeg_bin: str,
    overwrite: bool,
) -> None:
    """Encode decoded RGB images into an MP4 using ffmpeg."""

    if not images:
        raise RuntimeError("cannot write a video with no frames")
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if output_file.suffix.lower() != ".mp4":
        raise ValueError("--output must end with .mp4")
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"output file already exists: {output_file}. "
            "Pass --overwrite to replace it."
        )

    ffmpeg_path = shutil.which(ffmpeg_bin)
    if ffmpeg_path is None:
        raise RuntimeError(
            f"ffmpeg executable not found: {ffmpeg_bin!r}. Install ffmpeg or pass "
            "--ffmpeg-bin with its path."
        )

    width, height = images[0].size
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        _format_fps(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("failed to open ffmpeg pipes")

    try:
        for image in images:
            process.stdin.write(image.convert("RGB").tobytes())
        process.stdin.close()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        process.wait()
        raise RuntimeError(
            f"ffmpeg closed before all frames were written: {stderr}"
        ) from exc

    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")


def default_output_file(run_id: str, game_id: str) -> Path:
    """Return the default ignored run-video output path."""

    filename = f"{_filename_slug(run_id)}_{_filename_slug(game_id)}.mp4"
    return Path("runs") / "videos" / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an MP4 from persisted m_states observation frames."
    )
    parser.add_argument("--run-id", required=True, help="Run id to export.")
    parser.add_argument(
        "--memory-file",
        required=True,
        type=Path,
        help="SQLite memory file containing m_states.",
    )
    parser.add_argument(
        "--game-id",
        help="Game id to export when the run contains multiple games.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path. Defaults to runs/videos/<run>_<game>.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Video frames per second. Defaults to 4.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Nearest-neighbor scale factor for tiny memory frames. Defaults to 8.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name or path. Defaults to ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_frame_rows(
        args.memory_file,
        run_id=args.run_id,
        game_id=args.game_id,
    )
    game_ids = sorted({row.game_id for row in rows})
    if len(game_ids) != 1:
        raise RuntimeError("selected frames must belong to exactly one game")

    output_file = args.output or default_output_file(args.run_id, game_ids[0])
    images = extract_images(rows, scale=args.scale)
    write_mp4(
        images,
        output_file=output_file,
        fps=args.fps,
        ffmpeg_bin=args.ffmpeg_bin,
        overwrite=args.overwrite,
    )

    duration = len(images) / args.fps
    print(
        f"wrote {output_file} from {len(images)} frame(s), "
        f"run_id={args.run_id!r}, game_id={game_ids[0]!r}, "
        f"fps={_format_fps(args.fps)}, duration={duration:.2f}s"
    )


def _frame_row_from_sql(row: sqlite3.Row) -> FrameRow:
    observation = json.loads(str(row["current_observation_json"]))
    if not isinstance(observation, dict):
        raise RuntimeError(
            f"m_state id {row['id']} current_observation is not an object"
        )
    return FrameRow(
        id=int(row["id"]),
        game_id=str(row["game_id"]),
        step=(int(row["step"]) if row["step"] is not None else None),
        frame_index=int(row["frame_index"]),
        frame_count=int(row["frame_count"]),
        current_observation=observation,
        created_at=str(row["created_at"]),
    )


def _known_runs(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """
        SELECT run_id, game_id, COUNT(*) AS frame_count, MAX(id) AS newest_id
        FROM m_states
        GROUP BY run_id, game_id
        ORDER BY newest_id DESC
        LIMIT 10
        """
    ).fetchall()
    return "\n".join(
        f"- run_id={row['run_id']} game_id={row['game_id']} frames={row['frame_count']}"
        for row in rows
    )


def _observation_image(row: FrameRow) -> Image.Image:
    observation = row.current_observation
    image = _image_from_payload(observation.get("frame"), row_id=row.id)
    if image is not None:
        return image

    frames = observation.get("frames")
    if frames is not None and not isinstance(frames, list):
        raise RuntimeError(
            f"m_state id {row.id} observation frames field is not a list"
        )
    if frames:
        image = _image_from_payload(frames[-1], row_id=row.id)
        if image is not None:
            return image

    raise RuntimeError(f"m_state id {row.id} does not contain a serialized PNG frame")


def _image_from_payload(value: Any, *, row_id: int) -> Image.Image | None:
    if not isinstance(value, dict):
        return None
    if value.get("__type__") != FRAME_PAYLOAD_TYPE:
        return None

    encoded = str(value.get("data", "")).strip()
    if not encoded:
        raise RuntimeError(f"m_state id {row_id} frame payload is missing data")
    if encoded.startswith("data:"):
        _, encoded = encoded.split(",", 1)

    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise RuntimeError(
            f"m_state id {row_id} frame payload is not valid base64"
        ) from exc

    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"m_state id {row_id} frame payload is not a valid image"
        ) from exc


def _filename_slug(value: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._")
    return slug or "unnamed"


def _format_fps(fps: float) -> str:
    return f"{fps:g}"


if __name__ == "__main__":
    main()
