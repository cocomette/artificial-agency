"""Prepare an offline Kaggle dataset with all available public ARC games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import read_json_with_kaggle_dataset_id, write_json_if_changed  # noqa: E402

DEFAULT_OUTPUT_DIR = KAGGLE_ROOT / "build/public-games"
DEFAULT_METADATA_PATH = KAGGLE_ROOT / "upload/public-games/dataset-metadata.json"
PUBLIC_GAMES_DATASET_SLUG = "face-of-agi-public-games"


def main(argv: list[str] | None = None) -> None:
    """Prepare every public game returned by ARC normal mode."""

    args = _build_parser().parse_args(argv)
    output_dir = prepare_public_games(
        output_dir=Path(args.output_dir),
        metadata_path=Path(args.metadata_path),
    )
    print(f"[prepare_public_games] Wrote {output_dir.relative_to(ROOT)}")


def prepare_public_games(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> Path:
    """Download all available public games into one Kaggle dataset directory."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    environments_dir = output_dir / "environment_files"
    recordings_dir = output_dir / "recordings"
    environments_dir.mkdir(parents=True)
    recordings_dir.mkdir(parents=True)

    game_ids = _download_all_public_games(
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
    )
    _write_game_catalog(output_dir / "local_games.json", game_ids)
    write_json_if_changed(
        output_dir / "dataset-metadata.json",
        read_json_with_kaggle_dataset_id(metadata_path, PUBLIC_GAMES_DATASET_SLUG),
    )
    return output_dir


def _download_all_public_games(
    *,
    environments_dir: Path,
    recordings_dir: Path,
) -> tuple[str, ...]:
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(environments_dir),
        recordings_dir=str(recordings_dir),
    )
    game_ids = _public_game_ids(tuple(arcade.get_environments()))
    for game_id in game_ids:
        environment = arcade.make(game_id, seed=0, save_recording=False)
        if environment is None:
            raise RuntimeError(f"unable to prepare public ARC game '{game_id}'")
    return game_ids


def _public_game_ids(available_games: Sequence[Any]) -> tuple[str, ...]:
    game_ids = sorted({str(game.game_id).strip() for game in available_games})
    if not game_ids or any(not game_id for game_id in game_ids):
        raise RuntimeError("ARC returned no public game ids to prepare")
    return tuple(game_ids)


def _write_game_catalog(path: Path, game_ids: Sequence[str]) -> None:
    path.write_text(
        json.dumps(
            {str(index): game_id for index, game_id in enumerate(game_ids)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a Kaggle dataset with all public ARC games.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Prepared Kaggle dataset directory.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Dataset metadata copied into the prepared dataset directory.",
    )
    return parser


if __name__ == "__main__":
    main()
