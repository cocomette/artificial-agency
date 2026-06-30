"""Build a compact human-baseline summary from ARC-AGI recording files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("arc_agi_3_public_demo_human_testing/public_games-dataset")
DEFAULT_OUTPUT = Path("debug/scoring/human_baseline.json")
RECORDING_SUFFIX = ".recording.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-level human actions from recording summaries."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Dataset root containing one folder per game. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the generated JSON summary. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def read_summary(recording_path: Path) -> dict[str, Any]:
    last_line = ""
    with recording_path.open(encoding="utf-8") as recording_file:
        for line in recording_file:
            stripped = line.strip()
            if stripped:
                last_line = stripped

    if not last_line:
        raise ValueError(f"{recording_path} is empty")

    try:
        summary = json.loads(last_line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{recording_path} has invalid JSON in its final row") from error

    if not isinstance(summary, dict):
        raise ValueError(f"{recording_path} final row must be a JSON object")
    return summary


def player_id_from_recording(recording_path: Path) -> str:
    if not recording_path.name.endswith(RECORDING_SUFFIX):
        raise ValueError(f"{recording_path} does not end with {RECORDING_SUFFIX}")
    return recording_path.name[: -len(RECORDING_SUFFIX)]


def extract_actions_by_level(recording_path: Path) -> dict[str, Any]:
    summary = read_summary(recording_path)
    data = summary.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{recording_path} summary is missing object field data")

    cards = data.get("cards")
    if not isinstance(cards, dict):
        raise ValueError(f"{recording_path} summary is missing object field data.cards")

    levels: dict[str, Any] = {}
    for card_id, card in cards.items():
        if not isinstance(card, dict):
            raise ValueError(f"{recording_path} card {card_id!r} must be an object")

        actions_by_level = card.get("actions_by_level")
        if actions_by_level is None:
            continue
        if not isinstance(actions_by_level, list):
            raise ValueError(
                f"{recording_path} card {card_id!r} has non-list field actions_by_level"
            )

        for action_sequence in actions_by_level:
            if not isinstance(action_sequence, list):
                raise ValueError(
                    f"{recording_path} card {card_id!r} has a malformed action sequence"
                )

            for pair in action_sequence:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(
                        f"{recording_path} card {card_id!r} has a malformed level/action pair"
                    )

                level, action_count = pair
                level_key = str(level)
                existing = levels.get(level_key)
                if level_key in levels and existing != action_count:
                    raise ValueError(
                        f"{recording_path} has conflicting values for level {level_key}"
                    )
                levels[level_key] = action_count

    return levels


def level_sort_key(level: str) -> tuple[int, int | str]:
    try:
        return (0, int(level))
    except ValueError:
        return (1, level)


def build_human_baseline(input_root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    if not input_root.is_dir():
        raise ValueError(f"{input_root} is not a directory")

    baseline: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    for game_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        game_id = game_dir.name
        player_levels: dict[str, dict[str, Any]] = {}
        all_levels: set[str] = set()

        for recording_path in sorted(game_dir.rglob("*.jsonl")):
            player_id = player_id_from_recording(recording_path)
            if player_id in player_levels:
                raise ValueError(f"{game_dir} has multiple recordings for player {player_id}")

            levels = extract_actions_by_level(recording_path)
            player_levels[player_id] = levels
            all_levels.update(levels)

        baseline[game_id] = {"levels": {}}
        for level in sorted(all_levels, key=level_sort_key):
            baseline[game_id]["levels"][level] = {
                player_id: levels.get(level)
                for player_id, levels in sorted(player_levels.items())
            }

    return baseline


def main() -> None:
    args = parse_args()
    baseline = build_human_baseline(args.input_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(baseline, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
