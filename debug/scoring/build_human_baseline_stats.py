"""Build per-game, per-level statistics from the human-baseline summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_INPUT = Path("debug/scoring/human_baseline.json")
DEFAULT_OUTPUT = Path("debug/scoring/human_baseline_stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-level statistics from human_baseline.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Human baseline JSON path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the generated stats JSON. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def load_baseline(input_path: Path) -> dict[str, Any]:
    try:
        baseline = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{input_path} is not valid JSON") from error

    if not isinstance(baseline, dict):
        raise ValueError(f"{input_path} must contain a JSON object")
    return baseline


def numeric_actions(game_id: str, level: str, players: dict[str, Any]) -> list[int | float]:
    actions: list[int | float] = []
    for player_id, action_count in players.items():
        if action_count is None:
            continue
        if not isinstance(action_count, (int, float)):
            raise ValueError(
                f"{game_id} level {level} player {player_id} has non-numeric actions"
            )
        actions.append(action_count)
    return actions


def level_stats(game_id: str, level: str, players: dict[str, Any]) -> dict[str, Any]:
    solved_actions = numeric_actions(game_id, level, players)
    avg_actions = mean(solved_actions) if solved_actions else None
    median_actions = median(solved_actions) if solved_actions else None

    return {
        "avg_number_of_actions": avg_actions,
        "median_number_of_actions": median_actions,
        "number_of_players": len(players),
        "number_of_players_solved_level": len(solved_actions),
        "number_of_players_below_average_number_of_actions": (
            sum(action_count < avg_actions for action_count in solved_actions)
            if avg_actions is not None
            else 0
        ),
        "number_of_players_below_median_number_of_actions": (
            sum(action_count < median_actions for action_count in solved_actions)
            if median_actions is not None
            else 0
        ),
    }


def build_human_baseline_stats(baseline: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    for game_id, game in baseline.items():
        if not isinstance(game, dict):
            raise ValueError(f"{game_id} must be an object")

        levels = game.get("levels")
        if not isinstance(levels, dict):
            raise ValueError(f"{game_id} is missing object field levels")

        stats[game_id] = {"levels": {}}
        for level, players in levels.items():
            if not isinstance(players, dict):
                raise ValueError(f"{game_id} level {level} must be an object")

            stats[game_id]["levels"][level] = level_stats(game_id, level, players)

    return stats


def main() -> None:
    args = parse_args()
    baseline = load_baseline(args.input)
    stats = build_human_baseline_stats(baseline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
