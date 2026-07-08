"""Tests for the OpenAI world-model E2E helper logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from face_of_agi.environment.cheat_context import load_cheat_action_context


OPENAI_WORLD_MODEL_E2E_PATH = (
    Path(__file__).parents[1] / "scripts" / "openai_world_model_e2e.py"
)


def load_openai_world_model_e2e_module() -> ModuleType:
    """Load the OpenAI world E2E script as a module for focused helper tests."""

    spec = importlib.util.spec_from_file_location(
        "openai_world_model_e2e_test_module",
        OPENAI_WORLD_MODEL_E2E_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openai_world_e2e_cheat_context_extracts_action_meanings(
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    (game_dir / "game.py").write_text(
        """
from arcengine import GameAction


class Game:
    def step(self, action):
        dx = 0
        dy = 0
        if action == GameAction.ACTION1:
            dy = -1
        if action == GameAction.ACTION2:
            dy = 1
        if action == GameAction.ACTION3:
            dx = -1
        if action == GameAction.ACTION4:
            dx = 1
        x_pos, y_pos = (self.gisrhqpee * dx, self.tbwnoxqgc * dy)
        position = (x_pos, y_pos)
        return position
""".lstrip(),
        encoding="utf-8",
    )

    context = load_cheat_action_context(game_dir)

    assert context == "\n".join(
        [
            "ACTION1: up arrow",
            "ACTION2: down arrow",
            "ACTION3: left arrow",
            "ACTION4: right arrow",
        ]
    )


def test_openai_world_e2e_appends_cheat_context_to_game_context() -> None:
    e2e = load_openai_world_model_e2e_module()
    cheat_context = "ACTION1: up arrow"

    game_context = e2e._compose_game_context(cheat_action_context=cheat_context)

    assert "The proposed action is ACTION1." in game_context
    assert cheat_context in game_context
