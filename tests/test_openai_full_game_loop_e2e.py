"""Tests for the manual OpenAI full game-loop E2E harness helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

from face_of_agi.contracts import (
    EExperimentRecord,
    MStateRecord,
    ObservationRef,
    RoleContext,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "openai_full_game_loop_e2e.py"


def load_e2e_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "openai_full_game_loop_e2e_test_module",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_e2e_environment_config_forces_openai_one_step_settings(
    tmp_path: Path,
) -> None:
    e2e = load_e2e_module()
    catalog_path = tmp_path / "local_games.json"
    catalog_path.write_text(json.dumps({"2": "game-from-catalog"}), encoding="utf-8")
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 2",
                "max_actions_per_level: 99",
                "game_catalog_path: " + str(catalog_path),
                "enable_visualization: true",
                "save_recording: true",
                "models:",
                "  prompt_model_calls_enabled: true",
                "  agent:",
                "    backend: random",
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=str(config_path),
        game_index=None,
        game_id=None,
        agent_model="agent-model",
        world_model="world-model",
        goal_model="goal-model",
        image_model="image-model",
        image_size="512x512",
        image_quality="low",
        reasoning_effort="low",
        max_tool_calls=2,
        repair_attempts=1,
    )

    config = e2e.build_e2e_environment_config(args)

    assert config.game_index == e2e.DEFAULT_GAME_INDEX
    assert config.game_id == e2e.DEFAULT_GAME_ID
    assert config.max_actions_per_level == 1
    assert config.enable_visualization is False
    assert config.save_recording is False
    assert config.include_frame_data is True
    assert config.models.prompt_model_calls_enabled is False
    assert config.models.agent.backend == "openai"
    assert config.models.agent.model == "agent-model"
    assert config.models.world.backend == "openai"
    assert config.models.world.model == "world-model"
    assert config.models.goal.backend == "openai"
    assert config.models.goal.model == "goal-model"
    assert config.models.world.options["image_model"] == "image-model"
    assert config.models.goal.options["image_size"] == "512x512"


def test_build_e2e_environment_config_allows_game_index_override(
    tmp_path: Path,
) -> None:
    e2e = load_e2e_module()
    catalog_path = tmp_path / "local_games.json"
    catalog_path.write_text(json.dumps({"2": "game-from-catalog"}), encoding="utf-8")
    config_path = tmp_path / "starter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "max_actions_per_level: 99",
                "game_catalog_path: " + str(catalog_path),
                "models:",
                "  agent:",
                "    backend: random",
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=str(config_path),
        game_index=2,
        game_id=None,
        agent_model="agent-model",
        world_model="world-model",
        goal_model="goal-model",
        image_model="image-model",
        image_size="512x512",
        image_quality="low",
        reasoning_effort="low",
        max_tool_calls=2,
        repair_attempts=1,
    )

    config = e2e.build_e2e_environment_config(args)

    assert config.game_index == 2
    assert config.game_id == "game-from-catalog"


def test_build_context_documents_appends_cheat_action_context(
    tmp_path: Path,
) -> None:
    e2e = load_e2e_module()
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
    args = SimpleNamespace(
        cheat_action_context=True,
        game_dir=str(game_dir),
    )
    environment_config = SimpleNamespace(game_id=e2e.DEFAULT_GAME_ID)

    contexts = e2e.build_context_documents(
        args=args,
        environment_config=environment_config,
    )

    assert e2e.DEFAULT_GAME_ID in contexts.agent.game
    assert "Cheat action context from the local game source:" in contexts.agent.game
    assert "ACTION1: up arrow" in contexts.agent.game
    assert "ACTION4: right arrow" in contexts.world.game
    assert "ACTION4: right arrow" in contexts.goal.game


def test_artifact_writer_serializes_memory_records_and_saves_pngs(
    tmp_path: Path,
) -> None:
    e2e = load_e2e_module()
    writer = e2e.ArtifactWriter(tmp_path)
    frame = Image.new("RGB", (4, 4), color=(20, 40, 60))
    ref = ObservationRef(memory="state", id="obs-0")
    state = MStateRecord(
        id=1,
        game_id="game-1",
        run_id="run-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation={"id": "obs-0", "step": 0, "frame": frame},
        chosen_action={"action_id": "ACTION1", "data": None},
        world_context=RoleContext(),
        goal_context=RoleContext(),
        agent_context=RoleContext(),
        agent_trace={"tool_results": [], "metadata": {"backend": "openai"}},
        world_prediction=None,
        goal_prediction=None,
        metadata={},
        created_at="now",
    )
    experiment = EExperimentRecord(
        id=2,
        game_id="game-1",
        run_id="run-1",
        turn_id=1,
        tool_name="world",
        source_observation_ref=ref,
        tool_call={"tool": "world", "observation_ref": ref},
        output_observation={"id": "world-out", "step": 0, "frame": frame},
        tool_result={
            "id": "world-out",
            "tool": "world",
            "predicted_observation": frame,
            "metadata": {"response_id": "resp-1"},
        },
        metadata={},
        created_at="now",
    )

    e2e._write_artifact_json(tmp_path / "m_states.json", [state], writer)
    e2e._write_artifact_json(tmp_path / "e_experiments.json", [experiment], writer)

    states_payload = json.loads((tmp_path / "m_states.json").read_text())
    experiments_payload = json.loads((tmp_path / "e_experiments.json").read_text())
    assert states_payload[0]["current_observation"]["frame"]["image_path"].endswith(
        ".png"
    )
    assert experiments_payload[0]["tool_result"]["predicted_observation"][
        "image_path"
    ].endswith(".png")
    assert list((tmp_path / "images").glob("*.png"))


def test_artifact_writer_clears_stale_images(tmp_path: Path) -> None:
    e2e = load_e2e_module()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    stale_path = image_dir / "stale.png"
    stale_path.write_bytes(b"stale")

    e2e.ArtifactWriter(tmp_path)

    assert not stale_path.exists()
