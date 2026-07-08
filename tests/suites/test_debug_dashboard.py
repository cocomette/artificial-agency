"""Tests for local debug dashboard helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from debug.dashboard import config_manager
from debug.dashboard import test_workshop
from debug.dashboard.runner import (
    build_clean_db_command,
    build_list_games_command,
    build_run_command,
    format_command,
)

VALID_CONFIG = """
game_index: 0
max_actions_per_level: 1
models:
  agent:
    backend: random
  world:
    backend: none
  goal:
    backend: none
  updater:
    world:
      backend: mock
    goal:
      backend: mock
    agent:
      backend: mock
    general:
      backend: mock
"""


def test_config_manager_lists_yaml_files_and_rejects_escape_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "b.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (config_dir / "a.yml").write_text(VALID_CONFIG, encoding="utf-8")
    (config_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [path.name for path in config_manager.list_config_files(config_dir)] == [
        "a.yml",
        "b.yaml",
    ]
    assert config_manager.safe_config_path("a.yml", config_dir=config_dir) == (
        config_dir / "a.yml"
    ).resolve()

    with pytest.raises(ValueError, match="stay within"):
        config_manager.safe_config_path("../escape.yaml", config_dir=config_dir)
    nested = config_manager.safe_config_path("nested/config.yaml", config_dir=config_dir)
    assert nested == (config_dir / "nested" / "config.yaml").resolve()
    with pytest.raises(ValueError, match="\\.yaml"):
        config_manager.safe_config_path("config.txt", config_dir=config_dir)


def test_config_manager_validates_and_saves_configs(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    validation = config_manager.validate_config_text(VALID_CONFIG)

    assert validation.valid is True
    assert validation.data is not None

    saved = config_manager.save_config_as(
        "debug_copy.yaml",
        VALID_CONFIG,
        config_dir=config_dir,
    )
    assert saved.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(FileExistsError):
        config_manager.save_config_as(
            "debug_copy.yaml",
            VALID_CONFIG,
            config_dir=config_dir,
        )

    invalid = config_manager.validate_config_text("game_index: 0\n")
    assert invalid.valid is False
    assert "Missing required config keys" in invalid.message


def test_runner_builds_dev_profile_runtime_command() -> None:
    command = build_run_command(
        Path("src/face_of_agi/runtime/configs/random_local.yaml"),
        Path("runs/memory.sqlite"),
    )

    assert command == [
        "uv",
        "run",
        "--group",
        "dev",
        "python",
        "-m",
        "face_of_agi.runtime.shell",
        "--config",
        "src/face_of_agi/runtime/configs/random_local.yaml",
        "--database",
        "runs/memory.sqlite",
        "--debug-keep-all-m-states",
    ]
    assert "--debug-keep-all-m-states" not in build_run_command(
        "config.yaml",
        "memory.sqlite",
        keep_all_m_states=False,
    )
    assert format_command(["uv", "run", "path with spaces.yaml"]) == (
        "uv run 'path with spaces.yaml'"
    )


def test_runner_builds_clean_db_command() -> None:
    assert build_clean_db_command(Path("runs/memory.sqlite")) == [
        "uv",
        "run",
        "--no-dev",
        "python",
        "-m",
        "face_of_agi.runtime.shell",
        "--database",
        "runs/memory.sqlite",
        "--clean-db",
    ]


def test_runner_builds_list_games_command() -> None:
    assert build_list_games_command() == [
        "uv",
        "run",
        "--no-dev",
        "python",
        "-m",
        "face_of_agi.runtime.shell",
        "--list-games",
    ]


def test_test_workshop_builds_e2e_commands_and_rejects_escape_paths(
    tmp_path: Path,
) -> None:
    e2e_dir = tmp_path / "tests" / "e2e"
    e2e_dir.mkdir(parents=True)
    for filename in (
        "openai_full_game_loop_e2e.py",
        "openai_goal_model_e2e.py",
        "ollama_image_description_e2e.py",
        "world_model_e2e.py",
    ):
        (e2e_dir / filename).write_text("print('ok')\n", encoding="utf-8")
    (e2e_dir / "notes.txt").write_text("ignore\n", encoding="utf-8")

    assert [path.name for path in test_workshop.list_e2e_tests(root=tmp_path)] == [
        "ollama_image_description_e2e.py",
        "openai_full_game_loop_e2e.py",
        "openai_goal_model_e2e.py",
        "world_model_e2e.py",
    ]
    assert test_workshop.build_e2e_command(
        "openai_goal_model_e2e.py",
        root=tmp_path,
        extra_args=["--model", "gpt-5-nano"],
    ) == [
        "uv",
        "run",
        "--env-file",
        ".env",
        "--locked",
        "--extra",
        "ml",
        "--no-dev",
        "python",
        "tests/e2e/openai_goal_model_e2e.py",
        "--model",
        "gpt-5-nano",
    ]
    assert test_workshop.build_e2e_command(
        "openai_full_game_loop_e2e.py",
        root=tmp_path,
    ) == [
        "uv",
        "run",
        "--env-file",
        ".env",
        "--locked",
        "--extra",
        "ml",
        "--no-dev",
        "python",
        "tests/e2e/openai_full_game_loop_e2e.py",
    ]
    assert test_workshop.build_e2e_command(
        "ollama_image_description_e2e.py",
        root=tmp_path,
    ) == [
        "uv",
        "run",
        "--locked",
        "--extra",
        "ml",
        "--no-dev",
        "python",
        "tests/e2e/ollama_image_description_e2e.py",
    ]

    with pytest.raises(ValueError, match="stay within"):
        test_workshop.safe_e2e_path("../escape.py", root=tmp_path)


def test_test_workshop_collects_generic_result_artifacts(tmp_path: Path) -> None:
    result_dir = tmp_path / "runs" / "demo_e2e"
    image_dir = result_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "first.png").write_bytes(b"not a real png for unit tests")
    (result_dir / "loose.png").write_bytes(b"not a real png for unit tests")
    (result_dir / "summary.json").write_text(
        """
{
  "json_title": "Run Summary",
  "artifacts": {
    "input_image": "images/first.png"
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (result_dir / "metrics.json").write_text(
        """
{
  "json_title": "Metric Payload",
  "ok": true,
  "output": {
    "loose_image": "runs/demo_e2e/loose.png"
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    artifacts = test_workshop.collect_result_artifacts(result_dir, root=tmp_path)

    assert [(item.title, item.path.name) for item in artifacts.images] == [
        ("input_image", "first.png"),
        ("loose_image", "loose.png"),
    ]
    assert [(item.title, item.path.name) for item in artifacts.json_files] == [
        ("Run Summary", "summary.json"),
        ("Metric Payload", "metrics.json"),
    ]
    assert artifacts.json_files[0].data["artifacts"]["input_image"] == (
        "images/first.png"
    )
