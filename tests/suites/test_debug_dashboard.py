"""Tests for local debug dashboard helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from debug.dashboard import config_manager
from debug.dashboard import test_workshop
from debug.dashboard.memory_reader import (
    load_m_states,
    load_model_input_debug_records,
    matching_model_input_records,
)
from debug.dashboard.model_inputs import provider_output, sent_images, token_counts
from debug.dashboard.runner import (
    build_modal_run_command,
    build_clean_db_command,
    build_list_games_command,
    build_run_command,
    format_command,
)
from face_of_agi.contracts import ContextDocuments, Observation
from face_of_agi.memory import SQLiteDatabase, StateMemory

VALID_CONFIG = """
game_index: 0
max_actions_per_level: 1
models:
  agent:
    backend: openai
    model: gpt-5-nano
  world:
    backend: openai
    model: gpt-5-nano
  goal:
    backend: openai
    model: gpt-5-nano
  updater:
    world:
      backend: openai
      model: gpt-5-nano
    goal:
      backend: openai
      model: gpt-5-nano
    agent:
      backend: openai
      model: gpt-5-nano
    general:
      backend: openai
      model: gpt-5-nano
"""


def test_config_manager_lists_yaml_files_and_rejects_escape_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    nested_dir = config_dir / "nested"
    nested_dir.mkdir()
    (config_dir / "b.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (config_dir / "a.yml").write_text(VALID_CONFIG, encoding="utf-8")
    (nested_dir / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (config_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [
        path.relative_to(config_dir).as_posix()
        for path in config_manager.list_config_files(config_dir)
    ] == [
        "a.yml",
        "b.yaml",
        "nested/config.yaml",
    ]
    assert config_manager.safe_config_path("a.yml", config_dir=config_dir) == (
        config_dir / "a.yml"
    ).resolve()
    assert config_manager.safe_config_path(
        "nested/config.yaml",
        config_dir=config_dir,
    ) == (nested_dir / "config.yaml").resolve()
    assert config_manager.config_label(
        nested_dir / "config.yaml",
        config_dir=config_dir,
    ) == "nested/config.yaml"

    with pytest.raises(ValueError, match="stay within"):
        config_manager.safe_config_path("../escape.yaml", config_dir=config_dir)
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
        Path("src/face_of_agi/runtime/configs/openai/openai_all_gpt5_nano_test.yaml"),
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
        "src/face_of_agi/runtime/configs/openai/openai_all_gpt5_nano_test.yaml",
        "--database",
        "runs/memory.sqlite",
        "--debug-keep-all-m-states",
    ]
    assert format_command(["uv", "run", "path with spaces.yaml"]) == (
        "uv run 'path with spaces.yaml'"
    )


def test_runner_builds_modal_runtime_command() -> None:
    command = build_modal_run_command(
        Path("src/face_of_agi/runtime/configs/ollama/ollama_shared_gemma4_26b.yaml"),
        database_name="/vol/runs/memory.sqlite",
        live_commit_seconds=5,
        timing=True,
    )

    assert command == [
        "uv",
        "run",
        "--with",
        "modal",
        "modal",
        "run",
        "src/face_of_agi/runtime/modal_app.py",
        "--config",
        "src/face_of_agi/runtime/configs/ollama/ollama_shared_gemma4_26b.yaml",
        "--database-name",
        "memory.sqlite",
        "--live-commit-seconds",
        "5",
        "--timing",
    ]

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


def test_dashboard_loads_matching_model_input_debug_records(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    pending = state.prewrite_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=Observation(id="obs-1", step=0, frame={"frame": 0}),
        contexts=ContextDocuments(),
        metadata={"turn_id": 7},
    )
    state.write_model_input_debug_record(
        m_state_id=pending.id,
        run_id="run-1",
        game_id="game-1",
        turn_id=7,
        call_slot="agent",
        provider="openai",
        model="gpt-5-nano",
        phase="final_action",
        attempt=0,
        request={
            "model": "gpt-5-nano",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "prompt"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,RAW",
                        },
                    ],
                }
            ],
        },
        usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
    )

    states = load_m_states(database.path)
    records = load_model_input_debug_records(database.path)
    selected = matching_model_input_records(records, states[0])

    assert len(selected) == 1
    assert selected[0]["m_state_id"] == pending.id
    assert selected[0]["request"]["input"][0]["content"][1]["image_url"].endswith(
        "RAW"
    )


def test_model_input_token_counts_use_provider_usage() -> None:
    assert token_counts(
        {"usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}}
    ) == {"input": 11, "output": 3, "total": 14}
    assert token_counts(
        {"usage": {"prompt_eval_count": 5, "eval_count": 2}}
    ) == {"input": 5, "output": 2, "total": 7}
    assert token_counts({"usage": None}) == {
        "input": None,
        "output": None,
        "total": None,
    }


def test_model_input_helpers_decode_openai_data_url_images() -> None:
    record = {
        "request": {
            "input": [
                {
                    "content": [
                        {"type": "input_text", "text": "prompt"},
                        {
                            "type": "input_image",
                            "image_url": _tiny_png_data_url(size=(2, 3)),
                        },
                    ],
                }
            ]
        }
    }

    images = sent_images(record)

    assert len(images) == 1
    assert images[0].label == "Input item 1 image 2"
    assert images[0].error is None
    assert images[0].image.size == (2, 3)


def test_model_input_helpers_decode_ollama_base64_images() -> None:
    encoded = _tiny_png_data_url(size=(4, 5)).split(",", 1)[1]
    record = {
        "request": {
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "prompt", "images": [encoded]},
            ]
        }
    }

    images = sent_images(record)

    assert len(images) == 1
    assert images[0].label == "Message 2 image 1"
    assert images[0].error is None
    assert images[0].image.size == (4, 5)


def test_model_input_helpers_return_invalid_image_fallback() -> None:
    record = {
        "request": {
            "input": [
                {
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,RAW",
                        }
                    ],
                }
            ]
        }
    }

    images = sent_images(record)

    assert len(images) == 1
    assert images[0].image is None
    assert images[0].error is not None
    assert "Could not decode image payload" in images[0].error


def test_model_input_provider_output_formats_captured_json() -> None:
    output = provider_output(
        {
            "metadata": {
                "response_output_text": '{"ok": true}',
                "response_metadata": {"response_id": "resp-1"},
                "response_payload": {"output_text": '{"ok": true}'},
            }
        }
    )

    assert output.available is True
    assert output.text == '{"ok": true}'
    assert output.parsed_json == {"ok": True}
    assert output.metadata == {"response_id": "resp-1"}
    assert output.raw_response == {"output_text": '{"ok": true}'}


def test_test_workshop_builds_e2e_commands_and_rejects_escape_paths(
    tmp_path: Path,
) -> None:
    e2e_dir = tmp_path / "tests" / "e2e"
    e2e_dir.mkdir(parents=True)
    for filename in (
        "openai_goal_model_e2e.py",
        "ollama_goal_description_e2e.py",
        "ollama_image_description_e2e.py",
        "ollama_world_description_e2e.py",
    ):
        (e2e_dir / filename).write_text("print('ok')\n", encoding="utf-8")
    (e2e_dir / "notes.txt").write_text("ignore\n", encoding="utf-8")

    assert [path.name for path in test_workshop.list_e2e_tests(root=tmp_path)] == [
        "ollama_goal_description_e2e.py",
        "ollama_image_description_e2e.py",
        "ollama_world_description_e2e.py",
        "openai_goal_model_e2e.py",
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
    assert test_workshop.build_e2e_command(
        "ollama_world_description_e2e.py",
        root=tmp_path,
    ) == [
        "uv",
        "run",
        "--locked",
        "--extra",
        "ml",
        "--no-dev",
        "python",
        "tests/e2e/ollama_world_description_e2e.py",
    ]

    with pytest.raises(ValueError, match="stay within"):
        test_workshop.safe_e2e_path("../escape.py", root=tmp_path)


def _tiny_png_data_url(*, size: tuple[int, int] = (2, 2)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
