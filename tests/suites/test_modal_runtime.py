"""Tests for Modal runtime helper behavior that does not call Modal."""

from __future__ import annotations

from io import StringIO
import sys

from face_of_agi.runtime.modal_app import (
    _modal_env,
    ollama_models_from_config_text,
    run_streamed_subprocess,
)


def test_modal_helper_collects_ollama_models_from_shared_vlm_config() -> None:
    config_text = """
game_index: 0
max_actions_per_level: 1
models:
  shared_vlm:
    backend: ollama
    model: qwen3.6
  agent:
    backend: ollama
  world:
    backend: ollama
  goal:
    backend: ollama
  updater:
    world:
      backend: ollama
    goal:
      backend: ollama
    agent:
      backend: ollama
    general:
      backend: ollama
"""

    assert ollama_models_from_config_text(config_text) == ("qwen3.6",)


def test_modal_helper_collects_role_specific_ollama_models() -> None:
    config_text = """
game_index: 0
max_actions_per_level: 1
models:
  agent:
    backend: ollama
    model: qwen-agent
  world:
    backend: ollama
    model: qwen-world
  goal:
    backend: ollama
    model: qwen-goal
  updater:
    world:
      backend: ollama
      model: qwen-world-updater
    goal:
      backend: ollama
      model: qwen-updater
    agent:
      backend: ollama
      model: qwen-agent-updater
    general:
      backend: ollama
      model: qwen-general-updater
"""

    assert ollama_models_from_config_text(config_text) == (
        "qwen-agent",
        "qwen-world",
        "qwen-goal",
        "qwen-world-updater",
        "qwen-updater",
        "qwen-agent-updater",
        "qwen-general-updater",
    )


def test_modal_streamed_subprocess_tees_and_captures_output() -> None:
    stdout = StringIO()
    stderr = StringIO()

    result = run_streamed_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('stdout line'); "
                "print('stderr line', file=sys.stderr)"
            ),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert result.returncode == 0
    assert result.stdout == "stdout line\n"
    assert result.stderr == "stderr line\n"
    assert stdout.getvalue() == result.stdout
    assert stderr.getvalue() == result.stderr
