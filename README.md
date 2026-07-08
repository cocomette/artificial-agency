# FACE-OF-AGI

Minimal Python framework for an ARC-AGI-3 agent.

The repo contains a Python runtime shell, an ARC-AGI environment adapter,
orchestration loop scaffolding, model-role adapters, and SQLite-backed memory.
Architecture context lives under `doc/architecture/`.

Model-specific contracts, configs, and adapter shells live under `models`.
World and goal models are exposed as tools for orchestrator agent `X`, while
deterministic orchestration stays in the `orchestration` package.

## Setup

Use Python 3.12 and `uv` from the repo root.

For the full development environment, including tests, notebooks, and ML
backends:

```bash
uv sync --group dev
uv run python -c "import face_of_agi"
```

For a lighter runtime environment without notebook or ML dependencies:

```bash
uv sync --no-dev
```

For a runtime environment with ML backends but without dev tools:

```bash
uv sync --extra ml --no-dev
```

OpenAI-backed world and goal tools require an API key in the normal OpenAI SDK
environment variable:

```bash
export OPENAI_API_KEY=...
```

Ollama-backed local agent runs require Ollama plus the selected Gemma 4 model:

```bash
ollama serve
ollama pull gemma4:e4b
```

For Linux window rendering with `render_mode: human`, install Tk once:

```bash
sudo apt-get install python3-tk
```

## Run

The starter environment config lives at
`src/face_of_agi/runtime/starter_loop.yaml`.

Run the configured ARC game with the light runtime environment:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

Run the configured ARC game with ML backends available for local or hosted
model adapters:

```bash
uv run --extra ml --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

Run the configured ARC game from the full development environment:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

The starter shell stores persistent state memory `M` in `runs/memory.sqlite`
by default. Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```

Inspect the discoverable ARC toolkit game list and write the local catalog:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

The starter config selects from that stored catalog with `game_index`.

Quick copy-paste runtime commands also live in `doc/run_runtime.md`.

## Starter Config

The starter config supports:

- `game_index`
- `max_actions_per_level`
- `operation_mode`
- `game_catalog_path`
- `environments_dir`
- `recordings_dir`
- `enable_visualization`
- `render_mode`
- `seed`
- `save_recording`
- `include_frame_data`
- `cheat_action_context`
- `cheat_action_context_game_dir`
- `experimental_memory_turn_buffer`
- `models.prompt_model_calls_enabled`
- `models.agent.backend`
- `models.world.backend`
- `models.goal.backend`

Visualization is optional and environment-local. Set
`enable_visualization: true` in
`src/face_of_agi/runtime/starter_loop.yaml` to display outgoing ARC frame
bundles. Supported render modes are:

- `render_mode: human`
- `render_mode: terminal`
- `render_mode: terminal-fast`

`experimental_memory_turn_buffer` controls the rolling `E` experiment buffer.
It defaults to `2`, meaning tool-produced experimental frames remain
referenceable for the latest two frame turns per run and game.

Set `cheat_action_context: true` to append action semantics parsed from the
local game source to the agent, world, and goal contexts. For example, `ls20`
maps `ACTION1` to up, `ACTION2` to down, `ACTION3` to left, and `ACTION4` to
right. The runtime infers the source directory from `game_id`; use
`cheat_action_context_game_dir` only when that local path needs an override.

The default starter config keeps `models.agent.backend: random` and leaves
world and goal tools disabled. `models.prompt_model_calls_enabled` defaults to
`false`, so post-decision world and goal predictions are mocked from the
current frame unless real tool calls are explicitly enabled. To opt into a VLM
agent, set:

```yaml
models:
  prompt_model_calls_enabled: false
  agent:
    backend: ollama
    model: gemma4:e4b
    max_tool_calls: 2
    repair_attempts: 1
  world:
    backend: none
  goal:
    backend: none
```

or:

```yaml
models:
  prompt_model_calls_enabled: true
  agent:
    backend: openai
    model: gpt-5-nano
    max_tool_calls: 2
    repair_attempts: 1
  world:
    backend: openai
  goal:
    backend: openai
```

World and goal tools are wired explicitly. If an agent calls an unconfigured
tool, orchestration returns a repairable tool error and then fails clearly if
the model repeats the invalid call. The starter runtime exposes no world/goal
tools on non-controllable animation frames; the agent must submit the internal
`NONE` action for those frames.

## Model Backends

The local Diffusers world and goal tools support Qwen Image Edit,
InstructPix2Pix-style editors, and FLUX Kontext qint8. The OpenAI-backed world
and goal tools use the Responses API with the hosted image-generation tool and
can be injected through the model registry. The orchestrator agent `X` supports
OpenAI Responses tool calling and Ollama native tool calling with `gemma4:e4b`.

```python
from face_of_agi.models import (
    OpenAIOrchestratorAgentAdapter,
    ModelRegistry,
    OpenAIGoalToolAdapter,
    OpenAIWorldToolAdapter,
)

registry = ModelRegistry(
    orchestrator_agent=OpenAIOrchestratorAgentAdapter(),
    world_tool=OpenAIWorldToolAdapter(),
    goal_tool=OpenAIGoalToolAdapter(),
)
```

The OpenAI adapters default to `gpt-5-nano` with low reasoning effort and
`gpt-image-1-mini` for generated prediction images. E2E scripts resize uploaded
input images to `1024x1024` by default.

The Ollama agent adapter defaults to `gemma4:e4b` and requires the `ml` extra
or dev environment so the official `ollama` Python package is installed.

## Checks

Run the test suite:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

Run the manual local world-model E2E check:

```bash
uv run --locked --extra ml --no-dev python scripts/world_model_e2e.py
```

Run the manual OpenAI E2E checks:

```bash
uv run --locked --extra ml --no-dev python scripts/openai_world_model_e2e.py
uv run --locked --extra ml --no-dev python scripts/openai_goal_model_e2e.py
uv run --locked --extra ml --no-dev python scripts/openai_orchestrator_agent_e2e.py
uv run --env-file .env --group dev python scripts/openai_full_game_loop_e2e.py
uv run --env-file .env --group dev python scripts/openai_full_game_loop_e2e.py --cheat-action-context
uv run --locked --extra ml --no-dev python scripts/ollama_orchestrator_agent_e2e.py
```

The E2E scripts write outputs under `runs/`.

## Docs

- `doc/architecture/system_architecture.md`: high-level agent architecture.
- `doc/architecture/software/`: target software module boundaries.
- `doc/architecture/techstack.md`: current tools, frameworks, and runtime stack.
- `doc/run_runtime.md`: runtime command notes.

## Project-local Codex

Project-specific Codex assets live under `.agents/`.
