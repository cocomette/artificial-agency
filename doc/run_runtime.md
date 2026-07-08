# Run The Runtime

Copy-paste these commands from the repository root.

## First Setup

For the light runtime environment:

```bash
uv sync --no-dev
```

For a runtime environment with model backends:

```bash
uv sync --extra ml --no-dev
```

For the full development environment:

```bash
uv sync --group dev
```

## Create Or Refresh The Game Catalog

Run this once before using `game_index` from
`src/face_of_agi/runtime/starter_loop.yaml`.

With the light runtime environment:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

With model backends available:

```bash
uv run --extra ml --no-dev python -m face_of_agi.runtime.shell --list-games
```

From the full development environment:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --list-games
```

This writes:

```text
src/face_of_agi/environment/local_games.json
```

## Run The Runtime

With the light runtime environment:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

With model backends available:

```bash
uv run --extra ml --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

From the full development environment:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

The runtime starts orchestration. The default `X` adapter selects from the
valid ARC actions and the loop prints a condensed trace for each frame turn.

## Terminal-Friendly Rendering

If `render_mode: human` cannot open a window, edit
`src/face_of_agi/runtime/starter_loop.yaml`:

```yaml
enable_visualization: true
render_mode: terminal
```

Then run the same command:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/starter_loop.yaml
```

Or use the matching `uv run --extra ml --no-dev ...` or
`uv run --group dev ...` variant for the environment you synced.

## Useful Config Values

The starter config lives here:

```text
src/face_of_agi/runtime/starter_loop.yaml
```

Common values to change:

- `game_index`: selected game from the catalog printed by `--list-games`
- `max_actions_per_level`: action budget before stopping
- `enable_visualization`: show frames while running
- `render_mode`: `human`, `terminal`, or `terminal-fast`
- `cheat_action_context`: append action semantics parsed from local game source
- `cheat_action_context_game_dir`: optional override for the local game source
  directory used by `cheat_action_context`
- `experimental_memory_turn_buffer`: latest frame turns kept in rolling `E`
  memory; defaults to `2`
- `models.prompt_model_calls_enabled`: enable real post-decision S/G model
  calls; defaults to `false`, which mocks predictions from the current frame
- `models.agent.backend`: `random`, `openai`, or `ollama`; `huggingface` and
  `configurable` are reserved provider names and fail clearly until
  implemented
- `models.agent.model`: defaults are `gpt-5-nano` for OpenAI and
  `gemma4:e4b` for Ollama
- `models.world.backend` / `models.goal.backend`: `none`, `openai`, or
  `huggingface-diffusers`

## OpenAI Agent

Set an API key and opt into OpenAI in the YAML config:

```bash
export OPENAI_API_KEY=...
```

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

## Ollama Gemma 4 Agent

Start Ollama and pull the configured model:

```bash
ollama serve
ollama pull gemma4:e4b
```

Then configure:

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

World and goal tools are explicit. The starter runtime does not expose
world/goal tools on non-controllable animation frames, so X must submit the
internal `NONE` action on those frames.

Provider-backed model adapters are imported from role-local `providers/`
folders in code. Orchestration still receives only the role interfaces and
does not branch on concrete providers.
