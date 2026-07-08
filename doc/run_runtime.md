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

For the local debug dashboard:

```bash
uv sync --group debug
```

## Create Or Refresh The Game Catalog

Run this once before using `game_index` from
`src/face_of_agi/runtime/configs/starter_loop.yaml`.

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
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

With model backends available:

```bash
uv run --extra ml --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

From the full development environment:

```bash
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

The runtime starts orchestration with the configured real model providers and
prints a condensed trace for each frame turn.

## Clear Runtime Memory

Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```

Use the matching `uv run --extra ml --no-dev ...` or `uv run --group dev ...`
variant for the environment you synced.

## Ready-To-Run Configs

These configs preserve all `M` frame-turn rows for dashboard inspection.

Hosted OpenAI loop:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/openai/openai_all_gpt5_nano_test.yaml
```

Fully local Ollama loop:

```bash
ollama serve
ollama pull gemma4:26b
uv run --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/ollama/ollama_all_gemma4_26b.yaml
```

Mixed local/hosted `ls20` run with Ollama X, Ollama `L^X` updater, and
OpenAI S/G models:

```bash
ollama serve
ollama pull gemma4:e2b
uv run --env-file .env --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/ollama_x_openai_sg_agent_updater_ls20.yaml
```

Mixed local/hosted `ls20` run with Ollama X, Ollama game/general updaters for
X/S/G, and OpenAI S/G models:

```bash
ollama serve
ollama pull gemma4:e2b
uv run --env-file .env --group dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/ollama_x_openai_sg_full_updater_ls20.yaml
```

This config preserves all `M` state rows and uses terminal rendering by
default. Change `render_mode` to `human` if you want a matplotlib window.

## Terminal-Friendly Rendering

If `render_mode: human` cannot open a window, edit
`src/face_of_agi/runtime/configs/starter_loop.yaml`:

```yaml
enable_visualization: true
render_mode: terminal
```

Then run the same command:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

Or use the matching `uv run --extra ml --no-dev ...` or
`uv run --group dev ...` variant for the environment you synced.

## Useful Config Values

The starter config lives here:

```text
src/face_of_agi/runtime/configs/starter_loop.yaml
```

Common values to change:

- `game_index`: selected game from the catalog printed by `--list-games`
- `max_actions_per_level`: action budget before stopping
- `enable_visualization`: show frames while running
- `render_mode`: `human`, `terminal`, or `terminal-fast`
- `use_learned_contexts`: hydrate prior learned `K` and `L` contexts from
  SQLite at startup; defaults to `true`, set to `false` for a fresh run that
  keeps existing database rows untouched
- `experimental_memory_turn_buffer`: latest frame turns kept in rolling `E`
  memory; defaults to `2`
- `action_history_window`: prior frame-turn actions included in each X prompt;
  defaults to `8`, and `0` disables the compact history
- `debug_keep_all_m_states`: keep every `M` frame-turn row after a successful
  run; defaults to `false`
- `debug_trace`: stdout trace mode: `off`, `minimal`, `agent_decision`,
  `verbose`, or `model_inputs`; defaults to `minimal`
- `debug_color`: Rich color mode for debug traces: `auto`, `always`, or
  `never`; defaults to `auto`
- `models.agent.backend`: `openai` or `ollama`; `configurable` is a
  reserved provider name and fails clearly until implemented
- `models.agent.model`: defaults are `gpt-5-nano` for OpenAI and
  `gemma4:e4b` for Ollama
- `models.world.backend` / `models.goal.backend`: `openai` or `ollama`

## Modal Debug Dashboard

The Modal runner stores remote run memory on the `face-of-agi-runs` Volume and
commits that volume periodically while the game loop is active. To watch it
from a local Streamlit dashboard, pull the latest committed SQLite snapshot
before each refresh:

```bash
uv run --group debug --with modal streamlit run debug/dashboard/app.py -- --modal
```

The dashboard reads the downloaded snapshot from `runs/modal-memory.sqlite`.
Use `--modal-volume`, `--modal-database`, `--modal-snapshot`, or
`--local-database` after Streamlit's `--` separator if you need a non-default
Volume, remote database name, local snapshot path, or local-run database path.
The Runner page keeps local and Modal launch buttons side by side in Modal
mode.

## OpenAI Agent

Set an API key and opt into OpenAI in the YAML config:

```bash
export OPENAI_API_KEY=...
```

```yaml
models:
  agent:
    backend: openai
    model: gpt-5-nano
    max_tool_calls: 0
    repair_attempts: 1
  world:
    backend: openai
  goal:
    backend: openai
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
  agent:
    backend: ollama
    model: gemma4:e4b
    max_tool_calls: 0
    repair_attempts: 1
  world:
    backend: ollama
    model: gemma4:e4b
  goal:
    backend: ollama
    model: gemma4:e4b
  updater:
    world:
      backend: ollama
      model: gemma4:e4b
    goal:
      backend: ollama
      model: gemma4:e4b
    agent:
      backend: ollama
      model: gemma4:e4b
    general:
      backend: ollama
      model: gemma4:e4b
```

World and goal models maintain context that feeds Agent X and updater P. On
non-controllable animation frames, orchestration synthesizes the internal
`NONE` action without calling X.

## Debug Trace Modes

The compact default output is:

```yaml
debug_trace: minimal
debug_color: auto
```

Use `debug_trace: verbose` for colored sections covering run start/stop, frame
turns, control policy, selected actions, trace metadata, tool calls/results,
S/G predictions, and M-state persistence.

Use `debug_trace: agent_decision` to show only the Agent X decision panel for
each frame turn.

Use `debug_trace: model_inputs` when you need to inspect model inputs. This
adds sanitized X/S/G/updater input sections, full text prompts, and request
metadata. Image and base64 payloads are replaced with type/size summaries, and
sensitive-looking keys such as API keys, authorization headers, cookies, and
tokens are redacted. Long text fields are wrapped before printing so narrow
terminals do not crop them.

`models.updater` is also explicit: configure `world`, `goal`, `agent`, and
`general` with `backend: openai` or `backend: ollama` and a concrete `model`.

Provider-backed model adapters are imported from role-local `providers/`
folders in code. Orchestration still receives only the role interfaces and
does not branch on concrete providers.
