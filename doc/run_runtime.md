# Run The Runtime

Copy-paste these commands from the repository root.

## Setup

Light runtime:

```bash
uv sync --no-dev
```

Runtime with model backends:

```bash
uv sync --extra ml --no-dev
```

Full development environment:

```bash
uv sync --group dev
```

## Game Catalog

Refresh the local ARC game catalog:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

This writes:

```text
src/face_of_agi/environment/local_games.json
```

## Local Runtime

Run the starter config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

Use `uv run --extra ml --no-dev ...` for configs that need model backends, or
`uv run --group dev ...` from the full development environment.

Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```

## Current Runtime Shape

The active runtime roles are:

- Agent X: chooses the final action on controllable frame turns.
- Change summary: summarizes observed visual transition evidence.
- Historizer: summarizes prior agent game-context field evolution.
- Game memory: writes compact same-run memory from action history and frames.
- Updater P: updates the agent game context during play and agent general
  context at run end.

World and goal tool modules are not part of this branch’s active runtime.
Configs with `models.world`, `models.goal`, `models.updater.world`, or
`models.updater.goal` are rejected.

## Useful Config Values

Common values to change:

- `game_index`, `game_indices`, `game_ids`, or `game_selection`.
- `max_actions_per_level`.
- `max_parallel_games` and `max_game_retries`.
- `agent_action_history_window`.
- `agent_context_history_window`.
- `agent_updater_action_history_window`.
- `experimental_memory_turn_buffer`.
- `debug_keep_all_m_states`.
- `debug_trace`: `off`, `minimal`, `agent_decision`, `verbose`, or
  `model_inputs`.
- `debug_color`: `auto`, `always`, or `never`.

Model role sections:

```yaml
models:
  scheduler:
    enabled: true
    max_concurrent_calls: 8
    max_concurrent_calls_per_game: 1
  shared_vlm:
    backend: vllm
    model: ...
  agent:
    backend: vllm
  change:
    backend: vllm
    summary_max_chars: 2000
    summary_max_elements: 20
  historizer:
    backend: vllm
    field_max_chars: 2000
  memory:
    backend: vllm
    memory_max_chars: 10000
  updater:
    general:
      backend: vllm
      general_context_max_chars: 20000
    agent:
      backend: vllm
      agent_game_context_max_chars: 12000
      agent_game_context_field_max_chars: 6000
```

vLLM roles may also set `repair_invalid_output_preview_chars`; the RTX6000
configs set it to `8000`.

## Kaggle RTX6000 Configs

The RTX6000 vLLM configs live under:

```text
src/face_of_agi/runtime/configs/vllm/
```

The debug config is intended for smaller public-game debug batches. The
parallel config is the Kaggle-style competition run. Both use the shared vLLM
server config and explicit output caps.

## Debug Trace Modes

`debug_trace: minimal` prints the compact frame-turn trace. Use
`agent_decision` to show only Agent X decisions, `verbose` for loop and
persistence details, or `model_inputs` to include sanitized model inputs.

Image and base64 payloads are summarized, sensitive-looking keys are redacted,
and long text fields are wrapped for terminal readability.
