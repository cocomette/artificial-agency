# FACE-OF-AGI

Minimal Python framework for an ARC-AGI-3 agent.

This branch runs a frame-unrolled ARC game loop with these active model roles:
Agent X, transition change summary, agent-context historizer, same-run game
memory, and updater P. Runtime orchestration owns environment stepping,
fallbacks, persistence, and model-role ordering. Architecture context lives
under `doc/architecture/`.

## Setup

Use Python 3.12 and `uv` from the repo root:

```bash
uv sync --no-dev
uv run --no-dev python -c "import face_of_agi"
```

Dependency profiles:

- `uv sync --no-dev`: minimal runtime.
- `uv sync --extra ml --no-dev`: runtime plus OpenAI, Ollama, vLLM, and local
  model backend dependencies.
- `uv sync --group test --no-dev`: lightweight regression-test environment.
- `uv sync --group debug`: local Streamlit dashboard.
- `uv sync --group dev`: full development environment.

## First Run

Create or refresh the local game catalog before using `game_index`:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

Run the starter config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

Runtime notes and copy-paste command variants live in `doc/run_runtime.md`.

## Runtime Config

Current configs declare:

- `models.agent`: Agent X decision role.
- `models.change`: transition change-summary role.
- `models.historizer`: prior agent-context history summarizer.
- `models.memory`: same-run game memory summarizer.
- `models.updater.agent`: agent game-context updater.
- `models.updater.general`: shared end-of-run general updater.

Removed world/goal config keys are rejected. The active updater slots are
`agent` and `general`.

Useful top-level settings include `game_index`, `game_selection`,
`max_actions_per_level`, `max_parallel_games`, `max_game_retries`,
`agent_action_history_window`, `agent_context_history_window`,
`agent_updater_action_history_window`, `debug_keep_all_m_states`,
`debug_trace`, and `debug_color`.

Structured-output caps are configurable on the model role configs. The RTX6000
vLLM configs set change and historizer field caps, memory caps, updater caps,
and bounded invalid-output repair previews explicitly.

## Debug Dashboard

The local Streamlit dashboard can launch saved runtime configs and inspect
persisted FACE-OF-AGI memory turns from SQLite.

```bash
uv sync --group debug
uv run --group debug streamlit run debug/dashboard/app.py -- --database runs/memory.sqlite
```

Normal runtime runs prune `M` to the latest state per game. For a debug run
where every `m_states` row should remain inspectable, set
`debug_keep_all_m_states: true` in the runtime YAML.

## Tests

Run the model-free regression suite:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

During development, this branch’s requested suite is:

```bash
uv run pytest tests/suites -q
```

The suite does not call hosted model APIs. Manual E2E runners in `tests/e2e/`
are opt-in only.

## Pull Requests

Pull requests also validate the source branch name. Use lowercase kebab-case
after one of these prefixes:

```text
feat/<short-summary>
fix/<short-summary>
docs/<short-summary>
test/<short-summary>
refactor/<short-summary>
chore/<short-summary>
ci/<short-summary>
audit/<short-summary>
release/<short-summary>
wp/<work-package-or-step-summary>
```

## Docs

- `doc/architecture/software/`: active software module boundaries.
- `doc/architecture/software/config.md`: runtime config reference.
- `doc/run_runtime.md`: runtime command notes.
- `doc/test/`: regression and end-to-end test commands.
