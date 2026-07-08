# FACE-OF-AGI

Minimal Python framework for an ARC-AGI-3 agent.

This repo contains a Python runtime shell, an ARC-AGI environment adapter,
orchestration loop scaffolding, model-role adapters, and SQLite-backed memory.
Architecture context lives under `doc/architecture/`.

Model-specific contracts and adapter shells live under `src/face_of_agi/models`.
The active runtime model calls are change summary, world model,
agent-context historizer, and updater P tasks for agent game context plus agent
general knowledge. Agent X adapters remain in the tree, but Agent X is dormant
in the current game loop; updater P selects the actions that orchestration
queues and submits to the environment.
Deterministic orchestration stays in `src/face_of_agi/orchestration`.

## Setup

Use Python 3.12 and `uv` from the repo root:

```bash
uv sync --no-dev
uv run --no-dev python -c "import face_of_agi"
```

Dependency profiles:

- `uv sync --no-dev`: minimal runtime for the starter shell.
- `uv sync --extra ml --no-dev`: runtime plus OpenAI, Ollama, Torch, and
  local/hosted model backends.
- `uv sync --group test --no-dev`: lightweight regression-test environment.
- `uv sync --group debug`: local Streamlit dashboard.
- `uv sync --group dev`: full development environment with tests, notebooks,
  and model backend dependencies.

For Linux window rendering with `render_mode: human`, install Tk once:

```bash
sudo apt-get install python3-tk
```

## First Run

Create or refresh the local game catalog before using `game_index` or
`game_indices` from a runtime config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

This writes `src/face_of_agi/environment/local_games.json`. Runtime configs use
`game_index`, `game_indices`, `game_ids`, or `game_selection` to choose games.

Quick copy-paste runtime commands also live in `doc/run_runtime.md`.

Ready-to-run local config variants live under
`src/face_of_agi/runtime/configs/`:

- `starter_loop.yaml`: hosted OpenAI active roles, small action budget.
- `openai/openai_all_gpt5_nano_test.yaml`: cheaper hosted OpenAI loop.
- `openai/openai_all_gpt55_image2_test.yaml`: higher-quality hosted OpenAI
  loop.
- `ollama/ollama_all_gemma4_26b.yaml`: fully local Ollama loop.
- `ollama/ollama_shared_*.yaml`: shared local Ollama model configs.
- `vllm/vllm_h100_qwen36_35b_fp8.yaml`: Modal H100 vLLM loop using
  `Qwen/Qwen3.6-35B-A3B-FP8`.
- `vllm/vllm_h100_qwen36_35b_fp8_parallel.yaml`: Modal H100 vLLM loop that
  runs multiple selected games concurrently against one vLLM server.
- `vllm/vllm_rtx6000_qwen36_35b_fp8_parallel.yaml`: Kaggle RTX 6000 vLLM
  submission config that runs all available evaluation games.
- `vllm/vllm_rtx6000_qwen36_35b_fp8_debug.yaml`: Kaggle RTX 6000 vLLM debug
  config that runs a small public-game batch and preserves SQLite history.

## Starter Config

The starter config supports:

- `game_index`
- `game_indices`
- `game_ids`
- `game_selection`
- `max_parallel_games`
- `max_game_retries`
- `max_actions_per_level`
- `max_levels_per_game`
- `operation_mode`
- `game_catalog_path`
- `environments_dir`
- `recordings_dir`
- `enable_visualization`
- `render_mode`
- `seed`
- `save_recording`
- `cheat_action_context`
- `cheat_action_context_game_dir`
- `use_learned_contexts`
- `experimental_memory_turn_buffer`
- `agent_action_history_window`
- `action_history_window`
- `agent_context_history_window`
- `debug_keep_all_m_states`
- `debug_trace`
- `debug_color`
- `models.shared_vlm.backend`
- `models.agent.backend`
- `models.historizer.backend`
- `models.updater.agent_probing.backend`
- `models.updater.agent_policy.backend`
- `models.updater.general.backend`

Visualization is optional and environment-local. Set
`enable_visualization: true` in
`src/face_of_agi/runtime/configs/starter_loop.yaml` to display outgoing ARC frame
bundles. Supported render modes are:

- `render_mode: human`
- `render_mode: terminal`
- `render_mode: terminal-fast`

`experimental_memory_turn_buffer` controls the rolling `E` experiment buffer.
It defaults to `2`, meaning tool-produced experimental outputs remain
referenceable for the latest two frame turns per run and game.

`agent_action_history_window` is retained for the dormant Agent X path.
`action_history_window` controls the prior action history included in active
model inputs that receive action history before the current transition is
appended. Both default to `8` prior controllable action groups, including
bundled synthetic `NONE` animation evidence, real environment actions, and
changed-pixel percentages from the change-summary visible frame area. Set
either to `0` to disable that prior history for the corresponding model input.

`agent_context_history_window` controls how many recent same-run same-game
agent game contexts are summarized by the historizer before the agent updater
runs. It defaults to `8`; set it to `0` to pass `not available` instead. The
historizer runs only after at least two prior same-run contexts are available.

Post-action animation bundles are passed to change and world-model roles as
ordered image arrays and land directly on their final controllable frame.
Those animation inputs are cropped normally, then resized so the full bundle
fits within two configured input-frame areas.

`debug_keep_all_m_states` keeps every persisted frame turn in `M` after a
successful runtime run. It defaults to `false`; enable it only for debug runs.

`debug_trace` controls stdout runtime tracing. It defaults to `minimal`, which
preserves the compact per-frame trace. Use `off` to suppress trace lines,
`agent_decision` to print only the frame decision panel, `verbose` for
colored loop/agent/tool/persistence details, or `model_inputs` to also print
sanitized model inputs for active model roles. Model-input tracing prints
full text prompts and request metadata, but image/base64 payloads are
summarized and sensitive-looking keys are redacted. Long text fields are
wrapped before printing so they remain readable in narrow terminals.

`debug_color` controls Rich terminal coloring for debug traces: `auto`,
`always`, or `never`. It defaults to `auto`.

Set `cheat_action_context: true` to seed the agent's initial mutable game
context with action semantics parsed from the local game source. For example,
`ls20` maps `ACTION1` to up, `ACTION2` to down, `ACTION3` to left, and
`ACTION4` to right. After startup this text is ordinary updater-maintained
`role_context`: the updater may preserve, rewrite, shorten, or remove it. The
runtime infers the source directory from `game_id`; use
`cheat_action_context_game_dir` only when that local path needs an override.

`use_learned_contexts` defaults to `true`. Set it to `false` for a fresh run
that ignores persisted learned `K` and `L` contexts in SQLite while leaving the
database rows available for inspection.

The starter config requires real model providers for change summary, world,
historizer, and the active updater slots: agent probing, agent
policy, and general. Agent X is dormant; `models.agent` may remain in old
configs but is not used by the current game loop. For a local Ollama run,
configure the active updater roles:

```yaml
models:
  agent:
    backend: ollama
    model: gemma4:e4b
    max_tool_calls: 2
    repair_attempts: 1
  change:
    backend: ollama
    model: gemma4:e4b
  historizer:
    backend: ollama
    model: gemma4:e4b
  updater:
    agent:
      backend: ollama
      model: gemma4:e4b
    general:
      backend: ollama
      model: gemma4:e4b
```

For a hosted OpenAI run:

```yaml
models:
  agent:
    backend: openai
    model: gpt-5-nano
    max_tool_calls: 2
    repair_attempts: 1
  change:
    backend: openai
    model: gpt-5-nano
  historizer:
    backend: openai
    model: gpt-5-nano
  updater:
    agent:
      backend: openai
      model: gpt-5-nano
    general:
      backend: openai
      model: gpt-5-nano
```

For a ready-to-run multi-turn `ls20` debug run with cheat action context and
OpenAI-hosted active roles:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

The starter loop runs the configured ARC game and writes persistent state memory
under `runs/` by default. It is intentionally small: edit the YAML when you want
to change the game, action budgets or caps, debug output, model backends, or
updater backends.

Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```

If you installed the dev environment, use the same commands with
`uv run --group dev` instead of `uv run --no-dev`.

## Modal H100 Runs

Modal support is isolated to `face_of_agi.runtime.modal_app` and reuses the
existing runtime shell. Install or inject Modal locally, then launch a remote
single H100 run:

```bash
uv run --with modal modal run src/face_of_agi/runtime/modal_app.py --config src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml
```

The Modal app mounts two Volumes:

- `face-of-agi-local-models` at `/vol/models` for Ollama models, vLLM/Hugging
  Face model caches, and Diffusers caches.
- `face-of-agi-runs` at `/vol/runs` for copied configs, SQLite memory, and run
  artifacts.

While a run is active, the Modal runner commits the run volume every 30 seconds
by default so debug tools can inspect the latest committed `memory.sqlite`.
Change the interval, or disable live commits with `0`, through
`--live-commit-seconds`. The remote runtime shell's stdout and stderr are
streamed through Modal logs while the run is active.

The sample Modal H100 config uses vLLM for transition change summaries, world,
historizer, and active updater tasks. Modal starts `vllm serve` inside the H100
container and the runtime talks to `http://127.0.0.1:8000/v1` through Chat
Completions. Agent X is dormant. Goal entries in a config are ignored by Modal
model discovery. If a config defines
`models.shared_vlm`, matching local Ollama or vLLM roles can inherit its model
and runtime options, including typed role fields such as `repair_attempts`,
while still overriding role-specific settings. The H100 sample carries the vLLM
server flags needed for Qwen3.6 FP8, including Triton GDN prefill and disabled
thinking in the chat template.

For parallel vLLM runs, configure `game_indices` and optional
`max_parallel_games` instead of `game_index`. Each game gets its own SQLite file
derived from the requested database name, such as
`memory-game-index-3.sqlite`, while all workers send concurrent requests to the
same vLLM server. Do not force the vLLM server to one sequence with flags such
as `--max-num-seqs 1` when you want batched parallel inference. Set
`max_game_retries` to retry failed games with isolated retry run ids and
database files; Kaggle retries reuse the same Competition Mode wrapper and
reset it. Runtime entrypoints may pass `RuntimeConfig.deadline_monotonic`, which
is not a YAML setting, to stop cleanly before a platform hard timeout.

## Kaggle RTX 6000 Submission

The direct FACE-OF-AGI Kaggle submission workspace lives in `kaggle/`. It builds
an offline RTX 6000 notebook, uploads a dependency wheelhouse and model weights
as Kaggle inputs, starts vLLM inside the notebook, and runs
`python -m face_of_agi.runtime.kaggle` across all available evaluation games.
See `kaggle/README.md`.

The same workspace also has a separate RTX 6000 debug notebook for public-game
iteration. It uses an offline public-games dataset, runs the normal runtime
shell, writes SQLite files under `/kaggle/working/runs`, and pulls them locally
under `runs/kaggle-debug/` for dashboard inspection.

## Model Runs

For hosted OpenAI model configs, install model dependencies and provide
`OPENAI_API_KEY` through the environment or a local `.env` file:

```bash
uv sync --extra ml --no-dev
uv run --env-file .env --extra ml --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/openai/openai_all_gpt5_nano_test.yaml
```

For local Ollama configs, start Ollama and pull the configured model first:

```bash
uv sync --extra ml --no-dev
ollama serve
ollama pull gemma4:e4b
uv run --extra ml --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/ollama_x_local.yaml
```

Use `uv run --group dev ...` for the same commands if you synced the full dev
environment.

For the full config reference, see `doc/architecture/software/config.md`.
For runtime notes and copy-paste command variants, see `doc/run_runtime.md`.

## Debug Dashboard

The local Streamlit dashboard can launch saved runtime configs and inspect
persisted FACE-OF-AGI memory turns from SQLite. ARC, OpenAI, Ollama, or local
model backends run only when you explicitly click `RUN config` in the Runner
page.

```bash
uv sync --group debug
uv run --group debug streamlit run debug/dashboard/app.py -- --database runs/memory.sqlite
```

To inspect or launch a Modal run from your local browser, run the same
dashboard in Modal mode. It pulls `/vol/runs/memory.sqlite` from the
`face-of-agi-runs` Modal Volume into `runs/modal-memory.sqlite` before each
live refresh:

```bash
uv run --group debug --with modal streamlit run debug/dashboard/app.py -- --modal
```

Use `--modal-volume`, `--modal-database`, `--modal-snapshot`, or
`--local-database` after Streamlit's `--` separator for non-default Modal
Volume, remote SQLite path, local snapshot path, or local-run database path.

Normal runtime runs prune `M` to the latest state per game. For a debug run
where every `m_states` row should remain inspectable, set
`debug_keep_all_m_states: true` in the runtime YAML.

The Runner page uses the same runtime shell entrypoint as terminal runs and
includes a collapsible config editor under the config selector. It lists YAML
files from `src/face_of_agi/runtime/configs/`, validates edits, and supports
`Save` or `Save As`. The sidebar can clear the selected SQLite memory database
through the runtime shell's `--clean-db` path and reset stale disposable local
SQLite files when the schema is obsolete. Live Play and Offline Inspector
treat one `m_states` row as one frame turn and show the current frame, decision
trace, selected action, matching experimental tool outputs from `E`, persisted
agent context, and raw redacted JSON for inspection.

SQLite run databases created before the agent-context-only `m_states` schema
are incompatible with this branch. Reset or delete disposable local run DBs
before running the updated runtime.

## Human Baseline Scoring

Build a compact JSON summary from ARC-AGI human baseline recording summaries:

```bash
uv run --no-dev python debug/scoring/build_human_baseline.py
```

The tool reads one game folder per immediate child of the input root and writes
`debug/scoring/human_baseline.json`. To override paths:

```bash
uv run --no-dev python debug/scoring/build_human_baseline.py \
  --input-root arc_agi_3_public_demo_human_testing/public_games-dataset \
  --output debug/scoring/human_baseline.json
```

Build per-game, per-level statistics from the baseline:

```bash
uv run --no-dev python debug/scoring/build_human_baseline_stats.py
```

The stats tool writes JSON to `debug/scoring/human_baseline_stats.json` by default.
To override paths:

```bash
uv run --no-dev python debug/scoring/build_human_baseline_stats.py \
  --input debug/scoring/human_baseline.json \
  --output debug/scoring/human_baseline_stats.json
```

## Tests

Run the model-free regression suite:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

GitHub Actions runs this command automatically for pull requests. This gate uses
only the lightweight `test` dependency group and does not call hosted models,
Ollama, local model backends, or manual E2E runners in `tests/e2e/`.

Testing details live in `doc/test/test_suite.md` and `doc/test/end_to_end.md`.

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

- `doc/architecture/system_architecture.md`: high-level agent architecture.
- `doc/architecture/software/`: target software module boundaries.
- `doc/architecture/software/config.md`: runtime config reference.
- `doc/test/`: regression and end-to-end test commands.
- `doc/architecture/techstack.md`: current tools, frameworks, and runtime stack.
- `doc/run_runtime.md`: runtime command notes.
