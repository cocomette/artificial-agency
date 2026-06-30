# FACE-OF-AGI

Minimal Python framework for an ARC-AGI-3 agent.

This repo contains a Python runtime shell, an ARC-AGI environment adapter,
orchestration loop scaffolding, vLLM-backed model-role adapters, text
observation serialization, and SQLite-backed memory. Architecture context lives
under `doc/architecture/`.

The only real model backend is vLLM through its OpenAI-compatible Chat
Completions API. The `openai` Python package remains a transport client for
that vLLM endpoint; it is not OpenAI provider support.

## Setup

Use Python 3.12 and `uv` from the repo root:

```bash
uv sync --no-dev
uv run --no-dev python -c "import face_of_agi"
```

Dependency profiles:

- `uv sync --no-dev`: minimal runtime plus the vLLM HTTP transport client.
- `uv sync --group test --no-dev`: lightweight regression-test environment.
- `uv sync --group debug`: local Streamlit dashboard.
- `uv sync --group dev`: full development environment with tests and notebooks.

For Linux window rendering with `render_mode: human`, install Tk once:

```bash
sudo apt-get install python3-tk
```

## First Run

Create or refresh the local game catalog before using `game_index`:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

This writes the ignored file `src/face_of_agi/environment/local_games.json`.
The starter loop uses `game_index` from
`src/face_of_agi/runtime/configs/starter_loop.yaml` to choose one of those
catalog entries.

Run the starter config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

The starter config expects a vLLM OpenAI-compatible server at the configured
`models.shared_vlm.base_url`. It runs real vLLM-backed agent, change,
historizer, and updater roles.

Quick copy-paste runtime commands also live in `doc/run_runtime.md`.

## Runtime Configs

Ready-to-run configs live under `src/face_of_agi/runtime/configs/`:

- `starter_loop.yaml`: default vLLM-first runtime config.
- `vllm/**`: hardware/model-specific vLLM runtime variants.

The model config shape is:

```yaml
models:
  observation_text:
    crop_cells: 3
    overflow_chars_per_frame: 12000
    include_rows: true
    include_component_runs: true
  shared_vlm:
    backend: vllm
    model: Qwen/Qwen3.6-35B-A3B-FP8
    base_url: http://127.0.0.1:8000/v1
    api_key: EMPTY
  agent:
    backend: vllm
    max_tool_calls: 0
    repair_attempts: 1
  change:
    backend: vllm
  historizer:
    backend: vllm
  updater:
    agent:
      backend: vllm
    general:
      backend: vllm
```

Runtime rejects OpenAI, Ollama, HuggingFace, Diffusers, world, and goal backend
keys. Model-facing observations are serialized as cropped text with original ARC
grid coordinates and uppercase hex symbols `0..F`. Image payloads, image URLs,
base64 data URLs, and image input config fields are not accepted by model-input
capture.

Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```

If you installed the dev environment, use the same commands with
`uv run --group dev` instead of `uv run --no-dev`.

For the full config reference, see `doc/architecture/software/config.md`.
For runtime notes and copy-paste command variants, see `doc/run_runtime.md`.

## Debug Dashboard

The local Streamlit dashboard can launch saved runtime configs and inspect
persisted FACE-OF-AGI memory turns from SQLite. ARC and vLLM run only when you
explicitly click `RUN config` in the Runner page.

```bash
uv sync --group debug
uv run --group debug streamlit run debug/dashboard/app.py -- --database runs/memory.sqlite
```

Normal runtime runs prune `M` to the latest state per game. For a debug run
where every `m_states` row should remain inspectable, set
`debug_keep_all_m_states: true` in the runtime YAML.

The Runner page uses the same runtime shell entrypoint as terminal runs and
includes a collapsible config editor under the config selector. It lists YAML
files from `src/face_of_agi/runtime/configs/`, validates edits, and supports
`Save` or `Save As`. The sidebar can clear the selected SQLite memory database
through the runtime shell's `--clean-db` path.

## Tests

Run the model-free regression suite:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

GitHub Actions runs this command automatically for pull requests. This gate
uses only the lightweight `test` dependency group and does not call external
model APIs or a live vLLM server.

Testing details live in `doc/test/test_suite.md` and `doc/test/end_to_end.md`.

## License

Project source, scripts, configs, docs, and supporting materials are offered
under `Apache-2.0`. Third-party dependencies, datasets, model weights, and
other external artifacts remain under their own license terms. See `LICENSE`,
`NOTICE`, and `THIRD_PARTY_LICENSES.md`.

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
- `kaggle/README.md`: optional Kaggle notebook build and upload workflow.
