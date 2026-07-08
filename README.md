# FACE-OF-AGI VLM Online Adapter

This branch implements the Level 0 online learner runtime for ARC-AGI-3.
The primary target is Kaggle submission execution with bundled Hugging Face
Transformers model weights. The frozen backbone runs in-process, exposes visual
representations, and only small online learner components update during a run.

There is no active vLLM, hosted API, Ollama, prompt-role, playback, or updater
runtime path in this branch.

## Runtime

The runtime config uses `agent:`, not `models:`:

```yaml
agent:
  backbone:
    backend: transformers
    model_family: qwen3_5_moe_multimodal
    model_path: /kaggle/input/face-of-agi-qwen36-35b-fp8-weights
    processor_path: null
    device: auto
    dtype: auto
    image_size: 224x224
    local_files_only: true
    representation_layer: image_tokens_mean
    model_kwargs:
      device_map: auto
  online:
    buffer_size: 512
    adapter_rank: 16
    ensemble_size: 5
    hidden_dim: 512
    learning_rate: 0.001
    batch_size: 32
  replay:
    max_updates_per_turn: 8
    max_seconds_per_turn: 0.5
    solved_level_updates: 32
  planner:
    horizon: 3
    candidate_count: 64
    coordinate_candidates: 16
    diagnostic_turns: 4
```

Committed configs:

- `src/face_of_agi/runtime/configs/kaggle_transformers.yaml`: Competition Mode
  submission config. The Kaggle entrypoint overrides operation mode to
  competition and runs one worker per selected game with isolated SQLite files.
- `src/face_of_agi/runtime/configs/kaggle_debug_transformers.yaml`: Kaggle
  public-games debug config.
- `src/face_of_agi/runtime/configs/starter_loop.yaml`: local shell harness for
  the same Transformers-backed agent.

## Local Shell

Install test/runtime dependencies with uv, then run a local game config:

```bash
uv run --group dev python -m face_of_agi.runtime.shell \
  --config src/face_of_agi/runtime/configs/starter_loop.yaml \
  --database runs/memory.sqlite
```

The local shell requires the configured Transformers model path to exist. For
model-free automated tests, fake deterministic backbones are used directly in
test code instead of through YAML.

Reset disposable local SQLite state after schema changes:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell \
  --config src/face_of_agi/runtime/configs/starter_loop.yaml \
  --database runs/memory.sqlite \
  --clean-db
```

## Kaggle

The Kaggle path expects:

- a model input named `face-of-agi-qwen36-35b-fp8-weights` containing the
  Hugging Face Qwen3.6 35B FP8 model and processor files at the dataset root;
- a `face-of-agi-transformers-wheelhouse` dataset built from
  `kaggle/requirements-kaggle.txt`;
- Kaggle's ARC gateway/scorecard environment during competition reruns.

Build the offline wheelhouse:

```bash
cd kaggle
make wheelhouse
```

Build the submission notebook:

```bash
cd kaggle
make notebook
```

Submit when Kaggle credentials are configured in `kaggle/.env`:

```bash
cd kaggle
make submit
```

The generated notebook installs offline dependencies, rewrites the configured
model path to the Kaggle input, and invokes `face_of_agi.runtime.kaggle`
directly. It does not start a model server.

## Memory And Dashboard

SQLite `m_states` rows store committed learner turns: current observation,
chosen action, learner snapshot, learner trace, turn metrics, and run metadata.
Learner artifacts are stored in `learner_artifacts`; old `E` experiment tables
are not migrated.

Run the debug dashboard:

```bash
uv run --group dev streamlit run debug/dashboard/app.py
```

Dashboard verification is manual. Automated tests do not cover dashboard UI.

## Tests

Run the model-free regression suite:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

The default tests use fake backbones and do not load real model weights, call
hosted APIs, or start external model servers.

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

- `doc/architecture/system_architecture.md`: high-level learner architecture.
- `doc/architecture/software/`: target software module boundaries.
- `doc/architecture/software/config.md`: runtime config reference.
- `doc/run_runtime.md`: runtime command notes.
- `doc/test/`: test commands and scope.
