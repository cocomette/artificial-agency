# FACE-OF-AGI

FACE-OF-AGI is our research codebase for exploring agentic approaches to [ARC-AGI-3](https://arcprize.org/arc-agi/3): an interactive benchmark where agents must discover goals, learn from feedback, and act efficiently in unfamiliar environments. Our work focused on the ARC Prize 2026 Kaggle competition challenge, where agents must operate under limited resources and without internet access.

## Overview

This `main` branch contains the concept we submitted to the ARC-AGI competition. It is one of the later concepts we tested, and it placed us in the top 10 at the mid-competition milestone on June 30, 2026.

The repo contains different solution concepts across different branches, ranging from simpler model setups to orchestrator-based approaches where agents are exposed as tools. The submitted solution used Qwen 3.6 35B served locally through vLLM.

The common architecture shape is an orchestration layer between the game environment, the model harness, and memory. Game data, actions, model I/O, and contexts are stored in SQLite, where they can be used by the agent and inspected through the debug dashboard.

## Branches

The repository is organized around experimental branches:

- `feat/*` branches explore separate ideas, workflows, and research directions.
- Each branch updates its own `README.md` and `doc/` folder according to the concept it describes.
- When reading a branch, use that branch's documentation as the source of context.



## Repository Map

- `src/face_of_agi/` - project source code
- `doc/` - branch-specific project and architecture notes
- `debug/` - local debugging, inspection, and analysis tools with a debug dashboard app.
- `tests/` - regression tests for supported mechanics
- `kaggle/` - Kaggle notebook build and upload workflow



## Setup

Use Python 3.12 and `uv` from the repo root:

```bash
uv sync --no-dev
uv run --no-dev python -c "import face_of_agi"
```



## First Run

Create or refresh the local ARC-AGI game catalog before using `game_index`:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

This writes the ignored file `src/face_of_agi/environment/local_games.json`. The starter loop uses `game_index` from `src/face_of_agi/runtime/configs/starter_loop.yaml` to choose one of those catalog entries.

Run the starter config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

The starter config expects a vLLM OpenAI-compatible server at the configured `models.shared_vlm.base_url`. It runs real vLLM-backed agent, change, historizer, and updater roles.

## Runtime Configs

Ready-to-run configs live under `src/face_of_agi/runtime/configs/`. `starter_loop.yaml` is the default entry point, and the `vllm/` folder contains hardware and model-specific variants.

Quick copy-paste runtime commands live in `doc/run_runtime.md`. The full config reference lives in `doc/architecture/software/config.md`.

Clear memory database rows without starting ARC:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --clean-db
```



## Debug Dashboard

The local Streamlit dashboard was built to make runs inspectable while developing agents. It can launch saved runtime configs, edit YAML configs from the browser, inspect persisted SQLite memory turns, review saved runs, and replay agent behavior frame by frame.

```bash
uv sync --group debug
uv run --group debug streamlit run debug/dashboard/app.py -- --database runs/kaggle-debug/runs
```

The `--database` argument points to a folder of SQLite memory files. Dashboard-launched runs write to `memory.sqlite` inside that folder, and the scoring view reads the SQLite files in that folder. The dashboard is useful for inspecting runs, running local configs easily, inspecting results live, running focused E2E tests, and computing scores according to the ARC-AGI-3 scoring method.

The dashboard works with local runs as well as remote Modal and Kaggle runs through memory-file pulling.





## Tests

Run the model-free regression suite:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

GitHub Actions runs this command automatically for pull requests. This gate uses only the lightweight `test` dependency group and does not call external model APIs or a live vLLM server.

Testing details live in `doc/test/test_suite.md` and `doc/test/end_to_end.md`.

## Reports And References

The following project report summarizes the work, explored ideas, results, and lessons learned:
[FACE-OF-AGI Report](https://github.com/cocomette/artificial-agency/blob/main/doc/public%20report.pdf)


Useful starting points:

- [ARC-AGI-3](https://arcprize.org/arc-agi/3)
- [ARC Prize 2026 ARC-AGI-3 Kaggle Competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)



## License

Project source, scripts, configs, docs, and supporting materials are offered under `Apache-2.0`. Third-party dependencies, datasets, model weights, and other external artifacts remain under their own license terms.