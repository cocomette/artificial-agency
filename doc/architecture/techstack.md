# Tech Stack

- Python 3.12 with `uv` for environment and command execution.
- ARC-AGI-3 environment integration through the runtime environment adapter.
- vLLM Chat Completions for active model-role inference.
- SQLite for state memory, v1 role artifacts, rewards, experimental tool
  memory, and model-input debug records.
- Rich terminal tracing and Streamlit dashboard tooling for local debugging.

Active model roles are Agent X, Change Summary, Memory, World, Goal, Interest,
and Reward Judge. The v1 state-memory schema requires disposable older run
databases to be reset.
