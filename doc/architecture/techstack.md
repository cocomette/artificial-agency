# Tech Stack

- Python 3.12 with `uv` for environment and command execution.
- ARC-AGI-3 environment integration through the runtime environment adapter.
- OpenAI, Ollama, and vLLM provider adapters for active model roles.
- SQLite for state memory, the optional agent-creator role store,
  experimental tool memory, and model-input debug records.
- Rich terminal tracing and Streamlit dashboard tooling for local debugging.

Active game-loop model calls are change summary, world model,
agent-context historizer, and updater P for agent context. When configured, the
agent creator runs as a sidecar that writes learned roles to its own database;
those roles are not consumed by the game loop. Agent X adapters remain in the
codebase, but Agent X is dormant in the current game loop; updater P returns
the actions that orchestration queues and submits.
