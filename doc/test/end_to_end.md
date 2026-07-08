# End-To-End Checks

Use these checks when you need to exercise model adapters or full runtime paths
outside the model-free regression suite. Run commands from the repo root.

E2E runners live under `tests/e2e/`, share fixtures from `tests/fixtures/`, and
write artifacts under `runs/`.

## Local World Model

Install model dependencies:

```bash
uv sync --extra ml --no-dev
```

Run the local world-model check:

```bash
uv run --locked --extra ml --no-dev python tests/e2e/world_model_e2e.py
```

This path can download local image-edit model weights on first use.

## OpenAI

Install model dependencies and provide `OPENAI_API_KEY` through the environment
or a local `.env` file:

```bash
uv sync --extra ml --no-dev
```

Run role-level OpenAI checks:

```bash
uv run --env-file .env --locked --extra ml --no-dev python tests/e2e/openai_world_model_e2e.py
uv run --env-file .env --locked --extra ml --no-dev python tests/e2e/openai_goal_model_e2e.py
uv run --env-file .env --locked --extra ml --no-dev python tests/e2e/openai_orchestrator_agent_e2e.py
```

Run OpenAI S/G updater checks:

```bash
uv run --env-file .env --locked --extra ml --no-dev python tests/e2e/openai_updater_e2e.py
```

Run the full OpenAI game-loop check:

```bash
uv run --env-file .env --group dev python tests/e2e/openai_full_game_loop_e2e.py
```

Run the full loop with cheat action context:

```bash
uv run --env-file .env --group dev python tests/e2e/openai_full_game_loop_e2e.py --cheat-action-context
```

Cheat action context requires local game source files for the selected game.

## Ollama

Install model dependencies, start Ollama, and pull the configured model:

```bash
uv sync --extra ml --no-dev
ollama serve
ollama pull gemma4:e4b
```

Run the Ollama orchestrator check:

```bash
uv run --locked --extra ml --no-dev python tests/e2e/ollama_orchestrator_agent_e2e.py
```

Run the Ollama two-image change-description check:

```bash
uv run --locked --extra ml --no-dev python tests/e2e/ollama_image_change_e2e.py
```

Run the Ollama image-2 change check from an image-1 description:

```bash
uv run --locked --extra ml --no-dev python tests/e2e/ollama_image_change_2_e2e.py
```

Run the Ollama single-image structured area-description check:

```bash
uv run --locked --extra ml --no-dev python tests/e2e/ollama_image_description_e2e.py
```
