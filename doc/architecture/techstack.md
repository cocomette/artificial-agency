# ARC-AGI-3 Tech Stack

## Purpose

This file lists the concrete tools, frameworks, and runtime assumptions used by
the repository. Architecture and module ownership live in
`doc/architecture/system_architecture.md` and `doc/architecture/software/`.

## Core Runtime

- Language: Python 3.12.
- Environment and dependency manager: `uv`.
- Package build backend: Hatchling.
- Environment interface: ARC-AGI Toolkit.
- Persistence: SQLite through Python's built-in `sqlite3` module.
- Base install: ARC-AGI Toolkit and Matplotlib only.

## Model Runtime

- Local model entry point: Ollama where it fits a model role.
- World and goal prediction models produce structured description predictions
  from visual observations.
- ML/runtime libraries: PyTorch, Transformers, Accelerate, Safetensors,
  SentencePiece, and Protobuf.
- Install path: `uv sync --extra ml --no-dev` for runtime ML dependencies, or
  `uv sync --group dev` for the full development environment.

## Hardware Assumptions

- Hosted provider paths do not require local accelerator hardware.
- Local model paths may use CUDA, Apple MPS, or CPU depending on the selected
  provider and model size.

## Development Tools

- Tests: `pytest`.
- Interactive work: Jupyter Notebook and IPython kernel.
- Rendering/visualization: Matplotlib; Linux `render_mode: human` needs Tk
  installed through the system package manager.
