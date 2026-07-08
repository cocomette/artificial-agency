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
- World image backend: Hugging Face Diffusers with `Qwen/Qwen-Image-Edit`.
- ML/runtime libraries: PyTorch, Transformers, Accelerate, Safetensors,
  SentencePiece, and Protobuf.
- Install path: `uv sync --extra ml --no-dev` for runtime ML dependencies, or
  `uv sync --group dev` for the full development environment.

## Hardware Assumptions

- CUDA is the preferred device for the Qwen image backend.
- Apple MPS is supported as a local fallback.
- CPU can run as a fallback path, but it is not a practical target for the
  current image backend.
- Hugging Face model weights are downloaded into the normal local Hugging Face
  cache on first use.

## Development Tools

- Tests: `pytest`.
- Interactive work: Jupyter Notebook and IPython kernel.
- Rendering/visualization: Matplotlib; Linux `render_mode: human` needs Tk
  installed through the system package manager.
