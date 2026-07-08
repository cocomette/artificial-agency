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
- The active world prediction model produces structured description
  predictions from visual observations; goal prediction code remains dormant.
- H100 model entry point: Modal-managed vLLM serving an OpenAI-compatible
  Chat Completions endpoint for shared X/S/P runs.
- ML/runtime libraries: PyTorch, Transformers, Accelerate, Safetensors,
  SentencePiece, and Protobuf.
- Install path: `uv sync --extra ml --no-dev` for runtime ML dependencies, or
  `uv sync --group dev` for the full development environment.

## Hardware Assumptions

- Hosted provider paths do not require local accelerator hardware.
- Local model paths may use CUDA, Apple MPS, or CPU depending on the selected
  provider and model size.
- The vLLM backend is for Modal H100 runs. The default H100 sample serves
  `Qwen/Qwen3.6-35B-A3B-FP8` on a single H100 with Triton GDN prefill and the
  vLLM chat template configured with thinking disabled.

## Development Tools

- Tests: `pytest`.
- Interactive work: Jupyter Notebook and IPython kernel.
- Rendering/visualization: Matplotlib; Linux `render_mode: human` needs Tk
  installed through the system package manager.
