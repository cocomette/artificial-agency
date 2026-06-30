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
- Rendering/visualization: Matplotlib.
- Base install: ARC-AGI Toolkit, Matplotlib, Rich, PyYAML, and the OpenAI SDK
  used only as the vLLM OpenAI-compatible HTTP client.

## Model Runtime

- Real model backend: vLLM.
- Transport: vLLM's OpenAI-compatible Chat Completions API.
- Model-facing observations: `ObservationText` serialization of native 2D ARC
  integer grids plus cropped PNG image data URLs for frame-consuming vLLM roles.
- Runtime configs: `src/face_of_agi/runtime/configs/starter_loop.yaml` and
  `src/face_of_agi/runtime/configs/vllm/**`.

OpenAI, Ollama, HuggingFace, and Diffusers provider paths are not part of the
current runtime stack.

## Hardware Assumptions

- vLLM is expected to run outside the base Python environment, locally or on
  dedicated GPU infrastructure.
- The runtime process only needs network access to the configured vLLM
  OpenAI-compatible endpoint.
- Hardware-specific vLLM launch/config variants live under
  `src/face_of_agi/runtime/configs/vllm/`.

## Development Tools

- Tests: `pytest`.
- Interactive work: Jupyter Notebook and IPython kernel.
- Rendering/visualization: Matplotlib; Linux `render_mode: human` needs Tk
  installed through the system package manager.
