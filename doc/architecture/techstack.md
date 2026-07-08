# Tech Stack

- Python runtime with `uv` for dependency management.
- ARC-AGI-3 environment adapters from the ARC toolkit.
- Hugging Face `transformers` and PyTorch for the frozen in-process vision
  backbone.
- SQLite for per-game learner memory and debug traces.
- Streamlit for the local debug dashboard.
- Kaggle notebooks plus offline wheelhouse/model inputs for submissions.

The default test suite uses deterministic fake backbones and does not load real
model weights or call external services.
