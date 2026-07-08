# Scripts

This directory contains operational and developer scripts.

Download the default Qwen 3.6 35B FP8 Hugging Face snapshot into the
`face-of-agi-local-models` Modal Volume:

```bash
uv run --with modal modal run scripts/download_hf_model_to_modal_volume.py
```

Use `--model-id` or `--revision` to override the default Hugging Face snapshot.
Set `HF_TOKEN` locally before running the command to forward it to the remote
Modal function for authenticated Hugging Face downloads:

```bash
HF_TOKEN=hf_... uv run --with modal modal run scripts/download_hf_model_to_modal_volume.py
```

Manual end-to-end checks live under `tests/e2e/` so they can share
`tests/fixtures/` with the pytest suites.
