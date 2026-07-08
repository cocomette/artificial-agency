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

Build an MP4 from persisted M-state observation frames for one run:

```bash
uv run python scripts/assemble_run_video.py \
  --run-id game-index-3-20260517T200151Z \
  --memory-file resources/shared-memory_modal.sqlite
```

The script requires `ffmpeg` on `PATH` and writes to
`runs/videos/<run>_<game>.mp4` by default. Pass `--game-id` when one run id has
frames for multiple games.
