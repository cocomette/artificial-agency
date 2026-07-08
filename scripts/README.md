# Scripts

This directory is reserved for future operational or developer scripts.

Manual end-to-end checks live under `tests/e2e/` so they can share
`tests/fixtures/` with the pytest suites.

## Modal Hugging Face model download

Download the default Gemma FP8 snapshot into the `face-of-agi-local-models`
Modal Volume:

```bash
uv run --with modal modal run scripts/download_modal_hf_model.py
```

Download a different Hugging Face repo:

```bash
uv run --with modal modal run scripts/download_modal_hf_model.py --repo-id Qwen/Qwen3.6-35B-A3B-FP8
```

For gated repos, create a Modal secret containing `HF_TOKEN`, then pass the
secret name:

```bash
uv run --with modal modal secret create huggingface-token HF_TOKEN=...
uv run --with modal modal run scripts/download_modal_hf_model.py --hf-secret huggingface-token
```
