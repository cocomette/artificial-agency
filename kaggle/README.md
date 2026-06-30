# Kaggle ARC-AGI-3 Submission And Debug

This directory builds the RTX 6000 Kaggle submission notebook and the separate
public-game debug notebook for FACE-OF-AGI.

## Setup

Setup is required only once for a new Kaggle project. Rerun individual upload
commands later only when that artifact needs a new Kaggle version.

1. Copy `kaggle/.env.example` to `kaggle/.env`, then set your Kaggle username
   and token path:

   ```bash
   FACE_OF_AGI_KAGGLE_OWNER=<your-kaggle-username>
   FACE_OF_AGI_KAGGLE_TOKEN_FILE=.kaggle/access_token
   ```

   `FACE_OF_AGI_KAGGLE_TOKEN_FILE` is resolved from the repo root when it is a
   relative path.
2. Save a Kaggle API token at that configured path.

The Makefile syncs user-specific Kaggle metadata from `kaggle/.env` before
notebook and upload commands. Tracked metadata files use owner-neutral
placeholders; `make sync-metadata` rewrites kernel, dataset, and model owner
fields from `FACE_OF_AGI_KAGGLE_OWNER`.

The Makefile runs the Kaggle CLI through `uv run --with kaggle kaggle`, so a
global `kaggle` executable is not required. Submission targets request
`NvidiaRtxPro6000` by default.

Build and upload the offline Python wheelhouse:

```bash
cd kaggle
make wheelhouse-upload UPLOAD_MODE=create
```

This uploads the wheels needed by the offline notebooks. Use
`UPLOAD_MODE=version` for later wheelhouse updates.

The competition notebook expects the model-weight Kaggle Dataset configured by
`KAGGLE_SUBMISSION_MODEL_DATASET_SLUG`. The default is
`face-of-agi-qwen36-35b-fp8-weights`.

Prepare and upload the public-game dataset for the debug notebook:

```bash
make public-games-upload UPLOAD_MODE=create
```

This packages the current public ARC normal-mode games. Use
`UPLOAD_MODE=version` for later public-game dataset updates.

## Running

Build the competition notebook locally without uploading:

```bash
make notebook
```

This writes `kaggle/notebooks/submission.ipynb`. The notebook archive embeds
only `src/face_of_agi` and `pyproject.toml`; Kaggle runtime artifacts are
attached as datasets through `kernel-metadata.json`.

Build and submit the competition notebook:

```bash
make submit
make status
```

`make submit` builds `submission.ipynb` and pushes it for Kaggle Save & Run All.
After the run completes, use Kaggle's "Submit to Competition" action and select
`submission.parquet`.

Build and submit the debug notebook:

```bash
make debug-submit
make debug-status
```

The debug notebook uses the same wheelhouse and model dataset, attaches the
public-game dataset, starts vLLM, and runs a small public-game batch.
`make debug-submit` regenerates the debug kernel id from the current branch and
8-character HEAD commit id before pushing.

To submit a debug notebook with a different vLLM runtime config and model
dataset, pass both values to the Make target. Use a title suffix when the
debug run should create a separate Kaggle kernel instead of versioning the
default debug kernel:

```bash
make debug-submit \
  KAGGLE_DEBUG_CONFIG=src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_qwen36_35b_fp8_debug.yaml \
  KAGGLE_DEBUG_MODEL_DATASET_SLUG=face-of-agi-qwen36-35b-fp8-weights \
  DEBUG_KERNEL_TITLE_SUFFIX=qwen36
```

Download debug SQLite outputs:

```bash
make debug-pull
```

Set `DEBUG_KERNEL` to either `owner/slug` or a `https://www.kaggle.com/code/...`
URL to pull or inspect a run launched elsewhere.

This writes local snapshots under `../runs/kaggle-debug/`.

Inspect one pulled SQLite file with the dashboard:

```bash
cd ..
uv run --group debug streamlit run debug/dashboard/app.py -- --database runs/kaggle-debug/<sqlite-file>
```
