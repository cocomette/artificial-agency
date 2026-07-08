# Kaggle ARC-AGI-3 Submission And Debug

This directory builds the RTX 6000 Kaggle submission notebook and the separate
public-game debug notebook for FACE-OF-AGI.

## Setup

Setup is required only once for a new Kaggle project. Rerun individual upload
commands later only when that artifact needs a new Kaggle version.

1. Accept the ARC Prize 2026 ARC-AGI-3 Kaggle rules.
2. Set your Kaggle username and token path in `kaggle/.env`. The default
   `FACE_OF_AGI_KAGGLE_TOKEN_FILE=.kaggle/access_token` is resolved from the
   repo root.
3. Save a Kaggle API token at that configured path.

The Makefile syncs user-specific Kaggle metadata from `kaggle/.env` before
notebook and upload commands.

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

Upload the cached Modal model snapshot as a private Kaggle Dataset:

```bash
make modal-kaggle-secret
make modal-model-dataset-upload-dry-run
make modal-model-dataset-upload
```

`modal-kaggle-secret` creates the Modal `kaggle-api-token` secret from
`FACE_OF_AGI_KAGGLE_TOKEN_FILE`. The upload uses the cached
`Qwen/Qwen3.6-35B-A3B-FP8` snapshot in the Modal
`face-of-agi-local-models` Volume and mounts on Kaggle at
`/kaggle/input/face-of-agi-qwen36-35b-fp8-weights`. Use
`UPLOAD_MODE=version` for later model dataset updates.

For a different cached Hugging Face snapshot, add a dataset metadata JSON under
`upload/model-dataset/`, then override the Hugging Face repo id, metadata path,
and Kaggle dataset slug:

```bash
FACE_OF_AGI_KAGGLE_MODEL_DATASET_SLUG=face-of-agi-qwen3-vl-4b-thinking-weights \
make modal-model-dataset-upload \
  MODEL_DATASET_METADATA=upload/model-dataset/qwen3-vl-4b-thinking-dataset-metadata.json \
  MODAL_MODEL_UPLOAD_REPO_ID=Qwen/Qwen3-VL-4B-Thinking \
  UPLOAD_MODE=create
```

Prepare and upload the public-game dataset for the debug notebook:

```bash
make public-games-upload UPLOAD_MODE=create
```

This packages the current public ARC normal-mode games. Use
`UPLOAD_MODE=version` for later public-game dataset updates.

## Running

Build and submit the competition notebook:

```bash
make submit
make status
```

`make submit` builds `submission.ipynb` and pushes it for Kaggle Save & Run All.
After the run completes, use Kaggle's "Submit to Competition" action and select
`submission.parquet`.

Build and submit the Gemma 4 competition notebook as a separate Kaggle kernel:

```bash
make gemma-submit
make gemma-status
```

`make gemma-submit` uses
`src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_gemma4_parallel.yaml`,
attaches `face-of-agi-gemma4-31b-it-qat-w4a16-ct-weights`, and pushes to the
`your-kaggle-username/face-of-agi-arc-agi-3-rtx6000-gemma4` kernel id.

Build and submit the debug notebook:

```bash
make debug-submit
make debug-status
```

The debug notebook uses the same wheelhouse and model dataset, attaches the
public-game dataset, starts vLLM, and runs a small public-game batch.
`make debug-submit` regenerates the debug kernel id from the current branch and
8-character HEAD commit id before pushing.

Build and submit the Gemma 4 26B A4B FP8 Dynamic debug notebook:

```bash
make gemma4-fp8-debug-submit
make gemma4-fp8-debug-status
```

This launcher uses
`src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_gemma4_26b_a4b_fp8_dynamic_debug.yaml`,
attaches `face-of-agi-gemma4-26b-a4b-it-fp8-dynamic-weights`, and appends
`gemma4-fp8` to the generated debug kernel title.

Build and submit the MiniCPM-V 4.6 Thinking debug notebook:

```bash
make minicpm-v46-thinking-wheelhouse-upload UPLOAD_MODE=create
make minicpm-v46-thinking-debug-submit
make minicpm-v46-thinking-debug-status
```

This launcher uses
`src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_minicpm_v46_thinking_debug.yaml`,
attaches `face-of-agi-minicpm-v46-thinking-weights` and
`face-of-agi-minicpm-v46-thinking-wheelhouse`, and appends `minicpm-v46-thinking`
to the generated debug kernel title. MiniCPM-V 4.6 needs a newer vLLM than the
Qwen debug path, so keep its vLLM 0.22.0 wheelhouse separate from the shared
debug wheelhouse. Use `UPLOAD_MODE=version` after the first upload.

To submit a debug notebook with a different vLLM runtime config and model
dataset, pass both values to the Make target. Use a title suffix when the
debug run should create a separate Kaggle kernel instead of versioning the
default Qwen debug kernel:

```bash
make debug-submit \
  KAGGLE_DEBUG_CONFIG=src/face_of_agi/runtime/configs/vllm/vllm_rtx6000_gemma4_debug.yaml \
  KAGGLE_DEBUG_MODEL_DATASET_SLUG=face-of-agi-gemma4-31b-it-qat-w4a16-ct-weights \
  DEBUG_KERNEL_TITLE_SUFFIX=gemma4
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
