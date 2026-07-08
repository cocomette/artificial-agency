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

This uploads the wheels needed by the offline notebooks to the
`face-of-agi-wheelhouse-new` Kaggle Dataset. Use
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

Build and submit the debug notebook:

```bash
make debug-submit
make debug-status
```

The debug notebook uses the same wheelhouse and model dataset, attaches the
public-game dataset, starts vLLM, and runs a small public-game batch.
`make debug-submit` regenerates the debug kernel id from the current branch and
8-character HEAD commit id before pushing. The tracked
`debug-notebooks/kernel-metadata.template.json` is the source template;
`debug-notebooks/kernel-metadata.json` is generated for the Kaggle CLI and is
not committed.

Download debug SQLite outputs:

```bash
make debug-pull
```

This writes local snapshots under `../runs/kaggle-debug/`. By default it pulls
from the debug kernel id generated for the current branch and HEAD commit. To
pull a debug run launched from another machine, pass the exact notebook URL or
`owner/slug` from Kaggle:

```bash
make debug-pull DEBUG_KERNEL=https://www.kaggle.com/code/<owner>/<kernel-slug>
make debug-pull DEBUG_KERNEL=<owner>/<kernel-slug>
```

Inspect one pulled SQLite file with the dashboard:

```bash
cd ..
uv run --group debug streamlit run debug/dashboard/app.py -- --database runs/kaggle-debug/<sqlite-file>
```
