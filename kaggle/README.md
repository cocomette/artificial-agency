# Kaggle ARC-AGI-3 Submission And Debug

This directory builds the offline Kaggle submission notebook and the
public-game debug notebook for the Transformers online learner runtime.

## Inputs

The submission expects these Kaggle inputs:

- `face-of-agi-qwen36-35b-fp8-weights`: Hugging Face Qwen3.6 35B FP8 model
  and processor files at the dataset root.
- `face-of-agi-transformers-wheelhouse`: offline wheels built from
  `kaggle/requirements-kaggle.txt`.
- `face-of-agi-public-games`: local public-game files for the debug notebook
  only.

The generated competition notebook invokes `face_of_agi.runtime.kaggle`
directly. It does not start a model server.

## Setup

1. Accept the ARC Prize 2026 ARC-AGI-3 Kaggle rules.
2. Set your Kaggle username and token path in `kaggle/.env`.
3. Save a Kaggle API token at that configured path.

Build and upload the offline Python wheelhouse:

```bash
cd kaggle
make wheelhouse-upload UPLOAD_MODE=create
```

Upload bundled model weights as a Kaggle dataset so
`/kaggle/input/face-of-agi-qwen36-35b-fp8-weights` contains the local
Transformers files.

Prepare and upload the public-game dataset for the debug notebook:

```bash
cd kaggle
make public-games-upload UPLOAD_MODE=create
```

## Submission

Build and submit the competition notebook:

```bash
cd kaggle
make submit
make status
```

`make submit` builds `notebooks/submission.ipynb` and pushes it for Kaggle
Save & Run All. After the run completes, use Kaggle's submit action and select
`submission.parquet`.

## Debug

Build and submit the public-games debug notebook:

```bash
cd kaggle
make debug-submit
make debug-status
```

Download debug SQLite outputs:

```bash
cd kaggle
make debug-pull
```

Inspect one pulled SQLite file with the dashboard:

```bash
uv run --group dev streamlit run debug/dashboard/app.py -- \
  --database runs/kaggle-debug/<sqlite-file>
```
