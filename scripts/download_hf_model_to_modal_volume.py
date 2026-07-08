"""Modal app for downloading a Hugging Face model into the model Volume.

Run from the repo root:

    uv run --with modal modal run scripts/download_hf_model_to_modal_volume.py
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

import modal

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
VOLUME_NAME = "face-of-agi-local-models"
MOUNT = PurePosixPath("/vol/models")
TIMEOUT_SECONDS = 60 * 60 * 6

HF_HOME = MOUNT / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"

app = modal.App("download-qwen36-35b-a3b")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.32.0", "hf-xet")
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HUB_CACHE),
            "HUGGINGFACE_HUB_CACHE": str(HF_HUB_CACHE),
            "TRANSFORMERS_CACHE": str(HF_HOME / "transformers"),
        }
    )
)
secrets = []
if hf_token := os.environ.get("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token}))


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    secrets=secrets,
    timeout=TIMEOUT_SECONDS,
)
def download(model_id: str = MODEL_ID, revision: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(HF_HUB_CACHE),
    )
    print(f"downloaded {model_id} to {path}")
    volume.commit()
    return path


@app.local_entrypoint()
def main(model_id: str = MODEL_ID, revision: str | None = None) -> None:
    remote_path = download.remote(model_id=model_id, revision=revision)
    print(f"downloaded to {remote_path}")
