"""Download a Hugging Face model snapshot into the Modal model volume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:  # pragma: no cover - local syntax checks need not install Modal.
    modal = None  # type: ignore[assignment]

APP_NAME = "face-of-agi-hf-model-download"
DEFAULT_REPO_ID = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
DEFAULT_REVISION = "main"
MODEL_VOLUME_NAME = "face-of-agi-local-models"
MODEL_VOLUME_PATH = Path("/vol/models")
HF_HOME = MODEL_VOLUME_PATH / "huggingface"


def _modal_env() -> dict[str, str]:
    """Return Hugging Face cache paths backed by the Modal model volume."""

    return {
        "HF_HOME": str(HF_HOME),
        "HF_HUB_CACHE": str(HF_HOME / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
        "TRANSFORMERS_CACHE": str(HF_HOME / "transformers"),
        "DIFFUSERS_CACHE": str(HF_HOME / "diffusers"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }


def _snapshot_summary(path: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


if modal is not None:
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("huggingface_hub[hf_transfer]")
        .env(_modal_env())
    )
    app = modal.App(APP_NAME)

    @app.function(
        image=image,
        volumes={str(MODEL_VOLUME_PATH): model_volume},
        timeout=60 * 60 * 8,
    )
    def download_snapshot(
        *,
        repo_id: str,
        revision: str,
        repo_type: str,
        force_download: bool,
    ) -> dict[str, Any]:
        """Download one Hugging Face snapshot and commit it to the Modal Volume."""

        from huggingface_hub import snapshot_download

        HF_HOME.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                repo_type=repo_type,
                force_download=force_download,
            )
        )
        model_volume.commit()
        return {
            "repo_id": repo_id,
            "revision": revision,
            "repo_type": repo_type,
            "snapshot_path": str(snapshot_path),
            "hf_home": str(HF_HOME),
            **_snapshot_summary(snapshot_path),
        }

    @app.local_entrypoint()
    def main(
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = DEFAULT_REVISION,
        repo_type: str = "model",
        hf_secret: str = "",
        force_download: bool = False,
    ) -> None:
        """Download a Hugging Face snapshot to the FACE-OF-AGI Modal model Volume."""

        remote_function = download_snapshot
        if hf_secret:
            remote_function = download_snapshot.with_options(
                secrets=[modal.Secret.from_name(hf_secret)]
            )

        result = remote_function.remote(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            force_download=force_download,
        )
        print(f"Downloaded {result['repo_id']}@{result['revision']}")
        print(f"Snapshot: {result['snapshot_path']}")
        print(f"HF home: {result['hf_home']}")
        print(
            "Snapshot files: "
            f"{result['file_count']} files, {result['total_bytes']} bytes"
        )

else:
    app = None

