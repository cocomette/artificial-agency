"""Upload a Hugging Face snapshot from a Modal Volume to Kaggle artifacts."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import with_kaggle_dataset_id, with_kaggle_owner_slug  # noqa: E402

try:
    import modal
except ImportError:  # pragma: no cover - local tests need not install Modal.
    modal = None  # type: ignore[assignment]

APP_NAME = "face-of-agi-kaggle-model-upload"
DEFAULT_REPO_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_MODEL_INSTANCE_METADATA_PATH = Path(
    "upload/model/model-instance-metadata.json"
)
DEFAULT_MODEL_DATASET_METADATA_PATH = Path(
    "upload/model-dataset/dataset-metadata.json"
)
DEFAULT_METADATA_PATH = DEFAULT_MODEL_INSTANCE_METADATA_PATH
DEFAULT_VERSION_NOTES = "FACE-OF-AGI Modal volume upload"
MODEL_DATASET_SLUG = "face-of-agi-qwen36-35b-fp8-weights"
MODEL_VOLUME_NAME = os.environ.get(
    "FACE_OF_AGI_MODAL_MODEL_VOLUME",
    "face-of-agi-local-models",
)
KAGGLE_SECRET_NAME = os.environ.get(
    "FACE_OF_AGI_MODAL_KAGGLE_SECRET",
    "kaggle-api-token",
)
MODEL_VOLUME_PATH = Path("/vol/models")
HF_HOME = MODEL_VOLUME_PATH / "huggingface"
KAGGLE_MODEL_UPLOAD_TMP = MODEL_VOLUME_PATH / "kaggle-upload-tmp"
KAGGLE_DATASET_UPLOAD_TMP = MODEL_VOLUME_PATH / "kaggle-dataset-upload-tmp"
UPLOAD_STATE_COMMIT_SECONDS = 60


def _hf_cache_repo_dir(hf_home: Path, repo_id: str) -> Path:
    return hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"


def _find_hf_snapshot(
    *,
    hf_home: Path = HF_HOME,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> Path:
    repo_dir = _hf_cache_repo_dir(hf_home, repo_id)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        raise RuntimeError(
            f"HF cache snapshots directory does not exist: {snapshots_dir}"
        )

    if revision:
        direct_snapshot = snapshots_dir / revision
        if direct_snapshot.exists():
            return direct_snapshot
        ref_path = repo_dir / "refs" / revision
        if ref_path.exists():
            commit = ref_path.read_text(encoding="utf-8").strip()
            referenced_snapshot = snapshots_dir / commit
            if referenced_snapshot.exists():
                return referenced_snapshot
            raise RuntimeError(
                f"HF ref {revision!r} points to missing snapshot: "
                f"{referenced_snapshot}"
            )

    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise RuntimeError(f"HF cache contains no snapshots under {snapshots_dir}")
    with_config = [path for path in snapshots if (path / "config.json").exists()]
    return with_config[0] if with_config else snapshots[0]


def _model_instance_ref(metadata: dict[str, Any]) -> str:
    return "/".join(
        (
            str(metadata["ownerSlug"]),
            str(metadata["modelSlug"]),
            str(metadata["framework"]).lower(),
            str(metadata["instanceSlug"]),
        )
    )


def _parent_model_ref(metadata: dict[str, Any]) -> str:
    return "/".join((str(metadata["ownerSlug"]), str(metadata["modelSlug"])))


def _dataset_ref(metadata: dict[str, Any]) -> str:
    return str(metadata["id"])


def _snapshot_summary(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return file_count, total_bytes


def _run_streamed(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_tail: deque[str] = deque(maxlen=120)
    assert process.stdout is not None
    for line in process.stdout:
        output_tail.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    output = "".join(output_tail)
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}: {' '.join(command)}\n"
            f"Last output lines:\n{output}"
        )
    return output


def _configure_kaggle_upload_tmp(upload_tmp: Path) -> dict[str, str]:
    upload_tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(upload_tmp)
    return {
        **os.environ,
        "TMPDIR": str(upload_tmp),
        "TEMP": str(upload_tmp),
        "TMP": str(upload_tmp),
    }


def _completed_upload_token(path: str, upload_context: Any) -> str:
    upload_info_path = Path(upload_context.get_upload_info_file_path(path))
    if not upload_info_path.exists():
        raise RuntimeError(
            "Kaggle upload state is missing for "
            f"{path}. Expected {upload_info_path}."
        )

    upload_info = json.loads(upload_info_path.read_text(encoding="utf-8"))
    if upload_info.get("upload_complete") is not True:
        raise RuntimeError(f"Kaggle upload is not complete for {path}.")
    response = upload_info.get("start_blob_upload_response")
    if not isinstance(response, dict):
        raise RuntimeError(f"Kaggle upload token response is missing for {path}.")
    token = response.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"Kaggle upload token is missing for {path}.")
    return token


if modal is not None:
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "kaggle>=2.2.0"
    )
    model_volume = modal.Volume.from_name(
        MODEL_VOLUME_NAME,
        create_if_missing=False,
    )
    kaggle_secret = modal.Secret.from_name(
        KAGGLE_SECRET_NAME,
        required_keys=["KAGGLE_API_TOKEN"],
    )
    app = modal.App(APP_NAME)

    def _start_volume_committer(stop_event: threading.Event) -> threading.Thread:
        thread = threading.Thread(
            target=_commit_model_volume_until_stopped,
            args=(stop_event,),
            daemon=True,
        )
        thread.start()
        return thread

    def _commit_model_volume_until_stopped(stop_event: threading.Event) -> None:
        while not stop_event.wait(UPLOAD_STATE_COMMIT_SECONDS):
            try:
                print("Committing Modal model volume upload state", flush=True)
                model_volume.commit()
            except Exception as exc:
                print(f"Modal model volume commit failed: {exc}", flush=True)

    @app.function(
        image=image,
        volumes={str(MODEL_VOLUME_PATH): model_volume},
        secrets=[kaggle_secret],
        timeout=60 * 60 * 8,
    )
    def upload_model_variation(
        *,
        metadata_text: str,
        artifact_kind: str = "model",
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = "main",
        snapshot_path: str = "",
        upload_mode: str = "create",
        version_notes: str = DEFAULT_VERSION_NOTES,
        dir_mode: str = "skip",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        metadata = json.loads(metadata_text)
        if artifact_kind == "model":
            artifact_ref = _model_instance_ref(metadata)
            parent_ref = _parent_model_ref(metadata)
            metadata_filename = "model-instance-metadata.json"
            upload_tmp = KAGGLE_MODEL_UPLOAD_TMP
        elif artifact_kind == "dataset":
            artifact_ref = _dataset_ref(metadata)
            parent_ref = artifact_ref
            metadata_filename = "dataset-metadata.json"
            upload_tmp = KAGGLE_DATASET_UPLOAD_TMP
        else:
            raise RuntimeError("artifact_kind must be model or dataset")

        source_dir = (
            Path(snapshot_path)
            if snapshot_path
            else _find_hf_snapshot(repo_id=repo_id, revision=revision)
        )
        if not source_dir.exists():
            raise RuntimeError(f"model source directory does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise RuntimeError(f"model source path is not a directory: {source_dir}")

        metadata_path = source_dir / metadata_filename
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        file_count, total_bytes = _snapshot_summary(source_dir)
        print(f"Using Modal model source: {source_dir}", flush=True)
        print(f"Kaggle artifact kind: {artifact_kind}", flush=True)
        print(f"Kaggle artifact: {artifact_ref}", flush=True)
        print(
            f"Source summary: {file_count} files, {total_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
        if dry_run:
            return {
                "dry_run": True,
                "source_dir": str(source_dir),
                "file_count": file_count,
                "total_bytes": total_bytes,
                "artifact_kind": artifact_kind,
                "artifact_ref": artifact_ref,
            }

        subprocess_env = _configure_kaggle_upload_tmp(upload_tmp)
        print(
            f"Kaggle resumable upload state: {upload_tmp / '.kaggle/uploads'}",
            flush=True,
        )

        if upload_mode == "finalize":
            if artifact_kind != "model":
                raise RuntimeError("upload_mode finalize is only supported for model")
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            def reuse_completed_upload(
                path: str,
                quiet: bool,
                blob_type: Any,
                upload_context: Any,
                content_type: str | None = None,
            ) -> str:
                del quiet, blob_type, content_type
                return _completed_upload_token(path, upload_context)

            api._upload_blob = reuse_completed_upload  # type: ignore[method-assign]
            api.model_instance_create(str(source_dir), quiet=False, dir_mode=dir_mode)
            print("Verifying Kaggle variation list", flush=True)
            _run_streamed(
                ["kaggle", "models", "variations", "list", parent_ref],
                env=subprocess_env,
            )
            return {
                "dry_run": False,
                "source_dir": str(source_dir),
                "file_count": file_count,
                "total_bytes": total_bytes,
                "artifact_kind": artifact_kind,
                "artifact_ref": artifact_ref,
                "finalize_only": True,
            }

        if artifact_kind == "dataset" and upload_mode == "create":
            command = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(source_dir),
                "--dir-mode",
                dir_mode,
            ]
        elif artifact_kind == "dataset" and upload_mode == "version":
            command = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(source_dir),
                "-m",
                version_notes,
                "--dir-mode",
                dir_mode,
                "--delete-old-versions",
            ]
        elif artifact_kind == "model" and upload_mode == "create":
            command = [
                "kaggle",
                "models",
                "variations",
                "create",
                "-p",
                str(source_dir),
                "--dir-mode",
                dir_mode,
            ]
        elif artifact_kind == "model" and upload_mode == "version":
            command = [
                "kaggle",
                "models",
                "variations",
                "versions",
                "create",
                artifact_ref,
                "-p",
                str(source_dir),
                "-n",
                version_notes,
                "--dir-mode",
                dir_mode,
            ]
        else:
            raise RuntimeError("upload_mode must be create, version, or finalize")

        commit_stop = threading.Event()
        commit_thread = _start_volume_committer(commit_stop)
        try:
            _run_streamed(command, env=subprocess_env)
        finally:
            commit_stop.set()
            commit_thread.join(timeout=5)
            print("Final Modal model volume upload-state commit", flush=True)
            model_volume.commit()
        if artifact_kind == "model":
            print("Verifying Kaggle variation list", flush=True)
            _run_streamed(
                ["kaggle", "models", "variations", "list", parent_ref],
                env=subprocess_env,
            )
        else:
            print("Verifying Kaggle dataset list", flush=True)
            _run_streamed(
                ["kaggle", "datasets", "list", "-s", parent_ref.rsplit("/", 1)[1]],
                env=subprocess_env,
            )
        return {
            "dry_run": False,
            "source_dir": str(source_dir),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "artifact_kind": artifact_kind,
            "artifact_ref": artifact_ref,
        }

    @app.local_entrypoint()
    def main(
        metadata_path: str = str(DEFAULT_METADATA_PATH),
        artifact_kind: str = "model",
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = "main",
        snapshot_path: str = "",
        upload_mode: str = "create",
        version_notes: str = DEFAULT_VERSION_NOTES,
        dir_mode: str = "skip",
        dry_run: bool = False,
    ) -> None:
        metadata_text = _metadata_text(
            Path(metadata_path),
            artifact_kind=artifact_kind,
        )
        result = upload_model_variation.remote(
            metadata_text=metadata_text,
            artifact_kind=artifact_kind,
            repo_id=repo_id,
            revision=revision,
            snapshot_path=snapshot_path,
            upload_mode=upload_mode,
            version_notes=version_notes,
            dir_mode=dir_mode,
            dry_run=dry_run,
        )
        print(json.dumps(result, indent=2))

else:
    app = None


def _metadata_text(path: Path, *, artifact_kind: str) -> str:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if artifact_kind == "model":
        metadata = with_kaggle_owner_slug(metadata)
    elif artifact_kind == "dataset":
        metadata = with_kaggle_dataset_id(metadata, MODEL_DATASET_SLUG)
    else:
        raise RuntimeError("artifact_kind must be model or dataset")
    return json.dumps(metadata, indent=2) + "\n"
