"""Build a Kaggle notebook that imports the HF model into a private Kaggle model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import (  # noqa: E402
    KAGGLE_TOKEN_FILE_ENV,
    kaggle_token_path,
    with_kaggle_kernel_id,
    with_kaggle_owner_slug,
    write_json_if_changed,
)

NOTEBOOK_PATH = KAGGLE_ROOT / "model-bootstrap" / "model_bootstrap.ipynb"
METADATA_PATH = KAGGLE_ROOT / "model-bootstrap" / "kernel-metadata.json"
MODEL_METADATA_PATH = KAGGLE_ROOT / "upload/model/model-metadata.json"
INSTANCE_METADATA_PATH = KAGGLE_ROOT / "upload/model/model-instance-metadata.json"

HF_MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
UPLOAD_MODE = os.environ.get("MODEL_BOOTSTRAP_UPLOAD_MODE", "create")
CREATE_PARENT_MODEL = os.environ.get("MODEL_BOOTSTRAP_CREATE_PARENT", "true")
VERSION_NOTES = os.environ.get(
    "MODEL_BOOTSTRAP_VERSION_NOTES",
    "FACE-OF-AGI Qwen3.6 35B FP8 Hugging Face import",
)
EMBED_KAGGLE_TOKEN_ENV = "MODEL_BOOTSTRAP_EMBED_KAGGLE_TOKEN"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build(
    *,
    hf_model_id: str = HF_MODEL_ID,
    upload_mode: str = UPLOAD_MODE,
    create_parent_model: str = CREATE_PARENT_MODEL,
    version_notes: str = VERSION_NOTES,
    embedded_kaggle_token: str | None = None,
) -> dict:
    """Return the model bootstrap notebook document."""

    if embedded_kaggle_token is None:
        embedded_kaggle_token = _embedded_kaggle_token_from_env()

    model_metadata = with_kaggle_owner_slug(
        json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    )
    instance_metadata = with_kaggle_owner_slug(
        json.loads(INSTANCE_METADATA_PATH.read_text(encoding="utf-8"))
    )
    model_ref = _model_ref(model_metadata, instance_metadata)
    return {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": "none",
                "isInternetEnabled": True,
                "isGpuEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# FACE-OF-AGI Qwen3.6 35B FP8 Kaggle Model Bootstrap\n\n"
                "Downloads the Hugging Face snapshot and uploads it as a "
                "private Kaggle model variation."
            ),
            _install_cell(),
            _bootstrap_cell(
                hf_model_id=hf_model_id,
                upload_mode=upload_mode,
                create_parent_model=create_parent_model,
                version_notes=version_notes,
                model_ref=model_ref,
                model_metadata=model_metadata,
                instance_metadata=instance_metadata,
                embedded_kaggle_token=embedded_kaggle_token,
            ),
        ],
    }


def _install_cell() -> dict:
    return code_cell("!pip install -q --upgrade huggingface_hub 'kaggle>=2.2.0'")


def _bootstrap_cell(
    *,
    hf_model_id: str,
    upload_mode: str,
    create_parent_model: str,
    version_notes: str,
    model_ref: str,
    model_metadata: dict,
    instance_metadata: dict,
    embedded_kaggle_token: str,
) -> dict:
    return code_cell(
        dedent(
            f"""\
            import json
            import os
            from pathlib import Path
            import shutil
            import subprocess
            import threading
            import time

            from huggingface_hub import HfApi, snapshot_download

            HF_MODEL_ID = {hf_model_id!r}
            MODEL_REF = {model_ref!r}
            UPLOAD_MODE = {upload_mode!r}
            EMBEDDED_KAGGLE_API_TOKEN = {embedded_kaggle_token!r}
            SCRATCH_ROOT = Path(
                os.environ.get("MODEL_BOOTSTRAP_SCRATCH_ROOT", "/kaggle/temp")
            )
            MIN_FREE_BYTES = int(
                os.environ.get(
                    "MODEL_BOOTSTRAP_MIN_FREE_BYTES",
                    str(5 * 1024 * 1024 * 1024),
                )
            )
            CREATE_PARENT_MODEL = {create_parent_model!r}.lower() not in {{
                "0",
                "false",
                "no",
            }}
            VERSION_NOTES = {version_notes!r}
            MODEL_METADATA = {json.dumps(model_metadata, indent=4)!r}
            INSTANCE_METADATA = {json.dumps(instance_metadata, indent=4)!r}

            parent_dir = SCRATCH_ROOT / "kaggle-model-parent"
            variation_dir = SCRATCH_ROOT / "kaggle-model-default"
            for path in (parent_dir, variation_dir):
                if path.exists():
                    shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(
                "HF_HOME",
                str(SCRATCH_ROOT / "huggingface-home"),
            )

            (parent_dir / "model-metadata.json").write_text(
                MODEL_METADATA + "\\n",
                encoding="utf-8",
            )
            (variation_dir / "model-instance-metadata.json").write_text(
                INSTANCE_METADATA + "\\n",
                encoding="utf-8",
            )

            secret_errors = []

            if EMBEDDED_KAGGLE_API_TOKEN:
                os.environ["KAGGLE_API_TOKEN"] = EMBEDDED_KAGGLE_API_TOKEN
                print(
                    "Using embedded Kaggle API token from the generated "
                    "private notebook"
                )

            def optional_secret(*names):
                for name in names:
                    value = os.environ.get(name)
                    if value:
                        print(f"Using credential from environment variable {{name}}")
                        return value
                try:
                    from kaggle_secrets import UserSecretsClient

                    secrets = UserSecretsClient()
                except Exception as exc:
                    secret_errors.append(
                        f"kaggle_secrets unavailable: {{type(exc).__name__}}: {{exc}}"
                    )
                    return None
                for name in names:
                    try:
                        value = secrets.get_secret(name)
                    except Exception as exc:
                        secret_errors.append(
                            f"secret {{name!r}} unavailable: "
                            f"{{type(exc).__name__}}: {{exc}}"
                        )
                        continue
                    if value:
                        print(f"Using credential from Kaggle secret {{name}}")
                        return value
                    secret_errors.append(f"secret {{name!r}} is empty")
                return None

            kaggle_token = optional_secret(
                "KAGGLE_API_TOKEN",
                "KAGGLE_TOKEN",
                "kaggle_api_token",
                "kaggle_token",
            )
            if kaggle_token:
                os.environ["KAGGLE_API_TOKEN"] = kaggle_token
            else:
                kaggle_username = optional_secret("KAGGLE_USERNAME", "kaggle_username")
                kaggle_key = optional_secret("KAGGLE_KEY", "kaggle_key")
                if kaggle_username and kaggle_key:
                    os.environ["KAGGLE_USERNAME"] = kaggle_username
                    os.environ["KAGGLE_KEY"] = kaggle_key
                else:
                    error_details = "\\n".join(f"- {{error}}" for error in secret_errors)
                    raise RuntimeError(
                        "Kaggle credentials are required inside the running "
                        "Kaggle notebook. Your local "
                        f"{KAGGLE_TOKEN_FILE_ENV} path is only used to push "
                        "the notebook and is not copied into Kaggle. In the "
                        "Kaggle notebook UI, open Add-ons > Secrets and add "
                        "KAGGLE_API_TOKEN with the token value, "
                        "or add KAGGLE_USERNAME and KAGGLE_KEY. Make sure the "
                        "secret is attached/enabled for this notebook. Tried "
                        "KAGGLE_API_TOKEN, KAGGLE_TOKEN, KAGGLE_USERNAME, and "
                        f"KAGGLE_KEY. Secret lookup details:\\n{{error_details}}"
                    )

            hf_token = optional_secret("HF_TOKEN", "hf_token")
            download_kwargs = {{
                "repo_id": HF_MODEL_ID,
                "local_dir": str(variation_dir),
            }}
            if hf_token:
                download_kwargs["token"] = hf_token

            model_info = HfApi().model_info(
                HF_MODEL_ID,
                files_metadata=True,
                token=hf_token,
            )
            expected_bytes = sum(
                sibling.size or 0 for sibling in model_info.siblings
            )
            free_bytes = shutil.disk_usage(SCRATCH_ROOT).free
            print(
                f"HF snapshot size: {{expected_bytes / 1024**3:.2f}} GiB; "
                f"scratch free: {{free_bytes / 1024**3:.2f}} GiB"
            )
            if free_bytes < expected_bytes + MIN_FREE_BYTES:
                raise RuntimeError(
                    f"Not enough scratch disk at {{SCRATCH_ROOT}} to download "
                    f"{{HF_MODEL_ID}} before uploading it to Kaggle Models. "
                    f"Need about {{(expected_bytes + MIN_FREE_BYTES) / 1024**3:.2f}} "
                    f"GiB including headroom, found {{free_bytes / 1024**3:.2f}} GiB. "
                    "Set MODEL_BOOTSTRAP_SCRATCH_ROOT to a larger writable path, "
                    "or import the Hugging Face model through the Kaggle Models UI."
                )

            print(f"Downloading {{HF_MODEL_ID}} to {{variation_dir}}")

            def snapshot_size(path):
                total_bytes = 0
                file_count = 0
                for item in path.rglob("*"):
                    if item.is_file():
                        file_count += 1
                        total_bytes += item.stat().st_size
                return file_count, total_bytes

            def report_download_progress(stop_event):
                while not stop_event.wait(60):
                    file_count, total_bytes = snapshot_size(variation_dir)
                    percent = (
                        total_bytes / expected_bytes * 100
                        if expected_bytes
                        else 0
                    )
                    print(
                        "HF download progress: "
                        f"{{file_count}} files, "
                        f"{{total_bytes / 1024**3:.2f}} GiB, "
                        f"{{percent:.1f}}%",
                        flush=True,
                    )

            stop_progress = threading.Event()
            progress_thread = threading.Thread(
                target=report_download_progress,
                args=(stop_progress,),
                daemon=True,
            )
            progress_thread.start()
            try:
                snapshot_download(**download_kwargs)
            finally:
                stop_progress.set()
                progress_thread.join(timeout=5)
            file_count, total_bytes = snapshot_size(variation_dir)
            print(
                "HF download complete: "
                f"{{file_count}} files, {{total_bytes / 1024**3:.2f}} GiB",
                flush=True,
            )

            def checked_kaggle_command(command, success_marker):
                print("Running Kaggle CLI:", " ".join(command), flush=True)
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                output = result.stdout or ""
                if output:
                    print(output, end="" if output.endswith("\\n") else "\\n")
                if result.returncode != 0:
                    raise RuntimeError(
                        "Kaggle CLI command failed with exit code "
                        f"{{result.returncode}}: {{' '.join(command)}}\\n"
                        f"Output:\\n{{output}}"
                    )
                if "creation error:" in output.lower():
                    raise RuntimeError(
                        "Kaggle CLI reported a creation error despite "
                        f"exit status 0: {{output.strip()}}"
                    )
                if success_marker not in output:
                    raise RuntimeError(
                        "Kaggle CLI did not report expected success marker "
                        f"{{success_marker!r}}. Output:\\n{{output}}"
                    )

            if CREATE_PARENT_MODEL:
                checked_kaggle_command(
                    ["kaggle", "models", "create", "-p", str(parent_dir)],
                    "Your model was created.",
                )

            if UPLOAD_MODE == "create":
                checked_kaggle_command(
                    [
                        "kaggle",
                        "models",
                        "variations",
                        "create",
                        "-p",
                        str(variation_dir),
                        "-q",
                    ],
                    "Your model instance was created.",
                )
            elif UPLOAD_MODE == "version":
                checked_kaggle_command(
                    [
                        "kaggle",
                        "models",
                        "variations",
                        "versions",
                        "create",
                        MODEL_REF,
                        "-p",
                        str(variation_dir),
                        "-n",
                        VERSION_NOTES,
                    ],
                    "Your model instance version was created.",
                )
            else:
                raise RuntimeError(
                    "MODEL_BOOTSTRAP_UPLOAD_MODE must be create or version"
                )

            print(f"Uploaded Kaggle model variation: {{MODEL_REF}}")
            """
        )
    )


def _model_ref(model_metadata: dict, instance_metadata: dict) -> str:
    return "/".join(
        (
            str(model_metadata["ownerSlug"]),
            str(model_metadata["slug"]),
            str(instance_metadata["framework"]).lower(),
            str(instance_metadata["instanceSlug"]),
        )
    )


def _embedded_kaggle_token_from_env() -> str:
    if not _truthy_env(EMBED_KAGGLE_TOKEN_ENV):
        return ""

    token_path = kaggle_token_path()
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{EMBED_KAGGLE_TOKEN_ENV}=true but token file does not exist: "
            f"{token_path}"
        ) from exc
    if not token:
        raise RuntimeError(
            f"{EMBED_KAGGLE_TOKEN_ENV}=true but token file is empty: {token_path}"
        )
    return token


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    _sync_user_metadata_files()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_model_bootstrap_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")

    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        meta = with_kaggle_kernel_id(meta)
        meta["enable_gpu"] = False
        meta["enable_internet"] = True
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(
            "[build_model_bootstrap_notebook] Synced "
            f"{METADATA_PATH.relative_to(ROOT)}"
        )


def _sync_user_metadata_files() -> None:
    write_json_if_changed(
        MODEL_METADATA_PATH,
        with_kaggle_owner_slug(
            json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        ),
    )
    write_json_if_changed(
        INSTANCE_METADATA_PATH,
        with_kaggle_owner_slug(
            json.loads(INSTANCE_METADATA_PATH.read_text(encoding="utf-8"))
        ),
    )
    if METADATA_PATH.exists():
        write_json_if_changed(
            METADATA_PATH,
            with_kaggle_kernel_id(
                json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            ),
        )


if __name__ == "__main__":
    main()
