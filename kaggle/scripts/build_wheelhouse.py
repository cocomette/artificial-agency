"""Build the offline Kaggle wheelhouse for the Transformers runtime."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import read_json_with_kaggle_dataset_id, write_json_if_changed  # noqa: E402

WHEELHOUSE_DATASET_SLUG = "face-of-agi-transformers-wheelhouse"
KAGGLE_TORCH_STACK = {
    "cuda-toolkit",
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
}


def _read_runtime_requirements(path: Path) -> list[str]:
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        parsed = Requirement(requirement)
        if canonicalize_name(parsed.name) in KAGGLE_TORCH_STACK:
            continue
        requirements.append(requirement)
    return requirements


def _pip_download(
    packages: list[str] | tuple[str, ...],
    *,
    output: Path,
    platforms: list[str],
    python_version: str,
    abi: str,
    no_deps: bool = False,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(output),
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        abi,
    ]
    for platform in platforms:
        command.extend(["--platform", platform])
    if no_deps:
        command.append("--no-deps")
    subprocess.check_call(command + list(packages))


def build_wheelhouse(args: argparse.Namespace) -> None:
    output = args.output
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    runtime_requirements = _read_runtime_requirements(args.requirements)
    _pip_download(
        runtime_requirements,
        output=output,
        platforms=args.platform,
        python_version=args.python_version,
        abi=args.abi,
    )
    write_json_if_changed(
        output / "dataset-metadata.json",
        read_json_with_kaggle_dataset_id(args.metadata, WHEELHOUSE_DATASET_SLUG),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="append", required=True)
    parser.add_argument("--python-version", default="312")
    parser.add_argument("--abi", default="cp312")
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_wheelhouse(parse_args())
