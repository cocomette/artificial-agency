"""Build the offline Kaggle wheelhouse without replacing Kaggle's torch stack."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_env import read_json_with_kaggle_dataset_id, write_json_if_changed  # noqa: E402

VLLM_VERSION = os.environ.get("FACE_OF_AGI_KAGGLE_VLLM_VERSION", "0.19.1")
FULL_VLLM_DEPS = os.environ.get(
    "FACE_OF_AGI_KAGGLE_FULL_VLLM_DEPS",
    "",
).lower() in {"1", "true", "yes", "on"}
WHEELHOUSE_DATASET_SLUG = "face-of-agi-wheelhouse"
KAGGLE_TORCH_STACK = {
    "cuda-toolkit",
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
}
NO_DEPS_STACK_PACKAGES = (f"vllm=={VLLM_VERSION}",) if FULL_VLLM_DEPS else (
    "compressed-tensors==0.15.0.1",
    "flashinfer-python==0.6.6",
    "quack-kernels==0.4.1",
    "torch-c-dlpack-ext",
    f"vllm=={VLLM_VERSION}",
    "xgrammar==0.2.1",
)
NO_DEPS_STACK_NAMES = {"vllm"} if FULL_VLLM_DEPS else {
    "compressed-tensors",
    "flashinfer-python",
    "quack-kernels",
    "tilelang",
    "tokenspeed-mla",
    "torch-c-dlpack-ext",
    "vllm",
    "xgrammar",
}
VLLM_NO_DEPS_DEPENDENCY_NAMES = (
    {
        # Torch 2.11.0 declares a Linux cuda-toolkit dependency that is not
        # available in Kaggle's offline wheel inputs. Keep the pinned wheels
        # in the MiniCPM wheelhouse but install them with --no-deps.
        "torch",
        "torchaudio",
        "torchvision",
    }
    if FULL_VLLM_DEPS
    else {
        # This package declares broad Torch/CUDA dependencies; keep Kaggle's
        # preinstalled stack authoritative and include only the matching wheel.
        "humming-kernels",
    }
)
VLLM_TORCH_DEPENDENCY_PACKAGES = () if FULL_VLLM_DEPS else (
    "apache-tvm-ffi>=0.1.2",
    "click",
    "cloudpickle",
    "cuda-tile",
    "einops",
    "loguru",
    "ml-dtypes",
    "ninja",
    "numpy>=1.23.5",
    "nvidia-cudnn-frontend>=1.13.0",
    "nvidia-cutlass-dsl>=4.4.2",
    "nvidia-ml-py",
    "packaging>=24.2",
    "psutil",
    "pydantic>=2.0",
    "requests",
    "setuptools",
    "tabulate",
    "tqdm>=4.62.3",
    "transformers>=4.45.0",
    "typing-extensions>=4.10.0",
    "z3-solver<4.15.5,>=4.13.0",
)
FULL_VLLM_EXTRA_PACKAGES = (
    # Torch 2.11's CUDA 13 wheel imports libcupti.so.13 at startup. The
    # nvidia-cuda-cupti wheel is tagged manylinux_2_25, so it is downloaded
    # separately below instead of relying on the default wheelhouse platform.
    "nvidia-cuda-cupti==13.0.85",
) if FULL_VLLM_DEPS else ()
LINUX_MARKER_ENV = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.0",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
    "extra": "",
}


def _read_runtime_requirements(path: Path) -> list[str]:
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        parsed = Requirement(requirement)
        if canonicalize_name(parsed.name) == "vllm":
            continue
        requirements.append(requirement)
    return requirements


def _pip_download(
    packages: list[str] | tuple[str, ...],
    *,
    output: Path,
    platforms: list[str],
    python_version: str,
    abi: list[str],
    no_deps: bool = False,
) -> None:
    if not packages:
        return
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
    ]
    for platform in platforms:
        command.extend(["--platform", platform])
    for abi_tag in abi:
        command.extend(["--abi", abi_tag])
    if no_deps:
        command.append("--no-deps")
    subprocess.check_call(command + list(packages))


def _wheel_for(output: Path, distribution: str, version: str | None = None) -> Path:
    normalized = canonicalize_name(distribution).replace("-", "_")
    pattern = f"{normalized}-*.whl" if version is None else f"{normalized}-{version}*.whl"
    matches = sorted(output.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {distribution} wheel in {output}")
    return matches[-1]


def _vllm_dependency_requirements(output: Path) -> tuple[list[str], list[str]]:
    skipped = (set() if FULL_VLLM_DEPS else KAGGLE_TORCH_STACK) | NO_DEPS_STACK_NAMES
    with zipfile.ZipFile(_wheel_for(output, "vllm", VLLM_VERSION)) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = wheel.read(metadata_name).decode("utf-8")

    deps_requirements = []
    no_deps_requirements = []
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist: "):
            continue
        requirement = Requirement(line.removeprefix("Requires-Dist: "))
        normalized_name = canonicalize_name(requirement.name)
        if normalized_name in skipped:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate(
            LINUX_MARKER_ENV
        ):
            continue
        if normalized_name in VLLM_NO_DEPS_DEPENDENCY_NAMES:
            no_deps_requirements.append(str(requirement))
        else:
            deps_requirements.append(str(requirement))
    return deps_requirements, no_deps_requirements


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
    _pip_download(
        NO_DEPS_STACK_PACKAGES,
        output=output,
        platforms=args.platform,
        python_version=args.python_version,
        abi=args.abi,
        no_deps=True,
    )
    deps_requirements, no_deps_requirements = _vllm_dependency_requirements(output)
    _pip_download(
        list(VLLM_TORCH_DEPENDENCY_PACKAGES) + deps_requirements,
        output=output,
        platforms=args.platform,
        python_version=args.python_version,
        abi=args.abi,
    )
    _pip_download(
        no_deps_requirements,
        output=output,
        platforms=args.platform,
        python_version=args.python_version,
        abi=args.abi,
        no_deps=True,
    )
    _pip_download(
        FULL_VLLM_EXTRA_PACKAGES,
        output=output,
        platforms=["manylinux_2_25_x86_64"],
        python_version=args.python_version,
        abi=args.abi,
        no_deps=True,
    )
    write_json_if_changed(
        output / "dataset-metadata.json",
        read_json_with_kaggle_dataset_id(args.metadata, args.dataset_slug),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="append", required=True)
    parser.add_argument("--python-version", default="312")
    parser.add_argument("--abi", action="append", default=[])
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-slug", default=WHEELHOUSE_DATASET_SLUG)
    args = parser.parse_args()
    if not args.abi:
        args.abi = ["cp312"]
    return args


if __name__ == "__main__":
    build_wheelhouse(parse_args())
