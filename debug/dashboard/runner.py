"""Subprocess runner logic for launching the runtime from the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time

from debug.dashboard.modal_snapshot import volume_relative_path

DEV_COMMAND_PREFIX = (
    "uv",
    "run",
    "--group",
    "dev",
    "python",
    "-m",
    "face_of_agi.runtime.shell",
)
CLEAN_DB_COMMAND_PREFIX = (
    "uv",
    "run",
    "--no-dev",
    "python",
    "-m",
    "face_of_agi.runtime.shell",
)
MODAL_RUN_COMMAND_PREFIX = (
    "uv",
    "run",
    "--with",
    "modal",
    "modal",
    "run",
    "src/face_of_agi/runtime/modal_app.py",
)
GAME_CATALOG_PATH = Path("src/face_of_agi/environment/local_games.json")
RUNTIME_RUNNER_KEY = "runtime_runner"


@dataclass(frozen=True)
class CommandResult:
    """Completed subprocess result displayed by dashboard controls."""

    command: list[str]
    return_code: int
    output: str


def repo_root() -> Path:
    """Return the repository root for subprocess execution."""

    return Path(__file__).resolve().parents[2]


def build_run_command(
    config_path: str | Path,
    database_path: str | Path,
    *,
    keep_all_m_states: bool = True,
) -> list[str]:
    """Build the dev-profile runtime command used by the dashboard runner."""

    command = [
        *DEV_COMMAND_PREFIX,
        "--config",
        str(config_path),
        "--database",
        str(database_path),
    ]
    if keep_all_m_states:
        command.append("--debug-keep-all-m-states")
    return command


def build_modal_run_command(
    config_path: str | Path,
    *,
    database_name: str = "memory.sqlite",
    live_commit_seconds: int = 30,
    timing: bool = False,
) -> list[str]:
    """Build the Modal runtime command used by the dashboard runner."""

    command = [
        *MODAL_RUN_COMMAND_PREFIX,
        "--config",
        str(config_path),
        "--database-name",
        volume_relative_path(database_name),
        "--live-commit-seconds",
        str(live_commit_seconds),
    ]
    if timing:
        command.append("--timing")
    return command


def build_list_games_command() -> list[str]:
    """Build the runtime-shell command that refreshes the local game catalog."""

    return [
        *CLEAN_DB_COMMAND_PREFIX,
        "--list-games",
    ]


def build_clean_db_command(database_path: str | Path) -> list[str]:
    """Build the runtime-shell command that clears SQLite memory rows."""

    return [
        *CLEAN_DB_COMMAND_PREFIX,
        "--database",
        str(database_path),
        "--clean-db",
    ]


def pull_game_list(*, timeout_seconds: float = 30.0) -> CommandResult:
    """Refresh the local ARC game catalog through the runtime shell."""

    command = build_list_games_command()
    return run_command(command, timeout_seconds=timeout_seconds)


def clean_memory_database(
    database_path: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> CommandResult:
    """Run the same runtime clean-db behavior exposed in README commands."""

    command = build_clean_db_command(database_path)
    return run_command(command, timeout_seconds=timeout_seconds)


def run_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    timeout_seconds: float = 30.0,
) -> CommandResult:
    """Run a short dashboard command and capture combined output."""

    working_dir = Path(cwd).resolve() if cwd is not None else repo_root()
    try:
        process = subprocess.run(
            command,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = _timeout_output(exc)
        output += f"\n[dashboard] command timed out after {timeout_seconds:.0f}s\n"
        return CommandResult(command=list(command), return_code=124, output=output)

    return CommandResult(
        command=list(command),
        return_code=int(process.returncode),
        output=process.stdout or "",
    )


def format_command(command: list[str]) -> str:
    """Return a shell-display version of a command list."""

    return shlex.join(command)


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    output = exc.stdout or ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return str(output)


@dataclass
class RuntimeRunner:
    """Manage one runtime subprocess and its buffered terminal output."""

    command: list[str]
    cwd: Path
    process: subprocess.Popen[str]
    output_queue: queue.Queue[str] = field(default_factory=queue.Queue)
    output: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    stopped_at: float | None = None
    _reader: threading.Thread | None = None

    @classmethod
    def start(
        cls,
        command: list[str],
        *,
        cwd: str | Path | None = None,
    ) -> "RuntimeRunner":
        """Start a runtime command and stream combined stdout/stderr."""

        working_dir = Path(cwd).resolve() if cwd is not None else repo_root()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        runner = cls(command=list(command), cwd=working_dir, process=process)
        reader = threading.Thread(target=runner._read_output, daemon=True)
        runner._reader = reader
        reader.start()
        return runner

    def poll(self) -> int | None:
        """Drain output and return the process return code, if finished."""

        self._drain_output()
        return_code = self.process.poll()
        if return_code is not None and self.stopped_at is None:
            self.stopped_at = time.monotonic()
            self._drain_output()
        return return_code

    def is_running(self) -> bool:
        """Return whether the subprocess is still alive."""

        return self.poll() is None

    def stop(self, *, timeout_seconds: float = 5.0) -> int | None:
        """Terminate the subprocess, escalating to kill if it does not exit."""

        if self.poll() is not None:
            return self.process.returncode

        self.output_queue.put("\n[dashboard] stop requested\n")
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self.output_queue.put("[dashboard] process did not stop; killing\n")
            self.process.kill()
            self.process.wait(timeout=timeout_seconds)
        self.stopped_at = time.monotonic()
        self._drain_output()
        return self.process.returncode

    def clear_output(self) -> None:
        """Clear buffered process output."""

        self._drain_output()
        self.output.clear()

    def elapsed_seconds(self) -> float:
        """Return elapsed wall time for the current or completed process."""

        end = self.stopped_at or time.monotonic()
        return max(0.0, end - self.started_at)

    def _read_output(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self.output_queue.put(line)
        finally:
            stream.close()

    def _drain_output(self) -> None:
        while True:
            try:
                self.output.append(self.output_queue.get_nowait())
            except queue.Empty:
                return
