"""Manual E2E check for the OpenAI-backed orchestrator agent X.

This script calls the real OpenAI API. Set OPENAI_API_KEY before running it.
It uses a committed ARC frame fixture and asks X to submit one valid action
without world/goal tools.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from PIL import Image

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.orchestrator_agent import (
    OpenAIOrchestratorAgentAdapter,
    OpenAIOrchestratorAgentConfig,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "tests" / "fixtures" / "world" / "ls20_seed0_step0_source.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "openai_orchestrator_agent_e2e"


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    output_dir = _resolve_output_dir(args.output_dir)
    source = Image.open(SOURCE_PATH).convert("RGB")
    observation = Observation(id="ls20-seed0-step0", step=0, frame=source)
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(
            model=args.model,
            api_key_env=args.api_key_env,
            max_tool_calls=0,
            repair_attempts=args.repair_attempts,
            reasoning={"effort": args.reasoning_effort},
            metadata={"role": "agent", "script": "openai_orchestrator_agent_e2e"},
        )
    )

    decision = adapter.decide(
        context=RoleContext(
            general="Choose a valid ARC action from the supplied action space.",
            game="This manual check uses a fixture frame and no world/goal tools.",
        ),
        first_observation=observation,
        current_observation=observation,
        action_space=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION2"),
        ),
        tool_runtime=None,
    )
    metrics = {
        "fixture": str(SOURCE_PATH.relative_to(ROOT)),
        "final_action": asdict(decision.final_action),
        "reasoning_summary": decision.trace.reasoning_summary,
        "trace_metadata": decision.trace.metadata,
    }
    output_path = output_dir / "decision.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"saved OpenAI orchestrator-agent decision to {output_path}")
    print(json.dumps(metrics["final_action"], indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Dotenv file to load before calling OpenAI. Use an empty value to disable.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--repair-attempts", type=int, default=1)
    return parser.parse_args()


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_env_file(env_file: str) -> None:
    """Load a dotenv file when present without printing secret values."""

    if not env_file:
        return
    path = Path(env_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
