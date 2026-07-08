"""Manual E2E check for the Ollama-backed orchestrator agent X.

Start Ollama and pull the model before running:

    ollama serve
    ollama pull gemma4:e4b
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.orchestrator_agent import (
    OllamaOrchestratorAgentAdapter,
    OllamaOrchestratorAgentConfig,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "tests" / "fixtures" / "world" / "ls20_seed0_step0_source.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ollama_orchestrator_agent_e2e"


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    source = Image.open(SOURCE_PATH).convert("RGB")
    observation = Observation(id="ls20-seed0-step0", step=0, frame=source)
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(
            model=args.model,
            host=args.host,
            max_tool_calls=0,
            repair_attempts=args.repair_attempts,
        )
    )

    decision = adapter.decide(
        context=RoleContext(
            general="Choose a valid ARC action from the supplied action space.",
            game="This manual check uses a fixture frame and no world/goal live tools.",
        ),
        history_anchor_observation=observation,
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
    print(f"saved Ollama orchestrator-agent decision to {output_path}")
    print(json.dumps(metrics["final_action"], indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--repair-attempts", type=int, default=1)
    return parser.parse_args()


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
