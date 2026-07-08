"""Manual E2E check for Ollama-backed prompt updaters.

Start Ollama and pull the model before running:

    ollama serve
    ollama pull gemma4:e4b

The script exercises all six updater targets:
world_game, goal_game, agent_game, world_general, goal_general, and
agent_general. It saves request/response artifacts so malformed local-model
JSON can be inspected directly.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from arcengine import GameAction

from core import resolve_output_dir
from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    TurnMetrics,
    RoleContext,
    ToolResult,
)
from face_of_agi.frames import to_memory_jsonable
from face_of_agi.models.providers.ollama import message_content, object_get
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentProgressFeedback,
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    WorldGameContextUpdateInput,
)
from face_of_agi.models.updater.config import OllamaUpdaterConfig
from face_of_agi.models.updater.providers.ollama import OllamaUpdaterAdapter

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
TARGET_PATH = FIXTURE_DIR / "ls20_seed0_action1_target.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "ollama_updater_e2e"
GAME_ID = "ls20-9607627b"
RUN_ID = "ollama-updater-e2e"

UpdaterTarget = Literal[
    "world_game",
    "goal_game",
    "agent_game",
    "world_general",
    "goal_general",
    "agent_general",
]

TARGET_ORDER: tuple[UpdaterTarget, ...] = (
    "world_game",
    "goal_game",
    "agent_game",
    "world_general",
    "goal_general",
    "agent_general",
)
TARGET_CHOICES = ("all", *TARGET_ORDER)


def main() -> None:
    args = _parse_args()
    output_dir = resolve_output_dir(args.output_dir, root=ROOT)
    selected_targets = _selected_targets(args.targets)

    source = Image.open(args.source_image).convert("RGB")
    target = Image.open(args.target_image).convert("RGB")
    adapter = OllamaUpdaterAdapter(_config_from_args(args))
    contexts = _initial_contexts()

    artifacts: list[dict[str, Any]] = []
    target_calls: dict[UpdaterTarget, Callable[[], RoleContext]] = {
        "world_game": lambda: _run_world_game_update(
            adapter=adapter,
            previous_context=contexts["world"],
            source=source,
            target=target,
        ),
        "goal_game": lambda: _run_goal_game_update(
            adapter=adapter,
            previous_context=contexts["goal"],
            source=source,
            target=target,
        ),
        "agent_game": lambda: _run_agent_game_update(
            adapter=adapter,
            previous_context=contexts["agent"],
            source=source,
            target=target,
        ),
        "world_general": lambda: _run_general_update(
            adapter=adapter,
            role="world",
            previous_context=contexts["world"],
        ),
        "goal_general": lambda: _run_general_update(
            adapter=adapter,
            role="goal",
            previous_context=contexts["goal"],
        ),
        "agent_general": lambda: _run_general_update(
            adapter=adapter,
            role="agent",
            previous_context=contexts["agent"],
        ),
    }

    for target_name in selected_targets:
        role = _target_role(target_name)
        previous_context = contexts[role]
        try:
            result_context = target_calls[target_name]()
            _validate_update(
                target_name=target_name,
                previous_context=previous_context,
                result_context=result_context,
            )
            contexts[role] = result_context
            status = "ok"
            error = None
        except Exception as exc:
            result_context = previous_context
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        artifacts.append(
            _write_target_artifacts(
                output_dir=output_dir,
                target_name=target_name,
                status=status,
                error=error,
                previous_context=previous_context,
                result_context=result_context,
                adapter=adapter,
            )
        )
        if status != "ok" and not args.continue_on_error:
            break

    summary = {
        "fixture": {
            "game_id": GAME_ID,
            "source_frame": str(Path(args.source_image).resolve()),
            "actual_next_frame": str(Path(args.target_image).resolve()),
        },
        "model": _model_summary(args),
        "selected_targets": selected_targets,
        "artifacts": artifacts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"saved Ollama updater E2E artifacts to {output_dir}")
    print(json.dumps({"selected_targets": selected_targets, "artifacts": artifacts}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--source-image", default=str(SOURCE_PATH))
    parser.add_argument("--target-image", default=str(TARGET_PATH))
    parser.add_argument("--input-image-size", default="256x256")
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_CHOICES,
        default=("all",),
        help="Targets to run. Use 'all' for all six updater variants.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later targets after one updater target fails.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> OllamaUpdaterConfig:
    return OllamaUpdaterConfig(
        backend="ollama",
        model=args.model,
        host=args.host,
        think=args.think,
        keep_alive=args.keep_alive,
        options={"temperature": args.temperature},
        input_image_size=args.input_image_size,
        input_image_resample=args.input_image_resample,
    )


def _initial_contexts() -> dict[str, RoleContext]:
    return {
        "world": RoleContext(
            general=(
                "Infer reusable ARC transition rules from visual evidence. "
                "Prefer mechanics that transfer across similar grid worlds."
            ),
            game=(
                "Game ls20-9607627b uses a small grid-like scene. Current "
                "hypothesis: ACTION1 changes the active object, but the exact "
                "transition rule is uncertain."
            ),
        ),
        "goal": RoleContext(
            general=(
                "Infer reusable ARC goal patterns from visual evidence. "
                "Prefer concise hypotheses about progress and completion."
            ),
            game=(
                "Game ls20-9607627b goal hypothesis: compare current and next "
                "frames to identify what progress means."
            ),
        ),
        "agent": RoleContext(
            general=(
                "Act systematically in unknown ARC games. Use tools when the "
                "effect of an action or the goal is uncertain."
            ),
            game=(
                "For ls20-9607627b, inspect whether ACTION1 changes the grid "
                "toward a target arrangement before repeating it."
            ),
        ),
    }


def _run_world_game_update(
    *,
    adapter: OllamaUpdaterAdapter,
    previous_context: RoleContext,
    source: Image.Image,
    target: Image.Image,
) -> RoleContext:
    observation_ref = ObservationRef(memory="state", id="ls20-seed0-step0")
    actual_observation_ref = ObservationRef(memory="state", id="ls20-seed0-step1")
    action = ActionSpec(action_id=GameAction.ACTION1)
    return adapter.update_world_game_context(
        WorldGameContextUpdateInput(
            previous_context=previous_context,
            current_observation=Observation(
                id=actual_observation_ref.id,
                step=1,
                frame=target,
            ),
            post_decision_predictions=PostDecisionPredictions(
                world_prediction=ToolResult(
                    id="world-e2e-prediction",
                    tool="world",
                    predicted_description=_fixture_description("source frame prediction"),
                    source_observation_ref=observation_ref,
                    action=action,
                )
            ),
            turn_metrics=TurnMetrics(time_cost=1.0),
            submitted_action=action,
            metadata={"fixture": GAME_ID, "target": "world_game"},
        )
    )


def _run_goal_game_update(
    *,
    adapter: OllamaUpdaterAdapter,
    previous_context: RoleContext,
    source: Image.Image,
    target: Image.Image,
) -> RoleContext:
    observation_ref = ObservationRef(memory="state", id="ls20-seed0-step0")
    actual_observation_ref = ObservationRef(memory="state", id="ls20-seed0-step1")
    return adapter.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=previous_context,
            current_observation=Observation(
                id=actual_observation_ref.id,
                step=1,
                frame=target,
            ),
            post_decision_predictions=PostDecisionPredictions(
                goal_prediction=ToolResult(
                    id="goal-e2e-prediction",
                    tool="goal",
                    predicted_description=_fixture_description("goal frame prediction"),
                    source_observation_ref=observation_ref,
                )
            ),
            turn_metrics=TurnMetrics(time_cost=1.0),
            submitted_action=ActionSpec(action_id=GameAction.ACTION1),
            metadata={"fixture": GAME_ID, "target": "goal_game"},
        )
    )


def _run_agent_game_update(
    *,
    adapter: OllamaUpdaterAdapter,
    previous_context: RoleContext,
    source: Image.Image,
    target: Image.Image,
) -> RoleContext:
    previous_ref = ObservationRef(memory="state", id="ls20-seed0-step0")
    current_ref = ObservationRef(memory="state", id="ls20-seed0-step1")
    action = ActionSpec(action_id=GameAction.ACTION1)
    return adapter.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=previous_context,
            previous_observation=Observation(id=previous_ref.id, step=0, frame=source),
            current_observation=Observation(id=current_ref.id, step=1, frame=target),
            action_history=(
                ActionHistoryEntry(
                    action=action,
                    controllable=True,
                ),
            ),
            current_turn_world_game_context="world context for this turn",
            previous_turn_world_game_context=None,
            turn_metrics=AgentProgressFeedback(
                time_cost=1.0,
                cumulative_score=0.0,
            ),
        )
    )


def _run_general_update(
    *,
    adapter: OllamaUpdaterAdapter,
    role: Literal["world", "goal", "agent"],
    previous_context: RoleContext,
) -> RoleContext:
    return adapter.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role=role,
            previous_context=previous_context,
            run_id=RUN_ID,
            game_id=GAME_ID,
            stop_reason="manual_e2e_complete",
            step_count=1,
            completed_levels=0,
            final_state="fixture_transition",
            metadata={"fixture": GAME_ID, "target": f"{role}_general"},
        )
    )


def _validate_update(
    *,
    target_name: UpdaterTarget,
    previous_context: RoleContext,
    result_context: RoleContext,
) -> None:
    if target_name.endswith("_game"):
        if not result_context.game.strip():
            raise RuntimeError(f"{target_name} returned an empty game context")
        if result_context.general != previous_context.general:
            raise RuntimeError(f"{target_name} changed the general context segment")
        return

    if not result_context.general.strip():
        raise RuntimeError(f"{target_name} returned an empty general context")
    if result_context.game != previous_context.game:
        raise RuntimeError(f"{target_name} changed the game context segment")


def _write_target_artifacts(
    *,
    output_dir: Path,
    target_name: UpdaterTarget,
    status: str,
    error: str | None,
    previous_context: RoleContext,
    result_context: RoleContext,
    adapter: OllamaUpdaterAdapter,
) -> dict[str, Any]:
    context_path = output_dir / f"{target_name}_updated_context.txt"
    updated_text = (
        result_context.game if target_name.endswith("_game") else result_context.general
    )
    context_path.write_text(updated_text.rstrip() + "\n", encoding="utf-8")

    artifact = {
        "target": target_name,
        "status": status,
        "error": error,
        "context_file": str(context_path.relative_to(ROOT)),
        "previous_context": asdict(previous_context),
        "updated_context": asdict(result_context),
        "request": _sanitize_request(adapter.provider.last_request),
        "response": _sanitize_response(adapter.provider.last_response),
    }
    artifact_path = output_dir / f"{target_name}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return {
        "target": target_name,
        "status": status,
        "metadata_file": str(artifact_path.relative_to(ROOT)),
        "context_file": str(context_path.relative_to(ROOT)),
    }


def _sanitize_request(request: dict[str, Any] | None) -> dict[str, Any] | None:
    if request is None:
        return None
    return _sanitize_value(request)


def _sanitize_response(response: Any | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return {
        "message_content": _safe_message_content(response),
        "usage": {
            key: object_get(response, key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
                "done_reason",
            )
            if object_get(response, key) is not None
        },
        "raw": _sanitize_value(to_memory_jsonable(response)),
    }


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "images" and isinstance(item, list):
                sanitized[key] = [
                    {"kind": "base64_png_omitted", "base64_chars": len(image)}
                    if isinstance(image, str)
                    else _sanitize_value(image)
                    for image in item
                ]
            else:
                sanitized[key] = _sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _safe_message_content(response: Any) -> str:
    try:
        return message_content(response)
    except Exception as exc:
        return f"<missing message content: {type(exc).__name__}: {exc}>"


def _selected_targets(raw_targets: list[str] | tuple[str, ...]) -> list[UpdaterTarget]:
    if "all" in raw_targets:
        return list(TARGET_ORDER)
    selected: list[UpdaterTarget] = []
    for target in TARGET_ORDER:
        if target in raw_targets:
            selected.append(target)
    return selected


def _target_role(target_name: UpdaterTarget) -> Literal["world", "goal", "agent"]:
    if target_name.startswith("world"):
        return "world"
    if target_name.startswith("goal"):
        return "goal"
    return "agent"


def _model_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "ollama",
        "model": args.model,
        "host": args.host,
        "think": args.think,
        "temperature": args.temperature,
        "keep_alive": args.keep_alive,
        "input_image_size": args.input_image_size,
        "input_image_resample": args.input_image_resample,
    }


def _fixture_description(label: str) -> list[dict[str, object]]:
    return [
        {
            "bbox_2d": [0, 0, 64, 64],
            "description": label,
        }
    ]


if __name__ == "__main__":
    main()
