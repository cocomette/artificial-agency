"""Manual E2E check for OpenAI-backed S/G prompt updaters.

This script calls the real OpenAI API. Set OPENAI_API_KEY before running it.
It exercises the world and goal game-context updaters with committed ARC
fixture images, then exercises the world and goal general-context updaters with
the resulting text context.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from arcengine import GameAction

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    TurnMetrics,
    RoleContext,
    ToolResult,
)
from face_of_agi.models.updater import (
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    WorldGameContextUpdateInput,
)
from face_of_agi.models.updater.config import OpenAIUpdaterConfig
from face_of_agi.models.updater.providers.openai import OpenAIUpdaterAdapter

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "world"
SOURCE_PATH = FIXTURE_DIR / "ls20_seed0_step0_source.png"
TARGET_PATH = FIXTURE_DIR / "ls20_seed0_action1_target.png"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "openai_updater_e2e"
GAME_ID = "ls20-9607627b"
RUN_ID = "openai-updater-e2e"

UpdaterTarget = Literal[
    "world_game",
    "goal_game",
    "world_general",
    "goal_general",
]

TARGET_ORDER: tuple[UpdaterTarget, ...] = (
    "world_game",
    "goal_game",
    "world_general",
    "goal_general",
)
TARGET_CHOICES = ("all", *TARGET_ORDER)


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    output_dir = _resolve_output_dir(args.output_dir)
    selected_targets = _selected_targets(args.targets)

    source = Image.open(SOURCE_PATH).convert("RGB")
    target = Image.open(TARGET_PATH).convert("RGB")
    adapter = OpenAIUpdaterAdapter(_config_from_args(args))

    contexts = {
        "world": RoleContext(
            general=(
                "Infer reusable ARC transition rules from visual evidence. "
                "Prefer mechanics that transfer across similar grid worlds."
            ),
            game=(
                "Game ls20-9607627b uses a small grid-like scene. Current "
                "hypothesis: ACTION1 may move or transform the active object, "
                "but the exact transition rule is uncertain."
            ),
        ),
        "goal": RoleContext(
            general=(
                "Infer reusable ARC goal patterns from visual evidence. "
                "Prefer concise hypotheses about progress and completion."
            ),
            game=(
                "Game ls20-9607627b goal hypothesis: compare the current frame "
                "with the next observed frame to identify what progress means."
            ),
        ),
    }

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
        "world_general": lambda: _run_world_general_update(
            adapter=adapter,
            previous_context=contexts["world"],
        ),
        "goal_general": lambda: _run_goal_general_update(
            adapter=adapter,
            previous_context=contexts["goal"],
        ),
    }

    for target_name in selected_targets:
        previous_context = (
            contexts["world"] if target_name.startswith("world") else contexts["goal"]
        )
        result_context = target_calls[target_name]()
        _validate_update(
            target_name=target_name,
            previous_context=previous_context,
            result_context=result_context,
        )
        if target_name.startswith("world"):
            contexts["world"] = result_context
        else:
            contexts["goal"] = result_context
        artifacts.append(
            _write_target_artifacts(
                output_dir=output_dir,
                target_name=target_name,
                previous_context=previous_context,
                result_context=result_context,
                adapter=adapter,
            )
        )

    summary = {
        "fixture": {
            "game_id": GAME_ID,
            "seed": 0,
            "action": "ACTION1",
            "predicted_frame_source": str(SOURCE_PATH.relative_to(ROOT)),
            "actual_next_frame": str(TARGET_PATH.relative_to(ROOT)),
            "fixture_note": (
                "The source frame is used as a deliberately imperfect "
                "prediction; the target frame is the actual next observation."
            ),
        },
        "model": _model_summary(args),
        "selected_targets": selected_targets,
        "artifacts": artifacts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"saved OpenAI updater E2E artifacts to {output_dir}")
    print(json.dumps({"selected_targets": selected_targets}, indent=2))


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
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--input-image-detail", default="auto")
    parser.add_argument(
        "--input-image-size",
        default="1024x1024",
        help="Optional WxH resize applied to updater input images.",
    )
    parser.add_argument(
        "--input-image-resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="nearest",
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_CHOICES,
        default=("all",),
        help=(
            "Targets to run. Use 'all' for world_game, goal_game, "
            "world_general, and goal_general."
        ),
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> OpenAIUpdaterConfig:
    return OpenAIUpdaterConfig(
        backend="openai",
        api_key_env=args.api_key_env,
        model=args.model,
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        input_image_detail=args.input_image_detail,
        input_image_size=args.input_image_size,
        input_image_resample=args.input_image_resample,
        timeout=args.timeout,
        max_retries=args.max_retries,
        metadata={"role": "updater", "script": "openai_updater_e2e"},
    )


def _run_world_game_update(
    *,
    adapter: OpenAIUpdaterAdapter,
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
            current_observation_ref=observation_ref,
            actual_next_observation_ref=actual_observation_ref,
            previous_observation=Observation(
                id=observation_ref.id,
                step=0,
                frame=source,
            ),
            actual_next_observation=Observation(
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
                    explanation=(
                        "E2E fixture prediction uses the source frame on "
                        "purpose so the updater sees a visible mismatch."
                    ),
                )
            ),
            turn_metrics=TurnMetrics(
                time_cost=1.0,
            ),
            submitted_action=action,
            metadata={"fixture": GAME_ID, "target": "world_game"},
        )
    )


def _run_goal_game_update(
    *,
    adapter: OpenAIUpdaterAdapter,
    previous_context: RoleContext,
    source: Image.Image,
    target: Image.Image,
) -> RoleContext:
    observation_ref = ObservationRef(memory="state", id="ls20-seed0-step0")
    actual_observation_ref = ObservationRef(memory="state", id="ls20-seed0-step1")
    return adapter.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=previous_context,
            current_observation_ref=observation_ref,
            actual_next_observation_ref=actual_observation_ref,
            previous_observation=Observation(
                id=observation_ref.id,
                step=0,
                frame=source,
            ),
            actual_next_observation=Observation(
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
                    explanation=(
                        "E2E fixture prediction uses the source frame on "
                        "purpose so the updater sees a visible mismatch."
                    ),
                )
            ),
            turn_metrics=TurnMetrics(
                time_cost=1.0,
            ),
            submitted_action=ActionSpec(action_id=GameAction.ACTION1),
            metadata={"fixture": GAME_ID, "target": "goal_game"},
        )
    )


def _run_world_general_update(
    *,
    adapter: OpenAIUpdaterAdapter,
    previous_context: RoleContext,
) -> RoleContext:
    return adapter.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role="world",
            previous_context=previous_context,
            run_id=RUN_ID,
            game_id=GAME_ID,
            stop_reason="manual_e2e_complete",
            step_count=1,
            completed_levels=0,
            final_state="fixture_transition",
            metadata={"fixture": GAME_ID, "target": "world_general"},
        )
    )


def _run_goal_general_update(
    *,
    adapter: OpenAIUpdaterAdapter,
    previous_context: RoleContext,
) -> RoleContext:
    return adapter.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role="goal",
            previous_context=previous_context,
            run_id=RUN_ID,
            game_id=GAME_ID,
            stop_reason="manual_e2e_complete",
            step_count=1,
            completed_levels=0,
            final_state="fixture_transition",
            metadata={"fixture": GAME_ID, "target": "goal_general"},
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
    previous_context: RoleContext,
    result_context: RoleContext,
    adapter: OpenAIUpdaterAdapter,
) -> dict[str, Any]:
    context_path = output_dir / f"{target_name}_updated_context.txt"
    updated_text = (
        result_context.game if target_name.endswith("_game") else result_context.general
    )
    context_path.write_text(updated_text.rstrip() + "\n", encoding="utf-8")

    artifact = {
        "target": target_name,
        "context_file": str(context_path.relative_to(ROOT)),
        "previous_context": asdict(previous_context),
        "updated_context": asdict(result_context),
        "request": _sanitize_request(adapter.provider.last_request),
    }
    artifact_path = output_dir / f"{target_name}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return {
        "target": target_name,
        "metadata_file": str(artifact_path.relative_to(ROOT)),
        "context_file": str(context_path.relative_to(ROOT)),
    }


def _sanitize_request(request: dict[str, Any] | None) -> dict[str, Any] | None:
    if request is None:
        return None
    return _sanitize_value(request)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "image_url" and isinstance(item, str):
                sanitized[key] = _image_url_summary(item)
            else:
                sanitized[key] = _sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _image_url_summary(image_url: str) -> dict[str, Any]:
    prefix, _, encoded = image_url.partition(",")
    mime_type = prefix.removeprefix("data:").split(";", 1)[0]
    return {
        "kind": "data_url_omitted",
        "mime_type": mime_type,
        "base64_chars": len(encoded),
    }


def _selected_targets(raw_targets: list[str] | tuple[str, ...]) -> list[UpdaterTarget]:
    if "all" in raw_targets:
        return list(TARGET_ORDER)
    selected: list[UpdaterTarget] = []
    for target in TARGET_ORDER:
        if target in raw_targets:
            selected.append(target)
    return selected


def _model_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "openai",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "input_image_detail": args.input_image_detail,
        "input_image_size": args.input_image_size,
        "input_image_resample": args.input_image_resample,
    }


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fixture_description(label: str) -> list[dict[str, object]]:
    return [
        {
            "bbox_2d": [0, 0, 64, 64],
            "description": label,
        }
    ]


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
