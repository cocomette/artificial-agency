"""Fail-open helpers for model-role failures inside the game loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    DecisionResult,
    FrameTurnContext,
    Observation,
)
from face_of_agi.models.change.adapter import ChangeSummaryOutputError
from face_of_agi.models.action_coordinates import action6_coordinate_bounds
from face_of_agi.models.historizer.adapter import HistorizerOutputError
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.observation_text import (
    ObservationTextConfig,
    cropped_changed_cell_count,
)
from face_of_agi.models.orchestrator_agent.tooling import AgentOutputError
from face_of_agi.models.updater.adapter import UpdaterOutputError

try:
    from openai import OpenAIError
except ImportError:  # pragma: no cover - openai is a runtime dependency.
    OpenAIError = ()  # type: ignore[assignment]

_AGENT_REPAIR_FAILURE_TEXT = " X produced invalid structured agent step "
_VLLM_PROVIDER_FAILURE_PREFIXES = (
    "vLLM chat response did not include",
    "vLLM context overflow",
)


def fallback_decision_result(
    *,
    frame_context: FrameTurnContext,
    turn_id: int,
    action_space: Sequence[ActionSpec],
    error: Exception,
    observation_text_config: Any | None = None,
) -> DecisionResult:
    """Return a legal deterministic action when Agent X cannot produce one."""

    final_action = fallback_action(
        frame_context=frame_context,
        turn_id=turn_id,
        action_space=action_space,
        observation_text_config=observation_text_config,
    )
    trace = AgentTrace(
        step=frame_context.current_observation.step,
        first_observation_ref=frame_context.first_observation_ref,
        current_observation_ref=frame_context.current_observation_ref,
        final_action=final_action,
        reasoning_summary="Agent X failed; orchestration selected fallback action.",
        metadata={
            "decision_source": "orchestration_fallback",
            "agent_x_called": True,
            "fallback": "agent_decision_error",
            "fallback_error_type": type(error).__name__,
            "fallback_error": str(error),
        },
    )
    return DecisionResult(final_action=final_action, trace=trace)


def fallback_action(
    *,
    frame_context: FrameTurnContext,
    turn_id: int,
    action_space: Sequence[ActionSpec],
    observation_text_config: Any | None = None,
) -> ActionSpec:
    """Choose a valid action from the active prompt-facing action space."""

    actions = tuple(action for action in action_space if not action.is_none())
    if not actions:
        raise RuntimeError("fallback action requires at least one real action")

    for action in actions:
        if not _requires_action_data(action):
            return action

    for action in actions:
        if action.name == "ACTION6":
            x, y = fallback_action6_coordinates(
                turn_id=turn_id,
                observation_text_config=observation_text_config,
            )
            return ActionSpec(
                action_id=action.action_id,
                data={"x": x, "y": y},
                target=f"fallback probe at ({x},{y})",
            )

    return actions[0]


def fallback_action6_coordinates(
    *,
    turn_id: int,
    observation_text_config: Any | None = None,
) -> tuple[int, int]:
    """Return deterministic visible-crop probe coordinates for ACTION6."""

    minimum, maximum = action6_coordinate_bounds(observation_text_config)
    span = maximum - minimum
    probes = (
        (0.5, 0.5),
        (0.25, 0.5),
        (0.75, 0.5),
        (0.5, 0.25),
        (0.5, 0.75),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    )
    x_fraction, y_fraction = probes[(turn_id - 1) % len(probes)]
    return (
        int(round(minimum + (span * x_fraction))),
        int(round(minimum + (span * y_fraction))),
    )


def fallback_change_summary_result(
    *,
    observations: Sequence[Observation],
    error: Exception,
    observation_text_config: Any | None = None,
) -> ChangeSummaryResult:
    """Build deterministic transition evidence when the change model fails."""

    evidence = tuple(observations)
    if len(evidence) < 2:
        raise ValueError("fallback change summary requires at least two frames")
    first = evidence[0]
    final = evidence[-1]
    changed_cell_count = cropped_changed_cell_count(
        first.frame,
        final.frame,
        config=observation_text_config,
    )
    changed_cell_percent = _cropped_changed_cell_percent(
        changed_cell_count,
        observation_text_config=observation_text_config,
    )
    change_detected = any(
        cropped_changed_cell_count(
            left.frame,
            right.frame,
            config=observation_text_config,
        )
        > 0
        for left, right in zip(evidence, evidence[1:], strict=False)
    )
    summary = (
        "Visible changes occurred, but model summary unavailable."
        if change_detected
        else "no changes"
    )
    return ChangeSummaryResult(
        summary=summary,
        changed_pixel_count=changed_cell_count,
        change_detected=change_detected,
        metadata={
            "fallback": "change_summary_error",
            "fallback_error_type": type(error).__name__,
            "fallback_error": str(error),
            "frame_count": len(evidence),
            "serialized_frame_count": len(evidence),
            "source_frame_count": len(evidence),
            "deterministic_change_detected": change_detected,
        },
        changed_cell_percent=changed_cell_percent,
    )


def model_observation_text_config(model: object) -> Any | None:
    """Return a role adapter's observation text config when it exposes one."""

    config = getattr(model, "config", None)
    return getattr(config, "observation_text", None)


def is_agent_model_failure(error: Exception) -> bool:
    """Return whether an Agent X exception is a model/provider output failure."""

    return (
        _is_openai_error(error)
        or isinstance(error, AgentOutputError)
        or _runtime_error_contains(error, _AGENT_REPAIR_FAILURE_TEXT)
        or _is_vllm_provider_runtime_error(error)
    )


def is_change_model_failure(error: Exception) -> bool:
    """Return whether a change-summary exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, ChangeSummaryOutputError)
        or _is_vllm_provider_runtime_error(error)
    )


def is_historizer_model_failure(error: Exception) -> bool:
    """Return whether a historizer exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, HistorizerOutputError)
        or _is_vllm_provider_runtime_error(error)
    )


def is_updater_model_failure(error: Exception) -> bool:
    """Return whether an updater exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, UpdaterOutputError)
        or _is_vllm_provider_runtime_error(error)
    )


def _requires_action_data(action: ActionSpec) -> bool:
    return action.is_complex() or action.name == "ACTION6"


def _is_openai_error(error: Exception) -> bool:
    return bool(OpenAIError) and isinstance(error, OpenAIError)


def _runtime_error_contains(error: Exception, text: str) -> bool:
    return isinstance(error, RuntimeError) and text in str(error)


def _is_vllm_provider_runtime_error(error: Exception) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    return any(
        message.startswith(prefix)
        for prefix in _VLLM_PROVIDER_FAILURE_PREFIXES
    )


def _cropped_changed_cell_percent(
    changed_cell_count: int,
    *,
    observation_text_config: Any | None,
) -> float:
    config = _observation_text_config(observation_text_config)
    visible_axis = 64 - (2 * config.crop_cells)
    if visible_axis <= 0:
        raise ValueError("observation_text.crop_cells leaves an empty crop")
    return min(
        100.0,
        max(
            0.0,
            float(changed_cell_count) * 100.0 / float(visible_axis * visible_axis),
        ),
    )


def _observation_text_config(value: Any | None) -> ObservationTextConfig:
    if value is None:
        return ObservationTextConfig()
    if isinstance(value, ObservationTextConfig):
        return value
    if isinstance(value, dict):
        return ObservationTextConfig(**value)
    raise TypeError("observation_text_config must be an ObservationTextConfig")
