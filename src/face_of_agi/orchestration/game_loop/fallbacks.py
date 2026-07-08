"""Fail-open helpers for model-role failures inside the game loop."""

from __future__ import annotations

from collections.abc import Sequence

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    DecisionResult,
    FrameTurnContext,
    Observation,
)
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.change.adapter import (
    ChangeSummaryOutputError,
    change_summary_observation_images,
    model_visible_any_change_detected,
    model_visible_changed_pixel_count,
    model_visible_changed_pixel_percent,
)
from face_of_agi.models.historizer.adapter import HistorizerOutputError
from face_of_agi.models.memory.adapter import GameMemoryOutputError
from face_of_agi.models.orchestrator_agent.tooling import AgentOutputError
from face_of_agi.models.providers.scheduler import ModelSchedulerTimeoutError
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
) -> DecisionResult:
    """Return a legal deterministic action when Agent X cannot produce one."""

    final_action = fallback_action(
        turn_id=turn_id,
        action_space=action_space,
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
    turn_id: int,
    action_space: Sequence[ActionSpec],
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
            x, y = fallback_action6_coordinates(turn_id=turn_id)
            return ActionSpec(
                action_id=action.action_id,
                data={"x": x, "y": y},
                target=f"fallback probe at ({x},{y})",
            )

    return actions[0]


def fallback_action6_coordinates(*, turn_id: int) -> tuple[int, int]:
    """Return deterministic ARC-grid probe coordinates for ACTION6."""

    probes = (
        (32, 32),
        (16, 32),
        (48, 32),
        (32, 16),
        (32, 48),
        (16, 16),
        (48, 16),
        (16, 48),
        (48, 48),
        (0, 0),
        (63, 0),
        (0, 63),
        (63, 63),
    )
    return probes[(turn_id - 1) % len(probes)]


def fallback_change_summary_result(
    *,
    observations: Sequence[Observation],
    error: Exception,
    frame_scale: int,
    size: str | tuple[int, int] | None,
    resample: str,
    crop_box_normalized: object | None,
) -> ChangeSummaryResult:
    """Build deterministic transition evidence when the change model fails."""

    evidence = tuple(observations)
    if len(evidence) < 2:
        raise ValueError("fallback change summary requires at least two frames")
    images = change_summary_observation_images(
        evidence,
        frame_scale=frame_scale,
        size=size,
        resample=resample,
        crop_box_normalized=crop_box_normalized,
    )
    changed_pixel_count = model_visible_changed_pixel_count(images[0], images[-1])
    change_detected = model_visible_any_change_detected(images)
    elements: tuple[ChangeSummaryElement, ...] = ()
    if change_detected:
        elements = (
            ChangeSummaryElement(
                element_name="visible_scene",
                element_description="Model-visible frame content",
                element_mutation=(
                    "Visible changes occurred, but model summary is unavailable."
                ),
            ),
        )
    return ChangeSummaryResult(
        elements=elements,
        changed_pixel_count=changed_pixel_count,
        change_detected=change_detected,
        metadata={
            "fallback": "change_summary_error",
            "fallback_error_type": type(error).__name__,
            "fallback_error": str(error),
            "frame_count": len(evidence),
            "any_adjacent_frame_changed": change_detected,
        },
        changed_pixel_percent=model_visible_changed_pixel_percent(
            images[0],
            images[-1],
            changed_pixel_count=changed_pixel_count,
        ),
    )


def is_agent_model_failure(error: Exception) -> bool:
    """Return whether an Agent X exception is a model/provider output failure."""

    return (
        _is_openai_error(error)
        or isinstance(error, ModelSchedulerTimeoutError)
        or isinstance(error, AgentOutputError)
        or _runtime_error_contains(error, _AGENT_REPAIR_FAILURE_TEXT)
        or _is_vllm_provider_runtime_error(error)
    )


def is_change_model_failure(error: Exception) -> bool:
    """Return whether a change-summary exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, ModelSchedulerTimeoutError)
        or isinstance(error, ChangeSummaryOutputError)
        or _is_vllm_provider_runtime_error(error)
    )


def is_historizer_model_failure(error: Exception) -> bool:
    """Return whether a historizer exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, ModelSchedulerTimeoutError)
        or isinstance(error, HistorizerOutputError)
        or _is_vllm_provider_runtime_error(error)
    )


def is_memory_model_failure(error: Exception) -> bool:
    """Return whether a game-memory exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, ModelSchedulerTimeoutError)
        or isinstance(error, GameMemoryOutputError)
        or _is_vllm_provider_runtime_error(error)
    )


def is_updater_model_failure(error: Exception) -> bool:
    """Return whether an updater exception is model/provider related."""

    return (
        _is_openai_error(error)
        or isinstance(error, ModelSchedulerTimeoutError)
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
    return any(message.startswith(prefix) for prefix in _VLLM_PROVIDER_FAILURE_PREFIXES)
