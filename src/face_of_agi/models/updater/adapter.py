"""Adapter shell for updater model backends."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionOutcomeEvidence,
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.frames import (
    FRAME_PAYLOAD_TYPE,
    observation_to_pil_image,
    to_memory_jsonable,
)
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
    model_facing_action_text_for_crop,
)
from face_of_agi.models.historizer import (
    AGENT_CONTEXT_HISTORY_KEYS,
    AgentContextHistorySummary,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)
from face_of_agi.models.updater.config import UpdaterConfig
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_KEYS,
    AGENT_GAME_CONTEXT_MAX_CHARS,
    ContextSegment,
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    PromptImage,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    UpdaterContextTarget,
    UpdaterRole,
    UpdaterTask,
    updater_output_json_schema,
)
DEFAULT_INSTRUCTION_DIR = Path(__file__).parent / "instructions"


class PromptUpdaterProvider(Protocol):
    """Thin backend boundary for one prompt update request."""

    backend: str
    model: str | None

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        """Return raw provider text for the selected context segment."""
        ...

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        """Return repaired raw provider text for invalid structured output."""
        ...


class UpdaterOutputError(RuntimeError):
    """Raised when a real updater backend returns an invalid update payload."""


class PromptUpdaterAdapter:
    """Provider-neutral prompt updater that delegates only the model call."""

    def __init__(
        self,
        provider: PromptUpdaterProvider,
        config: UpdaterConfig | None = None,
    ) -> None:
        self.config = config or UpdaterConfig()
        self.provider = provider

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        """Update the agent game context."""

        return self._update_context(
            role="agent",
            segment="game",
            task="agent_game",
            previous_context=update_input.previous_context,
            text=_agent_game_prompt_text(
                update_input,
                crop_edges=self.config.input_image_crop_arc_grid_edges,
            ),
            glossary_actions=update_input.glossary_actions,
            images=_current_observation_image(
                update_input.current_observation,
                role="agent game",
                frame_scale=self.config.frame_scale,
            ),
        )

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        """Update one role's general knowledge context."""

        return self._update_context(
            role=update_input.role,
            segment="general",
            task="general",
            previous_context=update_input.previous_context,
            text=json.dumps(
                _prompt_payload(
                    role=update_input.role,
                    segment="general",
                    task="general",
                    previous_context=update_input.previous_context,
                    input_payload=_input_payload(update_input),
                ),
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

    def _update_context(
        self,
        *,
        role: UpdaterRole,
        segment: ContextSegment,
        task: UpdaterTask,
        previous_context: RoleContext,
        text: str,
        images: tuple[PromptImage, ...] = (),
        glossary_actions: Sequence[ActionSpec] | None = None,
    ) -> RoleContext:
        if task == "agent_game" and glossary_actions is None:
            raise ValueError(f"{task} updater requires glossary actions")
        target = UpdaterContextTarget(
            role=role,
            segment=segment,
            task=task,
            previous_context=previous_context,
        )
        output_schema = updater_output_json_schema(task)
        instructions_text = load_updater_instructions(
            task=task,
            role=role,
            instruction_dir=self.config.instruction_dir,
        )
        if task == "agent_game":
            assert glossary_actions is not None
            instructions_text = append_action_glossary(
                instructions_text,
                glossary_actions,
                mode="agent_update",
            )
        instructions = append_output_schema_to_instructions(
            instructions_text,
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptUpdateRequest(
            target=target,
            instructions=instructions,
            text=text,
            output_schema=output_schema,
            images=images,
            metadata={
                "backend": self.provider.backend,
                "model": self.provider.model,
            },
        )
        response = self.provider.update_prompt(request)
        validated = validate_with_repair(
            label=f"{self.provider.backend} updater",
            response=response,
            text_of=lambda item: item.text,
            validate=_updated_context_validator(
                task=task,
            ),
            repair=provider_repair_callback(
                self.provider,
                "repair_prompt",
                args=(request,),
            ),
            max_repair_attempts=getattr(self.config, "repair_attempts", 0),
            error_factory=UpdaterOutputError,
        )
        return _with_updated_segment(previous_context, segment, validated.value)


def load_updater_instructions(
    *,
    task: UpdaterTask,
    role: UpdaterRole | None = None,
    instruction_dir: str | Path | None = None,
) -> str:
    """Load the human-editable updater instruction prompt for one target."""

    path = updater_instruction_path(
        task=task,
        role=role,
        instruction_dir=instruction_dir,
    )
    return path.read_text(encoding="utf-8").strip()


def updater_instruction_path(
    *,
    task: UpdaterTask,
    role: UpdaterRole | None = None,
    instruction_dir: str | Path | None = None,
) -> Path:
    """Return the configured instruction file path for one updater target."""

    root = (
        Path(instruction_dir)
        if instruction_dir is not None
        else DEFAULT_INSTRUCTION_DIR
    )
    if task == "general":
        if role is None:
            raise ValueError("general updater instructions require a target role")
        return root / f"{role}_general_context_updater_prompt.md"
    return root / f"{task}_context_updater_prompt.md"


def _with_updated_segment(
    context: RoleContext,
    segment: ContextSegment,
    updated_text: str,
) -> RoleContext:
    if segment == "general":
        return RoleContext(general=updated_text, game=context.game)
    return RoleContext(general=context.general, game=updated_text)


def parse_updated_context_output(text: str) -> str:
    """Parse the required JSON updater output contract."""

    return _parse_string_updated_context_output(text)


def parse_agent_game_updated_context_output(text: str) -> str:
    """Parse agent-game context JSON and return it as context text."""

    loaded = _load_updated_context_json(text)
    updated_context = loaded.get("updated_context")
    if not isinstance(updated_context, dict):
        raise UpdaterOutputError(
            "agent game updater response JSON is missing object field "
            "'updated_context'"
        )

    missing = [key for key in AGENT_GAME_CONTEXT_KEYS if key not in updated_context]
    if missing:
        raise UpdaterOutputError(
            "agent game updater updated_context is missing keys: "
            + ", ".join(missing)
        )
    unexpected = sorted(set(updated_context) - set(AGENT_GAME_CONTEXT_KEYS))
    if unexpected:
        raise UpdaterOutputError(
            "agent game updater updated_context has unexpected keys: "
            + ", ".join(unexpected)
        )
    invalid = [
        key for key, value in updated_context.items() if not isinstance(value, str)
    ]
    if invalid:
        raise UpdaterOutputError(
            "agent game updater updated_context values must be strings: "
            + ", ".join(sorted(invalid))
        )
    ordered_context = {
        key: updated_context[key] for key in AGENT_GAME_CONTEXT_KEYS
    }
    updated_text = json.dumps(ordered_context, indent=2, ensure_ascii=False)
    _validate_agent_game_context_length(updated_text)
    return updated_text


def _validate_agent_game_context_length(updated_text: str) -> None:
    if len(updated_text) <= AGENT_GAME_CONTEXT_MAX_CHARS:
        return
    raise UpdaterOutputError(
        "agent game updater updated_context is too long: "
        f"{len(updated_text)} characters exceeds the "
        f"{AGENT_GAME_CONTEXT_MAX_CHARS} character cap. Revise the full "
        "context below the cap by removing stale details, duplicate evidence, "
        "and chronological action logs while preserving current goals, "
        "mechanics, policy, history, and extras that improve the next "
        "decision."
    )


def _updated_context_parser(task: UpdaterTask):
    if task == "agent_game":
        return parse_agent_game_updated_context_output
    return parse_updated_context_output


def _updated_context_validator(
    *,
    task: UpdaterTask,
):
    def validate(text: str) -> str:
        parser = _updated_context_parser(task)
        return parser(text)

    return validate


def _parse_string_updated_context_output(text: str) -> str:
    """Parse the default JSON updater output contract."""

    loaded = _load_updated_context_json(text)
    updated_context = loaded.get("updated_context")
    if not isinstance(updated_context, str):
        raise UpdaterOutputError(
            "updater response JSON is missing string field 'updated_context'"
        )
    return updated_context


def _load_updated_context_json(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise UpdaterOutputError(
            "updater response must be JSON with an 'updated_context' field; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise UpdaterOutputError(
            "updater response must be a JSON object with an 'updated_context' field"
        )
    return loaded


def _strip_json_fence(text: str) -> str:
    """Accept simple fenced JSON while rejecting prose-wrapped responses."""

    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return stripped
    return match.group(1).strip()


def _prompt_payload(
    *,
    role: UpdaterRole,
    segment: ContextSegment,
    task: UpdaterTask,
    previous_context: RoleContext,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the provider-neutral user payload for a prompt update task."""

    selected_context = (
        previous_context.general
        if segment == "general"
        else previous_context.game
    )
    return {
        "task": task,
        "role": role,
        "segment": segment,
        "current_context": selected_context,
        "previous_context": asdict(previous_context),
        "transition": input_payload,
    }


def _input_payload(update_input: object) -> dict[str, Any]:
    if is_dataclass(update_input):
        return _prompt_jsonable(asdict(update_input))
    return {}


def _prompt_jsonable(value: Any) -> Any:
    """Return JSON-safe updater input without embedding raw image payloads."""

    return _summarize_frame_payloads(to_memory_jsonable(value))


def _summarize_frame_payloads(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__type__") == FRAME_PAYLOAD_TYPE:
            return {
                "__type__": FRAME_PAYLOAD_TYPE,
                "kind": "image_summary",
                "mime_type": value.get("mime_type"),
                "width": value.get("width"),
                "height": value.get("height"),
                "encoding": "base64_omitted_for_prompt",
            }
        return {
            str(key): _summarize_frame_payloads(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_summarize_frame_payloads(item) for item in value]
    return value


def _current_observation_image(
    current_observation: Observation | None,
    *,
    role: str,
    frame_scale: int,
) -> tuple[PromptImage, ...]:
    if current_observation is None:
        raise ValueError(f"{role} updater requires a current observation")
    return (
        PromptImage(
            label="current_observation_frame",
            image=observation_to_pil_image(
                current_observation,
                frame_scale=frame_scale,
            ),
        ),
    )


def _agent_game_prompt_text(
    update_input: AgentGameContextUpdateInput,
    *,
    crop_edges: Any | None,
) -> str:
    return "\n\n".join(
        [
            "## Previous agent game context\n\n"
            + _text_or_none(update_input.previous_context.game),
            "## Allowed actions\n\n"
            + _allowed_actions_text(
                update_input.allowed_actions,
                crop_edges=crop_edges,
            ),
            "## Action outcome evidence\n\n"
            + _action_outcome_evidence_text(update_input.action_outcome_evidence),
            "## Action history window\n\n"
            + _action_history_window_text(update_input.action_history_window),
            "## Action history\n\n"
            + _numbered_action_history_text(
                update_input.action_history,
                crop_edges=crop_edges,
            ),
            "## Agent context history\n\n"
            + _agent_context_history_text(update_input.context_history),
            "## Progress feedback\n\n"
            + _progress_feedback_text(update_input.turn_metrics),
            "## Context revision feedback\n\n"
            + _context_revision_feedback_text(
                update_input.context_revision_feedback
            ),
        ]
    )


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _allowed_actions_text(
    action_space: tuple[ActionSpec, ...],
    *,
    crop_edges: Any | None,
) -> str:
    if not action_space:
        return "none"
    lines = [
        (
            "These are the only actions the agent may choose from in this "
            "turn. The action glossary may include raw game actions that "
            "are not allowed in this turn."
        )
    ]
    lines.extend(
        f"- {_action_text(action, crop_edges=crop_edges)}"
        for action in action_space
    )
    return "\n".join(lines)


def _action_history_window_text(window: int) -> str:
    if window < 0:
        raise ValueError("agent updater action history window must be non-negative")
    return "\n".join(
        [
            f"- prior_action_group_window: {window}",
            (
                "- rows: up to this many prior controllable action groups, "
                "followed by the latest transition group"
            ),
        ]
    )


def _numbered_action_history_text(
    history: tuple[ActionHistoryItem, ...],
    *,
    crop_edges: Any | None,
) -> str:
    if not history:
        return "none"
    lines = [
        (
            "Numbered oldest-to-newest. Controllable action rows may include "
            "animation evidence fields; nested animation_after rows mark "
            "synthetic animation-only turns. GAME_RESET rows mark environment "
            "resets between action groups, and SCORE_ADVANCE rows mark score "
            "or progress increases. The [latest] marker identifies the "
            "transition, reset, or score marker that produced the attached "
            "current_observation_frame. ACTION6 data in this history is "
            "rendered as normalized visual 0..1000 coordinates, matching the "
            "coordinate space used for future ACTION6 outputs."
        )
    ]
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text_for_crop(crop_edges),
        numbered=True,
        latest_description=lines[0],
    )


def _action_outcome_evidence_text(evidence: ActionOutcomeEvidence) -> str:
    lines: list[str] = [
        f"- suppression_threshold: {evidence.suppression_threshold}",
    ]
    if evidence.suppressed_actions:
        lines.append(
            "- suppressed_action_choices: " + ", ".join(evidence.suppressed_actions)
        )
        if evidence.suppression_reason:
            lines.append("- suppression_reason: " + evidence.suppression_reason)
    elif evidence.suppression_disabled_reason:
        lines.append(
            "- suppression_disabled_reason: "
            + evidence.suppression_disabled_reason
        )

    lines.append(
        "- latest_same_action_zero_changed_pixel_turn_count: "
        f"{evidence.latest_same_action_zero_changed_pixel_turn_count}"
    )
    lines.append(
        f"- stagnation_warning_threshold: "
        f"{evidence.stagnation_warning_threshold}"
    )
    if evidence.stagnation_warning:
        lines.append(
            "- STAGNATION_WARNING: ACTIVE; THE LATEST REPEATED CONTROLLABLE "
            "ACTION HIT THE CHANGED_PIXEL_PERCENT=0 WARNING THRESHOLD. "
            "IMMEDIATELY REVISE POLICY AND/OR GOALS: STOP THAT "
            "LOW-INFORMATION ACTION PATTERN, REPLACE ANY STALE GOAL "
            "HYPOTHESIS, AND FORCE A CONCRETE EXPLORATORY ACTION SEQUENCE "
            "USING ONLY THE CURRENTLY ALLOWED ACTIONS."
        )
    else:
        lines.append("- stagnation_warning: inactive")
    return "\n".join(lines)


def _progress_feedback_text(feedback: Any) -> str:
    return "\n".join(
        [
            f"- time_cost: {_metric_text(feedback.time_cost)}",
            f"- cumulative_score: {_metric_text(feedback.cumulative_score)}",
            (
                "- game_last_started_turns_ago: "
                f"{_metric_text(feedback.game_last_started_turns_ago)}"
            ),
            (
                "- score_last_advanced_turns_ago: "
                f"{_metric_text(feedback.score_last_advanced_turns_ago)}"
            ),
            f"- game_start_reason: {_text_or_none(feedback.game_start_reason)}",
            f"- game_restart_count: {feedback.game_restart_count}",
        ]
    )


def _agent_context_history_text(summary: AgentContextHistorySummary) -> str:
    if not summary.is_available():
        return "not available"
    return "\n".join(
        f"- {key}: {_text_or_none(summary.field_evolution.get(key))}"
        for key in AGENT_CONTEXT_HISTORY_KEYS
    )


def _context_revision_feedback_text(feedback: Any) -> str:
    return "\n".join(
        [
            f"- compared_turns: {feedback.compared_turns}",
            f"- goals_unchanged_turns: {feedback.goals_unchanged_turns}",
            (
                "- game_mechanics_unchanged_turns: "
                f"{feedback.game_mechanics_unchanged_turns}"
            ),
            f"- policy_unchanged_turns: {feedback.policy_unchanged_turns}",
            f"- history_unchanged_turns: {feedback.history_unchanged_turns}",
            f"- extras_unchanged_turns: {feedback.extras_unchanged_turns}",
        ]
    )


def _metric_text(value: float | int | None) -> str:
    if value is None:
        return "none"
    return str(value)


def _action_text(
    action: Any,
    *,
    crop_edges: Any | None,
) -> str:
    if isinstance(action, ActionSpec):
        return model_facing_action_text(
            action,
            crop_edges=crop_edges,
        )
    action_id = getattr(action, "action_id", action)
    name = getattr(action_id, "name", action_id)
    data = getattr(action, "data", None)
    is_complex = getattr(action, "is_complex", None)
    if callable(is_complex) and is_complex() and not data:
        return f"{name}(x,y normalized_0_1000)"
    if data:
        return f"{name} {json.dumps(data, sort_keys=True, ensure_ascii=False)}"
    return str(name)
