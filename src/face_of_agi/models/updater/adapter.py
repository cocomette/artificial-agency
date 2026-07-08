"""Adapter shell for updater model backends."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from face_of_agi.contracts import ActionHistoryEntry, Observation, RoleContext
from face_of_agi.frames import (
    FRAME_PAYLOAD_TYPE,
    observation_to_pil_image,
    to_memory_jsonable,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)
from face_of_agi.models.updater.config import UpdaterConfig
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_KEYS,
    ContextSegment,
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    PromptImage,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    UpdaterContextTarget,
    UpdaterRole,
    UpdaterTask,
    WORLD_GAME_ACTION_KEYS,
    WORLD_GAME_CONTEXT_KEYS,
    WorldGameContextUpdateInput,
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

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        """Update the world game context."""

        action = update_input.submitted_action or update_input.synthetic_none_action
        if action is None:
            raise ValueError("world game updater requires an action")
        predicted_description = _required_prediction_description(
            update_input.post_decision_predictions.world_prediction,
            role="world",
        )
        return self._update_context(
            role="world",
            segment="game",
            task="world_game",
            previous_context=update_input.previous_context,
            text=_world_game_prompt_text(
                context=update_input.previous_context.game,
                action=action,
                prediction_description=predicted_description,
            ),
            images=_current_observation_image(
                update_input.current_observation,
                role="world game",
                frame_scale=self.config.frame_scale,
            ),
        )

    def update_goal_game_context(
        self,
        update_input: GoalGameContextUpdateInput,
    ) -> RoleContext:
        """Update the goal game context."""

        predicted_description = _required_prediction_description(
            update_input.post_decision_predictions.goal_prediction,
            role="goal",
        )
        return self._update_context(
            role="goal",
            segment="game",
            task="goal_game",
            previous_context=update_input.previous_context,
            text=_goal_game_prompt_text(
                update_input.previous_context.game,
                prediction_description=predicted_description,
            ),
            images=_current_observation_image(
                update_input.current_observation,
                role="goal game",
                frame_scale=self.config.frame_scale,
            ),
        )

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        """Update the agent game context."""

        images = (
            PromptImage(
                label="previous_observation_frame",
                image=observation_to_pil_image(
                    update_input.previous_observation,
                    frame_scale=self.config.frame_scale,
                ),
            ),
            PromptImage(
                label="current_observation_frame",
                image=observation_to_pil_image(
                    update_input.current_observation,
                    frame_scale=self.config.frame_scale,
                ),
            ),
        )
        return self._update_context(
            role="agent",
            segment="game",
            task="agent_game",
            previous_context=update_input.previous_context,
            text=_agent_game_prompt_text(update_input),
            images=images,
        )

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        """Update one role's general knowledge context."""

        if update_input.role in ("world", "goal"):
            return self._update_context(
                role=update_input.role,
                segment="general",
                task="general",
                previous_context=update_input.previous_context,
                text=_general_prompt_text(
                    role=update_input.role,
                    context=update_input.previous_context,
                ),
            )
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
    ) -> RoleContext:
        target = UpdaterContextTarget(
            role=role,
            segment=segment,
            task=task,
            previous_context=previous_context,
        )
        output_schema = updater_output_json_schema(task)
        instructions = append_output_schema_to_instructions(
            load_updater_instructions(
                task=task,
                role=role,
                instruction_dir=self.config.instruction_dir,
            ),
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
            validate=_updated_context_parser(task),
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


def parse_world_game_updated_context_output(text: str) -> str:
    """Parse world-game action context JSON and return it as context text."""

    loaded = _load_updated_context_json(text)
    updated_context = loaded.get("updated_context")
    if not isinstance(updated_context, dict):
        raise UpdaterOutputError(
            "world game updater response JSON is missing object field "
            "'updated_context'"
        )

    missing = [key for key in WORLD_GAME_CONTEXT_KEYS if key not in updated_context]
    if missing:
        raise UpdaterOutputError(
            "world game updater updated_context is missing keys: "
            + ", ".join(missing)
        )
    unexpected = sorted(set(updated_context) - set(WORLD_GAME_CONTEXT_KEYS))
    if unexpected:
        raise UpdaterOutputError(
            "world game updater updated_context has unexpected keys: "
            + ", ".join(unexpected)
        )
    invalid = [
        key for key, value in updated_context.items() if not isinstance(value, str)
    ]
    if invalid:
        raise UpdaterOutputError(
            "world game updater updated_context values must be strings: "
            + ", ".join(sorted(invalid))
        )
    ordered_context = {
        key: updated_context[key] for key in WORLD_GAME_CONTEXT_KEYS
    }
    return json.dumps(ordered_context, indent=2, ensure_ascii=False)


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
    return json.dumps(ordered_context, indent=2, ensure_ascii=False)


def _updated_context_parser(task: UpdaterTask):
    if task == "world_game":
        return parse_world_game_updated_context_output
    if task == "agent_game":
        return parse_agent_game_updated_context_output
    return parse_updated_context_output


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


def _required_prediction_description(value: Any, *, role: str) -> Any:
    if value is None or getattr(value, "predicted_description", None) is None:
        raise ValueError(f"{role} game updater requires a predicted description")
    return value.predicted_description


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


def _world_game_prompt_text(
    *,
    context: str,
    action: Any,
    prediction_description: Any,
) -> str:
    return "\n\n".join(
        [
            "Previous world context:\n" + context,
            "Action:\n" + _action_text(action),
            "Committed world prediction description JSON:\n"
            + json.dumps(
                to_memory_jsonable(prediction_description),
                indent=2,
                sort_keys=True,
            ),
        ]
    )


def _goal_game_prompt_text(
    context: str,
    *,
    prediction_description: Any,
) -> str:
    return "\n\n".join(
        [
            "Previous goal context:\n" + context,
            "Committed goal prediction description JSON:\n"
            + json.dumps(
                to_memory_jsonable(prediction_description),
                indent=2,
                sort_keys=True,
            ),
        ]
    )


def _agent_game_prompt_text(update_input: AgentGameContextUpdateInput) -> str:
    return "\n\n".join(
        [
            "## Previous agent game context\n\n"
            + _text_or_none(update_input.previous_context.game),
            "## Current-turn world game context\n\n"
            + _text_or_none(update_input.current_turn_world_game_context),
            "## Previous-turn world game context\n\n"
            + _text_or_none(update_input.previous_turn_world_game_context),
            "## Action history\n\n" + _action_history_text(update_input.action_history),
            "## Progress feedback\n\n"
            + _progress_feedback_text(update_input.turn_metrics),
        ]
    )


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _action_history_text(history: tuple[ActionHistoryEntry, ...]) -> str:
    if not history:
        return "none"
    return "\n".join(f"- {_action_history_entry_text(entry)}" for entry in history)


def _action_history_entry_text(entry: ActionHistoryEntry) -> str:
    text = _action_text(entry.action)
    if not entry.controllable:
        text += " [animation]"
    return text


def _progress_feedback_text(feedback: Any) -> str:
    return "\n".join(
        [
            f"- time_cost: {_metric_text(feedback.time_cost)}",
            f"- cumulative_score: {_metric_text(feedback.cumulative_score)}",
            "- agent_context_word_count: "
            + _metric_text(feedback.agent_context_word_count),
        ]
    )


def _metric_text(value: float | int | None) -> str:
    if value is None:
        return "none"
    return str(value)


def _general_prompt_text(*, role: UpdaterRole, context: RoleContext) -> str:
    if role == "world":
        return "\n\n".join(
            [
                "Game world model text:\n" + context.game,
                "General world model text:\n" + context.general,
            ]
        )
    if role == "goal":
        return "\n\n".join(
            [
                "Game goal model text:\n" + context.game,
                "General goal model text:\n" + context.general,
            ]
        )
    raise ValueError(f"unsupported minimal general updater role: {role}")


def _action_text(action: Any) -> str:
    action_id = getattr(action, "action_id", action)
    name = getattr(action_id, "name", action_id)
    data = getattr(action, "data", None)
    if data:
        return f"{name} {json.dumps(data, sort_keys=True, ensure_ascii=False)}"
    return str(name)
