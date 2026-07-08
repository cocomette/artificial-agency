"""Provider-neutral adapter for the compacter role."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from face_of_agi.contracts import Observation
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.arc_grid_crop import (
    crop_image_arc_grid_edges,
    normalize_arc_grid_crop_edges,
)
from face_of_agi.models.change.components import (
    current_frame_components_prompt_text,
)
from face_of_agi.models.image_inputs import resize_image
from face_of_agi.models.compacter.contracts import (
    AgentCompacterSummary,
    AgentCompacterInput,
    PromptCompacterImage,
    PromptCompacterRequest,
    agent_compacter_json_schema,
)
from face_of_agi.models.structured_output import (
    MODEL_FALLBACK_WARNING,
    append_output_schema_to_instructions,
    provider_repair_callback,
    readable_model_error,
    validate_with_repair,
)
from face_of_agi.models.compacter.config import CompacterConfig
from face_of_agi.models.compacter.contracts import PromptCompacterProvider

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "compacter_prompt.md"
LOGGER = logging.getLogger(__name__)


class CompacterOutputError(RuntimeError):
    """Raised when a compacter backend returns invalid output."""


class AgentCompacterAdapter:
    """Compacter role that delegates only the provider call."""

    def __init__(
        self,
        provider: PromptCompacterProvider,
        config: CompacterConfig | None = None,
    ) -> None:
        self.config = config or CompacterConfig()
        self._arc_grid_crop_edges = normalize_arc_grid_crop_edges(
            self.config.input_image_crop_arc_grid_edges
        )
        self.provider = provider

    def compact_agent_context(
        self,
        compacter_input: AgentCompacterInput,
    ) -> AgentCompacterSummary:
        """Compact game context for one frame turn."""

        images = _current_observation_image(
            compacter_input.current_observation,
            crop_edges=self._arc_grid_crop_edges,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
        )
        metadata = {
            "backend": self.provider.backend,
            "model": self.provider.model,
            "game_id": compacter_input.game_id,
        }
        output_schema = agent_compacter_json_schema(
            compacter_input.allowed_actions
        )
        instructions = append_output_schema_to_instructions(
            append_action_glossary(
                load_compacter_instructions(self.config.instruction_path),
                compacter_input.allowed_actions,
                mode="compacter",
            ),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptCompacterRequest(
            instructions=instructions,
            text=_compacter_prompt_text(
                compacter_input,
                crop_edges=self._arc_grid_crop_edges,
                current_frame_components_text=_current_frame_components_text(
                    compacter_input.current_observation,
                    crop_edges=self._arc_grid_crop_edges,
                    max_nb_components=self.config.max_nb_components,
                ),
            ),
            output_schema=output_schema,
            images=images,
            metadata={
                **metadata,
                "task": "agent_compacter",
                "schema_name": "agent_compacter",
            },
        )
        response = None
        try:
            response = self.provider.compact_context(request)
            validated = validate_with_repair(
                label=f"{self.provider.backend} compacter",
                response=response,
                text_of=lambda item: item.text,
                validate=lambda text: parse_agent_compacter_output(
                    text,
                    allowed_actions=compacter_input.allowed_actions,
                ),
                repair=provider_repair_callback(
                    self.provider,
                    "repair_compacter_context",
                    args=(request,),
                ),
                max_repair_attempts=self.config.repair_attempts,
                error_factory=CompacterOutputError,
            )
        except Exception as exc:
            LOGGER.warning(
                MODEL_FALLBACK_WARNING + " game_id=%s",
                "previous compacter",
                self.provider.backend,
                self.provider.model,
                self.config.repair_attempts,
                readable_model_error(exc),
                compacter_input.game_id,
            )
            return _fallback_compacter_summary(
                compacter_input,
                metadata={
                    **(response.metadata if response is not None else metadata),
                    "repair_attempts": self.config.repair_attempts,
                    "fallback": "model_call_or_repair_failed",
                    "fallback_reason": readable_model_error(exc),
                },
            )
        world = validated.value
        world.metadata = {
            **validated.response.metadata,
            "repair_attempts": validated.repair_attempts,
        }
        return world


def load_compacter_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable compacter instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def parse_agent_compacter_output(
    text: str,
    *,
    allowed_actions: tuple[Any, ...] = (),
) -> AgentCompacterSummary:
    """Parse the compacter JSON contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise CompacterOutputError(
            "compacter response must be JSON with 'world_description', "
            "'special_events', 'action_effects', 'previous_actions_summary', "
            "and 'previous_strategy_summary'; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise CompacterOutputError("compacter response must be a JSON object")
    world_description = loaded.get("world_description")
    if not isinstance(world_description, str):
        raise CompacterOutputError(
            "compacter response JSON is missing string field 'world_description'"
        )
    special_events = loaded.get("special_events")
    if not isinstance(special_events, str):
        raise CompacterOutputError(
            "compacter response JSON is missing string field 'special_events'"
        )
    previous_actions_summary = loaded.get("previous_actions_summary")
    if not isinstance(previous_actions_summary, str):
        raise CompacterOutputError(
            "compacter response JSON is missing string field "
            "'previous_actions_summary'"
        )
    previous_strategy_summary = loaded.get("previous_strategy_summary")
    if not isinstance(previous_strategy_summary, str):
        raise CompacterOutputError(
            "compacter response JSON is missing string field "
            "'previous_strategy_summary'"
        )
    action_effects = _parse_action_effects(
        loaded,
        allowed_actions=allowed_actions,
        owner="compacter",
    )
    return AgentCompacterSummary(
        world_description=world_description,
        action_effects=action_effects,
        previous_actions_summary=previous_actions_summary,
        previous_strategy_summary=previous_strategy_summary,
        special_events=special_events,
    )


def _fallback_compacter_summary(
    compacter_input: AgentCompacterInput,
    *,
    metadata: dict[str, Any],
) -> AgentCompacterSummary:
    previous = _previous_compacter_summary(
        compacter_input.previous_compacter_context,
        allowed_actions=compacter_input.allowed_actions,
    )
    if previous is not None:
        previous.metadata = metadata
        return previous
    return AgentCompacterSummary(
        world_description="",
        action_effects={
            _action_name(action): "" for action in compacter_input.allowed_actions
        },
        previous_actions_summary="",
        previous_strategy_summary="",
        special_events="",
        metadata=metadata,
    )


def _previous_compacter_summary(
    text: str,
    *,
    allowed_actions: tuple[Any, ...],
) -> AgentCompacterSummary | None:
    if not text.strip():
        return None
    try:
        previous = parse_agent_compacter_output(text)
    except CompacterOutputError:
        LOGGER.warning(
            "previous compacter context could not be parsed for fallback; using "
            "empty compacter fallback",
        )
        return None
    if not allowed_actions:
        return previous
    previous.action_effects = {
        _action_name(action): previous.action_effects.get(_action_name(action), "")
        for action in allowed_actions
    }
    return previous


def _compacter_prompt_text(
    compacter_input: AgentCompacterInput,
    *,
    crop_edges: tuple[int, int, int, int],
    current_frame_components_text: str | None = None,
) -> str:
    sections = [
        "## Previous compacter context\n\n"
        + _text_or_none(compacter_input.previous_compacter_context),
        "## Allowed actions\n\n" + _allowed_actions_text(compacter_input.allowed_actions),
    ]
    if current_frame_components_text:
        sections.append(current_frame_components_text)
    sections.append(
        "## Action history\n\n"
        + _numbered_action_history_text(
            compacter_input.action_history,
            crop_edges=crop_edges,
        )
    )
    sections.append(
        "## Strategy history\n\n"
        + _numbered_text(compacter_input.strategy_history)
    )
    return "\n\n".join(sections)


def _current_observation_image(
    current_observation: Any,
    *,
    crop_edges: tuple[int, int, int, int],
    size: str | tuple[int, int] | None,
    resample: str,
) -> tuple[PromptCompacterImage, ...]:
    if current_observation is None:
        raise ValueError("compacter requires a current observation")
    return (
        PromptCompacterImage(
            label="current_observation_frame",
            image=_prepared_observation_image(
                current_observation,
                crop_edges=crop_edges,
                size=size,
                resample=resample,
            ),
        ),
    )


def _current_frame_components_text(
    current_observation: Observation,
    *,
    crop_edges: tuple[int, int, int, int],
    max_nb_components: int,
) -> str:
    return current_frame_components_prompt_text(
        current_observation,
        crop_edges=crop_edges,
        max_nb_components=max_nb_components,
    )


def _prepared_observation_image(
    observation: Any,
    *,
    crop_edges: tuple[int, int, int, int],
    size: str | tuple[int, int] | None,
    resample: str,
) -> Any:
    image = crop_image_arc_grid_edges(
        observation_to_pil_image(observation),
        crop_edges,
    )
    return resize_image(image, size=size, resample=resample)


def _numbered_action_history_text(
    history: tuple[Any, ...],
    *,
    crop_edges: tuple[int, int, int, int],
) -> str:
    if not history:
        return "not available"
    return grouped_action_history_text(
        history,
        action_text=lambda action: model_facing_action_text(
            action,
            crop_edges=crop_edges,
        ),
        numbered=True,
    )


def _numbered_text(history: tuple[str, ...]) -> str:
    if not history:
        return "not available"
    lines: list[str] = []
    for index, item in enumerate(history, start=1):
        latest_tag = " [latest]" if index == len(history) else ""
        lines.extend([f"{index}.{latest_tag}", _indent_text(_text_or_none(item))])
    return "\n".join(lines)


def _parse_action_effects(
    loaded: dict[str, Any],
    *,
    allowed_actions: tuple[Any, ...],
    owner: str,
) -> dict[str, str]:
    action_effects = loaded.get("action_effects")
    if not isinstance(action_effects, dict):
        raise CompacterOutputError(
            f"{owner} response JSON is missing object field 'action_effects'"
        )
    invalid_action_effects = [
        key for key, value in action_effects.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_action_effects:
        raise CompacterOutputError(
            f"{owner} action_effects must be an object of string fields"
        )
    expected_actions = tuple(
        dict.fromkeys(_action_name(action) for action in allowed_actions)
    )
    if expected_actions:
        missing_actions = [
            action for action in expected_actions if action not in action_effects
        ]
        if missing_actions:
            raise CompacterOutputError(
                f"{owner} action_effects is missing keys: "
                + ", ".join(missing_actions)
            )
        unexpected_actions = sorted(set(action_effects) - set(expected_actions))
        if unexpected_actions:
            raise CompacterOutputError(
                f"{owner} action_effects has unexpected keys: "
                + ", ".join(unexpected_actions)
            )
        return {
            action: action_effects[action] for action in expected_actions
        }
    return {key: action_effects[key] for key in action_effects}


def _allowed_actions_text(action_space: tuple[Any, ...]) -> str:
    if not action_space:
        return "none"
    return "\n".join(
        f"- {getattr(action, 'name', str(action))}" for action in action_space
    )


def _action_name(action: Any) -> str:
    return str(getattr(action, "name", action))


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "not available"
    stripped = str(value).strip()
    return stripped if stripped else "not available"


def _indent_text(value: str) -> str:
    return "\n".join(f"   {line}" for line in value.splitlines())


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)


__all__ = [
    "AgentCompacterAdapter",
    "CompacterOutputError",
    "load_compacter_instructions",
    "parse_agent_compacter_output",
]
