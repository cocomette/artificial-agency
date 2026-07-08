"""Provider-neutral adapter for the agent world-model role."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

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
from face_of_agi.models.image_inputs import resize_image
from face_of_agi.models.world.contracts import (
    AgentContextWorldSummary,
    AgentWorldModelInput,
    PromptWorldImage,
    PromptWorldRequest,
    agent_world_model_json_schema,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)
from face_of_agi.models.world.config import WorldModelConfig
from face_of_agi.models.world.contracts import PromptWorldProvider

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "world_model_prompt.md"
LOGGER = logging.getLogger(__name__)


class WorldModelOutputError(RuntimeError):
    """Raised when a world-model backend returns invalid output."""


class AgentWorldModelAdapter:
    """World-model role that delegates only the provider call."""

    def __init__(
        self,
        provider: PromptWorldProvider,
        config: WorldModelConfig | None = None,
    ) -> None:
        self.config = config or WorldModelConfig()
        self._arc_grid_crop_edges = normalize_arc_grid_crop_edges(
            self.config.input_image_crop_arc_grid_edges
        )
        self.provider = provider

    def summarize_agent_world_model(
        self,
        world_input: AgentWorldModelInput,
    ) -> AgentContextWorldSummary:
        """Summarize world mechanics/action effects for one transition."""

        images = _current_observation_image(
            world_input.current_observation,
            crop_edges=self._arc_grid_crop_edges,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
        )
        metadata = {
            "backend": self.provider.backend,
            "model": self.provider.model,
            "game_id": world_input.game_id,
        }
        output_schema = agent_world_model_json_schema(
            world_input.allowed_actions
        )
        instructions = append_output_schema_to_instructions(
            append_action_glossary(
                load_world_model_instructions(self.config.instruction_path),
                world_input.allowed_actions,
                mode="world_model",
            ),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptWorldRequest(
            instructions=instructions,
            text=_world_model_prompt_text(
                world_input,
                crop_edges=self._arc_grid_crop_edges,
            ),
            output_schema=output_schema,
            images=images,
            metadata={
                **metadata,
                "task": "agent_world_model",
                "schema_name": "agent_world_model",
            },
        )
        response = self.provider.summarize_world_model(request)
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} world model",
                response=response,
                text_of=lambda item: item.text,
                validate=lambda text: parse_agent_world_model_output(
                    text,
                    allowed_actions=world_input.allowed_actions,
                ),
                repair=provider_repair_callback(
                    self.provider,
                    "repair_world_model",
                    args=(request,),
                ),
                max_repair_attempts=self.config.repair_attempts,
                error_factory=WorldModelOutputError,
            )
        except WorldModelOutputError as exc:
            LOGGER.error(
                "world model structured output repair exhausted; using previous "
                "world-model fallback backend=%s model=%s game_id=%s "
                "repair_attempts=%s",
                self.provider.backend,
                self.provider.model,
                world_input.game_id,
                self.config.repair_attempts,
                exc_info=True,
            )
            return _fallback_world_summary(
                world_input,
                metadata={
                    **response.metadata,
                    "repair_attempts": self.config.repair_attempts,
                    "fallback": "repair_exhausted",
                    "fallback_reason": str(exc),
                },
            )
        world = validated.value
        world.metadata = {
            **validated.response.metadata,
            "repair_attempts": validated.repair_attempts,
        }
        return world


def load_world_model_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable world-model instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def parse_agent_world_model_output(
    text: str,
    *,
    allowed_actions: tuple[Any, ...] = (),
) -> AgentContextWorldSummary:
    """Parse the world-model JSON contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise WorldModelOutputError(
            "world model response must be JSON with 'world_description', "
            "'special_events', and 'action_effects'; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise WorldModelOutputError("world model response must be a JSON object")
    world_description = loaded.get("world_description")
    if not isinstance(world_description, str):
        raise WorldModelOutputError(
            "world model response JSON is missing string field 'world_description'"
        )
    special_events = loaded.get("special_events")
    if not isinstance(special_events, str):
        raise WorldModelOutputError(
            "world model response JSON is missing string field 'special_events'"
        )
    action_effects = _parse_action_effects(
        loaded,
        allowed_actions=allowed_actions,
        owner="world model",
    )
    return AgentContextWorldSummary(
        world_description=world_description,
        action_effects=action_effects,
        special_events=special_events,
    )


def _fallback_world_summary(
    world_input: AgentWorldModelInput,
    *,
    metadata: dict[str, Any],
) -> AgentContextWorldSummary:
    previous = _previous_world_model_summary(
        world_input.previous_world_model,
        allowed_actions=world_input.allowed_actions,
    )
    if previous is not None:
        previous.metadata = metadata
        return previous
    return AgentContextWorldSummary(
        world_description="",
        action_effects={
            _action_name(action): "" for action in world_input.allowed_actions
        },
        special_events="",
        metadata=metadata,
    )


def _previous_world_model_summary(
    text: str,
    *,
    allowed_actions: tuple[Any, ...],
) -> AgentContextWorldSummary | None:
    if not text.strip():
        return None
    try:
        previous = parse_agent_world_model_output(text)
    except WorldModelOutputError:
        LOGGER.warning(
            "previous world model could not be parsed for fallback; using empty "
            "world-model fallback",
            exc_info=True,
        )
        return None
    if not allowed_actions:
        return previous
    previous.action_effects = {
        _action_name(action): previous.action_effects.get(_action_name(action), "")
        for action in allowed_actions
    }
    return previous


def _world_model_prompt_text(
    world_input: AgentWorldModelInput,
    *,
    crop_edges: tuple[int, int, int, int],
) -> str:
    return "\n\n".join(
        [
            "## Previous world model\n\n"
            + _text_or_none(world_input.previous_world_model),
            "## Allowed actions\n\n"
            + _allowed_actions_text(world_input.allowed_actions),
            "## Action history\n\n"
            + _numbered_action_history_text(
                world_input.action_history,
                crop_edges=crop_edges,
            ),
        ]
    )


def _current_observation_image(
    current_observation: Any,
    *,
    crop_edges: tuple[int, int, int, int],
    size: str | tuple[int, int] | None,
    resample: str,
) -> tuple[PromptWorldImage, ...]:
    if current_observation is None:
        raise ValueError("world model requires a current observation")
    return (
        PromptWorldImage(
            label="current_observation_frame",
            image=_prepared_observation_image(
                current_observation,
                crop_edges=crop_edges,
                size=size,
                resample=resample,
            ),
        ),
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


def _parse_action_effects(
    loaded: dict[str, Any],
    *,
    allowed_actions: tuple[Any, ...],
    owner: str,
) -> dict[str, str]:
    action_effects = loaded.get("action_effects")
    if not isinstance(action_effects, dict):
        raise WorldModelOutputError(
            f"{owner} response JSON is missing object field 'action_effects'"
        )
    invalid_action_effects = [
        key for key, value in action_effects.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_action_effects:
        raise WorldModelOutputError(
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
            raise WorldModelOutputError(
                f"{owner} action_effects is missing keys: "
                + ", ".join(missing_actions)
            )
        unexpected_actions = sorted(set(action_effects) - set(expected_actions))
        if unexpected_actions:
            raise WorldModelOutputError(
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


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)


__all__ = [
    "AgentWorldModelAdapter",
    "WorldModelOutputError",
    "load_world_model_instructions",
    "parse_agent_world_model_output",
]
