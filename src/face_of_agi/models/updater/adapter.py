"""Adapter shell for updater model backends."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.frames import (
    FRAME_PAYLOAD_TYPE,
    observation_arc_cell_value,
    observation_to_pil_image,
    to_memory_jsonable,
)
from face_of_agi.models.arc_grid_crop import (
    ARC_GRID_SIZE,
    crop_image_arc_grid_edges,
    normalize_arc_grid_crop_edges,
)
from face_of_agi.models.action_glossary import append_action_glossary
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.change.components import (
    current_frame_components_prompt_text,
)
from face_of_agi.models.structured_output import (
    MODEL_FALLBACK_WARNING,
    append_output_schema_to_instructions,
    provider_repair_callback,
    readable_model_error,
    validate_with_repair,
)
from face_of_agi.models.orchestrator_agent.tooling import parse_action
from face_of_agi.models.updater.config import UpdaterConfig
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_MAX_CHARS,
    AGENT_GAME_CONTEXT_KEYS,
    AgentGameContextUpdateResult,
    ContextSegment,
    AgentGameContextUpdateInput,
    PromptImage,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    UpdaterContextTarget,
    UpdaterRole,
    UpdaterTask,
    updater_output_json_schema,
)
DEFAULT_INSTRUCTION_DIR = Path(__file__).parent / "instructions"
LOGGER = logging.getLogger(__name__)


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
        self._arc_grid_crop_edges = normalize_arc_grid_crop_edges(
            self.config.input_image_crop_arc_grid_edges
        )
        self.provider = provider

    def update_agent_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        """Update the agent game-strategy context."""

        return self._update_agent_context(
            update_input=update_input,
        )

    def _update_agent_context(
        self,
        *,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        return self._update_context(
            role="agent",
            segment="game",
            task="agent",
            previous_context=update_input.previous_context,
            current_observation=update_input.current_observation,
            text=_agent_game_prompt_text(
                update_input,
                current_frame_components_text=_current_frame_components_text(
                    update_input.current_observation,
                    crop_edges=self._arc_grid_crop_edges,
                    max_nb_components=getattr(self.config, "max_nb_components", 50),
                ),
            ),
            allowed_actions=update_input.allowed_actions,
            glossary_actions=update_input.glossary_actions,
            images=_current_observation_image(
                update_input.current_observation,
                role="agent game",
                crop_edges=self._arc_grid_crop_edges,
            ),
            actions_window=update_input.actions_window,
            previous_game_context_history=update_input.previous_game_context_history,
        )

    def _update_context(
        self,
        *,
        role: UpdaterRole,
        segment: ContextSegment,
        task: UpdaterTask,
        previous_context: RoleContext,
        text: str,
        current_observation: Observation | None = None,
        images: tuple[PromptImage, ...] = (),
        glossary_actions: Sequence[ActionSpec] | None = None,
        allowed_actions: Sequence[ActionSpec] = (),
        actions_window: int = 1,
        previous_game_context_history: tuple[str, ...] = (),
    ) -> RoleContext | AgentGameContextUpdateResult:
        if task == "agent" and glossary_actions is None:
            raise ValueError(f"{task} updater requires glossary actions")
        target = UpdaterContextTarget(
            role=role,
            segment=segment,
            task=task,
            previous_context=previous_context,
        )
        output_schema = updater_output_json_schema(
            task,
            allowed_actions=allowed_actions,
            actions_window=actions_window,
        )
        instructions_text = load_updater_instructions(
            task=task,
            role=role,
            instruction_dir=self.config.instruction_dir,
        )
        if task == "agent":
            assert glossary_actions is not None
            instructions_text = append_action_glossary(
                instructions_text,
                allowed_actions,
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
        response = None
        try:
            response = self.provider.update_prompt(request)
            validated = validate_with_repair(
                label=f"{self.provider.backend} updater",
                response=response,
                text_of=lambda item: item.text,
                validate=_updated_context_validator(
                    task=task,
                    allowed_actions=allowed_actions,
                    arc_grid_crop_edges=self._arc_grid_crop_edges,
                    actions_window=actions_window,
                ),
                repair=provider_repair_callback(
                    self.provider,
                    "repair_prompt",
                    args=(request,),
                ),
                max_repair_attempts=getattr(self.config, "repair_attempts", 0),
                error_factory=UpdaterOutputError,
            )
        except Exception as exc:
            if task != "agent":
                raise
            LOGGER.warning(
                MODEL_FALLBACK_WARNING + " actions_window=%s",
                "previous-context",
                self.provider.backend,
                self.provider.model,
                getattr(self.config, "repair_attempts", 0),
                readable_model_error(exc),
                actions_window,
            )
            return _fallback_agent_game_update(
                strategy_history=previous_game_context_history,
                previous_context=previous_context,
                current_observation=_fallback_current_observation(
                    current_observation,
                ),
                allowed_actions=allowed_actions,
                actions_window=actions_window,
            )
        if task == "agent":
            updated_text, next_actions = validated.value
            next_actions = _retarget_action6_actions(
                next_actions,
                current_observation=_fallback_current_observation(current_observation),
                crop_edges=self._arc_grid_crop_edges,
            )
            return AgentGameContextUpdateResult(
                context=updated_text,
                next_actions=next_actions,
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
    return root / f"{task}_context_updater_prompt.md"


def _with_updated_segment(
    context: RoleContext,
    segment: ContextSegment,
    updated_text: str,
) -> RoleContext:
    return RoleContext(general=context.general, game=updated_text)


def _fallback_agent_game_update(
    *,
    strategy_history: tuple[str, ...],
    previous_context: RoleContext,
    current_observation: Observation,
    allowed_actions: Sequence[ActionSpec],
    actions_window: int,
) -> AgentGameContextUpdateResult:
    if actions_window < 1:
        raise ValueError("actions_window must be at least 1")
    latest_strategy = strategy_history[-1] if strategy_history else previous_context.game
    context_payload = _previous_strategy_fields(latest_strategy)
    return AgentGameContextUpdateResult(
        context=json.dumps(context_payload, indent=2, ensure_ascii=False),
        next_actions=_fallback_next_actions(
            allowed_actions,
            current_observation=current_observation,
            actions_window=actions_window,
        ),
    )


def _blank_agent_game_context() -> RoleContext:
    return RoleContext(
        game=json.dumps(
            {key: "" for key in AGENT_GAME_CONTEXT_KEYS},
            indent=2,
            ensure_ascii=False,
        ),
    )


def _fallback_current_observation(
    current_observation: Observation | None,
) -> Observation:
    if current_observation is None:
        raise UpdaterOutputError(
            "agent game updater fallback requires the current observation"
        )
    return current_observation


def _retarget_action6_actions(
    actions: Sequence[ActionSpec],
    *,
    current_observation: Observation,
    crop_edges: tuple[int, int, int, int],
) -> tuple[ActionSpec, ...]:
    return tuple(
        _retarget_action6_action(
            action,
            current_observation=current_observation,
            crop_edges=crop_edges,
        )
        for action in actions
    )


def _retarget_action6_action(
    action: ActionSpec,
    *,
    current_observation: Observation,
    crop_edges: tuple[int, int, int, int],
) -> ActionSpec:
    if action.name != "ACTION6" or action.data is None:
        return action
    bbox = action.data.get("bbox")
    target_rgb = action.data.get("target_rgb_color")
    if bbox is None or target_rgb is None:
        return action
    image = crop_image_arc_grid_edges(
        observation_to_pil_image(current_observation),
        crop_edges,
    )
    target_bbox = _action6_bbox_tuple(bbox)
    pixel_x, pixel_y = _closest_target_color_pixel(
        image,
        bbox=target_bbox,
        target_rgb=target_rgb,
    )
    grid_x = _cropped_pixel_to_arc_grid(pixel_x, image.width, "x", crop_edges)
    grid_y = _cropped_pixel_to_arc_grid(pixel_y, image.height, "y", crop_edges)
    return ActionSpec(
        action_id=action.action_id,
        data={
            "x": grid_x,
            "y": grid_y,
        },
        target=action.target,
        target_value=observation_arc_cell_value(
            current_observation,
            x=grid_x,
            y=grid_y,
        ),
        target_bbox=target_bbox,
    )


def _closest_target_color_pixel(
    image: Any,
    *,
    bbox: Sequence[int],
    target_rgb: Sequence[int],
) -> tuple[int, int]:
    rgb_image = image.convert("RGB")
    left, top, right, bottom = _bbox_pixel_window(bbox, rgb_image.size)
    center_x = (left + right - 1) / 2
    center_y = (top + bottom - 1) / 2
    target = tuple(int(channel) for channel in target_rgb)
    best: tuple[int, float, int, int] | None = None
    for y in range(top, bottom):
        for x in range(left, right):
            color = rgb_image.getpixel((x, y))
            color_distance = sum(
                (int(color[index]) - target[index]) ** 2 for index in range(3)
            )
            center_distance = (x - center_x) ** 2 + (y - center_y) ** 2
            candidate = (color_distance, center_distance, x, y)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise UpdaterOutputError("ACTION6 bbox did not contain any target pixels")
    return (best[2], best[3])


def _bbox_pixel_window(
    bbox: Sequence[int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise UpdaterOutputError("ACTION6 retargeting requires a non-empty image")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    left = _clamp(int(x0 * width / 1000), 0, width - 1)
    top = _clamp(int(y0 * height / 1000), 0, height - 1)
    right = _clamp(_ceil_div(x1 * width, 1000), left + 1, width)
    bottom = _clamp(_ceil_div(y1 * height, 1000), top + 1, height)
    return (left, top, right, bottom)


def _action6_bbox_tuple(bbox: Any) -> tuple[int, int, int, int]:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        raise UpdaterOutputError("ACTION6 bbox must be [x0, y0, x1, y1]")
    if len(bbox) != 4:
        raise UpdaterOutputError("ACTION6 bbox must be [x0, y0, x1, y1]")
    return tuple(int(value) for value in bbox)


def _cropped_pixel_to_arc_grid(
    pixel: int,
    image_axis_size: int,
    axis: str,
    crop_edges: tuple[int, int, int, int],
) -> int:
    left, top, right, bottom = crop_edges
    if axis == "x":
        start = left
        visible = ARC_GRID_SIZE - left - right
    elif axis == "y":
        start = top
        visible = ARC_GRID_SIZE - top - bottom
    else:
        raise ValueError(f"unsupported ACTION6 coordinate axis: {axis!r}")
    offset = int((pixel + 0.5) * visible / image_axis_size)
    return _clamp(start + offset, start, start + visible - 1)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _previous_strategy_fields(text: str) -> dict[str, str]:
    if not text.strip():
        return {key: "" for key in AGENT_GAME_CONTEXT_KEYS}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.warning(
            "previous agent game context could not be parsed for updater "
            "fallback; using empty strategy fields",
        )
        return {key: "" for key in AGENT_GAME_CONTEXT_KEYS}
    if not isinstance(loaded, dict):
        return {key: "" for key in AGENT_GAME_CONTEXT_KEYS}
    return {
        key: loaded[key] if isinstance(loaded.get(key), str) else ""
        for key in AGENT_GAME_CONTEXT_KEYS
    }


def _fallback_next_actions(
    allowed_actions: Sequence[ActionSpec],
    *,
    current_observation: Observation,
    actions_window: int,
) -> tuple[ActionSpec, ...]:
    if not allowed_actions:
        raise UpdaterOutputError(
            "agent game updater fallback cannot select an action because allowed "
            "actions is empty"
        )
    action = _fallback_action(
        allowed_actions[0],
        current_observation=current_observation,
    )
    return tuple(action for _ in range(actions_window))


def _fallback_action(
    action: ActionSpec,
    *,
    current_observation: Observation,
) -> ActionSpec:
    if action.name == "ACTION6":
        x, y = _observation_grid_center(current_observation)
        return ActionSpec(
            action_id=action.action_id,
            data={"x": x, "y": y},
            target="",
            target_value=observation_arc_cell_value(
                current_observation,
                x=x,
                y=y,
            ),
        )
    return ActionSpec(action_id=action.action_id)


def _observation_grid_center(observation: Observation) -> tuple[int, int]:
    frame = observation.frame
    if frame is None and observation.frames:
        frame = observation.frames[-1]
    if frame is None:
        return (ARC_GRID_SIZE // 2, ARC_GRID_SIZE // 2)
    size = getattr(frame, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        width, height = size
        return (int(width) // 2, int(height) // 2)
    try:
        import numpy as np

        array = np.asarray(frame)
    except Exception:
        return (ARC_GRID_SIZE // 2, ARC_GRID_SIZE // 2)
    if array.ndim < 2:
        return (ARC_GRID_SIZE // 2, ARC_GRID_SIZE // 2)
    height, width = int(array.shape[0]), int(array.shape[1])
    return (width // 2, height // 2)


def parse_updated_context_output(text: str) -> str:
    """Parse the required JSON updater output contract."""

    return _parse_string_updated_context_output(text)


def parse_agent_game_updated_context_output(
    text: str,
    *,
    allowed_actions: Sequence[ActionSpec],
    arc_grid_crop_edges: object | None = None,
    actions_window: int = 1,
) -> tuple[str, tuple[ActionSpec, ...]]:
    """Parse agent-game context JSON and return context text plus actions."""

    if actions_window < 1:
        raise ValueError("actions_window must be at least 1")
    loaded = _load_updated_context_json(text)
    missing = [key for key in AGENT_GAME_CONTEXT_KEYS if key not in loaded]
    if missing:
        raise UpdaterOutputError(
            "agent game updater response JSON is missing keys: "
            + ", ".join(missing)
        )
    expected_top_level = {*AGENT_GAME_CONTEXT_KEYS, "next_actions"}
    unexpected = sorted(set(loaded) - expected_top_level)
    if unexpected:
        raise UpdaterOutputError(
            "agent game updater response JSON has unexpected keys: "
            + ", ".join(unexpected)
        )
    ordered_context = _validated_agent_game_context(
        loaded,
        allowed_actions=allowed_actions,
    )
    updated_text = json.dumps(ordered_context, indent=2, ensure_ascii=False)
    _validate_agent_game_context_length(updated_text)
    raw_actions = loaded.get("next_actions")
    if not isinstance(raw_actions, list):
        raise UpdaterOutputError(
            "agent game updater response JSON field 'next_actions' must be an array"
        )
    if not raw_actions:
        raise UpdaterOutputError(
            "agent game updater response JSON field 'next_actions' must not be empty"
        )
    if len(raw_actions) != actions_window:
        raise UpdaterOutputError(
            "agent game updater response JSON field 'next_actions' has "
            f"{len(raw_actions)} actions, expected exactly the {actions_window} "
            "action window"
        )
    try:
        next_actions = tuple(
            parse_action(
                item,
                allowed_actions,
                arc_grid_crop_edges=arc_grid_crop_edges,
            )
            for item in raw_actions
        )
    except Exception as exc:
        raise UpdaterOutputError(
            "agent game updater response JSON field 'next_actions' is invalid"
        ) from exc
    return updated_text, next_actions


def _validated_agent_game_context(
    context_payload: dict[str, Any],
    *,
    allowed_actions: Sequence[ActionSpec],
) -> dict[str, Any]:
    invalid = [
        key for key, value in context_payload.items() if not isinstance(value, str)
        and key in AGENT_GAME_CONTEXT_KEYS
    ]
    if invalid:
        raise UpdaterOutputError(
            "agent game updater context values must be strings: "
            + ", ".join(sorted(invalid))
        )
    return {key: context_payload[key] for key in AGENT_GAME_CONTEXT_KEYS}


def _validate_agent_game_context_length(updated_text: str) -> None:
    if len(updated_text) <= AGENT_GAME_CONTEXT_MAX_CHARS:
        return
    raise UpdaterOutputError(
        "agent game updater context is too long: "
        f"{len(updated_text)} characters exceeds the "
        f"{AGENT_GAME_CONTEXT_MAX_CHARS} character cap. Revise the full "
        "context below the cap by removing stale details, duplicate evidence, "
        "and chronological action logs while preserving the summary details "
        "that improve the next decision."
    )


def _updated_context_parser(
    task: UpdaterTask,
    *,
    allowed_actions: Sequence[ActionSpec],
    arc_grid_crop_edges: object | None,
    actions_window: int,
):
    if task == "agent":
        return lambda text: parse_agent_game_updated_context_output(
            text,
            allowed_actions=allowed_actions,
            arc_grid_crop_edges=arc_grid_crop_edges,
            actions_window=actions_window,
        )
    return parse_updated_context_output


def _updated_context_validator(
    *,
    task: UpdaterTask,
    allowed_actions: Sequence[ActionSpec] = (),
    arc_grid_crop_edges: object | None = None,
    actions_window: int = 1,
):
    def validate(text: str) -> str | tuple[str, tuple[ActionSpec, ...]]:
        parser = _updated_context_parser(
            task,
            allowed_actions=allowed_actions,
            arc_grid_crop_edges=arc_grid_crop_edges,
            actions_window=actions_window,
        )
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
            "updater response must be JSON; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise UpdaterOutputError("updater response must be a JSON object")
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

    selected_context = previous_context.game
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
    crop_edges: tuple[int, int, int, int],
) -> tuple[PromptImage, ...]:
    if current_observation is None:
        raise ValueError(f"{role} updater requires a current observation")
    return (
        PromptImage(
            label="current_observation_frame",
            image=crop_image_arc_grid_edges(
                observation_to_pil_image(current_observation),
                crop_edges,
            ),
        ),
    )


def _agent_game_prompt_text(
    update_input: AgentGameContextUpdateInput,
    *,
    current_frame_components_text: str | None = None,
) -> str:
    reset_notice = update_input.reset_notice.strip()
    sections = []
    sections.append(
        "## Previous current_strategy\n\n"
        + _text_or_none(_previous_current_strategy_text(update_input.previous_context))
    )
    sections.append(
        "## Allowed actions\n\n" + _allowed_actions_text(update_input.allowed_actions)
    )
    if current_frame_components_text:
        sections.append(current_frame_components_text)
    sections.append(
        "## Action history\n\n"
        + (
            reset_notice
            if reset_notice and not update_input.action_history
            else grouped_action_history_text(
                update_input.action_history,
                action_text=model_facing_action_text,
                numbered=True,
            )
        )
    )
    sections.append(
        "## Strategy history\n\n"
        + _numbered_text(update_input.previous_game_context_history)
    )
    sections.extend(
        [
            "## Previous strategy summary\n\n"
            + _text_or_none(update_input.previous_strategy_summary),
            "## Previous actions summary\n\n"
            + _text_or_none(update_input.previous_actions_summary),
            "## World model\n\n"
            + _text_or_none(update_input.world_model_context),
        ]
    )
    return "\n\n".join(sections)


def _previous_current_strategy_text(previous_context: RoleContext) -> str:
    text = previous_context.game.strip()
    if not text:
        return ""
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(loaded, dict):
        return text
    current_strategy = loaded.get("current_strategy")
    return current_strategy if isinstance(current_strategy, str) else ""


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


def _numbered_text(history: tuple[str, ...]) -> str:
    if not history:
        return "none"
    lines: list[str] = []
    for index, item in enumerate(history, start=1):
        latest_tag = " [latest]" if index == len(history) else ""
        lines.extend([f"{index}.{latest_tag}", _indent_text(_text_or_none(item))])
    return "\n".join(lines)


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _indent_text(value: str) -> str:
    return "\n".join(f"   {line}" for line in value.splitlines())


def _allowed_actions_text(action_space: tuple[ActionSpec, ...]) -> str:
    if not action_space:
        return "none"
    return "\n".join(f"- {_action_text(action)}" for action in action_space)


def _action_text(action: Any) -> str:
    if isinstance(action, ActionSpec):
        return model_facing_action_text(action)
    action_id = getattr(action, "action_id", action)
    name = getattr(action_id, "name", action_id)
    data = getattr(action, "data", None)
    is_complex = getattr(action, "is_complex", None)
    if callable(is_complex) and is_complex() and not data:
        return f"{name}(x,y normalized_0_1000,target)"
    if data:
        return f"{name} {json.dumps(data, sort_keys=True, ensure_ascii=False)}"
    return str(name)
