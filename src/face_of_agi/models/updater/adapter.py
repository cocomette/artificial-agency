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
    ActionHistoryItem,
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.frames import (
    FRAME_PAYLOAD_TYPE,
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
from face_of_agi.models.historizer import (
    AgentContextHistorySummary,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)
from face_of_agi.models.orchestrator_agent.tooling import parse_action
from face_of_agi.models.updater.config import UpdaterConfig
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_MAX_CHARS,
    AgentGameContextUpdateResult,
    AgentUpdaterMode,
    ContextSegment,
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    PromptImage,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    UpdaterContextTarget,
    UpdaterRole,
    UpdaterTask,
    agent_game_output_keys,
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

    def update_agent_probing_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        """Update the agent probing-strategy context."""

        return self._update_agent_context(
            mode="probing",
            task="agent_probing",
            update_input=update_input,
        )

    def update_agent_policy_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        """Update the agent policy-strategy context."""

        return self._update_agent_context(
            mode="policy",
            task="agent_policy",
            update_input=update_input,
        )

    def _update_agent_context(
        self,
        *,
        mode: AgentUpdaterMode,
        task: UpdaterTask,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        return self._update_context(
            role="agent",
            segment="game",
            task=task,
            previous_context=update_input.previous_context,
            current_observation=update_input.current_observation,
            text=_agent_game_prompt_text(
                update_input,
                crop_edges=self._arc_grid_crop_edges,
            ),
            allowed_actions=update_input.allowed_actions,
            glossary_actions=update_input.glossary_actions,
            images=_current_observation_image(
                update_input.current_observation,
                role="agent game",
                crop_edges=self._arc_grid_crop_edges,
            ),
            agent_mode=mode,
            agent_context_history=update_input.context_history,
            actions_window=update_input.actions_window,
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
        current_observation: Observation | None = None,
        images: tuple[PromptImage, ...] = (),
        glossary_actions: Sequence[ActionSpec] | None = None,
        allowed_actions: Sequence[ActionSpec] = (),
        agent_mode: AgentUpdaterMode | None = None,
        agent_context_history: AgentContextHistorySummary | None = None,
        actions_window: int = 1,
    ) -> RoleContext | AgentGameContextUpdateResult:
        if task in {"agent_probing", "agent_policy"} and glossary_actions is None:
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
        if task in {"agent_probing", "agent_policy"}:
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
        response = self.provider.update_prompt(request)
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} updater",
                response=response,
                text_of=lambda item: item.text,
                validate=_updated_context_validator(
                    task=task,
                    mode=agent_mode,
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
        except UpdaterOutputError as exc:
            if task not in {"agent_probing", "agent_policy"}:
                raise
            assert agent_mode is not None
            LOGGER.error(
                "agent %s updater structured output repair exhausted; using "
                "previous-context fallback backend=%s model=%s repair_attempts=%s "
                "actions_window=%s",
                agent_mode,
                self.provider.backend,
                self.provider.model,
                getattr(self.config, "repair_attempts", 0),
                actions_window,
                exc_info=True,
            )
            return _fallback_agent_game_update(
                mode=agent_mode,
                previous_context=previous_context,
                current_observation=_fallback_current_observation(
                    current_observation,
                ),
                allowed_actions=allowed_actions,
                actions_window=actions_window,
            )
        if task in {"agent_probing", "agent_policy"}:
            assert agent_mode is not None
            updated_text, next_actions = validated.value
            next_actions = _retarget_action6_actions(
                next_actions,
                current_observation=_fallback_current_observation(current_observation),
                crop_edges=self._arc_grid_crop_edges,
            )
            return AgentGameContextUpdateResult(
                context=updated_text,
                next_actions=next_actions,
                updater_mode=agent_mode,
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


def _fallback_agent_game_update(
    *,
    mode: AgentUpdaterMode,
    previous_context: RoleContext,
    current_observation: Observation,
    allowed_actions: Sequence[ActionSpec],
    actions_window: int,
) -> AgentGameContextUpdateResult:
    if actions_window < 1:
        raise ValueError("actions_window must be at least 1")
    context_payload = {
        agent_game_output_keys(mode)[0]: _previous_strategy_text(
            previous_context.game,
            mode=mode,
        )
    }
    return AgentGameContextUpdateResult(
        context=json.dumps(context_payload, indent=2, ensure_ascii=False),
        next_actions=_fallback_next_actions(
            allowed_actions,
            current_observation=current_observation,
            actions_window=actions_window,
        ),
        updater_mode=mode,
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
    pixel_x, pixel_y = _closest_target_color_pixel(
        image,
        bbox=bbox,
        target_rgb=target_rgb,
    )
    return ActionSpec(
        action_id=action.action_id,
        data={
            "x": _cropped_pixel_to_arc_grid(pixel_x, image.width, "x", crop_edges),
            "y": _cropped_pixel_to_arc_grid(pixel_y, image.height, "y", crop_edges),
        },
        target=action.target,
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


def _previous_strategy_text(
    text: str,
    *,
    mode: AgentUpdaterMode,
) -> str:
    if not text.strip():
        return ""
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.warning(
            "previous agent game context could not be parsed for updater "
            "fallback; using empty %s strategy",
            mode,
            exc_info=True,
        )
        return ""
    if not isinstance(loaded, dict):
        return ""
    value = loaded.get(agent_game_output_keys(mode)[0])
    return value if isinstance(value, str) else ""


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
    mode: AgentUpdaterMode,
    allowed_actions: Sequence[ActionSpec],
    arc_grid_crop_edges: object | None = None,
    actions_window: int = 1,
) -> tuple[str, tuple[ActionSpec, ...]]:
    """Parse agent-game context JSON and return context text plus actions."""

    if actions_window < 1:
        raise ValueError("actions_window must be at least 1")
    loaded = _load_updated_context_json(text)
    output_keys = agent_game_output_keys(mode)
    missing = [key for key in output_keys if key not in loaded]
    if missing:
        raise UpdaterOutputError(
            "agent game updater response JSON is missing keys: "
            + ", ".join(missing)
        )
    expected_top_level = {*output_keys, "next_actions"}
    unexpected = sorted(set(loaded) - expected_top_level)
    if unexpected:
        raise UpdaterOutputError(
            "agent game updater response JSON has unexpected keys: "
            + ", ".join(unexpected)
        )
    ordered_context = _validated_agent_game_context(
        loaded,
        mode=mode,
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
    mode: AgentUpdaterMode,
    allowed_actions: Sequence[ActionSpec],
) -> dict[str, Any]:
    invalid = [
        key for key, value in context_payload.items() if not isinstance(value, str)
        and key in agent_game_output_keys(mode)
    ]
    if invalid:
        raise UpdaterOutputError(
            "agent game updater context values must be strings: "
            + ", ".join(sorted(invalid))
        )
    if mode == "probing":
        return {"probing_strategy": context_payload["probing_strategy"]}
    return {"policy_strategy": context_payload["policy_strategy"]}


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
    mode: AgentUpdaterMode | None,
    allowed_actions: Sequence[ActionSpec],
    arc_grid_crop_edges: object | None,
    actions_window: int,
):
    if task == "agent_probing":
        return lambda text: parse_agent_game_updated_context_output(
            text,
            mode="probing",
            allowed_actions=allowed_actions,
            arc_grid_crop_edges=arc_grid_crop_edges,
            actions_window=actions_window,
        )
    if task == "agent_policy":
        return lambda text: parse_agent_game_updated_context_output(
            text,
            mode="policy",
            allowed_actions=allowed_actions,
            arc_grid_crop_edges=arc_grid_crop_edges,
            actions_window=actions_window,
        )
    return parse_updated_context_output


def _updated_context_validator(
    *,
    task: UpdaterTask,
    mode: AgentUpdaterMode | None = None,
    allowed_actions: Sequence[ActionSpec] = (),
    arc_grid_crop_edges: object | None = None,
    actions_window: int = 1,
):
    def validate(text: str) -> str | tuple[str, tuple[ActionSpec, ...]]:
        parser = _updated_context_parser(
            task,
            mode=mode,
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
    crop_edges: tuple[int, int, int, int],
) -> str:
    sections = [
        "## Previous game context\n\n"
        + _text_or_none(update_input.previous_context.game)
    ]
    if update_input.previous_level_solution_method.strip():
        sections.append(
            "## Previous level solution method\n\n"
            "This is the summary of the strategy that solved the previous "
            "level. Get inspired from it to understand what to do to solve "
            "this level:\n\n"
            + update_input.previous_level_solution_method.strip()
        )
    sections.extend(
        [
            "## Allowed actions\n\n"
            + _allowed_actions_text(update_input.allowed_actions),
            "## Action history\n\n"
            + _numbered_action_history_text(
                update_input.action_history,
                crop_edges=crop_edges,
            ),
            "## World model\n\n"
            + _world_model_context_text(update_input.context_history),
            "## Probing evolution\n\n"
            + _probing_evolution_text(update_input.context_history),
            "## Policy evolution\n\n"
            + _policy_evolution_text(update_input.context_history),
            "## Strategy summary\n\n"
            + _strategy_summary_text(update_input.context_history),
        ]
    )
    return "\n\n".join(sections)


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _allowed_actions_text(action_space: tuple[ActionSpec, ...]) -> str:
    if not action_space:
        return "none"
    return "\n".join(f"- {_action_text(action)}" for action in action_space)


def _numbered_action_history_text(
    history: tuple[ActionHistoryItem, ...],
    *,
    crop_edges: tuple[int, int, int, int],
) -> str:
    if not history:
        return "none"
    return grouped_action_history_text(
        history,
        action_text=lambda action: model_facing_action_text(
            action,
            crop_edges=crop_edges,
        ),
        numbered=True,
    )


def _world_model_context_text(summary: AgentContextHistorySummary) -> str:
    if not summary.is_available():
        return "not available"
    action_lines = [
        f"- {key}: {_text_or_none(value)}"
        for key, value in summary.action_effects.items()
    ]
    return "\n".join(
        [
            "world_description: " + _text_or_none(summary.world_description),
            "special_events: " + _text_or_none(summary.special_events),
            "action_effects:\n" + (
                "\n".join(action_lines) if action_lines else "not available"
            ),
        ]
    )


def _probing_evolution_text(summary: AgentContextHistorySummary) -> str:
    if not summary.is_available():
        return "not available"
    return _text_or_none(summary.probing_evolution)


def _policy_evolution_text(summary: AgentContextHistorySummary) -> str:
    if not summary.is_available():
        return "not available"
    return _text_or_none(summary.policy_evolution)


def _strategy_summary_text(summary: AgentContextHistorySummary) -> str:
    if not summary.is_available():
        return "not available"
    return _text_or_none(summary.strategy_summary)


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
