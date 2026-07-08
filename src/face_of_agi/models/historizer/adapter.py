"""Provider-neutral adapter for the agent context historizer role."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from face_of_agi.models.historizer.config import HistorizerConfig
from face_of_agi.models.historizer.contracts import (
    AgentContextHistoryDecision,
    AgentContextHistoryInput,
    AgentContextHistorySummary,
    AgentContextWorldSummary,
    PromptHistorizerProvider,
    PromptHistorizerProviderResponse,
    PromptHistorizerRequest,
    agent_context_history_json_schema,
)
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"
LOGGER = logging.getLogger(__name__)


class HistorizerOutputError(RuntimeError):
    """Raised when a historizer backend returns invalid output."""


class AgentContextHistorizerAdapter:
    """Provider-neutral historizer that delegates only the model call."""

    def __init__(
        self,
        provider: PromptHistorizerProvider,
        config: HistorizerConfig | None = None,
    ) -> None:
        self.config = config or HistorizerConfig()
        self.provider = provider

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        """Summarize agent context history and select the next updater mode."""

        world = history_input.current_world_model
        if world is None:
            raise ValueError("historizer requires current_world_model")
        metadata = {
            "backend": self.provider.backend,
            "model": self.provider.model,
            "game_id": history_input.game_id,
            "context_window": history_input.context_window,
            "strategy_history_count": len(history_input.strategy_history),
        }
        decision = self._summarize_history(
            history_input=history_input,
            world_model=world,
            metadata=metadata,
        )
        return AgentContextHistorySummary(
            world_description=world.world_description,
            probing_evolution=decision.probing_evolution,
            policy_evolution=decision.policy_evolution,
            strategy_summary=decision.strategy_summary,
            action_effects=world.action_effects,
            special_events=world.special_events,
            updater_mode=decision.updater_mode,
            metadata={
                **metadata,
                "available": True,
                "historizer": decision.metadata,
                "world_model": world.metadata,
            },
        )

    def _summarize_history(
        self,
        *,
        history_input: AgentContextHistoryInput,
        world_model: AgentContextWorldSummary,
        metadata: dict[str, Any],
    ) -> AgentContextHistoryDecision:
        output_schema = agent_context_history_json_schema()
        instructions = append_output_schema_to_instructions(
            load_historizer_instructions(self.config.instruction_path),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptHistorizerRequest(
            instructions=instructions,
            text=_history_prompt_text(
                history_input,
                world_model,
            ),
            output_schema=output_schema,
            metadata={
                **metadata,
                "task": "agent_context_history",
                "schema_name": "agent_context_history",
            },
        )
        response = self.provider.summarize_context_history(request)
        try:
            validated = validate_with_repair(
                label=f"{self.provider.backend} historizer",
                response=response,
                text_of=lambda item: item.text,
                validate=parse_agent_context_history_output,
                repair=provider_repair_callback(
                    self.provider,
                    "repair_context_history",
                    args=(request,),
                ),
                max_repair_attempts=self.config.repair_attempts,
                error_factory=HistorizerOutputError,
            )
        except HistorizerOutputError as exc:
            LOGGER.error(
                "historizer structured output repair exhausted; using empty "
                "policy-mode fallback backend=%s model=%s game_id=%s "
                "repair_attempts=%s",
                self.provider.backend,
                self.provider.model,
                history_input.game_id,
                self.config.repair_attempts,
                exc_info=True,
            )
            return AgentContextHistoryDecision(
                probing_evolution="",
                policy_evolution="",
                strategy_summary="",
                updater_mode="policy",
                metadata={
                    **response.metadata,
                    "repair_attempts": self.config.repair_attempts,
                    "fallback": "repair_exhausted",
                    "fallback_reason": str(exc),
                },
            )
        decision = validated.value
        decision.metadata = {
            **validated.response.metadata,
            "repair_attempts": validated.repair_attempts,
        }
        return decision


def load_historizer_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable historizer instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def parse_agent_context_history_output(text: str) -> AgentContextHistoryDecision:
    """Parse the historizer JSON contract."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise HistorizerOutputError(
            "historizer response must be JSON with 'probing_evolution', "
            "'policy_evolution', 'strategy_summary', and 'updater_mode'; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise HistorizerOutputError("historizer response must be a JSON object")
    probing_evolution = loaded.get("probing_evolution")
    if not isinstance(probing_evolution, str):
        raise HistorizerOutputError(
            "historizer response JSON is missing string field "
            "'probing_evolution'"
        )
    policy_evolution = loaded.get("policy_evolution")
    if not isinstance(policy_evolution, str):
        raise HistorizerOutputError(
            "historizer response JSON is missing string field "
            "'policy_evolution'"
        )
    strategy_summary = loaded.get("strategy_summary")
    if not isinstance(strategy_summary, str):
        raise HistorizerOutputError(
            "historizer response JSON is missing string field "
            "'strategy_summary'"
        )
    updater_mode = loaded.get("updater_mode")
    if updater_mode not in {"probing", "policy"}:
        raise HistorizerOutputError(
            "historizer response JSON field 'updater_mode' must be "
            "'probing' or 'policy'"
        )
    return AgentContextHistoryDecision(
        probing_evolution=probing_evolution,
        policy_evolution=policy_evolution,
        strategy_summary=strategy_summary,
        updater_mode=updater_mode,
    )


def _history_prompt_text(
    history_input: AgentContextHistoryInput,
    world_model: AgentContextWorldSummary,
) -> str:
    return "\n\n".join(
        [
            "## World model\n\n"
            + _world_model_context_text(world_model),
            "## Allowed actions\n\n"
            + _allowed_actions_text(history_input.allowed_actions),
            "## Action history\n\n"
            + _numbered_action_history_text(
                history_input.action_history,
            ),
            "## Probing/policy history\n\n"
            + _numbered_strategy_history_text(history_input.strategy_history),
        ]
    )


def _world_model_context_text(summary: AgentContextWorldSummary) -> str:
    action_lines = [
        f"- {key}: {_text_or_none(value)}"
        for key, value in summary.action_effects.items()
    ]
    return "\n\n".join(
        [
            "Latest world description:\n" + _text_or_none(summary.world_description),
            "Special events:\n" + _text_or_none(summary.special_events),
            "Action effects:\n" + (
                "\n".join(action_lines) if action_lines else "not available"
            ),
        ]
    )


def _allowed_actions_text(action_space: tuple[Any, ...]) -> str:
    if not action_space:
        return "none"
    return "\n".join(
        f"- {getattr(action, 'name', str(action))}" for action in action_space
    )


def _numbered_action_history_text(
    history: tuple[Any, ...],
) -> str:
    if not history:
        return "none"
    return grouped_action_history_text(
        history,
        action_text=model_facing_action_text,
        numbered=True,
    )


def _numbered_strategy_history_text(history: tuple[str, ...]) -> str:
    if not history:
        return "not available"
    lines = []
    for index, item in enumerate(history, start=1):
        lines.append(f"{index}. {_text_or_none(item)}")
    return "\n\n".join(lines)


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return stripped
    return match.group(1).strip()
