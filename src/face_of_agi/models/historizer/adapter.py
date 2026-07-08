"""Provider-neutral adapter for the historizer role."""

from __future__ import annotations

import json
import re
from pathlib import Path

from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.historizer.config import HistorizerConfig
from face_of_agi.models.historizer.contracts import (
    HistorizerInput,
    HistorizerSummary,
    PromptHistorizerProvider,
    PromptHistorizerRequest,
    historizer_summary_json_schema,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "historizer_prompt.md"
)


class HistorizerOutputError(RuntimeError):
    """Raised when a historizer backend returns invalid output."""


class HistorizerAdapter:
    """Historizer that delegates only the model call."""

    def __init__(
        self,
        provider: PromptHistorizerProvider,
        config: HistorizerConfig | None = None,
    ) -> None:
        self.config = config or HistorizerConfig()
        self.provider = provider

    def summarize_history(
        self,
        historizer_input: HistorizerInput,
    ) -> HistorizerSummary:
        """Summarize current-level action and strategy history."""

        output_schema = historizer_summary_json_schema()
        metadata = {
            "backend": self.provider.backend,
            "model": self.provider.model,
            "run_id": historizer_input.run_id,
            "game_id": historizer_input.game_id,
            "action_history_count": len(historizer_input.action_history),
            "strategy_history_count": len(historizer_input.strategy_history),
        }
        instructions = append_output_schema_to_instructions(
            load_historizer_instructions(self.config.instruction_path),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptHistorizerRequest(
            instructions=instructions,
            text=_historizer_prompt_text(historizer_input),
            output_schema=output_schema,
            metadata={
                **metadata,
                "task": "history_summary",
                "schema_name": "historizer_summary",
            },
        )
        response = self.provider.summarize_history(request)
        validated = validate_with_repair(
            label=f"{self.provider.backend} historizer",
            response=response,
            text_of=lambda item: item.text,
            validate=parse_historizer_summary_output,
            repair=provider_repair_callback(
                self.provider,
                "repair_history",
                args=(request,),
            ),
            max_repair_attempts=self.config.repair_attempts,
            error_factory=HistorizerOutputError,
        )
        summary = validated.value
        summary.metadata = {
            **validated.response.metadata,
            "repair_attempts": validated.repair_attempts,
        }
        return summary


def load_historizer_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable historizer instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def parse_historizer_summary_output(text: str) -> HistorizerSummary:
    """Parse historizer JSON output."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise HistorizerOutputError(
            "historizer response must be JSON with action_history_summary and "
            f"strategy_history_summary; raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise HistorizerOutputError("historizer response must be a JSON object")
    action_history_summary = loaded.get("action_history_summary")
    strategy_history_summary = loaded.get("strategy_history_summary")
    if not isinstance(action_history_summary, str):
        raise HistorizerOutputError(
            "historizer response JSON is missing string field "
            "'action_history_summary'"
        )
    if not isinstance(strategy_history_summary, str):
        raise HistorizerOutputError(
            "historizer response JSON is missing string field "
            "'strategy_history_summary'"
        )
    unexpected = sorted(
        set(loaded) - {"action_history_summary", "strategy_history_summary"}
    )
    if unexpected:
        raise HistorizerOutputError(
            "historizer response JSON has unexpected keys: " + ", ".join(unexpected)
        )
    return HistorizerSummary(
        action_history_summary=action_history_summary,
        strategy_history_summary=strategy_history_summary,
    )


def _historizer_prompt_text(historizer_input: HistorizerInput) -> str:
    return "\n\n".join(
        [
            "## World model\n\n"
            + _text_or_none(historizer_input.world_model_context),
            "## Previous history summaries\n\n"
            + _text_or_none(historizer_input.previous_history_summary),
            "## Action history\n\n"
            + grouped_action_history_text(
                historizer_input.action_history,
                action_text=model_facing_action_text,
                numbered=True,
            ),
            "## Strategy history\n\n"
            + _numbered_text(
                historizer_input.strategy_history,
                tag_latest=True,
            ),
        ]
    )


def _numbered_text(history: tuple[str, ...], *, tag_latest: bool = False) -> str:
    if not history:
        return "none"
    lines: list[str] = []
    for index, item in enumerate(history, start=1):
        latest_tag = " [latest]" if tag_latest and index == len(history) else ""
        lines.extend([f"{index}.{latest_tag}", _indent_text(_text_or_none(item))])
    return "\n".join(lines)


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"


def _indent_text(value: str, *, spaces: int = 3) -> str:
    indent = " " * spaces
    return "\n".join(f"{indent}{line}" for line in value.splitlines())


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


__all__ = [
    "HistorizerAdapter",
    "HistorizerOutputError",
    "load_historizer_instructions",
    "parse_historizer_summary_output",
]
