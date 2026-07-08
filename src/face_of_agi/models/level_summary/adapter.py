"""Provider-neutral adapter for the per-level solution summarizer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from face_of_agi.models.level_summary.config import LevelSummaryConfig
from face_of_agi.models.level_summary.contracts import (
    LevelSolutionSummary,
    LevelSolutionSummaryInput,
    PromptLevelSummaryProvider,
    PromptLevelSummaryRequest,
    level_solution_summary_json_schema,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)

DEFAULT_INSTRUCTION_PATH = (
    Path(__file__).parent / "instructions" / "level_solution_summary_prompt.md"
)


class LevelSummaryOutputError(RuntimeError):
    """Raised when a level-summary backend returns invalid output."""


class LevelSolutionSummarizerAdapter:
    """Per-level solution summarizer that delegates only the provider call."""

    def __init__(
        self,
        provider: PromptLevelSummaryProvider,
        config: LevelSummaryConfig | None = None,
    ) -> None:
        self.config = config or LevelSummaryConfig()
        self.provider = provider

    def summarize_level_solution(
        self,
        summary_input: LevelSolutionSummaryInput,
    ) -> LevelSolutionSummary:
        """Summarize the method used to solve one completed level."""

        output_schema = level_solution_summary_json_schema()
        metadata = {
            "backend": self.provider.backend,
            "model": self.provider.model,
            "run_id": summary_input.run_id,
            "game_id": summary_input.game_id,
            "completed_level": summary_input.completed_level,
            "strategy_count": len(summary_input.strategy_history),
        }
        instructions = append_output_schema_to_instructions(
            load_level_summary_instructions(self.config.instruction_path),
            output_schema,
            include=self.config.include_output_schema_in_instructions,
        )
        request = PromptLevelSummaryRequest(
            instructions=instructions,
            text=_level_summary_prompt_text(summary_input),
            output_schema=output_schema,
            metadata={
                **metadata,
                "task": "level_solution_summary",
                "schema_name": "level_solution_summary",
            },
        )
        response = self.provider.summarize_level_solution(request)
        validated = validate_with_repair(
            label=f"{self.provider.backend} level summary",
            response=response,
            text_of=lambda item: item.text,
            validate=parse_level_solution_summary_output,
            repair=provider_repair_callback(
                self.provider,
                "repair_level_solution",
                args=(request,),
            ),
            max_repair_attempts=self.config.repair_attempts,
            error_factory=LevelSummaryOutputError,
        )
        summary = validated.value
        summary.metadata = {
            **validated.response.metadata,
            "repair_attempts": validated.repair_attempts,
        }
        return summary


def load_level_summary_instructions(path: str | Path | None = None) -> str:
    """Load the human-editable level-summary instruction prompt."""

    instruction_path = Path(path) if path is not None else DEFAULT_INSTRUCTION_PATH
    return instruction_path.read_text(encoding="utf-8").strip()


def parse_level_solution_summary_output(text: str) -> LevelSolutionSummary:
    """Parse level-summary JSON output."""

    try:
        loaded = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        preview = text.strip().replace("\n", "\\n")[:300]
        raise LevelSummaryOutputError(
            "level summary response must be JSON with 'solution_method'; "
            f"raw response preview: {preview!r}"
        ) from exc
    if not isinstance(loaded, dict):
        raise LevelSummaryOutputError("level summary response must be a JSON object")
    solution_method = loaded.get("solution_method")
    if not isinstance(solution_method, str):
        raise LevelSummaryOutputError(
            "level summary response JSON is missing string field 'solution_method'"
        )
    unexpected = sorted(set(loaded) - {"solution_method"})
    if unexpected:
        raise LevelSummaryOutputError(
            "level summary response JSON has unexpected keys: "
            + ", ".join(unexpected)
        )
    return LevelSolutionSummary(solution_method=solution_method)


def _level_summary_prompt_text(summary_input: LevelSolutionSummaryInput) -> str:
    return "\n\n".join(
        [
            f"## Completed level\n\n{summary_input.completed_level}",
            "## Strategy history\n\n"
            + _numbered_strategy_history_text(summary_input.strategy_history),
        ]
    )


def _numbered_strategy_history_text(history: tuple[str, ...]) -> str:
    if not history:
        return "none"
    lines = [
        f"{index}. {item}"
        for index, item in enumerate(history, start=1)
    ]
    return "\n\n".join(lines)


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


__all__ = [
    "LevelSolutionSummarizerAdapter",
    "LevelSummaryOutputError",
    "load_level_summary_instructions",
    "parse_level_solution_summary_output",
]
