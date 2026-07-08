"""Tests for the active compacter prompt adapter."""

from __future__ import annotations

import json

from face_of_agi.contracts import ActionSpec, Observation
from face_of_agi.models.compacter import (
    AgentCompacterAdapter,
    AgentCompacterInput,
    PromptCompacterProviderResponse,
    PromptCompacterRequest,
    CompacterConfig,
)


class FakeCompacterProvider:
    backend = "fake"
    model = "fake-model"

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[PromptCompacterRequest] = []

    def compact_context(
        self,
        request: PromptCompacterRequest,
    ) -> PromptCompacterProviderResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return PromptCompacterProviderResponse(text=response)

    def repair_compacter_context(
        self,
        request: PromptCompacterRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptCompacterProviderResponse:
        self.requests.append(request)
        return PromptCompacterProviderResponse(
            text=self.responses.pop(0),
            metadata={
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            },
        )


def test_compacter_prompt_includes_current_frame_components_by_default() -> None:
    provider = FakeCompacterProvider(json.dumps(_compacter_response()))
    adapter = AgentCompacterAdapter(
        provider=provider,
        config=CompacterConfig(input_image_crop_arc_grid_edges=None),
    )
    frame = _arc_grid()
    for y in range(20, 22):
        for x in range(10, 12):
            frame[y][x] = 4

    adapter.compact_agent_context(
        AgentCompacterInput(
            game_id="game-1",
            current_observation=Observation(id="current", step=1, frame=frame),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )

    request = provider.requests[0]
    text = request.text
    assert "## Current frame components" in text
    assert "frame 0:" not in text
    assert "- color=charcoal nb=1 box=[(156,312,188,344)]" in text
    assert "symbol=" not in text
    assert "rgb=" not in text
    assert "## ARC rendered color legend" not in text
    assert "## ARC rendered color legend" not in request.instructions
    assert text.index("## Allowed actions") < text.index(
        "## Current frame components"
    )
    assert text.index("## Current frame components") < text.index(
        "## Action history"
    )


def test_compacter_falls_back_after_provider_context_length(caplog) -> None:
    provider = FakeCompacterProvider(RuntimeError("maximum context length reached"))
    adapter = AgentCompacterAdapter(
        provider=provider,
        config=CompacterConfig(repair_attempts=1),
    )

    with caplog.at_level("WARNING"):
        result = adapter.compact_agent_context(
            AgentCompacterInput(
                game_id="game-1",
                current_observation=Observation(
                    id="current",
                    step=1,
                    frame=_arc_grid(),
                ),
                previous_compacter_context=json.dumps(_compacter_response()),
                allowed_actions=(ActionSpec(action_id="ACTION1"),),
            )
        )

    assert result.world_description == "world"
    assert result.metadata["fallback"] == "model_call_or_repair_failed"
    assert (
        "max repair attempts / model context length reached, continuing with "
        "previous compacter fallback"
    ) in caplog.text
    assert "RuntimeError: maximum context length reached" in caplog.text
    assert "Traceback" not in caplog.text


def _compacter_response() -> dict[str, object]:
    return {
        "world_description": "world",
        "special_events": "",
        "action_effects": {"ACTION1": "moves"},
        "previous_actions_summary": "recent actions",
        "previous_strategy_summary": "recent strategy",
    }


def _arc_grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]
