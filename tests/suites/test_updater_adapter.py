"""Tests for the updater model shell."""

import json

from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    TurnMetrics,
    RoleContext,
    ToolResult,
)
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentProgressFeedback,
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    PromptUpdaterAdapter,
    UpdaterOutputError,
    UpdaterConfig,
    WorldGameContextUpdateInput,
)
from face_of_agi.models.updater.config import (
    OllamaUpdaterConfig,
    OpenAIUpdaterConfig,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_KEYS,
    WORLD_GAME_ACTION_KEYS,
    WORLD_GAME_CONTEXT_KEYS,
    agent_game_updated_context_json_schema,
    updated_context_json_schema,
    world_game_updated_context_json_schema,
)
from face_of_agi.models.updater.providers.ollama import OllamaUpdaterAdapter
from face_of_agi.models.updater.providers.openai import OpenAIUpdaterAdapter


def _observation(id_: str, step: int, color: tuple[int, int, int]) -> Observation:
    return Observation(
        id=id_,
        step=step,
        frame=Image.new("RGB", (4, 4), color=color),
    )


def _description_prediction(description: str = "predicted change") -> list[dict]:
    return [{"bbox_2d": [0.0, 0.0, 4.0, 4.0], "description": description}]


def _world_action_context(prefix: str) -> dict[str, str]:
    return {key: f"{prefix} {key}" for key in WORLD_GAME_CONTEXT_KEYS}


def _world_action_context_text(prefix: str) -> str:
    return json.dumps(
        _world_action_context(prefix),
        indent=2,
        ensure_ascii=False,
    )


def _agent_context(prefix: str) -> dict[str, str]:
    return {key: f"{prefix} {key}" for key in AGENT_GAME_CONTEXT_KEYS}


def _agent_context_text(prefix: str) -> str:
    return json.dumps(
        _agent_context(prefix),
        indent=2,
        ensure_ascii=False,
    )


def _agent_game_update_input(
    *,
    previous_context: RoleContext | None = None,
    action: ActionSpec | None = None,
    turn_metrics: AgentProgressFeedback | None = None,
    current_turn_world_game_context: str = "L^S current turn",
    previous_turn_world_game_context: str | None = None,
) -> AgentGameContextUpdateInput:
    final_action = action or ActionSpec(action_id="ACTION1")
    return AgentGameContextUpdateInput(
        previous_context=previous_context or RoleContext(general="K^X", game="L^X"),
        previous_observation=_observation("obs-0", 0, (255, 255, 255)),
        current_observation=_observation("obs-1", 1, (0, 0, 0)),
        current_turn_world_game_context=current_turn_world_game_context,
        previous_turn_world_game_context=previous_turn_world_game_context,
        action_history=(
            ActionHistoryEntry(
                action=final_action,
                controllable=True,
            ),
        ),
        turn_metrics=turn_metrics or AgentProgressFeedback(),
    )


class FakePromptUpdaterProvider:
    backend = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[PromptUpdateRequest] = []

    def update_prompt(self, request: PromptUpdateRequest) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        if request.target.task == "world_game":
            text = json.dumps(
                {"updated_context": _world_action_context(request.target.role)}
            )
        elif request.target.task == "agent_game":
            text = json.dumps(
                {"updated_context": _agent_context(request.target.role)}
            )
        else:
            text = json.dumps(
                {"updated_context": f"{request.target.role}-{request.target.segment}"}
            )
        return PromptUpdateProviderResponse(
            target=request.target,
            text=text,
        )


def test_prompt_updater_selects_role_segment_instruction_files(tmp_path) -> None:
    for task in ("world_game", "goal_game", "agent_game"):
        (tmp_path / f"{task}_context_updater_prompt.md").write_text(
            f"{task} instructions",
            encoding="utf-8",
        )
    for role in ("world", "goal", "agent"):
        (tmp_path / f"{role}_general_context_updater_prompt.md").write_text(
            f"{role}_general instructions",
            encoding="utf-8",
        )
    provider = FakePromptUpdaterProvider()
    updater = PromptUpdaterAdapter(
        provider=provider,
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    context = RoleContext(general="K", game="L")
    predicted_description = _description_prediction()
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )
    action = ActionSpec(action_id="ACTION1")

    results = [
        updater.update_world_game_context(
            WorldGameContextUpdateInput(
                previous_context=context,
                current_observation=actual_observation,
                post_decision_predictions=PostDecisionPredictions(
                    world_prediction=ToolResult(
                        id="world-out",
                        tool="world",
                        predicted_description=predicted_description,
                        source_observation_ref=observation_ref,
                        action=action,
                    )
                ),
                submitted_action=action,
            )
        ),
        updater.update_goal_game_context(
            GoalGameContextUpdateInput(
                previous_context=context,
                current_observation=actual_observation,
                post_decision_predictions=PostDecisionPredictions(
                    goal_prediction=ToolResult(
                        id="goal-out",
                        tool="goal",
                        predicted_description=predicted_description,
                        source_observation_ref=observation_ref,
                    )
                ),
            )
        ),
        updater.update_agent_game_context(
            _agent_game_update_input(
                previous_context=context,
            )
        ),
    ]
    results.extend(
        updater.update_general_knowledge(
            GeneralKnowledgeUpdateInput(
                role=role,
                previous_context=context,
                run_id="run-1",
                game_id="game-1",
            )
        )
        for role in ("world", "goal", "agent")
    )

    assert results == [
        RoleContext(general="K", game=_world_action_context_text("world")),
        RoleContext(general="K", game="goal-game"),
        RoleContext(general="K", game=_agent_context_text("agent")),
        RoleContext(general="world-general", game="L"),
        RoleContext(general="goal-general", game="L"),
        RoleContext(general="agent-general", game="L"),
    ]
    assert [request.instructions for request in provider.requests] == [
        "world_game instructions",
        "goal_game instructions",
        "agent_game instructions",
        "world_general instructions",
        "goal_general instructions",
        "agent_general instructions",
    ]
    assert provider.requests[0].output_schema == world_game_updated_context_json_schema()
    assert provider.requests[1].output_schema == updated_context_json_schema()
    assert (
        provider.requests[2].output_schema
        == agent_game_updated_context_json_schema()
    )
    assert [image.label for image in provider.requests[0].images] == [
        "current_observation_frame",
    ]
    assert [image.label for image in provider.requests[1].images] == [
        "current_observation_frame",
    ]
    assert [image.label for image in provider.requests[2].images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]
    assert provider.requests[3].images == ()
    assert provider.requests[4].images == ()


def test_prompt_updater_can_include_output_schema_in_instructions(tmp_path) -> None:
    (tmp_path / "world_game_context_updater_prompt.md").write_text(
        "world game instructions",
        encoding="utf-8",
    )
    provider = FakePromptUpdaterProvider()
    updater = PromptUpdaterAdapter(
        provider=provider,
        config=UpdaterConfig(
            instruction_dir=str(tmp_path),
            include_output_schema_in_instructions=True,
        ),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    action = ActionSpec(action_id="ACTION1")

    updater.update_world_game_context(
        WorldGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="L"),
            current_observation=Observation(
                id="obs-1",
                step=1,
                frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
            ),
            post_decision_predictions=PostDecisionPredictions(
                world_prediction=ToolResult(
                    id="world-out",
                    tool="world",
                    predicted_description=_description_prediction(),
                    source_observation_ref=observation_ref,
                    action=action,
                )
            ),
            submitted_action=action,
        )
    )

    instructions = provider.requests[0].instructions
    assert instructions.startswith("world game instructions\n\n")
    assert "Output JSON must match this schema exactly." in instructions
    assert '"updated_context"' in instructions
    assert '"world_understanding"' in instructions


def test_prompt_updater_sends_world_goal_game_updates_for_synthetic_none(
    tmp_path,
) -> None:
    (tmp_path / "world_game_context_updater_prompt.md").write_text(
        "world game instructions",
        encoding="utf-8",
    )
    (tmp_path / "goal_game_context_updater_prompt.md").write_text(
        "goal game instructions",
        encoding="utf-8",
    )
    provider = FakePromptUpdaterProvider()
    updater = PromptUpdaterAdapter(
        provider=provider,
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    predicted_description = _description_prediction()
    actual_observation = Observation(
        id="obs-1",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    updater.update_world_game_context(
        WorldGameContextUpdateInput(
            previous_context=RoleContext(general="K^S", game="L^S"),
            current_observation=actual_observation,
            post_decision_predictions=PostDecisionPredictions(
                world_prediction=ToolResult(
                    id="world-out",
                    tool="world",
                    predicted_description=predicted_description,
                    source_observation_ref=observation_ref,
                    action=ActionSpec.none(),
                )
            ),
            synthetic_none_action=ActionSpec.none(),
        )
    )
    updater.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=RoleContext(general="K^G", game="L^G"),
            current_observation=actual_observation,
            post_decision_predictions=PostDecisionPredictions(
                goal_prediction=ToolResult(
                    id="goal-out",
                    tool="goal",
                    predicted_description=predicted_description,
                    source_observation_ref=observation_ref,
                )
            ),
            synthetic_none_action=ActionSpec.none(),
        )
    )

    assert [image.label for image in provider.requests[0].images] == [
        "current_observation_frame",
    ]
    assert [image.label for image in provider.requests[1].images] == [
        "current_observation_frame",
    ]


def test_world_goal_game_updaters_require_current_observation(tmp_path) -> None:
    (tmp_path / "world_game_context_updater_prompt.md").write_text(
        "world game instructions",
        encoding="utf-8",
    )
    (tmp_path / "goal_game_context_updater_prompt.md").write_text(
        "goal game instructions",
        encoding="utf-8",
    )
    updater = PromptUpdaterAdapter(
        provider=FakePromptUpdaterProvider(),
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    action = ActionSpec(action_id="ACTION1")
    actual_observation = _observation("obs-1", 1, (0, 0, 0))

    with pytest.raises(ValueError, match="current observation"):
        updater.update_world_game_context(
            WorldGameContextUpdateInput(
                previous_context=RoleContext(general="K^S", game="L^S"),
                current_observation=None,
                post_decision_predictions=PostDecisionPredictions(
                    world_prediction=ToolResult(
                        id="world-out",
                        tool="world",
                        predicted_description=[],
                        source_observation_ref=observation_ref,
                        action=action,
                    )
                ),
                submitted_action=action,
            )
        )

    with pytest.raises(ValueError, match="current observation"):
        updater.update_goal_game_context(
            GoalGameContextUpdateInput(
                previous_context=RoleContext(general="K^G", game="L^G"),
                current_observation=None,
                post_decision_predictions=PostDecisionPredictions(
                    goal_prediction=ToolResult(
                        id="goal-out",
                        tool="goal",
                        predicted_description=[],
                        source_observation_ref=observation_ref,
                    )
                ),
            )
        )


def test_world_game_updater_requires_all_action_keys(tmp_path) -> None:
    (tmp_path / "world_game_context_updater_prompt.md").write_text(
        "world game instructions",
        encoding="utf-8",
    )

    class MissingActionProvider(FakePromptUpdaterProvider):
        def update_prompt(
            self,
            request: PromptUpdateRequest,
        ) -> PromptUpdateProviderResponse:
            self.requests.append(request)
            incomplete = dict(_world_action_context("updated"))
            incomplete.pop("world_understanding")
            return PromptUpdateProviderResponse(
                target=request.target,
                text=json.dumps({"updated_context": incomplete}),
            )

    updater = PromptUpdaterAdapter(
        provider=MissingActionProvider(),
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    action = ActionSpec(action_id="ACTION1")

    with pytest.raises(
        UpdaterOutputError,
        match="missing keys: world_understanding",
    ):
        updater.update_world_game_context(
            WorldGameContextUpdateInput(
                previous_context=RoleContext(general="K^S", game="L^S"),
                current_observation=_observation("obs-1", 1, (0, 0, 0)),
                post_decision_predictions=PostDecisionPredictions(
                    world_prediction=ToolResult(
                        id="world-out",
                        tool="world",
                        predicted_description=[],
                        source_observation_ref=observation_ref,
                        action=action,
                    )
                ),
                submitted_action=action,
            )
        )


class FakeOpenAIResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return {
            "id": "resp-1",
            "model": request["model"],
            "status": "completed",
            "output_text": self.output_text,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class FakeOpenAIClient:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeOpenAIResponses(output_text)


class FakeOllamaClient:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = [content] if isinstance(content, str) else list(content)
        self.calls: list[dict[str, object]] = []

    def chat(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return {
            "message": {"content": content},
            "prompt_eval_count": 1,
            "eval_count": 1,
        }


def test_openai_updater_updates_world_game_context_from_structured_json(tmp_path) -> None:
    (tmp_path / "world_game_context_updater_prompt.md").write_text(
        "world game instructions",
        encoding="utf-8",
    )
    client = FakeOpenAIClient(
        json.dumps({"updated_context": _world_action_context("updated")})
    )
    updater = OpenAIUpdaterAdapter(
        OpenAIUpdaterConfig(
            backend="openai",
            model="gpt-5-nano",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    action = ActionSpec(action_id="ACTION1")
    predicted_description = _description_prediction()
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    result = updater.update_world_game_context(
        WorldGameContextUpdateInput(
            previous_context=RoleContext(general="K^S", game="L^S"),
            current_observation=actual_observation,
            post_decision_predictions=PostDecisionPredictions(
                world_prediction=ToolResult(
                    id="world-out",
                    tool="world",
                    predicted_description=predicted_description,
                    source_observation_ref=observation_ref,
                    action=action,
                ),
            ),
            submitted_action=action,
        )
    )

    assert result == RoleContext(
        general="K^S",
        game=_world_action_context_text("updated"),
    )
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5-nano"
    assert request["instructions"] == "world game instructions"
    assert request["text"]["format"] == {
        "type": "json_schema",
        "name": "updater_context_update",
        "strict": True,
        "schema": world_game_updated_context_json_schema(),
    }
    content = request["input"][0]["content"]
    assert [item["type"] for item in content] == [
        "input_text",
        "input_image",
    ]
    records = drain_model_input_debug_records(updater)
    assert records[0]["call_slot"] == "updater_world"
    assert records[0]["provider"] == "openai"
    assert records[0]["phase"] == "update_prompt"
    assert len(records[0]["request"]["input"][0]["content"]) == 2
    assert records[0]["usage"] == {"input_tokens": 1, "output_tokens": 1}
    assert json.loads(records[0]["metadata"]["response_output_text"]) == {
        "updated_context": _world_action_context("updated")
    }
    assert json.loads(records[0]["metadata"]["response_payload"]["output_text"]) == {
        "updated_context": _world_action_context("updated")
    }


def test_openai_updater_updates_agent_game_context_from_structured_json(
    tmp_path,
) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    client = FakeOpenAIClient(
        json.dumps({"updated_context": _agent_context("updated")})
    )
    updater = OpenAIUpdaterAdapter(
        OpenAIUpdaterConfig(
            backend="openai",
            model="gpt-5-nano",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )
    action = ActionSpec(action_id="ACTION1")
    previous_observation = _observation("obs-0", 0, (255, 255, 255))
    current_observation = _observation("obs-1", 1, (0, 0, 0))

    result = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K^X", game="L^X"),
            previous_observation=previous_observation,
            current_observation=current_observation,
            current_turn_world_game_context="L^S",
            previous_turn_world_game_context=None,
            action_history=(
                ActionHistoryEntry(
                    action=action,
                    controllable=True,
                ),
            ),
        )
    )

    assert result == RoleContext(general="K^X", game=_agent_context_text("updated"))
    request = client.responses.calls[0]
    assert request["instructions"] == "agent game instructions"
    assert (
        request["text"]["format"]["schema"]
        == agent_game_updated_context_json_schema()
    )
    text = request["input"][0]["content"][0]["text"]
    assert text.startswith("## Previous agent game context\n\nL^X")
    assert "## Action history\n\n- ACTION1" in text
    assert "attached_images" not in text
    assert "trace" not in text
    assert [item["type"] for item in request["input"][0]["content"]] == [
        "input_text",
        "input_image",
        "input_image",
    ]


def test_agent_game_updater_serializes_structured_context_in_contract_order(
    tmp_path,
) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    updated_context = dict(reversed(tuple(_agent_context("updated").items())))
    client = FakeOllamaClient(json.dumps({"updated_context": updated_context}))
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e2b",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )
    previous = RoleContext(
        general="K^X",
        game=(
            "Initial action notes:\n"
            "ACTION1: up arrow\n"
            "ACTION2: down arrow"
        ),
    )

    result = updater.update_agent_game_context(
        _agent_game_update_input(
            previous_context=previous,
        )
    )

    assert result == RoleContext(general="K^X", game=_agent_context_text("updated"))
    assert len(client.calls[0]["messages"][1]["images"]) == 2


def test_agent_game_updater_requires_all_structured_context_keys(tmp_path) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    updated_context = _agent_context("updated")
    updated_context.pop("goals")
    client = FakeOllamaClient(json.dumps({"updated_context": updated_context}))
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e2b",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )

    with pytest.raises(UpdaterOutputError, match="missing keys: goals"):
        updater.update_agent_game_context(_agent_game_update_input())


def test_agent_game_updater_rejects_unexpected_structured_context_keys(
    tmp_path,
) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    updated_context = _agent_context("updated")
    updated_context["debug"] = "extra debug note"
    client = FakeOllamaClient(json.dumps({"updated_context": updated_context}))
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e2b",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )

    with pytest.raises(UpdaterOutputError, match="unexpected keys: debug"):
        updater.update_agent_game_context(_agent_game_update_input())


def test_agent_game_updater_requires_string_structured_context_values(
    tmp_path,
) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    updated_context: dict[str, object] = _agent_context("updated")
    updated_context["extras"] = {"note": "nested"}
    client = FakeOllamaClient(json.dumps({"updated_context": updated_context}))
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e2b",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )

    with pytest.raises(UpdaterOutputError, match="values must be strings: extras"):
        updater.update_agent_game_context(_agent_game_update_input())


def test_ollama_updater_updates_goal_game_context_from_structured_json(tmp_path) -> None:
    (tmp_path / "goal_game_context_updater_prompt.md").write_text(
        "goal game instructions",
        encoding="utf-8",
    )
    client = FakeOllamaClient('{"updated_context": "updated L^G"}')
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e4b",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    predicted_description = _description_prediction()
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    result = updater.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=RoleContext(general="K^G", game="L^G"),
            current_observation=actual_observation,
            post_decision_predictions=PostDecisionPredictions(
                goal_prediction=ToolResult(
                    id="goal-out",
                    tool="goal",
                    predicted_description=predicted_description,
                    source_observation_ref=observation_ref,
                )
            ),
        )
    )

    assert result == RoleContext(general="K^G", game="updated L^G")
    request = client.calls[0]
    assert request["model"] == "gemma4:e4b"
    assert request["format"] == updated_context_json_schema()
    assert request["messages"][0]["content"] == "goal game instructions"
    assert request["messages"][2] == {"role": "assistant", "content": "```json\n"}
    assert len(request["messages"][1]["images"]) == 1
    records = drain_model_input_debug_records(updater)
    assert records[0]["call_slot"] == "updater_goal"
    assert records[0]["provider"] == "ollama"
    assert records[0]["metadata"]["response_output_text"] == (
        '{"updated_context": "updated L^G"}'
    )
    assert records[0]["metadata"]["response_payload"]["message"]["content"] == (
        '{"updated_context": "updated L^G"}'
    )


def test_ollama_updater_repairs_invalid_structured_json(tmp_path) -> None:
    (tmp_path / "goal_game_context_updater_prompt.md").write_text(
        "goal game instructions",
        encoding="utf-8",
    )
    client = FakeOllamaClient(
        [
            '{"updated_context": "unterminated',
            '{"updated_context": "repaired L^G"}',
        ]
    )
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e4b",
            instruction_dir=str(tmp_path),
        ),
        client=client,
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    result = updater.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=RoleContext(general="K^G", game="L^G"),
            current_observation=actual_observation,
            post_decision_predictions=PostDecisionPredictions(
                goal_prediction=ToolResult(
                    id="goal-out",
                    tool="goal",
                    predicted_description=[],
                    source_observation_ref=observation_ref,
                )
            ),
        )
    )

    assert result == RoleContext(general="K^G", game="repaired L^G")
    assert len(client.calls) == 2
    repair_request = client.calls[1]
    assert repair_request["format"] == updated_context_json_schema()
    assert repair_request["messages"][0]["content"] == "goal game instructions"
    records = drain_model_input_debug_records(updater)
    assert [record["phase"] for record in records] == [
        "update_prompt",
        "repair_prompt",
    ]
    assert [record["attempt"] for record in records] == [0, 1]
    assert records[1]["call_slot"] == "updater_goal"
    assert records[1]["usage"] == {"prompt_eval_count": 1, "eval_count": 1}


def test_real_updater_rejects_malformed_json(tmp_path) -> None:
    (tmp_path / "world_game_context_updater_prompt.md").write_text(
        "world game instructions",
        encoding="utf-8",
    )
    updater = OpenAIUpdaterAdapter(
        OpenAIUpdaterConfig(
            backend="openai",
            model="gpt-5-nano",
            instruction_dir=str(tmp_path),
        ),
        client=FakeOpenAIClient("not json"),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    action = ActionSpec(action_id="ACTION1")
    predicted_description = _description_prediction()
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    with pytest.raises(UpdaterOutputError, match="must be JSON"):
        updater.update_world_game_context(
            WorldGameContextUpdateInput(
                previous_context=RoleContext(general="K^S", game="L^S"),
                current_observation=actual_observation,
                post_decision_predictions=PostDecisionPredictions(
                    world_prediction=ToolResult(
                        id="world-out",
                        tool="world",
                        predicted_description=predicted_description,
                        source_observation_ref=observation_ref,
                        action=action,
                    )
                ),
                submitted_action=action,
            )
        )


def test_real_updater_rejects_missing_updated_context(tmp_path) -> None:
    (tmp_path / "goal_game_context_updater_prompt.md").write_text(
        "goal game instructions",
        encoding="utf-8",
    )
    updater = OllamaUpdaterAdapter(
        OllamaUpdaterConfig(
            backend="ollama",
            model="gemma4:e4b",
            instruction_dir=str(tmp_path),
        ),
        client=FakeOllamaClient('{"summary": "no context"}'),
    )
    observation_ref = ObservationRef(memory="state", id="obs-0")
    predicted_description = _description_prediction()
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    with pytest.raises(UpdaterOutputError, match="updated_context"):
        updater.update_goal_game_context(
            GoalGameContextUpdateInput(
                previous_context=RoleContext(general="K^G", game="L^G"),
                current_observation=actual_observation,
                post_decision_predictions=PostDecisionPredictions(
                    goal_prediction=ToolResult(
                        id="goal-out",
                        tool="goal",
                        predicted_description=predicted_description,
                        source_observation_ref=observation_ref,
                    )
                ),
            )
        )
