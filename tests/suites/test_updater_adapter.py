"""Tests for the updater model shell."""

import json

from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
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
    load_updater_instructions,
)
from face_of_agi.models.updater.config import (
    OllamaUpdaterConfig,
    OpenAIUpdaterConfig,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.updater.contracts import updated_context_json_schema
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


def _committed_prediction_description(text: str, role: str) -> list[dict]:
    marker = f"Committed {role} prediction description JSON:\n"
    return json.loads(text.split(marker, 1)[1])


def _agent_game_update_input(
    *,
    previous_context: RoleContext | None = None,
    action: ActionSpec | None = None,
    turn_metrics: AgentProgressFeedback | None = None,
    current_turn_world_game_context: str = "L^S current turn",
    current_turn_goal_game_context: str = "L^G current turn",
    previous_turn_world_game_context: str | None = None,
) -> AgentGameContextUpdateInput:
    observation_ref = ObservationRef(memory="state", id="obs-0")
    final_action = action or ActionSpec(action_id="ACTION1")
    return AgentGameContextUpdateInput(
        previous_context=previous_context or RoleContext(general="K^X", game="L^X"),
        previous_observation=_observation("obs-0", 0, (255, 255, 255)),
        current_observation=_observation("obs-1", 1, (0, 0, 0)),
        current_turn_world_game_context=current_turn_world_game_context,
        current_turn_goal_game_context=current_turn_goal_game_context,
        previous_turn_world_game_context=previous_turn_world_game_context,
        trace=AgentTrace(
            step=0,
            first_observation_ref=observation_ref,
            current_observation_ref=observation_ref,
            final_action=final_action,
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
        return PromptUpdateProviderResponse(
            target=request.target,
            text=json.dumps(
                {"updated_context": f"{request.target.role}-{request.target.segment}"}
            ),
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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
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
                current_observation_ref=observation_ref,
                actual_next_observation_ref=observation_ref,
                previous_observation=previous_observation,
                actual_next_observation=actual_observation,
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
                current_observation_ref=observation_ref,
                actual_next_observation_ref=observation_ref,
                previous_observation=previous_observation,
                actual_next_observation=actual_observation,
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
        RoleContext(general="K", game="world-game"),
        RoleContext(general="K", game="goal-game"),
        RoleContext(general="K", game="agent-game"),
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
    assert provider.requests[0].text.startswith(
        "Previous world context:\nL\n\nAction:\nACTION1"
    )
    assert "Committed world prediction description JSON" in provider.requests[0].text
    assert '"bbox_2d"' in provider.requests[0].text
    assert [image.label for image in provider.requests[0].images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]
    assert provider.requests[1].text.startswith("Previous goal context:\nL")
    assert "Committed goal prediction description JSON" in provider.requests[1].text
    assert '"bbox_2d"' in provider.requests[1].text
    assert [image.label for image in provider.requests[1].images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]
    assert _committed_prediction_description(provider.requests[0].text, "world") == [
        {
            "bbox_2d": [0.0, 0.0, 4.0, 4.0],
            "description": "predicted change",
        }
    ]
    assert _committed_prediction_description(provider.requests[1].text, "goal") == [
        {
            "bbox_2d": [0.0, 0.0, 4.0, 4.0],
            "description": "predicted change",
        }
    ]
    assert '"task": "agent_game"' in provider.requests[2].text
    assert [image.label for image in provider.requests[2].images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]
    assert (
        provider.requests[3].text
        == "Game world model text:\nL\n\nGeneral world model text:\nK"
    )
    assert provider.requests[3].images == ()
    assert (
        provider.requests[4].text
        == "Game goal model text:\nL\n\nGeneral goal model text:\nK"
    )
    assert provider.requests[4].images == ()


def test_default_game_updater_instructions_include_action_glossary() -> None:
    world_instructions = load_updater_instructions(task="world_game")
    goal_instructions = load_updater_instructions(task="goal_game")
    agent_instructions = load_updater_instructions(task="agent_game")

    for instructions in (world_instructions, agent_instructions):
        assert "Action glossary:" in instructions
        assert "`RESET`: initialize or restart the game or level state." in instructions
        assert (
            "`ACTION6`: coordinate action targeting `x,y` on the 64x64 game grid."
            in instructions
        )
        assert "`ACTION7`: undo-style simple action." in instructions
    for instructions in (world_instructions, goal_instructions):
        normalized_instructions = " ".join(instructions.split())
        assert "forwarded exactly as returned" in normalized_instructions
        assert "`bbox_2d`" in instructions


def test_prompt_updater_payload_summarizes_observations_and_attaches_images(
    tmp_path,
) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    provider = FakePromptUpdaterProvider()
    updater = PromptUpdaterAdapter(
        provider=provider,
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )
    action = ActionSpec(action_id="ACTION1")

    updater.update_agent_game_context(
        _agent_game_update_input(
            previous_context=RoleContext(general="K^X", game="L^X"),
            action=action,
            turn_metrics=AgentProgressFeedback(
                time_cost=2.0,
                score_delta=1.0,
            ),
            current_turn_world_game_context="world context before S update",
            current_turn_goal_game_context="goal context before G update",
            previous_turn_world_game_context="world context before previous turn",
        )
    )

    request = provider.requests[0]
    payload = json.loads(request.text)
    json.dumps(payload)
    assert [image.label for image in request.images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]
    assert payload["attached_images"] == [
        {
            "label": "previous_observation_frame",
            "source": "transition.previous_observation",
        },
        {
            "label": "current_observation_frame",
            "source": "transition.current_observation",
        },
    ]
    transition = payload["transition"]
    previous_frame = transition["previous_observation"]["frame"]
    current_frame = transition["current_observation"]["frame"]
    assert previous_frame["__type__"] == "face_of_agi.frame.png_base64.v1"
    assert previous_frame["kind"] == "image_summary"
    assert previous_frame["encoding"] == "base64_omitted_for_prompt"
    assert current_frame["kind"] == "image_summary"
    assert transition["trace"]["final_action"]["action_id"] == "ACTION1"
    assert (
        transition["current_turn_world_game_context"]
        == "world context before S update"
    )
    assert (
        transition["current_turn_goal_game_context"]
        == "goal context before G update"
    )
    assert (
        transition["previous_turn_world_game_context"]
        == "world context before previous turn"
    )
    assert transition["turn_metrics"]["time_cost"] == 2.0
    assert transition["turn_metrics"]["score_delta"] == 1.0


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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    actual_observation = Observation(
        id="obs-1",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    updater.update_world_game_context(
        WorldGameContextUpdateInput(
            previous_context=RoleContext(general="K^S", game="L^S"),
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            previous_observation=previous_observation,
            actual_next_observation=actual_observation,
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
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            previous_observation=previous_observation,
            actual_next_observation=actual_observation,
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

    assert provider.requests[0].text.startswith(
        "Previous world context:\nL^S\n\nAction:\nNONE"
    )
    assert "Committed world prediction description JSON" in provider.requests[0].text
    assert '"bbox_2d"' in provider.requests[0].text
    assert [image.label for image in provider.requests[0].images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]
    assert provider.requests[1].text.startswith("Previous goal context:\nL^G")
    assert "Committed goal prediction description JSON" in provider.requests[1].text
    assert '"bbox_2d"' in provider.requests[1].text
    assert [image.label for image in provider.requests[1].images] == [
        "previous_observation_frame",
        "current_observation_frame",
    ]


def test_world_goal_game_updaters_require_transition_observations(tmp_path) -> None:
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
    previous_observation = _observation("obs-0", 0, (255, 255, 255))
    actual_observation = _observation("obs-1", 1, (0, 0, 0))

    with pytest.raises(ValueError, match="previous observation"):
        updater.update_world_game_context(
            WorldGameContextUpdateInput(
                previous_context=RoleContext(general="K^S", game="L^S"),
                current_observation_ref=observation_ref,
                actual_next_observation_ref=observation_ref,
                actual_next_observation=actual_observation,
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
                current_observation_ref=observation_ref,
                actual_next_observation_ref=observation_ref,
                previous_observation=previous_observation,
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
    client = FakeOpenAIClient('{"updated_context": "updated L^S"}')
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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    result = updater.update_world_game_context(
        WorldGameContextUpdateInput(
            previous_context=RoleContext(general="K^S", game="L^S"),
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            previous_observation=previous_observation,
            actual_next_observation=actual_observation,
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

    assert result == RoleContext(general="K^S", game="updated L^S")
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5-nano"
    assert request["instructions"] == "world game instructions"
    assert request["text"]["format"] == {
        "type": "json_schema",
        "name": "updater_context_update",
        "strict": True,
        "schema": updated_context_json_schema(),
    }
    content = request["input"][0]["content"]
    assert content[0]["text"].startswith(
        "Previous world context:\nL^S\n\nAction:\nACTION1"
    )
    assert [item["type"] for item in content] == [
        "input_text",
        "input_image",
        "input_image",
    ]
    assert "Committed world prediction description JSON" in content[0]["text"]
    assert '"bbox_2d"' in content[0]["text"]
    records = drain_model_input_debug_records(updater)
    assert records[0]["call_slot"] == "updater_world"
    assert records[0]["provider"] == "openai"
    assert records[0]["phase"] == "update_prompt"
    assert len(records[0]["request"]["input"][0]["content"]) == 3
    assert records[0]["usage"] == {"input_tokens": 1, "output_tokens": 1}
    assert records[0]["metadata"]["response_output_text"] == (
        '{"updated_context": "updated L^S"}'
    )
    assert records[0]["metadata"]["response_payload"]["output_text"] == (
        '{"updated_context": "updated L^S"}'
    )


def test_openai_updater_updates_agent_game_context_from_structured_json(
    tmp_path,
) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    client = FakeOpenAIClient('{"updated_context": "updated L^X"}')
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
    previous_observation = _observation("obs-0", 0, (255, 255, 255))
    current_observation = _observation("obs-1", 1, (0, 0, 0))

    result = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K^X", game="L^X"),
            previous_observation=previous_observation,
            current_observation=current_observation,
            current_turn_world_game_context="L^S",
            current_turn_goal_game_context="L^G",
            previous_turn_world_game_context=None,
            trace=AgentTrace(
                step=0,
                first_observation_ref=observation_ref,
                current_observation_ref=observation_ref,
                final_action=action,
            ),
        )
    )

    assert result == RoleContext(general="K^X", game="updated L^X")
    request = client.responses.calls[0]
    assert request["instructions"] == "agent game instructions"
    assert request["text"]["format"]["schema"] == updated_context_json_schema()
    payload = json.loads(request["input"][0]["content"][0]["text"])
    assert payload["task"] == "agent_game"
    assert payload["role"] == "agent"
    assert payload["attached_images"] == [
        {
            "label": "previous_observation_frame",
            "source": "transition.previous_observation",
        },
        {
            "label": "current_observation_frame",
            "source": "transition.current_observation",
        },
    ]
    assert [item["type"] for item in request["input"][0]["content"]] == [
        "input_text",
        "input_image",
        "input_image",
    ]


def test_agent_game_updater_returns_updated_context_exactly(tmp_path) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    updated_context = (
        "Replace the initial game context with one concise learned note.\n"
        "ACTION labels may be rewritten or removed by the updater."
    )
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

    assert result == RoleContext(general="K^X", game=updated_context)
    assert len(client.calls[0]["messages"][1]["images"]) == 2


def test_agent_game_updater_does_not_reject_or_rewrite_context_text(tmp_path) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    updated_context = json.dumps(
        {
            "context": {
                "score": 0.5,
                "state": "middle",
                "score_delta": 1.0,
                "note": "The world model prediction error improved by -0.00194.",
            }
        },
        sort_keys=True,
    )
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
            "This runtime run uses game game-1.\n"
            "ACTION1: up arrow\n"
            "ACTION2: down arrow"
        ),
    )

    result = updater.update_agent_game_context(
        _agent_game_update_input(
            previous_context=previous,
        )
    )

    assert result.game == updated_context


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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    result = updater.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=RoleContext(general="K^G", game="L^G"),
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            previous_observation=previous_observation,
            actual_next_observation=actual_observation,
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
    assert request["messages"][1]["content"].startswith("Previous goal context:\nL^G")
    assert (
        "Committed goal prediction description JSON"
        in request["messages"][1]["content"]
    )
    assert '"bbox_2d"' in request["messages"][1]["content"]
    assert request["messages"][2] == {"role": "assistant", "content": "```json\n"}
    assert len(request["messages"][1]["images"]) == 2
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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    result = updater.update_goal_game_context(
        GoalGameContextUpdateInput(
            previous_context=RoleContext(general="K^G", game="L^G"),
            current_observation_ref=observation_ref,
            actual_next_observation_ref=observation_ref,
            previous_observation=previous_observation,
            actual_next_observation=actual_observation,
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
    assert "Repair attempt 1" in repair_request["messages"][1]["content"]
    assert "updater response must be JSON" in repair_request["messages"][1]["content"]
    assert "exactly one top-level `updated_context` field" in repair_request[
        "messages"
    ][1]["content"]
    assert "not an object, array, `game` field, or `general` field" in repair_request[
        "messages"
    ][1]["content"]
    assert '{"updated_context": "unterminated' in repair_request["messages"][1][
        "content"
    ]
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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    with pytest.raises(UpdaterOutputError, match="must be JSON"):
        updater.update_world_game_context(
            WorldGameContextUpdateInput(
                previous_context=RoleContext(general="K^S", game="L^S"),
                current_observation_ref=observation_ref,
                actual_next_observation_ref=observation_ref,
                previous_observation=previous_observation,
                actual_next_observation=actual_observation,
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
    previous_observation = Observation(
        id="obs-0",
        step=0,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    actual_observation = Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
    )

    with pytest.raises(UpdaterOutputError, match="updated_context"):
        updater.update_goal_game_context(
            GoalGameContextUpdateInput(
                previous_context=RoleContext(general="K^G", game="L^G"),
                current_observation_ref=observation_ref,
                actual_next_observation_ref=observation_ref,
                previous_observation=previous_observation,
                actual_next_observation=actual_observation,
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
