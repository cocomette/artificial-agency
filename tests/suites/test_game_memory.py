"""Tests for the game memory model role and prompt integration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace

from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    Observation,
    ObservationRef,
    RoleContext,
    RuntimeConfig,
)
from face_of_agi.environment.config import EnvironmentConfig, load_environment_config
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models import ModelRegistry
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.memory import (
    GAME_MEMORY_MAX_CHARS,
    DisabledGameMemoryAdapter,
    GameMemoryAdapter,
    GameMemoryConfig,
    GameMemoryDocument,
    GameMemoryInput,
    GameMemoryOutputError,
    OllamaGameMemoryConfig,
    OpenAIGameMemoryConfig,
    PromptGameMemoryProviderResponse,
    PromptGameMemoryRequest,
    VLLMGameMemoryConfig,
    build_game_memory_repair_prompt,
    parse_game_memory_output,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.memory.providers import (
    OllamaGameMemoryAdapter,
    OpenAIGameMemoryAdapter,
    VLLMGameMemoryAdapter,
)
from face_of_agi.models.orchestrator_agent.tooling import (
    build_decision_prompt,
    load_agent_instructions,
)
from face_of_agi.models.updater import (
    AGENT_GAME_CONTEXT_KEYS,
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    load_updater_instructions,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    PromptUpdaterAdapter,
    UpdaterConfig,
    UpdaterTaskRegistry,
)
from face_of_agi.orchestration import Orchestrator
from face_of_agi.runtime import RuntimeLoop


def _observation(id_: str, step: int, color: tuple[int, int, int]) -> Observation:
    return Observation(
        id=id_,
        step=step,
        frame=Image.new("RGB", (4, 4), color=color),
    )


def _history_entry(summary: str = "white area expanded") -> ActionHistoryEntry:
    return ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=5,
        change_summary=summary,
    )


class FakeMemoryProvider:
    backend = "fake-memory"
    model = "fake-model"

    def __init__(
        self,
        text: str = json.dumps({"memory": "# Memory\n\nA door opened."}),
    ) -> None:
        self.text = text
        self.requests: list[PromptGameMemoryRequest] = []

    def summarize_game_memory(
        self,
        request: PromptGameMemoryRequest,
    ) -> PromptGameMemoryProviderResponse:
        self.requests.append(request)
        return PromptGameMemoryProviderResponse(
            text=self.text,
            metadata={"provider_note": "ok"},
        )


class RepairingMemoryProvider(FakeMemoryProvider):
    def __init__(self) -> None:
        super().__init__(text="# raw markdown")
        self.repair_calls: list[dict[str, object]] = []

    def repair_game_memory(
        self,
        request: PromptGameMemoryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptGameMemoryProviderResponse:
        self.repair_calls.append(
            {
                "request": request,
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            }
        )
        return PromptGameMemoryProviderResponse(
            text=json.dumps({"memory": "Repaired memory."}),
            metadata={"provider_note": "repaired"},
        )


class OversizedMemoryProvider(FakeMemoryProvider):
    def __init__(self, *, repair_text: str) -> None:
        super().__init__(
            text=json.dumps({"memory": "x" * (GAME_MEMORY_MAX_CHARS + 1)})
        )
        self.repair_text = repair_text
        self.repair_calls: list[dict[str, object]] = []

    def repair_game_memory(
        self,
        request: PromptGameMemoryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptGameMemoryProviderResponse:
        self.repair_calls.append(
            {
                "request": request,
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            }
        )
        return PromptGameMemoryProviderResponse(
            text=json.dumps({"memory": self.repair_text}),
            metadata={"provider_note": "length-repaired"},
        )


def test_game_memory_adapter_builds_prompt_with_images_without_run_or_game_ids() -> None:
    provider = FakeMemoryProvider()
    adapter = GameMemoryAdapter(provider, GameMemoryConfig())

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
            metadata={"non_identifying": "ok"},
        )
    )

    request = provider.requests[0]
    assert document.markdown == "# Memory\n\nA door opened."
    assert document.metadata["available"] is True
    assert document.metadata["repair_attempts"] == 0
    assert request.output_schema["required"] == ["memory"]
    assert request.output_schema["properties"]["memory"]["maxLength"] == (
        GAME_MEMORY_MAX_CHARS
    )
    assert "10,000 characters" in request.instructions
    assert "10,000 characters" in request.text
    assert [image.label for image in request.images] == [
        "first_game_frame",
        "current_game_frame",
    ]
    assert "## Action history" in request.text
    assert "white area expanded" in request.text
    assert "run_id" not in request.text
    assert "game_id" not in request.text
    assert "run_id" not in request.metadata
    assert "game_id" not in request.metadata


def test_disabled_game_memory_adapter_returns_unavailable_without_model_call() -> None:
    adapter = DisabledGameMemoryAdapter()

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
            metadata={"non_identifying": "ok"},
        )
    )

    assert document.markdown == "not available"
    assert document.is_available() is False
    assert document.metadata["backend"] == "none"
    assert document.metadata["disabled"] is True


def test_game_memory_adapter_repairs_invalid_structured_output_once() -> None:
    provider = RepairingMemoryProvider()
    adapter = GameMemoryAdapter(provider, GameMemoryConfig(repair_attempts=1))

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
        )
    )

    assert document.markdown == "Repaired memory."
    assert document.metadata["repair_attempts"] == 1
    assert document.metadata["provider_note"] == "repaired"
    assert provider.repair_calls[0]["attempt"] == 1
    assert provider.repair_calls[0]["invalid_text"] == "# raw markdown"
    assert "must be JSON" in str(provider.repair_calls[0]["validation_error"])


def test_game_memory_compresses_oversized_valid_output_with_repair_budget() -> None:
    provider = OversizedMemoryProvider(repair_text="short memory")
    adapter = GameMemoryAdapter(provider, GameMemoryConfig(repair_attempts=1))

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
        )
    )

    assert document.markdown == "short memory"
    assert document.metadata["repair_attempts"] == 1
    assert document.metadata["structural_repair_attempts"] == 0
    assert document.metadata["memory_initial_oversize"] is True
    assert document.metadata["memory_oversize"] is False
    assert document.metadata["memory_length_repair_attempted"] is True
    assert document.metadata["memory_length_repair_succeeded"] is True
    assert provider.repair_calls[0]["attempt"] == 1
    assert "expected at most 10000" in str(
        provider.repair_calls[0]["validation_error"]
    )


def test_game_memory_accepts_oversized_valid_output_when_budget_is_exhausted() -> None:
    provider = OversizedMemoryProvider(repair_text="unused")
    adapter = GameMemoryAdapter(provider, GameMemoryConfig(repair_attempts=0))

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
        )
    )

    assert len(document.markdown) == GAME_MEMORY_MAX_CHARS + 1
    assert document.metadata["repair_attempts"] == 0
    assert document.metadata["memory_oversize"] is True
    assert document.metadata["memory_length_repair_attempted"] is False
    assert document.metadata["memory_length_repair_skipped_reason"] == (
        "repair_budget_exhausted"
    )
    assert provider.repair_calls == []


def test_game_memory_repair_prompt_is_compact_and_does_not_replay_input() -> None:
    prompt = build_game_memory_repair_prompt(
        invalid_text="x" * (GAME_MEMORY_MAX_CHARS + 100),
        validation_error="too long",
        attempt=1,
    )

    assert "Original game memory input" not in prompt
    assert "10,000 characters" in prompt
    assert "truncated for compact repair" in prompt
    assert len(prompt) < 7_000


def test_game_memory_parses_raw_and_fenced_json() -> None:
    assert parse_game_memory_output('{"memory": "A door opened."}') == "A door opened."
    assert (
        parse_game_memory_output('```json\n{"memory": "A path opened."}\n```')
        == "A path opened."
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# Memory\n\nA door opened.", "must be JSON"),
        ('{"summary": "A door opened."}', "missing keys: memory"),
        ('{"memory": "ok", "extra": "no"}', "unexpected keys: extra"),
        ('{"memory": ""}', "must be non-empty"),
        ('{"memory": {"nested": "no"}}', "must be a string"),
        ("[]", "must be a JSON object"),
    ],
)
def test_game_memory_rejects_invalid_structured_output(
    text: str,
    match: str,
) -> None:
    with pytest.raises(GameMemoryOutputError, match=match):
        parse_game_memory_output(text)


def test_game_memory_input_rejects_identifying_metadata() -> None:
    with pytest.raises(ValueError, match="game_id"):
        GameMemoryInput(
            action_history=(),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
            metadata={"game_id": "game-1"},
        )


class FakeVLLMCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="chat-memory",
            model=kwargs["model"],
            object="chat.completion",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"memory": "# Memory\n\nPath opened."})
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class FakeVLLMClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FakeVLLMCompletions(),
        )


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="response-memory",
            model=kwargs["model"],
            status="completed",
            output_text=json.dumps({"memory": "OpenAI memory."}),
            usage=None,
            incomplete_details=None,
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "message": {
                "content": json.dumps({"memory": "Ollama memory."}),
            }
        }


def test_vllm_memory_debug_capture_metadata_excludes_run_and_game_ids() -> None:
    client = FakeVLLMClient()
    adapter = VLLMGameMemoryAdapter(
        VLLMGameMemoryConfig(model="fake-vllm"),
        client=client,
    )

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
            metadata={"non_identifying": "ok"},
        )
    )

    call = client.chat.completions.calls[0]
    response_format = call["response_format"]
    assert document.markdown == "# Memory\n\nPath opened."
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "game_memory"
    assert response_format["json_schema"]["schema"]["required"] == ["memory"]
    records = drain_model_input_debug_records(adapter)

    assert len(records) == 1
    assert records[0]["call_slot"] == "memory"
    assert records[0]["metadata"]["role"] == "memory"
    assert "run_id" not in records[0]["metadata"]
    assert "game_id" not in records[0]["metadata"]


def test_openai_memory_uses_json_schema_text_format() -> None:
    client = FakeOpenAIClient()
    adapter = OpenAIGameMemoryAdapter(
        OpenAIGameMemoryConfig(model="fake-openai"),
        client=client,
    )

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
        )
    )

    call = client.responses.calls[0]
    text_format = call["text"]["format"]
    assert document.markdown == "OpenAI memory."
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "game_memory"
    assert text_format["strict"] is True
    assert text_format["schema"]["required"] == ["memory"]


def test_ollama_memory_uses_schema_backed_structured_chat() -> None:
    client = FakeOllamaClient()
    adapter = OllamaGameMemoryAdapter(
        OllamaGameMemoryConfig(model="fake-ollama"),
        client=client,
    )

    document = adapter.summarize_game_memory(
        GameMemoryInput(
            action_history=(_history_entry(),),
            first_observation=_observation("obs-first", 0, (255, 255, 255)),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
        )
    )

    call = client.calls[0]
    assert document.markdown == "Ollama memory."
    assert call["format"]["required"] == ["memory"]
    assert call["messages"][-1]["role"] == "assistant"
    assert call["messages"][-1]["content"].startswith("```json")


def test_selected_vllm_configs_load_run063_model_prompt_controls() -> None:
    config_paths = (
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_debug.yaml",
        "src/face_of_agi/runtime/configs/vllm/vllm_h100_qwen36_35b_fp8.yaml",
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_h100_qwen36_35b_fp8_parallel.yaml",
    )

    for config_path in config_paths:
        config = load_environment_config(config_path)

        assert config.models.agent.options["action6_targeting_mode"] == "coordinates"
        assert config.models.change.options["activate_components"] is True
        assert config.models.change.options["persist_changed_elements_only"] is True
        assert config.models.change.options["max_nb_components"] == 50
        assert config.models.memory.backend == "vllm"
        assert "max_completion_tokens" not in config.models.memory.options


def test_rtx6000_debug_config_loads_memory_output_controls() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_debug.yaml"
    )

    memory = config.models.memory
    assert memory.backend == "vllm"
    assert memory.options["input_image_crop_box_normalized"] == [
        0.046875,
        0.046875,
        0.953125,
        0.953125,
    ]


def test_agent_prompt_includes_game_memory_without_replacing_recent_actions() -> None:
    prompt = build_decision_prompt(
        context=RoleContext(general="agent K", game="agent L"),
        action_space=[ActionSpec(action_id="ACTION1")],
        recent_action_history=(_history_entry("latest action changed the wall"),),
        game_memory=GameMemoryDocument("Goal: reach the opened door."),
    )

    assert "## Game memory" in prompt
    assert "Goal: reach the opened door." in prompt
    assert "## Recent actions" in prompt
    assert "latest action changed the wall" in prompt


def test_agent_and_updater_instructions_describe_game_memory_input() -> None:
    agent_instructions = load_agent_instructions()
    updater_instructions = load_updater_instructions(task="agent_game")
    normalized_updater_instructions = " ".join(updater_instructions.split())

    assert "`Game memory`" in agent_instructions
    assert "Use Game memory and Recent actions together" in agent_instructions
    assert "Elements may be targets, triggers, objects" in agent_instructions
    assert "`Game memory`" in updater_instructions
    assert "copy it wholesale" in updater_instructions
    assert "`Elements and associated changes`" in updater_instructions
    assert "Elements can be targets, triggers, objects, characters" in (
        normalized_updater_instructions
    )
    assert "When the agent is stagnant" in updater_instructions


class FakeUpdaterProvider:
    backend = "fake-updater"
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[PromptUpdateRequest] = []

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        payload = {
            "updated_context": {
                key: f"updated {key}" for key in AGENT_GAME_CONTEXT_KEYS
            }
        }
        return PromptUpdateProviderResponse(
            target=request.target,
            text=json.dumps(payload),
        )

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        raise AssertionError("repair should not be needed")


def test_updater_prompt_includes_game_memory_action_history_and_context_history() -> None:
    provider = FakeUpdaterProvider()
    updater = PromptUpdaterAdapter(provider, UpdaterConfig())

    updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="L"),
            current_observation=_observation("obs-current", 1, (0, 0, 0)),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            action_history_window=4,
            game_memory=GameMemoryDocument("Memory says ACTION1 opened a path."),
            action_history=(_history_entry("path opened"),),
        )
    )

    prompt = provider.requests[0].text
    assert "## Game memory" in prompt
    assert "Memory says ACTION1 opened a path." in prompt
    assert "## Action history" in prompt
    assert "path opened" in prompt
    assert "## Agent context history" in prompt


class FakeEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def list_available_games(self) -> Sequence[object]:
        return ()

    def list_local_games(self) -> Sequence[object]:
        return ()

    def resolve_game_id(self, game_index: int) -> str:
        return "game-1"

    def select_game_by_id(self, game_id: str) -> str:
        return game_id

    def reset(self) -> Observation:
        self.steps = 0
        return _observation("obs-0", 0, (255, 255, 255))

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, object] | None = None,
    ) -> Observation:
        del action, reasoning
        self.steps += 1
        return _observation(
            f"obs-{self.steps}",
            self.steps,
            (max(0, 255 - self.steps), 0, 0),
        )

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (ActionSpec(action_id="ACTION1"),)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id="game-1",
            available_actions=tuple(self.get_action_space()),
        )


class CapturingAgent:
    def __init__(self) -> None:
        self.memory_inputs: list[str] = []

    def decide(
        self,
        context: RoleContext,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: object | None = None,
        recent_action_history: tuple[object, ...] = (),
        *,
        glossary_actions: Sequence[ActionSpec],
        first_observation_ref: ObservationRef | None = None,
        recent_action_history_available: bool = True,
        action_outcome_evidence: object | None = None,
        game_memory: GameMemoryDocument | None = None,
    ) -> DecisionResult:
        del (
            context,
            tool_runtime,
            recent_action_history,
            glossary_actions,
            recent_action_history_available,
            action_outcome_evidence,
        )
        memory = game_memory or GameMemoryDocument.not_available()
        self.memory_inputs.append(memory.markdown)
        final_action = action_space[0]
        current_ref = ObservationRef(memory="state", id=current_observation.id)
        return DecisionResult(
            final_action=final_action,
            trace=AgentTrace(
                step=current_observation.step,
                first_observation_ref=first_observation_ref or current_ref,
                current_observation_ref=current_ref,
                final_action=final_action,
            ),
        )


class FakeChangeSummary:
    def summarize(
        self,
        previous_observation: Observation,
        current_observation: Observation,
        action: ActionSpec,
        *,
        glossary_actions: Sequence[ActionSpec],
        frame_observations: Sequence[Observation] | None = None,
        previous_change_elements: Sequence[ChangeSummaryElement] = (),
    ) -> ChangeSummaryResult:
        del (
            previous_observation,
            current_observation,
            action,
            glossary_actions,
            previous_change_elements,
        )
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="transition",
                    element_description="frame transition",
                    element_mutation=f"{len(frame_observations or ())} frame transition",
                ),
            ),
            changed_pixel_count=1,
            change_detected=True,
            metadata={},
            changed_pixel_percent=1.0,
        )


class CapturingMemory:
    def __init__(self) -> None:
        self.inputs: list[GameMemoryInput] = []

    def summarize_game_memory(
        self,
        memory_input: GameMemoryInput,
    ) -> GameMemoryDocument:
        self.inputs.append(memory_input)
        return GameMemoryDocument(
            markdown=f"memory-{len(self.inputs)}",
            metadata={"call": len(self.inputs)},
        )


class CapturingAgentUpdater:
    def __init__(self) -> None:
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        self.inputs.append(update_input)
        return update_input.previous_context


class NoopGeneralUpdater:
    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        return update_input.previous_context


def test_runtime_updates_memory_after_real_actions_and_reuses_for_next_agent(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.sqlite")
    state = StateMemory(database)
    agent = CapturingAgent()
    memory = CapturingMemory()
    updater = CapturingAgentUpdater()
    orchestrator = Orchestrator(
        state_memory=state,
        models=ModelRegistry(
            orchestrator_agent=agent,
            change_summary_model=FakeChangeSummary(),
            game_memory_model=memory,
            updater_tasks=UpdaterTaskRegistry(
                agent_game_updater=updater,
                general_updater=NoopGeneralUpdater(),
            ),
        ),
        contexts=ContextDocuments(),
    )

    result = RuntimeLoop(orchestrator).run(
        config=RuntimeConfig(run_id="run-1"),
        environment=FakeEnvironment(),
        environment_config=EnvironmentConfig(
            game_id="game-1",
            max_actions_per_level=2,
            debug_keep_all_m_states=True,
            debug_trace="off",
        ),
    )

    states = state.list_states(game_id="game-1")
    assert result.stop_reason == "action_limit_reached"
    assert agent.memory_inputs == ["not available", "memory-1"]
    assert [item.game_memory.markdown for item in updater.inputs] == [
        "memory-1",
        "memory-2",
    ]
    assert len(memory.inputs) == 2
    assert [len(item.action_history) for item in memory.inputs] == [1, 2]
    assert [item.first_observation.id for item in memory.inputs] == [
        "obs-0",
        "obs-0",
    ]
    assert [item.current_observation.id for item in memory.inputs] == [
        "obs-1",
        "obs-2",
    ]
    assert all(not hasattr(item, "run_id") for item in memory.inputs)
    assert all(not hasattr(item, "game_id") for item in memory.inputs)
    assert [state.metadata["game_memory"]["document"] for state in states] == [
        "memory-1",
        "memory-2",
    ]
    assert all(
        state.metadata["game_memory"]["updated_this_turn"] for state in states
    )
