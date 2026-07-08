"""Tests for active vLLM Chat Completions model providers."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

from arcengine import GameAction
from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    AgentCandidateAction,
    CandidateValuePrediction,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    Observation,
    RoleContext,
    WorldPrediction,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.goal import (
    GoalPredictionInput,
    VLLMGoalAdapter,
    VLLMGoalConfig,
)
from face_of_agi.models.memory import (
    MemoryBuildInput,
    MemoryLedgerEntry,
    VLLMMemoryAdapter,
    VLLMMemoryConfig,
)
from face_of_agi.models.interest import (
    InterestPredictionInput,
    VLLMInterestAdapter,
    VLLMInterestConfig,
)
from face_of_agi.models.orchestrator_agent.config import VLLMOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.providers.vllm import vllm_exclusive_gate, vllm_request_gate
from face_of_agi.models.reward_judge import (
    RewardJudgeInput,
    VLLMRewardJudgeAdapter,
    VLLMRewardJudgeConfig,
)
from face_of_agi.models.structured_output import OUTPUT_SCHEMA_INSTRUCTION
from face_of_agi.models.updater import AGENT_GAME_CONTEXT_KEYS
from face_of_agi.models.vllm_roles import parse_json_object
from face_of_agi.models.world import VLLMWorldAdapter, VLLMWorldConfig
from face_of_agi.models.world.contracts import WorldPredictionInput
from face_of_agi.models.updater.config import VLLMUpdaterConfig
from face_of_agi.models.updater.contracts import (
    agent_game_updated_context_json_schema,
    updated_context_json_schema,
)
from face_of_agi.models.updater.providers.vllm import VLLMUpdaterAdapter
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
)

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


class FakeChatCompletions:
    """Captures vLLM-compatible Chat Completions calls."""

    def __init__(self, contents: str | dict[str, Any] | list[Any]) -> None:
        self.contents = list(contents) if isinstance(contents, list) else [contents]
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        content = self.contents[index]
        message = (
            {"role": "assistant", "content": content}
            if isinstance(content, str)
            else {"role": "assistant", **content}
        )
        return {
            "id": f"chatcmpl-{index}",
            "model": request["model"],
            "object": "chat.completion",
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        }


class FakeOpenAIChatClient:
    """Tiny OpenAI Chat client stand-in."""

    def __init__(self, contents: str | list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(contents))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def test_vllm_request_gate_delays_exclusive_restart_until_request_exits() -> None:
    entered_request = threading.Event()
    release_request = threading.Event()
    exclusive_entered: list[str] = []

    def hold_request() -> None:
        with vllm_request_gate():
            entered_request.set()
            assert release_request.wait(timeout=2.0)

    request_thread = threading.Thread(target=hold_request)
    request_thread.start()
    try:
        assert entered_request.wait(timeout=2.0)

        def enter_exclusive() -> None:
            with vllm_exclusive_gate():
                exclusive_entered.append("entered")

        exclusive_thread = threading.Thread(target=enter_exclusive)
        exclusive_thread.start()
        time.sleep(0.02)
        assert exclusive_entered == []

        release_request.set()
        exclusive_thread.join(timeout=2.0)
        assert exclusive_entered == ["entered"]
    finally:
        release_request.set()
        request_thread.join(timeout=2.0)


def test_vllm_agent_uses_chat_completions_images_and_structured_output() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(json.dumps({"action": {"action_id": "ACTION1"}}))
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            input_image_size="10x12",
            use_response_format=True,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(game="choose directly"),
        current,
        [ActionSpec(action_id="ACTION1")],
    )

    request = client.calls[0]
    images = _input_images(request)
    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["backend"] == "vllm"
    assert request["model"] == MODEL
    assert request["messages"][0]["role"] == "system"
    assert len(images) == 1
    assert [_decode_data_url_image(image["image_url"]["url"]).size for image in images] == [
        (10, 12),
    ]
    assert request["response_format"]["json_schema"]["name"] == "agent_final_action"
    records = drain_model_input_debug_records(adapter)
    assert records[0]["provider"] == "vllm"
    assert records[0]["phase"] == "final_action"
    assert records[0]["usage"]["total_tokens"] == 6

    adapter.activate_lora_adapter("agent-lora")
    adapter.decide(
        RoleContext(game="choose directly"),
        current,
        [ActionSpec(action_id="ACTION1")],
    )

    assert client.calls[1]["model"] == "agent-lora"


def test_vllm_agent_selection_records_exact_request_and_uses_active_adapter() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(json.dumps({"action": {"action_id": "ACTION1"}}))
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            input_image_size="10x12",
            use_response_format=True,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    candidates = (
        AgentCandidateAction(
            action=ActionSpec(action_id="ACTION1"),
            source="runtime_simple_action",
            rank=0,
        ),
    )

    decision = adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=("advance",),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=candidates,
        world_predictions=(
            WorldPrediction(
                candidate_index=0,
                action=ActionSpec(action_id="ACTION1"),
                predicted_change="changed",
            ),
        ),
        interest_prediction=InterestPrediction(
            candidate_values=(
                CandidateValuePrediction(
                    candidate_index=0,
                    action=ActionSpec(action_id="ACTION1"),
                    expected_learning_progress=0.4,
                    expected_goal_delta=0.2,
                    confidence=0.5,
                    metadata={
                        "confidence_adjusted_learning_progress": 0.2,
                        "blended_score": 0.2,
                    },
                ),
            )
        ),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
    )

    request = client.calls[0]
    prompt_text = request["messages"][1]["content"][0]["text"]
    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["training_request"] == request
    assert "interest_value:" in prompt_text
    assert "blended_score=0.2" in prompt_text
    assert request["model"] == MODEL
    assert request["response_format"]["json_schema"]["name"] == "agent_final_action"

    adapter.activate_lora_adapter("agent-lora")
    adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=("advance",),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=candidates,
        world_predictions=(
            WorldPrediction(
                candidate_index=0,
                action=ActionSpec(action_id="ACTION1"),
                predicted_change="changed",
            ),
        ),
        interest_prediction=InterestPrediction(
            candidate_values=(
                CandidateValuePrediction(
                    candidate_index=0,
                    action=ActionSpec(action_id="ACTION1"),
                    expected_learning_progress=0.4,
                    expected_goal_delta=0.2,
                    confidence=0.5,
                ),
            )
        ),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert client.calls[1]["model"] == "agent-lora"


def test_vllm_agent_candidate_parser_accepts_single_action_object() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        json.dumps(
            {
                "action": {
                    "action_id": "ACTION6",
                    "data": {"x": 500, "y": 500},
                },
                "notes": "center probe",
            }
        )
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    action_space = (
        ActionSpec(action_id=GameAction.ACTION1),
        ActionSpec(action_id=GameAction.ACTION6, data={"x": 0, "y": 0}),
    )

    candidates = adapter.propose_candidate_actions(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        action_space=action_space,
        max_candidates=1,
        glossary_actions=action_space,
    )

    assert len(candidates) == 1
    assert candidates[0].action.name == "ACTION6"
    assert candidates[0].rationale == "center probe"


def test_vllm_agent_candidate_parser_accepts_schema_echo_payload() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        json.dumps(
            {
                "additionalProperties": False,
                "properties": {
                    "candidate_actions": [
                        {
                            "action_id": "ACTION6",
                            "data": {"x": 500, "y": 500},
                        }
                    ],
                    "notes": "schema echo with values",
                },
            }
        )
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    action_space = (
        ActionSpec(action_id=GameAction.ACTION1),
        ActionSpec(action_id=GameAction.ACTION6, data={"x": 0, "y": 0}),
    )

    candidates = adapter.propose_candidate_actions(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        action_space=action_space,
        max_candidates=1,
        glossary_actions=action_space,
    )

    assert len(candidates) == 1
    assert candidates[0].action.name == "ACTION6"
    assert candidates[0].rationale == "schema echo with values"


def test_vllm_agent_candidate_repairs_schema_only_payload() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        [
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "candidate_actions": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "notes": {"type": "string"},
                    },
                    "required": ["candidate_actions", "notes"],
                    "additionalProperties": False,
                }
            ),
            json.dumps(
                {
                    "candidate_actions": [
                        {
                            "action_id": "ACTION6",
                            "data": {"x": 500, "y": 500},
                        }
                    ],
                    "notes": "repaired coordinate",
                }
            ),
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=1,
        ),
        client=client,
    )
    action_space = (
        ActionSpec(action_id=GameAction.ACTION1),
        ActionSpec(action_id=GameAction.ACTION6, data={"x": 0, "y": 0}),
    )

    candidates = adapter.propose_candidate_actions(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        action_space=action_space,
        max_candidates=1,
        glossary_actions=action_space,
    )

    assert len(candidates) == 1
    assert candidates[0].action.name == "ACTION6"
    assert candidates[0].rationale == "repaired coordinate"
    assert len(client.calls) == 2
    assert "Repair attempt 1" in client.calls[1]["messages"][-1]["content"]


def test_vllm_agent_candidate_parser_accepts_additional_actions_key() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        json.dumps(
            {
                "additionalActions": [
                    {
                        "action_id": "ACTION6",
                        "data": {"x": 500, "y": 500},
                    }
                ],
                "notes": "alternate key",
            }
        )
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    action_space = (
        ActionSpec(action_id=GameAction.ACTION1),
        ActionSpec(action_id=GameAction.ACTION6, data={"x": 0, "y": 0}),
    )

    candidates = adapter.propose_candidate_actions(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        action_space=action_space,
        max_candidates=1,
        glossary_actions=action_space,
    )

    assert len(candidates) == 1
    assert candidates[0].action.name == "ACTION6"
    assert candidates[0].rationale == "alternate key"


def test_vllm_agent_selection_accepts_candidate_index() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(json.dumps({"candidate_index": 3}))
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    candidate = AgentCandidateAction(
        action=ActionSpec(action_id="ACTION1"),
        source="runtime_simple_action",
        rank=3,
    )

    decision = adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=(candidate,),
        world_predictions=(
            WorldPrediction(
                candidate_index=3,
                action=candidate.action,
                predicted_change="changed",
            ),
        ),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert decision.final_action == candidate.action


def test_vllm_agent_selection_accepts_nested_candidate_index() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(json.dumps({"action": {"candidate_index": 3}}))
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    candidate = AgentCandidateAction(
        action=ActionSpec(action_id="ACTION1"),
        source="runtime_simple_action",
        rank=3,
    )

    decision = adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=(candidate,),
        world_predictions=(
            WorldPrediction(
                candidate_index=3,
                action=candidate.action,
                predicted_change="changed",
            ),
        ),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert decision.final_action == candidate.action


def test_vllm_agent_selection_strips_simple_action_data() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        json.dumps({"action": {"action_id": "ACTION1", "data": {"x": 1, "y": 2}}})
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    candidate = AgentCandidateAction(
        action=ActionSpec(action_id="ACTION1"),
        source="runtime_simple_action",
        rank=0,
    )

    decision = adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=(candidate,),
        world_predictions=(
            WorldPrediction(
                candidate_index=0,
                action=candidate.action,
                predicted_change="changed",
            ),
        ),
        glossary_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert decision.final_action == ActionSpec(action_id="ACTION1")


def test_vllm_agent_selection_repairs_disallowed_action() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        [
            json.dumps(
                {
                    "action": {
                        "action_id": "ACTION6",
                        "data": {"x": 500, "y": 500},
                    }
                }
            ),
            json.dumps({"action": {"action_id": "ACTION2"}}),
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            max_tool_calls=0,
            repair_attempts=1,
        ),
        client=client,
    )
    candidates = (
        AgentCandidateAction(
            action=ActionSpec(action_id="ACTION1"),
            source="runtime_simple_action",
            rank=0,
        ),
        AgentCandidateAction(
            action=ActionSpec(action_id="ACTION2"),
            source="runtime_simple_action",
            rank=1,
        ),
    )

    decision = adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=candidates,
        world_predictions=(
            WorldPrediction(
                candidate_index=0,
                action=candidates[0].action,
                predicted_change="first",
            ),
            WorldPrediction(
                candidate_index=1,
                action=candidates[1].action,
                predicted_change="second",
            ),
        ),
        glossary_actions=(ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")),
    )

    assert decision.final_action == ActionSpec(action_id="ACTION2")
    assert len(client.calls) == 2
    assert "action 'ACTION6' is not allowed" in client.calls[1]["messages"][-1]["content"]


def test_vllm_agent_candidate_and_selection_include_schema_in_instructions() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        [
            json.dumps(
                {
                    "candidate_actions": [
                        {"action_id": "ACTION6", "data": {"x": 2, "y": 3}}
                    ],
                    "notes": "try coordinate",
                }
            ),
            json.dumps(
                {
                    "action": {
                        "action_id": "ACTION6",
                        "data": {"x": 2, "y": 3},
                    }
                }
            ),
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
            use_response_format=True,
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )
    action_space = (
        ActionSpec(action_id=GameAction.ACTION1),
        ActionSpec(action_id=GameAction.ACTION6, data={"x": 0, "y": 0}),
    )

    candidates = adapter.propose_candidate_actions(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        action_space=action_space,
        max_candidates=1,
        glossary_actions=action_space,
    )
    adapter.select_action(
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=3,
            confidence=0.5,
        ),
        current_observation=current,
        candidates=candidates,
        world_predictions=(
            WorldPrediction(
                candidate_index=0,
                action=candidates[0].action,
                predicted_change="changed",
            ),
        ),
        glossary_actions=action_space,
    )

    candidate_request, selection_request = client.calls
    assert OUTPUT_SCHEMA_INSTRUCTION in candidate_request["messages"][0]["content"]
    assert OUTPUT_SCHEMA_INSTRUCTION in selection_request["messages"][0]["content"]
    assert candidate_request["response_format"]["json_schema"]["name"] == (
        "agent_candidate_actions"
    )
    assert selection_request["response_format"]["json_schema"]["name"] == (
        "agent_final_action"
    )


def test_vllm_world_uses_base_then_active_adapter_and_records_exact_request() -> None:
    current = _observation("obs-current")
    client = FakeOpenAIChatClient(
        json.dumps({"predicted_change": "the tile moved right"})
    )
    adapter = VLLMWorldAdapter(
        VLLMWorldConfig(
            model=MODEL,
            input_image_size="10x12",
            use_response_format=True,
        ),
        client=client,
    )

    prediction = adapter.predict_transition(
        WorldPredictionInput(
            run_id="run-1",
            game_id="game-1",
            candidate_index=0,
            current_observation=current,
            action=ActionSpec(action_id="ACTION1"),
            memory=MemoryDocument(document="memory"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )

    request = client.calls[0]
    images = _input_images(request)
    assert prediction.predicted_change == "the tile moved right"
    assert prediction.metadata["training_request"] == request
    assert prediction.metadata["training_schema_name"] == "world_prediction"
    assert request["model"] == MODEL
    assert request["response_format"]["json_schema"]["name"] == "world_prediction"
    assert [_decode_data_url_image(image["image_url"]["url"]).size for image in images] == [
        (10, 12),
    ]

    adapter.activate_lora_adapter("world-lora")
    adapter.predict_transition(
        WorldPredictionInput(
            run_id="run-1",
            game_id="game-1",
            candidate_index=0,
            current_observation=current,
            action=ActionSpec(action_id="ACTION1"),
            memory=MemoryDocument(document="memory"),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )

    assert client.calls[1]["model"] == "world-lora"


def test_vllm_json_roles_include_schema_in_instructions_and_not_extra_body() -> None:
    current = _observation("obs-current")
    first = _observation("obs-first")
    client = FakeOpenAIChatClient(
        [
            json.dumps({"document": "memory document"}),
            json.dumps(
                {
                    "goal": "solve",
                    "subgoals": ["advance"],
                    "steps_remaining": 3,
                    "confidence": 0.5,
                }
            ),
            json.dumps({"predicted_change": "tile changed"}),
            json.dumps(
                {
                    "candidate_values": [
                        {
                            "candidate_index": 0,
                            "expected_learning_progress": 0.4,
                            "expected_goal_delta": 0.2,
                            "confidence": 0.5,
                            "notes": "learnable",
                        }
                    ]
                }
            ),
            json.dumps({"score": 0.75, "notes": "close", "error_tags": []}),
        ]
    )
    memory = VLLMMemoryAdapter(
        VLLMMemoryConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )
    goal = VLLMGoalAdapter(
        VLLMGoalConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )
    world = VLLMWorldAdapter(
        VLLMWorldConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )
    interest = VLLMInterestAdapter(
        VLLMInterestConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )
    judge = VLLMRewardJudgeAdapter(
        VLLMRewardJudgeConfig(
            model=MODEL,
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )

    memory_doc = memory.build_memory(
        MemoryBuildInput(
            run_id="run-1",
            game_id="game-1",
            first_observation=first,
            current_observation=current,
            ledger=(
                MemoryLedgerEntry(
                    turn_id=1,
                    action="ACTION1",
                    change_summary="tile changed",
                ),
            ),
        )
    )
    goal_prediction = goal.predict_goal(
        GoalPredictionInput(
            run_id="run-1",
            game_id="game-1",
            memory=memory_doc,
            current_observation=current,
        )
    )
    world_prediction = world.predict_transition(
        WorldPredictionInput(
            run_id="run-1",
            game_id="game-1",
            candidate_index=0,
            current_observation=current,
            action=ActionSpec(action_id="ACTION1"),
            memory=memory_doc,
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )
    interest_prediction = interest.score_candidates(
        InterestPredictionInput(
            run_id="run-1",
            game_id="game-1",
            turn_id=1,
            current_observation=current,
            memory=memory_doc,
            goal=goal_prediction,
            candidates=(
                AgentCandidateAction(
                    action=ActionSpec(action_id="ACTION1"),
                    source="runtime_simple_action",
                    rank=0,
                ),
            ),
            world_predictions=(world_prediction,),
        )
    )
    judge.judge_prediction(
        RewardJudgeInput(
            run_id="run-1",
            game_id="game-1",
            turn_id=1,
            action=ActionSpec(action_id="ACTION1"),
            prediction=world_prediction,
            change_summary="tile changed",
            previous_observation=first,
            current_observation=current,
        )
    )

    assert goal_prediction.steps_remaining == 3
    assert interest_prediction.candidate_values[0].expected_learning_progress == 0.4
    assert interest_prediction.metadata["training_request"] == client.calls[3]
    assert _memory_prompt_ledger(client.calls[0]) == [
        {
            "turn_id": 1,
            "action": "ACTION1",
            "change_summary": "tile changed",
        }
    ]
    for request in client.calls:
        assert "response_format" not in request
        assert OUTPUT_SCHEMA_INSTRUCTION in request["messages"][0]["content"]
        assert "include_output_schema_in_instructions" not in request.get(
            "extra_body",
            {},
        )


def test_vllm_json_roles_repair_non_json_response() -> None:
    current = _observation("obs-current")
    first = _observation("obs-first")
    client = FakeOpenAIChatClient(
        [
            "Based on the current state, the board is unchanged.",
            json.dumps({"document": "repaired memory document"}),
        ]
    )
    memory = VLLMMemoryAdapter(
        VLLMMemoryConfig(
            model=MODEL,
            repair_attempts=1,
        ),
        client=client,
    )

    memory_doc = memory.build_memory(
        MemoryBuildInput(
            run_id="run-1",
            game_id="game-1",
            first_observation=first,
            current_observation=current,
            ledger=(),
        )
    )

    assert memory_doc.document == "repaired memory document"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-2]["role"] == "assistant"
    assert "Repair attempt 1" in client.calls[1]["messages"][-1]["content"]


def test_vllm_json_parser_accepts_common_model_wrappers() -> None:
    assert parse_json_object(
        '```json\n{"document": "ok"}\n```',
        label="memory",
    ) == {"document": "ok"}
    assert parse_json_object(
        'Here is the JSON:\n{"document": "ok"}',
        label="memory",
    ) == {"document": "ok"}


def test_vllm_agent_game_updater_uses_object_schema_and_image(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    payload = {
        "updated_context": {
            key: f"{key} updated" for key in AGENT_GAME_CONTEXT_KEYS
        }
    }
    client = FakeOpenAIChatClient(json.dumps(payload))
    updater = VLLMUpdaterAdapter(
        VLLMUpdaterConfig(
            model=MODEL,
            instruction_dir=str(tmp_path),
            use_response_format=True,
        ),
        client=client,
    )

    result = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="L"),
            current_observation=_observation("obs-agent"),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            action_history_window=2,
        )
    )

    request = client.calls[0]
    assert json.loads(result.game) == payload["updated_context"]
    assert request["response_format"]["json_schema"]["schema"] == (
        agent_game_updated_context_json_schema()
    )
    assert request["messages"][1]["content"][1]["type"] == "image_url"
    records = drain_model_input_debug_records(updater)
    assert [record["phase"] for record in records] == ["update_prompt"]


def test_vllm_general_updater_uses_string_schema(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    client = FakeOpenAIChatClient(json.dumps({"updated_context": "new K"}))
    updater = VLLMUpdaterAdapter(
        VLLMUpdaterConfig(
            model=MODEL,
            instruction_dir=str(tmp_path),
            use_response_format=True,
        ),
        client=client,
    )

    result = updater.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role="agent",
            previous_context=RoleContext(general="old K", game="L"),
            run_id="run-1",
            game_id="game-1",
        )
    )

    assert result == RoleContext(general="new K", game="L")
    assert client.calls[0]["response_format"]["json_schema"]["schema"] == (
        updated_context_json_schema()
    )


def _input_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    content = request["messages"][1]["content"]
    return [item for item in content if item.get("type") == "image_url"]


def _memory_prompt_ledger(request: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = request["messages"][1]["content"][0]["text"]
    start = prompt.index("[")
    ledger, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return ledger


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _write_instruction_files(path) -> None:
    (path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    (path / "agent_general_context_updater_prompt.md").write_text(
        "agent general instructions",
        encoding="utf-8",
    )


def _observation(observation_id: str) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )
