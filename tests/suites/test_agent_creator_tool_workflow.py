"""Tests for agent-creator mutation-plan workflow mechanics."""

from __future__ import annotations

import json
from threading import Event, Lock
from typing import Any

import pytest

from face_of_agi.agent_creator import (
    AgentCreatorBatchItem,
    AgentCreatorStore,
    AgentRoleDefinition,
    AgentStrategySnapshot,
    RoleMutationToolExecutor,
)
from face_of_agi.contracts import ActionHistoryEntry, ActionSpec, Observation
from face_of_agi.models.action_history import (
    grouped_action_history_text,
    model_facing_action_text,
)
from face_of_agi.models.agent_creator import (
    AgentCreatorAdapter,
    AgentCreatorOutputError,
    AgentCreatorProviderResponse,
    CreatorOrchestratorRequest,
    CreatorOrchestratorResponse,
    CreatorMutation,
    AgentCreatorInput,
    OllamaAgentCreatorConfig,
    RoleAuthorInput,
    VLLMAgentCreatorConfig,
    agent_creator_orchestrator_output_schema,
    parse_creator_orchestrator_plan_output,
)
from face_of_agi.models.agent_creator.providers.ollama import OllamaAgentCreatorProvider
from face_of_agi.models.agent_creator.providers.vllm import VLLMAgentCreatorProvider
from face_of_agi.models.providers.ollama import OllamaChatCall, OllamaStructuredChatResult


def test_mutation_executor_updates_role_and_publishes_after_run(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    initial = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    store.seed_defaults(roles=(initial,), general_system_prompt="general")
    run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)
    executor = RoleMutationToolExecutor(
        store=store,
        role_author=_RoleAuthor(),
        run_id=run.id,
        roles=(initial,),
        general_system_prompt="general",
        max_tool_calls=4,
    )

    result = json.loads(
        executor.execute_tool_call(
            "update",
            {
                "role_name": "probing",
                "identified_failures": "repeated no-op loops",
                "meta_description": "probe repeated no-op loops",
            },
        )
    )

    assert result == {"status": "ok"}
    assert store.read_latest_complete_role_snapshot().roles == (initial,)
    store.complete_creator_run(run.id)
    latest = store.read_latest_complete_role_snapshot()
    assert latest is not None
    assert latest.roles[0].meta_description == "probe repeated no-op loops"
    assert latest.roles[0].role_instructions == "avoid repeated no-op loops"


def test_mutation_executor_reactivates_inactive_role_name(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    initial = AgentRoleDefinition(
        role="recover",
        meta_description="recover from bad states",
        role_instructions="escape bad states",
    )
    store.seed_defaults(roles=(initial,), general_system_prompt="general")
    delete_run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)
    store.stage_role_revision(
        role=initial,
        active=False,
        operation="delete",
        created_by_run_id=delete_run.id,
    )
    store.complete_creator_run(delete_run.id)
    add_run = store.create_creator_run(request_ids=(2,), max_tool_calls=4)
    executor = RoleMutationToolExecutor(
        store=store,
        role_author=_RoleAuthor(),
        run_id=add_run.id,
        roles=(),
        general_system_prompt="general",
        max_tool_calls=4,
    )

    result = json.loads(
        executor.execute_tool_call(
            "add",
            {
                "role_name": "recover",
                "instruction_guidance": "recover from avoidable dead ends",
                "meta_description": "recover when a game reaches avoidable dead ends",
            },
        )
    )

    assert result == {"status": "ok"}
    store.complete_creator_run(add_run.id)
    latest = store.read_latest_complete_role_snapshot()
    assert latest is not None
    assert latest.roles[0].role == "recover"
    assert latest.roles[0].meta_description == (
        "recover when a game reaches avoidable dead ends"
    )
    assert latest.roles[0].role_instructions == "handle recover from avoidable dead ends"


def test_mutation_executor_returns_tool_error_for_invalid_delete(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    initial = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    store.seed_defaults(roles=(initial,), general_system_prompt="general")
    run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)
    executor = RoleMutationToolExecutor(
        store=store,
        role_author=_RoleAuthor(),
        run_id=run.id,
        roles=(initial,),
        general_system_prompt="general",
        max_tool_calls=4,
    )

    result = json.loads(
        executor.execute_tool_call("delete", {"role_name": "probing"})
    )

    assert result == {"reason": "final_role", "status": "failed"}


def test_mutation_executor_rejects_add_when_role_capacity_is_reached(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    initial = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    store.seed_defaults(roles=(initial,), general_system_prompt="general")
    run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)
    author = _RoleAuthor()
    executor = RoleMutationToolExecutor(
        store=store,
        role_author=author,
        run_id=run.id,
        roles=(initial,),
        general_system_prompt="general",
        max_tool_calls=4,
        max_roles=1,
    )

    result = json.loads(
        executor.execute_tool_call(
            "add",
            {
                "role_name": "recover",
                "instruction_guidance": "recover from dead ends",
                "meta_description": "recover from avoidable dead ends",
            },
        )
    )

    assert result == {"reason": "max_roles_reached", "status": "failed"}
    assert author.create_calls == 0


def test_agent_creator_adapter_builds_multimodal_batch_request() -> None:
    role = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    provider = _CreatorProvider()
    adapter = AgentCreatorAdapter(provider)
    action_history = (
        ActionHistoryEntry(
            action=ActionSpec("ACTION4"),
            controllable=True,
            changed_pixel_count=0.1,
            change_summary="moved toward the blue gate",
            completed_levels=0,
            action_count=6,
            action_mode="probing",
        ),
        ActionHistoryEntry(
            action=ActionSpec.none(),
            controllable=False,
            changed_pixel_count=0.7,
            change_summary=(
                "animation produced changes but it is uncertain "
                "what changed exactly."
            ),
            animation_frame_count=4,
            avg_changed_pixel_count=0.7431,
        ),
    )

    adapter.run_creator(
        AgentCreatorInput(
            batch_items=(
                AgentCreatorBatchItem(
                    run_id="run-1",
                    game_id="game-1",
                    strategy_history=(
                        AgentStrategySnapshot(
                            role="probing",
                            strategy="test a different object after no visible change",
                        ),
                    ),
                    current_observation=Observation(
                        id="obs-1",
                        step=3,
                        frame=_test_image(),
                    ),
                    action_history=action_history,
                    roles=(role,),
                    general_system_prompt="general",
                    world_model_context={
                        "world_description": "red player can move near blue gate",
                        "action_effects": {"ACTION1": "moves player right"},
                    },
                ),
            ),
            current_roles=(role,),
            general_system_prompt="general",
            metadata={"max_roles": 8, "request_ids": (1,)},
        ),
        max_tool_calls=4,
    )

    assert provider.request is not None
    assert provider.request.tools == ()
    payload = json.loads(provider.request.text)
    assert payload["available_roles"] == [
        {
            "role": "probing",
            "meta_description": "probe uncertain mechanics",
        }
    ]
    batch_item = payload["batch_items"][0]
    assert {
        key: value
        for key, value in batch_item.items()
        if key != "action_history"
    } == {
        "current_frame": "attached image 1",
        "world_model_context": {
            "world_description": "red player can move near blue gate",
            "action_effects": {"ACTION1": "moves player right"},
        },
        "strategy_history": [
            {
                "role": "probing",
                "strategy": "test a different object after no visible change",
            }
        ],
    }
    assert batch_item["action_history"] == grouped_action_history_text(
        action_history,
        action_text=model_facing_action_text,
        numbered=True,
    )
    assert payload["metadata"] == {"max_roles": 8}
    assert len(provider.request.images) == 1
    assert "Output JSON must match this schema exactly" in provider.request.instructions
    assert '"enum": [' in provider.request.instructions
    assert '"mutations"' in provider.request.instructions
    assert '"update"' in provider.request.instructions


def test_agent_creator_schema_defines_distinct_mutation_variants() -> None:
    schema = agent_creator_orchestrator_output_schema(4)

    mutation_variants = schema["properties"]["mutations"]["items"]["anyOf"]

    assert mutation_variants == [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["delete"]},
                "role_name": {"type": "string"},
            },
            "required": ["action", "role_name"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add"]},
                "role_name": {"type": "string"},
                "instruction_guidance": {"type": "string"},
                "meta_description": {"type": "string"},
            },
            "required": [
                "action",
                "role_name",
                "instruction_guidance",
                "meta_description",
            ],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["update"]},
                "role_name": {"type": "string"},
                "identified_failures": {"type": "string"},
                "meta_description": {"type": "string"},
            },
            "required": ["action", "role_name", "identified_failures"],
            "additionalProperties": False,
        },
    ]


def test_agent_creator_adapter_tailors_role_author_payloads() -> None:
    role = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    provider = _CreatorProvider()
    adapter = AgentCreatorAdapter(provider)

    adapter.create_role_instructions(
        RoleAuthorInput(
            role_name="recover",
            instruction_guidance="recover from repeated no-op loops",
            general_system_prompt="general",
            metadata={"run_id": 10},
        )
    )
    add_payload = _latest_author_payload(provider)

    adapter.update_role_instructions(
        RoleAuthorInput(
            role_name="probing",
            identified_failures="keeps repeating stale probes",
            current_role=role,
            general_system_prompt="general",
            metadata={"run_id": 11},
        )
    )
    update_payload = _latest_author_payload(provider)

    assert add_payload == {
        "role_name": "recover",
        "instruction_guidance": "recover from repeated no-op loops",
        "general_system_prompt": "general",
    }
    assert update_payload == {
        "role_name": "probing",
        "current_role": {
            "role": "probing",
            "meta_description": "probe uncertain mechanics",
            "role_instructions": "test unknown effects",
        },
        "identified_failures": "keeps repeating stale probes",
        "general_system_prompt": "general",
    }


def test_creator_plan_rejects_multiple_mutations_for_same_role() -> None:
    with pytest.raises(AgentCreatorOutputError, match="more than one mutation"):
        parse_creator_orchestrator_plan_output(
            json.dumps(
                {
                    "mutations": [
                        {
                            "action": "update",
                            "role_name": "probing",
                            "instruction_guidance": "",
                            "identified_failures": "stale probing",
                            "meta_description": "",
                        },
                        {
                            "action": "delete",
                            "role_name": "probing",
                            "instruction_guidance": "",
                            "identified_failures": "",
                            "meta_description": "",
                        },
                    ]
                }
            )
        )


def test_creator_plan_ignores_irrelevant_action_fields() -> None:
    plan = parse_creator_orchestrator_plan_output(
        json.dumps(
            {
                "mutations": [
                    {
                        "action": "update",
                        "role_name": "probing",
                        "instruction_guidance": "irrelevant add-style text",
                        "identified_failures": "stale probing",
                        "meta_description": "probe stale states",
                    },
                    {
                        "action": "delete",
                        "role_name": "unused_role",
                        "instruction_guidance": "irrelevant",
                        "identified_failures": "irrelevant",
                        "meta_description": "irrelevant",
                    },
                ]
            }
        )
    )

    assert plan.mutations == (
        CreatorMutation(
            action="update",
            role_name="probing",
            identified_failures="stale probing",
            meta_description="probe stale states",
        ),
        CreatorMutation(action="delete", role_name="unused_role"),
    )


def test_mutation_executor_executes_add_update_authors_in_parallel(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    probing = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    policy = AgentRoleDefinition(
        role="policy",
        meta_description="pursue current objective",
        role_instructions="solve the current task",
    )
    store.seed_defaults(roles=(probing, policy), general_system_prompt="general")
    run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)
    author = _ParallelRoleAuthor(expected_calls=2)
    executor = RoleMutationToolExecutor(
        store=store,
        role_author=author,
        run_id=run.id,
        roles=(probing, policy),
        general_system_prompt="general",
        max_tool_calls=4,
        max_roles=4,
    )

    results = tuple(
        json.loads(item)
        for item in executor.execute_plan(
            (
                CreatorMutation(
                    action="add",
                    role_name="recover",
                    instruction_guidance="recover from repeated no-op loops",
                    meta_description="recover from repeated no-op loops",
                ),
                CreatorMutation(
                    action="update",
                    role_name="probing",
                    identified_failures="stale probing",
                    meta_description="probe less stale states",
                ),
            )
        )
    )

    assert results == ({"status": "ok"}, {"status": "ok"})
    assert author.all_calls_entered.is_set()
    store.complete_creator_run(run.id)
    latest = store.read_latest_complete_role_snapshot()
    assert latest is not None
    assert {role.role for role in latest.roles} == {"policy", "probing", "recover"}


def test_ollama_creator_provider_returns_structured_mutation_plan() -> None:
    client = _OllamaClient(
        {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "mutations": [
                            {
                                "action": "update",
                                "role_name": "probing",
                                "identified_failures": "stale probing",
                            }
                        ]
                    }
                ),
            }
        },
    )
    provider = OllamaAgentCreatorProvider(
        OllamaAgentCreatorConfig(model="model"),
        client=client,
    )
    response = provider.run_orchestrator(
        CreatorOrchestratorRequest(
            instructions="instructions",
            text="{}",
            tools=(),
        ),
        max_tool_calls=4,
    )

    assert response.tool_call_count == 1
    assert response.mutations == (
        CreatorMutation(
            action="update",
            role_name="probing",
            identified_failures="stale probing",
        ),
    )
    assert len(client.requests) == 1
    assert json.loads(client.requests[0]["messages"][1]["content"]) == {}


def test_vllm_creator_provider_returns_structured_mutation_plan() -> None:
    client = _VLLMClient(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "mutations": [
                                    {
                                        "action": "delete",
                                        "role_name": "policy",
                                    }
                                ]
                            }
                        ),
                    }
                }
            ]
        },
    )
    provider = VLLMAgentCreatorProvider(
        VLLMAgentCreatorConfig(model="model"),
        client=client,
    )
    response = provider.run_orchestrator(
        CreatorOrchestratorRequest(
            instructions="instructions",
            text="{}",
            tools=(),
        ),
        max_tool_calls=4,
    )

    assert response.tool_call_count == 1
    assert response.mutations == (
        CreatorMutation(action="delete", role_name="policy"),
    )
    assert "tools" not in client.requests[0]
    assert client.requests[0]["response_format"]["type"] == "json_schema"
    assert client.requests[0]["response_format"]["json_schema"]["name"] == (
        "agent_creator_plan"
    )
    assert json.loads(client.requests[0]["messages"][1]["content"][0]["text"]) == {}


class _RoleAuthor:
    def __init__(self) -> None:
        self.create_calls = 0

    def create_role_instructions(self, author_input: RoleAuthorInput) -> str:
        self.create_calls += 1
        return f"handle {author_input.instruction_guidance}"

    def update_role_instructions(self, author_input: RoleAuthorInput) -> str:
        return f"avoid {author_input.identified_failures}"


class _ParallelRoleAuthor:
    def __init__(self, *, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.entered_count = 0
        self.lock = Lock()
        self.all_calls_entered = Event()
        self.release_calls = Event()

    def create_role_instructions(self, author_input: RoleAuthorInput) -> str:
        self._enter_call()
        return f"handle {author_input.instruction_guidance}"

    def update_role_instructions(self, author_input: RoleAuthorInput) -> str:
        self._enter_call()
        return f"avoid {author_input.identified_failures}"

    def _enter_call(self) -> None:
        with self.lock:
            self.entered_count += 1
            if self.entered_count == self.expected_calls:
                self.all_calls_entered.set()
                self.release_calls.set()
        assert self.release_calls.wait(timeout=2)


class _CreatorProvider:
    backend = "test"
    model = "test-model"

    def __init__(self) -> None:
        self.request: CreatorOrchestratorRequest | None = None
        self.author_requests: list[Any] = []

    def run_orchestrator(
        self,
        request: CreatorOrchestratorRequest,
        *,
        max_tool_calls: int,
    ):
        del max_tool_calls
        self.request = request
        return CreatorOrchestratorResponse()

    def author_role(self, request):
        self.author_requests.append(request)
        return AgentCreatorProviderResponse(
            text=json.dumps({"role_instructions": "authored instructions"})
        )


def _latest_author_payload(provider: _CreatorProvider) -> dict[str, Any]:
    assert provider.author_requests
    return json.loads(provider.author_requests[-1].text)


def _test_image():
    from PIL import Image

    return Image.new("RGB", (4, 4), color=(255, 0, 0))


class _OllamaClient:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return self.responses.pop(0)

    def structured_chat(self, **request: Any) -> OllamaStructuredChatResult:
        self.requests.append(request)
        response = self.responses.pop(0)
        return OllamaStructuredChatResult(
            response=response,
            calls=(
                OllamaChatCall(
                    kind="structured",
                    request=request,
                    response=response,
                ),
            ),
        )


class _VLLMCompletions:
    def __init__(self, owner: "_VLLMClient") -> None:
        self.owner = owner

    def create(self, **request: Any) -> dict[str, Any]:
        self.owner.requests.append(request)
        return self.owner.responses.pop(0)


class _VLLMChat:
    def __init__(self, owner: "_VLLMClient") -> None:
        self.completions = _VLLMCompletions(owner)


class _VLLMClient:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = _VLLMChat(self)
