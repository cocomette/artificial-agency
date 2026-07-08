"""Tests for the shared agent creator role store."""

from __future__ import annotations

from threading import Event, Lock

import pytest

from face_of_agi.agent_creator import (
    AgentCreatorService,
    AgentCreatorStore,
    AgentRoleDefinition,
)
from face_of_agi.models.agent_creator import (
    AgentCreatorInput,
    CreatorMutation,
    CreatorOrchestratorResponse,
)
from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    Observation,
    ObservationRef,
    RoleContext,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory


def test_agent_creator_store_seeds_and_reads_latest_snapshot(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    roles = (
        AgentRoleDefinition(
            role="probing",
            meta_description="probe uncertain mechanics",
            role_instructions="test unknown effects",
        ),
    )

    seeded = store.seed_defaults(roles=roles, general_system_prompt="general")
    latest = store.read_latest_complete_role_snapshot()

    assert latest == seeded
    assert latest is not None
    assert latest.roles == roles
    assert latest.general_system_prompt == "general"


def test_agent_creator_store_rejects_empty_role_snapshot(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()

    with pytest.raises(ValueError, match="at least one role"):
        store.write_role_snapshot(roles=(), general_system_prompt="general")


def test_agent_creator_store_publishes_staged_revisions_atomically(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    seeded = store.seed_defaults(
        roles=(
            AgentRoleDefinition(
                role="probing",
                meta_description="probe uncertain mechanics",
                role_instructions="test unknown effects",
            ),
        ),
        general_system_prompt="general",
    )
    run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)

    store.stage_role_revision(
        role=AgentRoleDefinition(
            role="probing",
            meta_description="probe stuck states",
            role_instructions="probe no-op loops",
        ),
        active=True,
        operation="update",
        created_by_run_id=run.id,
    )

    assert store.read_latest_complete_role_snapshot() == seeded
    store.complete_creator_run(run.id)
    latest = store.read_latest_complete_role_snapshot()
    assert latest is not None
    assert latest.roles[0].meta_description == "probe stuck states"


def test_agent_creator_store_hides_inactive_latest_revision(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()
    role = AgentRoleDefinition(
        role="probing",
        meta_description="probe uncertain mechanics",
        role_instructions="test unknown effects",
    )
    store.seed_defaults(roles=(role,), general_system_prompt="general")
    run = store.create_creator_run(request_ids=(1,), max_tool_calls=4)
    store.stage_role_revision(
        role=role,
        active=False,
        operation="delete",
        created_by_run_id=run.id,
    )
    store.complete_creator_run(run.id)

    assert store.read_latest_complete_role_snapshot() is None


def test_agent_creator_store_claims_distinct_game_requests(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    store.initialize_schema()

    first_id = store.enqueue_game_request(
        run_id="run-1",
        game_id="game-1",
        memory_database_path=str(tmp_path / "memory-1.sqlite"),
    )
    duplicate_id = store.enqueue_game_request(
        run_id="run-1",
        game_id="game-1",
        memory_database_path=str(tmp_path / "memory-1.sqlite"),
    )
    store.enqueue_game_request(
        run_id="run-2",
        game_id="game-2",
        memory_database_path=str(tmp_path / "memory-2.sqlite"),
    )

    assert duplicate_id == first_id
    batch = store.claim_full_batch(batch_size=2)

    assert batch is not None
    assert batch.request_ids == (first_id, first_id + 1)
    assert [request.game_id for request in batch.requests] == ["game-1", "game-2"]


def test_agent_creator_service_processes_next_full_batch_after_active_batch(
    tmp_path,
) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    model = _BlockingCreatorModel()
    service = AgentCreatorService(store=store, creator_model=model, batch_size=1)
    first_memory = _memory_with_complete_state(tmp_path, run_id="run-1")
    second_memory = _memory_with_complete_state(tmp_path, run_id="run-2")

    service.enqueue_game(
        run_id="run-1",
        game_id="game-1",
        memory_database_path=first_memory.database.path,
    )
    assert model.entered_first_call.wait(timeout=2)
    service.enqueue_game(
        run_id="run-2",
        game_id="game-1",
        memory_database_path=second_memory.database.path,
    )
    model.release_first_call.set()
    service.close()

    assert model.call_count == 2
    latest = store.read_latest_complete_role_snapshot()
    assert latest is not None
    probing = next(role for role in latest.roles if role.role == "probing")
    assert probing.meta_description == "role 2"


def test_agent_creator_service_windows_strategy_history(tmp_path) -> None:
    store = AgentCreatorStore(tmp_path / "agent_creator.sqlite")
    model = _RecordingCreatorModel()
    service = AgentCreatorService(
        store=store,
        creator_model=model,
        batch_size=1,
        strategy_history_window=1,
    )
    memory = _memory_with_strategy_history(
        tmp_path,
        run_id="run-1",
        strategies=("first strategy", "second strategy", "latest strategy"),
    )

    service.enqueue_game(
        run_id="run-1",
        game_id="game-1",
        memory_database_path=memory.database.path,
    )
    service.close()

    assert model.strategy_history == (("policy", "latest strategy"),)


def _memory_with_complete_state(tmp_path, *, run_id: str) -> StateMemory:
    memory = StateMemory(SQLiteDatabase(tmp_path / f"{run_id}.sqlite"))
    observation = Observation(id=f"{run_id}-obs", step=1)
    ref = ObservationRef(memory="state", id=observation.id)
    memory.write_state(
        run_id=run_id,
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=ContextDocuments(agent=RoleContext(game="strategy")),
        agent_trace=AgentTrace(
            step=1,
            first_observation_ref=ref,
            current_observation_ref=ref,
            final_action=ActionSpec(action_id="ACTION1"),
        ),
        metadata={
            "agent_strategy": {"role": "policy", "strategy": "reach target"},
            "agent_creator_action_history": [],
            "world_model_context": {
                "world_description": "current world",
                "action_effects": {"ACTION1": "moves right"},
            },
        },
    )
    return memory


def _memory_with_strategy_history(
    tmp_path,
    *,
    run_id: str,
    strategies: tuple[str, ...],
) -> StateMemory:
    memory = StateMemory(SQLiteDatabase(tmp_path / f"{run_id}.sqlite"))
    ref = ObservationRef(memory="state", id=f"{run_id}-obs")
    for index, strategy in enumerate(strategies, start=1):
        observation = Observation(id=f"{run_id}-obs-{index}", step=index)
        memory.write_state(
            run_id=run_id,
            game_id="game-1",
            step=index,
            frame_index=0,
            frame_count=1,
            current_observation=observation,
            chosen_action=ActionSpec(action_id="ACTION1"),
                contexts=ContextDocuments(agent=RoleContext(game=strategy)),
            agent_trace=AgentTrace(
                step=index,
                first_observation_ref=ref,
                current_observation_ref=ref,
                final_action=ActionSpec(action_id="ACTION1"),
            ),
            metadata={
                "agent_strategy": {"role": "policy", "strategy": strategy},
                "agent_creator_action_history": [],
            },
        )
    return memory


class _BlockingCreatorModel:
    def __init__(self) -> None:
        self.entered_first_call = Event()
        self.release_first_call = Event()
        self._lock = Lock()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def run_creator(
        self,
        creator_input: AgentCreatorInput,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        assert max_tool_calls == 4
        assert creator_input.batch_items
        assert creator_input.batch_items[0].state_id is not None
        assert creator_input.batch_items[0].strategy_history
        assert creator_input.batch_items[0].world_model_context == {
            "world_description": "current world",
            "action_effects": {"ACTION1": "moves right"},
        }
        with self._lock:
            self._call_count += 1
            call_count = self._call_count
        if call_count == 1:
            self.entered_first_call.set()
            assert self.release_first_call.wait(timeout=2)
        return CreatorOrchestratorResponse(
            mutations=(
                CreatorMutation(
                    action="update",
                    role_name="probing",
                    identified_failures=f"failure {call_count}",
                    meta_description=f"role {call_count}",
                ),
            ),
            tool_call_count=1,
        )

    def update_role_instructions(self, author_input) -> str:
        call_count = self.call_count
        assert author_input.role_name == "probing"
        return f"instructions {call_count}"

    def create_role_instructions(self, author_input) -> str:
        raise AssertionError(f"unexpected create_role call: {author_input}")


class _RecordingCreatorModel:
    def __init__(self) -> None:
        self.strategy_history: tuple[tuple[str, str], ...] = ()

    def run_creator(
        self,
        creator_input: AgentCreatorInput,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        del max_tool_calls
        self.strategy_history = tuple(
            (item.role, item.strategy)
            for item in creator_input.batch_items[0].strategy_history
        )
        return CreatorOrchestratorResponse()

    def update_role_instructions(self, author_input) -> str:
        raise AssertionError(f"unexpected update_role call: {author_input}")

    def create_role_instructions(self, author_input) -> str:
        raise AssertionError(f"unexpected create_role call: {author_input}")
