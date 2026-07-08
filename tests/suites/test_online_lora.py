"""Tests for v1 online LoRA update coordination."""

from __future__ import annotations

from contextlib import nullcontext
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from face_of_agi.contracts import RewardJudgeScore
from face_of_agi.environment.config import OnlineLoRAConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.orchestration import online_lora

_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_online_lora_direct_vllm_chat_uses_api_key(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"predicted_change": "changed"}
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    monkeypatch.setattr(online_lora, "urlopen", fake_urlopen)

    content = online_lora._chat_completion_content(
        "http://127.0.0.1:8000/v1",
        {"messages": []},
    )

    assert json.loads(content) == {"predicted_change": "changed"}
    assert captured == {"authorization": "Bearer EMPTY", "timeout": 120}


def test_online_lora_hf_engine_handles_adapter_io_and_replay_scoring(
    tmp_path,
    monkeypatch,
) -> None:
    class _FakeHFEngine:
        def __init__(self) -> None:
            self.loaded: list[tuple[str, str]] = []
            self.deleted: list[str] = []
            self.requests: list[dict[str, Any]] = []

        def load_adapter(self, *, adapter_name: str, adapter_path: str) -> None:
            self.loaded.append((adapter_name, adapter_path))

        def delete_adapter(self, adapter_name: str) -> None:
            self.deleted.append(adapter_name)

        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.requests.append(request)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"predicted_change": "hf prediction"}
                            )
                        }
                    }
                ]
            }

    def fail_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("HF online LoRA path must not call vLLM HTTP")

    monkeypatch.setattr(online_lora, "urlopen", fail_urlopen)
    engine = _FakeHFEngine()
    manager = online_lora.OnlineLoRAManager(
        config=_config(
            tmp_path,
            min_new_samples_per_update=1,
            held_out_recent_samples=1,
        ),
        vllm_base_url="http://127.0.0.1:8000",
        hf_engine=engine,
    )
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    sample = _write_sample(memory, role="world", turn_id=1)
    target = _shared_target("world", update_index=1)

    try:
        manager._load_lora_adapter(target)
        prediction = manager._predict_world_from_replay(
            sample,
            model=target.adapter_name,
        )
        manager._unload_lora_adapter(target)
    finally:
        manager.shutdown()

    assert engine.loaded == [(target.adapter_name, target.adapter_path)]
    assert engine.deleted == [target.adapter_name]
    assert json.loads(prediction) == {"predicted_change": "hf prediction"}
    assert engine.requests[0]["model"] == target.adapter_name


def test_runtime_vllm_server_controller_parses_restart_env(monkeypatch) -> None:
    monkeypatch.delenv("FACE_OF_AGI_VLLM_RESTART_COMMAND_JSON", raising=False)
    assert not online_lora._RuntimeVLLMServerController.from_env(
        base_url="http://127.0.0.1:8000/v1"
    ).enabled

    monkeypatch.setenv(
        "FACE_OF_AGI_VLLM_RESTART_COMMAND_JSON",
        json.dumps(["vllm", "serve", "model-id"]),
    )
    monkeypatch.setenv("FACE_OF_AGI_VLLM_PID", "12345")
    monkeypatch.setenv("FACE_OF_AGI_VLLM_RESTART_CWD", "/root")

    controller = online_lora._RuntimeVLLMServerController.from_env(
        base_url="http://127.0.0.1:8000/v1"
    )

    assert controller.enabled is True
    assert controller.command == ("vllm", "serve", "model-id")
    assert controller.pid == 12345
    assert controller.cwd == "/root"


def test_online_lora_rejects_process_global_trainer_limit_drift(tmp_path) -> None:
    first = online_lora.OnlineLoRAManager(
        config=_config(
            tmp_path,
            min_new_samples_per_update=1,
            held_out_recent_samples=1,
            max_concurrent_trainer_jobs=1,
        ),
        vllm_base_url="http://127.0.0.1:8000",
    )
    try:
        with pytest.raises(RuntimeError, match="max_concurrent_trainer_jobs"):
            online_lora.OnlineLoRAManager(
                config=_config(
                    tmp_path,
                    min_new_samples_per_update=1,
                    held_out_recent_samples=1,
                    max_concurrent_trainer_jobs=2,
                ),
                vllm_base_url="http://127.0.0.1:8000",
            )
    finally:
        first.shutdown()


def test_shared_online_lora_rejects_existing_shared_adapter_directory(
    tmp_path,
) -> None:
    config = _config(tmp_path, min_new_samples_per_update=1, held_out_recent_samples=1)
    stale = Path(config.adapter_root) / "shared" / "world"
    stale.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="empty shared adapter directory"):
        online_lora.OnlineLoRAManager(
            config=config,
            vllm_base_url="http://127.0.0.1:8000",
        )


def test_online_lora_rejects_missing_local_trainer_base_path(tmp_path) -> None:
    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        trainer_base_model_path=str(tmp_path / "missing-model"),
        trainer_local_files_only=True,
    )

    with pytest.raises(RuntimeError, match="trainer_base_model_path does not exist"):
        online_lora.OnlineLoRAManager(
            config=config,
            vllm_base_url="http://127.0.0.1:8000",
        )


def test_online_lora_rejects_fp8_trainer_base_config(tmp_path) -> None:
    trainer_base = tmp_path / "trainer-base"
    trainer_base.mkdir()
    (trainer_base / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}}),
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        trainer_base_model_path=str(trainer_base),
        trainer_local_files_only=True,
    )

    with pytest.raises(RuntimeError, match="cannot be FP8-quantized"):
        online_lora.OnlineLoRAManager(
            config=config,
            vllm_base_url="http://127.0.0.1:8000",
        )


def test_world_sft_row_decodes_replay_image_and_target(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    sample = _write_sample(memory, role="world", turn_id=1)

    row = online_lora._world_sft_row(sample)

    assert row["messages"][-1]["role"] == "assistant"
    assert json.loads(row["messages"][-1]["content"]) == {
        "predicted_change": "changed 1"
    }
    assert row["images"][0].size == (1, 1)


def test_shared_online_lora_batches_across_games_and_applies_shared_adapters(
    tmp_path,
    monkeypatch,
) -> None:
    first_memory = StateMemory(SQLiteDatabase(tmp_path / "first.sqlite"))
    second_memory = StateMemory(SQLiteDatabase(tmp_path / "second.sqlite"))
    config = _config(
        tmp_path,
        min_new_samples_per_update=2,
        held_out_recent_samples=1,
        max_train_samples=2,
    )
    first_targets = {
        "world": _ActivationTarget(),
        "interest": _ActivationTarget(),
        "agent": _ActivationTarget(),
    }
    second_targets = {
        "world": _ActivationTarget(),
        "interest": _ActivationTarget(),
        "agent": _ActivationTarget(),
    }
    loaded: list[str] = []
    sft_samples: list[tuple[str, str, int]] = []

    def fake_sft(**kwargs: Any) -> None:
        Path(kwargs["adapter_path"]).mkdir(parents=True)
        sft_samples.extend(
            (sample.run_id, sample.game_id, sample.turn_id)
            for sample in kwargs["samples"]
        )

    monkeypatch.setattr(online_lora, "_train_world_sft_adapter", fake_sft)
    monkeypatch.setattr(
        online_lora,
        "_train_grpo_adapter",
        lambda **kwargs: Path(kwargs["adapter_path"]).mkdir(parents=True),
    )
    monkeypatch.setattr(online_lora, "_chat_completion_content", _fake_chat)
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_load_lora_adapter",
        lambda self, target: loaded.append(target.adapter_name),
    )
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_unload_lora_adapter",
        lambda self, target: None,
    )

    for memory, run_id, game_id in (
        (first_memory, "run-a", "game-a"),
        (second_memory, "run-b", "game-b"),
    ):
        for turn_id in (1, 2):
            _write_sample(
                memory,
                role="world",
                run_id=run_id,
                game_id=game_id,
                turn_id=turn_id,
            )
            _write_sample(
                memory,
                role="interest",
                run_id=run_id,
                game_id=game_id,
                turn_id=turn_id,
            )
            _write_sample(
                memory,
                role="agent",
                run_id=run_id,
                game_id=game_id,
                turn_id=turn_id,
            )

    manager = online_lora.OnlineLoRAManager(
        config=config,
        vllm_base_url="http://127.0.0.1:8000",
    )
    first = manager.register_game(
        state_memory=first_memory,
        run_id="run-a",
        game_id="game-a",
        reward_judge_model=_RewardJudge(
            {"fake-base": 0.2, "shared_world_v001": 0.8}
        ),
        activation_targets=first_targets,
    )
    second = manager.register_game(
        state_memory=second_memory,
        run_id="run-b",
        game_id="game-b",
        reward_judge_model=_RewardJudge(
            {"fake-base": 0.2, "shared_world_v001": 0.8}
        ),
        activation_targets=second_targets,
    )
    try:
        assert first is not None
        assert second is not None
        manager.maybe_schedule(real_turn_count=2)
        first.poll()
        second.poll()
    finally:
        manager.shutdown()

    assert loaded == ["shared_world_v001", "shared_interest_v001", "shared_agent_v001"]
    assert sft_samples == [("run-a", "game-a", 1), ("run-b", "game-b", 1)]
    assert first_targets["world"].activated == ["shared_world_v001"]
    assert second_targets["world"].activated == ["shared_world_v001"]
    first_rows = _lora_rows(first_memory.database.path)
    second_rows = _lora_rows(second_memory.database.path)
    assert ("world", "succeeded", "") in _row_statuses(first_rows)
    assert ("world", "succeeded", "") in _row_statuses(second_rows)
    assert first_rows[-1]["metadata"]["shared_batch"] is True
    assert first_memory.list_replay_samples(
        run_id="run-a",
        game_id="game-a",
        role="interest",
        ascending=True,
    )[0].reward == pytest.approx(0.6)
    assert second_memory.list_replay_samples(
        run_id="run-b",
        game_id="game-b",
        role="interest",
        ascending=True,
    )[0].reward == pytest.approx(0.6)


def test_same_gpu_vllm_suspend_pauses_noncontributors(
    tmp_path,
    monkeypatch,
) -> None:
    first_memory = StateMemory(SQLiteDatabase(tmp_path / "first.sqlite"))
    second_memory = StateMemory(SQLiteDatabase(tmp_path / "second.sqlite"))
    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        max_train_samples=1,
    )

    for turn_id in (1, 2):
        for role in ("world", "interest", "agent"):
            _write_sample(
                first_memory,
                role=role,
                run_id="run-a",
                game_id="game-a",
                turn_id=turn_id,
            )

    class _SuspendingVLLM:
        enabled = True

        def suspended(self):
            return nullcontext()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        online_lora,
        "_train_world_sft_adapter",
        lambda **kwargs: Path(kwargs["adapter_path"]).mkdir(parents=True),
    )
    monkeypatch.setattr(
        online_lora,
        "_train_grpo_adapter",
        lambda **kwargs: Path(kwargs["adapter_path"]).mkdir(parents=True),
    )
    monkeypatch.setattr(online_lora, "_chat_completion_content", _fake_chat)
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_load_lora_adapter",
        lambda self, target: None,
    )
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_unload_lora_adapter",
        lambda self, target: None,
    )

    manager = online_lora.OnlineLoRAManager(
        config=config,
        vllm_base_url="http://127.0.0.1:8000",
    )
    manager._vllm_server = _SuspendingVLLM()
    first = manager.register_game(
        state_memory=first_memory,
        run_id="run-a",
        game_id="game-a",
        reward_judge_model=_RewardJudge(
            {"fake-base": 0.2, "shared_world_v001": 0.8}
        ),
        activation_targets={
            "world": _ActivationTarget(),
            "interest": _ActivationTarget(),
            "agent": _ActivationTarget(),
        },
    )
    second = manager.register_game(
        state_memory=second_memory,
        run_id="run-b",
        game_id="game-b",
        activation_targets={
            "world": _ActivationTarget(),
            "interest": _ActivationTarget(),
            "agent": _ActivationTarget(),
        },
    )
    try:
        assert first is not None
        assert second is not None
        manager.maybe_schedule(real_turn_count=2)
        assert manager._games[first._handle_id].paused_update_index == 1
        assert manager._games[second._handle_id].paused_update_index == 1
        assert _lora_rows(second_memory.database.path) == []
    finally:
        manager.shutdown()


def test_shared_online_lora_unloads_previous_adapters_after_all_handles_ack(
    tmp_path,
    monkeypatch,
) -> None:
    first_memory = StateMemory(SQLiteDatabase(tmp_path / "first.sqlite"))
    second_memory = StateMemory(SQLiteDatabase(tmp_path / "second.sqlite"))
    manager = online_lora.OnlineLoRAManager(
        config=_config(
            tmp_path,
            min_new_samples_per_update=1,
            held_out_recent_samples=1,
        ),
        vllm_base_url="http://127.0.0.1:8000",
    )
    first_targets = {
        "world": _ActivationTarget(),
        "interest": _ActivationTarget(),
        "agent": _ActivationTarget(),
    }
    second_targets = {
        "world": _ActivationTarget(),
        "interest": _ActivationTarget(),
        "agent": _ActivationTarget(),
    }
    unloaded: list[str] = []
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_unload_lora_adapter",
        lambda self, target: unloaded.append(target.adapter_name),
    )
    first = manager.register_game(
        state_memory=first_memory,
        run_id="run-a",
        game_id="game-a",
        activation_targets=first_targets,
    )
    second = manager.register_game(
        state_memory=second_memory,
        run_id="run-b",
        game_id="game-b",
        activation_targets=second_targets,
    )
    assert first is not None
    assert second is not None
    manager._completed_update = online_lora.CompletedLoRAUpdate(
        update_index=2,
        completed_roles=tuple(
            _completed_shared_role(role, update_index=2)
            for role in ("world", "interest", "agent")
        ),
    )
    manager._pending_unload_targets[2] = tuple(
        _shared_target(role, update_index=1) for role in ("world", "interest", "agent")
    )

    try:
        first.poll()
        assert unloaded == []

        second.poll()
    finally:
        manager.shutdown()

    assert unloaded == [
        "shared_world_v001",
        "shared_interest_v001",
        "shared_agent_v001",
    ]


def test_shared_online_lora_max_wait_flushes_smaller_batch(
    tmp_path,
    monkeypatch,
) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    config = _config(
        tmp_path,
        min_new_samples_per_update=8,
        held_out_recent_samples=1,
        max_train_samples=8,
        max_update_wait_seconds=0.01,
    )
    for turn_id in (1, 2):
        _write_sample(memory, role="world", turn_id=turn_id)
        _write_sample(memory, role="interest", turn_id=turn_id)
        _write_sample(memory, role="agent", turn_id=turn_id)

    monkeypatch.setattr(
        online_lora,
        "_train_world_sft_adapter",
        lambda **kwargs: Path(kwargs["adapter_path"]).mkdir(parents=True),
    )
    monkeypatch.setattr(
        online_lora,
        "_train_grpo_adapter",
        lambda **kwargs: Path(kwargs["adapter_path"]).mkdir(parents=True),
    )
    monkeypatch.setattr(online_lora, "_chat_completion_content", _fake_chat)
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_load_lora_adapter",
        lambda self, target: None,
    )
    monkeypatch.setattr(
        online_lora.OnlineLoRAManager,
        "_unload_lora_adapter",
        lambda self, target: None,
    )
    manager = online_lora.OnlineLoRAManager(
        config=config,
        vllm_base_url="http://127.0.0.1:8000",
    )
    handle = manager.register_game(
        state_memory=memory,
        run_id="run-1",
        game_id="game-1",
        reward_judge_model=_RewardJudge(
            {"fake-base": 0.2, "shared_world_v001": 0.8}
        ),
        activation_targets={
            "world": _ActivationTarget(),
            "interest": _ActivationTarget(),
            "agent": _ActivationTarget(),
        },
    )
    try:
        assert handle is not None
        manager.maybe_schedule(real_turn_count=2)
        assert manager._future is None
        time.sleep(0.02)
        manager.maybe_schedule(real_turn_count=2)
        assert manager._future is not None
        handle.poll()
    finally:
        manager.shutdown()


def test_lora_config_supplies_target_modules(tmp_path) -> None:
    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        lora_target_modules=("q_proj", "v_proj"),
    )

    peft_config = online_lora._lora_config(config)

    assert set(peft_config.target_modules) == {"q_proj", "v_proj"}


def test_trainer_base_model_prefers_explicit_trainer_fields(tmp_path) -> None:
    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        trainer_base_model="trainable-base",
    )
    sample = _write_sample(
        StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite")),
        role="world",
        turn_id=1,
    )

    assert (
        online_lora._trainer_base_model(config=config, sample=sample)
        == "trainable-base"
    )


def test_trainer_model_kwargs_add_bnb_4bit_and_local_only(tmp_path) -> None:
    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        trainer_local_files_only=True,
        trainer_quantization="bnb_4bit",
        trainer_device_map="cuda:0",
    )

    processor_kwargs = online_lora._trainer_processor_kwargs(config)
    model_kwargs = online_lora._trainer_model_kwargs(config)

    assert processor_kwargs == {"local_files_only": True}
    assert model_kwargs["local_files_only"] is True
    assert model_kwargs["device_map"] == {"": "cuda:0"}
    assert model_kwargs["quantization_config"].load_in_4bit is True
    assert model_kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"


def test_bnb_4bit_prepares_model_for_kbit_training(tmp_path, monkeypatch) -> None:
    import peft

    config = _config(
        tmp_path,
        min_new_samples_per_update=1,
        held_out_recent_samples=1,
        trainer_quantization="bnb_4bit",
    )
    model = object()
    calls: list[Any] = []

    def fake_prepare(candidate: Any) -> str:
        calls.append(candidate)
        return "prepared-model"

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", fake_prepare)

    prepared = online_lora._prepare_model_for_quantized_training(
        model,
        config=config,
    )

    assert prepared == "prepared-model"
    assert calls == [model]


def test_grpo_prompt_messages_decode_image_url_blocks(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    sample = _write_sample(memory, role="agent", turn_id=1)

    messages = online_lora._grpo_prompt_messages(sample)

    user_content = messages[1]["content"]
    assert user_content[1]["type"] == "image"
    assert user_content[1]["image"].mode == "RGB"


class _ActivationTarget:
    def __init__(self) -> None:
        self.activated: list[str] = []

    def activate_lora_adapter(self, adapter_name: str) -> None:
        self.activated.append(adapter_name)


class _RewardJudge:
    def __init__(self, scores_by_model: dict[str, float]) -> None:
        self.scores_by_model = scores_by_model
        self.calls: list[Any] = []

    def judge_prediction(self, judge_input) -> RewardJudgeScore:
        self.calls.append(judge_input)
        model = str(judge_input.prediction.metadata["model"])
        return RewardJudgeScore(
            score=self.scores_by_model[model],
            notes=f"score for {model}",
        )


def _shared_target(role: str, *, update_index: int) -> online_lora.LoRAReloadTarget:
    sample_id = update_index * 10
    return online_lora.LoRAReloadTarget(
        role=role,
        update_index=update_index,
        adapter_name=f"shared_{role}_v{update_index:03d}",
        adapter_path=f"/tmp/shared/{role}/v{update_index:03d}",
        sample_ids=(sample_id,),
        eval_sample_ids=(),
        max_replay_sample_id=sample_id,
        sample_count=1,
    )


def _completed_shared_role(
    role: str,
    *,
    update_index: int,
) -> online_lora.CompletedLoRARole:
    target = _shared_target(role, update_index=update_index)
    return online_lora.CompletedLoRARole(
        target=target,
        metadata={
            "trained_sample_ids": target.sample_ids,
            "max_replay_sample_id": target.max_replay_sample_id,
            "sample_count": target.sample_count,
        },
    )


def _config(
    tmp_path,
    *,
    min_new_samples_per_update: int,
    held_out_recent_samples: int,
    **overrides: Any,
) -> OnlineLoRAConfig:
    config = OnlineLoRAConfig(
        enabled=True,
        base_model="fake-base",
        base_model_path="/local/base",
        update_interval_turns=4,
        min_new_samples_per_update=min_new_samples_per_update,
        held_out_recent_samples=held_out_recent_samples,
        adapter_root=str(tmp_path / "adapters"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _world_lp_metadata(*, turn_id: int) -> dict[str, Any]:
    return {
        "learning_progress_eval": {
            "action": {"action_id": "ACTION1", "data": None},
            "change_summary": f"changed {turn_id}",
            "previous_observation": _observation_json(f"prev-{turn_id}", turn_id),
            "current_observation": _observation_json(f"curr-{turn_id}", turn_id + 1),
            "candidate_index": 0,
            "request_model": "fake-base",
            "schema_name": "world_prediction",
        }
    }


def _write_sample(
    memory: StateMemory,
    *,
    role: str,
    turn_id: int,
    run_id: str = "run-1",
    game_id: str = "game-1",
    reward_components: dict[str, float] | None = None,
):
    target = {"predicted_change": f"changed {turn_id}"}
    if role == "agent":
        target = {"action": {"action_id": "ACTION1", "data": None}}
    elif role == "interest":
        target = {}
    metadata: dict[str, Any] = {
        "base_model": "fake-base",
        "base_model_path": "/local/base",
    }
    if role == "world":
        metadata.update(_world_lp_metadata(turn_id=turn_id))
    elif role == "interest":
        metadata["executed_candidate_index"] = 0
        metadata["label_components"] = {
            "lp_weight": 0.5,
            "goal_weight": 0.5,
            "goal_delta": 0.2,
            "progress_bonus": 0.1,
            "resource_cost": 0.0,
        }
        metadata["candidate_score_table"] = _candidate_score_table()
    else:
        metadata["reward_components"] = reward_components or {
            "lp_weight": 0.5,
            "goal_weight": 0.5,
            "goal_delta": 0.2,
            "progress_bonus": 0.1,
            "resource_cost": 0.0,
        }
        metadata["executed_candidate_index"] = 0
        metadata["candidate_score_table"] = _candidate_score_table()
    return memory.write_replay_sample(
        run_id=run_id,
        game_id=game_id,
        turn_id=turn_id,
        role=role,
        prompt={
            "request": {
                "model": "fake-base",
                "messages": [
                    {"role": "system", "content": f"{role} instructions"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{role} prompt for {game_id}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": _PNG_DATA_URL, "detail": "auto"},
                            },
                        ],
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": f"{role}_schema", "schema": {}},
                },
            }
        },
        completion={"target": target},
        reward=0.7,
        held_out=False,
        metadata=metadata,
    )


def _fake_chat(_base_url: str, request_payload: dict[str, Any]) -> str:
    schema_name = str(
        request_payload.get("response_format", {})
        .get("json_schema", {})
        .get("name", "")
    )
    if "interest" in schema_name:
        return json.dumps(
            {
                "candidate_values": [
                    {
                        "candidate_index": 0,
                        "expected_learning_progress": 0.8,
                        "expected_goal_delta": 0.2,
                        "confidence": 1.0,
                        "notes": "best",
                    },
                    {
                        "candidate_index": 1,
                        "expected_learning_progress": 0.1,
                        "expected_goal_delta": 0.0,
                        "confidence": 0.5,
                        "notes": "weak",
                    },
                ]
            }
        )
    return json.dumps({"predicted_change": f"{request_payload['model']} prediction"})


def _candidate_score_table() -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": 0,
            "action": {"action_id": "ACTION1", "data": None},
            "model_action": {"action_id": "ACTION1"},
            "action_name": "ACTION1",
            "expected_learning_progress": 0.3,
            "confidence": 1.0,
            "confidence_adjusted_learning_progress": 0.3,
            "expected_goal_delta": 0.2,
            "blended_score": 0.25,
            "notes": "first",
        },
        {
            "candidate_index": 1,
            "action": {"action_id": "ACTION2", "data": None},
            "model_action": {"action_id": "ACTION2"},
            "action_name": "ACTION2",
            "expected_learning_progress": 0.1,
            "confidence": 1.0,
            "confidence_adjusted_learning_progress": 0.1,
            "expected_goal_delta": 0.0,
            "blended_score": 0.05,
            "notes": "second",
        },
    ]


def _observation_json(observation_id: str, step: int) -> dict[str, Any]:
    return {
        "id": observation_id,
        "step": step,
        "frame": None,
        "frames": [],
        "raw_frame_data": None,
        "metadata": {},
    }


def _lora_rows(path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as connection:
        return [
            {
                "role": str(row[0]),
                "status": str(row[1]),
                "error": str(row[2]),
                "adapter_name": str(row[3]),
                "metadata": json.loads(str(row[4])),
            }
            for row in connection.execute(
                "SELECT role, status, error, adapter_name, metadata_json "
                "FROM lora_updates ORDER BY id"
            ).fetchall()
        ]


def _row_statuses(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    return [(row["role"], row["status"], row["error"]) for row in rows]
