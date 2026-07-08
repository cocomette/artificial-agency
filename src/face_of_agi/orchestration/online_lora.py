"""Periodic online LoRA update coordination for the v1 runtime."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import gc
from io import BytesIO
import json
from math import ceil
import os
from pathlib import Path
import signal
import shutil
from statistics import mean
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ReplaySampleRecord,
    WorldPrediction,
)
from face_of_agi.environment.config import OnlineLoRAConfig
from face_of_agi.frames import from_memory_jsonable
from face_of_agi.models.providers.hf_transformers import HFVLMEngine
from face_of_agi.memory import StateMemory
from face_of_agi.models.providers.vllm import vllm_exclusive_gate
from face_of_agi.models.reward_judge.contracts import RewardJudgeInput, RewardJudgeModel
from face_of_agi.runtime import timing as runtime_timing

TRAINABLE_ROLES = ("world", "interest", "agent")
_TRAINER_GATE_LOCK = threading.Lock()
_TRAINER_GATE: threading.BoundedSemaphore | None = None
_TRAINER_GATE_LIMIT: int | None = None


@dataclass(frozen=True, slots=True)
class LoRAReloadTarget:
    """vLLM adapter reload target."""

    role: str
    update_index: int
    adapter_name: str
    adapter_path: str
    sample_ids: tuple[int, ...]
    eval_sample_ids: tuple[int, ...]
    max_replay_sample_id: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class ReplaySampleBatch:
    """Train/eval split for one online update attempt."""

    train_samples: tuple[ReplaySampleRecord, ...]
    eval_samples: tuple[ReplaySampleRecord, ...]


@dataclass(frozen=True, slots=True)
class CompletedLoRARole:
    """One vLLM-loaded role adapter ready for local activation."""

    target: LoRAReloadTarget
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompletedLoRAUpdate:
    """Background result after the staged LoRA job finishes."""

    update_index: int
    completed_roles: tuple[CompletedLoRARole, ...] = ()


@dataclass(frozen=True, slots=True)
class HeldoutScore:
    """One World heldout prediction and Reward Judge score."""

    run_id: str
    game_id: str
    turn_id: int
    sample_id: int
    prediction: str
    score: float
    notes: str = ""
    error_tags: tuple[str, ...] = ()
    invalid_prediction: bool = False


@dataclass(frozen=True, slots=True)
class WorldLearningProgress:
    """World old/new scores and per-sample delayed learning progress."""

    metadata: dict[str, Any]
    learning_progress_by_sample_id: dict[int, float]
    learning_progress_by_turn_key: dict[tuple[str, str, int], float]

    @property
    def learning_progress_by_turn_id(self) -> dict[int, float]:
        """Compatibility view for single-game callers."""

        return {
            turn_id: value
            for (_run_id, _game_id, turn_id), value in (
                self.learning_progress_by_turn_key.items()
            )
        }


@dataclass(frozen=True, slots=True)
class ReplayTurnBundle:
    """Complete per-turn replay sample bundle across trainable roles."""

    run_id: str
    game_id: str
    turn_id: int
    samples_by_role: Mapping[str, ReplaySampleRecord]
    sort_key: tuple[str, str, str, int]


@dataclass(frozen=True, slots=True)
class SharedLoRABatch:
    """One shared update batch selected from registered games."""

    samples_by_role: Mapping[str, ReplaySampleBatch]
    contributor_keys: tuple[tuple[str, str], ...]
    per_game_batches: Mapping[tuple[str, str], Mapping[str, ReplaySampleBatch]]


@dataclass(slots=True)
class _RegisteredLoRAGame:
    """Mutable per-game online LoRA state owned by the shared manager."""

    handle_id: int
    run_id: str
    game_id: str
    state_memory: StateMemory
    reward_judge_model: RewardJudgeModel | None
    activation_targets: Mapping[str, Any]
    last_trained_sample_id: dict[str, int]
    applied_update_index: int = 0
    paused_update_index: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.run_id, self.game_id)


class OnlineLoRAGameHandle:
    """Per-game boundary for the shared online LoRA manager."""

    def __init__(self, *, manager: "OnlineLoRAManager", handle_id: int) -> None:
        self._manager = manager
        self._handle_id = handle_id

    def poll(self) -> None:
        """Apply ready shared adapters, blocking contributors when needed."""

        self._manager.poll_game(self._handle_id)

    def maybe_schedule(self, *, real_turn_count: int) -> None:
        """Ask the shared manager to schedule an update if a batch is ready."""

        self._manager.maybe_schedule(real_turn_count=real_turn_count)

    def shutdown(self) -> None:
        """Deregister the game after applying any required shared adapter."""

        self._manager.unregister_game(self._handle_id)


class OnlineLoRAManager:
    """Shared batched online LoRA manager for all games in one process."""

    def __init__(
        self,
        *,
        config: OnlineLoRAConfig,
        vllm_base_url: str,
        hf_engine: HFVLMEngine | None = None,
    ) -> None:
        self.config = config
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self._hf_engine = hf_engine
        self._trainer_gate = _process_trainer_gate(config.max_concurrent_trainer_jobs)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.RLock()
        self._future: Future[CompletedLoRAUpdate] | None = None
        self._update_index = 0
        self._next_handle_id = 1
        self._games: dict[int, _RegisteredLoRAGame] = {}
        self._games_by_key: dict[tuple[str, str], int] = {}
        self._active_adapter_paths: dict[str, str] = {}
        self._active_adapter_targets: dict[int, tuple[LoRAReloadTarget, ...]] = {}
        self._pending_unload_targets: dict[int, tuple[LoRAReloadTarget, ...]] = {}
        self._acknowledged_updates: dict[int, set[int]] = {}
        self._completed_update: CompletedLoRAUpdate | None = None
        self._completed_contributors: dict[int, set[tuple[str, str]]] = {}
        self._selected_train_samples: dict[
            tuple[int, str, tuple[str, str]], tuple[ReplaySampleRecord, ...]
        ] = {}
        self._fatal_error: BaseException | None = None
        self._first_ready_at: float | None = None
        self._vllm_server = (
            _RuntimeVLLMServerController(base_url=self.vllm_base_url)
            if hf_engine is not None
            else _RuntimeVLLMServerController.from_env(
                base_url=self.vllm_base_url,
            )
        )
        _validate_trainer_base_config(config)
        self._trainer_cache = (
            _WarmVLMTrainer(config)
            if (
                config.enabled
                and config.trainer_cache_enabled
                and hf_engine is None
            )
            else None
        )
        self._require_empty_shared_adapter_root()

    def register_game(
        self,
        *,
        state_memory: StateMemory | None,
        run_id: str,
        game_id: str,
        reward_judge_model: RewardJudgeModel | None = None,
        activation_targets: Mapping[str, Any] | None = None,
    ) -> OnlineLoRAGameHandle | None:
        """Register one game and return its frame-loop handle."""

        if not self.config.enabled:
            return None
        if state_memory is None:
            raise RuntimeError("online LoRA requires state memory")
        with self._lock:
            self._raise_if_fatal()
            key = (run_id, game_id)
            if key in self._games_by_key:
                raise RuntimeError(
                    f"online LoRA game already registered: run_id={run_id} "
                    f"game_id={game_id}"
                )
            handle_id = self._next_handle_id
            self._next_handle_id += 1
            self._games[handle_id] = _RegisteredLoRAGame(
                handle_id=handle_id,
                run_id=run_id,
                game_id=game_id,
                state_memory=state_memory,
                reward_judge_model=reward_judge_model,
                activation_targets=dict(activation_targets or {}),
                last_trained_sample_id={role: 0 for role in TRAINABLE_ROLES},
            )
            self._games_by_key[key] = handle_id
            update = self._completed_update
            if update is not None:
                self._acknowledged_updates.setdefault(update.update_index, set())
        return OnlineLoRAGameHandle(manager=self, handle_id=handle_id)

    def poll_game(self, handle_id: int) -> None:
        """Apply ready adapters at this game's turn boundary."""

        while True:
            with self._lock:
                self._raise_if_fatal()
                game = self._games.get(handle_id)
                if game is None:
                    return
                must_wait = (
                    game.paused_update_index is not None
                    and self._future is not None
                )
            if must_wait:
                self._finish_update(wait=True)
                continue

            self._finish_update(wait=False)
            with self._lock:
                self._raise_if_fatal()
                game = self._games.get(handle_id)
                if game is None:
                    return
                update = self._completed_update
                if update is None or update.update_index <= game.applied_update_index:
                    return
                self._apply_completed_update_to_game(game, update)
                return

    def maybe_schedule(self, *, real_turn_count: int) -> None:
        """Schedule a shared batch once enough complete turn bundles are ready."""

        if not self.config.enabled:
            return
        self._finish_update(wait=False)
        with self._lock:
            self._raise_if_fatal()
            if real_turn_count <= 0 or self._future is not None:
                return
            batch = self._ready_shared_batch(now=time.time())
            if batch is None:
                return

            self._update_index += 1
            update_index = self._update_index
            targets = {
                role: self._reload_target(
                    role,
                    batch.samples_by_role[role],
                    update_index=update_index,
                )
                for role in TRAINABLE_ROLES
            }
            self._completed_contributors[update_index] = set(batch.contributor_keys)
            for key, role_batches in batch.per_game_batches.items():
                game = self._game_for_key(key)
                game.paused_update_index = update_index
                for role in TRAINABLE_ROLES:
                    target = targets[role]
                    role_batch = role_batches[role]
                    self._selected_train_samples[
                        (update_index, role, key)
                    ] = role_batch.train_samples
                    self._write_update(
                        game,
                        role=role,
                        status="queued",
                        adapter_name=target.adapter_name,
                        adapter_path=target.adapter_path,
                        metadata={
                            **self._batch_metadata(role_batch),
                            "shared_update_index": update_index,
                            "shared_batch": True,
                        },
                    )
            if self._vllm_server.enabled:
                for game in self._games.values():
                    if game.applied_update_index < update_index:
                        game.paused_update_index = update_index
            self._future = self._executor.submit(
                self._run_update,
                update_index,
                batch.samples_by_role,
                batch.contributor_keys,
            )

    def unregister_game(self, handle_id: int) -> None:
        """Apply any required adapter and remove the game from acknowledgements."""

        self.poll_game(handle_id)
        with self._lock:
            game = self._games.pop(handle_id, None)
            if game is None:
                return
            self._games_by_key.pop(game.key, None)
            for acknowledged in self._acknowledged_updates.values():
                acknowledged.discard(handle_id)
            self._maybe_unload_acknowledged_targets()

    def shutdown(self) -> None:
        """Finish in-flight work and release trainer resources."""

        try:
            self._finish_update(wait=True)
            with self._lock:
                for game in tuple(self._games.values()):
                    update = self._completed_update
                    if update is not None and (
                        update.update_index > game.applied_update_index
                    ):
                        self._apply_completed_update_to_game(game, update)
                self._maybe_unload_acknowledged_targets()
        finally:
            if self._trainer_cache is not None:
                self._trainer_cache.close()
            if self._hf_engine is not None:
                close = getattr(self._hf_engine, "close", None)
                if callable(close):
                    close()
            self._vllm_server.close()
            self._executor.shutdown(wait=True)

    def _finish_update(self, *, wait: bool) -> None:
        with self._lock:
            future = self._future
            if future is None:
                return
            if not wait and not future.done():
                return
        try:
            result = future.result()
        except BaseException as exc:
            with self._lock:
                if self._future is future:
                    self._future = None
                self._fatal_error = exc
            raise
        with self._lock:
            if self._future is future:
                self._future = None
                self._completed_update = result
                self._acknowledged_updates.setdefault(result.update_index, set())
                self._active_adapter_targets[result.update_index] = tuple(
                    completed.target for completed in result.completed_roles
                )

    def _run_update(
        self,
        update_index: int,
        samples_by_role: Mapping[str, ReplaySampleBatch],
        contributor_keys: Sequence[tuple[str, str]],
    ) -> CompletedLoRAUpdate:
        """Run one shared staged trainer/reload pipeline off game threads."""

        targets = {
            role: self._reload_target(
                role,
                samples_by_role[role],
                update_index=update_index,
            )
            for role in TRAINABLE_ROLES
        }
        loaded_targets: list[LoRAReloadTarget] = []
        completed: list[CompletedLoRARole] = []
        current_role = "world"
        try:
            previous_loaded = self._active_adapter_targets.get(update_index - 1, ())
            if previous_loaded and not self._vllm_server.enabled:
                self._pending_unload_targets[update_index] = previous_loaded

            world_batch = samples_by_role["world"]
            world_target = targets["world"]
            old_world_model = self._active_request_model("world")
            old_train_scores = self._score_world_eval_samples(
                world_batch.train_samples,
                model=old_world_model,
            )
            old_eval_scores = self._score_world_eval_samples(
                world_batch.eval_samples,
                model=old_world_model,
            )
            self._run_trainer(
                role="world",
                target=world_target,
                contributor_keys=contributor_keys,
                samples_by_role=samples_by_role,
                train=lambda: self._train_world_adapter(
                    target=world_target,
                    samples=world_batch.train_samples,
                ),
            )
            self._load_lora_adapter(world_target)
            loaded_targets.append(world_target)
            new_train_scores = self._score_world_eval_samples(
                world_batch.train_samples,
                model=world_target.adapter_name,
            )
            new_eval_scores = self._score_world_eval_samples(
                world_batch.eval_samples,
                model=world_target.adapter_name,
            )
            world_lp = self._world_learning_progress(
                target=world_target,
                old_model=old_world_model,
                old_train_scores=old_train_scores,
                new_train_scores=new_train_scores,
                old_eval_scores=old_eval_scores,
                new_eval_scores=new_eval_scores,
                train_samples=world_batch.train_samples,
            )
            completed.append(
                CompletedLoRARole(
                    target=world_target,
                    metadata={
                        **self._target_metadata(world_target),
                        **world_lp.metadata,
                    },
                )
            )

            current_role = "interest"
            interest_batch = samples_by_role["interest"]
            interest_target = targets["interest"]
            interest_samples = self._backfill_interest_labels(
                interest_batch,
                learning_progress_by_turn_key=world_lp.learning_progress_by_turn_key,
                learning_progress_metadata=world_lp.metadata,
                world_target=world_target,
            )
            self._run_trainer(
                role="interest",
                target=interest_target,
                contributor_keys=contributor_keys,
                samples_by_role=samples_by_role,
                loaded_targets=loaded_targets,
                train=lambda: self._train_grpo_role_adapter(
                    role="interest",
                    target=interest_target,
                    samples=interest_samples,
                ),
            )
            self._load_lora_adapter(interest_target)
            loaded_targets.append(interest_target)
            completed.append(
                CompletedLoRARole(
                    target=interest_target,
                    metadata={
                        **self._target_metadata(interest_target),
                        "world_learning_progress": world_lp.metadata,
                    },
                )
            )

            current_role = "agent"
            agent_batch = samples_by_role["agent"]
            agent_target = targets["agent"]
            agent_samples = self._backfill_agent_rewards(
                agent_batch,
                interest_samples=interest_samples,
                learning_progress_by_turn_key=world_lp.learning_progress_by_turn_key,
                learning_progress_metadata=world_lp.metadata,
                world_target=world_target,
                interest_target=interest_target,
            )
            self._run_trainer(
                role="agent",
                target=agent_target,
                contributor_keys=contributor_keys,
                samples_by_role=samples_by_role,
                loaded_targets=loaded_targets,
                train=lambda: self._train_grpo_role_adapter(
                    role="agent",
                    target=agent_target,
                    samples=agent_samples,
                ),
            )
            self._load_lora_adapter(agent_target)
            loaded_targets.append(agent_target)
            completed.append(
                CompletedLoRARole(
                    target=agent_target,
                    metadata={
                        **self._target_metadata(agent_target),
                        "world_learning_progress": world_lp.metadata,
                        "interest_versioned_adapter_name": interest_target.adapter_name,
                        "interest_trained_sample_ids": interest_target.sample_ids,
                    },
                )
            )
            return CompletedLoRAUpdate(
                update_index=update_index,
                completed_roles=tuple(completed),
            )
        except Exception as exc:
            self._record_staged_failure(
                failed_role=current_role,
                error=exc,
                targets=targets,
                samples_by_role=samples_by_role,
                contributor_keys=contributor_keys,
            )
            try:
                self._cleanup_staged_update(
                    loaded_targets=loaded_targets,
                    targets=tuple(targets.values()),
                )
            except Exception as cleanup_exc:
                raise RuntimeError(
                    f"{current_role} update failed: {exc}; cleanup failed: "
                    f"{cleanup_exc}"
                ) from exc
            raise

    def _train_world_adapter(
        self,
        *,
        target: LoRAReloadTarget,
        samples: Sequence[ReplaySampleRecord],
    ) -> None:
        if self._hf_engine is not None:
            _train_world_sft_adapter_hf(
                engine=self._hf_engine,
                adapter_path=Path(target.adapter_path),
                samples=samples,
                config=self.config,
                previous_adapter_path=self._active_adapter_paths.get("world"),
                adapter_name=target.adapter_name,
            )
            return
        _train_world_sft_adapter(
            adapter_path=Path(target.adapter_path),
            samples=samples,
            config=self.config,
            previous_adapter_path=self._active_adapter_paths.get("world"),
            trainer_cache=self._trainer_cache,
            adapter_name=target.adapter_name,
        )

    def _train_grpo_role_adapter(
        self,
        *,
        role: str,
        target: LoRAReloadTarget,
        samples: Sequence[ReplaySampleRecord],
    ) -> None:
        if self._hf_engine is not None:
            _train_grpo_adapter_hf(
                engine=self._hf_engine,
                role=role,
                adapter_path=Path(target.adapter_path),
                samples=samples,
                config=self.config,
                previous_adapter_path=self._active_adapter_paths.get(role),
                adapter_name=target.adapter_name,
            )
            return
        _train_grpo_adapter(
            role=role,
            adapter_path=Path(target.adapter_path),
            samples=samples,
            config=self.config,
            reward_judge_model=None,
            previous_adapter_path=self._active_adapter_paths.get(role),
            trainer_cache=self._trainer_cache,
            adapter_name=target.adapter_name,
        )

    def _apply_completed_update_to_game(
        self,
        game: _RegisteredLoRAGame,
        update: CompletedLoRAUpdate,
    ) -> None:
        completed_roles = tuple(update.completed_roles)
        if {completed.target.role for completed in completed_roles} != set(
            TRAINABLE_ROLES
        ):
            raise RuntimeError("completed LoRA update must include every trainable role")
        previous = {
            role: _active_lora_adapter_name(game.activation_targets.get(role))
            for role in TRAINABLE_ROLES
        }
        try:
            for completed in completed_roles:
                self._activate_lora_adapter(
                    game,
                    completed.target.role,
                    completed.target.adapter_name,
                )
        except Exception as exc:
            for role, adapter_name in previous.items():
                self._restore_lora_adapter(game, role, adapter_name)
            self._fatal_error = RuntimeError(f"local activation failed: {exc}")
            raise self._fatal_error

        is_contributor = game.key in self._completed_contributors.get(
            update.update_index,
            set(),
        )
        for completed in completed_roles:
            target = completed.target
            self._active_adapter_paths[target.role] = target.adapter_path
            metadata = dict(completed.metadata)
            metadata["local_activation_status"] = "activated"
            if is_contributor:
                trained_samples = self._selected_train_samples.get(
                    (update.update_index, target.role, game.key),
                    (),
                )
                if trained_samples:
                    game.last_trained_sample_id[target.role] = max(
                        sample.id for sample in trained_samples
                    )
                self._write_update(
                    game,
                    role=target.role,
                    status="succeeded",
                    adapter_name=target.adapter_name,
                    adapter_path=target.adapter_path,
                    metadata={
                        **metadata,
                        "shared_update_index": update.update_index,
                        "shared_batch": True,
                    },
                )
        game.applied_update_index = update.update_index
        if game.paused_update_index == update.update_index:
            game.paused_update_index = None
        self._acknowledged_updates.setdefault(update.update_index, set()).add(
            game.handle_id
        )
        self._maybe_unload_acknowledged_targets()

    def _activate_lora_adapter(
        self,
        game: _RegisteredLoRAGame,
        role: str,
        adapter_name: str,
    ) -> None:
        target = game.activation_targets.get(role)
        if target is None:
            raise RuntimeError(f"no LoRA activation target configured for {role}")
        activate = getattr(target, "activate_lora_adapter", None)
        if not callable(activate):
            raise RuntimeError(f"{role} model does not support LoRA activation")
        activate(adapter_name)

    def _restore_lora_adapter(
        self,
        game: _RegisteredLoRAGame,
        role: str,
        adapter_name: str | None,
    ) -> None:
        target = game.activation_targets.get(role)
        if target is None:
            return
        if adapter_name:
            self._activate_lora_adapter(game, role, adapter_name)
            return
        provider = getattr(target, "provider", target)
        if hasattr(provider, "active_lora_adapter_name"):
            setattr(provider, "active_lora_adapter_name", None)

    def _load_lora_adapter(self, target: LoRAReloadTarget) -> None:
        if self._hf_engine is not None:
            self._hf_engine.load_adapter(
                adapter_name=target.adapter_name,
                adapter_path=target.adapter_path,
            )
            return
        payload = json.dumps(
            {
                "lora_name": target.adapter_name,
                "lora_path": target.adapter_path,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.vllm_base_url}/load_lora_adapter",
            data=payload,
            headers=_vllm_json_headers(),
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"vLLM LoRA reload failed for {target.adapter_name}: "
                    f"HTTP {response.status}"
                )

    def _unload_lora_adapter(self, target: LoRAReloadTarget) -> None:
        if self._hf_engine is not None:
            self._hf_engine.delete_adapter(target.adapter_name)
            return
        payload = json.dumps({"lora_name": target.adapter_name}).encode("utf-8")
        request = Request(
            f"{self.vllm_base_url}/unload_lora_adapter",
            data=payload,
            headers=_vllm_json_headers(),
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"vLLM LoRA unload failed for {target.adapter_name}: "
                    f"HTTP {response.status}"
                )

    def _world_learning_progress(
        self,
        *,
        target: LoRAReloadTarget,
        old_model: str,
        old_train_scores: Sequence[HeldoutScore],
        new_train_scores: Sequence[HeldoutScore],
        old_eval_scores: Sequence[HeldoutScore],
        new_eval_scores: Sequence[HeldoutScore],
        train_samples: Sequence[ReplaySampleRecord],
    ) -> WorldLearningProgress:
        old_train_by_key = {_score_key(score): score.score for score in old_train_scores}
        train_deltas_by_id: dict[int, float] = {}
        train_deltas_by_turn_key: dict[tuple[str, str, int], float] = {}
        for score in new_train_scores:
            score_key = _score_key(score)
            if score_key not in old_train_by_key:
                continue
            clipped_delta = _clip(
                score.score - old_train_by_key[score_key],
                minimum=self.config.learning_progress_min,
                maximum=self.config.learning_progress_max,
            )
            train_deltas_by_id[score.sample_id] = clipped_delta
            train_deltas_by_turn_key[
                (score.run_id, score.game_id, score.turn_id)
            ] = clipped_delta
        for sample in train_samples:
            if sample.id not in train_deltas_by_id:
                raise RuntimeError(
                    f"World replay sample {sample.id} missing per-sample LP score"
                )

        old_eval_by_key = {_score_key(score): score.score for score in old_eval_scores}
        eval_deltas = [
            score.score - old_eval_by_key[_score_key(score)]
            for score in new_eval_scores
            if _score_key(score) in old_eval_by_key
        ]
        raw_heldout_learning_progress = mean(eval_deltas) if eval_deltas else 0.0
        heldout_learning_progress = _clip(
            raw_heldout_learning_progress,
            minimum=self.config.learning_progress_min,
            maximum=self.config.learning_progress_max,
        )
        metadata = {
            "heldout_learning_progress": heldout_learning_progress,
            "raw_heldout_learning_progress": raw_heldout_learning_progress,
            "learning_progress_min": self.config.learning_progress_min,
            "learning_progress_max": self.config.learning_progress_max,
            "old_world_model": old_model,
            "new_world_model": target.adapter_name,
            "old_heldout_mean_score": _mean_score(old_eval_scores),
            "new_heldout_mean_score": _mean_score(new_eval_scores),
            "heldout_score_delta": eval_deltas,
            "old_heldout_scores": [
                _heldout_score_metadata(score) for score in old_eval_scores
            ],
            "new_heldout_scores": [
                _heldout_score_metadata(score) for score in new_eval_scores
            ],
            "per_sample_learning_progress": dict(sorted(train_deltas_by_id.items())),
            "per_turn_learning_progress": {
                _turn_key_text(key): value
                for key, value in sorted(train_deltas_by_turn_key.items())
            },
            "reload_status": "loaded_pending_atomic_local_activation",
        }
        return WorldLearningProgress(
            metadata=metadata,
            learning_progress_by_sample_id=train_deltas_by_id,
            learning_progress_by_turn_key=train_deltas_by_turn_key,
        )

    def _score_world_eval_samples(
        self,
        samples: Sequence[ReplaySampleRecord],
        *,
        model: str,
    ) -> tuple[HeldoutScore, ...]:
        return tuple(
            self._score_world_eval_sample(sample, model=model)
            for sample in samples
        )

    def _score_world_eval_sample(
        self,
        sample: ReplaySampleRecord,
        *,
        model: str,
    ) -> HeldoutScore:
        prediction_text = self._predict_world_from_replay(sample, model=model)
        payload = _parse_completion_json(prediction_text)
        predicted_change = ""
        if payload is not None and isinstance(payload.get("predicted_change"), str):
            predicted_change = str(payload["predicted_change"]).strip()
        if not predicted_change:
            return HeldoutScore(
                run_id=sample.run_id,
                game_id=sample.game_id,
                turn_id=sample.turn_id,
                sample_id=sample.id,
                prediction=prediction_text,
                score=0.0,
                notes="invalid World JSON prediction",
                invalid_prediction=True,
            )
        judge_input = self._world_judge_input(
            sample,
            model=model,
            predicted_change=predicted_change,
        )
        judge_model = self._reward_judge_for_sample(sample)
        score = judge_model.judge_prediction(judge_input)
        return HeldoutScore(
            run_id=sample.run_id,
            game_id=sample.game_id,
            turn_id=sample.turn_id,
            sample_id=sample.id,
            prediction=predicted_change,
            score=score.score,
            notes=score.notes,
            error_tags=score.error_tags,
        )

    def _predict_world_from_replay(
        self,
        sample: ReplaySampleRecord,
        *,
        model: str,
    ) -> str:
        request_payload = deepcopy(_sample_request(sample))
        request_payload["model"] = model
        if self._hf_engine is not None:
            return _hf_chat_completion_content(self._hf_engine, request_payload)
        return _chat_completion_content(self.vllm_base_url, request_payload)

    def _world_judge_input(
        self,
        sample: ReplaySampleRecord,
        *,
        model: str,
        predicted_change: str,
    ) -> RewardJudgeInput:
        return _world_judge_input_from_bundle(
            bundle=_learning_progress_eval_bundle(sample),
            run_id=sample.run_id,
            game_id=sample.game_id,
            turn_id=sample.turn_id,
            sample_id=sample.id,
            model=model,
            predicted_change=predicted_change,
            reward_metadata={"learning_progress_eval": True},
        )

    def _reward_judge_for_sample(
        self,
        sample: ReplaySampleRecord,
    ) -> RewardJudgeModel:
        model = self._game_for_key((sample.run_id, sample.game_id)).reward_judge_model
        if model is None:
            raise RuntimeError("World LP evaluation requires Reward Judge")
        return model

    def _backfill_interest_labels(
        self,
        batch: ReplaySampleBatch,
        *,
        learning_progress_by_turn_key: Mapping[tuple[str, str, int], float],
        learning_progress_metadata: Mapping[str, Any],
        world_target: LoRAReloadTarget,
    ) -> tuple[ReplaySampleRecord, ...]:
        updated: list[ReplaySampleRecord] = []
        for sample in batch.train_samples:
            label_components = sample.metadata.get("label_components")
            if not isinstance(label_components, dict):
                raise RuntimeError(
                    f"interest replay sample {sample.id} missing label_components"
                )
            executed_candidate_index = sample.metadata.get("executed_candidate_index")
            if isinstance(executed_candidate_index, bool) or not isinstance(
                executed_candidate_index,
                int,
            ):
                raise RuntimeError(
                    f"interest replay sample {sample.id} missing "
                    "executed_candidate_index"
                )
            turn_key = _sample_turn_key(sample)
            if turn_key not in learning_progress_by_turn_key:
                raise RuntimeError(
                    f"interest replay sample {sample.id} missing delayed LP "
                    f"for turn {sample.turn_id}"
                )
            learning_progress = float(learning_progress_by_turn_key[turn_key])
            goal_delta = _numeric_component(label_components, "goal_delta")
            metadata = dict(sample.metadata)
            metadata["interest_label"] = {
                "executed_candidate_index": executed_candidate_index,
                "realized_learning_progress": learning_progress,
                "realized_goal_delta": goal_delta,
                "label_source_update_index": world_target.update_index,
                "world_update_sample_ids": world_target.sample_ids,
                "world_eval_sample_ids": world_target.eval_sample_ids,
                "old_heldout_mean_score": learning_progress_metadata[
                    "old_heldout_mean_score"
                ],
                "new_heldout_mean_score": learning_progress_metadata[
                    "new_heldout_mean_score"
                ],
                "lp_scope": "per_executed_turn",
            }
            updated.append(
                self._state_memory_for_sample(
                    sample
                ).update_replay_sample_reward_metadata(
                    sample_id=sample.id,
                    reward=learning_progress,
                    metadata=metadata,
                )
            )
        return tuple(updated)

    def _backfill_agent_rewards(
        self,
        batch: ReplaySampleBatch,
        *,
        interest_samples: Sequence[ReplaySampleRecord],
        learning_progress_by_turn_key: Mapping[tuple[str, str, int], float],
        learning_progress_metadata: Mapping[str, Any],
        world_target: LoRAReloadTarget,
        interest_target: LoRAReloadTarget,
    ) -> tuple[ReplaySampleRecord, ...]:
        interest_by_turn = {
            _sample_turn_key(sample): sample for sample in interest_samples
        }
        updated: list[ReplaySampleRecord] = []
        for sample in batch.train_samples:
            components = sample.metadata.get("reward_components")
            if not isinstance(components, dict):
                raise RuntimeError(
                    f"agent replay sample {sample.id} missing reward_components"
                )
            lp_weight = _numeric_component(components, "lp_weight")
            goal_weight = _numeric_component(components, "goal_weight")
            goal_delta = _numeric_component(components, "goal_delta")
            progress_bonus = _numeric_component(components, "progress_bonus")
            resource_cost = _numeric_component_default(
                components,
                "resource_cost",
                default=0.0,
            )
            turn_key = _sample_turn_key(sample)
            if turn_key not in learning_progress_by_turn_key:
                raise RuntimeError(
                    f"agent replay sample {sample.id} missing delayed LP "
                    f"for turn {sample.turn_id}"
                )
            learning_progress = float(learning_progress_by_turn_key[turn_key])
            reward = (
                lp_weight * learning_progress
                + goal_weight * goal_delta
                + progress_bonus
                - resource_cost
            )
            interest_sample = interest_by_turn.get(turn_key)
            if interest_sample is None:
                raise RuntimeError(
                    f"agent replay sample {sample.id} missing paired Interest sample"
                )
            candidate_score_table = self._rescore_interest_sample(
                interest_sample,
                model=interest_target.adapter_name,
            )
            metadata = dict(sample.metadata)
            metadata["candidate_score_table"] = candidate_score_table
            metadata["candidate_score_table_source"] = {
                "interest_replay_sample_id": interest_sample.id,
                "interest_versioned_adapter_name": interest_target.adapter_name,
                "interest_adapter_sample_ids": interest_target.sample_ids,
            }
            metadata["learning_progress_backfill"] = {
                "learning_progress": learning_progress,
                "lp_weight": lp_weight,
                "goal_weight": goal_weight,
                "goal_delta": goal_delta,
                "progress_bonus": progress_bonus,
                "resource_cost": resource_cost,
                "computed_reward": reward,
                "world_trained_sample_ids": world_target.sample_ids,
                "world_eval_sample_ids": world_target.eval_sample_ids,
                "interest_trained_sample_ids": interest_target.sample_ids,
                "old_heldout_mean_score": learning_progress_metadata[
                    "old_heldout_mean_score"
                ],
                "new_heldout_mean_score": learning_progress_metadata[
                    "new_heldout_mean_score"
                ],
                "lp_scope": "per_executed_turn",
            }
            updated.append(
                self._state_memory_for_sample(
                    sample
                ).update_replay_sample_reward_metadata(
                    sample_id=sample.id,
                    reward=reward,
                    metadata=metadata,
                )
            )
        return tuple(updated)

    def _rescore_interest_sample(
        self,
        sample: ReplaySampleRecord,
        *,
        model: str,
    ) -> tuple[dict[str, Any], ...]:
        prediction_text = self._predict_interest_from_replay(sample, model=model)
        payload = _parse_completion_json(prediction_text)
        if payload is None:
            raise RuntimeError(
                f"Interest replay sample {sample.id} produced invalid JSON"
            )
        return _candidate_score_table_from_interest_payload(sample, payload)

    def _predict_interest_from_replay(
        self,
        sample: ReplaySampleRecord,
        *,
        model: str,
    ) -> str:
        request_payload = deepcopy(_sample_request(sample))
        request_payload["model"] = model
        if self._hf_engine is not None:
            return _hf_chat_completion_content(self._hf_engine, request_payload)
        return _chat_completion_content(self.vllm_base_url, request_payload)

    def _record_staged_failure(
        self,
        *,
        failed_role: str,
        error: Exception,
        targets: Mapping[str, LoRAReloadTarget],
        samples_by_role: Mapping[str, ReplaySampleBatch],
        contributor_keys: Sequence[tuple[str, str]],
    ) -> None:
        error_text = f"{failed_role} update failed: {error}"
        for key in contributor_keys:
            game = self._game_for_key(key)
            for role in TRAINABLE_ROLES:
                target = targets[role]
                role_batch = self._batch_for_game(samples_by_role[role], key)
                self._write_update(
                    game,
                    role=role,
                    status="failed",
                    adapter_name=target.adapter_name,
                    adapter_path=target.adapter_path,
                    error=error_text,
                    metadata={
                        **self._batch_metadata(role_batch),
                        "failed_role": failed_role,
                        "shared_update_index": target.update_index,
                        "shared_batch": True,
                    },
                )

    def _cleanup_staged_update(
        self,
        *,
        loaded_targets: Sequence[LoRAReloadTarget],
        targets: Sequence[LoRAReloadTarget],
    ) -> None:
        cleanup_errors: list[str] = []
        for target in loaded_targets:
            try:
                self._unload_lora_adapter(target)
            except Exception as exc:
                cleanup_errors.append(f"unload {target.adapter_name} failed: {exc}")
        for target in targets:
            path = Path(target.adapter_path)
            if not path.exists():
                continue
            try:
                shutil.rmtree(path)
            except Exception as exc:
                cleanup_errors.append(f"delete {path} failed: {exc}")
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))

    def _active_request_model(self, role: str) -> str:
        completed = self._completed_update
        if completed is not None:
            for completed_role in completed.completed_roles:
                if completed_role.target.role == role:
                    return completed_role.target.adapter_name
        if self.config.base_model:
            return self.config.base_model
        raise RuntimeError(f"no active or base model configured for {role}")

    def _run_trainer(
        self,
        *,
        role: str,
        target: LoRAReloadTarget,
        contributor_keys: Sequence[tuple[str, str]],
        samples_by_role: Mapping[str, ReplaySampleBatch],
        train: Callable[[], None],
        loaded_targets: Sequence[LoRAReloadTarget] = (),
    ) -> None:
        self._trainer_gate.acquire()
        try:
            for key in contributor_keys:
                game = self._game_for_key(key)
                role_batch = self._batch_for_game(samples_by_role[role], key)
                self._write_update(
                    game,
                    role=role,
                    status="running",
                    adapter_name=target.adapter_name,
                    adapter_path=target.adapter_path,
                    metadata={
                        **self._batch_metadata(role_batch),
                        "shared_update_index": target.update_index,
                        "shared_batch": True,
                    },
                )
            if self._vllm_server.enabled:
                with vllm_exclusive_gate():
                    with self._vllm_server.suspended():
                        try:
                            train()
                        finally:
                            if self._trainer_cache is not None:
                                self._trainer_cache.close()
                            _release_trainer_cuda_memory()
                for loaded_target in loaded_targets:
                    self._load_lora_adapter(loaded_target)
            else:
                train()
        finally:
            self._trainer_gate.release()

    def _write_update(
        self,
        game: _RegisteredLoRAGame,
        *,
        role: str,
        status: str,
        adapter_name: str,
        adapter_path: str,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        update_metadata = {"timestamp": time.time()}
        update_metadata.update(metadata or {})
        game.state_memory.write_lora_update(
            run_id=game.run_id,
            game_id=game.game_id,
            update_index=self._update_index,
            role=role,
            status=status,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            error=error,
            metadata=update_metadata,
        )

    def _reload_target(
        self,
        role: str,
        batch: ReplaySampleBatch,
        *,
        update_index: int,
    ) -> LoRAReloadTarget:
        return LoRAReloadTarget(
            role=role,
            update_index=update_index,
            adapter_name=self._adapter_name(role, update_index=update_index),
            adapter_path=str(self._adapter_path(role, update_index=update_index)),
            sample_ids=tuple(sample.id for sample in batch.train_samples),
            eval_sample_ids=tuple(sample.id for sample in batch.eval_samples),
            max_replay_sample_id=max(sample.id for sample in batch.train_samples),
            sample_count=len(batch.train_samples),
        )

    def _adapter_name(self, role: str, *, update_index: int) -> str:
        return f"shared_{role}_v{update_index:03d}"

    def _adapter_path(self, role: str, *, update_index: int | None = None) -> Path:
        if update_index is None:
            update_index = self._update_index
        return self._shared_adapter_root() / role / f"v{update_index:03d}"

    def _shared_adapter_root(self) -> Path:
        return Path(self.config.adapter_root) / "shared"

    def _ready_shared_batch(self, *, now: float) -> SharedLoRABatch | None:
        threshold = self.config.min_new_samples_per_update
        if threshold is None:
            threshold = self.config.update_interval_turns
        reserve = self.config.held_out_recent_samples
        train_bundles: list[ReplayTurnBundle] = []
        eval_bundles_by_game: dict[tuple[str, str], tuple[ReplayTurnBundle, ...]] = {}
        for game in self._games.values():
            bundles = self._complete_turn_bundles(game)
            if len(bundles) <= reserve:
                continue
            train_bundles.extend(bundles[:-reserve])
            eval_bundles_by_game[game.key] = bundles[-reserve:]
        if not train_bundles:
            self._first_ready_at = None
            return None
        if self._first_ready_at is None:
            self._first_ready_at = now
        wait_elapsed = now - self._first_ready_at >= self.config.max_update_wait_seconds
        if len(train_bundles) < threshold and not wait_elapsed:
            return None
        selected = tuple(
            sorted(train_bundles, key=lambda bundle: bundle.sort_key)[
                : self.config.max_train_samples
            ]
        )
        contributor_keys = tuple(
            sorted({(bundle.run_id, bundle.game_id) for bundle in selected})
        )
        samples_by_role: dict[str, ReplaySampleBatch] = {}
        for role in TRAINABLE_ROLES:
            samples_by_role[role] = ReplaySampleBatch(
                train_samples=tuple(
                    bundle.samples_by_role[role] for bundle in selected
                ),
                eval_samples=tuple(
                    bundle.samples_by_role[role]
                    for key in contributor_keys
                    for bundle in eval_bundles_by_game[key]
                ),
            )
        per_game_batches: dict[
            tuple[str, str], dict[str, ReplaySampleBatch]
        ] = {}
        for key in contributor_keys:
            selected_for_game = tuple(
                bundle
                for bundle in selected
                if (bundle.run_id, bundle.game_id) == key
            )
            per_game_batches[key] = {}
            for role in TRAINABLE_ROLES:
                per_game_batches[key][role] = ReplaySampleBatch(
                    train_samples=tuple(
                        bundle.samples_by_role[role] for bundle in selected_for_game
                    ),
                    eval_samples=tuple(
                        bundle.samples_by_role[role]
                        for bundle in eval_bundles_by_game[key]
                    ),
                )
        self._first_ready_at = None
        return SharedLoRABatch(
            samples_by_role=samples_by_role,
            contributor_keys=contributor_keys,
            per_game_batches=per_game_batches,
        )

    def _complete_turn_bundles(
        self,
        game: _RegisteredLoRAGame,
    ) -> tuple[ReplayTurnBundle, ...]:
        by_role: dict[str, dict[int, ReplaySampleRecord]] = {}
        for role in TRAINABLE_ROLES:
            samples = tuple(
                game.state_memory.list_replay_samples(
                    run_id=game.run_id,
                    game_id=game.game_id,
                    role=role,
                    held_out=False,
                    after_id=game.last_trained_sample_id[role],
                    ascending=True,
                )
            )
            by_role[role] = {sample.turn_id: sample for sample in samples}
        turn_ids = set(by_role["world"])
        for role in TRAINABLE_ROLES[1:]:
            turn_ids &= set(by_role[role])
        bundles: list[ReplayTurnBundle] = []
        for turn_id in sorted(turn_ids):
            samples_by_role = {
                role: by_role[role][turn_id] for role in TRAINABLE_ROLES
            }
            world_sample = samples_by_role["world"]
            bundles.append(
                ReplayTurnBundle(
                    run_id=game.run_id,
                    game_id=game.game_id,
                    turn_id=turn_id,
                    samples_by_role=samples_by_role,
                    sort_key=(
                        world_sample.created_at,
                        game.run_id,
                        game.game_id,
                        turn_id,
                    ),
                )
            )
        return tuple(bundles)

    def _batch_metadata(self, batch: ReplaySampleBatch) -> dict[str, Any]:
        sample_ids = tuple(int(sample.id) for sample in batch.train_samples)
        eval_sample_ids = tuple(int(sample.id) for sample in batch.eval_samples)
        return {
            "trained_sample_ids": sample_ids,
            "trained_sample_refs": tuple(_sample_ref(sample) for sample in batch.train_samples),
            "max_replay_sample_id": max(sample_ids) if sample_ids else 0,
            "sample_count": len(sample_ids),
            "eval_sample_ids": eval_sample_ids,
            "eval_sample_refs": tuple(_sample_ref(sample) for sample in batch.eval_samples),
            "eval_sample_count": len(eval_sample_ids),
            "held_out_recent_samples": self.config.held_out_recent_samples,
            "rolling_reserve": True,
        }

    def _target_metadata(self, target: LoRAReloadTarget) -> dict[str, Any]:
        return {
            "update_index": target.update_index,
            "trained_sample_ids": target.sample_ids,
            "max_replay_sample_id": target.max_replay_sample_id,
            "sample_count": target.sample_count,
            "eval_sample_ids": target.eval_sample_ids,
            "eval_sample_count": len(target.eval_sample_ids),
            "held_out_recent_samples": self.config.held_out_recent_samples,
            "rolling_reserve": True,
        }

    def _batch_for_game(
        self,
        batch: ReplaySampleBatch,
        key: tuple[str, str],
    ) -> ReplaySampleBatch:
        return ReplaySampleBatch(
            train_samples=tuple(
                sample
                for sample in batch.train_samples
                if (sample.run_id, sample.game_id) == key
            ),
            eval_samples=tuple(
                sample
                for sample in batch.eval_samples
                if (sample.run_id, sample.game_id) == key
            ),
        )

    def _game_for_key(self, key: tuple[str, str]) -> _RegisteredLoRAGame:
        handle_id = self._games_by_key.get(key)
        if handle_id is None:
            raise RuntimeError(f"online LoRA game is not registered: {key}")
        return self._games[handle_id]

    def _state_memory_for_sample(self, sample: ReplaySampleRecord) -> StateMemory:
        return self._game_for_key((sample.run_id, sample.game_id)).state_memory

    def _maybe_unload_acknowledged_targets(self) -> None:
        active_ids = set(self._games)
        for update_index, targets in tuple(self._pending_unload_targets.items()):
            acknowledged = self._acknowledged_updates.get(update_index, set())
            if not active_ids <= acknowledged:
                continue
            self._pending_unload_targets.pop(update_index, None)
            self._acknowledged_updates.pop(update_index, None)
            for target in targets:
                self._unload_lora_adapter(target)

    def _require_empty_shared_adapter_root(self) -> None:
        if not self.config.enabled:
            return
        root = self._shared_adapter_root()
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                "online LoRA requires an empty shared adapter directory: "
                f"{root}"
            )

    def _raise_if_fatal(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"shared online LoRA failed: {self._fatal_error}")


def _process_trainer_gate(
    max_concurrent_trainer_jobs: int,
) -> threading.BoundedSemaphore:
    """Return the process-global online LoRA trainer gate."""

    if max_concurrent_trainer_jobs < 1:
        raise ValueError("online_lora.max_concurrent_trainer_jobs must be positive")
    global _TRAINER_GATE, _TRAINER_GATE_LIMIT
    with _TRAINER_GATE_LOCK:
        if _TRAINER_GATE is None:
            _TRAINER_GATE_LIMIT = max_concurrent_trainer_jobs
            _TRAINER_GATE = threading.BoundedSemaphore(max_concurrent_trainer_jobs)
            return _TRAINER_GATE
        if _TRAINER_GATE_LIMIT != max_concurrent_trainer_jobs:
            raise RuntimeError(
                "online_lora.max_concurrent_trainer_jobs cannot change within "
                "one process: expected "
                f"{_TRAINER_GATE_LIMIT}, got {max_concurrent_trainer_jobs}"
            )
        return _TRAINER_GATE


class _RuntimeVLLMServerController:
    """Control a local vLLM server while same-GPU trainer jobs run."""

    def __init__(
        self,
        *,
        base_url: str,
        command: Sequence[str] = (),
        pid: int | None = None,
        cwd: str = "",
        wait_timeout_seconds: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.command = tuple(command)
        self.pid = pid
        self.cwd = cwd
        self.wait_timeout_seconds = wait_timeout_seconds
        self._owned_process: subprocess.Popen[str] | None = None

    @classmethod
    def from_env(cls, *, base_url: str) -> "_RuntimeVLLMServerController":
        command_text = os.environ.get("FACE_OF_AGI_VLLM_RESTART_COMMAND_JSON", "")
        pid_text = os.environ.get("FACE_OF_AGI_VLLM_PID", "")
        if not command_text:
            return cls(base_url=base_url)
        try:
            loaded = json.loads(command_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "FACE_OF_AGI_VLLM_RESTART_COMMAND_JSON must be valid JSON"
            ) from exc
        if not isinstance(loaded, list) or not all(
            isinstance(item, str) for item in loaded
        ):
            raise RuntimeError(
                "FACE_OF_AGI_VLLM_RESTART_COMMAND_JSON must be a JSON string list"
            )
        pid = int(pid_text) if pid_text else None
        return cls(
            base_url=base_url,
            command=tuple(loaded),
            pid=pid,
            cwd=os.environ.get("FACE_OF_AGI_VLLM_RESTART_CWD", ""),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.command)

    @contextmanager
    def suspended(self) -> Any:
        if not self.enabled:
            yield
            return
        self.stop()
        try:
            yield
        finally:
            self.start()

    def stop(self) -> None:
        """Terminate the active local vLLM server and wait for the API to drop."""

        if not self.enabled:
            return
        pid = self.pid
        if pid is None and self._owned_process is not None:
            pid = self._owned_process.pid
        if pid is not None:
            _terminate_process_tree(pid)
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if not _vllm_server_available(self.base_url):
                time.sleep(2.0)
                return
            time.sleep(1.0)
        raise RuntimeError("vLLM server did not stop before trainer start")

    def start(self) -> None:
        """Start vLLM again and wait until the OpenAI-compatible API is ready."""

        if not self.enabled:
            return
        self._owned_process = subprocess.Popen(
            list(self.command),
            cwd=self.cwd or None,
            text=True,
            env=os.environ.copy(),
        )
        self.pid = self._owned_process.pid
        _wait_for_vllm_server(
            self.base_url,
            process=self._owned_process,
            timeout_seconds=self.wait_timeout_seconds,
        )

    def close(self) -> None:
        if self._owned_process is None:
            return
        if self._owned_process.poll() is None:
            _terminate_process_tree(self._owned_process.pid)
        self._owned_process = None


def _terminate_process_tree(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_vllm_server(
    base_url: str,
    *,
    process: subprocess.Popen[str] | None = None,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "vLLM exited before becoming ready with return code "
                f"{process.returncode}"
            )
        if _vllm_server_available(base_url):
            return
        time.sleep(2.0)
    raise RuntimeError("vLLM did not become ready after trainer completed")


def _vllm_server_available(base_url: str) -> bool:
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers=_vllm_json_headers(),
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return 200 <= response.status < 500
    except (TimeoutError, URLError, OSError):
        return False


def _release_trainer_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _validate_trainer_base_config(config: OnlineLoRAConfig) -> None:
    """Fail before gameplay when the configured HF trainer base cannot train."""

    if not config.enabled:
        return
    base_model = _trainer_base_model(config=config, sample=None)
    config_json = _local_trainer_config_json(config=config, base_model=base_model)
    if config_json is not None:
        _raise_if_fp8_trainer_config(config_json, source=base_model)


class _WarmVLMTrainer:
    """Single cached VLM base used to train one LoRA adapter at a time."""

    def __init__(self, config: OnlineLoRAConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._base_model_name: str | None = None
        self._processor: Any | None = None
        self._model: Any | None = None

    def train_world(
        self,
        *,
        adapter_path: Path,
        samples: Sequence[ReplaySampleRecord],
        previous_adapter_path: str | None,
        adapter_name: str,
    ) -> None:
        with self._lock:
            self._ensure_loaded(samples[0])
            self._prepare_adapter(
                adapter_name=adapter_name,
                previous_adapter_path=previous_adapter_path,
            )
            try:
                self._train_world_active_adapter(
                    adapter_path=adapter_path,
                    samples=samples,
                    adapter_name=adapter_name,
                )
            finally:
                self._delete_adapter(adapter_name)

    def train_grpo(
        self,
        *,
        role: str,
        adapter_path: Path,
        samples: Sequence[ReplaySampleRecord],
        reward_judge_model: RewardJudgeModel | None,
        previous_adapter_path: str | None,
        adapter_name: str,
    ) -> None:
        with self._lock:
            self._ensure_loaded(samples[0])
            self._prepare_adapter(
                adapter_name=adapter_name,
                previous_adapter_path=previous_adapter_path,
            )
            try:
                self._train_grpo_active_adapter(
                    role=role,
                    adapter_path=adapter_path,
                    samples=samples,
                    reward_judge_model=reward_judge_model,
                    adapter_name=adapter_name,
                )
            finally:
                self._delete_adapter(adapter_name)

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._base_model_name = None

    def _ensure_loaded(self, sample: ReplaySampleRecord) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("warm LoRA trainer requires transformers") from exc
        base_model = _trainer_base_model(config=self.config, sample=sample)
        model_class = _auto_vision_model_class()
        self._processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            **_trainer_processor_kwargs(self.config),
        )
        self._model = model_class.from_pretrained(
            base_model,
            trust_remote_code=True,
            **_trainer_model_kwargs(self.config),
        )
        self._model = _prepare_model_for_quantized_training(
            self._model,
            config=self.config,
        )
        self._base_model_name = base_model

    def _prepare_adapter(
        self,
        *,
        adapter_name: str,
        previous_adapter_path: str | None,
    ) -> None:
        try:
            from peft import PeftModel, get_peft_model
        except ImportError as exc:
            raise RuntimeError("warm LoRA trainer requires peft") from exc
        if self._model is None:
            raise RuntimeError("warm LoRA trainer model is not loaded")
        if isinstance(self._model, PeftModel):
            if adapter_name in self._model.peft_config:
                self._model.delete_adapter(adapter_name)
            if previous_adapter_path:
                self._model.load_adapter(
                    previous_adapter_path,
                    adapter_name=adapter_name,
                    is_trainable=True,
                )
            else:
                self._model.add_adapter(adapter_name, _lora_config(self.config))
            self._model.set_adapter(adapter_name)
            return
        if previous_adapter_path:
            self._model = PeftModel.from_pretrained(
                self._model,
                previous_adapter_path,
                adapter_name=adapter_name,
                is_trainable=True,
            )
        else:
            self._model = get_peft_model(
                self._model,
                _lora_config(self.config),
                adapter_name=adapter_name,
            )
        self._model.set_adapter(adapter_name)

    def _delete_adapter(self, adapter_name: str) -> None:
        if self._model is None or not hasattr(self._model, "delete_adapter"):
            return
        peft_config = getattr(self._model, "peft_config", {})
        if adapter_name in peft_config:
            self._model.delete_adapter(adapter_name)

    def _train_world_active_adapter(
        self,
        *,
        adapter_path: Path,
        samples: Sequence[ReplaySampleRecord],
        adapter_name: str,
    ) -> None:
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError("World SFT requires transformers") from exc
        if self._model is None or self._processor is None:
            raise RuntimeError("warm World trainer is not loaded")
        rows = [_world_sft_row(sample) for sample in samples]
        adapter_path.mkdir(parents=True, exist_ok=True)
        training_args = _world_training_args(
            adapter_path=adapter_path,
            row_count=len(rows),
            config=self.config,
        )
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=_WorldSFTDataset(rows),
            data_collator=_world_sft_collator(self._processor),
        )
        trainer.train()
        trainer.model.save_pretrained(
            str(adapter_path),
            selected_adapters=[adapter_name],
        )

    def _train_grpo_active_adapter(
        self,
        *,
        role: str,
        adapter_path: Path,
        samples: Sequence[ReplaySampleRecord],
        reward_judge_model: RewardJudgeModel | None,
        adapter_name: str,
    ) -> None:
        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise RuntimeError(
                "online LoRA GRPO requires trl"
            ) from exc
        if self._model is None or self._processor is None:
            raise RuntimeError("warm GRPO trainer is not loaded")
        train_rows = _grpo_train_rows(samples)
        adapter_path.mkdir(parents=True, exist_ok=True)
        training_args = _grpo_training_args(
            adapter_path=adapter_path,
            row_count=len(train_rows),
            config=self.config,
        )
        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=lambda completions, **kwargs: _reward_function(
                completions,
                reward_judge_model=reward_judge_model,
                **kwargs,
            ),
            args=training_args,
            train_dataset=_GRPODataset(train_rows),
            processing_class=self._processor,
        )
        del role
        trainer.train()
        trainer.model.save_pretrained(
            str(adapter_path),
            selected_adapters=[adapter_name],
        )


def _lora_config(config: OnlineLoRAConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=list(config.lora_target_modules),
    )


def _trainer_processor_kwargs(config: OnlineLoRAConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.trainer_local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def _trainer_model_kwargs(config: OnlineLoRAConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    device_map = _trainer_device_map(config.trainer_device_map)
    if device_map is not None:
        kwargs["device_map"] = device_map
    torch_dtype = _trainer_torch_dtype(config.trainer_torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    quantization_config = _trainer_quantization_config(config)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if config.trainer_local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def _trainer_device_map(value: str) -> Any | None:
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized in {"cuda", "gpu"}:
        return {"": "cuda:0"}
    if normalized.startswith("cuda:"):
        return {"": normalized}
    return value


def _trainer_torch_dtype(value: str) -> Any | None:
    normalized = value.strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    import torch

    names = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in names:
        raise ValueError(f"unsupported online_lora.trainer_torch_dtype: {value}")
    return names[normalized]


def _local_trainer_config_json(
    *,
    config: OnlineLoRAConfig,
    base_model: str,
) -> dict[str, Any] | None:
    if config.trainer_base_model_path:
        path = Path(config.trainer_base_model_path)
        if not path.exists():
            if config.trainer_local_files_only:
                raise RuntimeError(
                    "online_lora.trainer_base_model_path does not exist: "
                    f"{path}"
                )
            return None
        config_path = path / "config.json" if path.is_dir() else path
        if not config_path.exists():
            raise RuntimeError(
                "online_lora.trainer_base_model_path must contain config.json: "
                f"{path}"
            )
        return _read_json_mapping(config_path)

    path = Path(base_model)
    if path.exists():
        config_path = path / "config.json" if path.is_dir() else path
        if config_path.exists():
            return _read_json_mapping(config_path)

    if not config.trainer_local_files_only:
        return None
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "online_lora.trainer_local_files_only requires transformers"
        ) from exc
    try:
        trainer_config = AutoConfig.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "online_lora trainer base is not available locally: "
            f"{base_model}"
        ) from exc
    payload = trainer_config.to_dict()
    if not isinstance(payload, dict):
        raise RuntimeError(
            "online_lora trainer base config did not resolve to a mapping: "
            f"{base_model}"
        )
    return payload


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read trainer config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"trainer config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"trainer config must be a JSON mapping: {path}")
    return payload


def _raise_if_fp8_trainer_config(
    config_json: Mapping[str, Any],
    *,
    source: str,
) -> None:
    quantization_config = config_json.get("quantization_config")
    quantization_text = json.dumps(
        quantization_config if quantization_config is not None else config_json,
        sort_keys=True,
        default=str,
    ).lower()
    if "fp8" not in quantization_text:
        return
    raise RuntimeError(
        "online LoRA trainer base cannot be FP8-quantized; configure "
        "online_lora.trainer_base_model or online_lora.trainer_base_model_path "
        f"with a trainable non-FP8 base instead: {source}"
    )


def _trainer_quantization_config(config: OnlineLoRAConfig) -> Any | None:
    mode = config.trainer_quantization.strip().lower()
    if mode in {"", "none"}:
        return None
    if mode != "bnb_4bit":
        raise ValueError(
            "unsupported online_lora.trainer_quantization: "
            f"{config.trainer_quantization}"
        )
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "online_lora.trainer_quantization=bnb_4bit requires torch, "
            "transformers, and bitsandbytes"
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _prepare_model_for_quantized_training(
    model: Any,
    *,
    config: OnlineLoRAConfig,
) -> Any:
    if config.trainer_quantization.strip().lower() != "bnb_4bit":
        return model
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "online_lora.trainer_quantization=bnb_4bit requires peft"
        ) from exc
    return prepare_model_for_kbit_training(model)


def _train_grpo_adapter(
    *,
    role: str,
    adapter_path: Path,
    samples: Sequence[ReplaySampleRecord],
    config: OnlineLoRAConfig,
    reward_judge_model: RewardJudgeModel | None = None,
    previous_adapter_path: str | None = None,
    trainer_cache: "_WarmVLMTrainer | None" = None,
    adapter_name: str = "default",
) -> None:
    """Run a bounded TRL GRPO update for one role adapter."""

    if not samples:
        raise RuntimeError(f"no replay samples available for {role} LoRA update")
    if role == "world":
        raise RuntimeError("World updates use supervised SFT, not GRPO")
    if trainer_cache is not None:
        trainer_cache.train_grpo(
            role=role,
            adapter_path=adapter_path,
            samples=samples,
            reward_judge_model=reward_judge_model,
            previous_adapter_path=previous_adapter_path,
            adapter_name=adapter_name,
        )
        return

    try:
        from peft import PeftModel
        from transformers import AutoProcessor
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "online LoRA GRPO requires peft, transformers, and trl"
        ) from exc

    train_rows = [
        {
            "prompt": _grpo_prompt_messages(sample),
            "target": _sample_target(sample),
            "recorded_reward": float(sample.reward),
            "role": sample.role,
            "metadata": sample.metadata,
            "run_id": sample.run_id,
            "game_id": sample.game_id,
            "turn_id": sample.turn_id,
            "sample_id": sample.id,
        }
        for sample in samples
    ]

    adapter_path.mkdir(parents=True, exist_ok=True)
    train_batch_size = min(config.train_batch_size, len(train_rows))
    training_args = GRPOConfig(
        output_dir=str(adapter_path),
        max_completion_length=config.max_completion_tokens,
        num_generations=max(2, config.grpo_num_generations),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=train_batch_size,
        max_steps=_grpo_max_steps(sample_count=len(train_rows), config=config),
        logging_steps=1,
        save_steps=1,
        remove_unused_columns=False,
    )
    base_model = _trainer_base_model(config=config, sample=samples[0])
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        **_trainer_processor_kwargs(config),
    )
    model_class = _auto_vision_model_class()
    base = model_class.from_pretrained(
        base_model,
        trust_remote_code=True,
        **_trainer_model_kwargs(config),
    )
    base = _prepare_model_for_quantized_training(base, config=config)
    model_arg: Any = base
    peft_config: Any | None = _lora_config(config)
    if previous_adapter_path:
        model_arg = PeftModel.from_pretrained(
            base,
            previous_adapter_path,
            is_trainable=True,
        )
        peft_config = None

    trainer_kwargs = {
        "model": model_arg,
        "reward_funcs": lambda completions, **kwargs: _reward_function(
            completions,
            reward_judge_model=reward_judge_model,
            **kwargs,
        ),
        "args": training_args,
        "train_dataset": _GRPODataset(train_rows),
        "processing_class": processor,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.model.save_pretrained(str(adapter_path))


def _train_grpo_adapter_hf(
    *,
    engine: HFVLMEngine,
    role: str,
    adapter_path: Path,
    samples: Sequence[ReplaySampleRecord],
    config: OnlineLoRAConfig,
    previous_adapter_path: str | None = None,
    adapter_name: str = "default",
) -> None:
    """Run a bounded TRL GRPO update on the shared HF engine model."""

    if not samples:
        raise RuntimeError(f"no replay samples available for {role} LoRA update")
    if role == "world":
        raise RuntimeError("World updates use supervised SFT, not GRPO")
    try:
        from trl import GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("online LoRA GRPO requires trl") from exc

    train_rows = _grpo_train_rows(samples)
    adapter_path.mkdir(parents=True, exist_ok=True)
    training_args = _grpo_training_args(
        adapter_path=adapter_path,
        row_count=len(train_rows),
        config=config,
    )
    with runtime_timing.span(
        "online_lora.hf_grpo",
        role=role,
        adapter_name=adapter_name,
        sample_count=len(samples),
        train_row_count=len(train_rows),
        max_completion_tokens=config.max_completion_tokens,
        grpo_num_generations=config.grpo_num_generations,
    ):
        with engine.exclusive_training():
            engine.prepare_trainable_adapter(
                adapter_name=adapter_name,
                previous_adapter_path=previous_adapter_path,
            )
            trainer = GRPOTrainer(
                model=engine.model,
                reward_funcs=lambda completions, **kwargs: _reward_function(
                    completions,
                    reward_judge_model=None,
                    **kwargs,
                ),
                args=training_args,
                train_dataset=_GRPODataset(train_rows),
                processing_class=engine.processor,
            )
            trainer.train()
            engine.save_adapter(
                adapter_name=adapter_name,
                adapter_path=adapter_path,
            )


def _train_world_sft_adapter(
    *,
    adapter_path: Path,
    samples: Sequence[ReplaySampleRecord],
    config: OnlineLoRAConfig,
    previous_adapter_path: str | None = None,
    trainer_cache: "_WarmVLMTrainer | None" = None,
    adapter_name: str = "default",
) -> None:
    """Run image-aware supervised World transition-summary LoRA training."""

    if not samples:
        raise RuntimeError("no replay samples available for World SFT update")
    if trainer_cache is not None:
        trainer_cache.train_world(
            adapter_path=adapter_path,
            samples=samples,
            previous_adapter_path=previous_adapter_path,
            adapter_name=adapter_name,
        )
        return
    try:
        import torch
        from peft import PeftModel, get_peft_model
        from transformers import AutoProcessor, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "World SFT requires torch, peft, and transformers to be installed"
        ) from exc

    base_model = _trainer_base_model(config=config, sample=samples[0])
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        **_trainer_processor_kwargs(config),
    )
    model_class = _auto_vision_model_class()
    base = model_class.from_pretrained(
        base_model,
        trust_remote_code=True,
        **_trainer_model_kwargs(config),
    )
    base = _prepare_model_for_quantized_training(base, config=config)
    if previous_adapter_path:
        model = PeftModel.from_pretrained(
            base,
            previous_adapter_path,
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            base,
            _lora_config(config),
        )

    rows = [_world_sft_row(sample) for sample in samples]
    adapter_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(adapter_path),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=min(config.train_batch_size, len(rows)),
        num_train_epochs=config.train_epochs,
        max_steps=config.max_update_steps or -1,
        logging_steps=1,
        save_steps=1,
        remove_unused_columns=False,
        report_to=[],
    )

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            _apply_chat_template(processor, feature["messages"])
            for feature in features
        ]
        image_batches = [feature["images"] for feature in features]
        inputs = processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        inputs["labels"] = labels
        return inputs

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_WorldSFTDataset(rows),
        data_collator=collate,
    )
    trainer.train()
    trainer.model.save_pretrained(str(adapter_path))
    del torch


def _train_world_sft_adapter_hf(
    *,
    engine: HFVLMEngine,
    adapter_path: Path,
    samples: Sequence[ReplaySampleRecord],
    config: OnlineLoRAConfig,
    previous_adapter_path: str | None = None,
    adapter_name: str = "default",
) -> None:
    """Run image-aware supervised World SFT on the shared HF engine model."""

    if not samples:
        raise RuntimeError("no replay samples available for World SFT update")
    try:
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError("World SFT requires transformers") from exc

    rows = [_world_sft_row(sample) for sample in samples]
    adapter_path.mkdir(parents=True, exist_ok=True)
    training_args = _world_training_args(
        adapter_path=adapter_path,
        row_count=len(rows),
        config=config,
    )
    with runtime_timing.span(
        "online_lora.hf_world_sft",
        adapter_name=adapter_name,
        sample_count=len(samples),
        train_row_count=len(rows),
    ):
        with engine.exclusive_training():
            engine.prepare_trainable_adapter(
                adapter_name=adapter_name,
                previous_adapter_path=previous_adapter_path,
            )
            trainer = Trainer(
                model=engine.model,
                args=training_args,
                train_dataset=_WorldSFTDataset(rows),
                data_collator=_world_sft_collator(engine.processor),
            )
            trainer.train()
            engine.save_adapter(
                adapter_name=adapter_name,
                adapter_path=adapter_path,
            )


def _grpo_max_steps(*, sample_count: int, config: OnlineLoRAConfig) -> int:
    """Return enough GRPO steps to cover the selected replay rows."""

    train_batch_size = min(config.train_batch_size, sample_count)
    derived_steps = ceil(sample_count / train_batch_size) * config.train_epochs
    if config.max_update_steps is not None:
        return min(derived_steps, config.max_update_steps)
    return derived_steps


def _grpo_train_rows(
    samples: Sequence[ReplaySampleRecord],
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": _grpo_prompt_messages(sample),
            "target": _sample_target(sample),
            "recorded_reward": float(sample.reward),
            "role": sample.role,
            "metadata": sample.metadata,
            "run_id": sample.run_id,
            "game_id": sample.game_id,
            "turn_id": sample.turn_id,
            "sample_id": sample.id,
        }
        for sample in samples
    ]


def _grpo_training_args(
    *,
    adapter_path: Path,
    row_count: int,
    config: OnlineLoRAConfig,
) -> Any:
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(adapter_path),
        max_completion_length=config.max_completion_tokens,
        num_generations=max(2, config.grpo_num_generations),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=min(config.train_batch_size, row_count),
        max_steps=_grpo_max_steps(sample_count=row_count, config=config),
        logging_steps=1,
        save_steps=1,
        remove_unused_columns=False,
    )


def _world_training_args(
    *,
    adapter_path: Path,
    row_count: int,
    config: OnlineLoRAConfig,
) -> Any:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(adapter_path),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=min(config.train_batch_size, row_count),
        num_train_epochs=config.train_epochs,
        max_steps=config.max_update_steps or -1,
        logging_steps=1,
        save_steps=1,
        remove_unused_columns=False,
        report_to=[],
    )


def _world_sft_collator(processor: Any) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            _apply_chat_template(processor, feature["messages"])
            for feature in features
        ]
        image_batches = [feature["images"] for feature in features]
        inputs = processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        inputs["labels"] = labels
        return inputs

    return collate


class _WorldSFTDataset:
    """Tiny list-backed dataset to avoid requiring datasets for World SFT."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._rows[index])


class _GRPODataset:
    """Tiny list-backed dataset that preserves PIL images in GRPO prompts."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._rows[index])


def _world_sft_row(sample: ReplaySampleRecord) -> dict[str, Any]:
    messages = deepcopy(_sample_messages(sample))
    target = _sample_target(sample)
    images = _decode_request_images(messages)
    if not images:
        raise RuntimeError(f"World replay sample {sample.id} missing image input")
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(target, sort_keys=True),
        }
    )
    return {
        "messages": messages,
        "images": images,
        "sample_id": sample.id,
    }


def _decode_request_images(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    from PIL import Image

    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or not url.startswith("data:"):
                continue
            try:
                encoded = url.split(",", 1)[1]
            except IndexError as exc:
                raise RuntimeError("invalid data URL in World replay image") from exc
            images.append(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
    return images


def _apply_chat_template(processor: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    apply_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_template):
        return str(
            apply_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return "\n".join(
        f"{message.get('role', 'user')}: {_message_text(message.get('content'))}"
        for message in messages
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def _auto_vision_model_class() -> Any:
    import transformers

    for name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError(
        "transformers does not provide an image-text model auto class"
    )


def _trainer_base_model(
    *,
    config: OnlineLoRAConfig,
    sample: ReplaySampleRecord | None,
) -> str:
    base_model = str(
        config.trainer_base_model_path
        or config.trainer_base_model
        or config.base_model_path
        or (sample.metadata.get("base_model_path") if sample is not None else "")
        or (sample.metadata.get("base_model") if sample is not None else "")
        or config.base_model
        or ""
    )
    if not base_model:
        raise RuntimeError(
            "online LoRA training requires online_lora.trainer_base_model_path, "
            "online_lora.trainer_base_model, online_lora.base_model_path, or "
            "online_lora.base_model"
        )
    return base_model


def _reward_function(
    completions: list[Any],
    *,
    reward_judge_model: RewardJudgeModel | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward generated JSON completions against replay targets."""

    roles = kwargs.get("role") or []
    metadata_items = kwargs.get("metadata") or []
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        text = _completion_text(completion)
        role = str(roles[index]) if index < len(roles) else ""
        metadata = (
            metadata_items[index]
            if index < len(metadata_items) and isinstance(metadata_items[index], dict)
            else {}
        )
        if role == "agent":
            rewards.append(_agent_candidate_value_reward(text, metadata))
        elif role == "interest":
            rewards.append(_interest_label_reward(text, metadata))
        else:
            rewards.append(0.0)
    return rewards


def _sample_messages(sample: ReplaySampleRecord) -> list[dict[str, Any]]:
    request = _sample_request(sample)
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"{sample.role} replay sample {sample.id} missing messages")
    return messages


def _grpo_prompt_messages(sample: ReplaySampleRecord) -> list[dict[str, Any]]:
    """Return TRL multimodal messages with decoded PIL images."""

    return [
        {
            **message,
            "content": _grpo_message_content(message.get("content")),
        }
        for message in _sample_messages(sample)
    ]


def _grpo_message_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    converted: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        if item.get("type") != "image_url":
            converted.append(dict(item))
            continue
        image = _decode_image_url_block(item)
        if image is not None:
            converted.append({"type": "image", "image": image})
    return converted


def _decode_image_url_block(item: Mapping[str, Any]) -> Any | None:
    from PIL import Image

    image_url = item.get("image_url")
    if not isinstance(image_url, dict):
        return None
    url = image_url.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        encoded = url.split(",", 1)[1]
    except IndexError as exc:
        raise RuntimeError("invalid data URL in replay image") from exc
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _sample_request(sample: ReplaySampleRecord) -> dict[str, Any]:
    request = sample.prompt_json.get("request")
    if not isinstance(request, dict):
        raise RuntimeError(f"{sample.role} replay sample {sample.id} missing request")
    return request


def _sample_target(sample: ReplaySampleRecord) -> dict[str, Any]:
    target = sample.completion_json.get("target")
    if not isinstance(target, dict):
        raise RuntimeError(f"{sample.role} replay sample {sample.id} missing target")
    return target


def _agent_candidate_value_reward(
    completion_text: str,
    metadata: Mapping[str, Any],
) -> float:
    invalid_reward = _invalid_agent_output_reward(metadata)
    completion = _parse_completion_json(completion_text)
    if completion is None:
        return invalid_reward
    generated_action = completion.get("action")
    if not isinstance(generated_action, dict):
        return invalid_reward
    table = _candidate_score_table(metadata)
    if not table:
        return invalid_reward
    normalized_table = _with_normalized_advantages(table)
    for item in normalized_table:
        action = item.get("model_action")
        if not isinstance(action, dict):
            action = item.get("action")
        if isinstance(action, dict) and _same_action_json(
            generated_action,
            action,
            action_name=str(item.get("action_name") or ""),
        ):
            return _numeric_or_zero(item.get("normalized_advantage"))
    return invalid_reward


def _invalid_agent_output_reward(metadata: Mapping[str, Any]) -> float:
    table = _candidate_score_table(metadata)
    if not table:
        return -1.0
    valid_rewards = [
        _numeric_or_zero(item.get("normalized_advantage"))
        for item in _with_normalized_advantages(table)
        if isinstance(item.get("model_action"), dict)
        or isinstance(item.get("action"), dict)
    ]
    if not valid_rewards:
        return -1.0
    return min(valid_rewards) - 1e-6


def _interest_label_reward(
    completion_text: str,
    metadata: Mapping[str, Any],
) -> float:
    label = metadata.get("interest_label")
    if not isinstance(label, dict):
        return 0.0
    completion = _parse_completion_json(completion_text)
    if completion is None:
        return 0.0
    values = completion.get("candidate_values")
    if not isinstance(values, list):
        return 0.0
    executed_index = label.get("executed_candidate_index")
    if isinstance(executed_index, bool) or not isinstance(executed_index, int):
        return 0.0
    target_lp = _numeric_or_none(label.get("realized_learning_progress"))
    target_goal_delta = _numeric_or_none(label.get("realized_goal_delta"))
    if target_lp is None or target_goal_delta is None:
        return 0.0
    for raw_value in values:
        if not isinstance(raw_value, dict):
            continue
        if raw_value.get("candidate_index") != executed_index:
            continue
        predicted_lp = _numeric_or_none(raw_value.get("expected_learning_progress"))
        predicted_goal_delta = _numeric_or_none(raw_value.get("expected_goal_delta"))
        confidence = _numeric_or_none(raw_value.get("confidence"))
        if (
            predicted_lp is None
            or predicted_goal_delta is None
            or confidence is None
        ):
            return 0.0
        value_error = min(
            1.0,
            (
                abs(predicted_lp - target_lp)
                + abs(predicted_goal_delta - target_goal_delta)
            )
            / 2.0,
        )
        value_reward = 1.0 - value_error
        confidence_target = value_reward
        calibration_reward = 1.0 - min(1.0, abs(confidence - confidence_target))
        return 0.8 * value_reward + 0.2 * calibration_reward
    return 0.0


def _parse_completion_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _chat_completion_content(vllm_base_url: str, request_payload: dict[str, Any]) -> str:
    payload = json.dumps(request_payload).encode("utf-8")
    request = Request(
        f"{vllm_base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers=_vllm_json_headers(),
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        if response.status >= 400:
            raise RuntimeError(f"vLLM chat completion failed: HTTP {response.status}")
        body = response.read().decode("utf-8")
    loaded = json.loads(body)
    choices = loaded.get("choices") if isinstance(loaded, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("vLLM chat completion response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("vLLM chat completion response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("vLLM chat completion response missing text content")
    return content


def _hf_chat_completion_content(
    engine: HFVLMEngine,
    request_payload: dict[str, Any],
) -> str:
    response = engine.chat(dict(request_payload))
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("HF chat completion response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("HF chat completion response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("HF chat completion response missing text content")
    return content


def _vllm_json_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _learning_progress_eval_bundle(sample: ReplaySampleRecord) -> dict[str, Any]:
    return _learning_progress_eval_bundle_from_metadata(
        sample.metadata,
        label=f"world replay sample {sample.id}",
    )


def _learning_progress_eval_bundle_from_metadata(
    metadata: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    bundle = metadata.get("learning_progress_eval")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"{label} missing learning_progress_eval")
    required = {
        "action",
        "change_summary",
        "previous_observation",
        "current_observation",
    }
    missing = sorted(required - set(bundle))
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"{label} missing LP eval fields: {names}")
    return bundle


def _world_judge_input_from_bundle(
    *,
    bundle: Mapping[str, Any],
    run_id: str,
    game_id: str,
    turn_id: int,
    sample_id: int,
    model: str,
    predicted_change: str,
    reward_metadata: Mapping[str, Any],
) -> RewardJudgeInput:
    action = _action_from_json(bundle["action"])
    candidate_index = int(bundle.get("candidate_index") or 0)
    return RewardJudgeInput(
        run_id=run_id,
        game_id=game_id,
        turn_id=turn_id,
        action=action,
        prediction=WorldPrediction(
            candidate_index=candidate_index,
            action=action,
            predicted_change=predicted_change,
            metadata={
                "model": model,
                "replay_sample_id": sample_id,
                **dict(reward_metadata),
            },
        ),
        change_summary=str(bundle["change_summary"]),
        previous_observation=_observation_from_jsonable(
            bundle["previous_observation"]
        ),
        current_observation=_observation_from_jsonable(
            bundle["current_observation"]
        ),
        metadata={
            "replay_sample_id": sample_id,
            "model": model,
            **dict(reward_metadata),
        },
    )


def _candidate_score_table_from_interest_payload(
    sample: ReplaySampleRecord,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    original_by_index = {
        int(item["candidate_index"]): item
        for item in _candidate_score_table(sample.metadata)
        if isinstance(item.get("candidate_index"), int)
    }
    values = payload.get("candidate_values")
    if not isinstance(values, list):
        raise RuntimeError(
            f"Interest replay sample {sample.id} missing candidate_values"
        )
    label_components = sample.metadata.get("label_components")
    if not isinstance(label_components, dict):
        raise RuntimeError(
            f"Interest replay sample {sample.id} missing label_components"
        )
    lp_weight = _numeric_component(label_components, "lp_weight")
    goal_weight = _numeric_component(label_components, "goal_weight")
    table: list[dict[str, Any]] = []
    returned_indices: set[int] = set()
    for raw_value in values:
        if not isinstance(raw_value, dict):
            raise RuntimeError(
                f"Interest replay sample {sample.id} candidate value is not an object"
            )
        candidate_index = raw_value.get("candidate_index")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise RuntimeError(
                f"Interest replay sample {sample.id} candidate_index must be integer"
            )
        original = original_by_index.get(candidate_index)
        if original is None:
            raise RuntimeError(
                f"Interest replay sample {sample.id} returned unknown "
                f"candidate_index {candidate_index}"
            )
        if candidate_index in returned_indices:
            raise RuntimeError(
                f"Interest replay sample {sample.id} duplicated "
                f"candidate_index {candidate_index}"
            )
        returned_indices.add(candidate_index)
        expected_lp = _bounded_numeric(
            raw_value.get("expected_learning_progress"),
            minimum=-1.0,
            maximum=1.0,
            label="expected_learning_progress",
        )
        expected_goal_delta = _bounded_numeric(
            raw_value.get("expected_goal_delta"),
            minimum=-1.0,
            maximum=1.0,
            label="expected_goal_delta",
        )
        confidence = _bounded_numeric(
            raw_value.get("confidence"),
            minimum=0.0,
            maximum=1.0,
            label="confidence",
        )
        confidence_adjusted_lp = confidence * expected_lp
        blended_score = lp_weight * confidence_adjusted_lp + goal_weight * (
            expected_goal_delta
        )
        table.append(
            {
                "candidate_index": candidate_index,
                "action": original["action"],
                "model_action": original.get("model_action", original["action"]),
                "action_name": original.get("action_name"),
                "expected_learning_progress": expected_lp,
                "confidence": confidence,
                "confidence_adjusted_learning_progress": confidence_adjusted_lp,
                "expected_goal_delta": expected_goal_delta,
                "blended_score": blended_score,
                "notes": str(raw_value.get("notes") or "").strip(),
                "metadata": {
                    "lp_weight": lp_weight,
                    "goal_weight": goal_weight,
                    "source": "interest_rescore",
                    "replay_sample_id": sample.id,
                },
            }
        )
    missing = sorted(set(original_by_index) - returned_indices)
    if missing:
        raise RuntimeError(
            f"Interest replay sample {sample.id} missing candidate indices: "
            + ", ".join(str(index) for index in missing)
        )
    return _with_normalized_advantages(tuple(table))


def _candidate_score_table(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    table = metadata.get("candidate_score_table")
    if not isinstance(table, (list, tuple)):
        return ()
    return tuple(item for item in table if isinstance(item, dict))


def _with_normalized_advantages(
    table: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    scores = [
        _numeric_or_none(item.get("blended_score"))
        for item in table
    ]
    if any(score is None for score in scores):
        return tuple(dict(item) for item in table)
    numeric_scores = [float(score) for score in scores if score is not None]
    mean_score = mean(numeric_scores) if numeric_scores else 0.0
    advantages = [score - mean_score for score in numeric_scores]
    max_abs_advantage = max((abs(value) for value in advantages), default=0.0)
    normalized: list[dict[str, Any]] = []
    for item, advantage in zip(table, advantages):
        copied = dict(item)
        copied["advantage"] = advantage
        copied["normalized_advantage"] = (
            0.0 if max_abs_advantage <= 1e-12 else advantage / max_abs_advantage
        )
        normalized.append(copied)
    return tuple(normalized)


def _action_from_json(value: Any) -> ActionSpec:
    if not isinstance(value, dict):
        raise RuntimeError("LP eval action must be a JSON object")
    if "action_id" not in value:
        raise RuntimeError("LP eval action missing action_id")
    data = value.get("data")
    if data is not None and not isinstance(data, dict):
        raise RuntimeError("LP eval action data must be an object or null")
    return ActionSpec(action_id=str(value["action_id"]), data=data)


def _observation_from_jsonable(value: Any) -> Observation:
    hydrated = from_memory_jsonable(value)
    if not isinstance(hydrated, dict):
        raise RuntimeError("LP eval observation must be a JSON object")
    metadata = hydrated.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise RuntimeError("LP eval observation metadata must be an object")
    frames = hydrated.get("frames") or ()
    if not isinstance(frames, (list, tuple)):
        raise RuntimeError("LP eval observation frames must be a list")
    return Observation(
        id=str(hydrated.get("id") or ""),
        step=int(hydrated.get("step") or 0),
        frame=hydrated.get("frame"),
        frames=tuple(frames),
        raw_frame_data=hydrated.get("raw_frame_data"),
        metadata=metadata,
    )


def _numeric_component(components: Mapping[str, Any], key: str) -> float:
    value = components.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"reward component {key} must be numeric")
    return float(value)


def _numeric_component_default(
    components: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    if key not in components:
        return default
    return _numeric_component(components, key)


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numeric_or_zero(value: Any) -> float:
    numeric = _numeric_or_none(value)
    return 0.0 if numeric is None else numeric


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _bounded_numeric(
    value: Any,
    *,
    minimum: float,
    maximum: float,
    label: str,
) -> float:
    from math import isfinite

    numeric = _numeric_or_none(value)
    if numeric is None or not isfinite(numeric) or not minimum <= numeric <= maximum:
        raise RuntimeError(f"{label} must be within {minimum}..{maximum}")
    return numeric


def _same_action_json(
    generated: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    action_name: str,
) -> bool:
    generated_name = _action_name_from_json(generated)
    candidate_name = action_name or _action_name_from_json(candidate)
    if generated_name != candidate_name:
        return False
    return _action_data_json(generated) == _action_data_json(candidate)


def _action_name_from_json(value: Mapping[str, Any]) -> str:
    raw = str(value.get("action_id") or value.get("name") or "")
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw


def _action_data_json(value: Mapping[str, Any]) -> dict[str, Any]:
    data = value.get("data")
    return dict(data) if isinstance(data, dict) else {}


def _clip(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _score_key(score: HeldoutScore) -> tuple[str, str, int]:
    return (score.run_id, score.game_id, score.sample_id)


def _sample_turn_key(sample: ReplaySampleRecord) -> tuple[str, str, int]:
    return (sample.run_id, sample.game_id, sample.turn_id)


def _turn_key_text(key: tuple[str, str, int]) -> str:
    run_id, game_id, turn_id = key
    return f"{run_id}:{game_id}:{turn_id}"


def _sample_ref(sample: ReplaySampleRecord) -> dict[str, Any]:
    return {
        "run_id": sample.run_id,
        "game_id": sample.game_id,
        "turn_id": sample.turn_id,
        "sample_id": sample.id,
    }


def _heldout_score_metadata(score: HeldoutScore) -> dict[str, Any]:
    return {
        "run_id": score.run_id,
        "game_id": score.game_id,
        "turn_id": score.turn_id,
        "sample_id": score.sample_id,
        "prediction": score.prediction,
        "score": score.score,
        "notes": score.notes,
        "error_tags": score.error_tags,
        "invalid_prediction": score.invalid_prediction,
    }


def _mean_score(scores: Sequence[HeldoutScore]) -> float:
    return mean(score.score for score in scores) if scores else 0.0


def _safe_adapter_component(value: str) -> str:
    safe = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")
    return safe or "unknown"


def _active_lora_adapter_name(target: Any | None) -> str:
    for candidate in (target, getattr(target, "provider", None)):
        name = getattr(candidate, "active_lora_adapter_name", None)
        if name:
            return str(name)
    return ""


def _configured_model_name(target: Any | None) -> str:
    for candidate in (target, getattr(target, "provider", None)):
        config = getattr(candidate, "config", None)
        model = getattr(config, "model", None)
        if model:
            return str(model)
    return ""
