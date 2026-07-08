"""Run the isolated single-VLM online LoRA ARC experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch

from single_vlm_arc.actions import (
    ACTION_NAMES,
    available_action_names,
    mask_action_logits,
    masked_action_probabilities,
    select_action,
    valid_action_mask,
)
from single_vlm_arc.config import apply_cli_overrides, load_config
from single_vlm_arc.env import build_session
from single_vlm_arc.history import (
    RollingHistory,
    Transition,
    decision_frame,
    decision_frame_with_metadata,
    transition_to_json,
)
from single_vlm_arc.logging import ExperimentLogger
from single_vlm_arc.model import build_model
from single_vlm_arc.online_update import (
    coord_self_imitation_loss,
    frame_to_palette_tensor,
    latent_changed_patch_mask,
    latent_grid_loss,
    next_frame_loss,
    policy_parameters,
    predicted_frame_tensor,
    world_model_parameters,
)
from single_vlm_arc.rewards import compute_reward, score_delta_from_info


def main() -> None:
    args = _build_parser().parse_args()
    config = apply_cli_overrides(
        load_config(args.config),
        game_id=args.game_id,
        game_index=args.game_index,
        max_turns=args.max_turns,
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        save_video=args.save_video,
        video_fps=args.video_fps,
        video_frame_scale=args.video_frame_scale,
        world_loss_mode=args.world_loss_mode,
        latent_loss_weight=args.latent_loss_weight,
        latent_changed_patch_weight=args.latent_changed_patch_weight,
        latent_huber_beta=args.latent_huber_beta,
        latent_cosine_loss_weight=args.latent_cosine_loss_weight,
        latent_cosine_min_delta_norm=args.latent_cosine_min_delta_norm,
        latent_learning_progress_normalization=(
            args.latent_learning_progress_normalization
        ),
        latent_learning_progress_normalization_floor=(
            args.latent_learning_progress_normalization_floor
        ),
        save_frame_predictions=args.save_frame_predictions,
        frame_prediction_save_every=args.frame_prediction_save_every,
        frame_prediction_frame_scale=args.frame_prediction_frame_scale,
        save_latent_predictions=args.save_latent_predictions,
        latent_prediction_save_every=args.latent_prediction_save_every,
        latent_prediction_frame_scale=args.latent_prediction_frame_scale,
    )
    result = run_experiment(config)
    print(
        "single-vlm-lora stop:"
        f" turns={result['turns']}"
        f" total_reward={result['total_reward']:.4f}"
        f" output_dir={config.logging.output_dir}"
    )


def run_experiment(config: Any) -> dict[str, Any]:
    """Run one configured single-VLM online adaptation experiment."""

    _seed_runtime(int(config.environment.seed))
    logger = ExperimentLogger(config)
    session = build_session(config.environment, dry_run=config.dry_run)
    model = build_model(config.model, palette_size=config.palette_size)
    setattr(
        model,
        "policy_adapter_trainable",
        bool(getattr(config.update, "policy_adapter_trainable", True)),
    )
    model.eval()
    world_parameters = world_model_parameters(model)
    policy_only_parameters = policy_parameters(model)
    if not world_parameters:
        raise RuntimeError("single-VLM experiment has no trainable world parameters")
    if not policy_only_parameters:
        raise RuntimeError("single-VLM experiment has no trainable policy parameters")
    world_optimizer = torch.optim.AdamW(
        world_parameters,
        lr=float(config.update.learning_rate),
    )
    policy_optimizer = torch.optim.AdamW(
        policy_only_parameters,
        lr=float(config.update.learning_rate),
    )
    history = RollingHistory(config.frame_history_n)
    observation = session.reset()
    total_reward = 0.0
    last_prediction_loss = 0.0
    learning_progress_rate_baseline: float | None = None
    action_learning_progress_rate_baselines: dict[str, float] = {}
    pending_queue: list[PendingTurn] = []
    policy_accumulator = PolicyUpdateAccumulator(
        model=model,
        optimizer=policy_optimizer,
        config=config,
    )
    policy_return_accumulator = PolicyReturnAccumulator(config=config)
    video_recorder = logger.video_recorder(config)
    frame_prediction_logger = logger.frame_prediction_logger(config)
    latent_prediction_logger = logger.latent_prediction_logger(config)
    world_model_step = 0
    learning_progress_horizon = max(
        1,
        int(getattr(config.update, "learning_progress_horizon", 1)),
    )
    termination_reason: str | None = None
    terminal_turn: int | None = None
    terminal_action_name: str | None = None
    terminal_observation_id: str | None = None

    for turn in range(config.environment.max_turns):
        if not _observation_has_usable_frame(observation):
            termination_reason = "terminal_observation_without_frame"
            terminal_turn = turn
            terminal_observation_id = getattr(observation, "id", None)
            break
        current_frame, decision_frame_index, decision_frame_count = (
            decision_frame_with_metadata(observation)
        )
        if turn == 0:
            video_recorder.append_observation(
                observation,
                turn=None,
                action_name=None,
                phase="reset",
            )
        turn_start = time.perf_counter()
        previous_info = session.get_info()
        action_space = tuple(session.get_action_space())
        valid_names = available_action_names(action_space)
        prompt = history.build_prompt(
            game_id=session.game_id,
            turn=turn,
            valid_actions=valid_names,
        )
        images = history.recent_frames(current_frame)
        frame_observation_ids = history.recent_frame_observation_ids(observation)

        with _use_policy_adapter(model), torch.no_grad():
            output = model(
                prompt,
                images,
                include_frame_logits=_uses_pixel_world_loss(config),
            )
            probability_temperature = (
                config.model.temperature
                if config.model.action_selection == "sample"
                else 1.0
            )
            action_probabilities = masked_action_probabilities(
                output.action_logits,
                action_space,
                temperature=probability_temperature,
            )
            selected = select_action(
                action_logits=output.action_logits,
                coord_logits=output.coord_logits,
                action_space=action_space,
                mode=config.model.action_selection,
                temperature=config.model.temperature,
            )

        next_observation = session.step(selected.action)
        next_info = session.get_info()
        if not _observation_has_usable_frame(next_observation):
            termination_reason = "terminal_next_observation_without_frame"
            terminal_turn = turn
            terminal_action_name = selected.action_name
            terminal_observation_id = getattr(next_observation, "id", None)
            break
        next_frame, next_decision_frame_index, next_decision_frame_count = (
            decision_frame_with_metadata(next_observation)
        )
        frame_change = _frame_change_diagnostics(
            current_frame,
            next_frame,
            config=config,
        )
        video_recorder.append_observation(
            next_observation,
            turn=turn,
            action_name=selected.action_name,
            phase="after_action",
        )
        score_delta = score_delta_from_info(previous_info, next_info)
        interaction_elapsed_seconds = time.perf_counter() - turn_start

        transition = Transition(
            turn=turn,
            observation=observation,
            action=selected.action,
            next_observation=next_observation,
            action_index=selected.action_index,
            log_probability=float(selected.log_prob.detach().cpu().item()),
            prediction_loss=0.0,
            reward=0.0,
            metadata={
                "action_probability": selected.probability,
                "score_delta": score_delta,
                "learning_progress": 0.0,
            },
        )
        history.append(transition)
        current_pending = PendingTurn(
            transition=transition,
            prompt=prompt,
            images=images,
            action_space=action_space,
            selected_action_index=selected.action_index,
            selected_x=selected.x,
            selected_y=selected.y,
            score_delta=score_delta,
            interaction_elapsed_seconds=interaction_elapsed_seconds,
            model_input={
                "prompt": prompt,
                "valid_actions": list(valid_names),
                "frame_history_count": len(images),
                "frame_history_observation_ids": frame_observation_ids,
                "current_observation_id": observation.id,
                "decision_frame_index": decision_frame_index,
                "decision_frame_count": decision_frame_count,
                "world_model_step_at_action": world_model_step,
            },
            model_output={
                "selected_action_index": selected.action_index,
                "selected_action_name": selected.action_name,
                "selected_action_probability": selected.probability,
                "selected_action_log_probability": float(
                    selected.log_prob.detach().cpu().item()
                ),
                "masked_action_probabilities": action_probabilities,
                "coordinate_argmax": _coord_argmax(output.coord_logits),
                "action_frame_logits_shape": (
                    list(output.action_frame_logits.shape)
                    if output.action_frame_logits is not None
                    else None
                ),
            },
            current_frame=current_frame,
            next_frame=next_frame,
            frame_diagnostics={
                "decision_frame_index": decision_frame_index,
                "decision_frame_count": decision_frame_count,
                "next_decision_frame_index": next_decision_frame_index,
                "next_decision_frame_count": next_decision_frame_count,
                **frame_change,
            },
        )
        current_pending.world_snapshot = _capture_world_snapshot(model)
        _ensure_pending_latents(model=model, pending=current_pending, config=config)
        current_pending.lp_prior_model_step = world_model_step
        current_pending.one_step_prior_loss_details = _pending_prediction_loss_details(
            model,
            current_pending,
            config=config,
        )
        current_pending.one_step_prior_loss = float(
            current_pending.one_step_prior_loss_details["loss"]
        )
        current_pending.policy_probe_pre_world = _policy_probe(
            model=model,
            pending=current_pending,
            config=config,
        )
        current_pending.head_norms_pre_world = _action_head_norms(model)
        world_update_start = time.perf_counter()
        current_pending.world_update_loss = _run_world_updates(
            model=model,
            optimizer=world_optimizer,
            pending=current_pending,
            config=config,
        )
        world_update_steps = max(int(config.update.update_steps), 0)
        world_model_step += world_update_steps
        current_pending.lp_current_model_step = world_model_step
        current_pending.world_update_elapsed_seconds = (
            time.perf_counter() - world_update_start
        )
        current_pending.policy_probe_post_world = _policy_probe(
            model=model,
            pending=current_pending,
            config=config,
        )
        current_pending.head_norms_post_world = _action_head_norms(model)
        current_pending.one_step_current_loss_details = _pending_prediction_loss_details(
            model,
            current_pending,
            config=config,
        )
        current_pending.one_step_current_loss = float(
            current_pending.one_step_current_loss_details["loss"]
        )
        if (
            current_pending.one_step_prior_loss is not None
            and current_pending.one_step_current_loss is not None
        ):
            current_pending.one_step_learning_progress = (
                current_pending.one_step_prior_loss
                - current_pending.one_step_current_loss
            )
        current_pending.frame_prediction_artifact = _maybe_log_frame_prediction(
            model=model,
            frame_prediction_logger=frame_prediction_logger,
            pending=current_pending,
            config=config,
        )
        current_pending.latent_prediction_artifact = _maybe_log_latent_prediction(
            model=model,
            latent_prediction_logger=latent_prediction_logger,
            pending=current_pending,
            config=config,
        )

        pending_queue.append(current_pending)

        if len(pending_queue) >= learning_progress_horizon:
            pending = pending_queue.pop(0)
            lp_window = [pending, *pending_queue[: learning_progress_horizon - 1]]
            turn_payload = _finalize_pending_turn(
                model=model,
                pending=pending,
                lp_window=lp_window,
                lp_window_complete=len(lp_window) >= learning_progress_horizon,
                current_world_model_step=world_model_step,
                intrinsic_reward_baseline=None,
                learning_progress_rate_baseline=learning_progress_rate_baseline,
                policy_signal_abs_baseline=None,
                action_learning_progress_rate_baselines=(
                    action_learning_progress_rate_baselines
                ),
                config=config,
            )
            for return_ready_turn in policy_return_accumulator.add(turn_payload):
                for completed_turn in policy_accumulator.add(return_ready_turn):
                    total_reward += float(completed_turn.payload["reward"])
                    last_prediction_loss = float(
                        completed_turn.payload["prediction_loss"]
                    )
                    _append_turn(logger=logger, completed_turn=completed_turn)
                    _maybe_save_checkpoint(
                        model=model,
                        logger=logger,
                        config=config,
                        completed_turn=completed_turn.pending.transition.turn + 1,
                    )
            (
                learning_progress_rate_baseline,
                action_learning_progress_rate_baselines,
            ) = (
                turn_payload.next_learning_progress_rate_baseline,
                turn_payload.next_action_learning_progress_rate_baselines,
            )

        observation = next_observation

    while pending_queue:
        pending = pending_queue.pop(0)
        lp_window = [pending, *pending_queue[: learning_progress_horizon - 1]]
        turn_payload = _finalize_pending_turn(
            model=model,
            pending=pending,
            lp_window=lp_window,
            lp_window_complete=len(lp_window) >= learning_progress_horizon,
            current_world_model_step=world_model_step,
            intrinsic_reward_baseline=None,
            learning_progress_rate_baseline=learning_progress_rate_baseline,
            policy_signal_abs_baseline=None,
            action_learning_progress_rate_baselines=(
                action_learning_progress_rate_baselines
            ),
            config=config,
        )
        for return_ready_turn in policy_return_accumulator.add(turn_payload):
            for completed_turn in policy_accumulator.add(return_ready_turn):
                total_reward += float(completed_turn.payload["reward"])
                last_prediction_loss = float(completed_turn.payload["prediction_loss"])
                _append_turn(logger=logger, completed_turn=completed_turn)
                _maybe_save_checkpoint(
                    model=model,
                    logger=logger,
                    config=config,
                    completed_turn=completed_turn.pending.transition.turn + 1,
                )
        (
            learning_progress_rate_baseline,
            action_learning_progress_rate_baselines,
        ) = (
            turn_payload.next_learning_progress_rate_baseline,
            turn_payload.next_action_learning_progress_rate_baselines,
        )

    for return_ready_turn in policy_return_accumulator.flush():
        for completed_turn in policy_accumulator.add(return_ready_turn):
            total_reward += float(completed_turn.payload["reward"])
            last_prediction_loss = float(completed_turn.payload["prediction_loss"])
            _append_turn(logger=logger, completed_turn=completed_turn)
            _maybe_save_checkpoint(
                model=model,
                logger=logger,
                config=config,
                completed_turn=completed_turn.pending.transition.turn + 1,
            )

    for completed_turn in policy_accumulator.flush():
        total_reward += float(completed_turn.payload["reward"])
        last_prediction_loss = float(completed_turn.payload["prediction_loss"])
        _append_turn(logger=logger, completed_turn=completed_turn)
        _maybe_save_checkpoint(
            model=model,
            logger=logger,
            config=config,
            completed_turn=completed_turn.pending.transition.turn + 1,
        )

    video_recorder.close()
    model.save_adapter(logger.final_adapter_dir())
    summary = {
        "run_name": config.run_name,
        "game_id": session.game_id,
        "dry_run": config.dry_run,
        "turns": len(history.transitions),
        "total_reward": total_reward,
        "last_prediction_loss": last_prediction_loss,
        "output_dir": str(logger.output_dir),
        "world_parameter_count": _parameter_count(world_parameters),
        "policy_parameter_count": _parameter_count(policy_only_parameters),
        **_adapter_metadata(model),
        "world_model_step": world_model_step,
        "termination_reason": termination_reason,
        "terminal_turn": terminal_turn,
        "terminal_action_name": terminal_action_name,
        "terminal_observation_id": terminal_observation_id,
        "video_path": str(video_recorder.video_path) if video_recorder.enabled else None,
        "frame_manifest_path": (
            str(video_recorder.manifest_path) if video_recorder.enabled else None
        ),
        "recorded_video_frames": video_recorder.frame_count,
        "frame_prediction_dir": (
            str(frame_prediction_logger.prediction_dir)
            if frame_prediction_logger.enabled
            else None
        ),
        "frame_prediction_manifest_path": (
            str(frame_prediction_logger.manifest_path)
            if frame_prediction_logger.enabled
            else None
        ),
        "logged_frame_predictions": frame_prediction_logger.count,
        "latent_prediction_dir": (
            str(latent_prediction_logger.prediction_dir)
            if latent_prediction_logger.enabled
            else None
        ),
        "latent_prediction_manifest_path": (
            str(latent_prediction_logger.manifest_path)
            if latent_prediction_logger.enabled
            else None
        ),
        "logged_latent_predictions": latent_prediction_logger.count,
    }
    logger.write_summary(summary)
    return summary


@dataclass(slots=True)
class PendingTurn:
    transition: Transition
    prompt: str
    images: list[Any]
    action_space: tuple[Any, ...]
    selected_action_index: int
    selected_x: int | None
    selected_y: int | None
    score_delta: float
    interaction_elapsed_seconds: float
    model_input: dict[str, Any]
    model_output: dict[str, Any]
    current_frame: Any
    next_frame: Any
    frame_diagnostics: dict[str, Any]
    world_snapshot: list[torch.Tensor] = field(default_factory=list)
    lp_prior_model_step: int = 0
    lp_current_model_step: int = 0
    world_update_loss: float = 0.0
    world_update_elapsed_seconds: float = 0.0
    one_step_prior_loss: float | None = None
    one_step_current_loss: float | None = None
    one_step_learning_progress: float | None = None
    current_latent_grid: torch.Tensor | None = None
    target_latent_grid: torch.Tensor | None = None
    latent_changed_patch_mask: torch.Tensor | None = None
    one_step_prior_loss_details: dict[str, Any] = field(default_factory=dict)
    one_step_current_loss_details: dict[str, Any] = field(default_factory=dict)
    policy_probe_pre_world: dict[str, Any] = field(default_factory=dict)
    policy_probe_post_world: dict[str, Any] = field(default_factory=dict)
    head_norms_pre_world: dict[str, float] = field(default_factory=dict)
    head_norms_post_world: dict[str, float] = field(default_factory=dict)
    frame_prediction_artifact: dict[str, Any] | None = None
    latent_prediction_artifact: dict[str, Any] | None = None


@dataclass(slots=True)
class FinalizedTurn:
    pending: PendingTurn
    payload: dict[str, Any]
    include_policy_update: bool
    policy_advantage: float
    next_intrinsic_reward_baseline: float | None
    next_learning_progress_rate_baseline: float | None
    next_policy_signal_abs_baseline: float | None
    next_action_learning_progress_rate_baselines: dict[str, float]


@dataclass(slots=True)
class PolicyUpdateResult:
    loss: float
    elapsed_seconds: float
    batch_turns: list[int]
    batch_size: int
    flushed: bool = False


class PolicyReturnAccumulator:
    """Delay policy credit until a discounted future LP-rate return is available."""

    def __init__(self, *, config: Any) -> None:
        self.config = config
        self.horizon = _policy_return_horizon(config)
        self._pending: list[FinalizedTurn] = []
        self._intrinsic_reward_baseline: float | None = None
        self._policy_signal_abs_baseline: float | None = None

    def add(self, finalized: FinalizedTurn) -> list[FinalizedTurn]:
        self._pending.append(finalized)
        if len(self._pending) < self.horizon:
            return []
        return [self._finalize_oldest(return_complete=True)]

    def flush(self) -> list[FinalizedTurn]:
        completed: list[FinalizedTurn] = []
        while self._pending:
            completed.append(self._finalize_oldest(return_complete=False))
        return completed

    def _finalize_oldest(self, *, return_complete: bool) -> FinalizedTurn:
        finalized = self._pending.pop(0)
        return_window = [finalized, *self._pending[: self.horizon - 1]]
        (
            self._intrinsic_reward_baseline,
            self._policy_signal_abs_baseline,
        ) = _apply_policy_return_credit(
            finalized=finalized,
            return_window=return_window,
            return_complete=return_complete,
            intrinsic_reward_baseline=self._intrinsic_reward_baseline,
            policy_signal_abs_baseline=self._policy_signal_abs_baseline,
            config=self.config,
        )
        return finalized


def _apply_policy_return_credit(
    *,
    finalized: FinalizedTurn,
    return_window: Sequence[FinalizedTurn],
    return_complete: bool,
    intrinsic_reward_baseline: float | None,
    policy_signal_abs_baseline: float | None,
    config: Any,
) -> tuple[float | None, float | None]:
    pending = finalized.pending
    payload = finalized.payload
    reward_inputs = payload["reward_inputs"]
    transition_metadata = payload["metadata"]
    return_discount = _policy_return_discount(config)
    return_weights = _discounted_window_weights(len(return_window), return_discount)
    return_rates = [
        float(item.payload["reward_inputs"]["learning_progress_rate"])
        for item in return_window
    ]
    policy_lp_return = _weighted_sum(return_rates, return_weights)
    policy_intrinsic_reward = (
        float(config.rewards.learning_progress_weight) * float(policy_lp_return)
    )

    policy_advantage_baseline_mode = _policy_advantage_baseline_mode(config)
    policy_advantage_normalization = _policy_advantage_normalization(config)
    policy_advantage_normalization_denominator = (
        _policy_advantage_normalization_denominator(
            policy_signal_abs_baseline=policy_signal_abs_baseline,
            policy_intrinsic_reward=policy_intrinsic_reward,
            config=config,
        )
    )
    next_policy_signal_abs_baseline = _next_policy_signal_abs_baseline(
        policy_signal_abs_baseline=policy_signal_abs_baseline,
        policy_intrinsic_reward=policy_intrinsic_reward,
        config=config,
    )
    policy_update_skipped_reason = _policy_update_skipped_reason(
        pending=pending,
        intrinsic_reward_baseline=intrinsic_reward_baseline,
        requires_intrinsic_baseline=(policy_advantage_baseline_mode == "ema"),
        config=config,
    )
    include_policy_update = policy_update_skipped_reason is None
    raw_policy_advantage = 0.0
    policy_advantage = 0.0
    if include_policy_update:
        raw_policy_advantage = _raw_policy_advantage(
            policy_intrinsic_reward=policy_intrinsic_reward,
            intrinsic_reward_baseline=intrinsic_reward_baseline,
            baseline_mode=policy_advantage_baseline_mode,
        )
        policy_advantage = _normalized_policy_advantage(
            raw_policy_advantage=raw_policy_advantage,
            denominator=policy_advantage_normalization_denominator,
            normalization=policy_advantage_normalization,
        )

    baseline_beta = float(getattr(config.update, "reward_baseline_beta", 0.9))
    baseline_beta = max(0.0, min(0.999, baseline_beta))
    next_intrinsic_reward_baseline = (
        _next_intrinsic_reward_baseline(
            intrinsic_reward_baseline=intrinsic_reward_baseline,
            policy_intrinsic_reward=policy_intrinsic_reward,
            policy_update_skipped_reason=policy_update_skipped_reason,
            baseline_beta=baseline_beta,
        )
        if policy_advantage_baseline_mode == "ema"
        else None
    )

    effective_update_steps = _effective_update_steps(
        config,
        include_policy_update=include_policy_update,
    )
    reward_elapsed_seconds = (
        pending.interaction_elapsed_seconds + pending.world_update_elapsed_seconds
    )
    reward = compute_reward(
        config=config.rewards,
        score_delta=pending.score_delta,
        learning_progress=float(reward_inputs["learning_progress"]),
        elapsed_seconds=reward_elapsed_seconds,
        update_steps=effective_update_steps,
    )

    finalized.include_policy_update = include_policy_update
    finalized.policy_advantage = policy_advantage
    finalized.next_intrinsic_reward_baseline = next_intrinsic_reward_baseline
    finalized.next_policy_signal_abs_baseline = next_policy_signal_abs_baseline

    payload["reward"] = reward.total
    payload["reward_breakdown"] = reward.to_dict()
    pending.transition.reward = reward.total

    return_turns = [item.pending.transition.turn for item in return_window]
    policy_updates = {
        "policy_intrinsic_reward": policy_intrinsic_reward,
        "policy_lp_return": policy_lp_return,
        "policy_lp_return_turns": return_turns,
        "policy_lp_return_weights": return_weights,
        "policy_lp_return_rates": return_rates,
        "policy_return_complete": bool(return_complete),
        "policy_return_discount": return_discount,
        "policy_return_horizon": _policy_return_horizon(config),
        "policy_learning_progress_signal": "discounted_rate_return",
        "intrinsic_reward_baseline": intrinsic_reward_baseline,
        "next_intrinsic_reward_baseline": next_intrinsic_reward_baseline,
        "raw_policy_advantage": raw_policy_advantage,
        "policy_advantage": policy_advantage,
        "policy_advantage_baseline": policy_advantage_baseline_mode,
        "policy_advantage_normalization": policy_advantage_normalization,
        "policy_advantage_normalization_denominator": (
            policy_advantage_normalization_denominator
        ),
        "policy_signal_abs_baseline": policy_signal_abs_baseline,
        "next_policy_signal_abs_baseline": next_policy_signal_abs_baseline,
        "policy_update_skipped_reason": policy_update_skipped_reason,
        "policy_update_pending": bool(include_policy_update),
        "update_steps": effective_update_steps,
    }
    for target in (transition_metadata, reward_inputs):
        target.update(policy_updates)

    return next_intrinsic_reward_baseline, next_policy_signal_abs_baseline


def _finalize_pending_turn(
    *,
    model: Any,
    pending: PendingTurn,
    lp_window: list[PendingTurn],
    lp_window_complete: bool,
    current_world_model_step: int,
    intrinsic_reward_baseline: float | None,
    learning_progress_rate_baseline: float | None,
    policy_signal_abs_baseline: float | None,
    action_learning_progress_rate_baselines: dict[str, float],
    config: Any,
) -> FinalizedTurn:
    window_discount = _learning_progress_discount(config)
    window_metrics = _window_learning_progress(
        model,
        snapshot=pending.world_snapshot,
        lp_window=lp_window,
        discount=window_discount,
        config=config,
    )
    raw_window_learning_progress = float(window_metrics["learning_progress"])
    learning_progress_loss_scale = _learning_progress_loss_scale(
        window_metrics,
        config=config,
    )
    raw_learning_progress = raw_window_learning_progress / learning_progress_loss_scale
    learning_progress_rate_beta = _learning_progress_rate_beta(config)
    selected_action_name = str(pending.model_output["selected_action_name"])
    action_conditioned_lp_baseline = (
        _action_conditioned_learning_progress_baseline(config)
    )
    selected_learning_progress_rate_baseline = _selected_learning_progress_rate_baseline(
        selected_action_name=selected_action_name,
        learning_progress_rate_baseline=learning_progress_rate_baseline,
        action_learning_progress_rate_baselines=action_learning_progress_rate_baselines,
        action_conditioned=action_conditioned_lp_baseline,
    )
    learning_progress_rate_reference = _learning_progress_rate_reference(
        learning_progress_rate_baseline=selected_learning_progress_rate_baseline,
        raw_learning_progress=raw_learning_progress,
    )
    learning_progress_rate = raw_learning_progress - learning_progress_rate_reference
    next_selected_learning_progress_rate_baseline = _next_learning_progress_rate_baseline(
        learning_progress_rate_baseline=selected_learning_progress_rate_baseline,
        raw_learning_progress=raw_learning_progress,
        beta=learning_progress_rate_beta,
    )
    next_learning_progress_rate_baseline = learning_progress_rate_baseline
    next_action_learning_progress_rate_baselines = dict(
        action_learning_progress_rate_baselines
    )
    if action_conditioned_lp_baseline:
        next_action_learning_progress_rate_baselines[selected_action_name] = (
            next_selected_learning_progress_rate_baseline
        )
    else:
        next_learning_progress_rate_baseline = (
            next_selected_learning_progress_rate_baseline
        )
    learning_progress = learning_progress_rate
    lp_prior_loss = float(window_metrics["weighted_prior_loss"])
    lp_current_loss = float(window_metrics["weighted_current_loss"])
    holdout_pre_loss = None
    holdout_post_loss = None
    holdout_learning_progress = None
    holdout_eval_turn = None
    if len(lp_window) > 1:
        holdout_eval_turn = lp_window[1].transition.turn
        holdout_pre_loss = window_metrics["prior_losses"][1]
        holdout_post_loss = window_metrics["current_losses"][1]
        holdout_learning_progress = window_metrics["transition_progress"][1]

    policy_intrinsic_reward = (
        float(config.rewards.learning_progress_weight) * float(learning_progress)
    )
    policy_advantage_baseline_mode = _policy_advantage_baseline_mode(config)
    policy_advantage_normalization = _policy_advantage_normalization(config)
    policy_advantage_normalization_denominator = (
        _policy_advantage_normalization_denominator(
            policy_signal_abs_baseline=policy_signal_abs_baseline,
            policy_intrinsic_reward=policy_intrinsic_reward,
            config=config,
        )
    )
    next_policy_signal_abs_baseline = _next_policy_signal_abs_baseline(
        policy_signal_abs_baseline=policy_signal_abs_baseline,
        policy_intrinsic_reward=policy_intrinsic_reward,
        config=config,
    )
    policy_update_skipped_reason = _policy_update_skipped_reason(
        pending=pending,
        intrinsic_reward_baseline=intrinsic_reward_baseline,
        requires_intrinsic_baseline=(policy_advantage_baseline_mode == "ema"),
        config=config,
    )
    include_policy_update = policy_update_skipped_reason is None
    effective_update_steps = _effective_update_steps(
        config,
        include_policy_update=include_policy_update,
    )
    reward_elapsed_seconds = (
        pending.interaction_elapsed_seconds + pending.world_update_elapsed_seconds
    )
    reward = compute_reward(
        config=config.rewards,
        score_delta=pending.score_delta,
        learning_progress=learning_progress,
        elapsed_seconds=reward_elapsed_seconds,
        update_steps=effective_update_steps,
    )
    raw_policy_advantage = 0.0
    policy_advantage = 0.0
    if include_policy_update:
        raw_policy_advantage = _raw_policy_advantage(
            policy_intrinsic_reward=policy_intrinsic_reward,
            intrinsic_reward_baseline=intrinsic_reward_baseline,
            baseline_mode=policy_advantage_baseline_mode,
        )
        policy_advantage = _normalized_policy_advantage(
            raw_policy_advantage=raw_policy_advantage,
            denominator=policy_advantage_normalization_denominator,
            normalization=policy_advantage_normalization,
        )
    policy_clip_epsilon = _policy_clip_epsilon(config)
    old_policy_log_probability = _old_policy_log_probability(pending)
    policy_log_probability_pre_update = float(
        pending.policy_probe_post_world["selected_action_log_probability"]
    )
    policy_probability_ratio_pre_update = _policy_probability_ratio(
        new_log_probability=policy_log_probability_pre_update,
        old_log_probability=old_policy_log_probability,
    )
    policy_clipped_ratio_pre_update = _clipped_probability_ratio(
        policy_probability_ratio_pre_update,
        policy_clip_epsilon,
    )
    baseline_beta = float(getattr(config.update, "reward_baseline_beta", 0.9))
    baseline_beta = max(0.0, min(0.999, baseline_beta))
    next_intrinsic_reward_baseline = (
        _next_intrinsic_reward_baseline(
            intrinsic_reward_baseline=intrinsic_reward_baseline,
            policy_intrinsic_reward=policy_intrinsic_reward,
            policy_update_skipped_reason=policy_update_skipped_reason,
            baseline_beta=baseline_beta,
        )
        if policy_advantage_baseline_mode == "ema"
        else None
    )

    policy_update_loss = 0.0
    policy_update_elapsed_seconds = 0.0
    policy_probe_post_policy = pending.policy_probe_post_world
    policy_log_probability_post_update = float(
        policy_probe_post_policy["selected_action_log_probability"]
    )
    policy_probability_ratio_post_update = _policy_probability_ratio(
        new_log_probability=policy_log_probability_post_update,
        old_log_probability=old_policy_log_probability,
    )
    policy_clipped_ratio_post_update = _clipped_probability_ratio(
        policy_probability_ratio_post_update,
        policy_clip_epsilon,
    )
    head_norms_post_policy = pending.head_norms_post_world
    total_elapsed_seconds = (
        pending.interaction_elapsed_seconds
        + pending.world_update_elapsed_seconds
        + policy_update_elapsed_seconds
    )
    prediction_loss = _pending_prediction_loss_value(
        model,
        pending,
        config=config,
    )
    if prediction_loss is None:
        raise RuntimeError("finalized transition received an empty prediction loss")

    transition = pending.transition
    transition.prediction_loss = prediction_loss
    transition.reward = reward.total
    transition.log_probability = _log_probability_value(
        model=model,
        pending=pending,
        config=config,
    )
    adapter_metadata = _adapter_metadata(model)
    transition.metadata.update(
        {
            "score_delta": pending.score_delta,
            "decision_frame_index": pending.frame_diagnostics[
                "decision_frame_index"
            ],
            "decision_frame_count": pending.frame_diagnostics[
                "decision_frame_count"
            ],
            "next_decision_frame_index": pending.frame_diagnostics[
                "next_decision_frame_index"
            ],
            "next_decision_frame_count": pending.frame_diagnostics[
                "next_decision_frame_count"
            ],
            "frame_changed_pixels": pending.frame_diagnostics[
                "frame_changed_pixels"
            ],
            "frame_changed_fraction": pending.frame_diagnostics[
                "frame_changed_fraction"
            ],
            "learning_progress": learning_progress,
            "raw_learning_progress": raw_learning_progress,
            "learning_progress_loss_scale": learning_progress_loss_scale,
            "latent_learning_progress_normalization": (
                _uses_latent_world_loss(config)
                and _latent_learning_progress_normalization(config)
            ),
            "learning_progress_rate": learning_progress_rate,
            "learning_progress_rate_reference": learning_progress_rate_reference,
            "learning_progress_rate_baseline": selected_learning_progress_rate_baseline,
            "next_learning_progress_rate_baseline": next_learning_progress_rate_baseline,
            "learning_progress_rate_baseline_scope": (
                "action" if action_conditioned_lp_baseline else "global"
            ),
            "learning_progress_rate_baseline_action": (
                selected_action_name if action_conditioned_lp_baseline else None
            ),
            "action_learning_progress_rate_baselines": dict(
                action_learning_progress_rate_baselines
            ),
            "next_action_learning_progress_rate_baselines": dict(
                next_action_learning_progress_rate_baselines
            ),
            "learning_progress_rate_beta": learning_progress_rate_beta,
            "action_probability": pending.model_output[
                "selected_action_probability"
            ],
            "policy_intrinsic_reward": policy_intrinsic_reward,
            "raw_policy_advantage": raw_policy_advantage,
            "policy_advantage": policy_advantage,
            "intrinsic_reward_baseline": intrinsic_reward_baseline,
            "policy_advantage_baseline": policy_advantage_baseline_mode,
            "policy_advantage_normalization": policy_advantage_normalization,
            "policy_advantage_normalization_denominator": (
                policy_advantage_normalization_denominator
            ),
            "policy_signal_abs_baseline": policy_signal_abs_baseline,
            "next_policy_signal_abs_baseline": next_policy_signal_abs_baseline,
            "policy_update_skipped_reason": policy_update_skipped_reason,
            "policy_loss_objective": "ppo_clipped_ratio",
            "policy_clip_epsilon": policy_clip_epsilon,
            "old_policy_log_probability": old_policy_log_probability,
            "policy_log_probability_pre_update": policy_log_probability_pre_update,
            "policy_log_probability_post_update": policy_log_probability_post_update,
            "policy_probability_ratio_pre_update": policy_probability_ratio_pre_update,
            "policy_probability_ratio_post_update": policy_probability_ratio_post_update,
            "policy_clipped_ratio_pre_update": policy_clipped_ratio_pre_update,
            "policy_clipped_ratio_post_update": policy_clipped_ratio_post_update,
            **adapter_metadata,
        }
    )

    payload = {
        **transition_to_json(transition),
        "elapsed_seconds": total_elapsed_seconds,
        "interaction_elapsed_seconds": pending.interaction_elapsed_seconds,
        "update_elapsed_seconds": (
            pending.world_update_elapsed_seconds + policy_update_elapsed_seconds
        ),
        "world_update_loss": pending.world_update_loss,
        "policy_update_loss": policy_update_loss,
        "update_loss": pending.world_update_loss + policy_update_loss,
        "frame_prediction_artifact": pending.frame_prediction_artifact,
        "latent_prediction_artifact": pending.latent_prediction_artifact,
        "reward_breakdown": reward.to_dict(),
        "reward_inputs": {
            "raw_elapsed_seconds": reward_elapsed_seconds,
            "interaction_elapsed_seconds": pending.interaction_elapsed_seconds,
            "world_update_elapsed_seconds": pending.world_update_elapsed_seconds,
            "policy_update_elapsed_seconds": policy_update_elapsed_seconds,
            "update_steps": effective_update_steps,
            "score_delta": pending.score_delta,
            "decision_frame_index": pending.frame_diagnostics[
                "decision_frame_index"
            ],
            "decision_frame_count": pending.frame_diagnostics[
                "decision_frame_count"
            ],
            "next_decision_frame_index": pending.frame_diagnostics[
                "next_decision_frame_index"
            ],
            "next_decision_frame_count": pending.frame_diagnostics[
                "next_decision_frame_count"
            ],
            "frame_changed_pixels": pending.frame_diagnostics[
                "frame_changed_pixels"
            ],
            "frame_changed_fraction": pending.frame_diagnostics[
                "frame_changed_fraction"
            ],
            "learning_progress": learning_progress,
            "raw_learning_progress": raw_learning_progress,
            "learning_progress_loss_scale": learning_progress_loss_scale,
            "latent_learning_progress_normalization": (
                _uses_latent_world_loss(config)
                and _latent_learning_progress_normalization(config)
            ),
            "learning_progress_rate": learning_progress_rate,
            "learning_progress_signal": "rate",
            "learning_progress_rate_reference": learning_progress_rate_reference,
            "learning_progress_rate_baseline": selected_learning_progress_rate_baseline,
            "next_learning_progress_rate_baseline": next_learning_progress_rate_baseline,
            "learning_progress_rate_baseline_scope": (
                "action" if action_conditioned_lp_baseline else "global"
            ),
            "learning_progress_rate_baseline_action": (
                selected_action_name if action_conditioned_lp_baseline else None
            ),
            "action_learning_progress_rate_baselines": dict(
                action_learning_progress_rate_baselines
            ),
            "next_action_learning_progress_rate_baselines": dict(
                next_action_learning_progress_rate_baselines
            ),
            "learning_progress_rate_beta": learning_progress_rate_beta,
            "frame_loss_mode": (
                "residual_ce" if _residual_frame_prediction(config) else "ce"
            ),
            "world_loss_mode": _world_loss_mode(config),
            "latent_loss_weight": float(
                getattr(config.update, "latent_loss_weight", 1.0)
            ),
            "latent_changed_patch_weight": _latent_changed_patch_weight(config),
            "latent_huber_beta": _latent_huber_beta(config),
            "latent_cosine_loss_weight": _latent_cosine_loss_weight(config),
            "latent_cosine_min_delta_norm": _latent_cosine_min_delta_norm(config),
            "latent_grid_shape": _latent_detail_value(
                pending.one_step_current_loss_details,
                "grid_shape",
            ),
            "latent_dim": _latent_detail_value(
                pending.one_step_current_loss_details,
                "latent_dim",
            ),
            "latent_changed_patch_count": _latent_detail_value(
                pending.one_step_current_loss_details,
                "changed_patch_count",
            ),
            "latent_changed_patch_fraction": _latent_detail_value(
                pending.one_step_current_loss_details,
                "changed_patch_fraction",
            ),
            "one_step_pre_latent_loss": pending.one_step_prior_loss_details.get(
                "latent_loss"
            ),
            "one_step_post_latent_loss": pending.one_step_current_loss_details.get(
                "latent_loss"
            ),
            "one_step_pre_pixel_loss": pending.one_step_prior_loss_details.get(
                "pixel_loss"
            ),
            "one_step_post_pixel_loss": pending.one_step_current_loss_details.get(
                "pixel_loss"
            ),
            "one_step_post_latent_changed_patch_loss": _latent_detail_value(
                pending.one_step_current_loss_details,
                "changed_patch_loss",
            ),
            "one_step_post_latent_unchanged_patch_loss": _latent_detail_value(
                pending.one_step_current_loss_details,
                "unchanged_patch_loss",
            ),
            "one_step_post_latent_huber_loss": _latent_detail_value(
                pending.one_step_current_loss_details,
                "huber_loss",
            ),
            "one_step_post_latent_cosine_loss": _latent_detail_value(
                pending.one_step_current_loss_details,
                "cosine_loss",
            ),
            "one_step_post_latent_cosine_patch_count": _latent_detail_value(
                pending.one_step_current_loss_details,
                "cosine_patch_count",
            ),
            "residual_frame_prediction": _residual_frame_prediction(config),
            "residual_frame_logit_bias": _residual_frame_logit_bias(config),
            "lp_eval_turn": pending.transition.turn,
            "lp_observation_source": "discounted_window_snapshot",
            "lp_same_observation": True,
            "lp_prior_model_step": int(pending.lp_prior_model_step),
            "lp_current_model_step": int(current_world_model_step),
            "lp_prior_loss": lp_prior_loss,
            "lp_current_loss": lp_current_loss,
            "lp_pre_loss": lp_prior_loss,
            "lp_post_loss": lp_current_loss,
            "lp_window_turns": [
                item.transition.turn for item in lp_window
            ],
            "lp_window_weights": window_metrics["weights"],
            "lp_window_prior_losses": window_metrics["prior_losses"],
            "lp_window_current_losses": window_metrics["current_losses"],
            "lp_window_transition_progress": window_metrics[
                "transition_progress"
            ],
            "lp_window_size": len(lp_window),
            "lp_window_discount": window_discount,
            "lp_window_complete": bool(lp_window_complete),
            "raw_window_learning_progress": raw_window_learning_progress,
            "normalized_window_learning_progress": raw_learning_progress,
            "one_step_pre_loss": pending.one_step_prior_loss,
            "one_step_post_loss": pending.one_step_current_loss,
            "one_step_learning_progress": pending.one_step_learning_progress,
            "holdout_eval_turn": holdout_eval_turn,
            "holdout_pre_loss": holdout_pre_loss,
            "holdout_post_loss": holdout_post_loss,
            "holdout_learning_progress": holdout_learning_progress,
            "policy_intrinsic_reward": policy_intrinsic_reward,
            "policy_lp_return": learning_progress,
            "policy_lp_return_turns": [pending.transition.turn],
            "policy_lp_return_weights": [1.0],
            "policy_lp_return_rates": [learning_progress_rate],
            "policy_return_complete": False,
            "policy_return_discount": _policy_return_discount(config),
            "intrinsic_reward_baseline": intrinsic_reward_baseline,
            "next_intrinsic_reward_baseline": next_intrinsic_reward_baseline,
            "raw_policy_advantage": raw_policy_advantage,
            "policy_advantage": policy_advantage,
            "policy_advantage_baseline": policy_advantage_baseline_mode,
            "policy_advantage_normalization": policy_advantage_normalization,
            "policy_advantage_normalization_denominator": (
                policy_advantage_normalization_denominator
            ),
            "policy_signal_abs_baseline": policy_signal_abs_baseline,
            "next_policy_signal_abs_baseline": next_policy_signal_abs_baseline,
            "policy_update_skipped_reason": policy_update_skipped_reason,
            "policy_loss_objective": "ppo_clipped_ratio",
            "policy_clip_epsilon": policy_clip_epsilon,
            "old_policy_log_probability": old_policy_log_probability,
            "policy_log_probability_pre_update": policy_log_probability_pre_update,
            "policy_log_probability_post_update": policy_log_probability_post_update,
            "policy_probability_ratio_pre_update": policy_probability_ratio_pre_update,
            "policy_probability_ratio_post_update": policy_probability_ratio_post_update,
            "policy_clipped_ratio_pre_update": policy_clipped_ratio_pre_update,
            "policy_clipped_ratio_post_update": policy_clipped_ratio_post_update,
            "policy_update_pending": bool(include_policy_update),
            "policy_update_accumulation_steps": _policy_update_accumulation_steps(
                config
            ),
            "policy_update_batch_size": 0,
            "policy_update_batch_turns": [],
            "policy_update_batch_flushed": False,
            "policy_update_batch_loss": 0.0,
            **adapter_metadata,
        },
        "model_input": pending.model_input,
        "model_output": pending.model_output,
        "policy_diagnostics": {
            "pre_world_update": pending.policy_probe_pre_world,
            "post_world_update": pending.policy_probe_post_world,
            "post_policy_update": policy_probe_post_policy,
            "head_norms_pre_world_update": pending.head_norms_pre_world,
            "head_norms_post_world_update": pending.head_norms_post_world,
            "head_norms_post_policy_update": head_norms_post_policy,
            "world_update_selected_probability_delta": (
                pending.policy_probe_post_world["selected_action_probability"]
                - pending.policy_probe_pre_world["selected_action_probability"]
            ),
            "world_update_max_probability_delta": (
                pending.policy_probe_post_world["max_action_probability"]
                - pending.policy_probe_pre_world["max_action_probability"]
            ),
            "policy_update_selected_probability_delta": (
                policy_probe_post_policy["selected_action_probability"]
                - pending.policy_probe_post_world["selected_action_probability"]
            ),
            "policy_update_max_probability_delta": (
                policy_probe_post_policy["max_action_probability"]
                - pending.policy_probe_post_world["max_action_probability"]
            ),
            "world_update_action_head_weight_l2_delta": (
                pending.head_norms_post_world["action_head_weight_l2"]
                - pending.head_norms_pre_world["action_head_weight_l2"]
            ),
            "policy_update_action_head_weight_l2_delta": (
                head_norms_post_policy["action_head_weight_l2"]
                - pending.head_norms_post_world["action_head_weight_l2"]
            ),
            "world_update_action_head_bias_l2_delta": (
                pending.head_norms_post_world["action_head_bias_l2"]
                - pending.head_norms_pre_world["action_head_bias_l2"]
            ),
            "policy_update_action_head_bias_l2_delta": (
                head_norms_post_policy["action_head_bias_l2"]
                - pending.head_norms_post_world["action_head_bias_l2"]
            ),
            "ppo": {
                "objective": "clipped_ratio",
                "clip_epsilon": policy_clip_epsilon,
                "old_policy_log_probability": old_policy_log_probability,
                "pre_update_log_probability": policy_log_probability_pre_update,
                "post_update_log_probability": policy_log_probability_post_update,
                "pre_update_probability_ratio": policy_probability_ratio_pre_update,
                "post_update_probability_ratio": policy_probability_ratio_post_update,
                "pre_update_clipped_ratio": policy_clipped_ratio_pre_update,
                "post_update_clipped_ratio": policy_clipped_ratio_post_update,
            },
            "policy_update_batch": {
                "accumulation_steps": _policy_update_accumulation_steps(config),
                "batch_size": 0,
                "batch_turns": [],
                "flushed": False,
                "loss": 0.0,
            },
            "world_optimizer_scope": "world_model_parameters",
            "policy_optimizer_scope": "policy_parameters",
            **adapter_metadata,
        },
        "observation_id": transition.observation.id,
        "next_observation_id": transition.next_observation.id,
    }
    return FinalizedTurn(
        pending=pending,
        payload=payload,
        include_policy_update=include_policy_update,
        policy_advantage=policy_advantage,
        next_intrinsic_reward_baseline=next_intrinsic_reward_baseline,
        next_learning_progress_rate_baseline=next_learning_progress_rate_baseline,
        next_policy_signal_abs_baseline=next_policy_signal_abs_baseline,
        next_action_learning_progress_rate_baselines=(
            next_action_learning_progress_rate_baselines
        ),
    )


class PolicyUpdateAccumulator:
    """Delay policy-gradient updates until enough finalized rewards are available."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: torch.optim.Optimizer,
        config: Any,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.accumulation_steps = _policy_update_accumulation_steps(config)
        self._pending: list[FinalizedTurn] = []

    def add(self, finalized: FinalizedTurn) -> list[FinalizedTurn]:
        if not finalized.include_policy_update:
            _apply_policy_update_result(
                model=self.model,
                finalized=finalized,
                result=None,
                config=self.config,
            )
            return [finalized]
        self._pending.append(finalized)
        if len(self._pending) < self.accumulation_steps:
            return []
        return self._flush(flushed=False)

    def flush(self) -> list[FinalizedTurn]:
        return self._flush(flushed=True)

    def _flush(self, *, flushed: bool) -> list[FinalizedTurn]:
        if not self._pending:
            return []
        batch = list(self._pending)
        self._pending.clear()
        policy_update_start = time.perf_counter()
        loss = _run_policy_batch_updates(
            model=self.model,
            optimizer=self.optimizer,
            finalized_turns=batch,
            config=self.config,
        )
        result = PolicyUpdateResult(
            loss=loss,
            elapsed_seconds=time.perf_counter() - policy_update_start,
            batch_turns=[item.pending.transition.turn for item in batch],
            batch_size=len(batch),
            flushed=flushed,
        )
        for finalized in batch:
            _apply_policy_update_result(
                model=self.model,
                finalized=finalized,
                result=result,
                config=self.config,
            )
        return batch


def _apply_policy_update_result(
    *,
    model: Any,
    finalized: FinalizedTurn,
    result: PolicyUpdateResult | None,
    config: Any,
) -> None:
    pending = finalized.pending
    payload = finalized.payload
    reward_inputs = payload["reward_inputs"]
    diagnostics = payload["policy_diagnostics"]
    old_policy_log_probability = float(reward_inputs["old_policy_log_probability"])
    policy_clip_epsilon = float(reward_inputs["policy_clip_epsilon"])

    if result is None:
        policy_update_elapsed_seconds = 0.0
        policy_update_loss = 0.0
        batch_turns: list[int] = []
        batch_size = 0
        batch_flushed = False
        batch_loss = 0.0
    else:
        policy_update_elapsed_seconds = result.elapsed_seconds / max(
            int(result.batch_size),
            1,
        )
        policy_update_loss = result.loss / max(int(result.batch_size), 1)
        batch_turns = result.batch_turns
        batch_size = result.batch_size
        batch_flushed = result.flushed
        batch_loss = result.loss

    policy_probe_post_policy = _policy_probe(
        model=model,
        pending=pending,
        config=config,
    )
    policy_log_probability_post_update = float(
        policy_probe_post_policy["selected_action_log_probability"]
    )
    policy_probability_ratio_post_update = _policy_probability_ratio(
        new_log_probability=policy_log_probability_post_update,
        old_log_probability=old_policy_log_probability,
    )
    policy_clipped_ratio_post_update = _clipped_probability_ratio(
        policy_probability_ratio_post_update,
        policy_clip_epsilon,
    )
    head_norms_post_policy = _action_head_norms(model)

    payload["policy_update_loss"] = policy_update_loss
    payload["update_loss"] = payload["world_update_loss"] + policy_update_loss
    payload["elapsed_seconds"] += policy_update_elapsed_seconds
    payload["update_elapsed_seconds"] += policy_update_elapsed_seconds
    payload["log_probability"] = _log_probability_value(
        model=model,
        pending=pending,
        config=config,
    )

    transition_metadata = payload["metadata"]
    for target in (transition_metadata, reward_inputs):
        target["policy_log_probability_post_update"] = (
            policy_log_probability_post_update
        )
        target["policy_probability_ratio_post_update"] = (
            policy_probability_ratio_post_update
        )
        target["policy_clipped_ratio_post_update"] = (
            policy_clipped_ratio_post_update
        )
    reward_inputs["policy_update_elapsed_seconds"] = policy_update_elapsed_seconds
    reward_inputs["policy_update_pending"] = False
    reward_inputs["policy_update_batch_size"] = batch_size
    reward_inputs["policy_update_batch_turns"] = batch_turns
    reward_inputs["policy_update_batch_flushed"] = batch_flushed
    reward_inputs["policy_update_batch_loss"] = batch_loss

    diagnostics["post_policy_update"] = policy_probe_post_policy
    diagnostics["head_norms_post_policy_update"] = head_norms_post_policy
    diagnostics["policy_update_selected_probability_delta"] = (
        policy_probe_post_policy["selected_action_probability"]
        - pending.policy_probe_post_world["selected_action_probability"]
    )
    diagnostics["policy_update_max_probability_delta"] = (
        policy_probe_post_policy["max_action_probability"]
        - pending.policy_probe_post_world["max_action_probability"]
    )
    diagnostics["policy_update_action_head_weight_l2_delta"] = (
        head_norms_post_policy["action_head_weight_l2"]
        - pending.head_norms_post_world["action_head_weight_l2"]
    )
    diagnostics["policy_update_action_head_bias_l2_delta"] = (
        head_norms_post_policy["action_head_bias_l2"]
        - pending.head_norms_post_world["action_head_bias_l2"]
    )
    diagnostics["ppo"]["post_update_log_probability"] = (
        policy_log_probability_post_update
    )
    diagnostics["ppo"]["post_update_probability_ratio"] = (
        policy_probability_ratio_post_update
    )
    diagnostics["ppo"]["post_update_clipped_ratio"] = (
        policy_clipped_ratio_post_update
    )
    diagnostics["policy_update_batch"] = {
        "accumulation_steps": _policy_update_accumulation_steps(config),
        "batch_size": batch_size,
        "batch_turns": batch_turns,
        "flushed": batch_flushed,
        "loss": batch_loss,
    }


def _run_world_updates(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    pending: PendingTurn,
    config: Any,
) -> float:
    last_loss = 0.0
    _ensure_pending_latents(model=model, pending=pending, config=config)
    model.train()
    with _use_world_adapter(model), _temporarily_requires_grad(
        policy_parameters(model),
        False,
    ):
        for _ in range(int(config.update.update_steps)):
            optimizer.zero_grad(set_to_none=True)
            predicted_delta = None
            if _uses_latent_world_loss(config):
                output, predicted_delta = model.forward_with_latent_delta(
                    pending.prompt,
                    pending.images,
                    action_index=pending.selected_action_index,
                    selected_x=pending.selected_x,
                    selected_y=pending.selected_y,
                    include_frame_logits=_uses_pixel_world_loss(config),
                )
            else:
                output = model(
                    pending.prompt,
                    pending.images,
                    include_frame_logits=_uses_pixel_world_loss(config),
                )
            loss_terms: list[torch.Tensor] = []
            if _uses_pixel_world_loss(config):
                if output.action_frame_logits is None:
                    raise RuntimeError("pixel world loss requires frame logits")
                prediction_loss = next_frame_loss(
                    output.action_frame_logits,
                    pending.next_frame,
                    palette_size=config.palette_size,
                    frame_size=config.frame_size,
                    action_index=pending.selected_action_index,
                    current_frame=pending.current_frame,
                    residual_prediction=_residual_frame_prediction(config),
                    residual_logit_bias=_residual_frame_logit_bias(config),
                )
                loss_terms.append(
                    float(config.update.next_frame_loss_weight) * prediction_loss
                )
            if _uses_latent_world_loss(config):
                if predicted_delta is None:
                    raise RuntimeError("latent world loss requires latent prediction")
                latent_loss = latent_grid_loss(
                    predicted_delta,
                    pending.current_latent_grid,
                    pending.target_latent_grid,
                    pending.latent_changed_patch_mask,
                    changed_patch_weight=_latent_changed_patch_weight(config),
                    huber_beta=_latent_huber_beta(config),
                    cosine_loss_weight=_latent_cosine_loss_weight(config),
                    cosine_min_delta_norm=_latent_cosine_min_delta_norm(config),
                )
                loss_terms.append(float(config.update.latent_loss_weight) * latent_loss)
            coord_loss = coord_self_imitation_loss(
                output.coord_logits,
                pending.selected_x,
                pending.selected_y,
            )
            loss = sum(loss_terms) + float(config.update.coord_loss_weight) * coord_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                world_model_parameters(model),
                float(config.update.gradient_clip_norm),
            )
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    return last_loss


def _run_policy_updates(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    pending: PendingTurn,
    advantage: float,
    config: Any,
) -> float:
    if float(config.update.policy_loss_weight) == 0.0:
        return 0.0
    last_loss = 0.0
    old_log_probability = _old_policy_log_probability(pending)
    clip_epsilon = _policy_clip_epsilon(config)
    model.train()
    with _use_policy_adapter(model), _temporarily_requires_grad(
        world_model_parameters(model),
        False,
    ):
        for _ in range(int(config.update.update_steps)):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                pending.prompt,
                pending.images,
                include_frame_logits=False,
            )
            masked_logits = mask_action_logits(
                output.action_logits[0],
                pending.action_space,
            )
            log_probs = _policy_log_probs(masked_logits, config)
            policy_loss = _ppo_clipped_policy_loss(
                new_log_probability=log_probs[pending.selected_action_index],
                old_log_probability=old_log_probability,
                advantage=float(advantage),
                clip_epsilon=clip_epsilon,
            )
            loss = float(config.update.policy_loss_weight) * policy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_parameters(model),
                float(config.update.gradient_clip_norm),
            )
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    return last_loss


def _run_policy_batch_updates(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    finalized_turns: list[FinalizedTurn],
    config: Any,
) -> float:
    if float(config.update.policy_loss_weight) == 0.0 or not finalized_turns:
        return 0.0
    last_loss = 0.0
    model.train()
    with _use_policy_adapter(model), _temporarily_requires_grad(
        world_model_parameters(model),
        False,
    ):
        for _ in range(int(config.update.update_steps)):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = None
            for finalized in finalized_turns:
                pending = finalized.pending
                output = model(
                    pending.prompt,
                    pending.images,
                    include_frame_logits=False,
                )
                masked_logits = mask_action_logits(
                    output.action_logits[0],
                    pending.action_space,
                )
                log_probs = _policy_log_probs(masked_logits, config)
                item_loss = _ppo_clipped_policy_loss(
                    new_log_probability=log_probs[pending.selected_action_index],
                    old_log_probability=_old_policy_log_probability(pending),
                    advantage=float(finalized.policy_advantage),
                    clip_epsilon=_policy_clip_epsilon(config),
                )
                batch_loss = item_loss if batch_loss is None else batch_loss + item_loss
            if batch_loss is None:
                continue
            batch_loss = batch_loss / max(len(finalized_turns), 1)
            loss = float(config.update.policy_loss_weight) * batch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_parameters(model),
                float(config.update.gradient_clip_norm),
            )
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    return last_loss


def _ppo_clipped_policy_loss(
    *,
    new_log_probability: torch.Tensor,
    old_log_probability: float,
    advantage: float,
    clip_epsilon: float,
) -> torch.Tensor:
    old_log_probability_tensor = new_log_probability.new_tensor(
        float(old_log_probability)
    )
    advantage_tensor = new_log_probability.new_tensor(float(advantage))
    ratio = torch.exp(new_log_probability - old_log_probability_tensor)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - float(clip_epsilon),
        1.0 + float(clip_epsilon),
    )
    return -torch.minimum(
        ratio * advantage_tensor,
        clipped_ratio * advantage_tensor,
    )


def _old_policy_log_probability(pending: PendingTurn) -> float:
    return float(
        pending.model_output.get(
            "selected_action_log_probability",
            pending.transition.log_probability,
        )
    )


def _policy_log_probs(masked_logits: torch.Tensor, config: Any) -> torch.Tensor:
    temperature = _policy_temperature(config)
    return torch.log_softmax(masked_logits / temperature, dim=-1)


def _policy_temperature(config: Any) -> float:
    if getattr(config.model, "action_selection", "sample") == "sample":
        return max(float(getattr(config.model, "temperature", 1.0)), 1e-6)
    return 1.0


def _policy_clip_epsilon(config: Any) -> float:
    return max(float(getattr(config.update, "policy_clip_epsilon", 0.2)), 0.0)


def _policy_probability_ratio(
    *,
    new_log_probability: float,
    old_log_probability: float,
) -> float:
    log_ratio = max(
        min(float(new_log_probability) - float(old_log_probability), 20.0),
        -20.0,
    )
    return float(math.exp(log_ratio))


def _clipped_probability_ratio(ratio: float, clip_epsilon: float) -> float:
    lower = 1.0 - float(clip_epsilon)
    upper = 1.0 + float(clip_epsilon)
    return float(min(max(float(ratio), lower), upper))


def _policy_update_skipped_reason(
    *,
    pending: PendingTurn,
    intrinsic_reward_baseline: float | None,
    requires_intrinsic_baseline: bool = True,
    config: Any,
) -> str | None:
    warmup_turns = max(int(getattr(config.update, "policy_warmup_turns", 3)), 0)
    if pending.transition.turn < warmup_turns:
        return "warmup"
    if float(config.update.policy_loss_weight) == 0.0:
        return "disabled"
    if int(config.update.update_steps) <= 0:
        return "no_update_steps"
    if requires_intrinsic_baseline and intrinsic_reward_baseline is None:
        return "baseline_init"
    return None


def _next_intrinsic_reward_baseline(
    *,
    intrinsic_reward_baseline: float | None,
    policy_intrinsic_reward: float,
    policy_update_skipped_reason: str | None,
    baseline_beta: float,
) -> float | None:
    if policy_update_skipped_reason == "warmup":
        return intrinsic_reward_baseline
    if intrinsic_reward_baseline is None:
        return float(policy_intrinsic_reward)
    return float(
        baseline_beta * intrinsic_reward_baseline
        + (1.0 - baseline_beta) * policy_intrinsic_reward
    )


def _policy_advantage_baseline_mode(config: Any) -> str:
    mode = str(getattr(config.update, "policy_advantage_baseline", "ema"))
    if mode not in {"ema", "zero"}:
        raise ValueError(f"unknown policy_advantage_baseline: {mode!r}")
    return mode


def _policy_advantage_normalization(config: Any) -> str:
    mode = str(getattr(config.update, "policy_advantage_normalization", "none"))
    if mode not in {"none", "ema_abs"}:
        raise ValueError(f"unknown policy_advantage_normalization: {mode!r}")
    return mode


def _policy_advantage_normalization_beta(config: Any) -> float:
    return max(
        0.0,
        min(
            0.999,
            float(getattr(config.update, "policy_advantage_normalization_beta", 0.5)),
        ),
    )


def _policy_advantage_normalization_floor(config: Any) -> float:
    return max(
        float(getattr(config.update, "policy_advantage_normalization_floor", 0.01)),
        1e-8,
    )


def _raw_policy_advantage(
    *,
    policy_intrinsic_reward: float,
    intrinsic_reward_baseline: float | None,
    baseline_mode: str,
) -> float:
    if baseline_mode == "zero":
        return float(policy_intrinsic_reward)
    if intrinsic_reward_baseline is None:
        return 0.0
    return float(policy_intrinsic_reward) - float(intrinsic_reward_baseline)


def _normalized_policy_advantage(
    *,
    raw_policy_advantage: float,
    denominator: float,
    normalization: str,
) -> float:
    if normalization == "none":
        return float(raw_policy_advantage)
    return float(raw_policy_advantage) / max(float(denominator), 1e-8)


def _policy_advantage_normalization_denominator(
    *,
    policy_signal_abs_baseline: float | None,
    policy_intrinsic_reward: float,
    config: Any,
) -> float:
    if _policy_advantage_normalization(config) == "none":
        return 1.0
    current_abs = abs(float(policy_intrinsic_reward))
    fallback = current_abs
    if policy_signal_abs_baseline is not None:
        fallback = max(float(policy_signal_abs_baseline), current_abs)
    return max(fallback, _policy_advantage_normalization_floor(config))


def _next_policy_signal_abs_baseline(
    *,
    policy_signal_abs_baseline: float | None,
    policy_intrinsic_reward: float,
    config: Any,
) -> float | None:
    if _policy_advantage_normalization(config) == "none":
        return policy_signal_abs_baseline
    signal_abs = abs(float(policy_intrinsic_reward))
    if policy_signal_abs_baseline is None:
        return signal_abs
    beta = _policy_advantage_normalization_beta(config)
    return float(beta * policy_signal_abs_baseline + (1.0 - beta) * signal_abs)


def _window_learning_progress(
    model: Any,
    *,
    snapshot: list[torch.Tensor],
    lp_window: Sequence[PendingTurn],
    discount: float,
    config: Any,
) -> dict[str, Any]:
    """Compute snapshot-vs-current LP over one discounted transition window."""

    weights = _discounted_window_weights(len(lp_window), discount)
    if not lp_window:
        return {
            "learning_progress": 0.0,
            "weighted_prior_loss": 0.0,
            "weighted_current_loss": 0.0,
            "weights": [],
            "prior_losses": [],
            "current_losses": [],
            "transition_progress": [],
        }

    with _temporarily_load_world_snapshot(model, snapshot):
        prior_losses = [
            _pending_prediction_loss_value(model, pending, config=config)
            for pending in lp_window
        ]
    current_losses = [
        _pending_prediction_loss_value(model, pending, config=config)
        for pending in lp_window
    ]
    prior_values = [float(loss) for loss in prior_losses if loss is not None]
    current_values = [float(loss) for loss in current_losses if loss is not None]
    if len(prior_values) != len(lp_window) or len(current_values) != len(lp_window):
        raise RuntimeError("window LP received an empty transition loss")
    transition_progress = [
        prior_loss - current_loss
        for prior_loss, current_loss in zip(prior_values, current_values)
    ]
    return {
        "learning_progress": _weighted_sum(transition_progress, weights),
        "weighted_prior_loss": _weighted_sum(prior_values, weights),
        "weighted_current_loss": _weighted_sum(current_values, weights),
        "weights": weights,
        "prior_losses": prior_values,
        "current_losses": current_values,
        "transition_progress": transition_progress,
    }


def _learning_progress_loss_scale(
    window_metrics: dict[str, Any],
    *,
    config: Any,
) -> float:
    if not (
        _uses_latent_world_loss(config)
        and _latent_learning_progress_normalization(config)
    ):
        return 1.0
    floor = _latent_learning_progress_normalization_floor(config)
    return max(
        abs(float(window_metrics["weighted_prior_loss"])),
        abs(float(window_metrics["weighted_current_loss"])),
        floor,
    )


def _weighted_learning_progress(
    prior_losses: Sequence[float],
    current_losses: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Return weighted prior-current improvement for tests and diagnostics."""

    progress = [
        float(prior) - float(current)
        for prior, current in zip(prior_losses, current_losses)
    ]
    return _weighted_sum(progress, weights)


def _weighted_sum(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    return float(
        sum(float(value) * float(weight) for value, weight in zip(values, weights))
    )


def _discounted_window_weights(size: int, discount: float) -> list[float]:
    """Return normalized recency weights `[1, gamma, gamma^2, ...]`."""

    if size <= 0:
        return []
    gamma = max(0.0, min(1.0, float(discount)))
    weights = [gamma**index for index in range(size)]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / size for _ in range(size)]
    return [float(weight / total) for weight in weights]


def _learning_progress_discount(config: Any) -> float:
    return max(
        0.0,
        min(1.0, float(getattr(config.update, "learning_progress_discount", 0.8))),
    )


def _learning_progress_rate_beta(config: Any) -> float:
    return max(
        0.0,
        min(0.999, float(getattr(config.update, "learning_progress_rate_beta", 0.5))),
    )


def _action_conditioned_learning_progress_baseline(config: Any) -> bool:
    return bool(
        getattr(
            config.update,
            "action_conditioned_learning_progress_baseline",
            False,
        )
    )


def _selected_learning_progress_rate_baseline(
    *,
    selected_action_name: str,
    learning_progress_rate_baseline: float | None,
    action_learning_progress_rate_baselines: dict[str, float],
    action_conditioned: bool,
) -> float | None:
    if not action_conditioned:
        return learning_progress_rate_baseline
    return action_learning_progress_rate_baselines.get(selected_action_name)


def _learning_progress_rate_reference(
    *,
    learning_progress_rate_baseline: float | None,
    raw_learning_progress: float,
) -> float:
    if learning_progress_rate_baseline is None:
        return float(raw_learning_progress)
    return float(learning_progress_rate_baseline)


def _next_learning_progress_rate_baseline(
    *,
    learning_progress_rate_baseline: float | None,
    raw_learning_progress: float,
    beta: float,
) -> float:
    if learning_progress_rate_baseline is None:
        return float(raw_learning_progress)
    return float(
        float(beta) * float(learning_progress_rate_baseline)
        + (1.0 - float(beta)) * float(raw_learning_progress)
    )


def _residual_frame_prediction(config: Any) -> bool:
    return bool(getattr(config.update, "residual_frame_prediction", False))


def _residual_frame_logit_bias(config: Any) -> float:
    return float(getattr(config.update, "residual_frame_logit_bias", 4.0))


def _world_loss_mode(config: Any) -> str:
    mode = str(getattr(config.update, "world_loss_mode", "pixel_ce"))
    if mode not in {"pixel_ce", "latent_grid", "hybrid"}:
        raise ValueError(f"unsupported world_loss_mode: {mode!r}")
    return mode


def _uses_pixel_world_loss(config: Any) -> bool:
    return _world_loss_mode(config) in {"pixel_ce", "hybrid"}


def _uses_latent_world_loss(config: Any) -> bool:
    return _world_loss_mode(config) in {"latent_grid", "hybrid"}


def _latent_changed_patch_weight(config: Any) -> float:
    return float(getattr(config.update, "latent_changed_patch_weight", 6.0))


def _latent_huber_beta(config: Any) -> float:
    return float(getattr(config.update, "latent_huber_beta", 1.0))


def _latent_cosine_loss_weight(config: Any) -> float:
    return float(getattr(config.update, "latent_cosine_loss_weight", 0.0))


def _latent_cosine_min_delta_norm(config: Any) -> float:
    return float(getattr(config.update, "latent_cosine_min_delta_norm", 1e-4))


def _latent_learning_progress_normalization(config: Any) -> bool:
    return bool(
        getattr(config.update, "latent_learning_progress_normalization", True)
    )


def _latent_learning_progress_normalization_floor(config: Any) -> float:
    return max(
        float(
            getattr(
                config.update,
                "latent_learning_progress_normalization_floor",
                0.01,
            )
        ),
        1e-8,
    )


def _policy_update_accumulation_steps(config: Any) -> int:
    return max(int(getattr(config.update, "policy_update_accumulation_steps", 1)), 1)


def _policy_return_horizon(config: Any) -> int:
    return max(
        int(getattr(config.update, "policy_learning_progress_return_horizon", 12)),
        1,
    )


def _policy_return_discount(config: Any) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(
                getattr(
                    config.update,
                    "policy_learning_progress_return_discount",
                    0.93,
                )
            ),
        ),
    )


def _capture_world_snapshot(model: Any) -> list[torch.Tensor]:
    """Capture trainable world parameters on CPU, excluding policy-only params."""

    return [
        parameter.detach().cpu().clone()
        for parameter in world_model_parameters(model)
    ]


@contextmanager
def _temporarily_load_world_snapshot(
    model: Any,
    snapshot: Sequence[torch.Tensor],
) -> Any:
    """Swap in a world snapshot, then restore current world parameters exactly."""

    parameters = world_model_parameters(model)
    if len(parameters) != len(snapshot):
        raise ValueError(
            "world snapshot parameter count does not match current model: "
            f"snapshot={len(snapshot)}, current={len(parameters)}"
        )
    current_values = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter, value in zip(parameters, snapshot):
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, current_values):
                parameter.copy_(value)


def _pending_prediction_loss_value(
    model: Any,
    pending: PendingTurn | None,
    *,
    config: Any,
) -> float | None:
    if pending is None:
        return None
    details = _pending_prediction_loss_details(model, pending, config=config)
    return float(details["loss"])


def _pending_prediction_loss_details(
    model: Any,
    pending: PendingTurn,
    *,
    config: Any,
) -> dict[str, Any]:
    _ensure_pending_latents(model=model, pending=pending, config=config)
    with _use_world_adapter(model), torch.no_grad():
        predicted_delta = None
        if _uses_latent_world_loss(config):
            output, predicted_delta = model.forward_with_latent_delta(
                pending.prompt,
                pending.images,
                action_index=pending.selected_action_index,
                selected_x=pending.selected_x,
                selected_y=pending.selected_y,
                include_frame_logits=_uses_pixel_world_loss(config),
            )
        else:
            output = model(
                pending.prompt,
                pending.images,
                include_frame_logits=_uses_pixel_world_loss(config),
            )
        total_loss = torch.zeros((), device=output.action_logits.device)
        details: dict[str, Any] = {
            "world_loss_mode": _world_loss_mode(config),
            "pixel_loss": None,
            "latent_loss": None,
            "loss": 0.0,
        }
        if _uses_pixel_world_loss(config):
            if output.action_frame_logits is None:
                raise RuntimeError("pixel world loss requires frame logits")
            pixel_loss = next_frame_loss(
                output.action_frame_logits,
                pending.next_frame,
                palette_size=config.palette_size,
                frame_size=config.frame_size,
                action_index=pending.selected_action_index,
                current_frame=pending.current_frame,
                residual_prediction=_residual_frame_prediction(config),
                residual_logit_bias=_residual_frame_logit_bias(config),
            )
            total_loss = total_loss + (
                float(config.update.next_frame_loss_weight) * pixel_loss
            )
            details["pixel_loss"] = float(pixel_loss.detach().cpu().item())
        if _uses_latent_world_loss(config):
            if predicted_delta is None:
                raise RuntimeError("latent world loss requires latent prediction")
            latent_loss, latent_details = latent_grid_loss(
                predicted_delta,
                pending.current_latent_grid,
                pending.target_latent_grid,
                pending.latent_changed_patch_mask,
                changed_patch_weight=_latent_changed_patch_weight(config),
                huber_beta=_latent_huber_beta(config),
                cosine_loss_weight=_latent_cosine_loss_weight(config),
                cosine_min_delta_norm=_latent_cosine_min_delta_norm(config),
                return_details=True,
            )
            total_loss = total_loss + (
                float(config.update.latent_loss_weight) * latent_loss
            )
            details["latent_loss"] = float(latent_loss.detach().cpu().item())
            details["latent"] = _scalar_latent_details(latent_details)
        details["loss"] = float(total_loss.detach().cpu().item())
        return details


def _prediction_loss_value(
    model: Any,
    prompt: str,
    images: list[Any],
    current_frame: Any,
    target_frame: Any,
    *,
    action_index: int,
    config: Any,
) -> float:
    with _use_world_adapter(model), torch.no_grad():
        output = model(
            prompt,
            images,
            include_frame_logits=True,
        )
        if output.action_frame_logits is None:
            raise RuntimeError("pixel prediction loss requires frame logits")
        loss = next_frame_loss(
            output.action_frame_logits,
            target_frame,
            palette_size=config.palette_size,
            frame_size=config.frame_size,
            action_index=action_index,
            current_frame=current_frame,
            residual_prediction=_residual_frame_prediction(config),
            residual_logit_bias=_residual_frame_logit_bias(config),
        )
    return float(loss.detach().cpu().item())


def _ensure_pending_latents(
    *,
    model: Any,
    pending: PendingTurn,
    config: Any,
) -> None:
    if not _uses_latent_world_loss(config):
        return
    if pending.current_latent_grid is not None:
        return
    current_grid = model.frame_latent_grid(pending.current_frame)
    target_grid = model.frame_latent_grid(pending.next_frame)
    if tuple(current_grid.shape) != tuple(target_grid.shape):
        raise RuntimeError(
            "current and target latent grids have different shapes: "
            f"current={tuple(current_grid.shape)}, target={tuple(target_grid.shape)}"
        )
    mask = latent_changed_patch_mask(
        pending.current_frame,
        pending.next_frame,
        palette_size=config.palette_size,
        frame_size=config.frame_size,
        grid_shape=tuple(current_grid.shape[:2]),
    )
    pending.current_latent_grid = current_grid.detach().cpu()
    pending.target_latent_grid = target_grid.detach().cpu()
    pending.latent_changed_patch_mask = mask.detach().cpu()


def _scalar_latent_details(details: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in details.items()
        if key
        not in {
            "target_delta_norm_map",
            "prediction_delta_norm_map",
            "error_norm_map",
            "changed_patch_mask",
        }
    }


def _latent_detail_value(loss_details: dict[str, Any], key: str) -> Any:
    latent = loss_details.get("latent")
    if not isinstance(latent, dict):
        return None
    return latent.get(key)


def _maybe_log_frame_prediction(
    *,
    model: Any,
    frame_prediction_logger: Any,
    pending: PendingTurn,
    config: Any,
) -> dict[str, Any] | None:
    turn = int(pending.transition.turn)
    if not frame_prediction_logger.should_log(turn):
        return None
    try:
        with _temporarily_load_world_snapshot(model, pending.world_snapshot):
            pre_update_prediction = _predicted_frame_value(
                model=model,
                pending=pending,
                config=config,
            )
            pre_update_loss = _prediction_loss_value(
                model,
                pending.prompt,
                pending.images,
                pending.current_frame,
                pending.next_frame,
                action_index=pending.selected_action_index,
                config=config,
            )
        post_update_prediction = _predicted_frame_value(
            model=model,
            pending=pending,
            config=config,
        )
        post_update_loss = _prediction_loss_value(
            model,
            pending.prompt,
            pending.images,
            pending.current_frame,
            pending.next_frame,
            action_index=pending.selected_action_index,
            config=config,
        )
        return frame_prediction_logger.append_prediction(
            turn=turn,
            action_name=str(pending.model_output["selected_action_name"]),
            selected_action_index=int(pending.selected_action_index),
            observation_id=getattr(pending.transition.observation, "id", None),
            next_observation_id=getattr(pending.transition.next_observation, "id", None),
            current_frame=pending.current_frame,
            target_frame=pending.next_frame,
            pre_update_prediction=pre_update_prediction,
            post_update_prediction=post_update_prediction,
            pre_update_loss=pre_update_loss,
            post_update_loss=post_update_loss,
        )
    except Exception as exc:
        return {
            "turn": turn,
            "action_name": str(pending.model_output.get("selected_action_name")),
            "error": str(exc),
        }


def _maybe_log_latent_prediction(
    *,
    model: Any,
    latent_prediction_logger: Any,
    pending: PendingTurn,
    config: Any,
) -> dict[str, Any] | None:
    turn = int(pending.transition.turn)
    if not latent_prediction_logger.should_log(turn):
        return None
    try:
        _ensure_pending_latents(model=model, pending=pending, config=config)
        with _temporarily_load_world_snapshot(model, pending.world_snapshot):
            pre_update_prediction = _predicted_latent_delta_value(
                model=model,
                pending=pending,
            )
        post_update_prediction = _predicted_latent_delta_value(
            model=model,
            pending=pending,
        )
        return latent_prediction_logger.append_prediction(
            turn=turn,
            action_name=str(pending.model_output["selected_action_name"]),
            selected_action_index=int(pending.selected_action_index),
            observation_id=getattr(pending.transition.observation, "id", None),
            next_observation_id=getattr(pending.transition.next_observation, "id", None),
            current_latent_grid=pending.current_latent_grid,
            target_latent_grid=pending.target_latent_grid,
            changed_patch_mask=pending.latent_changed_patch_mask,
            pre_update_prediction=pre_update_prediction,
            post_update_prediction=post_update_prediction,
            pre_update_loss=pending.one_step_prior_loss_details.get("latent_loss"),
            post_update_loss=pending.one_step_current_loss_details.get("latent_loss"),
        )
    except Exception as exc:
        return {
            "turn": turn,
            "action_name": str(pending.model_output.get("selected_action_name")),
            "error": str(exc),
        }


def _predicted_frame_value(
    *,
    model: Any,
    pending: PendingTurn,
    config: Any,
) -> torch.Tensor:
    with _use_world_adapter(model), torch.no_grad():
        output = model(pending.prompt, pending.images, include_frame_logits=True)
        if output.action_frame_logits is None:
            raise RuntimeError("frame prediction logging requires frame logits")
        return predicted_frame_tensor(
            output.action_frame_logits,
            palette_size=config.palette_size,
            frame_size=config.frame_size,
            action_index=pending.selected_action_index,
            current_frame=pending.current_frame,
            residual_prediction=_residual_frame_prediction(config),
            residual_logit_bias=_residual_frame_logit_bias(config),
        )


def _predicted_latent_delta_value(
    *,
    model: Any,
    pending: PendingTurn,
) -> torch.Tensor:
    with _use_world_adapter(model), torch.no_grad():
        return (
            model.predict_latent_delta(
                pending.prompt,
                pending.images,
                action_index=pending.selected_action_index,
                selected_x=pending.selected_x,
                selected_y=pending.selected_y,
            )
            .squeeze(0)
            .detach()
            .cpu()
        )


def _log_probability_value(*, model: Any, pending: PendingTurn, config: Any) -> float:
    with _use_policy_adapter(model), torch.no_grad():
        output = model(
            pending.prompt,
            pending.images,
            include_frame_logits=False,
        )
        masked_logits = mask_action_logits(
            output.action_logits[0],
            pending.action_space,
        )
        log_probs = _policy_log_probs(masked_logits, config)
    return float(log_probs[pending.selected_action_index].detach().cpu().item())


def _policy_probe(*, model: Any, pending: PendingTurn, config: Any) -> dict[str, Any]:
    """Return compact policy diagnostics for one stored transition."""

    probability_temperature = (
        config.model.temperature
        if config.model.action_selection == "sample"
        else 1.0
    )
    with _use_policy_adapter(model), torch.no_grad():
        output = model(
            pending.prompt,
            pending.images,
            include_frame_logits=False,
        )
        logits = (
            output.action_logits[0]
            if output.action_logits.ndim == 2
            else output.action_logits
        )
        logits = logits.detach()
        valid_mask = torch.tensor(
            valid_action_mask(pending.action_space),
            dtype=torch.bool,
            device=logits.device,
        )
        valid_logits = logits[valid_mask].float()
        probabilities = masked_action_probabilities(
            output.action_logits,
            pending.action_space,
            temperature=probability_temperature,
        )
    valid_probabilities = [
        probabilities[name]
        for name in ACTION_NAMES
        if probabilities[name] > 0.0
    ]
    selected_name = pending.model_output["selected_action_name"]
    selected_probability = float(probabilities.get(selected_name, 0.0))
    return {
        "masked_action_probabilities": probabilities,
        "selected_action_index": pending.selected_action_index,
        "selected_action_name": selected_name,
        "selected_action_probability": selected_probability,
        "selected_action_log_probability": math.log(max(selected_probability, 1e-12)),
        "max_action_probability": max(valid_probabilities) if valid_probabilities else 0.0,
        "action_entropy": _probability_entropy(valid_probabilities),
        "action_logit_l2": _tensor_l2(logits),
        "action_logit_max_abs": _tensor_max_abs(logits),
        "valid_action_logit_l2": _tensor_l2(valid_logits),
        "valid_action_logit_max_abs": _tensor_max_abs(valid_logits),
        "valid_action_logit_span": _valid_logit_span(valid_logits),
    }


def _probability_entropy(probabilities: list[float]) -> float:
    return float(
        -sum(probability * math.log(max(probability, 1e-12)) for probability in probabilities)
    )


def _action_head_norms(model: Any) -> dict[str, float]:
    """Return compact norms for the trainable action policy head."""

    action_head = getattr(model, "action_head", None)
    if action_head is None:
        return {
            "action_head_weight_l2": 0.0,
            "action_head_weight_max_abs": 0.0,
            "action_head_bias_l2": 0.0,
            "action_head_bias_max_abs": 0.0,
        }
    return {
        "action_head_weight_l2": _tensor_l2(action_head.weight.detach()),
        "action_head_weight_max_abs": _tensor_max_abs(action_head.weight.detach()),
        "action_head_bias_l2": _tensor_l2(action_head.bias.detach()),
        "action_head_bias_max_abs": _tensor_max_abs(action_head.bias.detach()),
    }


def _tensor_l2(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.float()).detach().cpu().item())


def _tensor_max_abs(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.float().abs().max().detach().cpu().item())


def _valid_logit_span(valid_logits: torch.Tensor) -> float:
    if valid_logits.numel() == 0:
        return 0.0
    logits = valid_logits.float()
    return float((logits.max() - logits.min()).detach().cpu().item())


def _coord_argmax(coord_logits: torch.Tensor) -> dict[str, int]:
    coord = coord_logits[0] if coord_logits.ndim == 2 else coord_logits
    return {
        "x": int(coord[:64].argmax().detach().cpu().item()),
        "y": int(coord[64:].argmax().detach().cpu().item()),
    }


def _seed_runtime(seed: int) -> None:
    """Seed local sampling and trainable head initialization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _effective_update_steps(config: Any, *, include_policy_update: bool) -> int:
    update_steps = max(int(config.update.update_steps), 0)
    if not include_policy_update:
        return update_steps
    return update_steps * 2


def _observation_has_usable_frame(observation: Any) -> bool:
    frame = decision_frame(observation)
    if frame is None:
        return False
    try:
        array = np.asarray(frame)
    except Exception:
        return False
    return array.ndim in (2, 3)


def _frame_change_diagnostics(
    current_frame: Any,
    next_frame: Any,
    *,
    config: Any,
) -> dict[str, Any]:
    current_target = frame_to_palette_tensor(
        current_frame,
        palette_size=config.palette_size,
        frame_size=config.frame_size,
    )
    next_target = frame_to_palette_tensor(
        next_frame,
        palette_size=config.palette_size,
        frame_size=config.frame_size,
    )
    changed_pixels = int((current_target != next_target).sum().item())
    total_pixels = max(int(current_target.numel()), 1)
    return {
        "frame_changed_pixels": changed_pixels,
        "frame_changed_fraction": float(changed_pixels / total_pixels),
    }


def _parameter_count(parameters: list[Any]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def _append_turn(*, logger: ExperimentLogger, completed_turn: FinalizedTurn) -> None:
    logger.append_turn(completed_turn.payload)


def _adapter_metadata(model: Any) -> dict[str, Any]:
    return {
        "role_adapters_enabled": bool(getattr(model, "role_adapters_enabled", False)),
        "world_adapter": getattr(model, "world_adapter_name", None),
        "policy_adapter": getattr(model, "policy_adapter_name", None),
        "policy_adapter_trainable": bool(
            getattr(model, "policy_adapter_trainable", True)
        ),
        "action_runtime_adapter": getattr(model, "action_runtime_adapter_name", None),
    }


def _use_world_adapter(model: Any) -> Any:
    context = getattr(model, "use_world_adapter", None)
    if callable(context):
        return context()
    return nullcontext()


def _use_policy_adapter(model: Any) -> Any:
    context = getattr(model, "use_policy_adapter", None)
    if callable(context):
        return context()
    return nullcontext()


@contextmanager
def _temporarily_requires_grad(parameters: list[Any], requires_grad: bool) -> Any:
    original_values = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(requires_grad)
        yield
    finally:
        for parameter, original_value in zip(parameters, original_values):
            parameter.requires_grad_(original_value)


def _maybe_save_checkpoint(
    *,
    model: Any,
    logger: ExperimentLogger,
    config: Any,
    completed_turn: int,
) -> None:
    save_every = int(config.model.lora.save_every)
    if save_every > 0 and completed_turn % save_every == 0:
        model.save_step_checkpoint(logger.step_checkpoint_path(completed_turn))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated single-VLM online LoRA ARC experiment.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a single-VLM experiment YAML config.",
    )
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--game-index", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--video-frame-scale", type=int, default=None)
    parser.add_argument(
        "--world-loss-mode",
        choices=("pixel_ce", "latent_grid", "hybrid"),
        default=None,
    )
    parser.add_argument("--latent-loss-weight", type=float, default=None)
    parser.add_argument("--latent-changed-patch-weight", type=float, default=None)
    parser.add_argument("--latent-huber-beta", type=float, default=None)
    parser.add_argument("--latent-cosine-loss-weight", type=float, default=None)
    parser.add_argument("--latent-cosine-min-delta-norm", type=float, default=None)
    parser.add_argument(
        "--latent-learning-progress-normalization",
        dest="latent_learning_progress_normalization",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-latent-learning-progress-normalization",
        dest="latent_learning_progress_normalization",
        action="store_false",
    )
    parser.add_argument(
        "--latent-learning-progress-normalization-floor",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--save-frame-predictions",
        dest="save_frame_predictions",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-save-frame-predictions",
        dest="save_frame_predictions",
        action="store_false",
    )
    parser.add_argument("--frame-prediction-save-every", type=int, default=None)
    parser.add_argument("--frame-prediction-frame-scale", type=int, default=None)
    parser.add_argument(
        "--save-latent-predictions",
        dest="save_latent_predictions",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-save-latent-predictions",
        dest="save_latent_predictions",
        action="store_false",
    )
    parser.add_argument("--latent-prediction-save-every", type=int, default=None)
    parser.add_argument("--latent-prediction-frame-scale", type=int, default=None)
    return parser


if __name__ == "__main__":
    main()
