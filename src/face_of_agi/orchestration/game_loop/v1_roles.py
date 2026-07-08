"""V1 Memory/World/Goal/Reward helpers for the game loop."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from typing import Any, Sequence

from face_of_agi.contracts import (
    ActionSpec,
    AgentCandidateAction,
    CandidateValuePrediction,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    RewardJudgeScore,
    TurnLedgerEntry,
    TurnReward,
    WorldPrediction,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import ModelCallCompleted
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.frames import to_memory_jsonable
from face_of_agi.models.arc_grid_crop import arc_grid_to_normalized_1000
from face_of_agi.memory import StateMemory
from face_of_agi.models.action_history import model_facing_action_text
from face_of_agi.models.adapters import (
    GoalModel,
    InterestModel,
    MemoryModel,
    OrchestratorAgentModel,
    RewardJudgeModel,
    WorldModel,
)
from face_of_agi.models.goal import GoalPredictionInput
from face_of_agi.models.interest import InterestPredictionInput
from face_of_agi.models.memory import MemoryBuildInput, MemoryLedgerEntry
from face_of_agi.models.reward_judge import RewardJudgeInput
from face_of_agi.models.world import WorldPredictionInput
from face_of_agi.orchestration.game_loop import helpers
from face_of_agi.orchestration.game_loop.actions.metrics import (
    effective_trace_cost_seconds,
)
from face_of_agi.orchestration.game_loop.actions.steps import (
    require_current,
    require_decision,
    require_next,
    require_update_input,
)
from face_of_agi.orchestration.game_loop.persistence import write_frame_trace
from face_of_agi.orchestration.game_loop.session import GameLoopSession
from face_of_agi.runtime import timing as runtime_timing


def bootstrap_memory_goal(
    session: GameLoopSession,
    *,
    memory_model: MemoryModel,
    goal_model: GoalModel,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Create initial Memory and Goal after environment reset."""

    observation = session.latest_environment_observation
    if session.first_observation is None:
        session.first_observation = observation
        session.first_observation_ref = session.current_ref_for(observation)
    _regenerate_memory_goal(
        session,
        memory_model=memory_model,
        goal_model=goal_model,
        current_observation=observation,
        state_memory=state_memory,
        turn_id=0,
        debug=debug,
    )


def reset_memory_goal_after_game_over(
    session: GameLoopSession,
    *,
    memory_model: MemoryModel,
    goal_model: GoalModel,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Regenerate v1 Memory/Goal after ARC resets without erasing run knowledge."""

    observation = session.latest_environment_observation
    if session.first_observation is None:
        session.first_observation = observation
        session.first_observation_ref = session.current_ref_for(observation)
    session.turn_ledger.append(
        TurnLedgerEntry(
            turn_id=session.frame_turn_count,
            action=ActionSpec.none(),
            change_summary=(
                "GAME_RESET: the ARC environment reset after game over. "
                "Prior mechanics and failed attempts remain relevant, but the "
                "current frame is a fresh post-reset state."
            ),
            metadata={
                "controllable": False,
                "reset_marker": True,
                "restart_count": session.game_restart_count,
                "game_start_reason": session.game_start_reason,
            },
        )
    )
    session.memory = None
    session.goal = None
    _regenerate_memory_goal(
        session,
        memory_model=memory_model,
        goal_model=goal_model,
        current_observation=observation,
        state_memory=state_memory,
        turn_id=session.frame_turn_count,
        debug=debug,
    )


def decide_with_world_candidates(
    session: GameLoopSession,
    *,
    agent: OrchestratorAgentModel,
    world_model: WorldModel,
    interest_model: InterestModel,
    debug: DebugBus,
) -> None:
    """Run two-stage Agent selection with World predictions."""

    current = require_current(session)
    frame_context = current.to_frame_context()
    if not frame_context.control_mode.controllable:
        decision = helpers.synthetic_animation_decision(frame_context)
        session.candidate_actions = ()
        session.world_predictions = ()
        session.interest_prediction = None
        session.decision = decision
        session.decision_duration_seconds = 0.0
        session.trace_cost_seconds = 0.0
        session.last_decision = decision
        session.frame_turn_count = current.turn_id
        return

    memory = _require_memory(session)
    goal = _require_goal(session)
    prompt_actions = helpers.prompt_action_outcome(
        action_space=frame_context.control_mode.allowed_actions,
        action_history=frame_context.recent_action_history,
        action_suppression_zero_changed_pixel_turns=(
            session.environment_config.action_suppression_zero_changed_pixel_turns
        ),
        updater_stagnation_warning_zero_changed_pixel_turns=0,
        crop_edges=helpers.model_input_crop_edges(agent),
    )
    candidates = _candidate_actions(
        agent=agent,
        memory=memory,
        goal=goal,
        current_observation=frame_context.current_observation,
        action_space=prompt_actions.allowed_actions,
        max_candidates=session.environment_config.candidate_action_count,
        recent_action_history=frame_context.recent_action_history,
        glossary_actions=frame_context.control_mode.allowed_actions,
    )
    if not candidates:
        raise RuntimeError("v1 agent candidate set is empty")

    predictions = _world_predictions(
        session,
        world_model=world_model,
        memory=memory,
        candidates=candidates,
        glossary_actions=frame_context.control_mode.allowed_actions,
        debug=debug,
    )
    interest_prediction = _interest_prediction(
        session,
        interest_model=interest_model,
        memory=memory,
        goal=goal,
        candidates=candidates,
        world_predictions=predictions,
        recent_action_history=frame_context.recent_action_history,
        debug=debug,
    )

    with runtime_timing.span(
        "game_loop.agent_select_action_v1",
        turn_id=current.turn_id,
        step=frame_context.current_observation.step,
    ):
        decision_started_at = helpers.perf_counter()
        decision = agent.select_action(
            memory=memory,
            goal=goal,
            current_observation=frame_context.current_observation,
            candidates=candidates,
            world_predictions=predictions,
            interest_prediction=interest_prediction,
            first_observation_ref=frame_context.first_observation_ref,
            recent_action_history=frame_context.recent_action_history,
            glossary_actions=frame_context.control_mode.allowed_actions,
        )
        duration = helpers.perf_counter() - decision_started_at

    session.candidate_actions = candidates
    session.world_predictions = predictions
    session.interest_prediction = interest_prediction
    session.decision = decision
    session.decision_duration_seconds = duration
    session.trace_cost_seconds = effective_trace_cost_seconds(
        decision=decision,
        wall_clock_seconds=duration,
    )
    session.last_decision = decision
    session.frame_turn_count = current.turn_id
    helpers.validate_decision(decision.final_action, control_mode=frame_context.control_mode)
    write_frame_trace(
        debug=debug,
        frame_turn=session.frame_turn_count,
        frame_context=frame_context,
        action=decision.final_action,
        trace=decision.trace,
    )
    debug.emit(ModelCallCompleted(role="agent", duration_seconds=duration))
    debug.capture_model_inputs(frame_context, current.turn_id, agent)


def evaluate_observed_transition(
    session: GameLoopSession,
    *,
    reward_judge_model: RewardJudgeModel,
    memory_model: MemoryModel,
    goal_model: GoalModel,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Judge the executed World prediction, compute reward, and update M/G."""

    current = require_current(session)
    next_snapshot = require_next(session)
    decision = require_decision(session)
    update_input = require_update_input(session)
    if update_input.action_history_entry is None:
        raise RuntimeError("v1 reward requires a change-summary history entry")

    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")
    previous_goal = session.goal
    if not current.control_mode.controllable:
        ledger_entry = TurnLedgerEntry(
            turn_id=current.turn_id,
            action=decision.final_action,
            change_summary=update_input.action_history_entry.change_summary,
            goal_before=previous_goal,
            metadata={
                "controllable": False,
                "m_state_id": current.source_state_id,
            },
        )
        session.turn_ledger.append(ledger_entry)
        next_memory, next_goal = _regenerate_memory_goal(
            session,
            memory_model=memory_model,
            goal_model=goal_model,
            current_observation=next_snapshot.observation,
            state_memory=state_memory,
            turn_id=current.turn_id,
            debug=debug,
            persist_goal=False,
        )
        ledger_entry = replace(ledger_entry, goal_after=next_goal)
        session.turn_ledger[-1] = ledger_entry
        session.memory = next_memory
        session.goal = next_goal
        _persist_animation_turn(
            session,
            state_memory=state_memory,
            ledger_entry=ledger_entry,
            memory=next_memory,
            goal=next_goal,
        )
        debug.clear_model_token_usage(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=current.turn_id,
        )
        return

    executed_prediction = _executed_world_prediction(
        session.world_predictions,
        decision.final_action,
    )
    judge = _judge_executed_prediction(
        session,
        reward_judge_model=reward_judge_model,
        prediction=executed_prediction,
        change_summary=update_input.action_history_entry.change_summary,
        debug=debug,
    )
    provisional_entry = TurnLedgerEntry(
        turn_id=current.turn_id,
        action=decision.final_action,
        change_summary=update_input.action_history_entry.change_summary,
        candidate_predictions=session.world_predictions,
        judge_scores=(judge,),
        goal_before=previous_goal,
        metadata={
            "controllable": True,
            "m_state_id": current.source_state_id,
        },
    )
    reward_goal = _predict_reward_goal(
        session,
        goal_model=goal_model,
        current_observation=next_snapshot.observation,
        memory=_require_memory(session),
        previous_goal=previous_goal,
        turn_id=current.turn_id,
        debug=debug,
    )
    _attach_model_token_usage(
        session,
        debug=debug,
        turn_id=current.turn_id,
    )
    reward = compute_immediate_turn_reward(
        environment_config=session.environment_config,
        turn_index=session.real_step_count,
        prediction_accuracy=judge.score,
        previous_goal=previous_goal,
        current_goal=reward_goal,
        previous_completed_levels=session.last_completed_levels,
        current_completed_levels=int(
            update_input.turn_metrics.cumulative_score or session.completed_levels
        ),
        turn_metrics=update_input.turn_metrics,
    )
    ledger_entry = replace(
        provisional_entry,
        reward=reward,
        goal_after=reward_goal,
    )
    session.turn_ledger.append(ledger_entry)
    next_memory, next_goal = _regenerate_memory_goal(
        session,
        memory_model=memory_model,
        goal_model=goal_model,
        current_observation=next_snapshot.observation,
        state_memory=state_memory,
        turn_id=current.turn_id,
        debug=debug,
        persist_goal=False,
    )
    _attach_model_token_usage(
        session,
        debug=debug,
        turn_id=current.turn_id,
    )
    ledger_entry = replace(ledger_entry, goal_after=next_goal)
    session.turn_ledger[-1] = ledger_entry
    session.latest_judge_score = judge
    session.latest_reward = reward
    session.memory = next_memory
    session.goal = next_goal
    _persist_v1_turn(
        session,
        state_memory=state_memory,
        ledger_entry=ledger_entry,
        memory=next_memory,
        goal=next_goal,
        reward=reward,
        judge=judge,
        executed_prediction=executed_prediction,
    )
    debug.clear_model_token_usage(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
    )


def compute_immediate_turn_reward(
    *,
    environment_config: EnvironmentConfig,
    turn_index: int,
    prediction_accuracy: float,
    previous_goal: GoalPrediction | None,
    current_goal: GoalPrediction | None,
    previous_completed_levels: int,
    current_completed_levels: int,
    turn_metrics=None,
) -> TurnReward:
    """Compute immediate proxy reward before delayed World LP is available."""

    lp_weight = _annealed_lp_weight(environment_config, turn_index)
    goal_weight = 1.0 - lp_weight
    goal_delta = _goal_delta(previous_goal, current_goal)
    progress_bonus = (
        environment_config.reward_progress_bonus
        if current_completed_levels > previous_completed_levels
        else 0.0
    )
    prediction_accuracy = max(0.0, min(1.0, prediction_accuracy))
    resource_cost, resource_metadata = _resource_cost(
        environment_config,
        turn_metrics=turn_metrics,
    )
    total = (
        lp_weight * prediction_accuracy
        + goal_weight * goal_delta
        + progress_bonus
        - resource_cost
    )
    return TurnReward(
        prediction_accuracy=prediction_accuracy,
        learning_progress=None,
        goal_delta=goal_delta,
        progress_bonus=progress_bonus,
        resource_cost=resource_cost,
        lp_weight=lp_weight,
        goal_weight=goal_weight,
        total=total,
        metadata={
            "turn_index": turn_index,
            "immediate_reward_proxy": "prediction_accuracy",
            **resource_metadata,
        },
    )


def _with_blended_interest_scores(
    prediction: InterestPrediction,
    *,
    lp_weight: float,
    goal_weight: float,
) -> InterestPrediction:
    enriched_values: list[CandidateValuePrediction] = []
    for value in prediction.candidate_values:
        confidence_adjusted_lp = (
            value.confidence * value.expected_learning_progress
        )
        blended_score = (
            lp_weight * confidence_adjusted_lp
            + goal_weight * value.expected_goal_delta
        )
        metadata = dict(value.metadata)
        metadata.update(
            {
                "lp_weight": lp_weight,
                "goal_weight": goal_weight,
                "confidence_adjusted_learning_progress": confidence_adjusted_lp,
                "blended_score": blended_score,
            }
        )
        enriched_values.append(replace(value, metadata=metadata))
    metadata = dict(prediction.metadata)
    metadata.update(
        {
            "lp_weight": lp_weight,
            "goal_weight": goal_weight,
            "scoring_formula": (
                "lp_weight * confidence * expected_learning_progress "
                "+ goal_weight * expected_goal_delta"
            ),
        }
    )
    return replace(
        prediction,
        candidate_values=tuple(enriched_values),
        metadata=metadata,
    )


def _candidate_actions(
    *,
    agent: OrchestratorAgentModel,
    memory: MemoryDocument,
    goal: GoalPrediction,
    current_observation,
    action_space: Sequence[ActionSpec],
    max_candidates: int,
    recent_action_history,
    glossary_actions: Sequence[ActionSpec],
) -> tuple[AgentCandidateAction, ...]:
    simple_actions = [action for action in action_space if not action.is_complex()]
    candidates = [
        AgentCandidateAction(
            action=action,
            source="runtime_simple_action",
            rank=index,
        )
        for index, action in enumerate(simple_actions[:max_candidates])
    ]
    remaining = max_candidates - len(candidates)
    if remaining > 0 and any(action.is_complex() for action in action_space):
        candidates.extend(
            agent.propose_candidate_actions(
                memory=memory,
                goal=goal,
                current_observation=current_observation,
                action_space=action_space,
                max_candidates=remaining,
                recent_action_history=tuple(recent_action_history),
                glossary_actions=glossary_actions,
            )
        )
    return _distinct_ranked_candidates(candidates, max_candidates=max_candidates)


def _world_predictions(
    session: GameLoopSession,
    *,
    world_model: WorldModel,
    memory: MemoryDocument,
    candidates: Sequence[AgentCandidateAction],
    glossary_actions: Sequence[ActionSpec],
    debug: DebugBus,
) -> tuple[WorldPrediction, ...]:
    current = require_current(session)
    predictions: list[WorldPrediction] = []
    for candidate in candidates:
        started_at = helpers.perf_counter()
        with runtime_timing.span(
            "game_loop.world_predict",
            turn_id=current.turn_id,
            step=current.observation.step,
        ):
            prediction = world_model.predict_transition(
                WorldPredictionInput(
                    run_id=session.config.run_id,
                    game_id=session.game_id,
                    candidate_index=candidate.rank,
                    current_observation=current.observation,
                    action=candidate.action,
                    memory=memory,
                    glossary_actions=glossary_actions,
                    metadata={"candidate_source": candidate.source},
                )
            )
        debug.emit(
            ModelCallCompleted(
                role="world",
                duration_seconds=helpers.perf_counter() - started_at,
            )
        )
        debug.capture_model_inputs(current.to_frame_context(), current.turn_id, world_model)
        predictions.append(prediction)
    return tuple(predictions)


def _interest_prediction(
    session: GameLoopSession,
    *,
    interest_model: InterestModel,
    memory: MemoryDocument,
    goal: GoalPrediction,
    candidates: Sequence[AgentCandidateAction],
    world_predictions: Sequence[WorldPrediction],
    recent_action_history,
    debug: DebugBus,
) -> InterestPrediction:
    current = require_current(session)
    started_at = helpers.perf_counter()
    with runtime_timing.span(
        "game_loop.interest_score_candidates",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        prediction = interest_model.score_candidates(
            InterestPredictionInput(
                run_id=session.config.run_id,
                game_id=session.game_id,
                turn_id=current.turn_id,
                current_observation=current.observation,
                memory=memory,
                goal=goal,
                candidates=candidates,
                world_predictions=world_predictions,
                recent_action_history=tuple(recent_action_history),
            )
        )
    enriched = _with_blended_interest_scores(
        prediction,
        lp_weight=_annealed_lp_weight(session.environment_config, session.real_step_count),
        goal_weight=(
            1.0
            - _annealed_lp_weight(
                session.environment_config,
                session.real_step_count,
            )
        ),
    )
    debug.emit(
        ModelCallCompleted(
            role="interest",
            duration_seconds=helpers.perf_counter() - started_at,
        )
    )
    debug.capture_model_inputs(current.to_frame_context(), current.turn_id, interest_model)
    return enriched


def _judge_executed_prediction(
    session: GameLoopSession,
    *,
    reward_judge_model: RewardJudgeModel,
    prediction: WorldPrediction,
    change_summary: str,
    debug: DebugBus,
) -> RewardJudgeScore:
    current = require_current(session)
    next_snapshot = require_next(session)
    decision = require_decision(session)
    started_at = helpers.perf_counter()
    with runtime_timing.span(
        "game_loop.reward_judge",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        judge = reward_judge_model.judge_prediction(
            RewardJudgeInput(
                run_id=session.config.run_id,
                game_id=session.game_id,
                turn_id=current.turn_id,
                action=decision.final_action,
                prediction=prediction,
                change_summary=change_summary,
                previous_observation=current.observation,
                current_observation=next_snapshot.observation,
            )
        )
    debug.emit(
        ModelCallCompleted(
            role="reward_judge",
            duration_seconds=helpers.perf_counter() - started_at,
        )
    )
    debug.capture_model_inputs(
        current.to_frame_context(),
        current.turn_id,
        reward_judge_model,
    )
    return judge


def _regenerate_memory_goal(
    session: GameLoopSession,
    *,
    memory_model: MemoryModel,
    goal_model: GoalModel,
    current_observation,
    state_memory: StateMemory | None,
    turn_id: int,
    debug: DebugBus,
    persist_goal: bool = True,
) -> tuple[MemoryDocument, GoalPrediction]:
    if session.first_observation is None:
        raise RuntimeError("Memory bootstrap requires first observation")

    started_at = helpers.perf_counter()
    memory = memory_model.build_memory(
        MemoryBuildInput(
            run_id=session.config.run_id,
            game_id=session.game_id,
            first_observation=session.first_observation,
            current_observation=current_observation,
            ledger=_memory_ledger_entries(session, memory_model=memory_model),
        )
    )
    debug.emit(
        ModelCallCompleted(
            role="memory",
            duration_seconds=helpers.perf_counter() - started_at,
        )
    )
    if session.current is not None:
        debug.capture_model_inputs(session.current.to_frame_context(), turn_id, memory_model)

    started_at = helpers.perf_counter()
    goal = goal_model.predict_goal(
        GoalPredictionInput(
            run_id=session.config.run_id,
            game_id=session.game_id,
            memory=memory,
            current_observation=current_observation,
            previous_goal=session.goal,
        )
    )
    debug.emit(
        ModelCallCompleted(
            role="goal",
            duration_seconds=helpers.perf_counter() - started_at,
        )
    )
    if session.current is not None:
        debug.capture_model_inputs(session.current.to_frame_context(), turn_id, goal_model)

    session.memory = memory
    session.goal = goal
    if persist_goal and state_memory is not None:
        state_memory.write_goal_prediction(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=turn_id,
            goal_prediction=goal,
            memory_document=memory.document,
            metadata={"bootstrap": turn_id == 0},
        )
    return memory, goal


def _memory_ledger_entries(
    session: GameLoopSession,
    *,
    memory_model: MemoryModel,
) -> tuple[MemoryLedgerEntry, ...]:
    """Return the Memory-facing ledger without rewards or model hypotheses."""

    crop_edges = helpers.model_input_crop_edges(memory_model)
    return tuple(
        MemoryLedgerEntry(
            turn_id=entry.turn_id,
            action=model_facing_action_text(entry.action, crop_edges=crop_edges),
            change_summary=entry.change_summary,
        )
        for entry in session.turn_ledger
    )


def _predict_reward_goal(
    session: GameLoopSession,
    *,
    goal_model: GoalModel,
    current_observation,
    memory: MemoryDocument,
    previous_goal: GoalPrediction | None,
    turn_id: int,
    debug: DebugBus,
) -> GoalPrediction:
    """Predict next-frame Goal from previous Memory for immediate reward only."""

    current = require_current(session)
    started_at = helpers.perf_counter()
    goal = goal_model.predict_goal(
        GoalPredictionInput(
            run_id=session.config.run_id,
            game_id=session.game_id,
            memory=memory,
            current_observation=current_observation,
            previous_goal=previous_goal,
            metadata={"reward_only": True},
        )
    )
    debug.emit(
        ModelCallCompleted(
            role="goal",
            duration_seconds=helpers.perf_counter() - started_at,
        )
    )
    debug.capture_model_inputs(current.to_frame_context(), turn_id, goal_model)
    return replace(
        goal,
        metadata={
            **goal.metadata,
            "reward_only": True,
        },
    )


def _attach_model_token_usage(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    turn_id: int,
) -> None:
    """Copy captured model-token totals into the current turn metrics."""

    update_input = require_update_input(session)
    usage = debug.model_token_usage(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=turn_id,
    )
    update_input.turn_metrics.model_prompt_tokens = int(
        usage.get("prompt_tokens", 0)
    )
    update_input.turn_metrics.model_completion_tokens = int(
        usage.get("completion_tokens", 0)
    )
    update_input.turn_metrics.model_total_tokens = int(
        usage.get("total_tokens", 0)
    )


def _persist_animation_turn(
    session: GameLoopSession,
    *,
    state_memory: StateMemory | None,
    ledger_entry: TurnLedgerEntry,
    memory: MemoryDocument,
    goal: GoalPrediction,
) -> None:
    if state_memory is None:
        return
    current = require_current(session)
    state_memory.write_goal_prediction(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        goal_prediction=goal,
        memory_document=memory.document,
        metadata={"bootstrap": False, "controllable": False},
    )
    state_memory.write_turn_ledger(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        m_state_id=current.source_state_id,
        action=ledger_entry.action,
        change_summary=ledger_entry.change_summary,
        memory_document=memory.document,
        goal_prediction=goal,
        reward=None,
        metadata=ledger_entry.metadata,
    )


def _persist_v1_turn(
    session: GameLoopSession,
    *,
    state_memory: StateMemory | None,
    ledger_entry: TurnLedgerEntry,
    memory: MemoryDocument,
    goal: GoalPrediction,
    reward: TurnReward,
    judge: RewardJudgeScore,
    executed_prediction: WorldPrediction,
) -> None:
    if state_memory is None:
        return
    current = require_current(session)
    candidate_record_ids: dict[int, int] = {}
    value_by_index = _interest_value_by_index(session.interest_prediction)
    for candidate, prediction in zip(session.candidate_actions, session.world_predictions):
        metadata = dict(prediction.metadata)
        interest_value = value_by_index.get(prediction.candidate_index)
        if interest_value is not None:
            metadata["interest_value"] = _candidate_value_json(
                interest_value,
                crop_edges=_runtime_agent_crop_edges(session),
            )
        record = state_memory.write_candidate_prediction(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=current.turn_id,
            candidate_index=prediction.candidate_index,
            action=prediction.action,
            prediction=prediction.predicted_change,
            source=candidate.source,
            metadata=metadata,
        )
        candidate_record_ids[prediction.candidate_index] = record.id
    state_memory.write_judge_score(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        candidate_prediction_id=candidate_record_ids.get(
            executed_prediction.candidate_index
        ),
        score=judge.score,
        notes=judge.notes,
        error_tags=judge.error_tags,
        metadata=judge.metadata,
    )
    state_memory.write_reward(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        reward=reward,
        metadata={"prediction_accuracy": judge.score},
    )
    state_memory.write_goal_prediction(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        goal_prediction=goal,
        memory_document=memory.document,
        metadata={"bootstrap": False},
    )
    state_memory.write_turn_ledger(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        m_state_id=current.source_state_id,
        action=ledger_entry.action,
        change_summary=ledger_entry.change_summary,
        memory_document=memory.document,
        goal_prediction=goal,
        reward=reward,
        metadata=ledger_entry.metadata,
    )
    _write_replay_samples(
        session,
        state_memory=state_memory,
        reward=reward,
        change_summary=ledger_entry.change_summary,
        executed_prediction=executed_prediction,
        judge=judge,
    )


def _write_replay_samples(
    session: GameLoopSession,
    *,
    state_memory: StateMemory,
    reward: TurnReward,
    change_summary: str,
    executed_prediction: WorldPrediction,
    judge: RewardJudgeScore,
) -> None:
    current = require_current(session)
    next_snapshot = require_next(session)
    world_request = _required_training_request(
        executed_prediction.metadata,
        role="world",
    )
    decision = require_decision(session)
    agent_request = _required_training_request(
        decision.trace.metadata,
        role="agent",
    )
    interest = session.interest_prediction
    if interest is None:
        raise RuntimeError("interest replay sample requires Interest prediction")
    interest_request = _required_training_request(
        interest.metadata,
        role="interest",
    )
    action_json = _action_json(decision.final_action)
    candidate_score_table = _candidate_score_table(
        interest,
        crop_edges=_runtime_agent_crop_edges(session),
    )
    state_memory.write_replay_sample(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        role="world",
        prompt={
            "role": "world",
            "phase": "complete",
            "request": world_request,
            "source_turn_id": current.turn_id,
            "candidate_index": executed_prediction.candidate_index,
        },
        completion={"target": {"predicted_change": change_summary}},
        reward=judge.score,
        held_out=False,
        metadata={
            "base_model": session.environment_config.online_lora.base_model,
            "base_model_path": session.environment_config.online_lora.base_model_path,
            "request_model": world_request.get("model"),
            "learning_progress_eval": {
                "action": action_json,
                "change_summary": change_summary,
                "previous_observation": to_memory_jsonable(current.observation),
                "current_observation": to_memory_jsonable(next_snapshot.observation),
                "candidate_index": executed_prediction.candidate_index,
                "request_model": world_request.get("model"),
                "schema_name": executed_prediction.metadata.get(
                    "training_schema_name"
                ),
            },
        },
    )
    state_memory.write_replay_sample(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        role="interest",
        prompt={
            "role": "interest",
            "phase": "complete",
            "request": interest_request,
            "source_turn_id": current.turn_id,
            "candidate_count": len(session.candidate_actions),
        },
        completion={"target": {}},
        reward=0.0,
        held_out=False,
        metadata={
            "base_model": session.environment_config.online_lora.base_model,
            "base_model_path": session.environment_config.online_lora.base_model_path,
            "request_model": interest_request.get("model"),
            "executed_candidate_index": executed_prediction.candidate_index,
            "candidate_score_table": candidate_score_table,
            "label_components": {
                "goal_delta": reward.goal_delta,
                "lp_weight": reward.lp_weight,
                "goal_weight": reward.goal_weight,
                "progress_bonus": reward.progress_bonus,
                "resource_cost": reward.resource_cost,
            },
        },
    )
    state_memory.write_replay_sample(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id,
        role="agent",
        prompt={
            "role": "agent",
            "phase": "select_action",
            "request": agent_request,
            "source_turn_id": current.turn_id,
            "candidate_count": len(session.candidate_actions),
        },
        completion={"target": {"action": action_json}},
        reward=reward.total,
        held_out=False,
        metadata={
            "base_model": session.environment_config.online_lora.base_model,
            "base_model_path": session.environment_config.online_lora.base_model_path,
            "request_model": agent_request.get("model"),
            "executed_candidate_index": executed_prediction.candidate_index,
            "candidate_score_table": candidate_score_table,
            "reward_components": {
                "lp_weight": reward.lp_weight,
                "goal_weight": reward.goal_weight,
                "goal_delta": reward.goal_delta,
                "progress_bonus": reward.progress_bonus,
                "resource_cost": reward.resource_cost,
                "prediction_accuracy": reward.prediction_accuracy,
                "original_learning_progress": reward.learning_progress,
                "original_total": reward.total,
                "resource_cost_components": reward.metadata.get(
                    "resource_cost_components",
                    {},
                ),
            },
        },
    )


def _required_training_request(
    metadata: dict[str, Any] | None,
    *,
    role: str,
) -> dict[str, Any]:
    request = (metadata or {}).get("training_request")
    if not isinstance(request, dict):
        raise RuntimeError(f"{role} replay sample requires exact vLLM request")
    return request


def _executed_world_prediction(
    predictions: Sequence[WorldPrediction],
    action: ActionSpec,
) -> WorldPrediction:
    for prediction in predictions:
        if _same_action(prediction.action, action):
            return prediction
    raise RuntimeError("executed action is missing its World prediction")


def _distinct_ranked_candidates(
    candidates: Sequence[AgentCandidateAction],
    *,
    max_candidates: int,
) -> tuple[AgentCandidateAction, ...]:
    ranked: list[AgentCandidateAction] = []
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    for candidate in candidates:
        identity = _action_identity(candidate.action)
        if identity in seen:
            continue
        seen.add(identity)
        ranked.append(
            AgentCandidateAction(
                action=candidate.action,
                source=candidate.source,
                rank=len(ranked),
                rationale=candidate.rationale,
            )
        )
        if len(ranked) >= max_candidates:
            break
    return tuple(ranked)


def _annealed_lp_weight(config: EnvironmentConfig, turn_index: int) -> float:
    horizon = max(1, config.max_actions_per_level)
    progress = min(1.0, max(0.0, float(turn_index) / float(horizon)))
    return (
        config.reward_lp_weight_start
        + (config.reward_lp_weight_end - config.reward_lp_weight_start) * progress
    )


def _goal_delta(
    previous_goal: GoalPrediction | None,
    current_goal: GoalPrediction | None,
) -> float:
    if previous_goal is None or current_goal is None:
        return 0.0
    previous = max(previous_goal.steps_remaining, 1)
    return (previous_goal.steps_remaining - current_goal.steps_remaining) / previous


def _resource_cost(
    config: EnvironmentConfig,
    *,
    turn_metrics,
) -> tuple[float, dict[str, Any]]:
    """Return configured action/time/token cost for one real turn."""

    trace_seconds = 0.0
    prompt_tokens = completion_tokens = total_tokens = 0
    if turn_metrics is not None:
        trace_seconds = max(0.0, float(turn_metrics.trace_cost or 0.0))
        prompt_tokens = max(0, int(turn_metrics.model_prompt_tokens or 0))
        completion_tokens = max(0, int(turn_metrics.model_completion_tokens or 0))
        total_tokens = max(0, int(turn_metrics.model_total_tokens or 0))
    action_cost = config.reward_action_penalty
    trace_cost = trace_seconds * config.reward_trace_seconds_penalty
    prompt_cost = (
        prompt_tokens / 1000.0 * config.reward_input_token_penalty_per_1k
    )
    completion_cost = (
        completion_tokens / 1000.0 * config.reward_output_token_penalty_per_1k
    )
    total = action_cost + trace_cost + prompt_cost + completion_cost
    return total, {
        "resource_cost_components": {
            "action_cost": action_cost,
            "trace_seconds": trace_seconds,
            "trace_cost": trace_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_token_cost": prompt_cost,
            "completion_token_cost": completion_cost,
        }
    }


def _require_memory(session: GameLoopSession) -> MemoryDocument:
    if session.memory is None:
        raise RuntimeError("v1 loop requires Memory before decision")
    return session.memory


def _require_goal(session: GameLoopSession) -> GoalPrediction:
    if session.goal is None:
        raise RuntimeError("v1 loop requires Goal before decision")
    return session.goal


def _same_action(left: ActionSpec, right: ActionSpec) -> bool:
    return _action_identity(left) == _action_identity(right)


def _interest_value_by_index(
    interest: InterestPrediction | None,
) -> dict[int, CandidateValuePrediction]:
    if interest is None:
        return {}
    return {
        value.candidate_index: value
        for value in interest.candidate_values
    }


def _candidate_score_table(
    interest: InterestPrediction,
    *,
    crop_edges: Any | None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _candidate_value_json(value, crop_edges=crop_edges)
        for value in interest.candidate_values
    )


def _candidate_value_json(
    value: CandidateValuePrediction,
    *,
    crop_edges: Any | None,
) -> dict[str, Any]:
    metadata = dict(value.metadata)
    return {
        "candidate_index": value.candidate_index,
        "action": _action_json(value.action),
        "model_action": _model_action_json(value.action, crop_edges=crop_edges),
        "action_name": value.action.name,
        "expected_learning_progress": value.expected_learning_progress,
        "confidence": value.confidence,
        "confidence_adjusted_learning_progress": metadata.get(
            "confidence_adjusted_learning_progress"
        ),
        "expected_goal_delta": value.expected_goal_delta,
        "blended_score": metadata.get("blended_score"),
        "notes": value.notes,
        "metadata": metadata,
    }


def _action_identity(action: ActionSpec) -> tuple[str, tuple[tuple[str, Any], ...]]:
    return (action.name, tuple(sorted((action.data or {}).items())))


def _runtime_agent_crop_edges(session: GameLoopSession) -> Any | None:
    return (
        session.environment_config.models.agent.options.get(
            "input_image_crop_arc_grid_edges"
        )
        or session.environment_config.models.shared_vlm.options.get(
            "input_image_crop_arc_grid_edges"
        )
    )


def _model_action_json(
    action: ActionSpec,
    *,
    crop_edges: Any | None,
) -> dict[str, Any]:
    action_json: dict[str, Any] = {"action_id": action.name}
    if action.name == "ACTION6" and action.data:
        action_json["data"] = {
            "x": arc_grid_to_normalized_1000(
                action.data,
                "x",
                crop_edges=crop_edges,
            ),
            "y": arc_grid_to_normalized_1000(
                action.data,
                "y",
                crop_edges=crop_edges,
            ),
        }
    elif action.data:
        action_json["data"] = dict(action.data)
    return action_json


def _action_json(action: ActionSpec) -> dict[str, Any]:
    return to_memory_jsonable(asdict(action) if is_dataclass(action) else action)
