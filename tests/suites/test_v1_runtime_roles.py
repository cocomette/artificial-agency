"""Tests for v1 Memory/World/Goal/Agent runtime helpers."""

from __future__ import annotations

from collections.abc import Sequence

from arcengine import GameAction
import pytest

from face_of_agi.contracts import (
    ActionSpec,
    AgentCandidateAction,
    GoalPrediction,
    MemoryDocument,
    Observation,
    RewardJudgeScore,
    RuntimeConfig,
    TurnMetrics,
    TurnLedgerEntry,
    TurnReward,
    WorldPrediction,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.debug.bus import DebugBus
from face_of_agi.models.memory import MemoryLedgerEntry
from face_of_agi.orchestration.game_loop.session import GameLoopSession
from face_of_agi.orchestration.game_loop import v1_roles


def test_candidate_actions_include_simple_actions_and_distinct_coordinates() -> None:
    simple = ActionSpec(action_id="ACTION1")
    action6 = ActionSpec(action_id=GameAction.ACTION6)
    agent = _CoordinateProposalAgent()

    candidates = v1_roles._candidate_actions(
        agent=agent,
        memory=MemoryDocument(document="memory"),
        goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=5,
            confidence=0.5,
        ),
        current_observation=Observation(id="obs", step=0),
        action_space=(simple, action6),
        max_candidates=3,
        recent_action_history=(),
        glossary_actions=(simple, action6),
    )

    assert [candidate.rank for candidate in candidates] == [0, 1]
    assert [candidate.action.name for candidate in candidates] == [
        "ACTION1",
        "ACTION6",
    ]


def test_turn_reward_anneals_from_lp_to_goal_weight() -> None:
    early = v1_roles.compute_immediate_turn_reward(
        environment_config=EnvironmentConfig(
            max_actions_per_level=10,
            reward_lp_weight_start=0.8,
            reward_lp_weight_end=0.2,
        ),
        turn_index=0,
        prediction_accuracy=0.5,
        previous_goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=10,
            confidence=0.5,
        ),
        current_goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=5,
            confidence=0.5,
        ),
        previous_completed_levels=0,
        current_completed_levels=0,
        turn_metrics=TurnMetrics(),
    )
    late = v1_roles.compute_immediate_turn_reward(
        environment_config=EnvironmentConfig(
            max_actions_per_level=10,
            reward_lp_weight_start=0.8,
            reward_lp_weight_end=0.2,
        ),
        turn_index=10,
        prediction_accuracy=0.5,
        previous_goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=10,
            confidence=0.5,
        ),
        current_goal=GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=5,
            confidence=0.5,
        ),
        previous_completed_levels=0,
        current_completed_levels=1,
        turn_metrics=TurnMetrics(),
    )

    assert early.lp_weight == pytest.approx(0.8)
    assert early.goal_weight == pytest.approx(0.2)
    assert early.goal_delta == 0.5
    assert early.prediction_accuracy == 0.5
    assert early.learning_progress == 0.5
    assert early.metadata["learning_progress_proxy"] == (
        "reward_judge_prediction_accuracy"
    )
    assert late.lp_weight == pytest.approx(0.2)
    assert late.progress_bonus == 1.0


def test_immediate_turn_reward_subtracts_configured_resource_cost() -> None:
    reward = v1_roles.compute_immediate_turn_reward(
        environment_config=EnvironmentConfig(
            max_actions_per_level=10,
            reward_lp_weight_start=1.0,
            reward_lp_weight_end=1.0,
            reward_action_penalty=0.1,
            reward_input_token_penalty_per_1k=0.2,
            reward_output_token_penalty_per_1k=0.3,
        ),
        turn_index=0,
        prediction_accuracy=1.0,
        previous_goal=None,
        current_goal=None,
        previous_completed_levels=0,
        current_completed_levels=0,
        turn_metrics=TurnMetrics(
            model_prompt_tokens=1000,
            model_completion_tokens=2000,
            model_total_tokens=3000,
        ),
    )

    assert reward.resource_cost == pytest.approx(0.9)
    assert reward.total == pytest.approx(0.1)


def test_memory_ledger_entries_strip_internal_turn_fields_and_normalize_action() -> None:
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(game_id="game-1", max_actions_per_level=1),
        game_id="game-1",
        latest_environment_observation=Observation(id="obs", step=0),
        remaining_actions=1,
    )
    action = ActionSpec(action_id=GameAction.ACTION6, data={"x": 32, "y": 32})
    goal = GoalPrediction(
        goal="solve",
        subgoals=("click",),
        steps_remaining=3,
        confidence=0.5,
    )
    session.turn_ledger.append(
        TurnLedgerEntry(
            turn_id=4,
            action=action,
            change_summary="clicked center tile",
            reward=TurnReward(
                prediction_accuracy=0.75,
                learning_progress=0.75,
                goal_delta=0.1,
                progress_bonus=0.0,
                resource_cost=0.0,
                lp_weight=0.5,
                goal_weight=0.5,
                total=0.425,
            ),
            candidate_predictions=(
                WorldPrediction(
                    candidate_index=0,
                    action=action,
                    predicted_change="world hypothesis",
                ),
            ),
            judge_scores=(RewardJudgeScore(score=0.75, notes="close"),),
            goal_before=goal,
            goal_after=goal,
            metadata={"controllable": True, "debug": "internal only"},
        )
    )

    rows = v1_roles._memory_ledger_entries(
        session,
        memory_model=_MemoryWithCrop(),
    )

    assert rows == (
        MemoryLedgerEntry(
            turn_id=4,
            action='ACTION6 {"x": 500, "y": 500}',
            change_summary="clicked center tile",
            reward_feedback={
                "total": 0.425,
                "learning_progress": 0.75,
                "prediction_accuracy": 0.75,
                "goal_delta": 0.1,
                "progress_bonus": 0.0,
                "resource_cost": 0.0,
                "judge_notes": "close",
            },
        ),
    )


def test_reset_memory_goal_preserves_full_run_ledger_and_first_frame() -> None:
    first = Observation(id="first", step=0)
    reset = Observation(id="reset", step=10)
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(game_id="game-1", max_actions_per_level=1),
        game_id="game-1",
        latest_environment_observation=reset,
        remaining_actions=1,
    )
    session.first_observation = first
    session.first_observation_ref = session.current_ref_for(first)
    session.turn_ledger.append(
        TurnLedgerEntry(
            turn_id=1,
            action=ActionSpec(action_id="ACTION1"),
            change_summary="first attempt changed something",
        )
    )
    session.game_restart_count = 1
    session.game_start_reason = "game_over_reset"
    session.frame_turn_count = 7
    memory = _RecordingMemory()

    v1_roles.reset_memory_goal_after_game_over(
        session,
        memory_model=memory,
        goal_model=_StaticGoal(),
        state_memory=None,
        debug=DebugBus.disabled(),
    )

    assert session.first_observation is first
    assert len(session.turn_ledger) == 2
    assert session.turn_ledger[-1].metadata["reset_marker"] is True
    assert memory.inputs[-1].first_observation is first
    assert memory.inputs[-1].current_observation is reset
    assert memory.inputs[-1].ledger == (
        MemoryLedgerEntry(
            turn_id=1,
            action="ACTION1",
            change_summary="first attempt changed something",
        ),
        MemoryLedgerEntry(
            turn_id=7,
            action="NONE",
            change_summary=(
                "GAME_RESET: the ARC environment reset after game over. "
                "Prior mechanics and failed attempts remain relevant, but the "
                "current frame is a fresh post-reset state."
            ),
        ),
    )


class _CoordinateProposalAgent:
    def propose_candidate_actions(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        max_candidates: int,
        recent_action_history=(),
        glossary_actions: Sequence[ActionSpec],
    ) -> tuple[AgentCandidateAction, ...]:
        del memory, goal, current_observation, action_space
        del max_candidates, recent_action_history, glossary_actions
        action = ActionSpec(action_id=GameAction.ACTION6, data={"x": 100, "y": 200})
        return (
            AgentCandidateAction(
                action=action,
                source="agent_coordinate_proposal",
                rank=0,
            ),
            AgentCandidateAction(
                action=action,
                source="agent_coordinate_proposal",
                rank=1,
            ),
        )


class _RecordingMemory:
    def __init__(self) -> None:
        self.inputs = []

    def build_memory(self, build_input):
        self.inputs.append(build_input)
        return MemoryDocument(document="remembered")


class _MemoryWithCrop(_RecordingMemory):
    input_image_crop_arc_grid_edges = 4


class _StaticGoal:
    def predict_goal(self, prediction_input):
        return GoalPrediction(
            goal="solve",
            subgoals=(),
            steps_remaining=1,
            confidence=0.5,
        )
