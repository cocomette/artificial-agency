"""vLLM adapter for the Interest role."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_of_agi.contracts import CandidateValuePrediction, InterestPrediction
from face_of_agi.models.action_history import grouped_action_history_text
from face_of_agi.models.interest.config import VLLMInterestConfig
from face_of_agi.models.interest.contracts import (
    InterestPredictionInput,
    interest_prediction_json_schema,
)
from face_of_agi.models.vllm_roles import (
    VLLMJsonRoleClient,
    action_text,
    bounded_float,
    non_negative_int,
    observation_image,
    parse_json_object,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class VLLMInterestAdapter:
    """Interest role backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMInterestConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = VLLMJsonRoleClient(
            config=config,
            call_slot="interest",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            client=client,
        )

    def score_candidates(
        self,
        prediction_input: InterestPredictionInput,
    ) -> InterestPrediction:
        """Score the full candidate set in one batch call."""

        text = self.provider.complete_json(
            prompt_text=_interest_prompt(self.config, prediction_input),
            output_schema=interest_prediction_json_schema(),
            schema_name="interest_prediction",
            images=(observation_image(self.config, prediction_input.current_observation),),
        )
        payload = parse_json_object(text, label="interest")
        candidate_values = _parse_candidate_values(
            payload,
            prediction_input=prediction_input,
        )
        metadata = {
            "backend": "vllm",
            "model": self.config.model,
            "usage": self.provider.last_usage,
        }
        return InterestPrediction(
            candidate_values=candidate_values,
            metadata=metadata,
        )


def _parse_candidate_values(
    payload: dict[str, Any],
    *,
    prediction_input: InterestPredictionInput,
) -> tuple[CandidateValuePrediction, ...]:
    raw_values = payload.get("candidate_values")
    if not isinstance(raw_values, list):
        raise RuntimeError("interest response requires candidate_values list")

    candidate_by_index = {
        candidate.rank: candidate for candidate in prediction_input.candidates
    }
    expected_indices = set(candidate_by_index)
    values_by_index: dict[int, CandidateValuePrediction] = {}
    for raw in raw_values:
        if not isinstance(raw, dict):
            raise RuntimeError("interest candidate_values items must be objects")
        candidate_index = non_negative_int(
            raw.get("candidate_index"),
            label="candidate_index",
        )
        if candidate_index not in expected_indices:
            raise RuntimeError(
                f"interest response included unknown candidate_index {candidate_index}"
            )
        if candidate_index in values_by_index:
            raise RuntimeError(
                f"interest response duplicated candidate_index {candidate_index}"
            )
        candidate = candidate_by_index[candidate_index]
        values_by_index[candidate_index] = CandidateValuePrediction(
            candidate_index=candidate_index,
            action=candidate.action,
            expected_learning_progress=bounded_float(
                raw.get("expected_learning_progress"),
                label="expected_learning_progress",
                minimum=-1.0,
                maximum=1.0,
            ),
            expected_goal_delta=bounded_float(
                raw.get("expected_goal_delta"),
                label="expected_goal_delta",
                minimum=-1.0,
                maximum=1.0,
            ),
            confidence=bounded_float(
                raw.get("confidence"),
                label="confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            notes=str(raw.get("notes") or "").strip(),
        )

    missing_indices = sorted(expected_indices - set(values_by_index))
    if missing_indices:
        raise RuntimeError(
            "interest response missing candidate_index values: "
            + ", ".join(str(index) for index in missing_indices)
        )
    return tuple(values_by_index[index] for index in sorted(values_by_index))


def _interest_prompt(
    config: VLLMInterestConfig,
    prediction_input: InterestPredictionInput,
) -> str:
    prediction_by_index = {
        prediction.candidate_index: prediction
        for prediction in prediction_input.world_predictions
    }
    candidate_lines: list[str] = []
    for candidate in prediction_input.candidates:
        world_prediction = prediction_by_index.get(candidate.rank)
        predicted_change = (
            world_prediction.predicted_change
            if world_prediction is not None
            else "not available"
        )
        candidate_lines.append(
            "\n".join(
                [
                    f"- candidate_index: {candidate.rank}",
                    "  action: "
                    + action_text(
                        candidate.action,
                        crop_edges=config.input_image_crop_arc_grid_edges,
                    ),
                    f"  source: {candidate.source}",
                    f"  predicted_change: {predicted_change}",
                    f"  rationale: {candidate.rationale}",
                ]
            )
        )
    return "\n\n".join(
        [
            f"run_id: {prediction_input.run_id}",
            f"game_id: {prediction_input.game_id}",
            f"turn_id: {prediction_input.turn_id}",
            "Attached image: current frame only.",
            "Current Memory document:",
            prediction_input.memory.document,
            "Goal prediction:",
            _goal_text(prediction_input.goal),
            "Candidate actions with World predictions:",
            "\n".join(candidate_lines),
            "Recent actions:",
            _recent_actions_text(
                prediction_input.recent_action_history,
                crop_edges=config.input_image_crop_arc_grid_edges,
            ),
        ]
    )


def _goal_text(goal) -> str:
    return "\n".join(
        [
            f"goal: {goal.goal}",
            f"subgoals: {list(goal.subgoals)}",
            f"steps_remaining: {goal.steps_remaining}",
            f"confidence: {goal.confidence}",
        ]
    )


def _recent_actions_text(history, *, crop_edges: Any | None) -> str:
    if not history:
        return "none"
    return grouped_action_history_text(
        history,
        action_text=lambda action: action_text(action, crop_edges=crop_edges),
        numbered=True,
        latest_description=(
            "Numbered oldest-to-newest. The [latest] marker identifies the "
            "transition, reset, or score marker that produced the attached frame."
        ),
    )
