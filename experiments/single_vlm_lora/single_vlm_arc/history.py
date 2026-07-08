"""Rolling interaction history for the single-VLM experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.contracts import ActionSpec

from single_vlm_arc.actions import action_to_json


@dataclass(slots=True)
class Transition:
    """One real or toy environment transition."""

    turn: int
    observation: Any
    action: ActionSpec
    next_observation: Any
    action_index: int
    log_probability: float
    prediction_loss: float
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RollingHistory:
    """Bounded frame/action history plus replay transitions."""

    def __init__(self, frame_history_n: int) -> None:
        if frame_history_n <= 0:
            raise ValueError("frame_history_n must be positive")
        self.frame_history_n = frame_history_n
        self.transitions: list[Transition] = []

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def recent_frames(self, current_frame: Any) -> list[Any]:
        frames = [decision_frame(item.observation) for item in self.transitions]
        frames.append(current_frame)
        return frames[-self.frame_history_n :]

    def recent_frame_observation_ids(self, current_observation: Any) -> list[str | None]:
        ids = [getattr(item.observation, "id", None) for item in self.transitions]
        ids.append(getattr(current_observation, "id", None))
        return ids[-self.frame_history_n :]

    def recent_action_text(self) -> str:
        items = self.transitions[-self.frame_history_n :]
        if not items:
            return "none"
        return "; ".join(
            f"t{item.turn}:{item.action.name}:{item.action.data or {}}"
            for item in items
        )

    def holdout(self, min_transitions: int) -> Transition | None:
        if len(self.transitions) < min_transitions:
            return None
        return self.transitions[0]

    def build_prompt(
        self,
        *,
        game_id: str,
        turn: int,
        valid_actions: tuple[str, ...],
    ) -> str:
        """Build the compact model prompt for one decision."""

        return (
            "You are controlling an ARC-AGI-3 game from images.\n"
            "Images are ordered chronologically from oldest to newest; "
            "the final image is the current state.\n"
            "Action glossary: ACTION1=up, ACTION2=down, ACTION3=left, "
            "ACTION4=right, ACTION5=interact/select/execute, "
            "ACTION6=target x,y on the 64x64 grid, ACTION7=undo-style action.\n"
            "Choose one valid action from the visible state, recent action effects, "
            "and which action is still informative for predicting future frames. "
            "The trainable heads emit the action; use the images and history to choose.\n"
            f"game_id: {game_id}\n"
            f"turn: {turn}\n"
            f"valid_actions: {', '.join(valid_actions)}\n"
            f"recent_actions: {self.recent_action_text()}\n"
            "Also predict the next 64x64 ARC palette frame after the selected action."
        )


def transition_to_json(transition: Transition) -> dict[str, Any]:
    """Return a JSON-safe transition summary."""

    return {
        "turn": transition.turn,
        "action": action_to_json(transition.action),
        "action_index": transition.action_index,
        "log_probability": transition.log_probability,
        "prediction_loss": transition.prediction_loss,
        "reward": transition.reward,
        "metadata": transition.metadata,
    }


def decision_frame(observation: Any) -> Any:
    """Return the final controllable frame from one observation bundle."""

    frame, _, _ = decision_frame_with_metadata(observation)
    return frame


def decision_frame_with_metadata(observation: Any) -> tuple[Any, int | None, int]:
    """Return `(frame, index, count)` for the canonical decision frame."""

    frames = tuple(getattr(observation, "frames", ()) or ())
    if frames:
        return frames[-1], len(frames) - 1, len(frames)
    frame = getattr(observation, "frame", None)
    if frame is None:
        return None, None, 0
    return frame, 0, 1
