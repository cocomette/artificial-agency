"""ARC action masking and selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from face_of_agi.contracts import ActionSpec

ACTION_NAMES: tuple[str, ...] = (
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


@dataclass(slots=True)
class SelectedAction:
    """One sampled or greedy action from the model output."""

    action: ActionSpec
    action_index: int
    action_name: str
    log_prob: Any
    probability: float
    x: int | None = None
    y: int | None = None


def action_name(action: ActionSpec) -> str:
    """Return a stable action name from ARC or string-backed actions."""

    return action.name


def available_action_names(action_space: Sequence[ActionSpec]) -> tuple[str, ...]:
    """Return valid action names in model action vocabulary order."""

    names = {action_name(action) for action in action_space}
    return tuple(name for name in ACTION_NAMES if name in names)


def valid_action_mask(action_space: Sequence[ActionSpec]) -> list[bool]:
    """Return an eight-action boolean mask for the active ARC action space."""

    valid = set(available_action_names(action_space))
    return [name in valid for name in ACTION_NAMES]


def mask_action_logits(logits: Any, action_space: Sequence[ActionSpec]) -> Any:
    """Mask invalid action logits with negative infinity."""

    import torch

    mask = torch.tensor(
        valid_action_mask(action_space),
        dtype=torch.bool,
        device=logits.device,
    )
    if not bool(mask.any()):
        raise ValueError("action space did not contain any known ARC action")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def masked_action_probabilities(
    action_logits: Any,
    action_space: Sequence[ActionSpec],
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Return JSON-safe masked probabilities for the full action vocabulary."""

    import torch

    logits = action_logits[0] if action_logits.ndim == 2 else action_logits
    masked = mask_action_logits(logits, action_space)
    scaled = masked / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    valid = set(available_action_names(action_space))
    return {
        name: float(probs[index].detach().cpu().item()) if name in valid else 0.0
        for index, name in enumerate(ACTION_NAMES)
    }


def select_action(
    *,
    action_logits: Any,
    coord_logits: Any,
    action_space: Sequence[ActionSpec],
    mode: str,
    temperature: float,
) -> SelectedAction:
    """Select a valid action and optional ACTION6 coordinates."""

    import torch

    if action_logits.ndim == 2:
        logits = action_logits[0]
    else:
        logits = action_logits
    masked = mask_action_logits(logits, action_space)
    if mode == "argmax":
        action_index = int(masked.argmax().item())
        probs = torch.softmax(masked, dim=-1)
        log_probs = torch.log_softmax(masked, dim=-1)
    elif mode == "sample":
        scaled = masked / max(float(temperature), 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        action_index = int(distribution.sample().item())
        log_probs = torch.log(probs.clamp_min(1e-12))
    else:
        raise ValueError(f"unsupported action selection mode: {mode}")

    name = ACTION_NAMES[action_index]
    log_prob = log_probs[action_index]
    probability = float(probs[action_index].detach().cpu().item())
    source_action = _matching_action(action_space, name)

    x: int | None = None
    y: int | None = None
    data: dict[str, int] | None = None
    if name == "ACTION6":
        coord = coord_logits[0] if coord_logits.ndim == 2 else coord_logits
        x = clamp_coordinate(int(coord[:64].argmax().item()))
        y = clamp_coordinate(int(coord[64:].argmax().item()))
        data = {"x": x, "y": y}

    return SelectedAction(
        action=ActionSpec(action_id=source_action.action_id, data=data),
        action_index=action_index,
        action_name=name,
        log_prob=log_prob,
        probability=probability,
        x=x,
        y=y,
    )


def _matching_action(action_space: Sequence[ActionSpec], name: str) -> ActionSpec:
    for action in action_space:
        if action.name == name:
            return action
    raise ValueError(f"selected action {name} is not in the active action space")


def clamp_coordinate(value: int) -> int:
    """Clamp an ARC coordinate to the valid 0-63 range."""

    return max(0, min(63, int(value)))


def action_to_json(action: ActionSpec) -> dict[str, Any]:
    """Return a JSON-safe action payload."""

    return {
        "action_id": action.name,
        "data": action.data,
    }
