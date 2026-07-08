"""Prompt-facing action glossary rendering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from face_of_agi.contracts import ActionSpec

ActionGlossaryMode = Literal[
    "agent_decision",
    "agent_update",
    "committed_action",
]

_BASE_ACTION_DESCRIPTIONS = {
    "RESET": "initialize or restart the game or level state.",
    "ACTION1": "up.",
    "ACTION2": "down.",
    "ACTION3": "left.",
    "ACTION4": "right.",
    "ACTION5": (
        "simple game-specific action, such as interact, select, rotate, "
        "attach/detach, or execute."
    ),
    "ACTION7": "undo-style simple action.",
    "NONE": "internal no-control action for animation-frame unrolling.",
}

_ACTION6_DESCRIPTIONS = {
    "agent_decision": (
        "coordinate action. For a new decision, output `x` and `y` in "
        "normalized visual coordinates from 0 to 1000, with `(0,0)` at the "
        "top-left, `x` increasing right, and `y` increasing down."
    ),
    "agent_update": (
        "coordinate action. Model-facing action-history `ACTION6` data is "
        "rendered in normalized visual 0..1000 coordinates. When writing "
        "future policy for the agent, use normalized visual 0..1000 "
        "coordinates or visual regions."
    ),
    "committed_action": (
        "coordinate action mapped to the game grid. Submitted and historical "
        "`ACTION6` data uses ARC 64x64 grid coordinates, with each axis as an "
        "integer from 0 to 63."
    ),
}


def action_glossary_text(
    actions: Sequence[ActionSpec],
    *,
    mode: ActionGlossaryMode,
) -> str:
    """Render a Markdown action glossary for exactly the supplied actions."""

    names = action_glossary_names(actions)
    lines = ["## Action glossary", ""]
    lines.extend(f"- `{name}`: {action_description(name, mode=mode)}" for name in names)
    return "\n".join(lines)


def append_action_glossary(
    instructions: str,
    actions: Sequence[ActionSpec],
    *,
    mode: ActionGlossaryMode,
) -> str:
    """Append a dynamic action glossary to instruction text."""

    return instructions.strip() + "\n\n" + action_glossary_text(actions, mode=mode)


def action_glossary_names(actions: Sequence[ActionSpec]) -> tuple[str, ...]:
    """Return unique action names in prompt order, rejecting unknown actions."""

    if not actions:
        raise ValueError("action glossary requires at least one action")
    names: list[str] = []
    seen: set[str] = set()
    for action in actions:
        name = action.name
        action_description(name, mode="committed_action")
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)


def action_description(name: str, *, mode: ActionGlossaryMode) -> str:
    """Return the prompt-facing description for one known action name."""

    if name == "ACTION6":
        return _ACTION6_DESCRIPTIONS[mode]
    try:
        return _BASE_ACTION_DESCRIPTIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown action for glossary: {name}") from exc
