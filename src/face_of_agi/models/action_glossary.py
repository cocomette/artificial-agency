"""Prompt-facing action glossary rendering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from face_of_agi.contracts import ActionSpec

ActionGlossaryMode = Literal[
    "agent_decision",
    "agent_update",
    "committed_action",
    "world_model",
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
    "NONE": "internal no-control action for animation evidence.",
}

_ACTION6_DESCRIPTIONS = {
    "agent_decision": "coordinate action with a required visual target note.",
    "agent_update": "coordinate action with a required visual target note.",
    "world_model": "coordinate action.",
    "committed_action": "coordinate action mapped to the game grid, shown by target.",
}


def action_glossary_text(
    actions: Sequence[ActionSpec],
    *,
    mode: ActionGlossaryMode,
) -> str:
    """Render a Markdown action glossary for exactly the supplied actions."""

    names = action_glossary_names(actions)
    lines = [
        "## Action glossary",
        "",
        "helper to interpret the playable actions from a user experience UI "
        "perspective. It can be useful to understand certain actions but does "
        "not replace observed facts.",
        "",
    ]
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
