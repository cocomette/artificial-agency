"""Tests for dynamic model action glossaries."""

from __future__ import annotations

import pytest

from face_of_agi.contracts import ActionSpec
from face_of_agi.models.action_glossary import action_glossary_text


def test_action_glossary_renders_only_supplied_actions_in_order() -> None:
    text = action_glossary_text(
        (
            ActionSpec(action_id="ACTION2"),
            ActionSpec(action_id="ACTION1"),
        ),
        mode="committed_action",
    )

    assert text.startswith(
        "## Action glossary\n\n"
        "helper to interpret the playable actions from a user experience UI "
        "perspective. It can be useful to understand certain actions but does "
        "not replace observed facts.\n\n"
    )
    assert "- `ACTION2`: down." in text
    assert text.index("`ACTION2`") < text.index("`ACTION1`")
    assert "- `ACTION3`" not in text


def test_action_glossary_renders_action6_text() -> None:
    actions = (ActionSpec(action_id="ACTION6"),)

    decision_text = action_glossary_text(actions, mode="agent_decision")
    update_text = action_glossary_text(actions, mode="agent_update")

    assert (
        "- `ACTION6`: coordinate action with a required visual target note."
        in decision_text
    )
    assert (
        "- `ACTION6`: coordinate action with a required visual target note."
        in update_text
    )


def test_action_glossary_rejects_unknown_actions() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        action_glossary_text(
            (ActionSpec(action_id="ACTION99"),),
            mode="committed_action",
        )
