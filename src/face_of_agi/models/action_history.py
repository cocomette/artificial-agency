"""Prompt-facing action history grouping helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionHistoryScoreAdvanceMarker,
    ActionSpec,
    ChangeSummaryElement,
)


@dataclass(frozen=True, slots=True)
class ActionHistoryGroup:
    """One controllable action plus following animation evidence."""

    action: ActionHistoryEntry | None
    animations: tuple[ActionHistoryEntry, ...] = ()


ActionHistoryRow = (
    ActionHistoryGroup
    | ActionHistoryResetMarker
    | ActionHistoryScoreAdvanceMarker
)


def model_facing_action_text(
    action: ActionSpec,
    *,
    crop_box_normalized: Any | None = None,
) -> str:
    """Render one action for prompts, using target-only ACTION6 history."""

    if action.name == "ACTION6":
        if action.data is None:
            return f"{action.name}(x,y normalized_0_1000,target)"
        if action.target is None or not action.target.strip():
            raise ValueError("ACTION6 action history entries require a target")
        return f"{action.name} target={json.dumps(action.target.strip())}"
    if action.data:
        return f"{action.name} {json.dumps(action.data, sort_keys=True)}"
    return action.name


def model_facing_action_text_for_crop(
    crop_box_normalized: Any | None,
) -> Callable[[ActionSpec], str]:
    """Return an action renderer for one model-visible crop configuration."""

    return lambda action: model_facing_action_text(
        action,
        crop_box_normalized=crop_box_normalized,
    )


def group_action_history(
    history: Sequence[ActionHistoryItem],
) -> tuple[ActionHistoryRow, ...]:
    """Group animation rows under actions and keep reset rows explicit."""

    rows: list[ActionHistoryRow] = []
    current_action: ActionHistoryEntry | None = None
    current_animations: list[ActionHistoryEntry] = []
    orphan_animations: list[ActionHistoryEntry] = []

    def flush_current_group() -> None:
        nonlocal current_action, current_animations, orphan_animations
        if orphan_animations:
            rows.append(
                ActionHistoryGroup(action=None, animations=tuple(orphan_animations))
            )
            orphan_animations = []
        if current_action is not None:
            rows.append(
                ActionHistoryGroup(
                    action=current_action,
                    animations=tuple(current_animations),
                )
            )
            current_action = None
            current_animations = []

    for entry in history:
        if isinstance(
            entry,
            (ActionHistoryResetMarker, ActionHistoryScoreAdvanceMarker),
        ):
            flush_current_group()
            rows.append(entry)
            continue

        if entry.controllable:
            if orphan_animations:
                rows.append(
                    ActionHistoryGroup(action=None, animations=tuple(orphan_animations))
                )
                orphan_animations = []
            if current_action is not None:
                rows.append(
                    ActionHistoryGroup(
                        action=current_action,
                        animations=tuple(current_animations),
                    )
                )
            current_action = entry
            current_animations = []
            continue

        if current_action is None:
            orphan_animations.append(entry)
        else:
            current_animations.append(entry)

    flush_current_group()
    return tuple(rows)


def grouped_action_history_text(
    history: Sequence[ActionHistoryItem],
    *,
    action_text: Callable[[ActionSpec], str],
    numbered: bool,
    latest_description: str | None = None,
) -> str:
    """Render action history with animation rows nested under action rows."""

    if not history:
        return "none"

    latest_item = history[-1]
    groups = group_action_history(history)
    lines: list[str] = []
    if numbered:
        if latest_description is None:
            raise ValueError("numbered action history requires latest_description")
        lines.append(latest_description)
        for index, row in enumerate(groups, start=1):
            if isinstance(row, ActionHistoryResetMarker):
                lines.append(
                    _reset_marker_line(
                        row,
                        prefix=f"{index}. ",
                        latest=row is latest_item,
                    )
                )
                continue
            if isinstance(row, ActionHistoryScoreAdvanceMarker):
                lines.append(
                    _score_advance_marker_line(
                        row,
                        prefix=f"{index}. ",
                        latest=row is latest_item,
                    )
                )
                continue
            lines.extend(
                _group_lines(
                    row,
                    prefix=f"{index}. ",
                    indent="   ",
                    latest_item=latest_item,
                    action_text=action_text,
                )
            )
        return "\n".join(lines)

    for row in groups:
        if isinstance(row, ActionHistoryResetMarker):
            lines.append(
                _reset_marker_line(
                    row,
                    prefix="- ",
                    latest=row is latest_item,
                )
            )
            continue
        if isinstance(row, ActionHistoryScoreAdvanceMarker):
            lines.append(
                _score_advance_marker_line(
                    row,
                    prefix="- ",
                    latest=row is latest_item,
                )
            )
            continue
        lines.extend(
            _group_lines(
                row,
                prefix="- ",
                indent="  ",
                latest_item=latest_item,
                action_text=action_text,
            )
        )
    return "\n".join(lines)


def _group_lines(
    group: ActionHistoryGroup,
    *,
    prefix: str,
    indent: str,
    latest_item: ActionHistoryItem,
    action_text: Callable[[ActionSpec], str],
) -> list[str]:
    lines: list[str] = []
    if group.action is None:
        lines.append(f"{prefix}animation_without_prior_action:")
    else:
        lines.append(
            prefix
            + action_history_entry_text(
                group.action,
                latest=group.action is latest_item,
                action_text=action_text,
            )
        )
    if group.animations:
        lines.append(f"{indent}animation_after:")
        for animation in group.animations:
            lines.append(
                f"{indent}- "
                + action_history_entry_text(
                    animation,
                    latest=animation is latest_item,
                    action_text=action_text,
                )
            )
    return lines


def action_history_entry_text(
    entry: ActionHistoryEntry,
    *,
    latest: bool = False,
    action_text: Callable[[ActionSpec], str],
) -> str:
    """Render one raw action history entry."""

    text = action_text(entry.action)
    if not entry.controllable:
        text += " [animation]"
    skipped_count = max(0, entry.skipped_intermediate_animation_frame_count)
    if skipped_count:
        text += f" [skipped_intermediate_animation_frames={skipped_count}]"
    if latest:
        text += " [latest]"
    text += f" [changed_pixels={entry.changed_pixel_count}]"
    if entry.changed_pixel_percent is not None:
        text += (
            f" [changed_area={_changed_pixel_percent_text(entry.changed_pixel_percent)}]"
        )
    if entry.completed_levels is not None:
        text += f" [completed_levels={entry.completed_levels}]"
    if entry.action_count is not None:
        text += f" [action_count={entry.action_count}]"
    summary = _change_summary_text(entry)
    if summary:
        if entry.change_elements:
            text += f" Elements and associated changes:\n{summary}"
        else:
            text += f" change: {summary}"
    return text


def _reset_marker_line(
    marker: ActionHistoryResetMarker,
    *,
    prefix: str,
    latest: bool,
) -> str:
    text = "GAME_RESET"
    if latest:
        text += " [latest]"
    text += f" [reason={marker.reason}] [restart_count={marker.restart_count}]"
    return prefix + text


def _score_advance_marker_line(
    marker: ActionHistoryScoreAdvanceMarker,
    *,
    prefix: str,
    latest: bool,
) -> str:
    text = "SCORE_ADVANCE"
    if latest:
        text += " [latest]"
    text += (
        f" [previous_score={_nullable_metric_text(marker.previous_score)}]"
        f" [new_score={_nullable_metric_text(marker.new_score)}]"
        f" [delta={_nullable_metric_text(marker.delta)}]"
    )
    return prefix + text


def _nullable_metric_text(value: float | None) -> str:
    if value is None:
        return "null"
    return str(value)


def _changed_pixel_percent_text(value: float) -> str:
    if value == 0:
        return "0%"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _change_summary_text(entry: ActionHistoryEntry) -> str:
    if entry.change_elements:
        return _change_elements_text(entry.change_elements)

    summary = entry.change_summary.strip()
    if entry.changed_pixel_count != 0:
        return summary

    identity = "First and final frames are identical."
    if not summary or _generic_zero_change_summary(summary):
        return identity
    if summary.lower().startswith(identity.lower()):
        return summary
    return f"{identity} {summary}"


def _generic_zero_change_summary(summary: str) -> bool:
    normalized = summary.strip().lower().rstrip(".!")
    return normalized in {
        "no change",
        "no changes",
        "nothing changed",
        "no visible change",
        "no visible changes",
        "no visible playfield change",
        "no visible playfield changes",
        "no visible playfield change occurred",
    }


def _change_elements_text(elements: Sequence[ChangeSummaryElement]) -> str:
    return "\n".join(_change_element_line(element) for element in elements)


def _change_element_line(element: ChangeSummaryElement) -> str:
    name = element.element_name.strip()
    description = element.element_description.strip()
    mutation = element.element_mutation.strip()
    if not mutation:
        mutation = "no detected changes for this element"
    return f"- {name}: {description}; mutations: {mutation}"
