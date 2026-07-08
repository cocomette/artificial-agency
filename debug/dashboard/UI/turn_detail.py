"""Shared memory-turn rendering helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from debug.dashboard.memory_reader import (
    action_label,
    experiment_summary,
    matching_experiments,
    observation_image,
    predicted_description,
    redacted_for_json,
    tool_result_image,
    turn_summary,
)
from debug.dashboard.UI.model_inputs import render_model_inputs


def render_live_status(
    state: dict[str, Any],
    experiments: list[dict[str, Any]],
) -> None:
    """Render compact metrics for the newest selected live state."""

    trace = _dict(state.get("agent_trace"))
    metadata = _dict(state.get("metadata"))
    control_mode = _dict(metadata.get("control_mode"))
    selected_experiments = matching_experiments(experiments, state)

    cols = st.columns(6)
    cols[0].metric("Run", str(state["run_id"]))
    cols[1].metric("Game", str(state["game_id"]))
    cols[2].metric("Turn", str(state["turn_id"]))
    cols[3].metric("Step", str(state["step"]))
    cols[4].metric("Frame", f"{state['frame_index'] + 1}/{state['frame_count']}")
    cols[5].metric("Action", action_label(state.get("chosen_action")))

    left, right = st.columns([1, 2])
    with left:
        render_image(
            observation_image(state.get("current_observation")),
            "Latest persisted frame",
        )
    with right:
        detail_cols = st.columns(4)
        detail_cols[0].metric("M state", str(state["id"]))
        detail_cols[1].metric("Control", str(control_mode.get("reason", "-")))
        detail_cols[2].metric(
            "Controllable",
            str(bool(control_mode.get("controllable", False))),
        )
        detail_cols[3].metric("Experiments", str(len(selected_experiments)))
        st.write("Latest reasoning summary")
        st.code(str(trace.get("reasoning_summary") or ""), language="text")


def render_recent_turns(states: list[dict[str, Any]]) -> None:
    """Render the latest persisted turn table for a live run."""

    st.subheader("Recent Turns")
    recent_states = sorted(states, key=lambda state: int(state["id"]), reverse=True)[:10]
    st.dataframe(
        pd.DataFrame([turn_summary(state) for state in recent_states]),
        width="stretch",
        hide_index=True,
    )


def select_turn(states: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    """Render a selectable turn table and return the selected state."""

    st.subheader("Turns")
    table = pd.DataFrame([turn_summary(state) for state in states])
    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    selected_rows = list(event.selection.rows)
    selected_index = selected_rows[0] if selected_rows else len(states) - 1
    return states[selected_index]


def render_selected_turn(
    state: dict[str, Any],
    experiments: list[dict[str, Any]],
    model_input_records: list[dict[str, Any]] | None = None,
) -> None:
    """Render all detail tabs for one selected M state."""

    st.subheader(
        f"Turn {state['turn_id']} | M state {state['id']} | step {state['step']}"
    )

    overview, agent, predictions, model_inputs, experiment_tab, raw = st.tabs(
        [
            "Overview",
            "Agent X",
            "Predictions",
            "Model Inputs",
            "Experiments",
            "Raw Data",
        ]
    )
    with overview:
        _render_overview(state)
    with agent:
        _render_agent_trace(state)
    with predictions:
        _render_predictions(state)
    with model_inputs:
        render_model_inputs(model_input_records or [])
    with experiment_tab:
        _render_experiments(experiments)
    with raw:
        _render_raw(state, experiments, model_input_records or [])


def render_image(image: Any | None, caption: str) -> None:
    """Render an image payload or a compact empty-frame message."""

    if image is None:
        st.info(f"{caption}: no visual frame")
        return
    st.image(image, caption=caption, width="stretch")


def _render_overview(state: dict[str, Any]) -> None:
    metadata = _dict(state.get("metadata"))
    control_mode = _dict(metadata.get("control_mode"))

    cols = st.columns(4)
    cols[0].metric("Action", action_label(state.get("chosen_action")))
    cols[1].metric("Control", str(control_mode.get("reason", "-")))
    cols[2].metric("Frame", f"{state['frame_index'] + 1}/{state['frame_count']}")
    cols[3].metric("Controllable", str(bool(control_mode.get("controllable", False))))

    left, right = st.columns([1, 2])
    with left:
        render_image(
            observation_image(state.get("current_observation")),
            "Current observed frame",
        )
    with right:
        st.write("Allowed actions")
        st.json(
            [action_label(action) for action in control_mode.get("allowed_actions") or []]
        )


def _render_agent_trace(state: dict[str, Any]) -> None:
    trace = _dict(state.get("agent_trace"))
    metadata = _dict(trace.get("metadata"))

    cols = st.columns(5)
    cols[0].metric("Backend", str(metadata.get("backend", "-")))
    cols[1].metric("Model", str(metadata.get("model", "-")))
    cols[2].metric("Final action", action_label(trace.get("final_action")))
    cols[3].metric("Tool calls", str(metadata.get("tool_call_count", 0)))
    cols[4].metric("Repairs", str(metadata.get("repair_count", 0)))

    st.write("Reasoning summary")
    st.code(str(trace.get("reasoning_summary") or ""), language="text")

    provider_cols = st.columns(2)
    provider_cols[0].write("Provider response ids")
    provider_cols[0].json(metadata.get("provider_response_ids") or [])
    provider_cols[1].write("Usage")
    provider_cols[1].json(metadata.get("usage") or [])

    tool_calls = trace.get("tool_calls") or []
    tool_results = trace.get("tool_results") or []
    if not tool_calls:
        st.info("No agent-requested tool calls recorded for this turn.")
        return

    for index, call in enumerate(tool_calls):
        with st.expander(f"Tool iteration {index + 1}: {call.get('tool', '-')}"):
            call_col, result_col = st.columns(2)
            call_col.write("Tool call")
            call_col.json(redacted_for_json(call))
            if index < len(tool_results):
                result = tool_results[index]
                result_col.write("Tool result")
                _render_tool_prediction_payload(
                    result,
                    image_caption="Tool predicted frame",
                )
                result_col.json(redacted_for_json(result))


def _render_predictions(state: dict[str, Any]) -> None:
    world, goal = st.columns(2)
    with world:
        st.write("World prediction")
        _render_tool_prediction(state.get("world_prediction"))
    with goal:
        st.write("Goal prediction")
        _render_tool_prediction(state.get("goal_prediction"))


def _render_tool_prediction(result: Any) -> None:
    payload = _dict(result)
    if not payload:
        st.info("No prediction recorded.")
        return
    _render_tool_prediction_payload(payload, image_caption="Predicted frame")
    st.write("Explanation")
    st.code(str(payload.get("explanation") or ""), language="text")
    st.write("Metadata")
    st.json(payload.get("metadata") or {})


def _render_tool_prediction_payload(result: Any, *, image_caption: str) -> None:
    image = tool_result_image(_dict(result))
    description = predicted_description(_dict(result))
    if image is not None:
        render_image(image, image_caption)
    elif description is not None:
        st.write("Predicted description")
        st.json(redacted_for_json(description))
    else:
        st.info(f"{image_caption}: no visual frame or description")


def _render_experiments(experiments: list[dict[str, Any]]) -> None:
    if not experiments:
        st.info("No E experiments recorded for this turn.")
        return

    st.dataframe(
        pd.DataFrame([experiment_summary(experiment) for experiment in experiments]),
        width="stretch",
        hide_index=True,
    )
    for experiment in experiments:
        with st.expander(
            f"E experiment {experiment['id']} | {experiment['tool_name']}"
        ):
            left, right = st.columns(2)
            with left:
                render_image(
                    observation_image(experiment.get("output_description")),
                    "Experiment output frame",
                )
                st.write("Output description")
                st.json(redacted_for_json(experiment.get("output_description")))
                st.write("Tool call")
                st.json(redacted_for_json(experiment.get("tool_call")))
            with right:
                st.write("Tool result")
                st.json(redacted_for_json(experiment.get("tool_result")))
                st.write("Metadata")
                st.json(experiment.get("metadata") or {})


def _render_raw(
    state: dict[str, Any],
    experiments: list[dict[str, Any]],
    model_input_records: list[dict[str, Any]],
) -> None:
    st.json(
        redacted_for_json(
            {
                "m_state": state,
                "matching_e_experiments": experiments,
                "matching_model_input_debug_records": model_input_records,
            }
        )
    )


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
