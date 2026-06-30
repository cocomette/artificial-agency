"""Runtime config YAML editor controls."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from debug.dashboard import config_manager

TEXT_KEY = "config_editor_text"
LOADED_KEY = "config_editor_loaded_name"
SAVE_AS_KEY = "config_save_as_name"


def render_config_editor(
    selected_name: str,
    selected_path: Path,
    *,
    saved_config_key: str | None = None,
) -> None:
    """Render validation, save, and save-as controls for one config."""

    if st.session_state.get(LOADED_KEY) != selected_name or _editor_state_missing():
        st.session_state[TEXT_KEY] = config_manager.read_config(selected_path)
        st.session_state[LOADED_KEY] = selected_name
        st.session_state[SAVE_AS_KEY] = f"{selected_path.stem}_copy.yaml"

    edited_text = str(
        st.text_area(
            "YAML",
            key=TEXT_KEY,
            height=640,
        )
    )
    validation = config_manager.validate_config_text(edited_text)
    if validation.valid:
        st.success(validation.message)
    else:
        st.error(validation.message)

    save_as_name = str(st.text_input("Save As", key=SAVE_AS_KEY))
    save_col, save_as_col, _ = st.columns([0.12, 0.14, 0.74])

    if save_col.button(
        "Save",
        disabled=not validation.valid,
        key="config_save_button",
        type="primary",
        use_container_width=True,
    ):
        try:
            saved_path = config_manager.save_config(selected_path, edited_text)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Saved {config_manager.config_label(saved_path)}.")

    if save_as_col.button(
        "Save As",
        disabled=not validation.valid,
        key="config_save_as_button",
        use_container_width=True,
    ):
        try:
            saved_path = config_manager.save_config_as(save_as_name, edited_text)
        except Exception as exc:
            st.error(str(exc))
        else:
            saved_label = config_manager.config_label(saved_path)
            st.success(f"Saved {saved_label}.")
            if saved_config_key is not None:
                st.session_state[saved_config_key] = saved_label
            st.rerun()


def _editor_state_missing() -> bool:
    return TEXT_KEY not in st.session_state or SAVE_AS_KEY not in st.session_state
