"""Render per-turn model input debug records."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image
import streamlit as st

from debug.dashboard.model_inputs import (
    MODEL_INPUT_SLOTS,
    PredictionOverlay,
    ProviderOutput,
    SentImage,
    prediction_overlay,
    provider_output,
    records_for_slot,
    sent_images,
    token_counts,
)

_SENT_IMAGE_COLUMNS = 2
_SENT_IMAGE_PREVIEW_SIZE = (256, 256)


def render_model_inputs(records: list[dict[str, Any]]) -> None:
    """Render six model-call subtabs for raw provider inputs."""

    if not records:
        st.info("No model input debug records captured for this turn.")
        return

    tabs = st.tabs([label for _, label in MODEL_INPUT_SLOTS])
    for tab, (slot, _label) in zip(tabs, MODEL_INPUT_SLOTS, strict=True):
        with tab:
            _render_slot(records_for_slot(records, slot), slot=slot)


def _render_slot(records: list[dict[str, Any]], *, slot: str) -> None:
    if not records:
        st.info("No provider request captured for this model call.")
        return

    for index, record in enumerate(records, start=1):
        title = _record_title(record, index=index)
        with st.expander(title, expanded=index == len(records)):
            _render_record_summary(record)
            _render_sent_images(record)
            _render_provider_output(record, slot=slot)
            _render_request_breakdown(_dict(record.get("request")))
            st.write("Raw provider request")
            st.code(_json_text(record.get("request")), language="json")


def _render_record_summary(record: dict[str, Any]) -> None:
    counts = token_counts(record)
    cols = st.columns(6)
    cols[0].metric("Backend", str(record.get("provider") or "-"))
    cols[1].metric("Model", str(record.get("model") or "-"))
    cols[2].metric("Phase", str(record.get("phase") or "-"))
    cols[3].metric("Input tokens", _token_label(counts["input"]))
    cols[4].metric("Output tokens", _token_label(counts["output"]))
    cols[5].metric("Total tokens", _token_label(counts["total"]))

    metadata = _dict(record.get("metadata"))
    if metadata:
        with st.expander("Record metadata", expanded=False):
            st.code(_json_text(metadata), language="json")


def _render_sent_images(record: dict[str, Any]) -> None:
    images = sent_images(record)
    with st.expander(f"Sent Images ({len(images)})", expanded=False):
        if not images:
            st.info("No image payloads captured in this provider request.")
            return
        for start in range(0, len(images), _SENT_IMAGE_COLUMNS):
            row_images = images[start : start + _SENT_IMAGE_COLUMNS]
            columns = st.columns(len(row_images))
            for column, image in zip(columns, row_images, strict=True):
                with column:
                    _render_sent_image(image)


def _render_sent_image(image: SentImage) -> None:
    if image.image is None:
        st.warning(f"{image.label}: {image.error or 'image unavailable'}")
        return
    width, height = image.image.size
    preview_width, preview_height = _SENT_IMAGE_PREVIEW_SIZE
    preview = image.image.resize(
        _SENT_IMAGE_PREVIEW_SIZE,
        Image.Resampling.NEAREST,
    )
    st.caption(
        f"{image.label} | source {width} x {height} px | "
        f"preview {preview_width} x {preview_height} px"
    )
    st.image(preview, width=preview_width)


def _render_provider_output(record: dict[str, Any], *, slot: str) -> None:
    output = provider_output(record)
    with st.expander("Provider Output", expanded=False):
        if not output.available:
            st.info("No provider output captured for this record.")
            return
        if slot in {"world", "goal"}:
            _render_prediction_overlay(
                prediction_overlay(record, display_size=_SENT_IMAGE_PREVIEW_SIZE)
            )
        _render_provider_output_payload(output)


def _render_prediction_overlay(overlay: PredictionOverlay) -> None:
    if overlay.image is not None:
        width, height = overlay.image.size
        source_width, source_height = overlay.source_size or overlay.image.size
        st.write("Predicted bounding boxes")
        st.image(
            overlay.image,
            caption=(
                "Overlay image: first decoded provider input image "
                f"({overlay.source_label or 'input frame'}) | "
                f"source {source_width} x {source_height} px | "
                f"preview {width} x {height} px | "
                f"{overlay.drawn_count}/{overlay.area_count} boxes"
            ),
            width=width,
        )

    if overlay.warnings:
        with st.expander("BBox overlay warnings", expanded=False):
            for warning in overlay.warnings:
                st.warning(warning)


def _render_provider_output_payload(output: ProviderOutput) -> None:
    if output.parsed_json is not None:
        st.write("Parsed output")
        st.json(output.parsed_json)

    if output.text is not None:
        st.write("Raw output text")
        language = "json" if output.parsed_json is not None else "text"
        st.code(output.text, language=language)

    if output.metadata:
        st.write("Response metadata")
        st.code(_json_text(output.metadata), language="json")

    if output.raw_response is not None:
        st.write("Raw response payload")
        st.code(_json_text(output.raw_response), language="json")


def _render_request_breakdown(request: dict[str, Any]) -> None:
    if not request:
        st.info("No request payload available.")
        return

    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions:
        st.write("Instructions")
        st.code(instructions, language="text")

    input_items = request.get("input")
    if isinstance(input_items, list):
        st.write("Input sequence")
        for index, item in enumerate(input_items, start=1):
            _render_openai_input_item(index, item)

    messages = request.get("messages")
    if isinstance(messages, list):
        st.write("Messages")
        for index, message in enumerate(messages, start=1):
            _render_chat_message(index, message)

    option_payload = {
        key: value
        for key, value in request.items()
        if key not in {"instructions", "input", "messages"}
    }
    if option_payload:
        with st.expander("Request options", expanded=False):
            st.code(_json_text(option_payload), language="json")


def _render_openai_input_item(index: int, item: Any) -> None:
    item_payload = _dict(item)
    label = str(item_payload.get("role") or item_payload.get("type") or "item")
    with st.expander(f"Input item {index}: {label}", expanded=False):
        content = item_payload.get("content")
        if isinstance(content, list):
            for content_index, part in enumerate(content, start=1):
                _render_content_part(content_index, part)
        else:
            st.code(_json_text(item), language="json")


def _render_content_part(index: int, part: Any) -> None:
    payload = _dict(part)
    part_type = str(payload.get("type") or f"part {index}")
    if part_type == "input_text" and isinstance(payload.get("text"), str):
        st.write(f"Content {index}: input_text")
        st.code(payload["text"], language="text")
        return
    if part_type == "input_image":
        st.write(f"Content {index}: input_image")
        with st.expander("Raw image payload", expanded=False):
            st.code(_json_text(payload), language="json")
        return
    st.write(f"Content {index}: {part_type}")
    st.code(_json_text(part), language="json")


def _render_chat_message(index: int, message: Any) -> None:
    payload = _dict(message)
    role = str(payload.get("role") or "message")
    with st.expander(f"Message {index}: {role}", expanded=False):
        content = payload.get("content")
        if isinstance(content, str):
            st.code(content, language="text")
        elif isinstance(content, list):
            for content_index, part in enumerate(content, start=1):
                _render_chat_content_part(content_index, part)
        elif content is not None:
            st.code(_json_text(content), language="json")

        images = payload.get("images")
        if isinstance(images, list) and images:
            st.write(f"Images: {len(images)}")
            for image_index, image_payload in enumerate(images, start=1):
                with st.expander(f"Raw image payload {image_index}", expanded=False):
                    st.code(str(image_payload), language="text")

        extras = {
            key: value
            for key, value in payload.items()
            if key not in {"role", "content", "images"}
        }
        if extras:
            st.write("Additional fields")
            st.code(_json_text(extras), language="json")


def _render_chat_content_part(index: int, part: Any) -> None:
    payload = _dict(part)
    part_type = str(payload.get("type") or f"part {index}")
    if part_type == "text" and isinstance(payload.get("text"), str):
        st.write(f"Content {index}: text")
        st.code(payload["text"], language="text")
        return

    if part_type == "image_url":
        st.write(f"Content {index}: image_url")
        with st.expander("Image URL payload", expanded=False):
            st.code(
                _json_text(_display_image_url_payload(payload.get("image_url"))),
                language="json",
            )
        return

    st.write(f"Content {index}: {part_type}")
    st.code(_json_text(part), language="json")


def _record_title(record: dict[str, Any], *, index: int) -> str:
    phase = str(record.get("phase") or "request")
    attempt = int(record.get("attempt") or 0)
    return f"Request {index} | {phase} | attempt {attempt}"


def _token_label(value: int | None) -> str:
    if value is None:
        return "Unavailable"
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _display_image_url_payload(value: Any) -> Any:
    if isinstance(value, dict):
        payload = dict(value)
        payload["url"] = _data_url_summary(payload.get("url"))
        return payload
    return _data_url_summary(value)


def _data_url_summary(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith("data:"):
        return value
    prefix, separator, encoded = value.partition(",")
    if not separator:
        return value
    return f"{prefix},<base64 {len(encoded)} chars>"


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
