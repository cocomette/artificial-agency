"""Debug helpers for collecting captured provider I/O."""

from __future__ import annotations

from typing import Any


def collect_model_input_payload(adapter: Any) -> dict[str, Any]:
    """Return captured prompt/request details from a model adapter."""

    payload: dict[str, Any] = {}
    _collect_prompt_request_payload(adapter, payload)

    nested_openai = getattr(adapter, "_openai", None)
    if nested_openai is not None:
        _collect_prompt_request_payload(nested_openai, payload)

    return payload


def collect_model_io_payload(adapter: Any) -> dict[str, Any]:
    """Return captured provider request and raw response details."""

    payload: dict[str, Any] = {}
    _collect_prompt_request_payload(adapter, payload)
    _collect_response_payload(adapter, payload)

    provider = getattr(adapter, "provider", None)
    if provider is not None:
        _collect_prompt_request_payload(provider, payload)
        _collect_response_payload(provider, payload)

    nested_openai = getattr(adapter, "_openai", None)
    if nested_openai is not None:
        _collect_prompt_request_payload(nested_openai, payload)
        _collect_response_payload(nested_openai, payload)

    return payload


def _collect_prompt_request_payload(adapter: Any, payload: dict[str, Any]) -> None:
    """Add captured prompt/request fields from one adapter-like object."""

    prompt = getattr(adapter, "last_prompt", None)
    if prompt:
        payload["prompt"] = prompt

    request = getattr(adapter, "last_request", None)
    if request is not None:
        payload["request"] = request


def _collect_response_payload(adapter: Any, payload: dict[str, Any]) -> None:
    """Add raw provider response fields from one adapter-like object."""

    response_text = getattr(adapter, "last_response_text", None)
    if response_text is not None:
        payload["response_text"] = response_text

    response_metadata = getattr(adapter, "last_response_metadata", None)
    if response_metadata is not None:
        payload["response_metadata"] = response_metadata
