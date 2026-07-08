"""Tests for shared structured-output repair policy."""

from __future__ import annotations

import pytest

from face_of_agi.models.structured_output import (
    provider_repair_callback,
    validate_with_repair,
)


def test_validate_with_repair_stops_after_configured_attempts() -> None:
    repairs: list[int] = []

    def validate(text: str) -> str:
        raise ValueError(f"bad payload {text}")

    def repair(invalid_text: str, validation_error: str, attempt: int) -> str:
        repairs.append(attempt)
        return f"still bad {attempt}"

    with pytest.raises(RuntimeError, match="after 2 repair attempt"):
        validate_with_repair(
            label="test output",
            response="bad",
            text_of=lambda item: item,
            validate=validate,
            repair=repair,
            max_repair_attempts=2,
            error_factory=RuntimeError,
        )

    assert repairs == [1, 2]


def test_validate_with_repair_returns_repaired_value() -> None:
    def validate(text: str) -> str:
        if text != "valid":
            raise ValueError("invalid")
        return text.upper()

    result = validate_with_repair(
        label="test output",
        response="bad",
        text_of=lambda item: item,
        validate=validate,
        repair=lambda _text, _error, _attempt: "valid",
        max_repair_attempts=2,
        error_factory=RuntimeError,
    )

    assert result.value == "VALID"
    assert result.repair_attempts == 1


def test_provider_repair_callback_binds_provider_method_inputs() -> None:
    class Provider:
        def repair_prompt(
            self,
            request: str,
            *,
            invalid_text: str,
            validation_error: str,
            attempt: int,
        ) -> str:
            return f"{request}:{invalid_text}:{validation_error}:{attempt}"

    repair = provider_repair_callback(
        Provider(),
        "repair_prompt",
        args=("request",),
    )

    assert repair is not None
    assert repair("bad", "invalid", 2) == "request:bad:invalid:2"
