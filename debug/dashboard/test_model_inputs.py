"""Tests for dashboard model-input helpers."""

from debug.dashboard.model_inputs import MODEL_INPUT_SLOTS, records_for_slot


def test_model_input_slots_include_game_memory() -> None:
    assert ("memory", "Game Memory") in MODEL_INPUT_SLOTS


def test_records_for_slot_returns_memory_records() -> None:
    records = [
        {"call_slot": "agent", "id": 1},
        {"call_slot": "memory", "id": 2},
        {"call_slot": "memory", "id": 3},
    ]

    assert records_for_slot(records, "memory") == [
        {"call_slot": "memory", "id": 2},
        {"call_slot": "memory", "id": 3},
    ]
