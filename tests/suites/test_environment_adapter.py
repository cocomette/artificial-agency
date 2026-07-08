"""Tests for ARC environment adapter id resolution helpers."""

from __future__ import annotations

import pytest
from arc_agi import OperationMode

from face_of_agi.environment.adapter import (
    ArcEnvironmentAdapter,
    ArcEnvironmentWrapperAdapter,
    KaggleArcadeAdapter,
)
from face_of_agi.environment import adapter as environment_adapter
from face_of_agi.environment.config import EnvironmentConfig


class _FakeGame:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id


class _FakeArcade:
    def __init__(self, game_ids: tuple[str, ...]) -> None:
        self.games = tuple(_FakeGame(game_id) for game_id in game_ids)
        self.made: list[str] = []

    def get_environments(self):
        return self.games

    def make(self, game_id, **kwargs):
        del kwargs
        self.made.append(game_id)
        return object()


def test_arc_environment_adapter_accepts_short_game_id_from_local_catalog() -> None:
    local_arcade = _FakeArcade(("ls20-aaa",))
    adapter = ArcEnvironmentAdapter(
        environments_dir="",
        recordings_dir="",
        local_arcade=local_arcade,
    )

    assert adapter.select_game_by_id("ls20") == "ls20-aaa"
    assert local_arcade.made == ["ls20-aaa"]


def test_arc_environment_adapter_rejects_ambiguous_short_game_id() -> None:
    adapter = ArcEnvironmentAdapter(
        environments_dir="",
        recordings_dir="",
        local_arcade=_FakeArcade(("ls20-aaa", "ls20-bbb")),
    )

    with pytest.raises(RuntimeError, match="matched multiple ARC games"):
        adapter.select_game_by_id("ls20")


def test_precreated_environment_wrapper_accepts_short_game_id() -> None:
    adapter = ArcEnvironmentWrapperAdapter(
        game_id="ls20-aaa",
        environment=object(),
    )

    assert adapter.select_game_by_id("ls20") == "ls20-aaa"


def test_precreated_environment_wrapper_does_not_match_distinct_full_ids() -> None:
    adapter = ArcEnvironmentWrapperAdapter(
        game_id="ls20-aaa",
        environment=object(),
    )

    with pytest.raises(RuntimeError, match="not 'ls20-bbb'"):
        adapter.select_game_by_id("ls20-bbb")


def test_kaggle_arcade_adapter_uses_config_operation_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeArcade:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(environment_adapter, "Arcade", FakeArcade)

    KaggleArcadeAdapter.from_config(
        EnvironmentConfig(
            game_index=0,
            max_actions_per_level=1,
            operation_mode=OperationMode.COMPETITION,
        )
    )

    assert captured["operation_mode"] == OperationMode.COMPETITION
