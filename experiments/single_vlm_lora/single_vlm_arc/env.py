"""Environment session helpers for real ARC and deterministic dry runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from face_of_agi.contracts import ActionSpec, EnvironmentInfo, Observation

from single_vlm_arc.config import EnvironmentExperimentConfig


class GameSession(Protocol):
    """Common session boundary used by the experiment runner."""

    game_id: str

    def reset(self) -> Observation:
        ...

    def step(self, action: ActionSpec) -> Observation:
        ...

    def get_action_space(self) -> Sequence[ActionSpec]:
        ...

    def get_info(self) -> EnvironmentInfo:
        ...


def build_session(
    config: EnvironmentExperimentConfig,
    *,
    dry_run: bool,
) -> GameSession:
    """Build a toy or real ARC session."""

    if dry_run:
        return ToyArcSession(seed=config.seed)
    return ArcSession(config)


class ArcSession:
    """Thin wrapper around the repo's ARC environment adapter."""

    def __init__(self, config: EnvironmentExperimentConfig) -> None:
        from arc_agi import OperationMode

        from face_of_agi.environment.adapter import ArcEnvironmentAdapter
        from face_of_agi.environment.config import EnvironmentConfig

        environment_config = EnvironmentConfig(
            game_index=config.game_index,
            max_actions_per_level=config.max_turns,
            game_id=config.game_id,
            operation_mode=OperationMode(str(config.operation_mode)),
            environments_dir=config.environments_dir,
            recordings_dir=config.recordings_dir,
            seed=config.seed,
            save_recording=config.save_recording,
        )
        self.adapter = ArcEnvironmentAdapter.from_config(environment_config)
        selected_game_id = config.game_id or self.adapter.resolve_game_id(
            config.game_index
        )
        self.game_id = self.adapter.select_game_by_id(selected_game_id)

    def reset(self) -> Observation:
        return self.adapter.reset()

    def step(self, action: ActionSpec) -> Observation:
        return self.adapter.step(action)

    def get_action_space(self) -> Sequence[ActionSpec]:
        return self.adapter.get_action_space()

    def get_info(self) -> EnvironmentInfo:
        return self.adapter.get_info()


@dataclass(slots=True)
class ToyArcSession:
    """Small deterministic 64x64 ARC-like environment for tests."""

    seed: int = 0
    game_id: str = "toy-single-vlm"
    _step: int = field(default=0, init=False)
    _x: int = field(default=4, init=False)
    _y: int = field(default=4, init=False)
    _levels_completed: int = field(default=0, init=False)
    _last_observation: Observation | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._step = 0
        self._x = 4
        self._y = 4
        self._levels_completed = 0
        self._last_observation = self._make_observation()

    def reset(self) -> Observation:
        self._step = 0
        self._x = 4
        self._y = 4
        self._levels_completed = 0
        self._last_observation = self._make_observation()
        return self._last_observation

    def step(self, action: ActionSpec) -> Observation:
        name = action.name
        if name == "ACTION1":
            self._y = max(0, self._y - 1)
        elif name == "ACTION2":
            self._y = min(63, self._y + 1)
        elif name == "ACTION3":
            self._x = max(0, self._x - 1)
        elif name == "ACTION4":
            self._x = min(63, self._x + 1)
        elif name == "ACTION5":
            self._levels_completed += int(self._x >= 6 and self._y >= 6)
        elif name == "ACTION6" and action.data:
            self._x = max(0, min(63, int(action.data.get("x", self._x))))
            self._y = max(0, min(63, int(action.data.get("y", self._y))))
        elif name == "RESET":
            return self.reset()
        self._step += 1
        self._last_observation = self._make_observation()
        return self._last_observation

    def get_action_space(self) -> Sequence[ActionSpec]:
        return (
            ActionSpec("RESET"),
            ActionSpec("ACTION1"),
            ActionSpec("ACTION2"),
            ActionSpec("ACTION3"),
            ActionSpec("ACTION4"),
            ActionSpec("ACTION5"),
            ActionSpec("ACTION6"),
        )

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id=self.game_id,
            available_actions=tuple(self.get_action_space()),
            levels_completed=self._levels_completed,
            win_levels=0,
        )

    def _make_observation(self) -> Observation:
        animation_frame = np.zeros((64, 64), dtype=np.uint8)
        animation_frame[:, 32] = 1
        animation_frame[32, :] = 1
        animation_frame[6, 6] = 9
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[:, 32] = 1
        frame[32, :] = 1
        frame[self._y, self._x] = 8
        frame[6, 6] = 9
        return Observation(
            id=f"{self.game_id}-step-{self._step}",
            step=self._step,
            frame=animation_frame,
            frames=(animation_frame, frame),
        )
