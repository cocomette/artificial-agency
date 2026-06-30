"""ARC-AGI environment adapter boundary and concrete toolkit wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from arc_agi import (
    Arcade,
    EnvironmentInfo as ArcEnvironmentInfo,
    EnvironmentWrapper,
    OperationMode,
)
from arcengine import FrameDataRaw, GameAction, GameState

from face_of_agi.contracts import ActionSpec, EnvironmentInfo, Observation
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.environment.visualization import ArcRenderer, resolve_visualization


class EnvironmentAdapter(Protocol):
    """Boundary for direct environment interaction."""

    def list_available_games(self) -> Sequence[ArcEnvironmentInfo]:
        """Return discoverable ARC environments in toolkit order."""
        ...

    def list_local_games(self) -> Sequence[ArcEnvironmentInfo]:
        """Return locally downloaded ARC environments."""
        ...

    def resolve_game_id(self, game_index: int) -> str:
        """Resolve the selected game index into one concrete game id."""
        ...

    def select_game_by_id(self, game_id: str) -> str:
        """Select and initialize the requested ARC-AGI game shell."""
        ...

    def reset(self) -> Observation:
        """Reset the current ARC game or level."""
        ...

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, Any] | None = None,
    ) -> Observation:
        """Apply one real ARC-AGI environment action."""
        ...

    def get_action_space(self) -> Sequence[ActionSpec]:
        """Return currently valid ARC actions."""
        ...

    def get_info(self) -> EnvironmentInfo:
        """Return current ARC environment metadata."""
        ...


class ArcEnvironmentAdapter:
    """Thin wrapper over the real ARC-AGI toolkit interfaces.

    NORMAL mode is used to discover the public list and to pull a game the
    first time. OFFLINE mode is used whenever the game is already available
    locally.
    """

    def __init__(
        self,
        *,
        environments_dir: str,
        recordings_dir: str,
        local_arcade: Arcade,
        seed: int = 0,
        save_recording: bool = False,
        renderer: ArcRenderer | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.environments_dir = environments_dir
        self.recordings_dir = recordings_dir
        self.discovery_arcade: Arcade | None = None
        self.local_arcade = local_arcade
        self.seed = seed
        self.save_recording = save_recording
        self.renderer = renderer
        self.render_mode = render_mode
        self._environment: EnvironmentWrapper | None = None
        self._game_id: str | None = None
        self._last_raw_observation: FrameDataRaw | None = None
        self._step_index = 0
        self._available_games: tuple[ArcEnvironmentInfo, ...] | None = None
        self._local_games: tuple[ArcEnvironmentInfo, ...] | None = None

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> "ArcEnvironmentAdapter":
        """Build the real ARC adapter directly from the shell config."""

        local_arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=config.environments_dir,
            recordings_dir=config.recordings_dir,
        )
        visualization = resolve_visualization(
            enabled=config.enable_visualization,
            render_mode=config.render_mode,
        )
        return cls(
            environments_dir=config.environments_dir,
            recordings_dir=config.recordings_dir,
            local_arcade=local_arcade,
            seed=config.seed,
            save_recording=config.save_recording,
            renderer=visualization.renderer,
            render_mode=visualization.render_mode,
        )

    def list_available_games(self) -> Sequence[ArcEnvironmentInfo]:
        """Return the live ARC toolkit game list in NORMAL mode."""

        if self._available_games is None:
            self._available_games = tuple(
                self._require_discovery_arcade().get_environments()
            )
        return self._available_games

    def list_local_games(self) -> Sequence[ArcEnvironmentInfo]:
        """Return the locally downloaded ARC toolkit game list."""

        if self._local_games is None:
            self._local_games = tuple(self.local_arcade.get_environments())
        return self._local_games

    def resolve_game_id(self, game_index: int) -> str:
        """Resolve one selected game index from the discoverable game list."""

        available_games = self.list_available_games()
        if not 0 <= game_index < len(available_games):
            raise RuntimeError(
                f"game index {game_index} is out of range for "
                f"{len(available_games)} discoverable ARC games"
            )
        return available_games[game_index].game_id

    def select_game_by_id(self, game_id: str) -> str:
        """Create a local ARC environment wrapper for one selected game.

        If the game is already present locally, use OFFLINE mode. Otherwise use
        NORMAL mode once so the toolkit can pull it into `environment_files`.
        """

        resolved_game_id = self._resolve_game_request_id(game_id)
        environment = self._make_local_environment(resolved_game_id)
        if environment is None:
            environment = self._download_and_make_environment(resolved_game_id)

        if environment is None:
            raise RuntimeError(
                f"unable to create ARC environment for game '{resolved_game_id}'"
            )

        self._environment = environment
        self._game_id = resolved_game_id
        self._last_raw_observation = None
        self._step_index = 0
        self._local_games = None
        return resolved_game_id

    def reset(self) -> Observation:
        """Reset the selected ARC environment and return the next observation."""

        raw_observation = self._require_environment().reset()
        if raw_observation is None:
            raise RuntimeError(f"reset failed for ARC game '{self._require_game_id()}'")

        self._step_index = 0
        return self._store_observation(raw_observation)

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, Any] | None = None,
    ) -> Observation:
        """Apply one real ARC action and return the resulting frames."""

        raw_observation = self._require_environment().step(
            action.action_id,
            data=action.data,
            reasoning=reasoning,
        )
        if raw_observation is None:
            raise RuntimeError(
                f"step failed for ARC game '{self._require_game_id()}' with action"
                f" '{action.action_id.name}'"
            )

        self._step_index += 1
        return self._store_observation(raw_observation)

    def get_action_space(self) -> Sequence[ActionSpec]:
        """Return the currently valid ARC actions from the last frame data."""

        raw_observation = self._last_raw_observation
        if raw_observation is None:
            return ()

        return tuple(
            ActionSpec(action_id=GameAction.from_id(action_id))
            for action_id in raw_observation.available_actions
        )

    def get_info(self) -> EnvironmentInfo:
        """Return the current ARC game state and progress counters."""

        raw_observation = self._last_raw_observation
        if raw_observation is None:
            return EnvironmentInfo(game_id=self._require_game_id())

        return EnvironmentInfo(
            game_id=raw_observation.game_id or self._require_game_id(),
            state=raw_observation.state,
            available_actions=tuple(self.get_action_space()),
            levels_completed=raw_observation.levels_completed,
            win_levels=raw_observation.win_levels,
            full_reset=raw_observation.full_reset,
            metadata={"raw_frame_data": raw_observation},
        )

    def _store_observation(self, raw_observation: FrameDataRaw) -> Observation:
        """Normalize raw ARC frame data into the local observation contract."""

        self._last_raw_observation = raw_observation
        frames = tuple(raw_observation.frame)
        return Observation(
            id=self._build_observation_id(raw_observation),
            step=self._step_index,
            frame=frames[0] if frames else None,
            frames=frames,
            raw_frame_data=raw_observation,
            metadata={"raw_frame_data": raw_observation},
        )

    def _build_observation_id(self, raw_observation: FrameDataRaw) -> str:
        """Create a stable observation id for one ARC run."""

        guid = raw_observation.guid or "local"
        return f"{self._require_game_id()}-{guid}-step-{self._step_index}"

    def _require_environment(self) -> EnvironmentWrapper:
        """Return the selected ARC environment or fail clearly."""

        if self._environment is None:
            raise RuntimeError("environment game was not selected")
        return self._environment

    def _require_game_id(self) -> str:
        """Return the active game id or fail clearly."""

        if self._game_id is None:
            raise RuntimeError("environment game id was not selected")
        return self._game_id

    def _make_local_environment(self, game_id: str) -> EnvironmentWrapper | None:
        """Create a local wrapper if the game already exists offline."""

        local_ids = {game.game_id for game in self.list_local_games()}
        if game_id not in local_ids:
            return None

        return self.local_arcade.make(
            game_id,
            seed=self.seed,
            save_recording=self.save_recording,
            renderer=self.renderer,
            render_mode=self.render_mode,
        )

    def _download_and_make_environment(self, game_id: str) -> EnvironmentWrapper | None:
        """Use NORMAL mode once so the toolkit can pull the game locally."""

        return self._require_discovery_arcade().make(
            game_id,
            seed=self.seed,
            save_recording=self.save_recording,
            renderer=self.renderer,
            render_mode=self.render_mode,
        )

    def _resolve_game_request_id(self, game_id: str) -> str:
        """Resolve a full or short game id against local then live catalogs."""

        local_matches = _matching_game_ids(game_id, self.list_local_games())
        if local_matches:
            return _single_game_id_match(game_id, local_matches)

        available_matches = _matching_game_ids(game_id, self.list_available_games())
        if available_matches:
            return _single_game_id_match(game_id, available_matches)

        return game_id

    def _require_discovery_arcade(self) -> Arcade:
        """Create the NORMAL-mode discovery client only when it is needed."""

        if self.discovery_arcade is None:
            self.discovery_arcade = Arcade(
                operation_mode=OperationMode.NORMAL,
                environments_dir=self.environments_dir,
                recordings_dir=self.recordings_dir,
            )
        return self.discovery_arcade


class ArcEnvironmentWrapperAdapter:
    """Environment adapter for an already-created ARC environment wrapper."""

    def __init__(
        self,
        *,
        game_id: str,
        environment: EnvironmentWrapper,
    ) -> None:
        self._game_id = game_id
        self._environment = environment
        self._last_raw_observation: FrameDataRaw | None = None
        self._step_index = 0

    def list_available_games(self) -> Sequence[ArcEnvironmentInfo]:
        """Return no discovery catalog for pre-created wrappers."""

        return ()

    def list_local_games(self) -> Sequence[ArcEnvironmentInfo]:
        """Return no local catalog for pre-created wrappers."""

        return ()

    def resolve_game_id(self, game_index: int) -> str:
        """Pre-created wrappers do not support catalog index resolution."""

        raise RuntimeError(
            "pre-created ARC environment wrappers do not resolve game indices"
        )

    def select_game_by_id(self, game_id: str) -> str:
        """Select the already-bound wrapper when the requested game matches."""

        if not _matching_game_id(self._game_id, game_id):
            raise RuntimeError(
                f"pre-created ARC wrapper is for '{self._game_id}', not '{game_id}'"
            )
        self._last_raw_observation = None
        self._step_index = 0
        return self._game_id

    def reset(self) -> Observation:
        """Reset the wrapped ARC environment and return the next observation."""

        raw_observation = self._environment.reset()
        if raw_observation is None:
            raise RuntimeError(f"reset failed for ARC game '{self._game_id}'")

        self._step_index = 0
        return self._store_observation(raw_observation)

    def step(
        self,
        action: ActionSpec,
        reasoning: dict[str, Any] | None = None,
    ) -> Observation:
        """Apply one real ARC action and return the resulting frames."""

        raw_observation = self._environment.step(
            action.action_id,
            data=action.data,
            reasoning=reasoning,
        )
        if raw_observation is None:
            raise RuntimeError(
                f"step failed for ARC game '{self._game_id}' with action"
                f" '{action.action_id.name}'"
            )

        self._step_index += 1
        return self._store_observation(raw_observation)

    def get_action_space(self) -> Sequence[ActionSpec]:
        """Return currently valid ARC actions from the last frame data."""

        raw_observation = self._last_raw_observation
        if raw_observation is None:
            return ()

        return tuple(
            ActionSpec(action_id=GameAction.from_id(action_id))
            for action_id in raw_observation.available_actions
        )

    def get_info(self) -> EnvironmentInfo:
        """Return current ARC environment metadata."""

        raw_observation = self._last_raw_observation
        if raw_observation is None:
            return EnvironmentInfo(game_id=self._game_id)

        return EnvironmentInfo(
            game_id=raw_observation.game_id or self._game_id,
            state=raw_observation.state,
            available_actions=tuple(self.get_action_space()),
            levels_completed=raw_observation.levels_completed,
            win_levels=raw_observation.win_levels,
            full_reset=raw_observation.full_reset,
            metadata={"raw_frame_data": raw_observation},
        )

    def _store_observation(self, raw_observation: FrameDataRaw) -> Observation:
        self._last_raw_observation = raw_observation
        frames = tuple(raw_observation.frame)
        return Observation(
            id=self._build_observation_id(raw_observation),
            step=self._step_index,
            frame=frames[0] if frames else None,
            frames=frames,
            raw_frame_data=raw_observation,
            metadata={"raw_frame_data": raw_observation},
        )

    def _build_observation_id(self, raw_observation: FrameDataRaw) -> str:
        guid = raw_observation.guid or "kaggle"
        return f"{self._game_id}-{guid}-step-{self._step_index}"


def _matching_game_id(left: str, right: str) -> bool:
    """Return whether two full or short ARC ids identify the same game."""

    if left == right:
        return True
    left_is_short = "-" not in left
    right_is_short = "-" not in right
    if not (left_is_short or right_is_short):
        return False
    return _short_game_id(left) == _short_game_id(right)


def _matching_game_ids(
    requested_game_id: str,
    games: Sequence[ArcEnvironmentInfo],
) -> tuple[str, ...]:
    return tuple(
        str(game.game_id)
        for game in games
        if _matching_game_id(str(game.game_id), requested_game_id)
    )


def _single_game_id_match(requested_game_id: str, matches: Sequence[str]) -> str:
    if len(matches) == 1:
        return matches[0]
    matched = ", ".join(matches)
    raise RuntimeError(
        f"game id '{requested_game_id}' matched multiple ARC games: {matched}"
    )


def _short_game_id(game_id: str) -> str:
    return game_id.split("-", 1)[0]


class KaggleArcadeAdapter:
    """ARC gateway adapter for Kaggle scorecard and game wrapper operations."""

    def __init__(self, arcade: Arcade) -> None:
        self._arcade = arcade

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> "KaggleArcadeAdapter":
        """Create an online ARC client pointed at the configured gateway."""

        return cls(
            Arcade(
                operation_mode=config.operation_mode,
                environments_dir=config.environments_dir,
                recordings_dir=config.recordings_dir,
            )
        )

    def open_scorecard(self, *, tags: Sequence[str]) -> str:
        """Open one ARC scorecard for a Kaggle submission run."""

        return str(self._arcade.open_scorecard(tags=list(tags)))

    def close_scorecard(self, scorecard_id: str) -> Any:
        """Close the active ARC scorecard and return the toolkit response."""

        return self._arcade.close_scorecard(scorecard_id)

    def list_available_game_ids(self) -> tuple[str, ...]:
        """Return game ids discoverable from the Kaggle gateway."""

        return tuple(str(game.game_id) for game in self._arcade.get_environments())

    def make_scorecard_environment(
        self,
        game_id: str,
        *,
        scorecard_id: str,
    ) -> EnvironmentWrapper:
        """Create one ARC environment wrapper bound to the scorecard."""

        environment = self._arcade.make(game_id, scorecard_id=scorecard_id)
        if environment is None:
            raise RuntimeError(f"Kaggle gateway could not create game '{game_id}'")
        return environment


__all__ = [
    "ArcEnvironmentAdapter",
    "ArcEnvironmentWrapperAdapter",
    "EnvironmentAdapter",
    "EnvironmentInfo",
    "GameState",
    "KaggleArcadeAdapter",
]
