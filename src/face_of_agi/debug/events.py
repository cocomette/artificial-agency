"""Typed debug events emitted by runtime orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    FrameTurnContext,
    GameRunResult,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolName,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    game_id: str
    config: Any


@dataclass(frozen=True, slots=True)
class FrameTurnStarted:
    frame_turn: int
    frame_context: FrameTurnContext
    lifecycle_state: Any
    completed_levels: int
    remaining_actions: int
    available_tools: Sequence[ToolName]


@dataclass(frozen=True, slots=True)
class AgentFrameworkInputCaptured:
    context: RoleContext
    current_observation: Observation
    action_space: Sequence[ActionSpec]
    recent_action_history: Sequence[Any]
    tool_runtime: Any | None


@dataclass(frozen=True, slots=True)
class AgentProviderRequestsCaptured:
    requests: Sequence[Any]


@dataclass(frozen=True, slots=True)
class FrameDecisionRecorded:
    frame_turn: int
    frame_context: FrameTurnContext
    action: ActionSpec
    trace: AgentTrace


@dataclass(frozen=True, slots=True)
class ToolModelInputCaptured:
    role: ToolName
    purpose: str
    call: ToolCall
    context: RoleContext
    observation: Observation


@dataclass(frozen=True, slots=True)
class ToolProviderInputCaptured:
    role: ToolName
    purpose: str
    adapter: Any | None


@dataclass(frozen=True, slots=True)
class ToolResultRecorded:
    role: ToolName
    purpose: str
    result: ToolResult
    experiment_ref: ObservationRef | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentStepRecorded:
    action: ActionSpec
    next_observation: Observation
    remaining_actions: int


@dataclass(frozen=True, slots=True)
class UpdaterInputCaptured:
    role: str
    update_input: Any


@dataclass(frozen=True, slots=True)
class UpdaterProviderOutputCaptured:
    role: str
    adapter: Any | None


@dataclass(frozen=True, slots=True)
class ModelCallCompleted:
    role: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class MStatePersisted:
    record_id: int
    turn_id: int


@dataclass(frozen=True, slots=True)
class FrameTurnCompleted:
    run_id: str
    game_id: str
    game_index: int | None
    turn_id: int
    env_step: int | None
    frame_index: int
    frame_count: int
    controllable: bool
    action: ActionSpec
    turn_duration_seconds: float
    completed_levels: int
    remaining_actions: int


@dataclass(frozen=True, slots=True)
class RunStopped:
    result: GameRunResult


DebugEvent: TypeAlias = (
    RunStarted
    | FrameTurnStarted
    | AgentFrameworkInputCaptured
    | AgentProviderRequestsCaptured
    | FrameDecisionRecorded
    | ToolModelInputCaptured
    | ToolProviderInputCaptured
    | ToolResultRecorded
    | EnvironmentStepRecorded
    | UpdaterInputCaptured
    | UpdaterProviderOutputCaptured
    | ModelCallCompleted
    | MStatePersisted
    | FrameTurnCompleted
    | RunStopped
)
