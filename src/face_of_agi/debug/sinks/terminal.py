"""Rich terminal debug tracing for game runs."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import json
import sys
import textwrap
from typing import Any, TextIO

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionHistoryResetMarker,
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
from face_of_agi.debug.capture.model_io import (
    collect_model_input_payload,
    collect_model_io_payload,
)
from face_of_agi.debug.sanitize import sanitize_for_debug
from face_of_agi.debug.events import (
    AgentFrameworkInputCaptured,
    AgentProviderRequestsCaptured,
    DebugEvent,
    EnvironmentStepRecorded,
    FrameDecisionRecorded,
    FrameTurnCompleted,
    FrameTurnStarted,
    KnownStateSimulationCompleted,
    MStatePersisted,
    ModelCallCompleted,
    RunStarted,
    RunStopped,
    ToolModelInputCaptured,
    ToolProviderInputCaptured,
    ToolResultRecorded,
    UpdaterInputCaptured,
    UpdaterProviderOutputCaptured,
)

DebugTraceMode = str
DebugColorMode = str

_TRACE_LEVELS = {
    "off": 0,
    "minimal": 1,
    "agent_decision": 2,
    "verbose": 3,
    "model_inputs": 4,
}


class DebugTrace:
    """Pretty stdout trace for manual runtime debugging."""

    def __init__(
        self,
        *,
        mode: DebugTraceMode = "minimal",
        color: DebugColorMode = "auto",
        output: TextIO | None = None,
    ) -> None:
        if mode not in _TRACE_LEVELS:
            raise ValueError(f"unknown debug trace mode: {mode}")
        if color not in {"auto", "always", "never"}:
            raise ValueError(f"unknown debug color mode: {color}")

        self.mode = mode
        self.color = color
        self.output = output or sys.stdout
        self.console = Console(
            file=self.output,
            force_terminal=color == "always",
            no_color=color == "never",
            color_system=None if color == "never" else "auto",
            highlight=False,
            soft_wrap=True,
            theme=Theme(
                {
                    "run": "bold cyan",
                    "frame": "bold blue",
                    "agent": "bold magenta",
                    "updater": "bold bright_blue",
                    "memory": "bold bright_black",
                    "warning": "bold red",
                }
            ),
        )

    @classmethod
    def disabled(cls) -> "DebugTrace":
        """Return a trace object that never emits output."""

        return cls(mode="off", color="never")

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        output: TextIO | None = None,
    ) -> "DebugTrace":
        """Build a trace object from environment config fields."""

        return cls(
            mode=str(getattr(config, "debug_trace", "minimal")),
            color=str(getattr(config, "debug_color", "auto")),
            output=output,
        )

    def enabled(self) -> bool:
        """Return whether any trace output is enabled."""

        return _TRACE_LEVELS[self.mode] > 0

    def verbose_enabled(self) -> bool:
        """Return whether verbose run/frame details should be printed."""

        return _TRACE_LEVELS[self.mode] >= _TRACE_LEVELS["verbose"]

    def decision_enabled(self) -> bool:
        """Return whether the Agent X decision panel should be printed."""

        return self.mode == "agent_decision" or self.verbose_enabled()

    def model_inputs_enabled(self) -> bool:
        """Return whether model inputs should be printed."""

        return _TRACE_LEVELS[self.mode] >= _TRACE_LEVELS["model_inputs"]

    def emit(self, event: DebugEvent) -> None:
        """Render one typed debug event to the terminal trace."""

        if isinstance(event, RunStarted):
            self.run_start(
                run_id=event.run_id,
                game_id=event.game_id,
                config=event.config,
            )
        elif isinstance(event, FrameTurnStarted):
            self.frame_turn(
                frame_turn=event.frame_turn,
                frame_context=event.frame_context,
                lifecycle_state=event.lifecycle_state,
                completed_levels=event.completed_levels,
                remaining_actions=event.remaining_actions,
                available_tools=event.available_tools,
            )
        elif isinstance(event, AgentFrameworkInputCaptured):
            self.agent_framework_input(
                context=event.context,
                current_observation=event.current_observation,
                action_space=event.action_space,
                recent_action_history=event.recent_action_history,
                tool_runtime=event.tool_runtime,
            )
        elif isinstance(event, AgentProviderRequestsCaptured):
            self.agent_provider_requests(event.requests)
        elif isinstance(event, FrameDecisionRecorded):
            self.frame_decision(
                frame_turn=event.frame_turn,
                frame_context=event.frame_context,
                action=event.action,
                trace=event.trace,
            )
        elif isinstance(event, ToolModelInputCaptured):
            self.tool_model_input(
                role=event.role,
                purpose=event.purpose,
                call=event.call,
                context=event.context,
                observation=event.observation,
            )
        elif isinstance(event, ToolProviderInputCaptured):
            self.tool_provider_input(
                role=event.role,
                purpose=event.purpose,
                adapter=event.adapter,
            )
        elif isinstance(event, ToolResultRecorded):
            self.tool_result(
                role=event.role,
                purpose=event.purpose,
                result=event.result,
                experiment_ref=event.experiment_ref,
            )
        elif isinstance(event, EnvironmentStepRecorded):
            self.environment_step(
                action=event.action,
                next_observation=event.next_observation,
                remaining_actions=event.remaining_actions,
            )
        elif isinstance(event, KnownStateSimulationCompleted):
            return
        elif isinstance(event, UpdaterInputCaptured):
            self.updater_input(role=event.role, update_input=event.update_input)
        elif isinstance(event, UpdaterProviderOutputCaptured):
            self.updater_provider_output(role=event.role, adapter=event.adapter)
        elif isinstance(event, MStatePersisted):
            self.persisted_state(record_id=event.record_id, turn_id=event.turn_id)
        elif isinstance(event, ModelCallCompleted):
            return
        elif isinstance(event, FrameTurnCompleted):
            return
        elif isinstance(event, RunStopped):
            self.stop(event.result)
        else:
            raise TypeError(f"unsupported debug event: {type(event).__name__}")

    def run_start(self, *, run_id: str, game_id: str, config: Any) -> None:
        """Print high-level run metadata."""

        if not self.verbose_enabled():
            return

        self._json_panel(
            "Run start",
            {
                "run_id": run_id,
                "game_id": game_id,
                "level_action_budget": getattr(
                    config,
                    "max_actions_per_level",
                    None,
                ),
                "game_level_cap": getattr(config, "max_levels_per_game", None),
                "debug_trace": getattr(config, "debug_trace", None),
                "debug_keep_all_m_states": getattr(
                    config,
                    "debug_keep_all_m_states",
                    None,
                ),
                "use_learned_contexts": getattr(
                    config,
                    "use_learned_contexts",
                    None,
                ),
                "operation_mode": getattr(config, "operation_mode", None),
                "render_mode": getattr(config, "render_mode", None),
            },
            style="run",
        )

    def frame_turn(
        self,
        *,
        frame_turn: int,
        frame_context: FrameTurnContext,
        lifecycle_state: Any,
        completed_levels: int,
        remaining_actions: int,
        available_tools: Sequence[ToolName],
    ) -> None:
        """Print per-frame loop state before Agent X runs."""

        if not self.verbose_enabled():
            return

        table = Table(box=box.ASCII, show_header=False, pad_edge=False)
        table.add_column("key", style="frame")
        table.add_column("value", overflow="fold", no_wrap=False)
        table.add_row("frame_turn", str(frame_turn))
        table.add_row("env_step", str(frame_context.current_observation.step))
        table.add_row(
            "frame",
            f"{frame_context.frame_index + 1}/{frame_context.frame_count}",
        )
        table.add_row("lifecycle_state", _display_scalar(lifecycle_state))
        table.add_row("completed_levels", str(completed_levels))
        table.add_row("remaining_actions", str(remaining_actions))
        table.add_row(
            "control",
            (
                f"controllable={frame_context.control_mode.controllable} "
                f"reason={frame_context.control_mode.reason}"
            ),
        )
        table.add_row(
            "allowed_actions",
            ", ".join(action.name for action in frame_context.control_mode.allowed_actions),
        )
        table.add_row("available_tools", ", ".join(available_tools) or "(none)")
        self.console.print(
            Panel(table, title="Frame turn", border_style="frame", box=box.ASCII)
        )

    def agent_framework_input(
        self,
        *,
        context: RoleContext,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        recent_action_history: Sequence[ActionHistoryItem],
        tool_runtime: Any | None,
    ) -> None:
        """Print the provider-neutral input handed to Agent X."""

        if not self.model_inputs_enabled():
            return

        payload: dict[str, Any] = {
            "role": "agent_x",
            "context": _role_context_payload(context),
            "current_observation": _observation_payload(current_observation),
            "action_space": [_action_payload(action) for action in action_space],
            "recent_action_history": [
                _action_history_payload(item) for item in recent_action_history
            ],
        }
        if tool_runtime is not None:
            payload["turn_id"] = getattr(tool_runtime, "turn_id", None)
        self._json_panel("Agent X framework input", payload, style="agent")

    def agent_provider_requests(self, requests: Sequence[Any]) -> None:
        """Print provider-specific requests sent by Agent X backends."""

        if not self.model_inputs_enabled():
            return

        for index, request in enumerate(requests, start=1):
            self._json_panel(
                f"Agent X provider request {index}",
                request,
                style="agent",
            )

    def frame_decision(
        self,
        *,
        frame_turn: int,
        frame_context: FrameTurnContext,
        action: ActionSpec,
        trace: AgentTrace,
    ) -> None:
        """Print the final decision record for one frame turn."""

        if not self.enabled():
            return

        if not self.decision_enabled():
            self._print_minimal_frame_trace(
                frame_turn=frame_turn,
                frame_context=frame_context,
                action=action,
                trace=trace,
            )
            return

        title = (
            "Orchestration synthetic decision"
            if trace.metadata.get("agent_x_called") is False
            else "Agent X decision"
        )
        self._json_panel(
            title,
            {
                "frame_turn": frame_turn,
                "env_step": frame_context.current_observation.step,
                "final_action": _action_payload(action),
                "reasoning_summary": trace.reasoning_summary,
                "tool_calls": trace.tool_calls,
                "tool_results": [_tool_result_summary(result) for result in trace.tool_results],
                "metadata": trace.metadata,
            },
            style="agent",
        )

    def tool_model_input(
        self,
        *,
        role: ToolName,
        purpose: str,
        call: ToolCall,
        context: RoleContext,
        observation: Observation,
    ) -> None:
        """Print the framework-level prediction input."""

        if not self.model_inputs_enabled():
            return

        self._json_panel(
            f"{role} model input",
            {
                "purpose": purpose,
                "tool_call": call,
                "context": _role_context_payload(context),
                "source_observation": _observation_payload(observation),
            },
            style=role,
        )

    def tool_provider_input(
        self,
        *,
        role: ToolName,
        purpose: str,
        adapter: Any | None,
    ) -> None:
        """Print captured provider prompt/request data for predictions."""

        if not self.model_inputs_enabled() or adapter is None:
            return

        payload = collect_model_input_payload(adapter)
        if not payload:
            return
        self._json_panel(
            f"{role} provider input",
            {"purpose": purpose, **payload},
            style=role,
        )

    def tool_result(
        self,
        *,
        role: ToolName,
        purpose: str,
        result: ToolResult,
        experiment_ref: ObservationRef | None = None,
    ) -> None:
        """Print a routed prediction result summary."""

        if not self.verbose_enabled():
            return

        self._json_panel(
            f"{role} tool result",
            {
                "purpose": purpose,
                "experiment_ref": experiment_ref,
                "result": _tool_result_summary(result),
            },
            style=role,
        )

    def environment_step(
        self,
        *,
        action: ActionSpec,
        next_observation: Observation,
        remaining_actions: int,
    ) -> None:
        """Print the real environment step result."""

        if not self.verbose_enabled():
            return

        self._json_panel(
            "Environment step",
            {
                "submitted_action": _action_payload(action),
                "next_observation": _observation_payload(next_observation),
                "remaining_actions": remaining_actions,
            },
            style="run",
        )

    def updater_input(self, *, role: str, update_input: Any) -> None:
        """Print the input passed to updater P."""

        if not self.model_inputs_enabled():
            return

        self._json_panel(
            f"Updater P {role} input",
            update_input,
            style="updater",
        )

    def updater_provider_output(self, *, role: str, adapter: Any | None) -> None:
        """Print captured provider request/response data for updater P."""

        if not self.model_inputs_enabled() or adapter is None:
            return

        payload = collect_model_io_payload(adapter)
        if not payload:
            return
        self._json_panel(
            f"Updater P {role} provider output",
            payload,
            style="updater",
        )

    def persisted_state(self, *, record_id: int, turn_id: int) -> None:
        """Print the M-state row written for a frame turn."""

        if not self.verbose_enabled():
            return

        self._json_panel(
            "Persisted M state",
            {"record_id": record_id, "turn_id": turn_id},
            style="memory",
        )

    def stop(self, result: GameRunResult) -> None:
        """Print the run stop reason."""

        if not self.verbose_enabled():
            return

        self._json_panel(
            "Run stop",
            {
                "run_id": result.run_id,
                "game_id": result.game_id,
                "stop_reason": result.stop_reason,
                "step_count": result.step_count,
                "completed_levels": result.completed_levels,
                "last_state": result.last_state,
                "state_record_ids": result.state_record_ids,
            },
            style="run",
        )

    def _print_minimal_frame_trace(
        self,
        *,
        frame_turn: int,
        frame_context: FrameTurnContext,
        action: ActionSpec,
        trace: AgentTrace,
    ) -> None:
        controllable = "yes" if frame_context.control_mode.controllable else "no"
        self.console.print(
            "frame turn"
            f" {frame_turn}: env_step={frame_context.current_observation.step}"
            f" frame={frame_context.frame_index + 1}/{frame_context.frame_count}"
            f" controllable={controllable}",
        )
        if action.is_none() and trace.metadata.get("agent_x_called") is False:
            self.console.print(
                "action: orchestration synthesized NONE; environment not stepped"
            )
        elif action.is_none():
            self.console.print("action: X returned NONE; environment not stepped")
        else:
            self.console.print(f"action: X selected {_format_action(action)}")

    def _json_panel(self, title: str, payload: Any, *, style: str) -> None:
        text = json.dumps(sanitize_for_debug(payload), indent=2, sort_keys=True)
        text = _wrap_debug_text(text, width=_wrap_width(self.console.width))
        self.console.print(
            Panel(
                Text(text, overflow="fold", no_wrap=False),
                title=title,
                border_style=style,
                box=box.ASCII,
            )
        )


def _wrap_debug_text(text: str, *, width: int) -> str:
    """Fold long rendered lines before Rich can crop them in narrow terminals."""

    wrapped_lines: list[str] = []
    for line in text.splitlines():
        if len(line) <= width:
            wrapped_lines.append(line)
            continue

        indent = line[: len(line) - len(line.lstrip(" "))]
        wrapper = textwrap.TextWrapper(
            width=width,
            subsequent_indent=f"{indent}  ",
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped = wrapper.wrap(line)
        wrapped_lines.extend(wrapped or [line])
    return "\n".join(wrapped_lines)


def _wrap_width(console_width: int) -> int:
    """Return a conservative content width for panel body text."""

    return max(40, min(120, console_width - 8))


def _role_context_payload(context: RoleContext) -> dict[str, str]:
    return {
        "general": context.general,
        "game": context.game,
        "composed": context.composed(),
    }


def _observation_payload(observation: Observation) -> dict[str, Any]:
    return {
        "id": observation.id,
        "step": observation.step,
        "frame_count": observation.frame_count(),
        "metadata": observation.metadata,
    }


def _action_payload(action: ActionSpec | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "action_id": action.name,
        "data": action.data,
        "requires_data": action.is_complex(),
    }


def _action_history_payload(item: ActionHistoryItem) -> dict[str, Any]:
    if isinstance(item, ActionHistoryResetMarker):
        return {
            "type": "game_reset",
            "reason": item.reason,
            "restart_count": item.restart_count,
        }
    return {
        "action": _action_payload(item.action),
        "controllable": item.controllable,
        "changed_pixel_count": item.changed_pixel_count,
        "completed_levels": item.completed_levels,
        "action_count": item.action_count,
        "animation_frame_count": item.animation_frame_count,
        "avg_changed_pixel_count": item.avg_changed_pixel_count,
        "change_summary": item.change_summary,
        "change_elements": item.change_elements,
    }


def _tool_result_summary(result: ToolResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "id": result.id,
        "tool": result.tool,
        "source_observation_ref": result.source_observation_ref,
        "source_state_id": result.source_state_id,
        "action": result.action,
        "explanation": result.explanation,
        "metadata": result.metadata,
        "output": result.output,
    }


def _format_action(action: ActionSpec) -> str:
    if action.data:
        return f"{action.name} {action.data}"
    return action.name


def _display_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, Enum):
        return value.name
    return str(value)
