"""Provider-neutral orchestration agent loop."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.models.orchestrator_agent.config import OrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime
from face_of_agi.models.orchestrator_agent.tooling import (
    AgentOutputError,
    build_decision_result,
    final_action_schema,
    parse_arguments,
    parse_action,
    parse_final_action,
)
from face_of_agi.models.providers.vision import resolve_model_vision_profile


@dataclass(slots=True)
class AgentTurnRequest:
    """Provider-neutral input for one X provider conversation."""

    context: RoleContext
    first_observation: Observation
    current_observation: Observation
    action_space: Sequence[ActionSpec]
    recent_action_history: tuple[ActionHistoryEntry, ...] = ()
    world_game_context: str = ""
    goal_game_context: str = ""


@dataclass(slots=True)
class ProviderFunctionCall:
    """Provider-neutral function call emitted by an X backend."""

    name: str
    arguments: Any
    call_id: str | None = None


@dataclass(slots=True)
class AgentToolSpec:
    """Provider-neutral tool definition available to Agent X."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class AgentProviderStep:
    """Provider-neutral response from one X backend model step."""

    tool_calls: tuple[ProviderFunctionCall, ...] = ()
    final_output: Any | None = None
    response_id: str | None = None
    usage: Any | None = None


@dataclass(slots=True)
class ProviderToolFeedback:
    """Provider-neutral feedback for a tool result returned to X."""

    call_id: str | None
    invocation: ExperimentToolInvocationResult


class AgentProviderSession(Protocol):
    """Thin provider boundary used by the shared Agent X loop."""

    backend: str
    model: str | None

    def begin(self, request: AgentTurnRequest) -> None:
        """Start one provider conversation from a framework turn request."""
        ...

    def step(
        self,
        action_space: Sequence[ActionSpec],
        tool_specs: Sequence[AgentToolSpec],
    ) -> AgentProviderStep:
        """Run one provider model step and normalize tools/final output."""
        ...

    def append_tool_feedback(self, feedback: ProviderToolFeedback) -> None:
        """Append orchestration-owned tool output to provider conversation state."""
        ...

    def append_repair(
        self,
        *,
        validation_error: str,
        action_space: Sequence[ActionSpec],
        invalid_text: str | None,
        attempt: int,
    ) -> None:
        """Append provider-specific structured-output repair context."""
        ...


class OrchestratorAgentAdapter:
    """Shared Agent X reasoning/tool loop over a thin provider session."""

    def __init__(
        self,
        provider: AgentProviderSession,
        config: OrchestratorAgentConfig | None = None,
    ) -> None:
        self.config = config or OrchestratorAgentConfig()
        self.provider = provider
        self.last_provider_requests: list[Any] = []
        self.vision_profile = resolve_model_vision_profile(
            backend=self.provider.backend,
            model=self.provider.model,
        )
        self.coordinate_space = self.vision_profile.coordinate_space

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
        world_game_context: str = "",
        goal_game_context: str = "",
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        """Run the provider-neutral X loop until one final action is submitted."""

        if not action_space:
            raise RuntimeError("orchestrator agent received no valid actions")

        self.last_provider_requests = []
        request = AgentTurnRequest(
            context=context,
            world_game_context=world_game_context,
            goal_game_context=goal_game_context,
            first_observation=first_observation,
            current_observation=current_observation,
            action_space=action_space,
            recent_action_history=recent_action_history,
        )
        self.provider.begin(request)

        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        repair_count = 0
        tool_call_count = 0
        response_ids: list[str] = []
        usage: list[Any] = []
        tool_specs = self._agent_tool_specs(tool_runtime)
        return self._run_steps(
            action_space=action_space,
            first_observation=first_observation,
            current_observation=current_observation,
            tool_runtime=tool_runtime,
            tool_specs=tool_specs,
            tool_calls=tool_calls,
            tool_results=tool_results,
            repair_count=repair_count,
            tool_call_count=tool_call_count,
            response_ids=response_ids,
            usage=usage,
        )

    def _run_steps(
        self,
        *,
        action_space: Sequence[ActionSpec],
        first_observation: Observation,
        current_observation: Observation,
        tool_runtime: AgentToolRuntime | None,
        tool_specs: tuple[AgentToolSpec, ...],
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        repair_count: int,
        tool_call_count: int,
        response_ids: list[str],
        usage: list[Any],
    ) -> DecisionResult:
        current_repair_count = repair_count
        while True:
            invalid_text: str | None = None
            try:
                response = self.provider.step(action_space, tool_specs)
                self._capture_provider_request()
                if response.response_id is not None:
                    response_ids.append(response.response_id)
                if response.usage is not None:
                    usage.append(response.usage)

                if response.tool_calls:
                    tool_call_count = self._invoke_tool_calls(
                        provider_calls=response.tool_calls,
                        action_space=action_space,
                        tool_runtime=tool_runtime,
                        tool_specs=tool_specs,
                        tool_call_count=tool_call_count,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                    )
                    continue

                if response.final_output is None:
                    raise AgentOutputError(
                        "model response contained neither tool calls nor final output"
                    )

                invalid_text = str(response.final_output)
                try:
                    final_action = parse_final_action(
                        response.final_output,
                        action_space,
                        coordinate_space=self.coordinate_space,
                    )
                except Exception as exc:
                    raise AgentOutputError(
                        f"invalid final structured action: {exc}"
                    ) from exc
                return self._decision_result(
                    final_action=final_action,
                    first_observation=first_observation,
                    current_observation=current_observation,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    repair_count=current_repair_count,
                    response_ids=response_ids,
                    usage=usage,
                )
            except Exception as exc:
                if current_repair_count >= self.config.repair_attempts:
                    raise RuntimeError(
                        f"{self.provider.backend} X produced invalid structured "
                        f"agent step after {current_repair_count} repair "
                        f"attempt(s): {exc}"
                    ) from exc
                current_repair_count += 1
                self.provider.append_repair(
                    validation_error=str(exc),
                    action_space=action_space,
                    invalid_text=invalid_text,
                    attempt=current_repair_count,
                )

    def _agent_tool_specs(
        self,
        tool_runtime: AgentToolRuntime | None,
    ) -> tuple[AgentToolSpec, ...]:
        """Return generic Agent X tool specs exposed for this run."""

        if tool_runtime is None or self.config.max_tool_calls <= 0:
            return ()
        available_tool_specs = getattr(tool_runtime, "available_tool_specs", None)
        if not callable(available_tool_specs):
            return ()
        return tuple(available_tool_specs())

    def _invoke_tool_calls(
        self,
        *,
        provider_calls: Sequence[ProviderFunctionCall],
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None,
        tool_specs: Sequence[AgentToolSpec],
        tool_call_count: int,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ) -> int:
        if tool_runtime is None:
            raise AgentOutputError("model requested tools but no tool runtime exists")
        if not tool_specs:
            raise AgentOutputError("model requested tools but no tools are available")
        if tool_call_count + len(provider_calls) > self.config.max_tool_calls:
            raise AgentOutputError(
                "model exceeded Agent X tool-call budget "
                f"({self.config.max_tool_calls})"
            )

        available_names = {spec.name for spec in tool_specs}
        for provider_call in provider_calls:
            if provider_call.name not in available_names:
                raise AgentOutputError(
                    f"model requested unavailable tool {provider_call.name!r}"
                )
            call = self._tool_call_from_provider_call(
                provider_call,
                action_space,
                tool_runtime=tool_runtime,
            )
            invocation = tool_runtime.invoke(
                call,
                metadata={"provider_call_id": provider_call.call_id},
            )
            self.provider.append_tool_feedback(
                ProviderToolFeedback(
                    call_id=provider_call.call_id,
                    invocation=invocation,
                )
            )
            tool_calls.append(call)
            tool_results.append(invocation.tool_result)
            tool_call_count += 1
        return tool_call_count

    def _tool_call_from_provider_call(
        self,
        provider_call: ProviderFunctionCall,
        action_space: Sequence[ActionSpec],
        *,
        tool_runtime: AgentToolRuntime,
    ) -> ToolCall:
        args = parse_arguments(provider_call.arguments)
        source_state_id = args.get("source_state_id")
        if source_state_id is None:
            source_state_id = tool_runtime.current_source_state_id
        if source_state_id is None:
            raise AgentOutputError("tool source_state_id is required")
        if isinstance(source_state_id, bool) or not isinstance(source_state_id, int):
            raise AgentOutputError("tool source_state_id must be an integer")

        action = None
        if args.get("action") is not None:
            action = parse_action(
                args.get("action"),
                action_space,
                coordinate_space=self.coordinate_space,
            )
        return ToolCall(
            tool=provider_call.name,  # type: ignore[arg-type]
            source_state_id=source_state_id,
            action=action,
        )

    def _decision_result(
        self,
        *,
        final_action: ActionSpec,
        first_observation: Observation,
        current_observation: Observation,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        repair_count: int,
        response_ids: list[str],
        usage: list[Any],
    ) -> DecisionResult:
        return build_decision_result(
            final_action=final_action,
            first_observation=first_observation,
            current_observation=current_observation,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata={
                "backend": self.provider.backend,
                "model": self.provider.model,
                "tool_call_count": len(tool_calls),
                "repair_count": repair_count,
                "provider_response_ids": response_ids,
                "usage": usage,
            },
        )

    def _capture_provider_request(self) -> None:
        """Capture the last provider request for ephemeral debug output."""

        request = getattr(self.provider, "last_request", None)
        if request is None:
            return
        try:
            self.last_provider_requests.append(deepcopy(request))
        except Exception:
            self.last_provider_requests.append(request)
