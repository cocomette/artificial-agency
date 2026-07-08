"""Provider-neutral orchestration agent loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from face_of_agi.contracts import (
    ActionSpec,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    RoleContext,
    ToolCall,
    ToolName,
    ToolResult,
)
from face_of_agi.models.orchestrator_agent.config import OrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime
from face_of_agi.models.orchestrator_agent.tooling import (
    AgentOutputError,
    build_decision_result,
    build_tool_call,
    parse_submit_action,
)


@dataclass(slots=True)
class AgentTurnRequest:
    """Provider-neutral input for one X provider conversation."""

    context: RoleContext
    first_observation: Observation
    current_observation: Observation
    action_space: Sequence[ActionSpec]
    tool_runtime: AgentToolRuntime | None
    available_tools: tuple[ToolName, ...]


@dataclass(slots=True)
class ProviderFunctionCall:
    """Provider-neutral function call emitted by an X backend."""

    name: str
    arguments: Any
    call_id: str | None = None


@dataclass(slots=True)
class AgentProviderResponse:
    """Provider-neutral response from one X backend model call."""

    function_calls: tuple[ProviderFunctionCall, ...]
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

    def call(self) -> AgentProviderResponse:
        """Run one provider model call and normalize its function calls."""
        ...

    def append_tool_feedback(self, feedback: ProviderToolFeedback) -> None:
        """Append orchestration-owned tool output to provider conversation state."""
        ...

    def append_repair(self, error: str, action_space: Sequence[ActionSpec]) -> None:
        """Append a provider-specific repair instruction."""
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

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
    ) -> DecisionResult:
        """Run the provider-neutral X loop until one final action is submitted."""

        if not action_space:
            raise RuntimeError("orchestrator agent received no valid actions")

        available_tools = (
            tool_runtime.available_tools() if tool_runtime is not None else ()
        )
        self.provider.begin(
            AgentTurnRequest(
                context=context,
                first_observation=first_observation,
                current_observation=current_observation,
                action_space=action_space,
                tool_runtime=tool_runtime,
                available_tools=available_tools,
            )
        )

        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        repair_count = 0
        response_ids: list[str] = []
        usage: list[Any] = []

        max_responses = self.config.max_tool_calls + self.config.repair_attempts + 3
        for _ in range(max_responses):
            response = self.provider.call()
            if response.response_id is not None:
                response_ids.append(response.response_id)
            if response.usage is not None:
                usage.append(response.usage)

            try:
                completed = self._handle_provider_response(
                    response=response,
                    action_space=action_space,
                    tool_runtime=tool_runtime,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    first_observation=first_observation,
                    current_observation=current_observation,
                    repair_count=repair_count,
                    response_ids=response_ids,
                    usage=usage,
                )
            except Exception as exc:
                if repair_count >= self.config.repair_attempts:
                    raise RuntimeError(
                        f"{self.provider.backend} X produced an invalid response "
                        f"after {repair_count} repair attempt(s): {exc}"
                    ) from exc
                repair_count += 1
                self.provider.append_repair(str(exc), action_space)
                continue

            if completed is not None:
                return completed
            if response.function_calls:
                continue

            if repair_count >= self.config.repair_attempts:
                raise RuntimeError(
                    f"{self.provider.backend} X did not call a tool or submit_action"
                )
            repair_count += 1
            self.provider.append_repair(
                "response did not include a function tool call",
                action_space,
            )

        raise RuntimeError(f"{self.provider.backend} X exceeded its response budget")

    def _handle_provider_response(
        self,
        *,
        response: AgentProviderResponse,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        first_observation: Observation,
        current_observation: Observation,
        repair_count: int,
        response_ids: list[str],
        usage: list[Any],
    ) -> DecisionResult | None:
        """Execute normalized provider function calls."""

        for provider_call in response.function_calls:
            if provider_call.name == "submit_action":
                final_action, reasoning_summary = parse_submit_action(
                    provider_call.arguments,
                    action_space,
                )
                return build_decision_result(
                    final_action=final_action,
                    reasoning_summary=reasoning_summary,
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

            if len(tool_calls) >= self.config.max_tool_calls:
                raise AgentOutputError("tool-call budget exhausted")
            if tool_runtime is None:
                raise AgentOutputError("tool runtime is not available")

            call = build_tool_call(
                name=provider_call.name,
                arguments=provider_call.arguments,
                action_space=action_space,
            )
            invocation = tool_runtime.invoke(call)
            tool_calls.append(call)
            tool_results.append(invocation.tool_result)
            self.provider.append_tool_feedback(
                ProviderToolFeedback(
                    call_id=provider_call.call_id,
                    invocation=invocation,
                )
            )

        return None
