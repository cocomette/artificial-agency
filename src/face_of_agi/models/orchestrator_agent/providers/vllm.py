"""vLLM provider adapter for orchestrator agent X."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import Any

from face_of_agi.contracts import (
    AgentCandidateAction,
    ActionHistoryItem,
    ActionSpec,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    Observation,
    ObservationRef,
    WorldPrediction,
)
from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.orchestrator_agent.adapter import (
    AgentProviderStep,
    AgentToolSpec,
    AgentTurnRequest,
    OrchestratorAgentAdapter,
    ProviderToolFeedback,
)
from face_of_agi.models.orchestrator_agent.config import VLLMOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.tooling import (
    AgentOutputError,
    build_agent_instructions,
    build_candidate_prompt,
    build_decision_result,
    build_selection_prompt,
    candidate_actions_schema,
    build_decision_prompt,
    final_action_repair_prompt,
    final_action_schema,
    object_get,
    observation_images,
    parse_action,
    parse_arguments,
    parse_final_action,
)
from face_of_agi.models.providers.openai import plain
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message,
    chat_message_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.structured_output import append_output_schema_to_instructions
from face_of_agi.models.vllm_roles import parse_json_object


class VLLMOrchestratorAgentAdapter(OrchestratorAgentAdapter):
    """Agent X adapter backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMOrchestratorAgentConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM Agent X requires an explicit model")
        if config.max_tool_calls > 0:
            raise ValueError("vLLM Agent X does not support tool calls")
        self.provider = VLLMOrchestratorAgentProvider(config, client=client)
        super().__init__(provider=self.provider, config=config)

    def activate_lora_adapter(self, adapter_name: str) -> None:
        """Use a successfully loaded vLLM LoRA adapter for future Agent calls."""

        self.provider.activate_lora_adapter(adapter_name)

    def propose_candidate_actions(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        max_candidates: int,
        recent_action_history: tuple[ActionHistoryItem, ...] = (),
        glossary_actions: Sequence[ActionSpec],
    ) -> tuple[AgentCandidateAction, ...]:
        """Return distinct Agent-proposed coordinate candidates."""

        if max_candidates <= 0:
            return ()
        actions = tuple(action_space)
        if not any(action.is_complex() for action in actions):
            return ()
        response = self.provider.propose_candidates(
            memory=memory,
            goal=goal,
            current_observation=current_observation,
            action_space=actions,
            max_candidates=max_candidates,
            recent_action_history=recent_action_history,
            glossary_actions=glossary_actions,
            crop_edges=self.config.input_image_crop_arc_grid_edges,
            coordinate_space=self.coordinate_space,
        )
        parsed = parse_arguments(response)
        candidate_payload = _candidate_payload(parsed)
        raw_candidates = _candidate_action_values(candidate_payload)
        candidates: list[AgentCandidateAction] = []
        seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        for raw in raw_candidates:
            action = parse_action(
                raw,
                actions,
                coordinate_space=self.coordinate_space,
                crop_edges=self.config.input_image_crop_arc_grid_edges,
            )
            if not action.is_complex():
                continue
            identity = (
                action.name,
                tuple(sorted((action.data or {}).items())),
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                AgentCandidateAction(
                    action=action,
                    source="agent_coordinate_proposal",
                    rank=len(candidates),
                    rationale=str(candidate_payload.get("notes") or ""),
                )
            )
            if len(candidates) >= max_candidates:
                break
        return tuple(candidates)

    def select_action(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        candidates: Sequence[AgentCandidateAction],
        world_predictions: Sequence[WorldPrediction],
        interest_prediction: InterestPrediction | None = None,
        first_observation_ref: ObservationRef | None = None,
        recent_action_history: tuple[ActionHistoryItem, ...] = (),
        glossary_actions: Sequence[ActionSpec],
    ):
        """Select the final action from evaluated candidates."""

        response = self.provider.select_candidate(
            memory=memory,
            goal=goal,
            current_observation=current_observation,
            candidates=tuple(candidates),
            world_predictions=tuple(world_predictions),
            interest_prediction=interest_prediction,
            recent_action_history=recent_action_history,
            glossary_actions=glossary_actions,
            crop_edges=self.config.input_image_crop_arc_grid_edges,
            coordinate_space=self.coordinate_space,
        )
        final_action = _parse_final_candidate_action(
            response,
            candidates,
            coordinate_space=self.coordinate_space,
            crop_edges=self.config.input_image_crop_arc_grid_edges,
        )
        return build_decision_result(
            final_action=final_action,
            current_observation=current_observation,
            first_observation_ref=first_observation_ref,
            tool_calls=[],
            tool_results=[],
            metadata={
                "agent_v1": True,
                "candidate_count": len(candidates),
                "world_prediction_count": len(world_predictions),
                "interest_value_count": (
                    len(interest_prediction.candidate_values)
                    if interest_prediction is not None
                    else 0
                ),
                "response_id": self.provider.last_response_id,
                "usage": self.provider.last_usage,
                "training_request": self.provider.last_request,
                "training_phase": "select_action",
                "training_schema_name": "agent_final_action",
            },
        )


class VLLMOrchestratorAgentProvider:
    """Thin vLLM translation layer for the shared Agent X loop."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMOrchestratorAgentConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.instructions = ""
        self.messages: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None
        self.last_response_id: str | None = None
        self.last_usage: Any | None = None
        self.active_lora_adapter_name: str | None = None

    def activate_lora_adapter(self, adapter_name: str) -> None:
        """Use a runtime-loaded vLLM LoRA adapter for future provider calls."""

        self.active_lora_adapter_name = adapter_name

    def begin(self, request: AgentTurnRequest) -> None:
        """Build the initial vLLM chat messages for one X turn."""

        self.instructions = build_agent_instructions(
            glossary_actions=request.glossary_actions
        )
        schema = final_action_schema(request.action_space)
        self.instructions = append_output_schema_to_instructions(
            self.instructions,
            schema,
            include=self.config.include_output_schema_in_instructions,
        )
        prompt = build_decision_prompt(
            context=request.context,
            action_space=request.action_space,
            recent_action_history=request.recent_action_history,
            recent_action_history_available=(
                request.recent_action_history_available
            ),
            action_outcome_evidence=request.action_outcome_evidence,
            crop_edges=self.config.input_image_crop_arc_grid_edges,
        )
        images = observation_images(
            current_observation=request.current_observation,
            frame_scale=self.config.frame_scale,
        )
        self.messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *vllm_image_content(
                        images,
                        detail=self.config.input_image_detail,
                        size=self.config.input_image_size,
                        resample=self.config.input_image_resample,
                        mime_type=self.config.image_mime_type,
                        crop_edges=self.config.input_image_crop_arc_grid_edges,
                    ),
                ],
            },
        ]

    def step(
        self,
        action_space: Sequence[ActionSpec],
        tool_specs: Sequence[AgentToolSpec],
    ) -> AgentProviderStep:
        """Call vLLM once and normalize final structured output."""

        del tool_specs
        schema = final_action_schema(action_space)
        response = self._chat(schema)
        message = chat_message(response)
        self.messages.append(self._assistant_message(message))
        return AgentProviderStep(
            final_output=chat_message_content(response),
            response_id=(
                str(object_get(response, "id"))
                if object_get(response, "id") is not None
                else None
            ),
            usage=plain(object_get(response, "usage")),
        )

    def append_tool_feedback(self, feedback: ProviderToolFeedback) -> None:
        """Reject tool feedback because vLLM Agent X exposes no tools."""

        del feedback
        raise RuntimeError("vLLM Agent X does not support tool calls")

    def append_repair(
        self,
        *,
        validation_error: str,
        action_space: Sequence[ActionSpec],
        invalid_text: str | None,
        attempt: int,
    ) -> None:
        """Append one structured-output repair instruction."""

        self.messages.append(
            {
                "role": "user",
                "content": final_action_repair_prompt(
                    action_space,
                    validation_error=validation_error,
                    invalid_text=invalid_text,
                    attempt=attempt,
                ),
            }
        )

    def propose_candidates(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        max_candidates: int,
        recent_action_history: Sequence[ActionHistoryItem],
        glossary_actions: Sequence[ActionSpec],
        crop_edges: Any | None,
        coordinate_space: Any,
    ) -> Any:
        """Call vLLM for candidate coordinate proposals."""

        schema = candidate_actions_schema(action_space)
        instructions = append_output_schema_to_instructions(
            build_agent_instructions(glossary_actions=glossary_actions),
            schema,
            include=self.config.include_output_schema_in_instructions,
        )
        prompt = build_candidate_prompt(
            memory=memory,
            goal=goal,
            action_space=action_space,
            max_candidates=max_candidates,
            recent_action_history=recent_action_history,
            crop_edges=crop_edges,
        )
        return self._chat_once(
            instructions=instructions,
            prompt=prompt,
            current_observation=current_observation,
            schema=schema,
            schema_name="agent_candidate_actions",
            phase="candidate_actions",
            validate=lambda parsed: _validate_candidate_actions_payload(
                parsed,
                action_space=action_space,
                coordinate_space=coordinate_space,
                crop_edges=crop_edges,
            ),
        )

    def select_candidate(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        candidates: Sequence[AgentCandidateAction],
        world_predictions: Sequence[WorldPrediction],
        interest_prediction: InterestPrediction | None,
        recent_action_history: Sequence[ActionHistoryItem],
        glossary_actions: Sequence[ActionSpec],
        crop_edges: Any | None,
        coordinate_space: Any,
    ) -> Any:
        """Call vLLM for final action selection."""

        schema = final_action_schema([candidate.action for candidate in candidates])
        instructions = append_output_schema_to_instructions(
            build_agent_instructions(glossary_actions=glossary_actions),
            schema,
            include=self.config.include_output_schema_in_instructions,
        )
        prompt = build_selection_prompt(
            memory=memory,
            goal=goal,
            candidates=candidates,
            world_predictions=world_predictions,
            interest_prediction=interest_prediction,
            recent_action_history=recent_action_history,
            crop_edges=crop_edges,
        )
        return self._chat_once(
            instructions=instructions,
            prompt=prompt,
            current_observation=current_observation,
            schema=schema,
            schema_name="agent_final_action",
            phase="select_action",
            validate=lambda parsed: _parse_final_candidate_action(
                parsed,
                candidates,
                coordinate_space=coordinate_space,
                crop_edges=crop_edges,
            ),
        )

    def _chat(self, schema: dict[str, Any]) -> Any:
        response = self._client.chat(
            model=self._request_model(),
            messages=list(self.messages),
            response_format=json_schema_response_format(
                name="agent_final_action",
                schema=schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(phase="final_action", response=response)
        return response

    def _chat_once(
        self,
        *,
        instructions: str,
        prompt: str,
        current_observation: Observation,
        schema: dict[str, Any],
        schema_name: str,
        phase: str,
        validate: Callable[[dict[str, Any]], object] | None = None,
    ) -> Any:
        images = observation_images(
            current_observation=current_observation,
            frame_scale=self.config.frame_scale,
        )
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *vllm_image_content(
                        images,
                        detail=self.config.input_image_detail,
                        size=self.config.input_image_size,
                        resample=self.config.input_image_resample,
                        mime_type=self.config.image_mime_type,
                        crop_edges=self.config.input_image_crop_arc_grid_edges,
                    ),
                ],
            },
        ]
        max_repairs = max(0, self.config.repair_attempts)
        for attempt in range(max_repairs + 1):
            response = self._client.chat(
                model=self._request_model(),
                messages=messages,
                response_format=json_schema_response_format(
                    name=schema_name,
                    schema=schema,
                ),
            )
            self.last_request = self._client.last_request
            self.last_response_id = (
                str(object_get(response, "id"))
                if object_get(response, "id") is not None
                else None
            )
            self.last_usage = plain(object_get(response, "usage"))
            self._capture_request(phase=phase, response=response, attempt=attempt)
            response_text = chat_message_content(response)
            try:
                parsed = parse_json_object(response_text, label="agent")
                if validate is not None:
                    validate(parsed)
                return parsed
            except RuntimeError as exc:
                if attempt >= max_repairs:
                    raise
                messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": _json_repair_prompt(
                                schema_name=schema_name,
                                validation_error=str(exc),
                                invalid_text=response_text,
                                attempt=attempt + 1,
                            ),
                        },
                    ]
                )
        raise RuntimeError("unreachable vLLM agent repair state")

    def _request_model(self) -> str | None:
        return self.active_lora_adapter_name or self.config.model

    def _assistant_message(self, message: Any) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": object_get(message, "content", "") or "",
        }

    def _capture_request(
        self,
        *,
        phase: str,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        if self.last_request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot="agent",
            provider=self.backend,
            model=self.model,
            phase=phase,
            request=self.last_request,
            response=response,
            metadata={
                "response_metadata": chat_response_metadata(response),
            },
            attempt=attempt,
        )


def _parse_final_candidate_action(
    arguments: Any,
    candidates: Sequence[AgentCandidateAction],
    *,
    coordinate_space: Any,
    crop_edges: Any | None,
) -> ActionSpec:
    parsed = parse_arguments(arguments)
    indexed = _candidate_from_index(parsed, candidates)
    if indexed is not None:
        return indexed.action

    candidate_actions = tuple(candidate.action for candidate in candidates)
    action_value = parsed.get("action")
    if isinstance(action_value, dict):
        action_value = _strip_simple_action_data(action_value, candidate_actions)
        parsed = {**parsed, "action": action_value}
    action = parse_final_action(
        parsed,
        candidate_actions,
        coordinate_space=coordinate_space,
        crop_edges=crop_edges,
    )
    return _require_matching_candidate(action, candidate_actions)


def _candidate_from_index(
    parsed: dict[str, Any],
    candidates: Sequence[AgentCandidateAction],
) -> AgentCandidateAction | None:
    raw_index = _raw_candidate_index(parsed)
    if raw_index is None:
        action = parsed.get("action")
        if isinstance(action, dict):
            raw_index = _raw_candidate_index(action)
    if raw_index is None:
        return None
    for candidate in candidates:
        if candidate.rank == raw_index:
            return candidate
    raise AgentOutputError(f"candidate_index {raw_index} is not available")


def _raw_candidate_index(value: dict[str, Any]) -> int | None:
    for key in (
        "candidate_index",
        "selected_candidate_index",
        "candidateIndex",
        "index",
    ):
        raw_index = value.get(key)
        if isinstance(raw_index, bool):
            continue
        if isinstance(raw_index, int):
            return raw_index
    return None


def _strip_simple_action_data(
    action_value: dict[str, Any],
    candidates: Sequence[ActionSpec],
) -> dict[str, Any]:
    action_id = action_value.get("action_id")
    if action_id is None or "data" not in action_value:
        return action_value
    matching = [candidate for candidate in candidates if str(action_id) == candidate.name]
    if len(matching) == 1 and not matching[0].is_complex():
        cleaned = dict(action_value)
        cleaned.pop("data", None)
        return cleaned
    return action_value


def _require_matching_candidate(
    action: ActionSpec,
    candidates: Sequence[ActionSpec],
) -> ActionSpec:
    """Return the matching candidate action or fail clearly."""

    for candidate in candidates:
        if candidate.name != action.name:
            continue
        if candidate.name == "ACTION6" and (candidate.data or {}) != (action.data or {}):
            continue
        return candidate
    same_name = [candidate for candidate in candidates if candidate.name == action.name]
    if len(same_name) == 1:
        return same_name[0]
    raise AgentOutputError(f"selected action was not one of the candidates: {action}")


def _validate_candidate_actions_payload(
    parsed: dict[str, Any],
    *,
    action_space: Sequence[ActionSpec],
    coordinate_space: Any,
    crop_edges: Any | None,
) -> None:
    payload = _candidate_payload(parsed)
    for raw in _candidate_action_values(payload):
        parse_action(
            raw,
            action_space,
            coordinate_space=coordinate_space,
            crop_edges=crop_edges,
        )


def _candidate_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    if _has_candidate_value(parsed):
        return parsed
    properties = parsed.get("properties")
    if isinstance(properties, dict) and _has_candidate_value(properties):
        return properties
    return parsed


def _has_candidate_value(value: dict[str, Any]) -> bool:
    return any(
        isinstance(value.get(key), (list, dict))
        for key in (
            "candidate_actions",
            "candidates",
            "actions",
            "additionalActions",
            "additional_actions",
            "action",
        )
    )


def _candidate_action_values(parsed: dict[str, Any]) -> list[Any]:
    for key in (
        "candidate_actions",
        "candidates",
        "actions",
        "additionalActions",
        "additional_actions",
    ):
        value = parsed.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
    action = parsed.get("action")
    if isinstance(action, dict):
        return [action]
    raise AgentOutputError(
        "candidate_actions must be a list; parsed response preview: "
        + json.dumps(parsed, sort_keys=True)[:300]
    )


def _json_repair_prompt(
    *,
    schema_name: str,
    validation_error: str,
    invalid_text: str,
    attempt: int,
) -> str:
    return "\n\n".join(
        [
            f"Repair attempt {attempt}: the previous {schema_name} output was invalid.",
            "Validation error:\n" + validation_error,
            "Invalid output:\n" + invalid_text,
            "Return only the corrected JSON object. Do not include prose, Markdown fences, or the JSON schema.",
        ]
    )
