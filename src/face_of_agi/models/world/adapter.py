"""vLLM adapter for the World role."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_of_agi.contracts import WorldPrediction
from face_of_agi.models.action_glossary import action_glossary_text
from face_of_agi.models.hf_roles import (
    HFJsonRoleClient,
    observation_image as hf_observation_image,
    parse_json_object as hf_parse_json_object,
)
from face_of_agi.models.world.config import HFWorldConfig, VLLMWorldConfig
from face_of_agi.models.world.contracts import (
    WorldPredictionInput,
    world_prediction_json_schema,
)
from face_of_agi.models.vllm_roles import (
    VLLMJsonRoleClient,
    action_text,
    observation_image,
    parse_json_object,
)

DEFAULT_INSTRUCTION_PATH = Path(__file__).parent / "instructions" / "instruction_prompt.md"


class VLLMWorldAdapter:
    """World role backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMWorldConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = VLLMJsonRoleClient(
            config=config,
            call_slot="world",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            client=client,
        )

    def activate_lora_adapter(self, adapter_name: str) -> None:
        """Use a successfully loaded vLLM LoRA adapter for future World calls."""

        self.provider.activate_lora_adapter(adapter_name)

    def predict_transition(
        self,
        prediction_input: WorldPredictionInput,
    ) -> WorldPrediction:
        """Predict the visible transition for one candidate action."""

        text = self.provider.complete_json(
            prompt_text=_world_prompt(self.config, prediction_input),
            output_schema=world_prediction_json_schema(),
            schema_name="world_prediction",
            images=(observation_image(self.config, prediction_input.current_observation),),
        )
        payload = parse_json_object(text, label="world")
        prediction = str(payload.get("predicted_change") or "").strip()
        if not prediction:
            raise RuntimeError("world response requires non-empty predicted_change")
        metadata = {
            "backend": "vllm",
            "model": self.config.model,
            "training_phase": "complete",
            "training_schema_name": "world_prediction",
            "usage": self.provider.last_usage,
        }
        if self.provider.last_request is not None:
            metadata["training_request"] = self.provider.last_request
        return WorldPrediction(
            candidate_index=prediction_input.candidate_index,
            action=prediction_input.action,
            predicted_change=prediction,
            metadata=metadata,
        )


class HFWorldAdapter:
    """World role backed by the shared HF/Transformers VLM."""

    def __init__(
        self,
        config: HFWorldConfig,
        *,
        engine: Any | None = None,
    ) -> None:
        self.config = config
        self.provider = HFJsonRoleClient(
            config=config,
            call_slot="world",
            instruction_path=DEFAULT_INSTRUCTION_PATH,
            engine=engine,
        )

    def activate_lora_adapter(self, adapter_name: str) -> None:
        """Use a loaded HF LoRA adapter for future World calls."""

        self.provider.activate_lora_adapter(adapter_name)

    def predict_transition(
        self,
        prediction_input: WorldPredictionInput,
    ) -> WorldPrediction:
        """Predict the visible transition for one candidate action."""

        text = self.provider.complete_json(
            prompt_text=_world_prompt(self.config, prediction_input),
            output_schema=world_prediction_json_schema(),
            schema_name="world_prediction",
            images=(
                hf_observation_image(
                    self.config,
                    prediction_input.current_observation,
                ),
            ),
        )
        payload = hf_parse_json_object(text, label="world")
        prediction = str(payload.get("predicted_change") or "").strip()
        if not prediction:
            raise RuntimeError("world response requires non-empty predicted_change")
        metadata = {
            "backend": "hf_transformers",
            "model": self.provider.model,
            "training_phase": "complete",
            "training_schema_name": "world_prediction",
            "usage": self.provider.last_usage,
        }
        if self.provider.last_request is not None:
            metadata["training_request"] = self.provider.last_request
        return WorldPrediction(
            candidate_index=prediction_input.candidate_index,
            action=prediction_input.action,
            predicted_change=prediction,
            metadata=metadata,
        )


def _world_prompt(
    config: VLLMWorldConfig,
    prediction_input: WorldPredictionInput,
) -> str:
    return "\n\n".join(
        [
            f"run_id: {prediction_input.run_id}",
            f"game_id: {prediction_input.game_id}",
            f"candidate_index: {prediction_input.candidate_index}",
            "Attached image: current frame only.",
            "Candidate action:",
            action_text(
                prediction_input.action,
                crop_edges=config.input_image_crop_arc_grid_edges,
            ),
            "Action glossary:",
            action_glossary_text(
                prediction_input.glossary_actions,
                mode="agent_decision",
            ),
            "Current Memory document:",
            prediction_input.memory.document,
        ]
    )
