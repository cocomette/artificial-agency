"""Tests for the concrete goal-model image editing adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from face_of_agi.contracts import Observation, ObservationRef, RoleContext, ToolCall
from face_of_agi.models.tools.goal import GoalToolAdapter, GoalToolConfig
from face_of_agi.tools import ToolRouter


class FakeGoalPipeline:
    """Tiny stand-in for a Diffusers image-edit pipeline."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **inputs: object) -> SimpleNamespace:
        self.calls.append(inputs)
        image = Image.new("RGB", (6, 8), color=(0, 255, 0))
        return SimpleNamespace(images=[image])


def test_goal_tool_predict_composes_prompt_and_returns_tool_result() -> None:
    pipeline = FakeGoalPipeline()
    adapter = GoalToolAdapter(
        config=GoalToolConfig(
            device="cpu",
            torch_dtype="float32",
            seed=123,
            num_inference_steps=3,
            true_cfg_scale=2.5,
            frame_scale=2,
        ),
        pipeline=pipeline,
    )
    observation = Observation(
        id="obs-goal",
        step=4,
        frame=np.zeros((64, 64), dtype=np.uint8),
    )

    result = adapter.predict(
        context=RoleContext(
            general="General goal facts.",
            game="Collect green cells to make progress.",
        ),
        observation=observation,
    )

    call = pipeline.calls[0]
    prompt = str(call["prompt"])
    input_image = call["image"]

    assert isinstance(input_image, Image.Image)
    assert input_image.size == (128, 128)
    assert "Goal Model Instruction" in prompt
    assert "best goal-directed action" in prompt
    assert "Do not merely reproduce the source frame" in prompt
    assert "GOAL MODEL DOC (K^G + L^G):" in prompt
    assert "WORLD MODEL DOC" not in prompt
    assert "World Model Instruction" not in prompt
    assert "General goal facts." in prompt
    assert "Collect green cells to make progress." in prompt
    assert "PROPOSED ACTION" not in prompt
    assert "action_id:" not in prompt
    assert result.id.startswith("goal-")
    assert result.tool == "goal"
    assert result.source_observation_ref.memory == "state"
    assert result.source_observation_ref.id == "obs-goal"
    assert result.action is None
    assert isinstance(result.predicted_observation, Image.Image)
    assert result.predicted_observation.size == (6, 8)
    assert result.metadata["backend"] == "huggingface-diffusers"
    assert result.metadata["model"] == "Qwen/Qwen-Image-Edit"
    assert result.metadata["pipeline_type"] == "qwen_image_edit"
    assert result.metadata["device"] == "cpu"
    assert result.metadata["steps"] == 3
    assert result.metadata["true_cfg_scale"] == 2.5
    assert result.metadata["image_size"] == (6, 8)
    assert call["true_cfg_scale"] == 2.5
    assert "guidance_scale" not in call
    assert "image_guidance_scale" not in call


def test_goal_tool_predict_supports_instruct_pix2pix_call_shape() -> None:
    pipeline = FakeGoalPipeline()
    adapter = GoalToolAdapter(
        config=GoalToolConfig(
            model="timbrooks/instruct-pix2pix",
            pipeline_type="instruct_pix2pix",
            device="cpu",
            torch_dtype="float32",
            seed=123,
            num_inference_steps=4,
            guidance_scale=6.5,
            image_guidance_scale=1.25,
        ),
        pipeline=pipeline,
    )
    observation = Observation(
        id="obs-goal-pix2pix",
        step=2,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(game="Reach the bright target cell."),
        observation=observation,
    )

    call = pipeline.calls[0]
    prompt = str(call["prompt"])
    assert call["guidance_scale"] == 6.5
    assert call["image_guidance_scale"] == 1.25
    assert "true_cfg_scale" not in call
    assert call["num_inference_steps"] == 4
    assert isinstance(call["image"], Image.Image)
    assert not prompt.startswith("Action ")
    assert "Predict goal-relevant ARC frame." in prompt
    assert "Goal Model Instruction" not in prompt
    assert "GOAL MODEL DOC" not in prompt
    assert "WORLD MODEL DOC" not in prompt
    assert result.metadata["model"] == "timbrooks/instruct-pix2pix"
    assert result.metadata["pipeline_type"] == "instruct_pix2pix"
    assert result.metadata["guidance_scale"] == 6.5
    assert result.metadata["image_guidance_scale"] == 1.25


def test_goal_tool_predict_supports_flux_kontext_qint8_call_shape() -> None:
    pipeline = FakeGoalPipeline()
    adapter = GoalToolAdapter(
        config=GoalToolConfig(
            model="black-forest-labs/FLUX.1-Kontext-dev",
            pipeline_type="flux_kontext_qint8",
            quantized_model="VincentGOURBIN/flux_qint_8bit",
            device="cpu",
            torch_dtype="bfloat16",
            seed=123,
            num_inference_steps=5,
            true_cfg_scale=1.0,
            guidance_scale=2.5,
            max_sequence_length=128,
            max_area=65_536,
        ),
        pipeline=pipeline,
    )
    observation = Observation(
        id="obs-goal-flux",
        step=2,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(game="Progress is reaching the green exit."),
        observation=observation,
    )

    call = pipeline.calls[0]
    prompt = str(call["prompt"])
    assert not prompt.startswith("Action ")
    assert "Predict goal-relevant ARC frame." in prompt
    assert call["true_cfg_scale"] == 1.0
    assert call["guidance_scale"] == 2.5
    assert call["max_sequence_length"] == 128
    assert call["max_area"] == 65_536
    assert "image_guidance_scale" not in call
    assert result.metadata["model"] == "black-forest-labs/FLUX.1-Kontext-dev"
    assert result.metadata["pipeline_type"] == "flux_kontext_qint8"
    assert result.metadata["quantized_model"] == "VincentGOURBIN/flux_qint_8bit"
    assert result.metadata["quantized_subdir"] == "flux-1-kontext-dev"
    assert result.metadata["quantize_text_encoder"] is True


def test_goal_tool_instruct_pix2pix_prompt_uses_only_goal_context() -> None:
    adapter = GoalToolAdapter(
        GoalToolConfig(
            model="timbrooks/instruct-pix2pix",
            pipeline_type="instruct_pix2pix",
        )
    )
    prompt = adapter._compose_prompt(
        context=RoleContext(
            game=(
                "Progress means collecting green cells. "
                "This sentence should be less important than the first sentence."
            )
        ),
        observation=Observation(id="obs-goal-pix2pix", step=2, frame=object()),
    )

    assert not prompt.startswith("Action ")
    assert "ACTION6" not in prompt
    assert "Predict goal-relevant ARC frame." in prompt
    assert "Goal hint: Progress means collecting green cells" in prompt
    assert "Goal Model Instruction" not in prompt
    assert "World Model Instruction" not in prompt
    assert len(prompt) < 220


def test_goal_tool_flux_prompt_uses_only_goal_context() -> None:
    adapter = GoalToolAdapter(
        GoalToolConfig(
            model="black-forest-labs/FLUX.1-Kontext-dev",
            pipeline_type="flux_kontext_qint8",
        )
    )
    prompt = adapter._compose_prompt(
        context=RoleContext(game="Progress means reaching the green exit."),
        observation=Observation(id="obs-goal-flux", step=2, frame=object()),
    )

    assert not prompt.startswith("Action ")
    assert "ACTION6" not in prompt
    assert "Predict goal-relevant ARC frame." in prompt
    assert "Goal model doc: Progress means reaching the green exit" in prompt
    assert "Goal Model Instruction" not in prompt
    assert "World Model Instruction" not in prompt


def test_tool_router_routes_goal_without_action() -> None:
    pipeline = FakeGoalPipeline()
    goal_tool = GoalToolAdapter(
        config=GoalToolConfig(device="cpu", torch_dtype="float32", seed=None),
        pipeline=pipeline,
    )
    router = ToolRouter(goal_tool=goal_tool)
    observation = Observation(
        id="obs-router-goal",
        step=3,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )
    call = ToolCall(
        tool="goal",
        observation_ref=ObservationRef(memory="state", id=observation.id),
    )

    result = router.route(
        call=call,
        context=RoleContext(game="The target is the green exit."),
        observation=observation,
    )

    assert result.tool == "goal"
    assert result.action is None
    assert "PROPOSED ACTION" not in str(pipeline.calls[0]["prompt"])
