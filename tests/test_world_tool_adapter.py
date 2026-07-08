"""Tests for the concrete world-model image editing adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.tools.world import WorldToolAdapter, WorldToolConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "world"
WORLD_MODEL_E2E_PATH = Path(__file__).parents[1] / "scripts" / "world_model_e2e.py"


def load_world_model_e2e_module() -> ModuleType:
    """Load the manual E2E script as a module for focused helper tests."""

    spec = importlib.util.spec_from_file_location(
        "world_model_e2e_test_module",
        WORLD_MODEL_E2E_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeWorldPipeline:
    """Tiny stand-in for QwenImageEditPipeline."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **inputs: object) -> SimpleNamespace:
        self.calls.append(inputs)
        image = Image.new("RGB", (5, 7), color=(255, 0, 0))
        return SimpleNamespace(images=[image])


class FakePipelineClass:
    """Tiny stand-in for a Diffusers pipeline class."""

    from_pretrained_calls: list[dict[str, object]] = []
    from_single_file_calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> FakeWorldPipeline:
        cls.from_pretrained_calls.append({"model": model, **kwargs})
        return FakeWorldPipeline()

    @classmethod
    def from_single_file(cls, model: str, **kwargs: object) -> FakeWorldPipeline:
        cls.from_single_file_calls.append({"model": model, **kwargs})
        return FakeWorldPipeline()


class FakeFluxKontextPipeline:
    """Tiny stand-in for FluxKontextPipeline."""

    from_pretrained_calls: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.transformer: object | None = None
        self.text_encoder_2: object | None = None
        self.to_calls: list[str] = []
        self.progress_configs: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> "FakeFluxKontextPipeline":
        cls.from_pretrained_calls.append({"model": model, **kwargs})
        return cls()

    def to(self, device: str) -> "FakeFluxKontextPipeline":
        self.to_calls.append(device)
        return self

    def set_progress_bar_config(self, **kwargs: object) -> None:
        self.progress_configs.append(kwargs)


class FakeTokenizer:
    """Whitespace tokenizer for E2E prompt-window diagnostics."""

    def __init__(self, model_max_length: int) -> None:
        self.model_max_length = model_max_length

    def __call__(
        self,
        prompt: str,
        *,
        truncation: bool,
        add_special_tokens: bool,
    ) -> SimpleNamespace:
        return SimpleNamespace(input_ids=prompt.split())

    def decode(self, input_ids: list[str], *, skip_special_tokens: bool) -> str:
        return " ".join(input_ids)


def test_world_tool_predict_composes_prompt_and_returns_tool_result() -> None:
    pipeline = FakeWorldPipeline()
    adapter = WorldToolAdapter(
        config=WorldToolConfig(
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
        id="obs-1",
        step=4,
        frame=np.zeros((64, 64), dtype=np.uint8),
    )
    action = ActionSpec(action_id="ACTION6", data={"x": 2, "y": 3})

    result = adapter.predict(
        context=RoleContext(general="General world facts.", game="Game dynamics."),
        action=action,
        observation=observation,
    )

    call = pipeline.calls[0]
    prompt = str(call["prompt"])
    input_image = call["image"]

    assert isinstance(input_image, Image.Image)
    assert input_image.size == (128, 128)
    assert "World Model Instruction" in prompt
    assert "WORLD MODEL DOC (K^S + L^S):" in prompt
    assert "WORLD CONTEXT:" not in prompt
    assert "TASK:" not in prompt
    assert "General world facts." in prompt
    assert "Game dynamics." in prompt
    assert "action_id: ACTION6" in prompt
    assert '"x": 2' in prompt
    assert result.tool == "world"
    assert result.source_observation_ref.memory == "state"
    assert result.source_observation_ref.id == "obs-1"
    assert result.action == action
    assert isinstance(result.predicted_observation, Image.Image)
    assert result.predicted_observation.size == (5, 7)
    assert result.metadata["backend"] == "huggingface-diffusers"
    assert result.metadata["model"] == "Qwen/Qwen-Image-Edit"
    assert result.metadata["pipeline_type"] == "qwen_image_edit"
    assert result.metadata["device"] == "cpu"
    assert result.metadata["steps"] == 3
    assert result.metadata["true_cfg_scale"] == 2.5
    assert result.metadata["image_size"] == (5, 7)
    assert call["true_cfg_scale"] == 2.5
    assert "guidance_scale" not in call
    assert "image_guidance_scale" not in call


def test_world_tool_predict_supports_instruct_pix2pix_call_shape() -> None:
    pipeline = FakeWorldPipeline()
    adapter = WorldToolAdapter(
        config=WorldToolConfig(
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
        id="obs-pix2pix",
        step=2,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(game="World model doc."),
        action=ActionSpec(action_id="ACTION1"),
        observation=observation,
    )

    call = pipeline.calls[0]
    prompt = str(call["prompt"])
    assert call["guidance_scale"] == 6.5
    assert call["image_guidance_scale"] == 1.25
    assert "true_cfg_scale" not in call
    assert call["num_inference_steps"] == 4
    assert isinstance(call["image"], Image.Image)
    assert prompt.startswith("Action ACTION1.")
    assert "Predict next ARC frame." in prompt
    assert "World Model Instruction" not in prompt
    assert "WORLD MODEL DOC" not in prompt
    assert result.metadata["model"] == "timbrooks/instruct-pix2pix"
    assert result.metadata["pipeline_type"] == "instruct_pix2pix"
    assert result.metadata["guidance_scale"] == 6.5
    assert result.metadata["image_guidance_scale"] == 1.25


def test_world_tool_predict_supports_flux_kontext_qint8_call_shape() -> None:
    pipeline = FakeWorldPipeline()
    adapter = WorldToolAdapter(
        config=WorldToolConfig(
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
        id="obs-flux",
        step=2,
        frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
    )

    result = adapter.predict(
        context=RoleContext(game="Objects move one cell after each action."),
        action=ActionSpec(action_id="ACTION1"),
        observation=observation,
    )

    call = pipeline.calls[0]
    prompt = str(call["prompt"])
    assert prompt.startswith("Action ACTION1.")
    assert "Predict next ARC frame." in prompt
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


def test_world_tool_instruct_pix2pix_prompt_keeps_action_and_data_first() -> None:
    adapter = WorldToolAdapter(
        WorldToolConfig(
            model="timbrooks/instruct-pix2pix",
            pipeline_type="instruct_pix2pix",
        )
    )
    prompt = adapter._compose_prompt(
        context=RoleContext(
            game=(
                "Objects move one cell after each action. "
                "This sentence should be less important than the action."
            )
        ),
        action=ActionSpec(action_id="ACTION6", data={"x": 2, "y": 3}),
        observation=Observation(id="obs-pix2pix", step=2, frame=object()),
    )

    assert prompt.startswith('Action ACTION6 data {"x": 2, "y": 3}.')
    assert "Predict next ARC frame." in prompt
    assert "Hint: Objects move one cell after each action" in prompt
    assert "World Model Instruction" not in prompt
    assert len(prompt) < 220


def test_world_tool_flux_prompt_keeps_action_first() -> None:
    adapter = WorldToolAdapter(
        WorldToolConfig(
            model="black-forest-labs/FLUX.1-Kontext-dev",
            pipeline_type="flux_kontext_qint8",
        )
    )
    prompt = adapter._compose_prompt(
        context=RoleContext(game="Objects move one cell after each action."),
        action=ActionSpec(action_id="ACTION6", data={"x": 2, "y": 3}),
        observation=Observation(id="obs-flux", step=2, frame=object()),
    )

    assert prompt.startswith('Action ACTION6 data {"x": 2, "y": 3}.')
    assert "Predict next ARC frame." in prompt
    assert "World model doc: Objects move one cell after each action" in prompt
    assert "World Model Instruction" not in prompt


def test_world_tool_loads_single_file_checkpoints_with_single_file_api() -> None:
    adapter = WorldToolAdapter(
        WorldToolConfig(
            model="models/instruct-pix2pix-pruned/model.safetensors",
            pipeline_type="instruct_pix2pix",
        )
    )
    FakePipelineClass.from_pretrained_calls = []
    FakePipelineClass.from_single_file_calls = []

    adapter._load_diffusers_pipeline(FakePipelineClass, dtype="fake-dtype")

    assert FakePipelineClass.from_pretrained_calls == []
    assert FakePipelineClass.from_single_file_calls == [
        {
            "model": "models/instruct-pix2pix-pruned/model.safetensors",
            "torch_dtype": "fake-dtype",
        }
    ]


def test_world_tool_flux_pipeline_class_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(FluxKontextPipeline=FakeFluxKontextPipeline),
    )
    adapter = WorldToolAdapter(
        WorldToolConfig(pipeline_type="flux_kontext_qint8")
    )

    assert adapter._pipeline_class() is FakeFluxKontextPipeline


def test_world_tool_flux_qint8_component_paths_use_configured_subdir(
    tmp_path: Path,
) -> None:
    quantized_root = tmp_path / "quantized-flux"
    quantized_root.mkdir()
    adapter = WorldToolAdapter(
        WorldToolConfig(
            pipeline_type="flux_kontext_qint8",
            quantized_model=str(quantized_root),
            quantized_subdir="flux-1-kontext-dev",
        )
    )

    assert adapter._flux_qint8_component_path("transformer") == Path(
        quantized_root / "flux-1-kontext-dev/transformer/qint8"
    )
    assert adapter._flux_qint8_component_path("text_encoder") == Path(
        quantized_root / "flux-1-kontext-dev/text_encoder/qint8"
    )


def test_world_tool_flux_qint8_loader_attaches_quantized_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(FluxKontextPipeline=FakeFluxKontextPipeline),
    )
    import diffusers

    adapter = WorldToolAdapter(
        WorldToolConfig(
            model="black-forest-labs/FLUX.1-Kontext-dev",
            pipeline_type="flux_kontext_qint8",
            quantize_text_encoder=True,
        )
    )
    FakeFluxKontextPipeline.from_pretrained_calls = []
    monkeypatch.setattr(diffusers, "FluxKontextPipeline", FakeFluxKontextPipeline)
    monkeypatch.setattr(
        adapter,
        "_load_flux_qint8_transformer",
        lambda device: {"component": "transformer", "device": device},
    )
    monkeypatch.setattr(
        adapter,
        "_load_flux_qint8_text_encoder",
        lambda device: {"component": "text_encoder", "device": device},
    )

    pipeline = adapter._load_flux_kontext_qint8_pipeline(
        dtype="fake-dtype",
        device="cpu",
    )

    assert FakeFluxKontextPipeline.from_pretrained_calls == [
        {
            "model": "black-forest-labs/FLUX.1-Kontext-dev",
            "transformer": None,
            "text_encoder_2": None,
            "torch_dtype": "fake-dtype",
        }
    ]
    assert pipeline.transformer == {"component": "transformer", "device": "cpu"}
    assert pipeline.text_encoder_2 == {"component": "text_encoder", "device": "cpu"}


def test_world_model_e2e_prompt_diagnostics_check_both_flux_tokenizers() -> None:
    e2e = load_world_model_e2e_module()

    adapter = SimpleNamespace(
        _pipeline=SimpleNamespace(
            tokenizer=FakeTokenizer(model_max_length=8),
            tokenizer_2=FakeTokenizer(model_max_length=12),
        )
    )
    prompt = "Action ACTION1. Predict next ARC frame from current image state."

    prompt_check = e2e._inspect_prompt_token_window(
        adapter=adapter,
        prompt=prompt,
        action=ActionSpec(action_id="ACTION1"),
    )

    assert prompt_check["action_in_retained_prompt"] is True
    assert [check["name"] for check in prompt_check["tokenizers"]] == [
        "tokenizer",
        "tokenizer_2",
    ]
    assert all(
        check["action_in_retained_prompt"] for check in prompt_check["tokenizers"]
    )


def test_world_model_e2e_prompt_diagnostics_fail_if_action_is_dropped() -> None:
    e2e = load_world_model_e2e_module()

    adapter = SimpleNamespace(
        _pipeline=SimpleNamespace(
            tokenizer=FakeTokenizer(model_max_length=4),
            tokenizer_2=None,
        )
    )
    prompt = "This prefix pushes Action ACTION1 outside the retained window."

    with pytest.raises(RuntimeError, match="ACTION1"):
        e2e._inspect_prompt_token_window(
            adapter=adapter,
            prompt=prompt,
            action=ActionSpec(action_id="ACTION1"),
        )


def test_world_tool_uses_pil_observation_images_directly() -> None:
    pipeline = FakeWorldPipeline()
    adapter = WorldToolAdapter(
        config=WorldToolConfig(device="cpu", torch_dtype="float32", seed=None),
        pipeline=pipeline,
    )
    observation = Observation(
        id="obs-pil",
        step=1,
        frame=Image.new("RGBA", (9, 11), color=(0, 255, 0, 255)),
    )

    adapter.predict(
        context=RoleContext(),
        action=ActionSpec(action_id="ACTION1"),
        observation=observation,
    )

    input_image = pipeline.calls[0]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.mode == "RGB"
    assert input_image.size == (9, 11)
    assert "generator" not in pipeline.calls[0]


def test_world_tool_pipeline_is_loaded_once() -> None:
    pipeline = FakeWorldPipeline()

    class LoadingAdapter(WorldToolAdapter):
        def __init__(self) -> None:
            super().__init__(
                WorldToolConfig(device="cpu", torch_dtype="float32", seed=None)
            )
            self.load_count = 0

        def _load_pipeline(self) -> FakeWorldPipeline:
            self.load_count += 1
            self._loaded_device = "cpu"
            self._loaded_dtype_name = "float32"
            return pipeline

    adapter = LoadingAdapter()
    observation = Observation(
        id="obs-cache",
        step=0,
        frame=np.zeros((2, 2), dtype=np.uint8),
    )

    adapter.predict(RoleContext(), ActionSpec(action_id="ACTION1"), observation)
    adapter.predict(RoleContext(), ActionSpec(action_id="ACTION2"), observation)

    assert adapter.load_count == 1
    assert adapter.last_prompt is not None
    assert len(pipeline.calls) == 2


def test_world_tool_real_arc_fixture_pair_loads_and_differs() -> None:
    source = Image.open(FIXTURE_DIR / "ls20_seed0_step0_source.png").convert("RGB")
    target = Image.open(FIXTURE_DIR / "ls20_seed0_action1_target.png").convert("RGB")

    assert source.size == target.size
    assert source.mode == "RGB"
    assert target.mode == "RGB"
    assert np.any(np.asarray(source) != np.asarray(target))
