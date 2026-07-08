"""Shared Diffusers image-editor backend for model tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.frames import observation_to_pil_image

ImageEditorPipeline = Literal[
    "qwen_image_edit",
    "instruct_pix2pix",
    "flux_kontext_qint8",
]

DEFAULT_FLUX_QINT8_MODEL = "VincentGOURBIN/flux_qint_8bit"


class ImageEditorConfig(Protocol):
    """Config attributes required by the shared Diffusers image editor."""

    backend: str | None
    model: str | None
    pipeline_type: ImageEditorPipeline
    quantized_model: str | None
    quantized_subdir: str
    quantize_text_encoder: bool
    device: str
    torch_dtype: str
    seed: int | None
    num_inference_steps: int
    true_cfg_scale: float
    guidance_scale: float
    image_guidance_scale: float
    max_sequence_length: int
    max_area: int
    negative_prompt: str
    frame_scale: int
    options: dict[str, Any]


class DiffusersImageEditorAdapter:
    """Shared local image-editing backend for world and goal tools."""

    def __init__(
        self,
        config: ImageEditorConfig,
        *,
        pipeline: Any | None = None,
        prompt_dir: Path,
        role_name: str,
    ) -> None:
        self.config = config
        self._pipeline = pipeline
        self._prompt_dir = prompt_dir
        self._role_name = role_name
        self._loaded_device: str | None = None
        self._loaded_dtype_name: str | None = None
        self._instruction_prompts: dict[str, str] = {}
        self.last_prompt: str | None = None

    def _predict_image(
        self,
        context: RoleContext,
        observation: Observation,
        action: ActionSpec | None = None,
    ) -> Any:
        """Run the image editor for one role-specific prediction prompt."""

        image = self._observation_to_image(observation)
        prompt = self._compose_prompt(
            context=context,
            observation=observation,
            action=action,
        )
        self.last_prompt = prompt
        return self._run_pipeline(image=image, prompt=prompt).convert("RGB")

    def _compose_prompt(
        self,
        context: RoleContext,
        observation: Observation,
        action: ActionSpec | None = None,
    ) -> str:
        """Build a role-specific image-editing prompt."""

        raise NotImplementedError

    def _run_pipeline(self, *, image: Any, prompt: str) -> Any:
        """Call the cached Diffusers pipeline with the configured inputs."""

        pipeline = self._require_pipeline()
        inputs: dict[str, Any] = self._pipeline_inputs(image=image, prompt=prompt)
        torch = self._optional_torch()
        if self.config.seed is not None and torch is not None:
            inputs["generator"] = torch.manual_seed(self.config.seed)

        inputs.update(self.config.options)
        if torch is None:
            output = pipeline(**inputs)
            return output.images[0]

        with torch.inference_mode():
            output = pipeline(**inputs)

        return output.images[0]

    def _pipeline_inputs(self, *, image: Any, prompt: str) -> dict[str, Any]:
        """Build backend-specific image-editor call arguments."""

        common_inputs: dict[str, Any] = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": self.config.negative_prompt,
            "num_inference_steps": self.config.num_inference_steps,
        }

        if self.config.pipeline_type == "qwen_image_edit":
            return {
                **common_inputs,
                "true_cfg_scale": self.config.true_cfg_scale,
            }

        if self.config.pipeline_type == "instruct_pix2pix":
            return {
                **common_inputs,
                "guidance_scale": self.config.guidance_scale,
                "image_guidance_scale": self.config.image_guidance_scale,
            }

        if self.config.pipeline_type == "flux_kontext_qint8":
            return {
                **common_inputs,
                "true_cfg_scale": self.config.true_cfg_scale,
                "guidance_scale": self.config.guidance_scale,
                "max_sequence_length": self.config.max_sequence_length,
                "max_area": self.config.max_area,
            }

        raise ValueError(
            f"unsupported {self._role_name} image editor: {self.config.pipeline_type}"
        )

    def _metadata(self, image_size: tuple[int, int]) -> dict[str, Any]:
        """Return common image-editor metadata for a ToolResult."""

        return {
            "backend": self.config.backend,
            "model": self.config.model,
            "pipeline_type": self.config.pipeline_type,
            "quantized_model": self._metadata_quantized_model(),
            "quantized_subdir": self.config.quantized_subdir,
            "quantize_text_encoder": self.config.quantize_text_encoder,
            "device": self._loaded_device or self.config.device,
            "torch_dtype": self._loaded_dtype_name or self.config.torch_dtype,
            "seed": self.config.seed,
            "steps": self.config.num_inference_steps,
            "true_cfg_scale": self.config.true_cfg_scale,
            "guidance_scale": self.config.guidance_scale,
            "image_guidance_scale": self.config.image_guidance_scale,
            "max_sequence_length": self.config.max_sequence_length,
            "max_area": self.config.max_area,
            "image_size": image_size,
        }

    def _require_pipeline(self) -> Any:
        """Load and cache the configured image-edit pipeline on first use."""

        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Any:
        """Create the Diffusers pipeline for the configured image editor."""

        torch = self._import_torch()
        device = self._resolve_device(torch)
        dtype = self._resolve_torch_dtype(torch, device)

        if self.config.pipeline_type == "flux_kontext_qint8":
            pipeline = self._load_flux_kontext_qint8_pipeline(dtype, device)
        else:
            pipeline_class = self._pipeline_class()
            pipeline = self._load_diffusers_pipeline(pipeline_class, dtype)

        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=None)

        self._loaded_device = device
        self._loaded_dtype_name = str(dtype).removeprefix("torch.")
        return pipeline

    def _load_diffusers_pipeline(self, pipeline_class: Any, dtype: Any) -> Any:
        """Load either a Diffusers repo/folder or a single checkpoint file."""

        model = self.config.model
        if model is None:
            raise ValueError(f"{self._role_name} image editor model must be configured")

        if self._is_single_file_checkpoint(model):
            return pipeline_class.from_single_file(model, torch_dtype=dtype)

        return pipeline_class.from_pretrained(model, torch_dtype=dtype)

    def _load_flux_kontext_qint8_pipeline(self, dtype: Any, device: str) -> Any:
        """Load FLUX Kontext with qint8 transformer and optional qint8 T5."""

        from diffusers import FluxKontextPipeline

        model = self.config.model
        if model is None:
            raise ValueError("FLUX Kontext base model must be configured")

        pipeline_kwargs: dict[str, Any] = {
            "transformer": None,
            "torch_dtype": dtype,
        }
        if self.config.quantize_text_encoder:
            pipeline_kwargs["text_encoder_2"] = None

        pipeline = FluxKontextPipeline.from_pretrained(model, **pipeline_kwargs)
        pipeline.transformer = self._load_flux_qint8_transformer(device)
        if self.config.quantize_text_encoder:
            pipeline.text_encoder_2 = self._load_flux_qint8_text_encoder(device)
        return pipeline

    def _load_flux_qint8_transformer(self, device: str) -> Any:
        """Load the qint8 FLUX transformer component."""

        from diffusers.models import FluxTransformer2DModel
        from optimum.quanto import QuantizedDiffusersModel

        class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
            base_class = FluxTransformer2DModel

        path = self._flux_qint8_component_path("transformer")
        return QuantizedFluxTransformer2DModel.from_pretrained(str(path)).to(device)

    def _load_flux_qint8_text_encoder(self, device: str) -> Any:
        """Load the qint8 FLUX T5 text encoder component."""

        from optimum.quanto import QuantizedTransformersModel
        from transformers import T5EncoderModel

        class T5EncoderAutoClass:
            """Adapter for optimum.quanto's AutoClass-style loader contract."""

            @classmethod
            def from_config(cls, config: Any) -> T5EncoderModel:
                return T5EncoderModel._from_config(config)

        class QuantizedT5EncoderModel(QuantizedTransformersModel):
            auto_class = T5EncoderAutoClass

        path = self._flux_qint8_component_path("text_encoder")
        return QuantizedT5EncoderModel.from_pretrained(str(path)).to(device)

    def _flux_qint8_component_path(self, component: str) -> Path:
        """Return one qint8 component path from the quantized FLUX repo."""

        return (
            self._quantized_model_path()
            / self.config.quantized_subdir
            / component
            / "qint8"
        )

    def _quantized_model_path(self) -> Path:
        """Resolve a local or Hugging Face qint8 FLUX model root."""

        model = self._quantized_model_id()
        path = Path(model).expanduser()
        if path.exists():
            return path

        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model))

    def _quantized_model_id(self) -> str:
        """Return the configured qint8 FLUX model id or default repo id."""

        return self.config.quantized_model or DEFAULT_FLUX_QINT8_MODEL

    def _metadata_quantized_model(self) -> str | None:
        """Report the qint8 model only for FLUX qint8 runs."""

        if self.config.pipeline_type == "flux_kontext_qint8":
            return self._quantized_model_id()
        return self.config.quantized_model

    def _is_single_file_checkpoint(self, model: str) -> bool:
        """Return whether the configured model points at one checkpoint file."""

        checkpoint_suffixes = {".ckpt", ".safetensors"}
        return Path(model).suffix in checkpoint_suffixes

    def _pipeline_class(self) -> Any:
        """Return the Diffusers pipeline class for the configured editor."""

        if self.config.pipeline_type == "qwen_image_edit":
            from diffusers import QwenImageEditPipeline

            return QwenImageEditPipeline

        if self.config.pipeline_type == "instruct_pix2pix":
            from diffusers import StableDiffusionInstructPix2PixPipeline

            return StableDiffusionInstructPix2PixPipeline

        if self.config.pipeline_type == "flux_kontext_qint8":
            from diffusers import FluxKontextPipeline

            return FluxKontextPipeline

        raise ValueError(
            f"unsupported {self._role_name} image editor: {self.config.pipeline_type}"
        )

    def _load_instruction_prompt(self, filename: str = "instruction_prompt.md") -> str:
        """Read one fixed role instruction prompt once."""

        if filename not in self._instruction_prompts:
            prompt_path = self._prompt_dir / filename
            self._instruction_prompts[filename] = prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        return self._instruction_prompts[filename]

    def _compact_context_hint(self, context: RoleContext, max_chars: int = 96) -> str:
        """Return a short context hint that fits compact prompt windows."""

        context_text = " ".join(context.composed().split())
        if not context_text:
            return ""

        first_sentence = context_text.split(".")[0].strip()
        if len(first_sentence) <= max_chars:
            return first_sentence
        return first_sentence[:max_chars].rsplit(" ", 1)[0].strip()

    def _observation_to_image(self, observation: Observation) -> Any:
        """Normalize a framework observation frame into a PIL RGB image."""

        return observation_to_pil_image(
            observation,
            frame_scale=self.config.frame_scale,
        )

    def _resolve_device(self, torch: Any) -> str:
        """Choose the execution device requested by config."""

        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"

        return "cpu"

    def _resolve_torch_dtype(self, torch: Any, device: str) -> Any:
        """Choose a torch dtype that fits the resolved device."""

        dtype_name = self.config.torch_dtype
        if dtype_name == "auto":
            if device == "cuda":
                return torch.bfloat16
            if device == "mps":
                return torch.float16
            return torch.float32

        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_by_name[dtype_name]

    def _import_torch(self) -> Any:
        """Import torch lazily so package imports stay lightweight."""

        import torch

        return torch

    def _optional_torch(self) -> Any | None:
        """Return torch when installed, allowing injected fake pipelines in tests."""

        try:
            return self._import_torch()
        except ModuleNotFoundError:
            return None

    def _action_id_text(self, action: ActionSpec) -> str:
        """Return a compact action id for prompts and logs."""

        return str(getattr(action.action_id, "name", action.action_id))

    def _compact_action_text(self, action: ActionSpec) -> str:
        """Render action details early for short prompts."""

        action_id = self._action_id_text(action)
        if action.data is None:
            return action_id
        return f"{action_id} data {self._action_data_text(action)}"

    def _action_data_text(self, action: ActionSpec) -> str:
        """Render action payloads deterministically for the model prompt."""

        if action.data is None:
            return "{}"

        return json.dumps(action.data, sort_keys=True)
