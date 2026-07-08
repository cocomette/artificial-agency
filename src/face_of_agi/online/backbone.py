"""Frozen observation backbones for online learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from face_of_agi.contracts import Observation
from face_of_agi.environment.config import BackboneRuntimeConfig
from face_of_agi.frames import observation_to_pil_image, resize_image_if_needed


FeatureVector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Frozen-backbone representation plus metadata for one observation."""

    features: FeatureVector
    metadata: dict[str, Any] = field(default_factory=dict)


class FrozenBackbone(Protocol):
    """Observation encoder boundary used by the online learner."""

    def encode(self, observation: Observation) -> EncodedObservation:
        """Return a frozen representation for one observation."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Return static encoder metadata."""
        ...


class DeterministicBackbone:
    """Small deterministic backbone for tests and no-weight local smoke runs."""

    def __init__(self, *, feature_dim: int = 32) -> None:
        if feature_dim < 4:
            raise ValueError("feature_dim must be at least 4")
        self.feature_dim = feature_dim

    def encode(self, observation: Observation) -> EncodedObservation:
        import numpy as np

        image = observation_to_pil_image(observation)
        array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        channels = array.reshape(-1, 3)
        stats = [
            float(array.mean()),
            float(array.std()),
            float(channels[:, 0].mean()),
            float(channels[:, 1].mean()),
            float(channels[:, 2].mean()),
            float(array.min()),
            float(array.max()),
            float(observation.step or 0),
        ]
        flat = array.reshape(-1)
        if flat.size:
            stride = max(1, flat.size // max(1, self.feature_dim - len(stats)))
            stats.extend(float(item) for item in flat[::stride])
        features = tuple(stats[: self.feature_dim])
        if len(features) < self.feature_dim:
            features = features + (0.0,) * (self.feature_dim - len(features))
        return EncodedObservation(
            features=features,
            metadata={
                "backend": "deterministic",
                "feature_dim": self.feature_dim,
                "observation_id": observation.id,
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": "deterministic", "feature_dim": self.feature_dim}


class TransformersBackbone:
    """Frozen in-process Hugging Face Transformers vision backbone."""

    def __init__(
        self,
        config: BackboneRuntimeConfig,
        *,
        feature_dim: int | None = None,
    ) -> None:
        self.config = config
        self.feature_dim = feature_dim
        self.model_path = Path(config.model_path)
        self.processor_path = (
            Path(config.processor_path) if config.processor_path else self.model_path
        )
        if config.local_files_only and not self.model_path.exists():
            raise RuntimeError(
                "agent.backbone.model_path must exist when local_files_only is true: "
                f"{self.model_path}"
            )
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None
        self._load()

    def encode(self, observation: Observation) -> EncodedObservation:
        if self.config.model_family == "qwen3_5_moe_multimodal":
            return self._encode_qwen_multimodal(observation)
        return self._encode_generic_vision(observation)

    def _encode_generic_vision(self, observation: Observation) -> EncodedObservation:
        torch = self._require_torch()
        processor = self._require_processor()
        model = self._require_model()

        image = observation_to_pil_image(observation)
        image = resize_image_if_needed(
            image,
            size=self.config.image_size,
            resample="nearest",
        )
        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
        tensor = self._select_representation(outputs)
        values = tensor.detach().float().cpu().flatten().tolist()
        return EncodedObservation(
            features=tuple(float(value) for value in values),
            metadata={
                **self.metadata(),
                "observation_id": observation.id,
                "feature_dim": len(values),
            },
        )

    def _encode_qwen_multimodal(self, observation: Observation) -> EncodedObservation:
        torch = self._require_torch()
        processor = self._require_processor()
        model = self._require_model()

        image = observation_to_pil_image(observation)
        image = resize_image_if_needed(
            image,
            size=self.config.image_size,
            resample="nearest",
        )
        inputs = self._build_qwen_inputs(processor, image)
        inputs = self._move_inputs(inputs)
        context_factory = getattr(torch, "inference_mode", torch.no_grad)
        with context_factory():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        values, raw_feature_dim, image_token_count = self._select_qwen_representation(
            inputs,
            outputs,
            model,
        )
        return EncodedObservation(
            features=values,
            metadata={
                **self.metadata(),
                "observation_id": observation.id,
                "feature_dim": len(values),
                "raw_feature_dim": raw_feature_dim,
                "image_token_count": image_token_count,
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model_family": self.config.model_family,
            "model_path": str(self.model_path),
            "processor_path": str(self.processor_path),
            "device": str(self._device),
            "dtype": self.config.dtype,
            "representation_layer": self.config.representation_layer,
            "local_files_only": self.config.local_files_only,
            "feature_dim": self.feature_dim,
        }

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backbone requires the ml dependencies. Run with "
                "`uv sync --extra ml --no-dev` or package them in the Kaggle "
                "wheelhouse."
            ) from exc

        self._torch = torch
        self._device = _resolve_device(torch, self.config.device)
        dtype = _resolve_dtype(torch, self.config.dtype)
        processor_kwargs = {
            **self.config.options,
            **self.config.processor_kwargs,
        }
        model_kwargs = {
            **self.config.options,
            **self.config.model_kwargs,
        }
        if (
            dtype is not None
            and "torch_dtype" not in model_kwargs
            and "dtype" not in model_kwargs
        ):
            model_kwargs["torch_dtype"] = dtype
        processor_error: Exception | None = None
        if self.config.model_family == "qwen3_5_moe_multimodal":
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.processor_path,
                    local_files_only=self.config.local_files_only,
                    **processor_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    "unable to load local Qwen Transformers processor from "
                    f"{self.processor_path}: {exc}"
                ) from exc
        else:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.processor_path,
                    local_files_only=self.config.local_files_only,
                    **processor_kwargs,
                )
            except Exception as exc:
                processor_error = exc
                try:
                    self._processor = AutoImageProcessor.from_pretrained(
                        self.processor_path,
                        local_files_only=self.config.local_files_only,
                        **processor_kwargs,
                    )
                except Exception as image_exc:
                    raise RuntimeError(
                        "unable to load local Transformers processor from "
                        f"{self.processor_path}: {image_exc}"
                    ) from processor_error

        self._model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=self.config.local_files_only,
            **model_kwargs,
        )
        self._model.eval()
        if "device_map" not in model_kwargs:
            self._model.to(self._device)
        else:
            model_device = getattr(self._model, "device", None)
            if model_device is not None:
                self._device = model_device
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _build_qwen_inputs(self, processor: Any, image: Any) -> Any:
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError(
                "qwen3_5_moe_multimodal backbone requires a processor with "
                "apply_chat_template"
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.config.feature_prompt},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _move_inputs(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self._device)
        if not isinstance(inputs, dict):
            return inputs
        return {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _select_qwen_representation(
        self,
        inputs: Any,
        outputs: Any,
        model: Any,
    ) -> tuple[FeatureVector, int, int]:
        if self.config.representation_layer != "image_tokens_mean":
            raise RuntimeError(
                "qwen3_5_moe_multimodal backbone requires "
                "representation_layer: image_tokens_mean"
            )
        input_ids = _mapping_get(inputs, "input_ids")
        if input_ids is None:
            raise RuntimeError("Qwen backbone processor did not return input_ids")
        image_token_id = getattr(
            getattr(model, "config", None),
            "image_token_id",
            None,
        )
        if image_token_id is None:
            raise RuntimeError("Qwen backbone model config is missing image_token_id")
        hidden_state = _final_hidden_state(outputs)
        pooled, image_token_count = _mean_hidden_vectors_for_token(
            input_ids,
            hidden_state,
            int(image_token_id),
        )
        raw_feature_dim = len(pooled)
        return (
            _project_values(pooled, self.feature_dim),
            raw_feature_dim,
            image_token_count,
        )

    def _select_representation(self, outputs: Any) -> Any:
        layer = self.config.representation_layer
        if layer == "pooled" and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output[0]
        if getattr(outputs, "last_hidden_state", None) is not None:
            hidden = outputs.last_hidden_state
            if layer == "cls":
                return hidden[0, 0]
            return hidden.mean(dim=1)[0]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0].flatten()
        raise RuntimeError("Transformers backbone did not return hidden states")

    def _require_torch(self) -> Any:
        if self._torch is None:
            raise RuntimeError("Transformers backbone was not loaded")
        return self._torch

    def _require_processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError("Transformers processor was not loaded")
        return self._processor

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Transformers model was not loaded")
        return self._model


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    if hasattr(value, "get"):
        return value.get(key)
    return None


def _final_hidden_state(outputs: Any) -> Any:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    if getattr(outputs, "last_hidden_state", None) is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise RuntimeError("Qwen backbone did not return hidden states")


def _mean_hidden_vectors_for_token(
    input_ids: Any,
    hidden_state: Any,
    token_id: int,
) -> tuple[list[float], int]:
    token_rows = _batched_token_ids(input_ids)
    hidden_rows = _batched_hidden_states(hidden_state)
    selected: list[list[float]] = []
    for tokens, vectors in zip(token_rows, hidden_rows, strict=False):
        for token, vector in zip(tokens, vectors, strict=False):
            if int(token) == token_id:
                selected.append([float(value) for value in vector])
    if not selected:
        raise RuntimeError("Qwen backbone did not find image tokens in input_ids")
    width = len(selected[0])
    return (
        [
            sum(vector[index] for vector in selected) / len(selected)
            for index in range(width)
        ],
        len(selected),
    )


def _batched_token_ids(value: Any) -> list[list[int]]:
    data = _to_python_list(value)
    if not isinstance(data, list):
        raise RuntimeError("Qwen backbone input_ids must be a tensor or list")
    if not data:
        return []
    if _is_scalar(data[0]):
        return [[int(item) for item in data]]
    return [[int(item) for item in row] for row in data]


def _batched_hidden_states(value: Any) -> list[list[list[float]]]:
    data = _to_python_list(value)
    if not isinstance(data, list):
        raise RuntimeError("Qwen backbone hidden states must be a tensor or list")
    if not data:
        return []
    if isinstance(data[0], list) and data[0] and _is_scalar(data[0][0]):
        return [[[float(item) for item in vector] for vector in data]]
    return [
        [[float(item) for item in vector] for vector in row]
        for row in data
    ]


def _to_python_list(value: Any) -> Any:
    item = value
    for method in ("detach", "float", "cpu"):
        if hasattr(item, method):
            item = getattr(item, method)()
    if hasattr(item, "tolist"):
        return item.tolist()
    return item


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool))


def _project_values(values: list[float], target_dim: int | None) -> FeatureVector:
    if target_dim is None:
        return tuple(float(value) for value in values)
    if target_dim <= 0:
        raise RuntimeError("Qwen backbone feature_dim must be positive")
    width = len(values)
    if width == target_dim:
        return tuple(float(value) for value in values)
    if width < target_dim:
        return tuple(float(value) for value in values) + (0.0,) * (
            target_dim - width
        )
    projected: list[float] = []
    for index in range(target_dim):
        start = index * width // target_dim
        end = (index + 1) * width // target_dim
        chunk = values[start:end] or values[start : start + 1]
        projected.append(sum(chunk) / len(chunk))
    return tuple(projected)


def _resolve_device(torch: Any, value: str) -> Any:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_dtype(torch: Any, value: str) -> Any | None:
    if value in {"", "auto", "float32", "fp32"}:
        return None if value in {"", "auto"} else torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported agent.backbone.dtype: {value}")
