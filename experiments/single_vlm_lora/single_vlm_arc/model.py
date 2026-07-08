"""Trainable single-VLM wrapper and fake test model."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from single_vlm_arc.config import ModelConfig


WORLD_ADAPTER_NAME = "world"
POLICY_ADAPTER_NAME = "policy"


@dataclass(slots=True)
class ModelForwardOutput:
    action_logits: torch.Tensor
    coord_logits: torch.Tensor
    action_frame_logits: torch.Tensor | None

    @property
    def frame_logits(self) -> torch.Tensor:
        """Backward-compatible default next-frame logits for ACTION0/RESET."""

        if self.action_frame_logits is None:
            raise RuntimeError("frame logits were not requested for this forward pass")
        return self.action_frame_logits[:, 0]


class HeadMixin:
    """Shared policy/world heads for HF and fake models."""

    world_adapter_name: str = WORLD_ADAPTER_NAME
    policy_adapter_name: str = POLICY_ADAPTER_NAME
    action_runtime_adapter_name: str = POLICY_ADAPTER_NAME
    role_adapters_enabled: bool = False

    hidden_pooling: str
    hidden_pool: nn.Linear | None
    action_norm: nn.LayerNorm
    action_head: nn.Linear
    action_condition: nn.Embedding
    coord_x_condition: nn.Embedding
    coord_y_condition: nn.Embedding
    coord_head: nn.Linear
    frame_head: nn.Sequential
    latent_patch_position: nn.Embedding
    latent_delta_head: nn.Sequential
    latent_grid_shape: tuple[int, int]
    latent_dim: int
    palette_size: int

    def _init_heads(
        self,
        hidden_size: int,
        palette_size: int,
        *,
        hidden_pooling: str = "last",
        latent_grid_shape: tuple[int, int] = (8, 8),
        latent_dim: int | None = None,
    ) -> None:
        self.hidden_pooling = hidden_pooling
        self.palette_size = int(palette_size)
        self.latent_grid_shape = (int(latent_grid_shape[0]), int(latent_grid_shape[1]))
        self.latent_dim = int(latent_dim or hidden_size)
        self.hidden_pool = (
            nn.Linear(hidden_size, 1)
            if hidden_pooling == "attention"
            else None
        )
        self.action_norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, 8)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        self.action_condition = nn.Embedding(8, hidden_size)
        self.coord_x_condition = nn.Embedding(65, hidden_size)
        self.coord_y_condition = nn.Embedding(65, hidden_size)
        self.coord_head = nn.Linear(hidden_size, 128)
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size, 128 * 8 * 8),
            nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, palette_size, kernel_size=4, stride=2, padding=1),
        )
        patch_count = self.latent_grid_shape[0] * self.latent_grid_shape[1]
        self.latent_patch_position = nn.Embedding(patch_count, hidden_size)
        self.latent_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )

    def _heads_forward(
        self,
        hidden: torch.Tensor,
        *,
        include_frame_logits: bool = True,
    ) -> ModelForwardOutput:
        batch_size, hidden_size = hidden.shape
        frame_logits = None
        if include_frame_logits:
            action_embeddings = self.action_condition.weight.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            conditioned_hidden = hidden.unsqueeze(1) + action_embeddings.unsqueeze(0)
            conditioned_hidden = conditioned_hidden.reshape(batch_size * 8, hidden_size)
            frame_logits = self.frame_head(conditioned_hidden).reshape(
                batch_size,
                8,
                -1,
                64,
                64,
            )
        return ModelForwardOutput(
            action_logits=self.action_head(self.action_norm(hidden)),
            coord_logits=self.coord_head(hidden),
            action_frame_logits=frame_logits,
        )

    def _latent_delta_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        action_index: int,
        selected_x: int | None = None,
        selected_y: int | None = None,
    ) -> torch.Tensor:
        """Predict a selected-action latent delta grid from pooled VLM hidden state."""

        batch_size = hidden.shape[0]
        action_id = max(0, min(7, int(action_index)))
        coord_x = 64 if selected_x is None else max(0, min(63, int(selected_x)))
        coord_y = 64 if selected_y is None else max(0, min(63, int(selected_y)))
        action = torch.tensor([action_id], device=hidden.device, dtype=torch.long)
        x_value = torch.tensor([coord_x], device=hidden.device, dtype=torch.long)
        y_value = torch.tensor([coord_y], device=hidden.device, dtype=torch.long)
        conditioned = (
            hidden
            + self.action_condition(action).to(device=hidden.device, dtype=hidden.dtype)
            + self.coord_x_condition(x_value).to(device=hidden.device, dtype=hidden.dtype)
            + self.coord_y_condition(y_value).to(device=hidden.device, dtype=hidden.dtype)
        )
        patch_positions = self.latent_patch_position.weight.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        patch_hidden = conditioned.unsqueeze(1) + patch_positions.unsqueeze(0)
        patch_hidden = patch_hidden.reshape(batch_size * patch_positions.shape[0], -1)
        delta = self.latent_delta_head(patch_hidden).reshape(
            batch_size,
            self.latent_grid_shape[0],
            self.latent_grid_shape[1],
            self.latent_dim,
        )
        return delta

    def _pool_hidden_sequence(
        self,
        hidden_sequence: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hidden_pooling == "attention":
            if self.hidden_pool is None:
                raise RuntimeError("hidden_pooling='attention' requires hidden_pool")
            pool_parameter = self.hidden_pool.weight
            hidden_sequence = hidden_sequence.to(
                device=pool_parameter.device,
                dtype=pool_parameter.dtype,
            )
            scores = self.hidden_pool(hidden_sequence).squeeze(-1)
            if attention_mask is not None and attention_mask.shape == scores.shape:
                scores = scores.masked_fill(attention_mask.to(torch.bool) == 0, -1e4)
            weights = torch.softmax(scores.float(), dim=-1).to(hidden_sequence.dtype)
            return torch.sum(hidden_sequence * weights.unsqueeze(-1), dim=1)
        return hidden_sequence[:, -1, :]

    def use_world_adapter(self) -> Any:
        return nullcontext()

    def use_policy_adapter(self) -> Any:
        return nullcontext()


class FakeSingleVLMPolicy(nn.Module, HeadMixin):
    """Tiny deterministic trainable model for tests and dry-run configs."""

    def __init__(
        self,
        *,
        hidden_size: int,
        palette_size: int,
        hidden_pooling: str = "last",
    ) -> None:
        super().__init__()
        self.hidden = nn.Parameter(torch.zeros(1, hidden_size))
        self._init_heads(
            hidden_size,
            palette_size,
            hidden_pooling=hidden_pooling,
        )

    def forward(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        include_frame_logits: bool = True,
    ) -> ModelForwardOutput:
        del prompt, images
        hidden_sequence = self.hidden.unsqueeze(1)
        return self._heads_forward(
            self._pool_hidden_sequence(hidden_sequence),
            include_frame_logits=include_frame_logits,
        )

    def predict_latent_delta(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        action_index: int,
        selected_x: int | None = None,
        selected_y: int | None = None,
    ) -> torch.Tensor:
        del prompt, images
        hidden_sequence = self.hidden.unsqueeze(1)
        return self._latent_delta_from_hidden(
            self._pool_hidden_sequence(hidden_sequence),
            action_index=action_index,
            selected_x=selected_x,
            selected_y=selected_y,
        )

    def forward_with_latent_delta(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        action_index: int,
        selected_x: int | None = None,
        selected_y: int | None = None,
        include_frame_logits: bool = False,
    ) -> tuple[ModelForwardOutput, torch.Tensor]:
        del prompt, images
        hidden_sequence = self.hidden.unsqueeze(1)
        hidden = self._pool_hidden_sequence(hidden_sequence)
        return (
            self._heads_forward(hidden, include_frame_logits=include_frame_logits),
            self._latent_delta_from_hidden(
                hidden,
                action_index=action_index,
                selected_x=selected_x,
                selected_y=selected_y,
            ),
        )

    def frame_latent_grid(self, frame: Any) -> torch.Tensor:
        return _fake_frame_latent_grid(
            frame,
            palette_size=self.palette_size,
            grid_shape=self.latent_grid_shape,
            latent_dim=self.latent_dim,
        )

    def save_adapter(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "fake_model.pt")

    def save_step_checkpoint(self, path: str | Path) -> None:
        _save_state_dict(self.state_dict(), Path(path))


class SingleVLMPolicy(nn.Module, HeadMixin):
    """HF VLM with LoRA adapters plus action, coordinate, and frame heads."""

    def __init__(self, config: ModelConfig, *, palette_size: int) -> None:
        super().__init__()
        self.config = config
        self.processor = None
        self.base_model = None
        self.role_adapters_enabled = bool(
            config.lora.enabled and config.lora.separate_role_adapters
        )
        self.action_runtime_adapter_name = (
            self.policy_adapter_name
            if self.role_adapters_enabled
            else self.world_adapter_name
        )
        self.device_name = _resolve_device(config.device)
        self.dtype = _resolve_dtype(config.dtype)
        self._load_hf_model(config)
        hidden_size = _hidden_size(self.base_model)
        self._latent_metadata_error: str | None = None
        try:
            latent_grid_shape, latent_dim = self._infer_projected_latent_metadata()
        except Exception as exc:
            latent_grid_shape, latent_dim = (8, 8), hidden_size
            self._latent_metadata_error = str(exc)
        self._init_heads(
            hidden_size,
            palette_size,
            hidden_pooling=config.hidden_pooling,
            latent_grid_shape=latent_grid_shape,
            latent_dim=latent_dim,
        )
        head_device = self._head_device()
        if self.hidden_pool is not None:
            self.hidden_pool.to(head_device)
        self.action_norm.to(head_device)
        self.action_head.to(head_device)
        self.action_condition.to(head_device)
        self.coord_x_condition.to(head_device)
        self.coord_y_condition.to(head_device)
        self.coord_head.to(head_device)
        self.frame_head.to(head_device)
        self.latent_patch_position.to(head_device)
        self.latent_delta_head.to(head_device)

    def forward(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        include_frame_logits: bool = True,
    ) -> ModelForwardOutput:
        if self.processor is None or self.base_model is None:
            raise RuntimeError("HF model was not initialized")
        hidden = self._forward_pooled_hidden(prompt, images)
        return self._heads_forward(hidden, include_frame_logits=include_frame_logits)

    def predict_latent_delta(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        action_index: int,
        selected_x: int | None = None,
        selected_y: int | None = None,
    ) -> torch.Tensor:
        if self.processor is None or self.base_model is None:
            raise RuntimeError("HF model was not initialized")
        hidden = self._forward_pooled_hidden(prompt, images)
        return self._latent_delta_from_hidden(
            hidden,
            action_index=action_index,
            selected_x=selected_x,
            selected_y=selected_y,
        )

    def forward_with_latent_delta(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        action_index: int,
        selected_x: int | None = None,
        selected_y: int | None = None,
        include_frame_logits: bool = False,
    ) -> tuple[ModelForwardOutput, torch.Tensor]:
        if self.processor is None or self.base_model is None:
            raise RuntimeError("HF model was not initialized")
        hidden = self._forward_pooled_hidden(prompt, images)
        return (
            self._heads_forward(hidden, include_frame_logits=include_frame_logits),
            self._latent_delta_from_hidden(
                hidden,
                action_index=action_index,
                selected_x=selected_x,
                selected_y=selected_y,
            ),
        )

    def frame_latent_grid(self, frame: Any) -> torch.Tensor:
        if self.processor is None or self.base_model is None:
            raise RuntimeError("HF model was not initialized")
        inputs = self._prepare_inputs("Encode this frame.", [frame])
        tokens = self._projected_image_tokens(inputs)
        if tokens is None:
            message = "projected image token extraction is unavailable for this model"
            if self._latent_metadata_error:
                message = f"{message}: {self._latent_metadata_error}"
            raise RuntimeError(message)
        grid_shape = _latent_grid_shape_for_tokens(tokens.shape[0])
        if grid_shape != self.latent_grid_shape:
            raise RuntimeError(
                "projected image token grid changed after model initialization: "
                f"init={self.latent_grid_shape}, current={grid_shape}"
            )
        return tokens.reshape(*grid_shape, tokens.shape[-1]).detach().cpu().float()

    def _forward_pooled_hidden(
        self,
        prompt: str,
        images: Sequence[Any],
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(prompt, images)
        hidden_sequence = self._forward_hidden_sequence(inputs)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_sequence.device)
        hidden = self._pool_hidden_sequence(hidden_sequence, attention_mask)
        head_parameter = next(self.action_head.parameters())
        return hidden.to(device=head_parameter.device, dtype=head_parameter.dtype)

    def _infer_projected_latent_metadata(self) -> tuple[tuple[int, int], int]:
        from PIL import Image

        dummy = Image.new("RGB", self.config.image_size, (0, 0, 0))
        inputs = self._prepare_inputs("Latent grid metadata probe.", [dummy])
        tokens = self._projected_image_tokens(inputs)
        if tokens is None:
            raise RuntimeError("model does not expose projected image tokens")
        return _latent_grid_shape_for_tokens(tokens.shape[0]), int(tokens.shape[-1])

    def _projected_image_tokens(self, inputs: dict[str, Any]) -> torch.Tensor | None:
        assert self.base_model is not None
        inner_model = _inner_model(self.base_model)
        if inner_model is None or not hasattr(inner_model, "get_image_features"):
            return None
        pixel_values = inputs.get("pixel_values")
        position_ids = _image_position_ids(inputs)
        if pixel_values is None:
            return None
        kwargs: dict[str, Any] = {
            "pixel_values": pixel_values,
            "return_dict": True,
        }
        if position_ids is not None:
            kwargs["image_position_ids"] = position_ids
        was_training = bool(getattr(self.base_model, "training", False))
        self.base_model.eval()
        try:
            with torch.no_grad(), _disable_adapters_if_available(self.base_model):
                image_features = inner_model.get_image_features(**kwargs)
        finally:
            if was_training:
                self.base_model.train()
        projected_tokens = getattr(image_features, "pooler_output", None)
        if projected_tokens is None:
            return None
        return _flatten_projected_image_tokens(projected_tokens)

    def save_adapter(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.base_model is not None and hasattr(self.base_model, "save_pretrained"):
            if self.role_adapters_enabled:
                self.base_model.save_pretrained(
                    save_path,
                    selected_adapters=[
                        self.world_adapter_name,
                        self.policy_adapter_name,
                    ],
                )
            else:
                self.base_model.save_pretrained(save_path)
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(save_path)
        torch.save(self._head_state_dict(), save_path / "head_state.pt")

    def save_step_checkpoint(self, path: str | Path) -> None:
        _save_state_dict(self._trainable_state_dict(), Path(path))

    def _load_hf_model(self, config: ModelConfig) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from transformers import AutoProcessor

        processor_id = config.processor_id or config.model_id
        self.processor = AutoProcessor.from_pretrained(
            processor_id,
            trust_remote_code=config.trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
        }
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation
        if config.device == "auto":
            kwargs["device_map"] = "auto"

        model = _load_model_with_fallbacks(config.model_id, kwargs)
        if config.device != "auto":
            model = model.to(self.device_name)
        if config.gradient_checkpointing:
            _enable_gradient_checkpointing(model)

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if config.lora.enabled:
            model = _apply_lora(model, config)

        self.base_model = model

    def use_world_adapter(self) -> Any:
        return self._use_adapter(self.world_adapter_name)

    def use_policy_adapter(self) -> Any:
        return self._use_adapter(self.policy_adapter_name)

    @contextmanager
    def _use_adapter(self, adapter_name: str) -> Any:
        if self.base_model is None or not self.role_adapters_enabled:
            yield
            return
        set_adapter = getattr(self.base_model, "set_adapter", None)
        if not callable(set_adapter):
            yield
            return

        previous_adapter = _active_adapter(self.base_model)
        set_adapter(adapter_name)
        try:
            yield
        finally:
            if previous_adapter is not None:
                set_adapter(previous_adapter)

    def _prepare_inputs(self, prompt: str, images: Sequence[Any]) -> dict[str, Any]:
        assert self.processor is not None
        processor = self.processor
        pil_images = [_resize_image(image, self.config.image_size) for image in images]
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": image}
                        for image in pil_images
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = processor(
                text=prompt,
                images=pil_images,
                return_tensors="pt",
            )
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            prepared[key] = value.to(self._model_device()) if hasattr(value, "to") else value
        return prepared

    def _forward_hidden_sequence(self, inputs: dict[str, Any]) -> torch.Tensor:
        assert self.base_model is not None
        backbone = _hidden_backbone(self.base_model)
        if backbone is not None:
            outputs = backbone(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is not None:
                return last_hidden
        outputs = self.base_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return _hidden_sequence(outputs)

    def _model_device(self) -> torch.device:
        if self.base_model is None:
            return torch.device(self.device_name)
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device(self.device_name)

    def _head_device(self) -> torch.device:
        return self._model_device()

    def _head_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("hidden_pool", self.hidden_pool),
            ("action_norm", self.action_norm),
            ("action_head", self.action_head),
            ("action_condition", self.action_condition),
            ("coord_x_condition", self.coord_x_condition),
            ("coord_y_condition", self.coord_y_condition),
            ("coord_head", self.coord_head),
            ("frame_head", self.frame_head),
            ("latent_patch_position", self.latent_patch_position),
            ("latent_delta_head", self.latent_delta_head),
        ):
            if module is None:
                continue
            for key, value in module.state_dict().items():
                state[f"{prefix}.{key}"] = value.detach().cpu()
        return state

    def _trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self._head_state_dict()
        if self.base_model is not None:
            if self.role_adapters_enabled:
                adapter_names = (self.world_adapter_name, self.policy_adapter_name)
                for name, parameter in self.base_model.named_parameters():
                    if _parameter_name_has_adapter(name, adapter_names):
                        state[f"base_model.{name}"] = parameter.detach().cpu()
            else:
                for name, parameter in self.base_model.named_parameters():
                    if parameter.requires_grad:
                        state[f"base_model.{name}"] = parameter.detach().cpu()
        return state


def build_model(config: ModelConfig, *, palette_size: int) -> nn.Module:
    """Build a fake or HF trainable model from config."""

    if config.backend == "fake":
        return FakeSingleVLMPolicy(
            hidden_size=config.hidden_size,
            palette_size=palette_size,
            hidden_pooling=config.hidden_pooling,
        )
    if config.backend == "hf":
        return SingleVLMPolicy(config, palette_size=palette_size)
    raise ValueError(f"unknown model backend: {config.backend}")


def resolve_lora_target_modules(
    model: nn.Module,
    candidates: Sequence[str],
) -> list[str]:
    """Resolve LoRA target suffixes present on a torch model.

    Gemma 4 wraps projections in modules such as ``q_proj`` whose child
    ``linear`` is the actual ``nn.Linear`` PEFT can patch. In that case target
    the child suffix, for example ``q_proj.linear``, instead of the wrapper.
    """

    candidate_set = set(candidates)
    found_plain: set[str] = set()
    found_wrapped: set[str] = set()
    wrapped_parents: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            suffix = parts[-1]
            if suffix in candidate_set:
                found_plain.add(suffix)
            elif suffix == "linear" and len(parts) >= 2 and parts[-2] in candidate_set:
                found_wrapped.add(".".join(parts[-2:]))
                wrapped_parents.add(parts[-2])
    return sorted((found_plain - wrapped_parents) | found_wrapped)


def _load_model_with_fallbacks(model_id: str, kwargs: dict[str, Any]) -> Any:
    errors: list[Exception] = []
    loaders = (
        ("AutoModelForImageTextToText", "AutoModelForImageTextToText"),
        ("AutoModelForMultimodalLM", "AutoModelForMultimodalLM"),
        ("AutoModelForCausalLM", "AutoModelForCausalLM"),
    )
    import transformers

    for _, loader_name in loaders:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            return loader.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            errors.append(exc)
    messages = "; ".join(str(error) for error in errors[-3:])
    raise RuntimeError(f"failed to load HF model {model_id!r}: {messages}")


def _apply_lora(model: nn.Module, config: ModelConfig) -> nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = resolve_lora_target_modules(model, config.lora.target_modules)
    if not target_modules:
        target_modules = list(config.lora.target_modules)
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(
        model,
        lora_config,
        adapter_name=WORLD_ADAPTER_NAME,
    )
    if not config.lora.separate_role_adapters:
        set_adapter = getattr(model, "set_adapter", None)
        if callable(set_adapter):
            set_adapter(WORLD_ADAPTER_NAME)
        return model
    add_adapter = getattr(model, "add_adapter", None)
    if not callable(add_adapter):
        raise RuntimeError("PEFT model does not support adding a policy adapter")
    add_adapter(POLICY_ADAPTER_NAME, lora_config)
    set_adapter = getattr(model, "set_adapter", None)
    if callable(set_adapter):
        set_adapter(POLICY_ADAPTER_NAME)
    return model


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    for source in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
    ):
        hidden = getattr(source, "hidden_size", None)
        if hidden is not None:
            return int(hidden)
    raise RuntimeError("could not infer hidden size from model config")


def _hidden_sequence(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    raise RuntimeError("model output did not include hidden states")


def _hidden_backbone(model: Any) -> Any | None:
    """Return a callable backbone that emits last_hidden_state without LM logits."""

    peft_inner = getattr(getattr(model, "base_model", None), "model", None)
    candidates = [
        getattr(peft_inner, "model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and callable(candidate):
            return candidate
    return None


def _inner_model(model: Any) -> Any | None:
    peft_inner = getattr(getattr(model, "base_model", None), "model", None)
    return peft_inner or getattr(model, "model", None)


def _image_position_ids(inputs: dict[str, Any]) -> torch.Tensor | None:
    if "image_position_ids" in inputs:
        return inputs["image_position_ids"]
    if "pixel_position_ids" in inputs:
        return inputs["pixel_position_ids"]
    return None


def _disable_adapters_if_available(model: Any) -> Any:
    disable = getattr(model, "disable_adapter", None)
    if callable(disable):
        return disable()
    return nullcontext()


def _active_adapter(model: Any) -> str | list[str] | None:
    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    if isinstance(active_adapters, tuple):
        active_adapters = list(active_adapters)
    if isinstance(active_adapters, list):
        if not active_adapters:
            return None
        if len(active_adapters) == 1:
            return str(active_adapters[0])
        return [str(adapter) for adapter in active_adapters]

    active_adapter = getattr(model, "active_adapter", None)
    if callable(active_adapter):
        active_adapter = active_adapter()
    if isinstance(active_adapter, str):
        return active_adapter
    return None


def _parameter_name_has_adapter(
    name: str,
    adapter_names: Sequence[str],
) -> bool:
    parts = name.split(".")
    return any(adapter_name in parts for adapter_name in adapter_names)


def _flatten_projected_image_tokens(projected_tokens: torch.Tensor) -> torch.Tensor:
    tokens = projected_tokens.detach()
    if tokens.ndim == 2:
        return tokens.float()
    if tokens.ndim == 3:
        return tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])[0].float()
    if tokens.ndim >= 4:
        return tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])[0].float()
    raise RuntimeError(
        "projected image tokens must have at least 2 dimensions, "
        f"got shape={tuple(tokens.shape)}"
    )


def _latent_grid_shape_for_tokens(token_count: int) -> tuple[int, int]:
    token_count = int(token_count)
    side = math.isqrt(token_count)
    if side * side == token_count:
        return (side, side)
    for height in range(side, 1, -1):
        if token_count % height == 0:
            width = token_count // height
            return (height, width)
    raise RuntimeError(f"cannot map {token_count} projected image tokens to a 2D grid")


def _fake_frame_latent_grid(
    frame: Any,
    *,
    palette_size: int,
    grid_shape: tuple[int, int],
    latent_dim: int,
) -> torch.Tensor:
    from single_vlm_arc.online_update import frame_to_palette_tensor
    import torch.nn.functional as F

    frame_tensor = frame_to_palette_tensor(
        frame,
        palette_size=palette_size,
        frame_size=(64, 64),
    ).float()
    denominator = max(int(palette_size) - 1, 1)
    normalized = frame_tensor.div(float(denominator)).unsqueeze(0).unsqueeze(0)
    pooled = F.interpolate(normalized, size=grid_shape, mode="area")[0, 0]
    basis = torch.linspace(0.5, 1.5, int(latent_dim), dtype=torch.float32)
    return pooled.unsqueeze(-1) * basis


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        return None
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in aliases:
        raise ValueError(f"unsupported dtype: {dtype}")
    return aliases[dtype]


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _resize_image(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        from face_of_agi.frames import frame_to_pil_image

        pil_image = frame_to_pil_image(image).convert("RGB")
    if pil_image.size == size:
        return pil_image
    return pil_image.resize(size, Image.Resampling.NEAREST)


def _save_state_dict(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        tensor_state = {
            key: value.detach().cpu()
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        save_file(tensor_state, path)
    except Exception:
        torch.save(state, path)
