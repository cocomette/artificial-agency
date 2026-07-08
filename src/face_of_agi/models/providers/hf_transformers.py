"""Shared HF/Transformers multimodal chat engine for local v1 roles."""

from __future__ import annotations

import base64
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from io import BytesIO
import inspect
import json
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Mapping, Sequence

from face_of_agi.models.providers.openai import set_optional
from face_of_agi.runtime import timing as runtime_timing

DEFAULT_HF_BACKEND = "hf_transformers"


@dataclass(slots=True)
class HFChatConfig:
    """Shared config for HF/Transformers chat-style role calls."""

    backend: str | None = DEFAULT_HF_BACKEND
    model: str | None = None
    model_path: str | None = None
    local_files_only: bool = False
    quantization: str = "bnb_4bit"
    device_map: str = "auto"
    torch_dtype: str = "bf16"
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_batch_size: int = 4
    max_queue_wait_ms: float = 20.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    options: dict[str, Any] = field(default_factory=dict)
    extra_request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _QueuedChatRequest:
    request: dict[str, Any]
    signature: tuple[Any, ...]
    future: Future[Any]


class HFVLMEngine:
    """One process-local Transformers VLM with queued inference and LoRA training."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model_name = _base_model_name(config)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._model_lock = threading.RLock()
        self._condition = threading.Condition()
        self._queue: list[_QueuedChatRequest] = []
        self._active_generation_batches = 0
        self._training_active = False
        self._closed = False
        self._loaded_adapters: set[str] = set()
        self._prepared_for_quantized_training = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="face-of-agi-hf-vlm-engine",
            daemon=True,
        )
        self._worker.start()

    def chat(self, request: dict[str, Any]) -> Any:
        """Queue one chat request and return an OpenAI-compatible response."""

        future: Future[Any] = Future()
        queued = _QueuedChatRequest(
            request=dict(request),
            signature=self._signature(request),
            future=future,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("HF VLM engine is closed")
            self._queue.append(queued)
            self._condition.notify_all()
        timeout = getattr(self.config, "timeout", None)
        return future.result(timeout=timeout)

    def load_adapter(self, *, adapter_name: str, adapter_path: str | Path) -> None:
        """Load a saved PEFT adapter into the warm model if needed."""

        with self.exclusive_training():
            self._ensure_loaded()
            if adapter_name in self._loaded_adapters:
                return
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("HF LoRA adapter loading requires peft") from exc
            if self._model is None:
                raise RuntimeError("HF VLM model is not loaded")
            if isinstance(self._model, PeftModel):
                self._model.load_adapter(
                    str(adapter_path),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            else:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    str(adapter_path),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            self._loaded_adapters.add(adapter_name)
            self._model.eval()

    def prepare_trainable_adapter(
        self,
        *,
        adapter_name: str,
        previous_adapter_path: str | None,
    ) -> None:
        """Prepare one active trainable PEFT adapter on the warm model."""

        self._ensure_loaded()
        if not self._prepared_for_quantized_training:
            self._model = prepare_model_for_quantized_training(
                self._model,
                config=self.config,
            )
            self._prepared_for_quantized_training = True
        try:
            from peft import PeftModel, get_peft_model
        except ImportError as exc:
            raise RuntimeError("HF LoRA training requires peft") from exc
        if self._model is None:
            raise RuntimeError("HF VLM model is not loaded")
        if isinstance(self._model, PeftModel):
            if adapter_name in self._model.peft_config:
                self._model.delete_adapter(adapter_name)
            if previous_adapter_path:
                self._model.load_adapter(
                    previous_adapter_path,
                    adapter_name=adapter_name,
                    is_trainable=True,
                )
            else:
                self._model.add_adapter(adapter_name, lora_config(self.config))
            self._model.set_adapter(adapter_name)
        elif previous_adapter_path:
            self._model = PeftModel.from_pretrained(
                self._model,
                previous_adapter_path,
                adapter_name=adapter_name,
                is_trainable=True,
            )
            self._model.set_adapter(adapter_name)
        else:
            self._model = get_peft_model(
                self._model,
                lora_config(self.config),
                adapter_name=adapter_name,
            )
            self._model.set_adapter(adapter_name)
        self._loaded_adapters.add(adapter_name)
        self._model.train()

    def save_adapter(self, *, adapter_name: str, adapter_path: str | Path) -> None:
        """Save the selected adapter while keeping it loaded."""

        if self._model is None:
            raise RuntimeError("HF VLM model is not loaded")
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(
            str(adapter_path),
            selected_adapters=[adapter_name],
        )
        self._loaded_adapters.add(adapter_name)
        self._model.eval()

    @property
    def model(self) -> Any:
        """Return the loaded model, loading it if necessary."""

        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("HF VLM model is not loaded")
        return self._model

    @property
    def processor(self) -> Any:
        """Return the loaded processor, loading it if necessary."""

        self._ensure_loaded()
        if self._processor is None:
            raise RuntimeError("HF VLM processor is not loaded")
        return self._processor

    def delete_adapter(self, adapter_name: str) -> None:
        """Delete a loaded PEFT adapter after no game handle can use it."""

        with self.exclusive_training():
            if self._model is None or not hasattr(self._model, "delete_adapter"):
                return
            peft_config = getattr(self._model, "peft_config", {})
            if adapter_name in peft_config:
                self._model.delete_adapter(adapter_name)
            self._loaded_adapters.discard(adapter_name)

    @contextmanager
    def exclusive_training(self) -> Any:
        """Block generation while a trainer owns the warm model."""

        with self._condition:
            while (
                self._training_active
                or self._active_generation_batches > 0
            ):
                self._condition.wait()
            self._training_active = True
        try:
            with self._model_lock:
                yield
        finally:
            with self._condition:
                self._training_active = False
                self._condition.notify_all()

    def close(self) -> None:
        """Stop the queue worker and release model references."""

        with self._condition:
            self._closed = True
            for queued in self._queue:
                queued.future.set_exception(RuntimeError("HF VLM engine closed"))
            self._queue.clear()
            self._condition.notify_all()
        self._worker.join(timeout=5)
        self._model = None
        self._processor = None
        self._loaded_adapters.clear()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._closed
                    and (not self._queue or self._training_active)
                ):
                    self._condition.wait()
                if self._closed:
                    return
                first = self._queue.pop(0)
                batch = [first]
                deadline = time.monotonic() + _queue_wait_seconds(self.config)
                while len(batch) < _max_batch_size(self.config):
                    match_index = _first_compatible_index(
                        self._queue,
                        signature=first.signature,
                    )
                    if match_index is not None:
                        batch.append(self._queue.pop(match_index))
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                    if self._training_active:
                        break
                if self._training_active:
                    self._queue = batch + self._queue
                    self._condition.notify_all()
                    continue
                self._active_generation_batches += 1

            try:
                responses = self._generate_batch(
                    [queued.request for queued in batch]
                )
            except BaseException as exc:
                for queued in batch:
                    queued.future.set_exception(exc)
            else:
                for queued, response in zip(batch, responses, strict=True):
                    queued.future.set_result(response)
            finally:
                with self._condition:
                    self._active_generation_batches -= 1
                    if self._active_generation_batches == 0:
                        self._condition.notify_all()

    def _generate_batch(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._model_lock:
            self._ensure_loaded()
            if self._model is None or self._processor is None:
                raise RuntimeError("HF VLM engine failed to load")
            adapter_name = self._adapter_name(requests[0])
            with runtime_timing.span(
                "hf_vlm.prepare_batch",
                adapter=adapter_name or "",
                batch_size=len(requests),
            ):
                texts: list[str] = []
                image_batches: list[list[Any]] = []
                for request in requests:
                    messages, images = normalize_hf_messages(
                        request.get("messages") or ()
                    )
                    texts.append(
                        _apply_chat_template(
                            self._processor,
                            messages,
                            add_generation_prompt=True,
                            chat_template_kwargs=_chat_template_kwargs(request),
                        )
                    )
                    image_batches.append(images)
                inputs = _processor_inputs(
                    self._processor,
                    texts=texts,
                    image_batches=image_batches,
                )
            inputs = _to_device(inputs, _model_device(self._model))
            generation_kwargs = _generation_kwargs(
                requests[0],
                self.config,
                processor=self._processor,
            )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            image_count = sum(len(images) for images in image_batches)
            if _stop_when_json_complete(requests[0]):
                generation_kwargs["stopping_criteria"] = _json_stopping_criteria(
                    processor=self._processor,
                    prompt_tokens=prompt_tokens,
                )
            with _manual_seed(requests[0]):
                with _adapter_context(self._model, adapter_name):
                    import torch

                    with runtime_timing.span(
                        "hf_vlm.generate",
                        adapter=adapter_name or "",
                        batch_size=len(requests),
                        image_count=image_count,
                        max_new_tokens=int(generation_kwargs["max_new_tokens"]),
                        prompt_tokens=prompt_tokens,
                    ):
                        with torch.no_grad():
                            output_ids = self._model.generate(
                                **inputs,
                                **generation_kwargs,
                            )
            completion_ids = output_ids[:, prompt_tokens:]
            texts_out = self._processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            runtime_timing.emit(
                "hf_vlm.generate_result",
                adapter=adapter_name or "",
                batch_size=len(requests),
                completion_tokens=[
                    int(completion_ids[index].shape[0])
                    for index in range(len(requests))
                ],
                prompt_tokens=prompt_tokens,
            )
            return [
                _chat_response(
                    model=str(request.get("model") or self.model_name),
                    content=str(text).strip(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=int(completion_ids[index].shape[0]),
                )
                for index, (request, text) in enumerate(zip(requests, texts_out, strict=True))
            ]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("HF VLM engine requires transformers") from exc
        model_class = auto_vision_model_class()
        with runtime_timing.span(
            "hf_vlm.load",
            model=self.model_name,
            quantization=str(getattr(self.config, "quantization", "none")),
        ):
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **processor_kwargs(self.config, model_name=self.model_name),
            )
            _configure_processor_for_batched_generation(self._processor)
            self._model = model_class.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **model_kwargs(self.config),
            )
            self._model.eval()

    def _signature(self, request: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            self._adapter_name(request),
            request.get("max_completion_tokens") or request.get("max_tokens"),
            request.get("temperature"),
            request.get("top_p"),
            request.get("seed"),
            repr(_chat_template_kwargs(request)),
        )

    def _adapter_name(self, request: Mapping[str, Any]) -> str | None:
        model = str(request.get("model") or "")
        if not model or model in _base_model_aliases(self.config, self.model_name):
            return None
        return model


class HFChatClient:
    """Chat-completions-shaped client backed by the shared HF VLM engine."""

    def __init__(self, config: Any, *, engine: HFVLMEngine | None = None) -> None:
        self.config = config
        self.engine = engine or shared_hf_vlm_engine(config)
        self.last_request: dict[str, Any] | None = None

    def chat(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        request = self.build_request(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_body=extra_body,
        )
        self.last_request = request
        return self.engine.chat(request)

    def build_request(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not model:
            raise ValueError("HF chat calls require an explicit model")
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        set_optional(request, "tools", tools)
        set_optional(request, "tool_choice", tool_choice)
        set_optional(request, "response_format", response_format)
        for key in (
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "seed",
        ):
            set_optional(request, key, getattr(self.config, key, None))
        merged_extra_body = {
            **(getattr(self.config, "options", None) or {}),
            **(extra_body or {}),
        }
        set_optional(request, "extra_body", merged_extra_body)
        request.update(getattr(self.config, "extra_request_options", {}) or {})
        return request


_SHARED_ENGINE_LOCK = threading.Lock()
_SHARED_ENGINE: HFVLMEngine | None = None
_SHARED_ENGINE_KEY: tuple[Any, ...] | None = None


def shared_hf_vlm_engine(config: Any) -> HFVLMEngine:
    """Return the process-global HF VLM engine for compatible configs."""

    global _SHARED_ENGINE, _SHARED_ENGINE_KEY
    key = _engine_key(config)
    with _SHARED_ENGINE_LOCK:
        if _SHARED_ENGINE is not None and getattr(_SHARED_ENGINE, "_closed", False):
            _SHARED_ENGINE = None
            _SHARED_ENGINE_KEY = None
        if _SHARED_ENGINE is None:
            _SHARED_ENGINE = HFVLMEngine(config)
            _SHARED_ENGINE_KEY = key
            return _SHARED_ENGINE
        if _SHARED_ENGINE_KEY != key:
            raise RuntimeError(
                "hf_transformers shared engine config cannot change within one "
                f"process: expected {_SHARED_ENGINE_KEY}, got {key}"
            )
        return _SHARED_ENGINE


def reset_shared_hf_vlm_engine_for_tests() -> None:
    """Close and clear the process-global HF engine."""

    global _SHARED_ENGINE, _SHARED_ENGINE_KEY
    with _SHARED_ENGINE_LOCK:
        if _SHARED_ENGINE is not None:
            _SHARED_ENGINE.close()
        _SHARED_ENGINE = None
        _SHARED_ENGINE_KEY = None


def normalize_hf_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Convert OpenAI image_url chat blocks to HF multimodal content."""

    normalized: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(dict(message))
            continue
        items: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                items.append({"type": "text", "text": str(item)})
                continue
            if item.get("type") != "image_url":
                items.append(dict(item))
                continue
            image = _decode_image_url(item)
            if image is None:
                continue
            images.append(image)
            items.append({"type": "image", "image": image})
        normalized.append({**dict(message), "content": items})
    return normalized, images


def auto_vision_model_class() -> Any:
    """Return the best available Transformers VLM auto class."""

    import transformers

    for name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError("transformers does not provide an image-text model auto class")


def processor_kwargs(config: Any, *, model_name: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    kwargs["use_fast"] = True
    if bool(getattr(config, "local_files_only", False)):
        kwargs["local_files_only"] = True
    if model_name:
        kwargs.update(
            _tokenizer_asset_kwargs(
                model_name,
                local_files_only=bool(getattr(config, "local_files_only", False)),
            )
        )
    return kwargs


def _tokenizer_asset_kwargs(
    model_name: str,
    *,
    local_files_only: bool,
) -> dict[str, str]:
    """Resolve tokenizer assets for processors whose config omits file names."""

    assets = {
        "tokenizer_file": "tokenizer.json",
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    model_path = Path(model_name)
    resolved: dict[str, str] = {}
    if model_path.exists():
        for kwarg_name, filename in assets.items():
            candidate = model_path / filename
            if candidate.exists():
                resolved[kwarg_name] = str(candidate)
        return resolved
    try:
        from transformers.utils import cached_file
    except ImportError:
        return resolved
    for kwarg_name, filename in assets.items():
        try:
            path = cached_file(
                model_name,
                filename,
                local_files_only=local_files_only,
            )
        except Exception:
            continue
        if path:
            resolved[kwarg_name] = str(path)
    return resolved


def model_kwargs(config: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    device_map = _device_map(getattr(config, "device_map", "auto"))
    if device_map is not None:
        kwargs["device_map"] = device_map
    torch_dtype = _torch_dtype(getattr(config, "torch_dtype", "auto"))
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    quantization_config = quantization_config_for(config)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if bool(getattr(config, "local_files_only", False)):
        kwargs["local_files_only"] = True
    return kwargs


def quantization_config_for(config: Any) -> Any | None:
    """Return the configured Transformers quantization config."""

    mode = str(getattr(config, "quantization", "none")).strip().lower()
    if mode in {"", "none"}:
        return None
    if mode != "bnb_4bit":
        raise ValueError(f"unsupported hf_transformers.quantization: {mode}")
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "hf_transformers.quantization=bnb_4bit requires torch, "
            "transformers, and bitsandbytes"
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def prepare_model_for_quantized_training(model: Any, *, config: Any) -> Any:
    """Prepare a BnB model for PEFT training without whole-model fp32 casts."""

    if str(getattr(config, "quantization", "none")).strip().lower() != "bnb_4bit":
        return model
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(_module: Any, _input: Any, output: Any) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if hasattr(model, "gradient_checkpointing_enable"):
        parameters = inspect.signature(model.gradient_checkpointing_enable).parameters
        if "gradient_checkpointing_kwargs" in parameters:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_enable()
    return model


def lora_config(config: Any) -> Any:
    """Return the PEFT LoRA config shared by HF inference/training."""

    from peft import LoraConfig

    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=list(getattr(config, "lora_target_modules", ())),
    )


def _first_compatible_index(
    queue: Sequence[_QueuedChatRequest],
    *,
    signature: tuple[Any, ...],
) -> int | None:
    for index, queued in enumerate(queue):
        if queued.signature == signature:
            return index
    return None


def _base_model_name(config: Any) -> str:
    model = str(getattr(config, "model_path", None) or getattr(config, "model", "") or "")
    if not model:
        raise ValueError("hf_transformers backend requires a model or model_path")
    return model


def _base_model_aliases(config: Any, model_name: str) -> set[str]:
    return {
        value
        for value in {
            model_name,
            str(getattr(config, "model", "") or ""),
            str(getattr(config, "model_path", "") or ""),
        }
        if value
    }


def _engine_key(config: Any) -> tuple[Any, ...]:
    return (
        _base_model_name(config),
        bool(getattr(config, "local_files_only", False)),
        str(getattr(config, "quantization", "none")),
        str(getattr(config, "device_map", "auto")),
        str(getattr(config, "torch_dtype", "auto")),
    )


def _queue_wait_seconds(config: Any) -> float:
    return max(0.0, float(getattr(config, "max_queue_wait_ms", 20.0))) / 1000.0


def _max_batch_size(config: Any) -> int:
    return max(1, int(getattr(config, "max_batch_size", 1)))


def _apply_chat_template(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    apply_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_template):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            **dict(chat_template_kwargs or {}),
        }
        try:
            return str(
                apply_template(
                    list(messages),
                    **kwargs,
                )
            )
        except TypeError:
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            return str(apply_template(list(messages), **kwargs))
    return "\n".join(
        f"{message.get('role', 'user')}: {_message_text(message.get('content'))}"
        for message in messages
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def _chat_template_kwargs(request: Mapping[str, Any]) -> dict[str, Any]:
    extra_body = request.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return {}
    value = extra_body.get("chat_template_kwargs")
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def _stop_when_json_complete(request: Mapping[str, Any]) -> bool:
    extra_body = request.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return False
    return bool(extra_body.get("stop_when_json_complete"))


def _json_stopping_criteria(*, processor: Any, prompt_tokens: int) -> Any:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError as exc:
        raise RuntimeError("JSON stopping requires transformers") from exc

    tokenizer = getattr(processor, "tokenizer", processor)

    class JsonCompleteStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            del scores, kwargs
            for row in input_ids:
                text = tokenizer.decode(
                    row[prompt_tokens:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not _has_complete_json_value(text):
                    return False
            return True

    return StoppingCriteriaList([JsonCompleteStoppingCriteria()])


def _has_complete_json_value(text: str) -> bool:
    start = text.find("{")
    if start < 0:
        return False
    try:
        _, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return False
    return not text[start + end :].strip()


def _processor_inputs(
    processor: Any,
    *,
    texts: Sequence[str],
    image_batches: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "text": list(texts),
        "padding": True,
        "return_tensors": "pt",
    }
    if any(image_batches):
        kwargs["images"] = [list(images) for images in image_batches]
    return processor(**kwargs)


def _generation_kwargs(
    request: Mapping[str, Any],
    config: Any,
    *,
    processor: Any,
) -> dict[str, Any]:
    max_new_tokens = (
        request.get("max_completion_tokens")
        or request.get("max_tokens")
        or getattr(config, "max_completion_tokens", None)
        or getattr(config, "max_tokens", None)
        or 512
    )
    temperature = request.get("temperature")
    if temperature is None:
        temperature = getattr(config, "temperature", None)
    top_p = request.get("top_p")
    if top_p is None:
        top_p = getattr(config, "top_p", None)
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(temperature and float(temperature) > 0.0),
    }
    if kwargs["do_sample"] and temperature is not None:
        kwargs["temperature"] = float(temperature)
    if kwargs["do_sample"] and top_p is not None:
        kwargs["top_p"] = float(top_p)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        kwargs["pad_token_id"] = eos_token_id
    return kwargs


def _configure_processor_for_batched_generation(processor: Any) -> None:
    """Use left padding for decoder-only batched generation."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "padding_side"):
        tokenizer = processor
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _to_device(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        to = getattr(value, "to", None)
        moved[key] = to(device) if callable(to) else value
    return moved


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return "cuda"


def _device_map(value: Any) -> Any | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized in {"cuda", "gpu"}:
        return {"": "cuda:0"}
    if normalized.startswith("cuda:"):
        return {"": normalized}
    return value


def _torch_dtype(value: Any) -> Any | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    import torch

    names = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in names:
        raise ValueError(f"unsupported hf_transformers.torch_dtype: {value}")
    return names[normalized]


@contextmanager
def _manual_seed(request: Mapping[str, Any]) -> Any:
    seed = request.get("seed")
    if seed is None:
        yield
        return
    import torch

    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def _adapter_context(model: Any, adapter_name: str | None) -> Any:
    if adapter_name:
        set_adapter = getattr(model, "set_adapter", None)
        if not callable(set_adapter):
            raise RuntimeError(f"HF adapter is not loaded: {adapter_name}")
        set_adapter(adapter_name)
        yield
        return
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        with disable_adapter():
            yield
        return
    with nullcontext():
        yield


def _decode_image_url(item: Mapping[str, Any]) -> Any | None:
    image_url = item.get("image_url")
    if not isinstance(image_url, dict):
        return None
    url = image_url.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        encoded = url.split(",", 1)[1]
    except IndexError as exc:
        raise RuntimeError("invalid data URL in HF chat image") from exc
    from PIL import Image

    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _chat_response(
    *,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-hf-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


__all__ = [
    "DEFAULT_HF_BACKEND",
    "HFChatClient",
    "HFChatConfig",
    "HFVLMEngine",
    "auto_vision_model_class",
    "lora_config",
    "model_kwargs",
    "normalize_hf_messages",
    "prepare_model_for_quantized_training",
    "processor_kwargs",
    "quantization_config_for",
    "reset_shared_hf_vlm_engine_for_tests",
    "shared_hf_vlm_engine",
]
