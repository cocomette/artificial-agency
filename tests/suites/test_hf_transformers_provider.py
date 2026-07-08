"""Tests for the shared HF/Transformers provider shell."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from face_of_agi.models.providers.hf_transformers import (
    HFVLMEngine,
    prepare_model_for_quantized_training,
    processor_kwargs,
)


class FakeHFEngine(HFVLMEngine):
    """HF engine with generation replaced by a deterministic fake."""

    def __init__(self, *, max_batch_size: int = 4, max_queue_wait_ms: float = 50.0):
        self.batches: list[list[dict[str, Any]]] = []
        super().__init__(
            SimpleNamespace(
                model="fake-hf",
                max_batch_size=max_batch_size,
                max_queue_wait_ms=max_queue_wait_ms,
                timeout=2,
            )
        )

    def _generate_batch(self, requests):
        self.batches.append(list(requests))
        return [
            {
                "id": f"fake-{index}",
                "model": request["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            for index, request in enumerate(requests)
        ]


def _request(*, model: str = "fake-hf") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 4,
        "temperature": 0.0,
    }


def test_processor_kwargs_force_fast_tokenizer_and_local_only() -> None:
    kwargs = processor_kwargs(
        SimpleNamespace(local_files_only=True),
    )

    assert kwargs == {"use_fast": True, "local_files_only": True}


def test_processor_kwargs_resolve_local_qwen_tokenizer_assets(tmp_path: Path) -> None:
    for filename in ("tokenizer.json", "vocab.json", "merges.txt"):
        (tmp_path / filename).write_text("{}", encoding="utf-8")

    kwargs = processor_kwargs(
        SimpleNamespace(local_files_only=True),
        model_name=str(tmp_path),
    )

    assert kwargs["tokenizer_file"] == str(tmp_path / "tokenizer.json")
    assert kwargs["vocab_file"] == str(tmp_path / "vocab.json")
    assert kwargs["merges_file"] == str(tmp_path / "merges.txt")


def test_bnb_training_prepare_freezes_without_peft_upcast() -> None:
    class _Param:
        requires_grad = True

    class _FakeModel:
        def __init__(self) -> None:
            self.param = _Param()
            self.input_grads_enabled = False
            self.gradient_checkpointing_kwargs = None

        def parameters(self):
            return [self.param]

        def enable_input_require_grads(self) -> None:
            self.input_grads_enabled = True

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    model = _FakeModel()

    prepared = prepare_model_for_quantized_training(
        model,
        config=SimpleNamespace(quantization="bnb_4bit"),
    )

    assert prepared is model
    assert model.param.requires_grad is False
    assert model.input_grads_enabled is True
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_hf_engine_batches_compatible_concurrent_requests() -> None:
    engine = FakeHFEngine(max_batch_size=2, max_queue_wait_ms=80)
    try:
        responses: list[Any] = []
        first = threading.Thread(target=lambda: responses.append(engine.chat(_request())))
        second = threading.Thread(target=lambda: responses.append(engine.chat(_request())))

        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)

        assert len(responses) == 2
        assert [len(batch) for batch in engine.batches] == [2]
    finally:
        engine.close()


def test_hf_engine_training_window_queues_generation() -> None:
    engine = FakeHFEngine(max_batch_size=1, max_queue_wait_ms=5)
    try:
        completed = threading.Event()

        with engine.exclusive_training():
            thread = threading.Thread(
                target=lambda: (engine.chat(_request()), completed.set())
            )
            thread.start()
            time.sleep(0.05)
            assert not completed.is_set()

        thread.join(timeout=2)
        assert completed.is_set()
        assert [len(batch) for batch in engine.batches] == [1]
    finally:
        engine.close()
