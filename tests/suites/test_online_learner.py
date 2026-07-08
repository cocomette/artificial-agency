"""Focused tests for online learner mechanics."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
from arcengine import GameAction

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ObservationRef,
    TransitionRecord,
)
from face_of_agi.environment.config import (
    BackboneRuntimeConfig,
    OnlineRuntimeConfig,
    PlannerRuntimeConfig,
    ReplayRuntimeConfig,
)
from face_of_agi.online.backbone import TransformersBackbone
from face_of_agi.online.learning import (
    EncodedTransition,
    OnlineWorldModel,
    ReplayTrainer,
    TransitionBuffer,
    ValueModel,
    transition_priority,
)
from face_of_agi.online.planner import ShortHorizonPlanner
from face_of_agi.orchestration.game_loop.state_machine import _validate_action_payload


def test_planner_expands_action6_coordinate_candidates() -> None:
    planner = _planner(
        planner_config=PlannerRuntimeConfig(
            candidate_count=8,
            coordinate_candidates=4,
            diagnostic_turns=1,
        )
    )

    result = planner.choose(
        features=(0.0, 0.0, 0.0, 0.0),
        action_space=(ActionSpec(action_id=GameAction.ACTION6),),
        real_turn_index=0,
    )

    assert len(result.candidates) == 4
    assert {
        (candidate.action.data["x"], candidate.action.data["y"])
        for candidate in result.candidates
    } == {(32, 32), (16, 16), (48, 16), (16, 48)}


def test_action6_validation_requires_arc_grid_integer_coordinates() -> None:
    _validate_action_payload(
        ActionSpec(action_id="ACTION6", data={"x": 0, "y": 63})
    )

    with pytest.raises(RuntimeError, match="missing 'y'"):
        _validate_action_payload(ActionSpec(action_id="ACTION6", data={"x": 0}))
    with pytest.raises(RuntimeError, match="ARC grid integer"):
        _validate_action_payload(
            ActionSpec(action_id="ACTION6", data={"x": 0.5, "y": 1})
        )
    with pytest.raises(RuntimeError, match="0..63"):
        _validate_action_payload(
            ActionSpec(action_id="ACTION6", data={"x": 0, "y": 64})
        )


def test_transition_buffer_samples_highest_priority_first() -> None:
    buffer = TransitionBuffer(max_size=4)
    low = _transition("low", changed_pixel_percent=0.0, prediction_error=0.1)
    high = _transition("high", changed_pixel_percent=80.0, prediction_error=3.0)

    buffer.add(low)
    buffer.add(high)

    assert [item.id for item in buffer.sample(2)] == ["high", "low"]
    assert transition_priority(high.record) > transition_priority(low.record)


def test_replay_trainer_honors_per_turn_update_budget() -> None:
    buffer = TransitionBuffer(max_size=4)
    transition = _transition("t1", changed_pixel_percent=10.0, prediction_error=1.0)
    buffer.add(transition)
    trainer = ReplayTrainer(
        config=ReplayRuntimeConfig(
            max_updates_per_turn=2,
            max_seconds_per_turn=10.0,
            solved_level_updates=3,
        ),
        buffer=buffer,
        world_model=OnlineWorldModel(OnlineRuntimeConfig(ensemble_size=2)),
        value_model=ValueModel(learning_rate=0.1),
    )

    stats = trainer.update_after_real_transition(
        transition,
        completed_level=False,
    )

    assert stats.real_update_count == 1
    assert stats.replay_update_count == 2
    assert stats.sampled_transition_ids == ("t1", "t1")


def test_planner_selects_highest_value_action() -> None:
    online_config = OnlineRuntimeConfig(ensemble_size=2, learning_rate=1.0)
    world_model = OnlineWorldModel(online_config)
    value_model = ValueModel(learning_rate=1.0)
    action1 = ActionSpec(action_id="ACTION1")
    action2 = ActionSpec(action_id="ACTION2")
    value_model.update(
        _transition(
            "rewarded",
            action=action2,
            changed_pixel_percent=0.0,
            score_delta=2.0,
        )
    )
    planner = ShortHorizonPlanner(
        config=PlannerRuntimeConfig(diagnostic_turns=0),
        world_model=world_model,
        value_model=value_model,
    )

    result = planner.choose(
        features=(0.0, 0.0, 0.0, 0.0),
        action_space=(action1, action2),
        real_turn_index=10,
    )

    assert result.action == action2


def test_qwen_backbone_builds_chat_template_and_pools_image_tokens() -> None:
    processor = _FakeQwenProcessor()
    model = _FakeQwenModel()
    backbone = _fake_qwen_backbone(
        processor=processor,
        model=model,
        feature_dim=2,
    )

    encoded = backbone.encode(
        Observation(id="obs-1", step=1, frame=((0, 1), (2, 3))),
    )

    assert encoded.features == (25.0, 45.0)
    assert encoded.metadata["feature_dim"] == 2
    assert encoded.metadata["raw_feature_dim"] == 4
    assert encoded.metadata["image_token_count"] == 2
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert processor.messages[0]["content"][1] == {
        "type": "text",
        "text": "Represent the current ARC frame.",
    }
    assert processor.kwargs == {
        "add_generation_prompt": False,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert model.call_kwargs["output_hidden_states"] is True
    assert model.call_kwargs["use_cache"] is False
    assert model.call_kwargs["return_dict"] is True


def test_qwen_backbone_requires_image_tokens() -> None:
    backbone = _fake_qwen_backbone(
        processor=_FakeQwenProcessor(input_ids=[[1, 2, 3]]),
        model=_FakeQwenModel(),
        feature_dim=2,
    )

    with pytest.raises(RuntimeError, match="image tokens"):
        backbone.encode(Observation(id="obs-1", step=1, frame=((0, 1), (2, 3))))


def test_qwen_backbone_routes_model_and_processor_kwargs(monkeypatch) -> None:
    captures: dict[str, dict] = {}

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captures["processor"] = {"path": str(path), **kwargs}
            return _FakeQwenProcessor()

    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            raise AssertionError("Qwen path must not use AutoImageProcessor")

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captures["model"] = {"path": str(path), **kwargs}
            return _LoadableFakeQwenModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
    fake_transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())

    TransformersBackbone(
        BackboneRuntimeConfig(
            backend="transformers",
            model_family="qwen3_5_moe_multimodal",
            model_path="unused-qwen",
            local_files_only=False,
            representation_layer="image_tokens_mean",
            processor_kwargs={"processor_only": True},
            model_kwargs={"device_map": "auto", "model_only": True},
        ),
        feature_dim=4,
    )

    assert captures["processor"] == {
        "path": "unused-qwen",
        "local_files_only": False,
        "processor_only": True,
    }
    assert captures["model"] == {
        "path": "unused-qwen",
        "local_files_only": False,
        "device_map": "auto",
        "model_only": True,
    }


def _planner(*, planner_config: PlannerRuntimeConfig) -> ShortHorizonPlanner:
    online_config = OnlineRuntimeConfig(ensemble_size=2)
    return ShortHorizonPlanner(
        config=planner_config,
        world_model=OnlineWorldModel(online_config),
        value_model=ValueModel(learning_rate=0.1),
    )


def _transition(
    transition_id: str,
    *,
    action: ActionSpec | None = None,
    changed_pixel_percent: float,
    prediction_error: float | None = None,
    score_delta: float | None = None,
) -> EncodedTransition:
    selected_action = action or ActionSpec(action_id="ACTION1")
    record = TransitionRecord(
        previous_observation_ref=ObservationRef(memory="state", id=f"{transition_id}-a"),
        next_observation_ref=ObservationRef(memory="state", id=f"{transition_id}-b"),
        action=selected_action,
        controllable=True,
        changed_pixel_percent=changed_pixel_percent,
        score_delta=score_delta,
        prediction_error=prediction_error,
    )
    return EncodedTransition(
        id=transition_id,
        previous=(0.0, 0.0, 0.0, 0.0),
        action=selected_action,
        next=(1.0, 0.0, 0.0, 0.0),
        record=record,
        priority=transition_priority(record),
    )


def _fake_qwen_backbone(
    *,
    processor: "_FakeQwenProcessor",
    model: "_FakeQwenModel",
    feature_dim: int,
) -> TransformersBackbone:
    config = BackboneRuntimeConfig(
        backend="transformers",
        model_family="qwen3_5_moe_multimodal",
        model_path="unused-qwen",
        local_files_only=False,
        representation_layer="image_tokens_mean",
        feature_prompt="Represent the current ARC frame.",
    )
    backbone = object.__new__(TransformersBackbone)
    backbone.config = config
    backbone.feature_dim = feature_dim
    backbone.model_path = Path(config.model_path)
    backbone.processor_path = Path(config.model_path)
    backbone._torch = _FakeTorch()
    backbone._processor = processor
    backbone._model = model
    backbone._device = "cpu"
    return backbone


class _FakeTorch:
    def no_grad(self):
        return nullcontext()

    def inference_mode(self):
        return nullcontext()


def _fake_torch_module() -> ModuleType:
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(is_available=lambda: False)
    module.device = lambda value: value
    module.float32 = "float32"
    module.float16 = "float16"
    module.bfloat16 = "bfloat16"
    module.no_grad = lambda: nullcontext()
    module.inference_mode = lambda: nullcontext()
    return module


class _FakeQwenProcessor:
    def __init__(self, *, input_ids=None) -> None:
        self.input_ids = input_ids or [[1, 42, 42, 2]]
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {"input_ids": self.input_ids, "pixel_values": [[0.0]]}


class _FakeQwenModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(image_token_id=42)
        self.call_kwargs = {}

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return SimpleNamespace(
            hidden_states=[
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [10.0, 20.0, 30.0, 40.0],
                        [30.0, 40.0, 50.0, 60.0],
                        [7.0, 8.0, 9.0, 10.0],
                    ]
                ]
            ]
        )


class _LoadableFakeQwenModel(_FakeQwenModel):
    device = "cpu"

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return ()
