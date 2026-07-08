from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import pytest

torch = pytest.importorskip("torch")

from face_of_agi.contracts import ActionSpec

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "single_vlm_lora"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from run_single_vlm_lora import (  # noqa: E402
    FinalizedTurn,
    PendingTurn,
    _action_conditioned_learning_progress_baseline,
    _capture_world_snapshot,
    _discounted_window_weights,
    _frame_change_diagnostics,
    _learning_progress_rate_reference,
    _next_learning_progress_rate_baseline,
    _normalized_policy_advantage,
    _observation_has_usable_frame,
    _policy_advantage_normalization_denominator,
    _policy_return_discount,
    _policy_return_horizon,
    _policy_update_accumulation_steps,
    _policy_log_probs,
    _policy_probe,
    _ppo_clipped_policy_loss,
    _raw_policy_advantage,
    _run_policy_batch_updates,
    _run_world_updates,
    _selected_learning_progress_rate_baseline,
    _temporarily_load_world_snapshot,
    _weighted_learning_progress,
    run_experiment,
)
from single_vlm_arc.actions import (  # noqa: E402
    masked_action_probabilities,
    mask_action_logits,
    select_action,
    valid_action_mask,
)
from single_vlm_arc.config import apply_cli_overrides, load_config  # noqa: E402
from single_vlm_arc.history import (  # noqa: E402
    RollingHistory,
    Transition,
    decision_frame,
    decision_frame_with_metadata,
)
from single_vlm_arc.logging import FramePredictionLogger, LatentPredictionLogger  # noqa: E402
from single_vlm_arc.model import (  # noqa: E402
    POLICY_ADAPTER_NAME,
    WORLD_ADAPTER_NAME,
    FakeSingleVLMPolicy,
    SingleVLMPolicy,
    resolve_lora_target_modules,
)
from single_vlm_arc.online_update import (  # noqa: E402
    frame_to_palette_tensor,
    latent_changed_patch_mask,
    latent_grid_loss,
    next_frame_loss,
    policy_parameters,
    trainable_parameters,
    world_model_parameters,
)
from single_vlm_arc.rewards import compute_reward  # noqa: E402


def test_config_loads_nested_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dry_run": True,
                "frame_history_n": 2,
                "model": {
                    "backend": "fake",
                    "model_id": "fake",
                    "image_size": "32x32",
                },
                "environment": {"max_turns": 1},
                "logging": {"output_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dry_run is True
    assert config.frame_history_n == 2
    assert config.model.backend == "fake"
    assert config.model.image_size == (32, 32)
    assert config.model.gradient_checkpointing is False
    assert config.model.lora.r == 16
    assert config.environment.max_turns == 1
    assert config.palette_size == 16
    assert config.rewards.learning_progress_weight == 10.0
    assert config.update.learning_progress_horizon == 4
    assert config.update.learning_progress_discount == 0.8
    assert config.update.learning_progress_rate_beta == 0.5
    assert config.update.world_loss_mode == "pixel_ce"
    assert config.update.latent_loss_weight == 1.0
    assert config.update.latent_changed_patch_weight == 6.0
    assert config.update.latent_huber_beta == 1.0
    assert config.update.latent_cosine_loss_weight == 0.1
    assert config.update.latent_cosine_min_delta_norm == 1e-4
    assert config.update.latent_learning_progress_normalization is True
    assert config.update.latent_learning_progress_normalization_floor == 0.01
    assert config.update.policy_warmup_turns == 3
    assert config.update.policy_clip_epsilon == 0.2
    assert config.update.policy_advantage_baseline == "ema"
    assert config.update.policy_advantage_normalization == "none"
    assert config.update.policy_advantage_normalization_beta == 0.5
    assert config.update.policy_advantage_normalization_floor == 0.01
    assert config.model.hidden_pooling == "last"
    assert config.model.lora.separate_role_adapters is True
    assert config.update.action_conditioned_learning_progress_baseline is False
    assert config.update.residual_frame_prediction is False
    assert config.update.residual_frame_logit_bias == 4.0
    assert config.update.policy_update_accumulation_steps == 1
    assert config.update.policy_learning_progress_return_horizon == 12
    assert config.update.policy_learning_progress_return_discount == 0.93
    assert config.update.policy_adapter_trainable is True
    assert config.logging.save_video is False
    assert config.logging.video_fps == 4
    assert config.logging.video_frame_scale == 8
    assert config.logging.save_frame_predictions is True
    assert config.logging.frame_prediction_save_every == 1
    assert config.logging.frame_prediction_frame_scale == 8
    assert config.logging.save_latent_predictions is True
    assert config.logging.latent_prediction_save_every == 1
    assert config.logging.latent_prediction_frame_scale == 16


def test_cli_overrides_support_latent_prediction_options(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("logging:\n  output_dir: out\n", encoding="utf-8")
    config = load_config(config_path)

    apply_cli_overrides(
        config,
        world_loss_mode="latent_grid",
        latent_loss_weight=2.0,
        latent_changed_patch_weight=4.0,
        latent_huber_beta=0.5,
        latent_cosine_loss_weight=0.25,
        latent_cosine_min_delta_norm=0.01,
        latent_learning_progress_normalization=False,
        latent_learning_progress_normalization_floor=0.02,
        save_latent_predictions=False,
        latent_prediction_save_every=3,
        latent_prediction_frame_scale=5,
    )

    assert config.update.world_loss_mode == "latent_grid"
    assert config.update.latent_loss_weight == 2.0
    assert config.update.latent_changed_patch_weight == 4.0
    assert config.update.latent_huber_beta == 0.5
    assert config.update.latent_cosine_loss_weight == 0.25
    assert config.update.latent_cosine_min_delta_norm == 0.01
    assert config.update.latent_learning_progress_normalization is False
    assert config.update.latent_learning_progress_normalization_floor == 0.02
    assert config.logging.save_latent_predictions is False
    assert config.logging.latent_prediction_save_every == 3
    assert config.logging.latent_prediction_frame_scale == 5


def test_modal_a100_config_loads_with_selectable_qwen_model() -> None:
    config = load_config(
        EXPERIMENT_ROOT / "configs" / "qwen3_vl_4b_a100_40gb.yaml"
    )

    assert config.model.model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.model.dtype == "bf16"
    assert config.model.attn_implementation == "sdpa"
    assert config.environment.max_turns == 20
    assert config.rewards.learning_progress_weight == 10.0
    assert config.update.learning_progress_horizon == 4
    assert config.update.learning_progress_discount == 0.8
    assert config.update.learning_progress_rate_beta == 0.5
    assert config.update.policy_warmup_turns == 3
    assert config.update.policy_clip_epsilon == 0.2


def test_gemma4_26b_config_enables_gradient_checkpointing() -> None:
    config = load_config(
        EXPERIMENT_ROOT / "configs" / "gemma4_26b_a100_40gb.yaml"
    )

    assert config.model.model_id == "google/gemma-4-26B-A4B-it"
    assert config.model.gradient_checkpointing is True


def test_policy_log_probs_use_sampling_temperature() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(action_selection="sample", temperature=0.5),
    )
    action_space = (
        ActionSpec("ACTION1"),
        ActionSpec("ACTION2"),
    )
    logits = torch.tensor([0.0, 0.0, 1.0, -1.0, 9.0, 9.0, 9.0, 9.0])

    log_probs = _policy_log_probs(mask_action_logits(logits, action_space), config)
    expected = torch.log_softmax(
        torch.tensor([0.0, 1.0]) / 0.5,
        dim=-1,
    )

    assert log_probs[1].item() == pytest.approx(expected[0].item())
    assert log_probs[2].item() == pytest.approx(expected[1].item())


def test_ppo_clipped_policy_loss_uses_old_logprob_ratio() -> None:
    new_log_probability = torch.tensor(float(np.log(1.5)), requires_grad=True)

    loss = _ppo_clipped_policy_loss(
        new_log_probability=new_log_probability,
        old_log_probability=0.0,
        advantage=2.0,
        clip_epsilon=0.2,
    )

    assert loss.item() == pytest.approx(-2.4)


def test_action_masking_and_action6_coordinates() -> None:
    action_space = (
        ActionSpec("ACTION1"),
        ActionSpec("ACTION6"),
    )
    logits = torch.tensor([0.0, 0.1, 5.0, 6.0, 7.0, 8.0, 0.2, 9.0])
    coord_logits = torch.zeros(128)
    coord_logits[6] = 10.0
    coord_logits[127] = 10.0

    mask = valid_action_mask(action_space)
    masked = mask_action_logits(logits, action_space)
    selected = select_action(
        action_logits=logits,
        coord_logits=coord_logits,
        action_space=action_space,
        mode="argmax",
        temperature=1.0,
    )

    assert mask == [False, True, False, False, False, False, True, False]
    assert int(masked.argmax().item()) == 6
    assert selected.action.name == "ACTION6"
    assert selected.action.data == {"x": 6, "y": 63}


def test_masked_action_probabilities_zero_invalid_actions() -> None:
    action_space = (
        ActionSpec("ACTION1"),
        ActionSpec("ACTION6"),
    )
    logits = torch.tensor([9.0, 0.1, 5.0, 6.0, 7.0, 8.0, 0.2, 9.0])

    probabilities = masked_action_probabilities(logits, action_space)

    assert probabilities["ACTION1"] > 0.0
    assert probabilities["ACTION6"] > 0.0
    assert probabilities["RESET"] == 0.0
    assert probabilities["ACTION7"] == 0.0
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_zero_initialized_action_head_starts_uniform_after_masking() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    action_space = (
        ActionSpec("ACTION1"),
        ActionSpec("ACTION2"),
        ActionSpec("ACTION6"),
    )

    assert isinstance(model.action_norm, torch.nn.LayerNorm)
    probabilities = masked_action_probabilities(
        model("prompt", []).action_logits,
        action_space,
    )

    assert probabilities["ACTION1"] == pytest.approx(1.0 / 3.0)
    assert probabilities["ACTION2"] == pytest.approx(1.0 / 3.0)
    assert probabilities["ACTION6"] == pytest.approx(1.0 / 3.0)
    assert probabilities["RESET"] == 0.0


def test_policy_and_world_parameter_sets_are_disjoint() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)

    policy_ids = {id(parameter) for parameter in policy_parameters(model)}
    world_ids = {id(parameter) for parameter in world_model_parameters(model)}
    trainable_ids = {id(parameter) for parameter in trainable_parameters(model)}
    expected_policy_ids = {
        id(parameter)
        for module in (model.action_norm, model.action_head)
        for parameter in module.parameters()
    }
    expected_world_ids = {
        id(parameter)
        for module in (
            model.action_condition,
            model.coord_x_condition,
            model.coord_y_condition,
            model.coord_head,
            model.frame_head,
            model.latent_patch_position,
            model.latent_delta_head,
        )
        for parameter in module.parameters()
    }
    expected_world_ids.add(id(model.hidden))

    assert policy_ids == expected_policy_ids
    assert expected_world_ids.issubset(world_ids)
    assert policy_ids.isdisjoint(world_ids)
    assert policy_ids | world_ids == trainable_ids


def test_role_adapter_parameter_sets_include_named_lora_adapters(
    tmp_path: Path,
) -> None:
    model = _toy_role_adapter_model()

    base_parameters = dict(model.base_model.named_parameters())
    policy_adapter_ids = {
        id(parameter)
        for name, parameter in base_parameters.items()
        if POLICY_ADAPTER_NAME in name.split(".")
    }
    world_adapter_ids = {
        id(parameter)
        for name, parameter in base_parameters.items()
        if WORLD_ADAPTER_NAME in name.split(".")
    }
    policy_ids = {id(parameter) for parameter in policy_parameters(model)}
    world_ids = {id(parameter) for parameter in world_model_parameters(model)}
    trainable_ids = {id(parameter) for parameter in trainable_parameters(model)}

    assert policy_adapter_ids.issubset(policy_ids)
    assert world_adapter_ids.issubset(world_ids)
    assert policy_ids.isdisjoint(world_ids)
    assert policy_ids | world_ids == trainable_ids

    state = model._trainable_state_dict()
    assert any(".world." in key for key in state)
    assert any(".policy." in key for key in state)

    assert model.base_model.active_adapters == [POLICY_ADAPTER_NAME]
    with model.use_world_adapter():
        assert model.base_model.active_adapters == [WORLD_ADAPTER_NAME]
    assert model.base_model.active_adapters == [POLICY_ADAPTER_NAME]

    model.save_adapter(tmp_path / "final_adapter")
    selected = json.loads(
        (tmp_path / "final_adapter" / "selected_adapters.json").read_text(
            encoding="utf-8",
        )
    )
    assert selected == [WORLD_ADAPTER_NAME, POLICY_ADAPTER_NAME]


def test_policy_adapter_trainable_flag_excludes_policy_lora_parameters() -> None:
    model = _toy_role_adapter_model()
    model.policy_adapter_trainable = False

    base_parameters = dict(model.base_model.named_parameters())
    policy_adapter_ids = {
        id(parameter)
        for name, parameter in base_parameters.items()
        if POLICY_ADAPTER_NAME in name.split(".")
    }
    policy_ids = {id(parameter) for parameter in policy_parameters(model)}
    head_ids = {
        id(parameter)
        for module in (model.action_norm, model.action_head)
        for parameter in module.parameters()
    }

    assert policy_ids == head_ids
    assert policy_ids.isdisjoint(policy_adapter_ids)


def test_shared_lora_config_disables_role_adapters(tmp_path: Path) -> None:
    config_path = tmp_path / "shared.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "backend": "fake",
                    "model_id": "fake",
                    "lora": {
                        "enabled": True,
                        "separate_role_adapters": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.lora.separate_role_adapters is False


def test_attention_hidden_pooling_head_is_world_parameter() -> None:
    model = FakeSingleVLMPolicy(
        hidden_size=16,
        palette_size=10,
        hidden_pooling="attention",
    )
    output = model("prompt", [])
    policy_ids = {id(parameter) for parameter in policy_parameters(model)}
    world_ids = {id(parameter) for parameter in world_model_parameters(model)}

    assert model.hidden_pool is not None
    assert output.action_logits.shape == (1, 8)
    assert output.action_frame_logits.shape == (1, 8, 10, 64, 64)
    assert all(id(parameter) in world_ids for parameter in model.hidden_pool.parameters())
    assert all(id(parameter) not in policy_ids for parameter in model.hidden_pool.parameters())


def test_attention_hidden_pooling_accepts_bfloat16_hidden_states() -> None:
    model = FakeSingleVLMPolicy(
        hidden_size=16,
        palette_size=10,
        hidden_pooling="attention",
    )
    hidden_sequence = torch.zeros(1, 3, 16, dtype=torch.bfloat16)

    pooled = model._pool_hidden_sequence(hidden_sequence)

    assert pooled.shape == (1, 16)
    assert pooled.dtype == model.hidden_pool.weight.dtype


def test_fake_model_exposes_latent_grid_and_selected_prediction() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[0:8, 0:8] = 3

    latent_grid = model.frame_latent_grid(frame)
    predicted = model.predict_latent_delta(
        "prompt",
        [frame],
        action_index=6,
        selected_x=12,
        selected_y=20,
    )
    output, selected = model.forward_with_latent_delta(
        "prompt",
        [frame],
        action_index=6,
        selected_x=12,
        selected_y=20,
        include_frame_logits=False,
    )

    assert latent_grid.shape == (8, 8, 16)
    assert predicted.shape == (1, 8, 8, 16)
    assert selected.shape == (1, 8, 8, 16)
    assert output.action_frame_logits is None
    assert output.action_logits.shape == (1, 8)


def test_latent_coordinate_conditioning_changes_selected_prediction() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)

    left = model.predict_latent_delta(
        "prompt",
        [],
        action_index=6,
        selected_x=1,
        selected_y=1,
    )
    right = model.predict_latent_delta(
        "prompt",
        [],
        action_index=6,
        selected_x=40,
        selected_y=40,
    )

    assert not torch.allclose(left, right)


def test_world_snapshot_excludes_policy_parameters() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)

    snapshot = _capture_world_snapshot(model)

    assert len(snapshot) == len(world_model_parameters(model))
    assert sum(tensor.numel() for tensor in snapshot) == sum(
        parameter.numel() for parameter in world_model_parameters(model)
    )
    assert sum(tensor.numel() for tensor in snapshot) < sum(
        parameter.numel() for parameter in trainable_parameters(model)
    )


def test_world_snapshot_context_restores_current_parameters() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    snapshot = _capture_world_snapshot(model)
    world_params = world_model_parameters(model)

    with torch.no_grad():
        for parameter in world_params:
            parameter.add_(1.0)
    modified = _snapshot_parameters(world_params)
    with _temporarily_load_world_snapshot(model, snapshot):
        for parameter, snapshot_value in zip(world_params, snapshot):
            assert torch.allclose(parameter.detach().cpu(), snapshot_value)

    assert not _any_parameter_changed(modified, world_params)


def test_discounted_window_learning_progress_formula() -> None:
    weights = _discounted_window_weights(4, 0.8)
    prior_losses = [10.0, 8.0, 4.0, 2.0]
    current_losses = [9.0, 7.5, 5.0, 1.0]

    progress = _weighted_learning_progress(prior_losses, current_losses, weights)

    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx(
        [
            1.0 / 2.952,
            0.8 / 2.952,
            0.64 / 2.952,
            0.512 / 2.952,
        ]
    )
    assert progress == pytest.approx(
        sum(
            weight * (prior - current)
            for prior, current, weight in zip(prior_losses, current_losses, weights)
        )
    )


def test_discounted_policy_lp_return_formula() -> None:
    config = SimpleNamespace(
        update=SimpleNamespace(
            policy_learning_progress_return_horizon=4,
            policy_learning_progress_return_discount=0.5,
        )
    )
    rates = [0.2, 0.1, -0.1]
    weights = _discounted_window_weights(len(rates), _policy_return_discount(config))
    policy_return = sum(weight * rate for weight, rate in zip(weights, rates))

    assert _policy_return_horizon(config) == 4
    assert _policy_return_discount(config) == pytest.approx(0.5)
    assert weights == pytest.approx([1 / 1.75, 0.5 / 1.75, 0.25 / 1.75])
    assert policy_return == pytest.approx(
        0.2 * (1 / 1.75)
        + 0.1 * (0.5 / 1.75)
        - 0.1 * (0.25 / 1.75)
    )


def test_learning_progress_rate_uses_fast_ema_reference() -> None:
    first_raw_lp = 0.2
    second_raw_lp = 0.1

    first_reference = _learning_progress_rate_reference(
        learning_progress_rate_baseline=None,
        raw_learning_progress=first_raw_lp,
    )
    first_baseline = _next_learning_progress_rate_baseline(
        learning_progress_rate_baseline=None,
        raw_learning_progress=first_raw_lp,
        beta=0.5,
    )
    second_reference = _learning_progress_rate_reference(
        learning_progress_rate_baseline=first_baseline,
        raw_learning_progress=second_raw_lp,
    )
    second_rate = second_raw_lp - second_reference
    second_baseline = _next_learning_progress_rate_baseline(
        learning_progress_rate_baseline=first_baseline,
        raw_learning_progress=second_raw_lp,
        beta=0.5,
    )

    assert first_reference == pytest.approx(first_raw_lp)
    assert first_raw_lp - first_reference == pytest.approx(0.0)
    assert second_rate == pytest.approx(-0.1)
    assert second_baseline == pytest.approx(0.15)


def test_action_conditioned_learning_progress_baseline_selects_per_action() -> None:
    config = SimpleNamespace(
        update=SimpleNamespace(action_conditioned_learning_progress_baseline=True),
    )
    baselines = {"ACTION1": 0.2}

    assert _action_conditioned_learning_progress_baseline(config) is True
    assert _selected_learning_progress_rate_baseline(
        selected_action_name="ACTION1",
        learning_progress_rate_baseline=0.8,
        action_learning_progress_rate_baselines=baselines,
        action_conditioned=True,
    ) == pytest.approx(0.2)
    assert _selected_learning_progress_rate_baseline(
        selected_action_name="ACTION2",
        learning_progress_rate_baseline=0.8,
        action_learning_progress_rate_baselines=baselines,
        action_conditioned=True,
    ) is None
    assert _selected_learning_progress_rate_baseline(
        selected_action_name="ACTION1",
        learning_progress_rate_baseline=0.8,
        action_learning_progress_rate_baselines=baselines,
        action_conditioned=False,
    ) == pytest.approx(0.8)


def test_zero_policy_advantage_baseline_preserves_lp_rate_sign() -> None:
    config = SimpleNamespace(
        update=SimpleNamespace(
            policy_advantage_normalization="ema_abs",
            policy_advantage_normalization_floor=0.01,
        )
    )
    raw_advantage = _raw_policy_advantage(
        policy_intrinsic_reward=-0.006,
        intrinsic_reward_baseline=-0.8,
        baseline_mode="zero",
    )
    denominator = _policy_advantage_normalization_denominator(
        policy_signal_abs_baseline=0.02,
        policy_intrinsic_reward=-0.006,
        config=config,
    )
    normalized = _normalized_policy_advantage(
        raw_policy_advantage=raw_advantage,
        denominator=denominator,
        normalization="ema_abs",
    )

    assert raw_advantage == pytest.approx(-0.006)
    assert denominator == pytest.approx(0.02)
    assert normalized == pytest.approx(-0.3)


def test_policy_advantage_normalization_uses_current_signal_for_spikes() -> None:
    config = SimpleNamespace(
        update=SimpleNamespace(
            policy_advantage_normalization="ema_abs",
            policy_advantage_normalization_floor=0.01,
        )
    )

    denominator = _policy_advantage_normalization_denominator(
        policy_signal_abs_baseline=0.008,
        policy_intrinsic_reward=3.6,
        config=config,
    )

    assert denominator == pytest.approx(3.6)


def test_ema_policy_advantage_baseline_can_flip_small_negative_lp_rate() -> None:
    raw_advantage = _raw_policy_advantage(
        policy_intrinsic_reward=-0.006,
        intrinsic_reward_baseline=-0.8,
        baseline_mode="ema",
    )

    assert raw_advantage == pytest.approx(0.794)


def test_separate_optimizers_only_step_their_parameter_sets() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    policy_optimizer = torch.optim.AdamW(policy_parameters(model), lr=1e-2)
    world_optimizer = torch.optim.AdamW(world_model_parameters(model), lr=1e-2)
    frame = np.zeros((64, 64), dtype=np.uint8)

    world_before = _snapshot_parameters(world_model_parameters(model))
    policy_before = _snapshot_parameters(policy_parameters(model))
    policy_optimizer.zero_grad(set_to_none=True)
    policy_loss = -model("prompt", []).action_logits[0, 1]
    policy_loss.backward()
    policy_optimizer.step()

    assert _any_parameter_changed(policy_before, policy_parameters(model))
    assert not _any_parameter_changed(world_before, world_model_parameters(model))

    world_before = _snapshot_parameters(world_model_parameters(model))
    policy_before = _snapshot_parameters(policy_parameters(model))
    world_optimizer.zero_grad(set_to_none=True)
    world_loss = next_frame_loss(
        model("prompt", []).action_frame_logits,
        frame,
        palette_size=10,
        action_index=1,
    )
    world_loss.backward()
    world_optimizer.step()

    assert _any_parameter_changed(world_before, world_model_parameters(model))
    assert not _any_parameter_changed(policy_before, policy_parameters(model))


def test_world_update_changes_world_scope_without_policy_drift() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    optimizer = torch.optim.AdamW(world_model_parameters(model), lr=1e-2)
    pending = _make_pending_turn()
    config = _training_config()

    world_before = _snapshot_parameters(world_model_parameters(model))
    policy_before = _snapshot_parameters(policy_parameters(model))
    probe_before = _policy_probe(model=model, pending=pending, config=config)

    _run_world_updates(
        model=model,
        optimizer=optimizer,
        pending=pending,
        config=config,
    )

    probe_after = _policy_probe(model=model, pending=pending, config=config)
    assert _any_parameter_changed(world_before, world_model_parameters(model))
    assert not _any_parameter_changed(policy_before, policy_parameters(model))
    assert probe_after["selected_action_probability"] == pytest.approx(
        probe_before["selected_action_probability"]
    )
    assert probe_after["max_action_probability"] == pytest.approx(
        probe_before["max_action_probability"]
    )


def test_policy_batch_update_changes_policy_scope_without_world_step() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    optimizer = torch.optim.AdamW(policy_parameters(model), lr=1e-2)
    pending = _make_pending_turn()
    config = _training_config()
    finalized = FinalizedTurn(
        pending=pending,
        payload={},
        include_policy_update=True,
        policy_advantage=1.0,
        next_intrinsic_reward_baseline=None,
        next_learning_progress_rate_baseline=None,
        next_policy_signal_abs_baseline=None,
        next_action_learning_progress_rate_baselines={},
    )

    world_before = _snapshot_parameters(world_model_parameters(model))
    policy_before = _snapshot_parameters(policy_parameters(model))

    _run_policy_batch_updates(
        model=model,
        optimizer=optimizer,
        finalized_turns=[finalized],
        config=config,
    )

    assert _any_parameter_changed(policy_before, policy_parameters(model))
    assert not _any_parameter_changed(world_before, world_model_parameters(model))


class _ToyPeftBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Module()
        self.proj.lora_A = torch.nn.ModuleDict(
            {
                WORLD_ADAPTER_NAME: torch.nn.Linear(2, 2, bias=False),
                POLICY_ADAPTER_NAME: torch.nn.Linear(2, 2, bias=False),
            }
        )
        self.active_adapters = [POLICY_ADAPTER_NAME]

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        if isinstance(adapter_name, list):
            self.active_adapters = list(adapter_name)
        else:
            self.active_adapters = [str(adapter_name)]

    def save_pretrained(self, path: str | Path, **kwargs: object) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        selected_adapters = kwargs.get("selected_adapters")
        (save_path / "selected_adapters.json").write_text(
            json.dumps(selected_adapters),
            encoding="utf-8",
        )


def _toy_role_adapter_model() -> SingleVLMPolicy:
    model = SingleVLMPolicy.__new__(SingleVLMPolicy)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.processor = None
    model.base_model = _ToyPeftBackbone()
    model.role_adapters_enabled = True
    model.world_adapter_name = WORLD_ADAPTER_NAME
    model.policy_adapter_name = POLICY_ADAPTER_NAME
    model.action_runtime_adapter_name = POLICY_ADAPTER_NAME
    model._init_heads(2, 3)
    return model


def _make_pending_turn() -> PendingTurn:
    current = np.zeros((64, 64), dtype=np.uint8)
    target = current.copy()
    target[8:16, 16:24] = 2
    action_space = (
        ActionSpec("ACTION1"),
        ActionSpec("ACTION2"),
    )
    transition = Transition(
        turn=0,
        observation=SimpleNamespace(id="obs0", frame=current),
        action=ActionSpec("ACTION1"),
        next_observation=SimpleNamespace(id="obs1", frame=target),
        action_index=1,
        log_probability=0.0,
        prediction_loss=0.0,
        reward=0.0,
    )
    return PendingTurn(
        transition=transition,
        prompt="prompt",
        images=[current],
        action_space=action_space,
        selected_action_index=1,
        selected_x=None,
        selected_y=None,
        score_delta=0.0,
        interaction_elapsed_seconds=0.0,
        model_input={},
        model_output={
            "selected_action_name": "ACTION1",
            "selected_action_probability": 0.5,
            "selected_action_log_probability": 0.0,
        },
        current_frame=current,
        next_frame=target,
        frame_diagnostics={
            "decision_frame_index": 0,
            "decision_frame_count": 1,
            "next_decision_frame_index": 0,
            "next_decision_frame_count": 1,
            "frame_changed_pixels": 64,
            "frame_changed_fraction": 64 / (64 * 64),
        },
    )


def _training_config() -> SimpleNamespace:
    return SimpleNamespace(
        palette_size=10,
        frame_size=(64, 64),
        model=SimpleNamespace(action_selection="sample", temperature=1.0),
        update=SimpleNamespace(
            update_steps=1,
            learning_rate=1e-2,
            gradient_clip_norm=1.0,
            world_loss_mode="pixel_ce",
            next_frame_loss_weight=1.0,
            latent_loss_weight=1.0,
            latent_changed_patch_weight=6.0,
            latent_huber_beta=1.0,
            latent_cosine_loss_weight=0.0,
            latent_cosine_min_delta_norm=1e-4,
            latent_learning_progress_normalization=True,
            latent_learning_progress_normalization_floor=0.01,
            coord_loss_weight=0.01,
            policy_loss_weight=1.0,
            policy_clip_epsilon=0.2,
            policy_update_accumulation_steps=1,
        ),
    )


def _snapshot_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _any_parameter_changed(
    before: list[torch.Tensor],
    after: list[torch.nn.Parameter],
) -> bool:
    return any(
        not torch.allclose(previous, current.detach())
        for previous, current in zip(before, after)
    )


def test_recent_frame_ids_are_chronological_observations() -> None:
    history = RollingHistory(frame_history_n=2)
    obs0 = SimpleNamespace(id="obs0", frame=np.zeros((64, 64), dtype=np.uint8))
    obs1 = SimpleNamespace(id="obs1", frame=np.ones((64, 64), dtype=np.uint8))
    history.append(
        Transition(
            turn=0,
            observation=obs0,
            action=ActionSpec("ACTION1"),
            next_observation=obs1,
            action_index=1,
            log_probability=0.0,
            prediction_loss=0.0,
            reward=0.0,
        )
    )

    assert history.recent_frame_observation_ids(obs1) == ["obs0", "obs1"]


def test_decision_frame_selects_last_animation_frame() -> None:
    first = np.zeros((64, 64), dtype=np.uint8)
    last = np.ones((64, 64), dtype=np.uint8)
    observation = SimpleNamespace(frame=first, frames=(first, last))

    frame, index, count = decision_frame_with_metadata(observation)

    assert np.array_equal(frame, last)
    assert decision_frame(observation) is last
    assert index == 1
    assert count == 2


def test_history_uses_canonical_decision_frames() -> None:
    history = RollingHistory(frame_history_n=2)
    first0 = np.zeros((64, 64), dtype=np.uint8)
    last0 = np.ones((64, 64), dtype=np.uint8)
    first1 = np.full((64, 64), 2, dtype=np.uint8)
    last1 = np.full((64, 64), 3, dtype=np.uint8)
    obs0 = SimpleNamespace(id="obs0", frame=first0, frames=(first0, last0))
    obs1 = SimpleNamespace(id="obs1", frame=first1, frames=(first1, last1))
    history.append(
        Transition(
            turn=0,
            observation=obs0,
            action=ActionSpec("ACTION1"),
            next_observation=obs1,
            action_index=1,
            log_probability=0.0,
            prediction_loss=0.0,
            reward=0.0,
        )
    )

    frames = history.recent_frames(decision_frame(obs1))

    assert np.array_equal(frames[0], last0)
    assert np.array_equal(frames[1], last1)


def test_observation_frame_guard_rejects_terminal_no_frame_observations() -> None:
    assert _observation_has_usable_frame(
        SimpleNamespace(frame=np.zeros((64, 64), dtype=np.uint8))
    )
    assert _observation_has_usable_frame(
        SimpleNamespace(
            frame=None,
            frames=(np.zeros((64, 64), dtype=np.uint8),),
        )
    )
    assert not _observation_has_usable_frame(SimpleNamespace(frame=None))
    assert not _observation_has_usable_frame(SimpleNamespace(frame=np.asarray(None)))


def test_frame_change_diagnostics_counts_palette_differences() -> None:
    current = np.zeros((4, 4), dtype=np.uint8)
    changed = np.zeros((4, 4), dtype=np.uint8)
    changed[0, 0] = 1
    changed[1, 1] = 2
    config = SimpleNamespace(palette_size=16, frame_size=(4, 4))

    diagnostics = _frame_change_diagnostics(current, changed, config=config)
    unchanged = _frame_change_diagnostics(current, current, config=config)

    assert diagnostics["frame_changed_pixels"] == 2
    assert diagnostics["frame_changed_fraction"] == pytest.approx(2 / 16)
    assert unchanged["frame_changed_pixels"] == 0
    assert unchanged["frame_changed_fraction"] == 0.0


def test_prompt_includes_action_glossary_and_frame_order() -> None:
    prompt = RollingHistory(frame_history_n=2).build_prompt(
        game_id="test-game",
        turn=3,
        valid_actions=("ACTION1", "ACTION4"),
    )

    assert "Images are ordered chronologically from oldest to newest" in prompt
    assert "ACTION1=up" in prompt
    assert "ACTION2=down" in prompt
    assert "ACTION3=left" in prompt
    assert "ACTION4=right" in prompt
    assert "valid_actions: ACTION1, ACTION4" in prompt
    assert "after the selected action" in prompt


def test_next_frame_loss_accepts_palette_frames() -> None:
    logits = torch.zeros(1, 10, 64, 64)
    frame = np.zeros((64, 64), dtype=np.uint8)

    loss = next_frame_loss(logits, frame, palette_size=10)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_frame_to_palette_tensor_preserves_arc_agi_16_color_ids() -> None:
    frame = np.array([[0, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int8)

    target = frame_to_palette_tensor(frame, palette_size=16, frame_size=(8, 1))

    assert target.tolist() == [[0, 9, 10, 11, 12, 13, 14, 15]]
    with pytest.raises(ValueError, match="outside configured palette_size"):
        frame_to_palette_tensor(frame, palette_size=10, frame_size=(8, 1))


def test_action_conditioned_next_frame_loss_indexes_selected_action() -> None:
    logits = torch.zeros(1, 8, 10, 64, 64)
    logits[:, 2, 0] = 5.0
    logits[:, 3, 1] = 5.0
    frame = np.zeros((64, 64), dtype=np.uint8)

    selected_loss = next_frame_loss(
        logits,
        frame,
        palette_size=10,
        action_index=2,
    )
    wrong_action_loss = next_frame_loss(
        logits,
        frame,
        palette_size=10,
        action_index=3,
    )

    with pytest.raises(ValueError):
        next_frame_loss(logits, frame, palette_size=10)
    assert selected_loss.item() < wrong_action_loss.item()


def test_residual_next_frame_loss_adds_current_frame_prior() -> None:
    logits = torch.zeros(1, 8, 10, 64, 64)
    current_frame = np.zeros((64, 64), dtype=np.uint8)
    target_same = np.zeros((64, 64), dtype=np.uint8)
    target_changed = np.ones((64, 64), dtype=np.uint8)

    plain_loss = next_frame_loss(
        logits,
        target_same,
        palette_size=10,
        action_index=2,
        residual_prediction=False,
    )
    residual_same_loss = next_frame_loss(
        logits,
        target_same,
        palette_size=10,
        action_index=2,
        current_frame=current_frame,
        residual_prediction=True,
        residual_logit_bias=4.0,
    )
    residual_changed_loss = next_frame_loss(
        logits,
        target_changed,
        palette_size=10,
        action_index=2,
        current_frame=current_frame,
        residual_prediction=True,
        residual_logit_bias=4.0,
    )

    assert residual_same_loss.item() < plain_loss.item()
    assert residual_changed_loss.item() > plain_loss.item()
    with pytest.raises(ValueError, match="current_frame is required"):
        next_frame_loss(
            logits,
            target_same,
            palette_size=10,
            action_index=2,
            residual_prediction=True,
        )


def test_latent_changed_patch_mask_downsamples_frame_delta() -> None:
    current = np.zeros((64, 64), dtype=np.uint8)
    target = current.copy()
    target[8:16, 16:24] = 2

    mask = latent_changed_patch_mask(
        current,
        target,
        palette_size=10,
        frame_size=(64, 64),
        grid_shape=(8, 8),
    )

    assert mask.shape == (8, 8)
    assert int(mask.sum().item()) == 1
    assert bool(mask[1, 2])


def test_latent_grid_loss_prefers_correct_delta_and_reports_details() -> None:
    current = torch.zeros(8, 8, 4)
    target = current.clone()
    target[1, 2] = 1.0
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[1, 2] = True
    wrong = torch.zeros(1, 8, 8, 4)
    correct = target.sub(current).unsqueeze(0)

    wrong_loss, details = latent_grid_loss(
        wrong,
        current,
        target,
        mask,
        changed_patch_weight=6.0,
        return_details=True,
    )
    correct_loss = latent_grid_loss(
        correct,
        current,
        target,
        mask,
        changed_patch_weight=6.0,
    )

    assert correct_loss.item() < wrong_loss.item()
    assert details["grid_shape"] == [8, 8]
    assert details["changed_patch_count"] == 1
    assert details["changed_patch_loss"] is not None


def test_latent_grid_loss_adds_changed_patch_cosine_auxiliary() -> None:
    current = torch.zeros(2, 2, 3)
    target = current.clone()
    target[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.zeros(2, 2, dtype=torch.bool)
    mask[0, 0] = True
    aligned = target.sub(current).unsqueeze(0)
    opposite = aligned.clone()
    opposite[:, 0, 0] = torch.tensor([-1.0, 0.0, 0.0])

    aligned_loss, aligned_details = latent_grid_loss(
        aligned,
        current,
        target,
        mask,
        cosine_loss_weight=0.1,
        return_details=True,
    )
    opposite_loss, opposite_details = latent_grid_loss(
        opposite,
        current,
        target,
        mask,
        cosine_loss_weight=0.1,
        return_details=True,
    )

    assert aligned_loss.item() < opposite_loss.item()
    assert aligned_details["cosine_loss"] == pytest.approx(0.0)
    assert opposite_details["cosine_loss"] == pytest.approx(2.0)
    assert opposite_details["cosine_patch_count"] == 1


def test_reward_math_combines_progress_and_costs() -> None:
    reward = compute_reward(
        config=type(
            "RewardConfig",
            (),
            {
                "score_weight": 10.0,
                "learning_progress_weight": 2.0,
                "action_cost": 0.5,
                "time_cost_weight": 0.1,
                "update_cost": 0.25,
            },
        )(),
        score_delta=1.0,
        learning_progress=0.5,
        elapsed_seconds=2.0,
        update_steps=2,
    )

    assert reward.total == 9.8


def test_rescaled_time_cost_is_comparable_on_slow_turns() -> None:
    reward = compute_reward(
        config=type(
            "RewardConfig",
            (),
            {
                "score_weight": 10.0,
                "learning_progress_weight": 1.0,
                "action_cost": 0.02,
                "time_cost_weight": 0.0001,
                "update_cost": 0.01,
            },
        )(),
        score_delta=0.0,
        learning_progress=0.0,
        elapsed_seconds=40.0,
        update_steps=1,
    )

    assert reward.time_cost == pytest.approx(0.004)
    assert reward.total == pytest.approx(-0.034)


def test_lora_target_module_resolver_finds_linear_suffixes() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Module()
            self.block.q_proj = torch.nn.Linear(2, 2)
            self.block.down_proj = torch.nn.Linear(2, 2)
            self.block.norm = torch.nn.LayerNorm(2)

    targets = resolve_lora_target_modules(
        Tiny(),
        ("q_proj", "v_proj", "down_proj"),
    )

    assert targets == ["down_proj", "q_proj"]


def test_lora_target_module_resolver_finds_wrapped_linear_children() -> None:
    class WrappedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Module()
            self.block.q_proj = WrappedLinear()
            self.block.v_proj = WrappedLinear()
            self.block.other = WrappedLinear()

    targets = resolve_lora_target_modules(
        Tiny(),
        ("q_proj", "v_proj", "down_proj"),
    )

    assert targets == ["q_proj.linear", "v_proj.linear"]


def test_lora_target_module_resolver_avoids_wrapper_parent_collision() -> None:
    class WrappedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Module()
            self.block.q_proj = WrappedLinear()
            self.other = torch.nn.Module()
            self.other.q_proj = torch.nn.Linear(2, 2)
            self.other.down_proj = torch.nn.Linear(2, 2)

    targets = resolve_lora_target_modules(
        Tiny(),
        ("q_proj", "v_proj", "down_proj"),
    )

    assert targets == ["down_proj", "q_proj.linear"]


def test_fake_model_update_reduces_repeated_frame_loss() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    frame = np.zeros((64, 64), dtype=np.uint8)
    output = model("prompt", [])
    assert output.action_frame_logits.shape == (1, 8, 10, 64, 64)
    before = next_frame_loss(
        output.action_frame_logits,
        frame,
        palette_size=10,
        action_index=1,
    )

    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = next_frame_loss(
            model("prompt", []).action_frame_logits,
            frame,
            palette_size=10,
            action_index=1,
        )
        loss.backward()
        optimizer.step()

    after = next_frame_loss(
        model("prompt", []).action_frame_logits,
        frame,
        palette_size=10,
        action_index=1,
    )

    assert after.item() < before.item()


def test_fake_model_update_reduces_repeated_latent_loss_without_policy_step() -> None:
    model = FakeSingleVLMPolicy(hidden_size=16, palette_size=10)
    optimizer = torch.optim.AdamW(world_model_parameters(model), lr=1e-2)
    current = np.zeros((64, 64), dtype=np.uint8)
    target = current.copy()
    target[8:16, 16:24] = 2
    current_latent = model.frame_latent_grid(current)
    target_latent = model.frame_latent_grid(target)
    mask = latent_changed_patch_mask(
        current,
        target,
        palette_size=10,
        frame_size=(64, 64),
        grid_shape=(8, 8),
    )
    policy_before = _snapshot_parameters(policy_parameters(model))
    before = latent_grid_loss(
        model.predict_latent_delta("prompt", [current], action_index=1),
        current_latent,
        target_latent,
        mask,
    )

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = latent_grid_loss(
            model.predict_latent_delta("prompt", [current], action_index=1),
            current_latent,
            target_latent,
            mask,
        )
        loss.backward()
        optimizer.step()

    after = latent_grid_loss(
        model.predict_latent_delta("prompt", [current], action_index=1),
        current_latent,
        target_latent,
        mask,
    )

    assert after.item() < before.item()
    assert not _any_parameter_changed(policy_before, policy_parameters(model))


def test_dry_run_runner_writes_expected_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "dry_run.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_name": "test-dry-run",
                "dry_run": True,
                "frame_history_n": 2,
                "model": {
                    "backend": "fake",
                    "model_id": "fake",
                    "hidden_size": 16,
                    "lora": {"enabled": False, "save_every": 1},
                },
                "environment": {"max_turns": 7, "seed": 0},
                "update": {
                    "learning_rate": 0.001,
                    "update_steps": 1,
                    "policy_learning_progress_return_horizon": 4,
                    "policy_learning_progress_return_discount": 0.8,
                },
                "logging": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    torch.manual_seed(0)

    summary = run_experiment(config)

    assert summary["turns"] == 7
    assert summary["world_parameter_count"] > summary["policy_parameter_count"] > 0
    assert summary["world_model_step"] == 7
    assert summary["world_adapter"] == WORLD_ADAPTER_NAME
    assert summary["policy_adapter"] == POLICY_ADAPTER_NAME
    assert summary["action_runtime_adapter"] == POLICY_ADAPTER_NAME
    assert (output_dir / "config.resolved.yaml").exists()
    assert (output_dir / "turns.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "checkpoints" / "adapter_step_000001.safetensors").exists()
    assert (output_dir / "checkpoints" / "final_adapter" / "fake_model.pt").exists()
    turns_text = (output_dir / "turns.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in turns_text.splitlines()]
    assert len(rows) == 7
    assert rows[0]["turn"] == 0
    assert rows[0]["reward_inputs"]["world_adapter"] == WORLD_ADAPTER_NAME
    assert rows[0]["reward_inputs"]["policy_adapter"] == POLICY_ADAPTER_NAME
    assert rows[0]["policy_diagnostics"]["action_runtime_adapter"] == POLICY_ADAPTER_NAME
    assert [row["reward_inputs"]["lp_eval_turn"] for row in rows] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert [row["reward_inputs"]["holdout_eval_turn"] for row in rows] == [
        1,
        2,
        3,
        4,
        5,
        6,
        None,
    ]
    assert rows[0]["reward_inputs"]["lp_prior_model_step"] == 0
    assert rows[0]["reward_inputs"]["lp_current_model_step"] == 4
    assert rows[6]["reward_inputs"]["lp_prior_model_step"] == 6
    assert rows[6]["reward_inputs"]["lp_current_model_step"] == 7
    assert rows[0]["reward_inputs"]["lp_observation_source"] == (
        "discounted_window_snapshot"
    )
    assert rows[0]["reward_inputs"]["lp_same_observation"] is True
    assert rows[0]["reward_inputs"]["lp_window_turns"] == [0, 1, 2, 3]
    assert rows[0]["reward_inputs"]["lp_window_size"] == 4
    assert rows[0]["reward_inputs"]["lp_window_discount"] == pytest.approx(0.8)
    assert rows[0]["reward_inputs"]["lp_window_complete"] is True
    assert rows[4]["reward_inputs"]["lp_window_turns"] == [4, 5, 6]
    assert rows[4]["reward_inputs"]["lp_window_complete"] is False
    assert rows[6]["reward_inputs"]["lp_window_turns"] == [6]
    assert rows[6]["reward_inputs"]["lp_window_complete"] is False
    assert rows[0]["reward_inputs"]["policy_lp_return_turns"] == [0, 1, 2, 3]
    assert rows[0]["reward_inputs"]["policy_return_complete"] is True
    assert rows[4]["reward_inputs"]["policy_lp_return_turns"] == [4, 5, 6]
    assert rows[4]["reward_inputs"]["policy_return_complete"] is False
    assert rows[0]["reward_inputs"]["policy_return_discount"] == pytest.approx(0.8)
    assert "model_input" in rows[0]
    assert "prompt" in rows[0]["model_input"]
    assert rows[0]["model_input"]["frame_history_count"] >= 1
    assert rows[0]["model_input"]["decision_frame_index"] == 1
    assert rows[0]["model_input"]["decision_frame_count"] == 2
    assert "world_model_step_at_action" in rows[0]["model_input"]
    assert "model_output" in rows[0]
    assert "masked_action_probabilities" in rows[0]["model_output"]
    assert rows[0]["model_output"]["action_frame_logits_shape"] == [1, 8, 16, 64, 64]
    assert "reward_inputs" in rows[0]
    assert rows[0]["reward_inputs"]["decision_frame_index"] == 1
    assert rows[0]["reward_inputs"]["decision_frame_count"] == 2
    assert rows[0]["reward_inputs"]["next_decision_frame_index"] == 1
    assert rows[0]["reward_inputs"]["next_decision_frame_count"] == 2
    assert rows[0]["reward_inputs"]["frame_changed_pixels"] >= 0
    assert rows[0]["reward_inputs"]["frame_changed_fraction"] >= 0.0
    assert any(
        row["reward_inputs"]["frame_changed_pixels"] > 0
        for row in rows
    )
    assert "policy_diagnostics" in rows[0]
    diagnostics = rows[0]["policy_diagnostics"]
    assert "pre_world_update" in diagnostics
    assert "post_world_update" in diagnostics
    assert "post_policy_update" in diagnostics
    assert "head_norms_pre_world_update" in diagnostics
    assert "masked_action_probabilities" in diagnostics["pre_world_update"]
    assert "action_logit_l2" in diagnostics["pre_world_update"]
    assert "valid_action_logit_span" in diagnostics["pre_world_update"]
    assert "action_head_weight_l2" in diagnostics["head_norms_pre_world_update"]
    assert "action_head_bias_l2" in diagnostics["head_norms_pre_world_update"]
    assert diagnostics["world_optimizer_scope"] == "world_model_parameters"
    assert diagnostics["policy_optimizer_scope"] == "policy_parameters"
    assert "logits" not in diagnostics["pre_world_update"]
    assert [row["reward_inputs"]["policy_update_skipped_reason"] for row in rows[:3]] == [
        "warmup",
        "warmup",
        "warmup",
    ]
    assert rows[3]["reward_inputs"]["policy_update_skipped_reason"] == "baseline_init"
    assert rows[4]["reward_inputs"]["policy_update_skipped_reason"] is None
    assert rows[5]["reward_inputs"]["policy_update_skipped_reason"] is None
    assert rows[6]["reward_inputs"]["policy_update_skipped_reason"] is None
    assert rows[0]["reward_inputs"]["learning_progress_signal"] == "rate"
    assert rows[0]["reward_inputs"]["learning_progress_rate_beta"] == pytest.approx(0.5)
    assert rows[0]["reward_inputs"]["learning_progress_rate_baseline"] is None
    assert rows[0]["reward_inputs"]["learning_progress"] == pytest.approx(0.0)
    assert rows[0]["reward_inputs"]["learning_progress_rate"] == pytest.approx(0.0)
    assert rows[1]["reward_inputs"]["learning_progress_rate_baseline"] == pytest.approx(
        rows[0]["reward_inputs"]["raw_window_learning_progress"]
    )
    assert rows[3]["reward_inputs"]["intrinsic_reward_baseline"] is None
    assert rows[3]["reward_inputs"]["policy_advantage"] == 0.0
    assert rows[4]["reward_inputs"]["intrinsic_reward_baseline"] == pytest.approx(
        rows[3]["reward_inputs"]["policy_intrinsic_reward"]
    )
    assert rows[4]["reward_inputs"]["policy_loss_objective"] == "ppo_clipped_ratio"
    assert rows[4]["reward_inputs"]["policy_clip_epsilon"] == pytest.approx(0.2)
    assert rows[4]["reward_inputs"]["policy_update_accumulation_steps"] == 1
    assert rows[4]["reward_inputs"]["policy_update_batch_size"] == 1
    assert rows[4]["reward_inputs"]["policy_update_batch_turns"] == [4]
    assert rows[4]["policy_diagnostics"]["policy_update_batch"]["batch_size"] == 1
    assert rows[4]["reward_inputs"]["old_policy_log_probability"] == pytest.approx(
        rows[4]["model_output"]["selected_action_log_probability"]
    )
    assert rows[4]["reward_inputs"]["policy_probability_ratio_pre_update"] > 0.0
    assert rows[4]["reward_inputs"]["policy_probability_ratio_post_update"] > 0.0
    assert rows[4]["policy_diagnostics"]["ppo"]["objective"] == "clipped_ratio"
    assert rows[4]["policy_diagnostics"]["ppo"]["clip_epsilon"] == pytest.approx(0.2)
    assert rows[0]["reward_breakdown"]["update_cost"] == pytest.approx(0.01)
    assert rows[4]["reward_breakdown"]["update_cost"] == pytest.approx(0.02)
    for row in rows:
        reward_breakdown = row["reward_breakdown"]
        reward_inputs = row["reward_inputs"]
        weights = reward_inputs["lp_window_weights"]
        prior_losses = reward_inputs["lp_window_prior_losses"]
        current_losses = reward_inputs["lp_window_current_losses"]
        transition_progress = reward_inputs["lp_window_transition_progress"]
        assert sum(weights) == pytest.approx(1.0)
        assert len(weights) == reward_inputs["lp_window_size"]
        assert len(prior_losses) == reward_inputs["lp_window_size"]
        assert len(current_losses) == reward_inputs["lp_window_size"]
        assert len(transition_progress) == reward_inputs["lp_window_size"]
        raw_window_learning_progress = sum(
            weight * progress for weight, progress in zip(weights, transition_progress)
        )
        assert reward_inputs["raw_window_learning_progress"] == pytest.approx(
            raw_window_learning_progress
        )
        assert reward_inputs["raw_learning_progress"] == pytest.approx(
            raw_window_learning_progress
        )
        assert reward_inputs["learning_progress"] == pytest.approx(
            reward_inputs["learning_progress_rate"]
        )
        assert reward_inputs["learning_progress_rate"] == pytest.approx(
            raw_window_learning_progress
            - reward_inputs["learning_progress_rate_reference"]
        )
        assert reward_inputs["lp_pre_loss"] == reward_inputs["lp_prior_loss"]
        assert reward_inputs["lp_post_loss"] == reward_inputs["lp_current_loss"]
        assert reward_inputs["one_step_learning_progress"] == pytest.approx(
            reward_inputs["one_step_pre_loss"] - reward_inputs["one_step_post_loss"]
        )
        return_weights = reward_inputs["policy_lp_return_weights"]
        return_rates = reward_inputs["policy_lp_return_rates"]
        assert sum(return_weights) == pytest.approx(1.0)
        assert len(return_weights) == len(reward_inputs["policy_lp_return_turns"])
        assert len(return_rates) == len(reward_inputs["policy_lp_return_turns"])
        assert reward_inputs["policy_lp_return"] == pytest.approx(
            sum(weight * rate for weight, rate in zip(return_weights, return_rates))
        )
        assert reward_inputs["policy_intrinsic_reward"] == pytest.approx(
            10.0 * reward_inputs["policy_lp_return"]
        )
        expected_total = (
            10.0 * reward_breakdown["learning_progress"]
            + 10.0 * reward_breakdown["score_delta"]
            - reward_breakdown["action_cost"]
            - reward_breakdown["time_cost"]
            - reward_breakdown["update_cost"]
        )
        assert reward_breakdown["total"] == pytest.approx(expected_total)


def test_dry_run_runner_supports_latent_grid_world_loss(tmp_path: Path) -> None:
    config_path = tmp_path / "dry_run_latent.yaml"
    output_dir = tmp_path / "out_latent"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_name": "test-dry-run-latent",
                "dry_run": True,
                "frame_history_n": 2,
                "model": {
                    "backend": "fake",
                    "model_id": "fake",
                    "hidden_size": 16,
                    "lora": {"enabled": False, "save_every": 100},
                },
                "environment": {"max_turns": 5, "seed": 0},
                "update": {
                    "learning_rate": 0.001,
                    "update_steps": 1,
                    "world_loss_mode": "latent_grid",
                    "latent_loss_weight": 1.0,
                    "latent_changed_patch_weight": 6.0,
                    "latent_cosine_loss_weight": 0.1,
                    "latent_learning_progress_normalization": True,
                    "policy_warmup_turns": 1,
                },
                "logging": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    summary = run_experiment(load_config(config_path))

    assert summary["turns"] == 5
    assert summary["logged_latent_predictions"] == 5
    assert (output_dir / "latent_prediction_manifest.jsonl").exists()
    assert (output_dir / "latent_predictions").exists()
    rows = [
        json.loads(line)
        for line in (output_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["model_output"]["action_frame_logits_shape"] is None
    assert rows[0]["reward_inputs"]["world_loss_mode"] == "latent_grid"
    assert rows[0]["reward_inputs"]["one_step_post_latent_loss"] is not None
    assert rows[0]["reward_inputs"]["one_step_post_latent_cosine_loss"] is not None
    assert rows[0]["reward_inputs"]["latent_cosine_loss_weight"] == 0.1
    assert rows[0]["reward_inputs"]["latent_learning_progress_normalization"] is True
    assert rows[0]["reward_inputs"]["learning_progress_loss_scale"] > 0.0
    assert "normalized_window_learning_progress" in rows[0]["reward_inputs"]
    assert rows[0]["reward_inputs"]["latent_grid_shape"] == [8, 8]
    assert rows[0]["latent_prediction_artifact"]["image_path"].startswith(
        "latent_predictions/"
    )


def test_dry_run_runner_can_write_replay_video(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    config_path = tmp_path / "dry_run_video.yaml"
    output_dir = tmp_path / "out_video"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_name": "test-dry-run-video",
                "dry_run": True,
                "model": {
                    "backend": "fake",
                    "model_id": "fake",
                    "hidden_size": 16,
                    "lora": {"enabled": False, "save_every": 100},
                },
                "environment": {"max_turns": 2, "seed": 0},
                "update": {"learning_rate": 0.001, "update_steps": 1},
                "logging": {
                    "output_dir": str(output_dir),
                    "save_video": True,
                    "video_fps": 2,
                    "video_frame_scale": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    summary = run_experiment(config)

    assert summary["recorded_video_frames"] == 6
    assert (output_dir / "frames.mp4").exists()
    assert (output_dir / "frame_manifest.jsonl").exists()
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "frame_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["phase"] for row in manifest_rows] == [
        "reset",
        "reset",
        "after_action",
        "after_action",
        "after_action",
        "after_action",
    ]
    assert manifest_rows[0]["turn"] is None
    assert manifest_rows[2]["turn"] == 0


def test_frame_prediction_logger_writes_comparison_artifacts(tmp_path: Path) -> None:
    logger = FramePredictionLogger(
        tmp_path,
        enabled=True,
        save_every=1,
        frame_scale=1,
        palette_size=10,
        frame_size=(4, 4),
    )
    current = np.zeros((4, 4), dtype=np.uint8)
    target = current.copy()
    target[1, 1] = 2
    pre_update = current.copy()
    post_update = target.copy()

    artifact = logger.append_prediction(
        turn=3,
        action_name="ACTION2",
        selected_action_index=2,
        observation_id="obs-3",
        next_observation_id="obs-4",
        current_frame=current,
        target_frame=target,
        pre_update_prediction=pre_update,
        post_update_prediction=post_update,
        pre_update_loss=1.5,
        post_update_loss=0.25,
    )

    assert artifact is not None
    assert logger.count == 1
    assert (tmp_path / artifact["image_path"]).exists()
    assert (tmp_path / artifact["raw_arrays_path"]).exists()
    manifest_rows = [
        json.loads(line)
        for line in logger.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[0]["turn"] == 3
    assert manifest_rows[0]["pre_update"]["missed_changed_pixels"] == 1
    assert manifest_rows[0]["post_update"]["accuracy"] == 1.0


def test_latent_prediction_logger_writes_heatmap_artifacts(tmp_path: Path) -> None:
    logger = LatentPredictionLogger(
        tmp_path,
        enabled=True,
        save_every=1,
        frame_scale=2,
    )
    current = torch.zeros(2, 2, 3)
    target = current.clone()
    target[0, 1] = 1.0
    mask = torch.zeros(2, 2, dtype=torch.bool)
    mask[0, 1] = True
    pre = torch.zeros(2, 2, 3)
    post = target - current

    artifact = logger.append_prediction(
        turn=4,
        action_name="ACTION4",
        selected_action_index=4,
        observation_id="obs-4",
        next_observation_id="obs-5",
        current_latent_grid=current,
        target_latent_grid=target,
        changed_patch_mask=mask,
        pre_update_prediction=pre,
        post_update_prediction=post,
        pre_update_loss=0.5,
        post_update_loss=0.1,
    )

    assert artifact is not None
    assert logger.count == 1
    assert (tmp_path / artifact["image_path"]).exists()
    assert (tmp_path / artifact["raw_arrays_path"]).exists()
    manifest_rows = [
        json.loads(line)
        for line in logger.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[0]["changed_patch_count"] == 1
    assert manifest_rows[0]["post_error_norm_mean"] < manifest_rows[0][
        "pre_error_norm_mean"
    ]


def test_dry_run_supports_optional_action_baselines_residual_and_accumulation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dry_run_optional.yaml"
    output_dir = tmp_path / "out_optional"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_name": "test-dry-run-optional",
                "dry_run": True,
                "frame_history_n": 2,
                "model": {
                    "backend": "fake",
                    "model_id": "fake",
                    "hidden_size": 16,
                    "hidden_pooling": "attention",
                    "lora": {"enabled": False, "save_every": 100},
                },
                "environment": {"max_turns": 9, "seed": 0},
                "update": {
                    "learning_rate": 0.001,
                    "update_steps": 1,
                    "action_conditioned_learning_progress_baseline": True,
                    "residual_frame_prediction": True,
                    "residual_frame_logit_bias": 4.0,
                    "policy_update_accumulation_steps": 2,
                    "policy_advantage_baseline": "zero",
                    "policy_advantage_normalization": "ema_abs",
                    "policy_advantage_normalization_floor": 0.01,
                },
                "logging": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    torch.manual_seed(0)

    summary = run_experiment(config)

    assert summary["turns"] == 9
    assert _policy_update_accumulation_steps(config) == 2
    rows = [
        json.loads(line)
        for line in (output_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 9
    assert all(
        row["reward_inputs"]["learning_progress_rate_baseline_scope"] == "action"
        for row in rows
    )
    assert all(
        row["reward_inputs"]["residual_frame_prediction"] is True
        for row in rows
    )
    assert all(
        row["reward_inputs"]["frame_loss_mode"] == "residual_ce"
        for row in rows
    )
    assert all(
        row["reward_inputs"]["policy_advantage_baseline"] == "zero"
        for row in rows
    )
    assert all(
        row["reward_inputs"]["policy_advantage_normalization"] == "ema_abs"
        for row in rows
    )
    assert rows[3]["reward_inputs"]["policy_update_skipped_reason"] is None
    assert rows[3]["reward_inputs"]["intrinsic_reward_baseline"] is None
    batch_rows = [
        row
        for row in rows
        if row["reward_inputs"]["policy_update_batch_size"] == 2
    ]
    assert batch_rows
    for row in batch_rows:
        assert len(row["reward_inputs"]["policy_update_batch_turns"]) == 2
        assert row["reward_inputs"]["policy_update_pending"] is False
    assert all(
        row["reward_inputs"]["policy_update_batch_flushed"] is False
        for row in rows
        if row["reward_inputs"]["policy_update_batch_size"] > 0
    )
    assert any(
        row["reward_inputs"]["next_action_learning_progress_rate_baselines"]
        for row in rows
    )
    for row in rows[3:]:
        reward_inputs = row["reward_inputs"]
        if reward_inputs["policy_update_skipped_reason"] is not None:
            continue
        assert reward_inputs["policy_advantage"] == pytest.approx(
            reward_inputs["raw_policy_advantage"]
            / reward_inputs["policy_advantage_normalization_denominator"]
        )
