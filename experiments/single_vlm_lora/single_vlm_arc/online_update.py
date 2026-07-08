"""Losses and online update helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from single_vlm_arc.model import POLICY_ADAPTER_NAME, WORLD_ADAPTER_NAME


def frame_to_palette_tensor(
    frame: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int] = (64, 64),
) -> Any:
    """Convert an ARC frame or image into a palette-index target tensor."""

    import torch
    from PIL import Image

    if isinstance(frame, Image.Image):
        image = frame.convert("L").resize(frame_size, Image.Resampling.NEAREST)
        array = np.asarray(image)
        target = np.rint(array * (palette_size - 1) / 255.0).astype("int64")
    else:
        array = np.asarray(frame)
        if array.ndim == 3:
            image = Image.fromarray(array.astype("uint8")).convert("L")
            image = image.resize(frame_size, Image.Resampling.NEAREST)
            array = np.asarray(image)
            target = np.rint(array * (palette_size - 1) / 255.0).astype("int64")
        elif array.ndim == 2:
            if array.shape != (frame_size[1], frame_size[0]):
                image = Image.fromarray(array.astype("uint8")).resize(
                    frame_size,
                    Image.Resampling.NEAREST,
                )
                array = np.asarray(image)
            target = array.astype("int64")
            min_value = int(target.min(initial=0))
            max_value = int(target.max(initial=0))
            if min_value < 0 or max_value >= int(palette_size):
                raise ValueError(
                    "target frame palette values outside configured palette_size: "
                    f"min={min_value}, max={max_value}, palette_size={palette_size}"
                )
        else:
            raise ValueError(f"unsupported frame array shape: {array.shape}")
    return torch.as_tensor(target, dtype=torch.long)


def next_frame_loss(
    frame_logits: Any,
    target_frame: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int] = (64, 64),
    action_index: int | None = None,
    current_frame: Any | None = None,
    residual_prediction: bool = False,
    residual_logit_bias: float = 4.0,
) -> Any:
    """Cross-entropy loss for predicted 64x64 palette logits."""

    import torch.nn.functional as F

    frame_logits = adjusted_frame_logits(
        frame_logits,
        current_frame=current_frame,
        palette_size=palette_size,
        frame_size=frame_size,
        action_index=action_index,
        residual_prediction=residual_prediction,
        residual_logit_bias=residual_logit_bias,
    )
    target = frame_to_palette_tensor(
        target_frame,
        palette_size=palette_size,
        frame_size=frame_size,
    ).to(frame_logits.device)
    return F.cross_entropy(frame_logits, target.unsqueeze(0))


def latent_changed_patch_mask(
    current_frame: Any,
    target_frame: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int] = (64, 64),
    grid_shape: tuple[int, int],
) -> Any:
    """Return a boolean patch-grid mask for frame regions changed by the action."""

    import torch.nn.functional as F

    current = frame_to_palette_tensor(
        current_frame,
        palette_size=palette_size,
        frame_size=frame_size,
    )
    target = frame_to_palette_tensor(
        target_frame,
        palette_size=palette_size,
        frame_size=frame_size,
    )
    changed = (current != target).float().unsqueeze(0).unsqueeze(0)
    pooled = F.interpolate(changed, size=grid_shape, mode="area")[0, 0]
    return pooled > 0.0


def latent_grid_loss(
    predicted_delta: Any,
    current_latent_grid: Any,
    target_latent_grid: Any,
    changed_patch_mask: Any,
    *,
    changed_patch_weight: float = 6.0,
    huber_beta: float = 1.0,
    cosine_loss_weight: float = 0.0,
    cosine_min_delta_norm: float = 1e-4,
    return_details: bool = False,
) -> Any:
    """Patch-weighted Smooth L1 plus optional changed-patch cosine loss."""

    import torch
    import torch.nn.functional as F

    if predicted_delta.ndim == 3:
        predicted_delta = predicted_delta.unsqueeze(0)
    if predicted_delta.ndim != 4:
        raise ValueError(
            "predicted_delta must have shape [batch, height, width, dim], "
            f"got {tuple(predicted_delta.shape)}"
        )
    current = torch.as_tensor(
        current_latent_grid,
        device=predicted_delta.device,
        dtype=predicted_delta.dtype,
    )
    target = torch.as_tensor(
        target_latent_grid,
        device=predicted_delta.device,
        dtype=predicted_delta.dtype,
    )
    mask = torch.as_tensor(
        changed_patch_mask,
        device=predicted_delta.device,
        dtype=torch.bool,
    )
    if current.shape != target.shape:
        raise ValueError(
            "current and target latent grids must have the same shape: "
            f"current={tuple(current.shape)}, target={tuple(target.shape)}"
        )
    if predicted_delta.shape[1:] != current.shape:
        raise ValueError(
            "predicted latent delta shape does not match target latent grid: "
            f"predicted={tuple(predicted_delta.shape[1:])}, "
            f"target={tuple(current.shape)}"
        )
    if mask.shape != current.shape[:2]:
        raise ValueError(
            "changed patch mask shape does not match latent grid: "
            f"mask={tuple(mask.shape)}, grid={tuple(current.shape[:2])}"
        )

    target_delta = target - current
    element_loss = F.smooth_l1_loss(
        predicted_delta,
        target_delta.unsqueeze(0).expand_as(predicted_delta),
        beta=max(float(huber_beta), 1e-6),
        reduction="none",
    ).mean(dim=-1)
    weights = torch.ones_like(element_loss)
    weights = weights + (
        (max(float(changed_patch_weight), 1.0) - 1.0)
        * mask.to(dtype=weights.dtype).unsqueeze(0)
    )
    huber_loss = (element_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    cosine_loss, cosine_patch_count = _latent_changed_patch_cosine_loss(
        predicted_delta,
        target_delta,
        mask,
        min_delta_norm=cosine_min_delta_norm,
    )
    loss = huber_loss + (float(cosine_loss_weight) * cosine_loss)
    if not return_details:
        return loss
    details = latent_grid_loss_details(
        predicted_delta,
        current,
        target,
        mask,
        element_loss=element_loss,
        cosine_min_delta_norm=cosine_min_delta_norm,
    )
    details["huber_loss"] = float(huber_loss.detach().cpu().item())
    details["cosine_loss"] = float(cosine_loss.detach().cpu().item())
    details["cosine_loss_weight"] = float(cosine_loss_weight)
    details["cosine_patch_count"] = int(cosine_patch_count)
    details["loss"] = float(loss.detach().cpu().item())
    return loss, details


def _latent_changed_patch_cosine_loss(
    predicted_delta: Any,
    target_delta: Any,
    changed_patch_mask: Any,
    *,
    min_delta_norm: float,
) -> tuple[Any, int]:
    import torch
    import torch.nn.functional as F

    if predicted_delta.ndim == 3:
        predicted_delta = predicted_delta.unsqueeze(0)
    target_norm = target_delta.detach().float().norm(dim=-1)
    cosine_mask = changed_patch_mask & (
        target_norm > max(float(min_delta_norm), 0.0)
    )
    patch_count = int(cosine_mask.sum().detach().cpu().item())
    if patch_count == 0:
        return predicted_delta.sum() * 0.0, 0
    prediction = predicted_delta[:, cosine_mask, :].reshape(-1, predicted_delta.shape[-1])
    target = (
        target_delta[cosine_mask]
        .unsqueeze(0)
        .expand(predicted_delta.shape[0], -1, -1)
        .reshape(-1, target_delta.shape[-1])
    )
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-8)
    return (1.0 - cosine).mean().to(dtype=predicted_delta.dtype), patch_count


def latent_grid_loss_details(
    predicted_delta: Any,
    current_latent_grid: Any,
    target_latent_grid: Any,
    changed_patch_mask: Any,
    *,
    element_loss: Any | None = None,
    cosine_min_delta_norm: float = 1e-4,
) -> dict[str, Any]:
    """Return detached scalar metrics and maps for one latent-grid prediction."""

    import torch
    import torch.nn.functional as F

    if predicted_delta.ndim == 3:
        predicted_delta = predicted_delta.unsqueeze(0)
    current = torch.as_tensor(
        current_latent_grid,
        device=predicted_delta.device,
        dtype=predicted_delta.dtype,
    )
    target = torch.as_tensor(
        target_latent_grid,
        device=predicted_delta.device,
        dtype=predicted_delta.dtype,
    )
    mask = torch.as_tensor(
        changed_patch_mask,
        device=predicted_delta.device,
        dtype=torch.bool,
    )
    target_delta = target - current
    error = predicted_delta - target_delta.unsqueeze(0)
    if element_loss is None:
        element_loss = F.smooth_l1_loss(
            predicted_delta,
            target_delta.unsqueeze(0).expand_as(predicted_delta),
            reduction="none",
        ).mean(dim=-1)
    patch_loss = element_loss.detach().float().mean(dim=0)
    changed_count = int(mask.sum().detach().cpu().item())
    cosine_loss, cosine_patch_count = _latent_changed_patch_cosine_loss(
        predicted_delta,
        target_delta,
        mask,
        min_delta_norm=cosine_min_delta_norm,
    )
    total_count = int(mask.numel())
    unchanged = ~mask
    changed_loss = _masked_mean(patch_loss, mask)
    unchanged_loss = _masked_mean(patch_loss, unchanged)
    return {
        "grid_shape": [int(current.shape[0]), int(current.shape[1])],
        "latent_dim": int(current.shape[-1]),
        "changed_patch_count": changed_count,
        "changed_patch_fraction": float(changed_count / max(total_count, 1)),
        "loss": float(patch_loss.mean().detach().cpu().item()),
        "changed_patch_loss": changed_loss,
        "unchanged_patch_loss": unchanged_loss,
        "cosine_loss": float(cosine_loss.detach().cpu().item()),
        "cosine_patch_count": int(cosine_patch_count),
        "target_delta_norm_mean": float(
            target_delta.detach().float().norm(dim=-1).mean().cpu().item()
        ),
        "prediction_delta_norm_mean": float(
            predicted_delta.detach().float().norm(dim=-1).mean().cpu().item()
        ),
        "error_norm_mean": float(error.detach().float().norm(dim=-1).mean().cpu().item()),
        "target_delta_norm_map": target_delta.detach().float().norm(dim=-1).cpu(),
        "prediction_delta_norm_map": predicted_delta.detach()
        .float()
        .norm(dim=-1)
        .mean(dim=0)
        .cpu(),
        "error_norm_map": error.detach().float().norm(dim=-1).mean(dim=0).cpu(),
        "changed_patch_mask": mask.detach().cpu(),
    }


def _masked_mean(values: Any, mask: Any) -> float | None:
    selected = values[mask]
    if selected.numel() == 0:
        return None
    return float(selected.mean().detach().cpu().item())


def predicted_frame_tensor(
    frame_logits: Any,
    *,
    palette_size: int,
    frame_size: tuple[int, int] = (64, 64),
    action_index: int | None = None,
    current_frame: Any | None = None,
    residual_prediction: bool = False,
    residual_logit_bias: float = 4.0,
) -> Any:
    """Return the argmax 64x64 palette prediction used by the frame loss."""

    adjusted = adjusted_frame_logits(
        frame_logits,
        current_frame=current_frame,
        palette_size=palette_size,
        frame_size=frame_size,
        action_index=action_index,
        residual_prediction=residual_prediction,
        residual_logit_bias=residual_logit_bias,
    )
    return adjusted.argmax(dim=1).squeeze(0).detach().cpu()


def adjusted_frame_logits(
    frame_logits: Any,
    *,
    current_frame: Any | None = None,
    palette_size: int,
    frame_size: tuple[int, int] = (64, 64),
    action_index: int | None = None,
    residual_prediction: bool = False,
    residual_logit_bias: float = 4.0,
) -> Any:
    """Return selected `[batch, palette, height, width]` logits after priors."""

    import torch.nn.functional as F

    frame_logits = selected_action_frame_logits(
        frame_logits,
        action_index=action_index,
    )
    if frame_logits.ndim == 3:
        frame_logits = frame_logits.unsqueeze(0)
    if not residual_prediction:
        return frame_logits
    if current_frame is None:
        raise ValueError("current_frame is required for residual frame prediction")
    current_target = frame_to_palette_tensor(
        current_frame,
        palette_size=palette_size,
        frame_size=frame_size,
    ).to(frame_logits.device)
    current_prior = F.one_hot(
        current_target,
        num_classes=palette_size,
    ).permute(2, 0, 1)
    current_prior = current_prior.to(
        device=frame_logits.device,
        dtype=frame_logits.dtype,
    )
    return frame_logits + (float(residual_logit_bias) * current_prior.unsqueeze(0))


def selected_action_frame_logits(frame_logits: Any, *, action_index: int | None) -> Any:
    """Return `[batch, palette, height, width]` logits for one action."""

    if frame_logits.ndim != 5:
        return frame_logits
    if action_index is None:
        raise ValueError("action_index is required for action-conditioned frame logits")
    return frame_logits[:, int(action_index)]


def coord_self_imitation_loss(coord_logits: Any, x: int | None, y: int | None) -> Any:
    """Small auxiliary loss that reinforces chosen ACTION6 coordinates."""

    import torch
    import torch.nn.functional as F

    if x is None or y is None:
        return torch.zeros((), dtype=coord_logits.dtype, device=coord_logits.device)
    logits = coord_logits[0] if coord_logits.ndim == 2 else coord_logits
    x_target = torch.tensor([max(0, min(63, int(x)))], device=coord_logits.device)
    y_target = torch.tensor([max(0, min(63, int(y)))], device=coord_logits.device)
    return F.cross_entropy(logits[:64].unsqueeze(0), x_target) + F.cross_entropy(
        logits[64:].unsqueeze(0),
        y_target,
    )


def trainable_parameters(model: Any) -> list[Any]:
    """Return trainable parameters for the optimizer."""

    scoped_parameters = _unique_parameters(
        [
            *policy_parameters(model),
            *world_model_parameters(model),
        ]
    )
    if scoped_parameters:
        return scoped_parameters
    return _requires_grad_parameters(model)


def policy_parameters(model: Any) -> list[Any]:
    """Return parameters that policy-gradient updates are allowed to change."""

    parameters: list[Any] = []
    if bool(getattr(model, "policy_adapter_trainable", True)):
        parameters.extend(
            _role_adapter_parameters(
                model,
                getattr(model, "policy_adapter_name", POLICY_ADAPTER_NAME),
            )
        )
    for module_name in ("action_norm", "action_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        parameters.extend(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
    return _unique_parameters(parameters)


def world_model_parameters(model: Any) -> list[Any]:
    """Return predictive/world parameters excluding the policy-only head."""

    if not bool(getattr(model, "role_adapters_enabled", False)):
        policy_parameter_ids = {id(parameter) for parameter in policy_parameters(model)}
        return [
            parameter
            for parameter in _requires_grad_parameters(model)
            if id(parameter) not in policy_parameter_ids
        ]

    parameters = _role_adapter_parameters(
        model,
        getattr(model, "world_adapter_name", WORLD_ADAPTER_NAME),
    )
    for module_name in (
        "hidden_pool",
        "action_condition",
        "coord_x_condition",
        "coord_y_condition",
        "coord_head",
        "frame_head",
        "latent_patch_position",
        "latent_delta_head",
    ):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        parameters.extend(parameter for parameter in module.parameters())
    return _unique_parameters(parameters)


def _requires_grad_parameters(model: Any) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _role_adapter_parameters(model: Any, adapter_name: str) -> list[Any]:
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        return []
    parameters: list[Any] = []
    for name, parameter in base_model.named_parameters():
        if str(adapter_name) in name.split("."):
            parameters.append(parameter)
    return _unique_parameters(parameters)


def _unique_parameters(parameters: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[int] = set()
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        unique.append(parameter)
    return unique
