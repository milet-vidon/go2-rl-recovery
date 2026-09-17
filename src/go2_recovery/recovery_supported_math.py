"""Dense joint-return shaping; this is not a support or recovery-success test.

The caller validates the fixed joint soft ranges once during initialization.
The per-step scoring path checks tensor metadata only and never copies tensor
values to the CPU.  Ordinary finite inputs follow the stated linear-in-error
formula, avoiding an exponentially vanishing signal at large joint offsets.
"""

from __future__ import annotations

import torch


def _require_float_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a real floating-point dtype")


def _validate_range_metadata(joint_soft_range: torch.Tensor) -> None:
    _require_float_tensor(joint_soft_range, "joint_soft_range")
    if joint_soft_range.ndim not in (1, 2) or joint_soft_range.shape[-1] != 12:
        raise ValueError("joint_soft_range must have shape [12] or [N, 12]")


def validate_supported_joint_soft_range(joint_soft_range: torch.Tensor) -> None:
    """Validate fixed ranges once at initialization (may synchronize a device).

    Ranges must be finite and strictly positive.  An empty [0, 12] batch is
    permitted, consistent with the empty-batch scoring contract.
    """
    _validate_range_metadata(joint_soft_range)
    if not bool(torch.all(torch.isfinite(joint_soft_range))):
        raise ValueError("joint_soft_range must contain only finite values")
    if not bool(torch.all(joint_soft_range > 0)):
        raise ValueError("joint_soft_range must contain only positive values")


def supported_pose_score(
    cos_up: torch.Tensor,
    joint_delta: torch.Tensor,
    joint_soft_range: torch.Tensor,
) -> torch.Tensor:
    """Return [N] joint-return shaping in [0, 1], not a standing criterion.

    For finite inputs and validated positive ranges the exact formula is:
      clamp(cos_up, 0, 1)^2 *
      (1 - mean(clamp(abs(joint_delta) / joint_soft_range, 0, 1))).

    Inputs have shapes [N], [N, 12], and [12] or [N, 12], respectively.
    All must share a real floating dtype and device. State tensors MUST be
    finite, and fixed ranges MUST be validated separately at setup. There is
    deliberately no nan_to_num sanitization. Clamps are NOT finite-state
    validation (infinities can clamp to finite values); callers must separately
    monitor simulator state and model finiteness.
    """
    _require_float_tensor(cos_up, "cos_up")
    _require_float_tensor(joint_delta, "joint_delta")
    _validate_range_metadata(joint_soft_range)
    if cos_up.ndim != 1:
        raise ValueError("cos_up must have shape [N]")
    if joint_delta.ndim != 2 or joint_delta.shape != (cos_up.shape[0], 12):
        raise ValueError("joint_delta must have shape [N, 12] matching cos_up")
    if joint_soft_range.ndim == 2 and joint_soft_range.shape != joint_delta.shape:
        raise ValueError("batched joint_soft_range must match joint_delta [N, 12]")
    for name, value in (("joint_delta", joint_delta), ("joint_soft_range", joint_soft_range)):
        if value.device != cos_up.device:
            raise ValueError(f"{name} must be on the same device as cos_up")
        if value.dtype != cos_up.dtype:
            raise TypeError(f"{name} must have the same dtype as cos_up")

    joint_error = (joint_delta.abs() / joint_soft_range).clamp(0.0, 1.0).mean(dim=-1)
    return cos_up.clamp(0.0, 1.0).square() * (1.0 - joint_error)
