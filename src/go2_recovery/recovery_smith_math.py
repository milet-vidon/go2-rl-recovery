"""Tensor-only Smith-inspired roll/stand shaping, not a success criterion.

Independent tensor implementation of the reward equations in ResetTask at:
https://github.com/lauramsmith/fine-tuning-locomotion/blob/
583f1de43e91cdd24d632d783872528eb1337480/
motion_imitation/envs/env_wrappers/reset_task.py

The caller supplies the Go2 target joint offsets and name-resolved joint weights;
neither A1 joint order nor A1 height is assumed here.
"""

import math
from numbers import Real

import torch


def smith_recovery_terms(
    cos_up: torch.Tensor,
    height: torch.Tensor,
    joint_delta: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_weights: torch.Tensor,
    *,
    target_height: float = 0.32,
) -> dict[str, torch.Tensor]:
    """Return per-environment roll/stand shaping and its ungated components.

    Shapes are ``(N,)``, ``(N,)``, ``(N,J)``, ``(N,J)``, and ``(J,)``.
    All tensors must share a floating dtype and device, with finite values.
    ``joint_delta`` is measured minus target joint position in radians. Resolve
    ``joint_weights`` by the asset's joint names before calling, not by assuming
    an articulation ordering. ``stand_gate`` is boolean; other outputs preserve
    the input dtype/device and have shape ``(N,)``. Empty batches are supported.

    The orientation gate is intentionally strict and discontinuous. Height,
    pose and velocity are reported ungated, but cannot influence total reward
    when the gate is closed. This function does not validate contacts, leg
    crossing, stability, or recovery success. Value-finiteness checks belong in
    offline diagnostics: only metadata is inspected here to avoid GPU-to-CPU
    synchronization during reward calculation.
    """
    if isinstance(target_height, bool) or not isinstance(target_height, Real):
        raise TypeError("target_height must be a finite positive Python real scalar")
    if not math.isfinite(target_height) or target_height <= 0:
        raise ValueError("target_height must be finite and strictly positive")

    inputs = {
        "cos_up": cos_up,
        "height": height,
        "joint_delta": joint_delta,
        "joint_velocity": joint_velocity,
        "joint_weights": joint_weights,
    }
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
    for name, value in inputs.items():
        if value.dtype != cos_up.dtype or value.device != cos_up.device:
            raise ValueError(f"{name} must share cos_up's dtype and device")
    if cos_up.ndim != 1 or height.shape != cos_up.shape:
        raise ValueError("cos_up and height must both have shape (N,)")
    if joint_delta.ndim != 2 or joint_delta.shape[0] != cos_up.shape[0] or joint_delta.shape[1] == 0:
        raise ValueError("joint_delta must have shape (N,J), with J > 0")
    if joint_velocity.shape != joint_delta.shape:
        raise ValueError("joint_velocity must share joint_delta's shape (N,J)")
    if joint_weights.shape != (joint_delta.shape[1],):
        raise ValueError("joint_weights must have shape (J,)")

    up = cos_up.clamp(-1.0, 1.0)
    roll = ((up + 1.0) * 0.5).square()
    stand_gate = up > math.cos(0.2 * math.pi)
    height_term = (height / target_height).clamp(0.0, 1.0)
    pose = torch.exp(-0.6 * (joint_weights * joint_delta).square().sum(dim=-1))
    velocity = torch.exp(-0.02 * joint_velocity.square().sum(dim=-1))
    stand = (0.2 * height_term + 0.6 * pose + 0.2 * velocity) * stand_gate
    return {
        "roll": roll,
        "stand": stand,
        "height": height_term,
        "pose": pose,
        "velocity": velocity,
        "stand_gate": stand_gate,
        "total": 0.5 * roll + 0.5 * stand,
    }
