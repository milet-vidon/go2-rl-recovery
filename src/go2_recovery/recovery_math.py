"""Tensor-only posture geometry shared by rewards and regression checks."""

import torch


def upright_error_squared(gravity: torch.Tensor) -> torch.Tensor:
    """Squared distance to downward gravity; inversion has error four, not zero."""
    target = gravity.new_tensor((0.0, 0.0, -1.0))
    return (gravity - target).square().sum(dim=-1)


def reset_clearance_height(rotation: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    """Support height of a conservative Go2 default-pose envelope.

    Body-frame bounds: x +/-0.45, y +/-0.30, z [-0.55, 0.18] m.
    This intentionally produces a short drop, not a pre-settled ground pose.
    Only valid for the default joints with the configured +/-0.1 rad offsets.
    """
    center = rotation.new_tensor((0.0, 0.0, -0.185))
    extent = rotation.new_tensor((0.45, 0.30, 0.365))
    vertical = rotation[:, 2, :]
    return (vertical.abs() * extent).sum(-1) - (vertical * center).sum(-1) + margin
