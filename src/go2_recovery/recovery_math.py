"""Tensor-only posture geometry shared by rewards and regression checks."""

import torch


def stance_alignment_penalty(foot_b):
    """Nominal 0.32 m stance width and mirrored left/right foot positions."""
    side = foot_b.new_tensor([1., -1., 1., -1.])
    lateral = ((foot_b[:, :, 1] * side - 0.16) / 0.10).square().mean(1)
    mirrored_right = foot_b[:, [1, 3]] * foot_b.new_tensor([1., -1., 1.])
    symmetry = ((foot_b[:, [0, 2]] - mirrored_right)
                / foot_b.new_tensor([0.15, 0.10, 0.10])).square().mean((1, 2))
    return 0.5 * (lateral + symmetry)


def normal_stance_geometry(foot_b, knee_b, joint_delta):
    """FL/FR/RL/RR in the robot frame (+x front, +y left), in metres.

    Screening bounds for the nominal Go2 flat-ground stand, not constraints
    on rolling or swinging legs. Per-joint limits prevent an RMS average from
    hiding one severely twisted leg. This is not a mesh collision test.
    """
    side = foot_b.new_tensor([1., -1., 1., -1.])
    fore = foot_b.new_tensor([1., 1., -1., -1.])
    return ((foot_b[:, :, 1] * side > 0.06).all(1)
            & (foot_b[:, :, 1].abs() < 0.30).all(1)
            & (knee_b[:, :, 1] * side > 0.04).all(1)
            & (foot_b[:, :, 0] * fore > 0.08).all(1)
            & (joint_delta.abs().amax(1) < 0.65))


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
