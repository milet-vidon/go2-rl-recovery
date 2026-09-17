"""Roll-only coordinate reflection, never a physical-state/history transformation.

Frozen inference diagnostic only, not symmetry training or asset/dynamics proof.
TensorDict is deep-cloned before the roll actor input is replaced. Standing
must continue to consume the caller's original real observation dictionary.
"""

import math

import torch
from tensordict import TensorDictBase


NATIVE_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                           for leg in ("FL", "FR", "RL", "RR"))
JOINT_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
JOINT_SIGNS = (-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)
OBSERVATION_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity",
                     "velocity_commands", "joint_pos", "joint_vel", "actions")
RIGHT_GRAVITY_Y_THRESHOLD = -math.cos(math.radians(30.0))


def _matrix(value, width, label):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{label} must have shape [N,{width}]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite floating point")


def _mask(mask, rows, device):
    if not isinstance(mask, torch.Tensor) or mask.shape != (rows,) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("Mask must be same-device bool[N]")


def reflect_joint_vector(value):
    """J is its own inverse; returned native-order tensor never aliases the input."""
    _matrix(value, 12, "Joint/action vector")
    return value[:, JOINT_PERMUTATION] * value.new_tensor(JOINT_SIGNS)


def reflect_policy_observation(value):
    """M48 maps a COPY to virtual coordinates; input remains a real measurement."""
    _matrix(value, 48, "Policy observation")
    result = value.clone()
    result[:, :3] = value[:, :3] * value.new_tensor((1, -1, 1))
    result[:, 3:6] = value[:, 3:6] * value.new_tensor((-1, 1, -1))
    result[:, 6:9] = value[:, 6:9] * value.new_tensor((1, -1, 1))
    result[:, 9:12] = value[:, 9:12] * value.new_tensor((1, -1, -1))
    for start in (12, 24, 36):
        result[:, start:start+12] = reflect_joint_vector(value[:, start:start+12])
    return result


def initial_right_mask(gravity, eligible_settled_fallen, mode):
    """Call ONCE at real policy start; no requested pose, source ID or outcome input.

    Nonfinite/zero gravity fails closed. Upright/back directions cannot satisfy
    the strict normalized body-y lateral cone. Eligibility is the already
    measured settled-fallen predicate, not merely the nominal reset class.
    """
    _matrix(gravity, 3, "Policy-start projected gravity")
    _mask(eligible_settled_fallen, gravity.shape[0], gravity.device)
    if mode not in ("off", "initial_right"):
        raise ValueError("Mirror mode must be off or initial_right")
    norm = torch.linalg.vector_norm(gravity, dim=-1)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= 0).any()):
        raise ValueError("Cannot classify nonfinite or zero policy-start gravity")
    selected = eligible_settled_fallen & (gravity[:, 1] / norm < RIGHT_GRAVITY_Y_THRESHOLD)
    return selected if mode == "initial_right" else torch.zeros_like(selected)


def roll_input_copy(real_obs, selected):
    """Deep clone EVERY TensorDict leaf, then set only the new policy input tensor.

    No shallow clone or in-place assignment into real_obs or its leaves. Even
    off mode returns independent storage, with exactly equal numerical values.
    """
    if not isinstance(real_obs, TensorDictBase):
        raise TypeError("Expected the real runtime TensorDict, not a reconstructed observation")
    value = real_obs["policy"]
    _matrix(value, 48, "Real policy observation")
    _mask(selected, value.shape[0], value.device)
    result = real_obs.clone(recurse=True)
    result["policy"] = torch.where(selected[:, None], reflect_policy_observation(value), value)
    return result


def physical_roll_action(model_action, selected):
    """Inverse-transform only the selected rows; action manager receives this result."""
    _matrix(model_action, 12, "Roll actor output in model coordinates")
    _mask(selected, model_action.shape[0], model_action.device)
    return torch.where(selected[:, None], reflect_joint_vector(model_action), model_action)


def validate_mirror_joint_contract(joint_names, default, soft_limits, hard_limits):
    """Prove representation/interval commutation, NOT morphology or dynamics."""
    if tuple(joint_names) != NATIVE_JOINT_NAMES:
        raise ValueError("Unexpected Go2 native joint order; never use ANYmal indices")
    _matrix(default, 12, "Default joint positions")
    if not torch.equal(reflect_joint_vector(default), default):
        raise ValueError("Nominal positions are not exactly reflection-compatible")
    for label, limits in (("soft", soft_limits), ("hard", hard_limits)):
        if not isinstance(limits, torch.Tensor) or limits.shape != (*default.shape, 2):
            raise ValueError(f"{label} bounds must be [N,12,2]")
        if limits.device != default.device or limits.dtype != default.dtype or not bool(torch.isfinite(limits).all()):
            raise ValueError(f"Invalid {label} limit dtype/device/finiteness")
        if bool((limits[:, :, 0] >= limits[:, :, 1]).any()):
            raise ValueError(f"Invalid {label} limit ordering")
        reflected_low = reflect_joint_vector(limits[:, :, 0])
        reflected_high = reflect_joint_vector(limits[:, :, 1])
        lower = torch.minimum(reflected_low, reflected_high)
        upper = torch.maximum(reflected_low, reflected_high)
        if not torch.equal(lower, limits[:, :, 0]) or not torch.equal(upper, limits[:, :, 1]):
            raise ValueError(f"{label} limits are not exactly reflection-compatible")
    return {"native_joint_names": list(NATIVE_JOINT_NAMES), "permutation": list(JOINT_PERMUTATION),
            "signs": list(JOINT_SIGNS), "default_exactly_compatible": True,
            "soft_limits_exactly_compatible": True, "hard_limits_exactly_compatible": True,
            "nominal_point25_soft_clamp_commutes_algebraically": True,
            "physical_asset_symmetry_proven": False}
