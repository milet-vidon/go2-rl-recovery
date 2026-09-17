"""Subset action-history restoration; pure torch, no Isaac/Kit imports.

Call only after ``super()._reset_idx(env_ids)`` in a future isolated environment,
after its physical root/joint state has been restored. This is NOT a state-bank
loader, physics restore, observation override, or proof of deterministic replay.

The three source arrays are the collected a[t-1], a[t-2], and the joint target
actually executed during the last control interval. With the validated native
joint order and fixed nominal + 0.25 * raw action semantics, raw/processed/cache
are derivable; collecting redundant copies is unnecessary. We cannot establish
the provenance of an allegedly executed source target from these arrays alone.

Isaac Lab's set_joint_position_target only sets a buffer; it does NOT step or
write physics. The caller owns the later normal scene write and observation
recomputation. Measured q is never read, clamped, or replaced here.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import torch


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _scalar_is(value, expected: float) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value == expected


def _tensor(value, shape, device, dtype, name: str, *, finite: bool = False) -> None:
    _check(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    _check(tuple(value.shape) == tuple(shape), f"{name} has wrong shape")
    _check(value.device == device and value.dtype == dtype, f"{name} device/dtype mismatch")
    _check(not value.requires_grad, f"{name} must not require gradients")
    if finite:
        _check(bool(torch.isfinite(value).all()), f"{name} must be finite")


@torch.no_grad()
def restore_handoff_action_history(
    action_manager,
    asset,
    env_ids: torch.Tensor,
    *,
    previous_raw_action: torch.Tensor,
    previous_previous_raw_action: torch.Tensor,
    previous_executed_joint_target_rad: torch.Tensor,
    joint_names: Sequence[str],
) -> None:
    """Restore only selected rows from three recorded (K, 12) arrays.

    Arrays must already have the destination device/dtype and be ordered like
    env_ids. IDs must be unique int64 and native action order must equal all 12
    asset joints. Unsupported semantics and inconsistent source targets raise
    before any mutation. No manager/term processing, simulator step or direct
    observation writes are performed. Empty selections are validated no-ops.

    Only the exact, single ControlStepJointPositionAction protocol is supported;
    this intentionally rejects current-position offsets, permuted/subset joints,
    per-joint scaling, action clipping and legacy JointPositionAction terms.
    """
    _check(list(action_manager.active_terms) == ["joint_pos"], "Require one joint_pos action term")
    term = action_manager.get_term("joint_pos")
    _check(type(term).__name__ == "ControlStepJointPositionAction", "Unsupported action term class")
    _check(term.cfg.class_type is type(term), "Action term/config class mismatch")
    _check(term._asset is asset, "Action term uses a different asset")
    _check(term.cfg.reference == "nominal", "Require nominal action reference")
    _check(_scalar_is(term.cfg.scale, 0.25) and _scalar_is(term._scale, 0.25), "Require scalar action scale 0.25")
    _check(_scalar_is(term.cfg.offset, 0.0) and _scalar_is(term._offset, 0.0), "Require zero action offset")
    _check(term.cfg.use_default_offset is False and term.cfg.clip is None, "Default offset/clipping is unsupported")

    native_names = tuple(asset.joint_names)
    _check(len(native_names) == 12 and len(set(native_names)) == 12, "Require 12 unique native joint names")
    _check(all(isinstance(name, str) and name for name in native_names), "Invalid native joint names")
    _check(not isinstance(joint_names, (str, bytes)) and tuple(joint_names) == native_names, "Snapshot native joint order mismatch")
    _check(tuple(term._joint_names) == native_names, "Action term native joint order mismatch")
    _check(asset.num_joints == 12 and term.action_dim == 12 and action_manager.total_action_dim == 12, "Action/asset dimensions differ")
    ids = term._joint_ids
    if isinstance(ids, slice):
        actual_joint_ids = list(range(12))[ids]
    elif isinstance(ids, torch.Tensor):
        _check(ids.ndim == 1 and ids.dtype == torch.long, "Invalid action joint indices")
        actual_joint_ids = ids.tolist()
    else:
        _check(isinstance(ids, (list, tuple)), "Unsupported action joint index representation")
        _check(all(isinstance(index, int) and not isinstance(index, bool) for index in ids), "Invalid action joint indices")
        actual_joint_ids = list(ids)
    _check(actual_joint_ids == list(range(12)), "Action joint indices must preserve full native order")

    current = action_manager.action
    _check(isinstance(current, torch.Tensor) and current.ndim == 2 and current.shape[1] == 12, "Invalid manager action buffer")
    _check(current.dtype in (torch.float32, torch.float64), "Unsupported floating-point dtype")
    n, width = current.shape
    device, dtype = current.device, current.dtype
    _check(isinstance(env_ids, torch.Tensor) and env_ids.ndim == 1 and env_ids.dtype == torch.long, "env_ids must be a 1-D int64 tensor")
    _check(env_ids.device == device, "env_ids device mismatch")
    _check(bool(((env_ids >= 0) & (env_ids < n)).all()), "env_ids out of range")
    _check(env_ids.unique().numel() == env_ids.numel(), "env_ids must be unique")
    selected_ids = env_ids.clone()
    shape = (selected_ids.numel(), width)

    # Validate every destination before any write. Isaac's properties expose
    # these actual buffers, not copies. Do not call process_action(s): that
    # would update all N environments and shift their history by one interval.
    buffers = (
        ("manager.action", current),
        ("manager.prev_action", action_manager.prev_action),
        ("term.raw_actions", term._raw_actions),
        ("term.processed_actions", term._processed_actions),
        ("term._target", term._target),
        ("asset.joint_pos_target", asset.data.joint_pos_target),
    )
    for name, buffer in buffers:
        _tensor(buffer, (n, width), device, dtype, name)
    _check(term.raw_actions is term._raw_actions and term.processed_actions is term._processed_actions, "Action properties do not expose their native buffers")
    _tensor(asset.data.default_joint_pos, (n, width), device, dtype, "default_joint_pos")
    _tensor(asset.data.soft_joint_pos_limits, (n, width, 2), device, dtype, "soft_joint_pos_limits")
    for name, value in (
        ("previous_raw_action", previous_raw_action),
        ("previous_previous_raw_action", previous_previous_raw_action),
        ("previous_executed_joint_target_rad", previous_executed_joint_target_rad),
    ):
        _tensor(value, shape, device, dtype, name, finite=True)

    nominal = asset.data.default_joint_pos.index_select(0, selected_ids)
    limits = asset.data.soft_joint_pos_limits.index_select(0, selected_ids)
    _check(bool(torch.isfinite(nominal).all() and torch.isfinite(limits).all()), "Selected nominal/limits must be finite")
    _check(bool((limits[..., 0] < limits[..., 1]).all()), "Selected soft limits must be ordered")
    raw = previous_raw_action.clone()
    prior_raw = previous_previous_raw_action.clone()
    processed = raw * 0.25
    executed = previous_executed_joint_target_rad.clone()
    unclamped = nominal + processed
    _check(bool(torch.isfinite(processed).all() and torch.isfinite(unclamped).all()), "Derived action target is nonfinite")
    expected = torch.clamp(unclamped, min=limits[..., 0], max=limits[..., 1])
    _check(bool(((executed >= limits[..., 0]) & (executed <= limits[..., 1])).all()), "Recorded executed target exceeds soft limits")
    _check(torch.allclose(executed, expected, rtol=0.0, atol=1e-6), "Recorded executed target contradicts nominal + 0.25 * previous_raw_action with soft clamp")
    if selected_ids.numel() == 0:
        return

    # The source executed target is authoritative after its semantics check;
    # keep its exact values, rather than replacing them with measured q or a
    # newly recomputed/rounded target. The native setter only writes a buffer.
    asset.set_joint_position_target(executed, joint_ids=slice(None), env_ids=selected_ids)
    current.index_copy_(0, selected_ids, raw)
    action_manager.prev_action.index_copy_(0, selected_ids, prior_raw)
    term._raw_actions.index_copy_(0, selected_ids, raw)
    term._processed_actions.index_copy_(0, selected_ids, processed)
    term._target.index_copy_(0, selected_ids, executed)
