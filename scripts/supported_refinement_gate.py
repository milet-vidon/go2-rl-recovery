"""Latch a standing-only refinement actor after 150 completed valid intervals.

This helper has no environment access and cannot reset or modify physics,
observations, actions, command history, or the standing criterion. The caller
must evaluate its existing strict criterion AFTER ``env.step`` completes and
call this function exactly once for that completed 50 Hz control interval.
The returned latch selects the NEXT action, never the interval just measured.

The validity mask must include the full original strict-standing criterion.
Passing a height/contact shortcut would not implement this gate's contract.
The caller owns initial all-zero counters and all-false latches; there is no
reset API. Once enabled, refinement remains latched for this rollout, even if
later standing becomes invalid. Such a regression must remain visible to the
unchanged evaluator, not be hidden by switching actors back or resetting.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


CONTROL_DT = 0.02
REQUIRED_COMPLETED_INTERVALS = 150
REQUIRED_HOLD_SECONDS = CONTROL_DT * REQUIRED_COMPLETED_INTERVALS


def advance_after_completed_interval(
    strict_valid: torch.Tensor,
    completed_count: torch.Tensor,
    latched: torch.Tensor,
    *,
    dt: float = CONTROL_DT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(next_count, next_latched, just_enabled)`` without mutation.

    Inputs are same-device, nonempty one-dimensional tensors. ``strict_valid``
    and ``latched`` must have dtype bool; ``completed_count`` must be int64 in
    [0, 150]. Counts describe consecutive valid completed intervals BEFORE the
    newly measured interval. Invalid intervals reset the count to zero, while
    a previously true latch never clears. Count 150 with a false input latch
    is rejected as an impossible state produced by this transition function.

    ``just_enabled`` is true exactly once per environment: after its 150th
    consecutive valid completed interval, to enable the subsequent action.
    This API intentionally accepts no pose ID, elapsed wall time, result
    lookup, desired command, model identity, reset flag, or interpolation.
    """
    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise TypeError("dt must be a real 50 Hz control interval")
    if not math.isfinite(float(dt)) or float(dt) != CONTROL_DT:
        raise ValueError("refinement gate requires exactly dt=0.02 (50 Hz)")

    named = (
        ("strict_valid", strict_valid, torch.bool),
        ("completed_count", completed_count, torch.int64),
        ("latched", latched, torch.bool),
    )
    for name, value, expected_dtype in named:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must use a dense strided tensor")
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"{name} must have nonempty shape [N]")
        if value.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must use CPU or CUDA storage")
    if not (strict_valid.shape == completed_count.shape == latched.shape):
        raise ValueError("all refinement gate inputs must have the same [N] shape")
    if not (strict_valid.device == completed_count.device == latched.device):
        raise ValueError("all refinement gate inputs must use the same device")
    if bool(((completed_count < 0) | (completed_count > REQUIRED_COMPLETED_INTERVALS)).any()):
        raise ValueError("completed_count must be within [0, 150]")
    if bool(((completed_count == REQUIRED_COMPLETED_INTERVALS) & ~latched).any()):
        raise ValueError("count 150 with an unset latch is inconsistent gate state")

    next_count = torch.where(
        strict_valid,
        torch.clamp(completed_count + 1, max=REQUIRED_COMPLETED_INTERVALS),
        torch.zeros_like(completed_count),
    )
    just_enabled = ~latched & (next_count == REQUIRED_COMPLETED_INTERVALS)
    next_latched = latched | just_enabled
    return next_count, next_latched, just_enabled
