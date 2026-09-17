"""Pure raw-action proposal for a FUTURE stand-to-refinement comparison.

This is NOT connected to any evaluator, video, environment or training task.
The original supported_refinement_gate owns the completed150-strict-interval
latch. Call this helper only on the original STANDING branch, never instead of
the roll->stand gate. The latch from a completed interval affects the NEXT
issued action; no t=0 sample or interpolation sample can satisfy that gate.

Both endpoints are recomputed from the SAME current real observation, including
the LAST ACTUALLY ISSUED (possibly mixed) action. They are actor proposals, not
two independently executed actions. The old endpoint is NOT a frozen anchor.
Do not write either unissued proposal into real action/observation history.

In linear25 mode, the first affected action uses alpha=1/25, the25th uses
alpha=1 EXACTLY, then every subsequent output is exactly the current refiner.
The proposed counter returned here may be committed only after that actual
control interval completes. Calling this pure function does not execute a step.
Hard mode provides the explicit direct-switch comparison, not an alternate
standing criterion. Nothing here resets history, applies PD, clamps/scales a
target, filters a sensor, restarts a failed transition, or certifies smoothness.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


PROTOCOL = "supported_refinement_live_endpoints_raw_transition_v1"
CONTROL_DT = .02
LINEAR_INTERVALS = 25
MODES = ("hard", "linear25")


def transition_action(
    old_stand_action: torch.Tensor,
    refinement_action: torch.Tensor,
    refinement_enabled: torch.Tensor,
    just_enabled: torch.Tensor,
    completed_transition_intervals: torch.Tensor,
    *,
    mode: str = "linear25",
    dt: float = CONTROL_DT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (issued_raw12, proposed_completed_count, alpha), without mutation.

    Actions: finite matching dense float32/float64 [N,12], native joint order,
    dimensionless raw units. Masks: matching bool[N]. Counter: int64[N], zero
    before the external latch and on its one-time edge. After an issued and
    completed interval, retain the proposed counter for the next call; it
    saturates at25 in linear mode or1 in hard mode. A latch cannot be cleared
    by passing a previously progressed row as disabled; that input is rejected.

    The helper cannot prove that supplied actor outputs came from current real
    observations, that the external gate measured150 valid intervals, or that
    any returned action was actually issued. A future live adapter must record
    those facts. Alpha progression alone is not a physical smoothness guarantee.
    """
    if mode not in MODES:
        raise ValueError("Explicit mode must be hard or linear25")
    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise TypeError("dt must be a real50Hz interval")
    if not math.isfinite(float(dt)) or float(dt) != CONTROL_DT:
        raise ValueError("Exactly0.02s control intervals are required")
    if not isinstance(old_stand_action, torch.Tensor):
        raise TypeError("Actor outputs must be tensors")
    if (old_stand_action.layout != torch.strided or old_stand_action.ndim != 2
            or old_stand_action.shape[0] == 0 or old_stand_action.shape[1] != 12
            or old_stand_action.dtype not in (torch.float32, torch.float64)
            or old_stand_action.device.type not in ("cpu", "cuda")):
        raise ValueError("Require nonempty native dense float32/float64 [N,12] actions on CPU/CUDA")
    count, device, dtype = old_stand_action.shape[0], old_stand_action.device, old_stand_action.dtype
    for value in (old_stand_action, refinement_action):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Actor outputs must be tensors")
        if (value.layout != torch.strided or value.shape != (count,12)
                or value.device != device or value.dtype != dtype):
            raise ValueError("Actor proposal shape/device/dtype mismatch")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite actor proposal")
    for name, value, expected in (("refinement_enabled", refinement_enabled, torch.bool),
                                  ("just_enabled", just_enabled, torch.bool),
                                  ("completed_transition_intervals", completed_transition_intervals, torch.int64)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + " must be a tensor")
        if value.layout != torch.strided or value.shape != (count,) or value.device != device or value.dtype != expected:
            raise ValueError(name + " has an invalid shape, device, layout or dtype")
    length = LINEAR_INTERVALS if mode == "linear25" else 1
    progress = completed_transition_intervals
    if bool(((progress < 0) | (progress > length)).any()):
        raise ValueError("Completed transition count outside this mode's finite bounds")
    if bool((just_enabled & ~refinement_enabled).any()):
        raise ValueError("A new edge requires the original gate's true latch")
    if bool(((~refinement_enabled | just_enabled) & (progress != 0)).any()):
        raise ValueError("Disabled or new-edge rows must have zero prior progress; no unlatch/retry")
    if bool((refinement_enabled & ~just_enabled & (progress == 0)).any()):
        raise ValueError("Enabled row with no progress requires a genuine first edge")

    proposed = torch.where(refinement_enabled, progress.clamp(max=length-1) + 1, torch.zeros_like(progress))
    alpha = proposed.to(dtype=dtype) / length
    if mode == "hard":
        issued = torch.where(refinement_enabled[:,None], refinement_action, old_stand_action)
    else:
        # Live endpoints, deliberately not (previous-issued anchor, new proposal).
        mixed = (1.-alpha[:,None]) * old_stand_action + alpha[:,None] * refinement_action
        # Exact endpoint branch avoids multiplication by zero altering endpoint
        # bits, including signed zero, and never adds a post-transition filter.
        selected = torch.where((proposed < length)[:,None], mixed, refinement_action)
        issued = torch.where(refinement_enabled[:,None], selected, old_stand_action)
    if not bool(torch.isfinite(issued).all()):
        raise ValueError("Nonfinite issued proposal after interpolation")
    return issued, proposed, alpha
