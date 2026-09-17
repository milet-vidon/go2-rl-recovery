"""Tensor-only, one-way policy handoff gate for a diagnostic experiment.

This selects a policy; it is NOT a standing/recovery success criterion. It
does not change physical state, actions, observation history, or policy state.
"""

import math
from numbers import Real

import torch


HANDOFF_MAX_TILT_DEG = 30.0
HANDOFF_MAX_ANGULAR_SPEED = 1.0
HANDOFF_HOLD_S = 0.2
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def update_handoff_gate(
    gravity: torch.Tensor,
    angular_velocity: torch.Tensor,
    previous_steps: torch.Tensor,
    switched: torch.Tensor,
    step_dt: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return consecutive gate steps, latched switch state, and switch edge.

    Inputs have shapes (N,3), (N,3), integer (N,), and boolean (N,).
    Gravity and angular velocity use the same floating dtype and all inputs
    share a device. Gravity is normalized before testing tilt. Nonfinite
    vectors and zero/nonfinite gravity norm cannot trigger a switch.

    Tilt must be strictly below 30 degrees and angular speed strictly below
    1 rad/s for at least 0.2 s (10 control samples at 50 Hz). Invalid samples
    clear the consecutive counter. Counters saturate at the required hold;
    negative incoming counters conservatively restart from zero. A previous
    switch remains latched even if the gate later fails. No input is mutated.
    """
    if isinstance(step_dt, bool) or not isinstance(step_dt, Real):
        raise TypeError("step_dt must be a finite positive Python real scalar")
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and strictly positive")
    inputs = (gravity, angular_velocity, previous_steps, switched)
    if any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("All state inputs must be tensors")
    if not gravity.is_floating_point() or not angular_velocity.is_floating_point():
        raise TypeError("Gravity and angular velocity must be floating tensors")
    if gravity.ndim != 2 or gravity.shape[1] != 3:
        raise ValueError("gravity must have shape (N,3)")
    n = gravity.shape[0]
    if angular_velocity.shape != gravity.shape:
        raise ValueError("angular_velocity must share gravity's shape (N,3)")
    if previous_steps.shape != (n,) or switched.shape != (n,):
        raise ValueError("Counter and switch state must have shape (N,)")
    if previous_steps.dtype not in _INTEGER_DTYPES or switched.dtype != torch.bool:
        raise TypeError("Counter must be integer and switch state boolean")
    if angular_velocity.dtype != gravity.dtype:
        raise ValueError("Gravity and angular velocity must share a dtype")
    if any(value.device != gravity.device for value in inputs):
        raise ValueError("All inputs must share a device")

    ratio = HANDOFF_HOLD_S / float(step_dt)
    if not math.isfinite(ratio):
        raise ValueError("step_dt is too small for the hold duration")
    # Avoid an extra step from one-ULP roundoff for exact decimal multiples.
    required_steps = max(1, math.ceil(math.nextafter(ratio, -math.inf)))
    if required_steps > torch.iinfo(previous_steps.dtype).max:
        raise ValueError("Counter dtype cannot represent the required hold")

    gravity_norm = torch.linalg.vector_norm(gravity, dim=-1)
    angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1)
    finite = (torch.isfinite(gravity).all(-1)
              & torch.isfinite(angular_velocity).all(-1)
              & torch.isfinite(gravity_norm) & torch.isfinite(angular_speed))
    nonzero_gravity = gravity_norm > 0
    denominator = torch.where(nonzero_gravity, gravity_norm, torch.ones_like(gravity_norm))
    cos_up = -gravity[:, 2] / denominator
    valid = (finite & nonzero_gravity
             & (cos_up > math.cos(math.radians(HANDOFF_MAX_TILT_DEG)))
             & (angular_speed < HANDOFF_MAX_ANGULAR_SPEED))
    # Clamp before adding so even a saturated integer input cannot overflow.
    next_count = previous_steps.clamp(min=0, max=required_steps - 1) + 1
    new_counter = torch.where(valid, next_count, torch.zeros_like(previous_steps))
    just_switched = ~switched & (new_counter >= required_steps)
    new_switched = switched | just_switched
    return new_counter, new_switched, just_switched
