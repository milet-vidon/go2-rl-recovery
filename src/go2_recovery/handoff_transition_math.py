"""Pure tensor hard/raw-ramp diagnostic; no simulator, selector or state reset.

The existing one-way gate owns switched/just_switched. This module only
constructs the action issued through the existing action manager. It is not a
recovery controller or a learned/published behavior selector reproduction.
"""

from collections import deque
import math

import torch


def ramp_step_count(seconds, step_dt):
    if type(seconds) not in (int, float) or seconds not in (0, .2, .5):
        raise ValueError("Ramp seconds must be exactly 0, 0.2 or 0.5")
    if type(step_dt) not in (int, float) or not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("Control dt must be finite and positive")
    count = seconds / step_dt
    if not math.isclose(count, round(count), abs_tol=1e-9, rel_tol=0):
        raise ValueError("Ramp must contain an integer number of control samples")
    return round(count)


def transition_action(roll_action, stand_action, previous_raw_action, switched,
                      just_switched, anchor, completed_steps, total_ramp_steps):
    """Return issued action, copied anchor, next counter, alpha; never mutate.

    On an edge the anchor is the real previous issued raw action. The first
    ramp output uses alpha=1/N, the Nth is EXACTLY the current stand output.
    T=0 uses the original torch.where hard branch, with no interpolation.
    Alpha is zero before handoff and one after completion. No RNG is consumed.
    """
    if type(total_ramp_steps) is not int or total_ramp_steps < 0:
        raise ValueError("total_ramp_steps must be a nonnegative integer")
    floats = (roll_action, stand_action, previous_raw_action, anchor)
    if any(not isinstance(value, torch.Tensor) for value in (*floats, switched, just_switched, completed_steps)):
        raise TypeError("All state inputs must be tensors")
    if roll_action.ndim != 2 or roll_action.shape[1] != 12 or not roll_action.is_floating_point():
        raise ValueError("Actions must have floating [N,12] native joint layout")
    n, device, dtype = roll_action.shape[0], roll_action.device, roll_action.dtype
    for value in floats:
        if value.shape != (n, 12) or value.device != device or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError("Action/anchor shape, dtype, device or finiteness mismatch")
    for value in (switched, just_switched):
        if value.shape != (n,) or value.device != device or value.dtype != torch.bool:
            raise ValueError("Switch masks must be matching [N] boolean tensors")
    if completed_steps.shape != (n,) or completed_steps.device != device or completed_steps.dtype != torch.long:
        raise ValueError("Counters must be matching [N] int64 tensors")
    if bool((just_switched & ~switched).any()) or bool((completed_steps < 0).any()):
        raise ValueError("Invalid edge/counter state")
    if bool((completed_steps[~switched] != 0).any()) or bool((completed_steps[just_switched] != 0).any()):
        raise ValueError("Unswitched/new-edge rows cannot have previous ramp progress")
    next_anchor = torch.where(just_switched[:, None], previous_raw_action, anchor)
    if total_ramp_steps == 0:
        return (torch.where(switched[:, None], stand_action, roll_action), next_anchor,
                torch.zeros_like(completed_steps), switched.to(dtype=dtype))
    if bool((completed_steps > total_ramp_steps).any()) or bool((switched & ~just_switched & (completed_steps == 0)).any()):
        raise ValueError("Latched ramp progress is missing or beyond its bound")
    next_steps = torch.where(switched, completed_steps.clamp(max=total_ramp_steps-1) + 1,
                             torch.zeros_like(completed_steps))
    alpha = next_steps.to(dtype=dtype) / total_ramp_steps
    ramped = (1-alpha[:, None]) * next_anchor + alpha[:, None] * stand_action
    # Explicit endpoint branch avoids 0*anchor arithmetic changing exact hard
    # stand output and lets CPU/live equivalence tests require torch.equal.
    selected = torch.where((next_steps < total_ramp_steps)[:, None], ramped, stand_action)
    action = torch.where(switched[:, None], selected, roll_action)
    if not torch.isfinite(action).all():
        raise ValueError("Interpolation overflow produced a nonfinite issued action")
    return action, next_anchor, next_steps, alpha


class TransitionNeighborhood:
    """Bounded per-trial diagnostic rows; observation/log data only, never replay."""

    def __init__(self, trials, before_steps=50, after_steps=50):
        if any(type(value) is not int or value < 1 for value in (trials, before_steps, after_steps)):
            raise ValueError("Positive integer trace dimensions required")
        self.before_steps, self.after_steps = before_steps, after_steps
        self.rings = [deque(maxlen=before_steps) for _ in range(trials)]
        self.rows = [[] for _ in range(trials)]
        self.switch_steps = [None] * trials
        self.events = [[] for _ in range(trials)]
        self.phases = ["roll"] * trials

    def wants_row(self, trial, step):
        switch = self.switch_steps[trial]
        return switch is None or step <= switch + self.after_steps

    def record(self, trial, step, row, just_switched):
        if just_switched:
            if self.switch_steps[trial] is not None:
                raise ValueError("One-way experiment cannot switch twice")
            self.switch_steps[trial] = step
            self.rows[trial].extend(self.rings[trial])
            self.rings[trial].clear()
        phase = row["phase"]
        if phase != self.phases[trial]:
            self.events[trial].append({"step": step, "policy_time_s": row["policy_time_s"],
                "from": self.phases[trial], "to": phase, "alpha": row["alpha"],
                "reason": "unchanged forward gate" if just_switched else "raw ramp completed"})
            self.phases[trial] = phase
        if self.switch_steps[trial] is None:
            self.rings[trial].append(row)
        elif step <= self.switch_steps[trial] + self.after_steps:
            self.rows[trial].append(row)

    def report(self):
        return [{"trial": i, "triggered": switch is not None, "switch_step": switch,
                 "events": self.events[i], "untriggered_tail_only": switch is None,
                 "rows": self.rows[i] if switch is not None else list(self.rings[i])}
                for i, switch in enumerate(self.switch_steps)]
