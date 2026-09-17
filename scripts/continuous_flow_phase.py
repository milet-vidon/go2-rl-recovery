"""Pure bounded sequencing for a FUTURE continuous multi-actor diagnostic.

No environment, sensor, action history, PD, policy or reset API exists here.
``strict_valid``/``quiet_for_handoff``/``fallen``/``finite`` are measurements supplied by the caller;
this module does NOT verify their physical meaning or certify locomotion.

Usage: create initial_state at the actual policy-start control counter, obtain
next_action BEFORE inference, issue exactly one real step, then submit that
completed interval. The returned phase selects the NEXT action, never the one
just measured. All states/plans are immutable; prior states remain available
for logging. The old roll/stand selector and actual-start mirror mask remain
external and must never be cleared or recomputed at these phase edges.
If initial1s nominal-PD preparation is retained, the caller must record it in
the same physical episode and must not reset managers/history afterward. This
helper neither performs preparation nor grants permission to reset anything.

The shared550-interval zero-command budget includes optional refinement. Both
standing holds require150 consecutive strict AND quiet completed intervals controlled by their
respective standing actor. Move400, then stop300 with its LAST150 strict. A
later fall/done fails instead of restarting. Timing completion is NOT physical
acceptance: existing motion, slip, drift, quiet-stop and continuity checks must
still be independently applied by a future integration.

V3 readiness is deliberately STRONGER than the original strict recovery flag:
the caller independently measures3D linear speed<.06m/s and3D angular speed<.15rad/s.
Strict flags are never relabeled to hide otherwise valid but moving recovery.
The550-interval budget and the original roll->stand selector are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import struct


PROTOCOL = "continuous_flow_phase_logic_v3_quiet_ready"
CONTROL_DT = .02
STRICT_HOLD_INTERVALS = 150
PREMOVE_LIMIT = 550
MOVE_INTERVALS = 400
STOP_INTERVALS = 300
ZERO_COMMAND = (0.0, 0.0, 0.0)
PHASES = frozenset(("recovery", "refinement", "move", "stop", "complete", "failed"))
TERMINAL = frozenset(("complete", "failed"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command_ok(command):
    require(type(command) is tuple and len(command) == 3 and
            all(type(x) is float and math.isfinite(x) for x in command),
            "Command must be an immutable finite three-float body-frame tuple")


def float32_command(command):
    """Encode the requested command exactly as the verified float32 manager does.

    Only the PLAN is encoded. Never round/snap a measured command before checking
    it: measured values must already equal the exact float32 representation.
    This is a declared transport dtype, not an approximate command tolerance.
    """
    command_ok(command)
    try:
        encoded = tuple(struct.unpack("<f", struct.pack("<f", x))[0] for x in command)
    except OverflowError as exc:
        raise ValueError("Command is not representable as finite float32") from exc
    command_ok(encoded)
    return encoded


@dataclass(frozen=True)
class FlowConfig:
    move_speed: float = .5
    include_refinement: bool = False

    def __post_init__(self):
        require(type(self.move_speed) is float and self.move_speed in (.5, .8),
                "Only separately validated .5 or .8 m/s diagnostic commands are supported")
        require(type(self.include_refinement) is bool, "Refinement enable must be explicit bool")


@dataclass(frozen=True)
class FlowState:
    config: FlowConfig
    initial_control_step: int
    phase: str = "recovery"
    completed_intervals: int = 0
    premove_intervals: int = 0
    move_intervals: int = 0
    stop_intervals: int = 0
    recovery_strict_count: int = 0
    recovery_ready_count: int = 0
    refinement_strict_count: int = 0
    refinement_ready_count: int = 0
    stop_strict_count: int = 0
    old_stand_branch_latched: bool = False
    recovery_hold_completed_at: int | None = None
    refinement_hold_completed_at: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ActionPlan:
    absolute_control_step: int
    phase: str
    actor_role: str
    real_command: tuple[float, float, float]
    recovery_stand_active: bool | None


@dataclass(frozen=True)
class CompletedInterval:
    control_step_before: int
    control_step_after: int
    command_before: tuple[float, float, float]
    command_after: tuple[float, float, float]
    strict_valid: bool
    fallen: bool
    done: bool
    finite: bool
    quiet_for_handoff: bool


def check_state(state):
    require(type(state) is FlowState and type(state.config) is FlowConfig, "Wrong state/config type")
    require(state.phase in PHASES, "Unknown phase")
    for name in ("initial_control_step", "completed_intervals", "premove_intervals", "move_intervals",
                 "stop_intervals", "recovery_strict_count", "recovery_ready_count", "refinement_strict_count", "refinement_ready_count", "stop_strict_count"):
        require(type(getattr(state, name)) is int and getattr(state, name) >= 0, "Invalid counter: " + name)
    require(state.completed_intervals == state.premove_intervals + state.move_intervals + state.stop_intervals,
            "Phase counters do not sum to the monotonic completed-interval count")
    require(state.premove_intervals <= PREMOVE_LIMIT and state.move_intervals <= MOVE_INTERVALS
            and state.stop_intervals <= STOP_INTERVALS, "Stage budget exceeded")
    require(all(getattr(state, name) <= STRICT_HOLD_INTERVALS for name in
                ("recovery_strict_count", "recovery_ready_count", "refinement_strict_count", "refinement_ready_count", "stop_strict_count")), "Hold counter overflow")
    require(state.recovery_ready_count <= state.recovery_strict_count and
            state.refinement_ready_count <= state.refinement_strict_count, "Quiet readiness cannot exceed strict hold")
    require(type(state.old_stand_branch_latched) is bool, "Invalid old-selector latch")
    for name in ("recovery_hold_completed_at", "refinement_hold_completed_at"):
        value = getattr(state, name)
        require(value is None or (type(value) is int and state.initial_control_step < value <=
                                  state.initial_control_step + state.premove_intervals), "Invalid completed hold boundary")
    if state.phase == "failed":
        require(type(state.failure_reason) is str and bool(state.failure_reason), "Failed state needs an explicit reason")
    else:
        require(state.failure_reason is None, "Nonfailed state cannot contain a failure reason")
    if state.phase in ("refinement", "move", "stop", "complete"):
        require(state.recovery_hold_completed_at is not None and state.recovery_strict_count == 150 and state.recovery_ready_count == 150
                and state.old_stand_branch_latched, "Later phase lacks completed original-standing hold")
    if state.phase == "refinement":
        require(state.config.include_refinement and state.refinement_hold_completed_at is None,
                "Unexpected refinement phase")
    if state.phase in ("move", "stop", "complete") and state.config.include_refinement:
        require(state.refinement_hold_completed_at is not None and state.refinement_strict_count == 150 and state.refinement_ready_count == 150,
                "Moving before the optional refinement hold completed")
    if not state.config.include_refinement:
        require(state.refinement_strict_count == 0 and state.refinement_ready_count == 0 and state.refinement_hold_completed_at is None,
                "Disabled refinement contains history")
    if state.phase in ("recovery", "refinement"):
        require(state.premove_intervals < PREMOVE_LIMIT and state.move_intervals == state.stop_intervals == 0,
                "Zero-command phase already timed out or contains motion history")
    if state.phase == "recovery":
        require(state.recovery_hold_completed_at is None and state.recovery_ready_count < 150,
                "Completed original-standing hold did not select the next phase")
    if state.phase == "move":
        require(state.move_intervals < MOVE_INTERVALS and state.stop_intervals == 0, "Move phase budget invalid")
    if state.phase in ("stop", "complete"):
        require(state.move_intervals == MOVE_INTERVALS, "Stop began before400 moving intervals")
    if state.phase == "stop":
        require(state.stop_intervals < STOP_INTERVALS, "Stop phase already exhausted")
    if state.phase == "complete":
        require(state.stop_intervals == STOP_INTERVALS and state.stop_strict_count == 150,
                "Sequence completion lacks its last150 strict stop intervals")


def initial_state(config=None, *, actual_control_step=0):
    """Create logical bookkeeping only; caller owns real initialization/history."""
    state = FlowState(FlowConfig() if config is None else config, actual_control_step)
    check_state(state)
    return state


def next_action(state, *, recovery_stand_active=None):
    """Command for the REAL observation; never construct alternate command views.

    In recovery, supply the unchanged old selector AFTER its normal update.
    Its roll/stand actors consume zero real command. All later phases reject
    an old-selector argument: only refinement or locomotion is called then.
    Stop keeps the locomotion actor active, with zero real command.
    """
    check_state(state)
    require(state.phase not in TERMINAL, "Terminal sequences cannot issue more actions")
    if state.phase == "recovery":
        require(type(recovery_stand_active) is bool, "Recovery needs the actual old roll/stand selector")
        require(not state.old_stand_branch_latched or recovery_stand_active, "Old recovery selector cannot unlatch")
        role = "stand" if recovery_stand_active else "roll"
        command = ZERO_COMMAND
    else:
        require(recovery_stand_active is None, "Do not call the old recovery selector during later phases")
        role = "refinement" if state.phase == "refinement" else "locomotion"
        command = float32_command((state.config.move_speed, 0.0, 0.0)) if state.phase == "move" else ZERO_COMMAND
    return ActionPlan(state.initial_control_step + state.completed_intervals, state.phase,
                      role, command, recovery_stand_active)


def after_completed_interval(state, plan, measured):
    """Consume one completed real interval and return immutable NEXT-action state.

    Both actual control counters are required. A t=0 observation, repeated
    interval, reset counter or skipped sample is rejected, never counted as
    persistence. Invalid sample types raise; actual done/fall/nonfinite/command
    errors fail the trial with the completed issued interval still counted.
    """
    check_state(state)
    require(type(plan) is ActionPlan and type(measured) is CompletedInterval, "Wrong plan/interval type")
    require(type(plan.absolute_control_step) is int, "Plan needs an actual integer control index")
    command_ok(plan.real_command)
    expected = next_action(state, recovery_stand_active=plan.recovery_stand_active)
    require(plan == expected, "Plan is stale or does not match the issued phase/actor/command")
    for name in ("control_step_before", "control_step_after"):
        require(type(getattr(measured, name)) is int, "Measured control indices must be integers")
    require(measured.control_step_before == plan.absolute_control_step and
            measured.control_step_after == plan.absolute_control_step + 1,
            "Require exactly one completed interval, with monotonic actual counters")
    for name in ("strict_valid", "fallen", "done", "finite", "quiet_for_handoff"):
        require(type(getattr(measured, name)) is bool, "Measurement flags must be explicit bool: " + name)
    command_ok(measured.command_before)
    command_ok(measured.command_after)
    counts = {"completed_intervals": state.completed_intervals + 1}
    field = "premove_intervals" if state.phase in ("recovery", "refinement") else state.phase + "_intervals"
    counts[field] = getattr(state, field) + 1
    current = replace(state, **counts)
    reason = ("nonfinite_state" if not measured.finite else "unexpected_done" if measured.done else
              "command_mismatch" if measured.command_before != plan.real_command or measured.command_after != plan.real_command else
              "inconsistent_strict_and_fallen" if measured.fallen and measured.strict_valid else
              "later_fall" if measured.fallen and state.phase != "recovery" else None)
    if reason:
        failed = replace(current, phase="failed", failure_reason=reason)
        check_state(failed)
        return failed
    if state.phase == "recovery":
        count = min(150, state.recovery_strict_count + 1) if measured.strict_valid and plan.recovery_stand_active else 0
        ready = min(150, state.recovery_ready_count + 1) if measured.strict_valid and measured.quiet_for_handoff and plan.recovery_stand_active else 0
        current = replace(current, recovery_strict_count=count,
                          recovery_ready_count=ready,
                          old_stand_branch_latched=state.old_stand_branch_latched or plan.recovery_stand_active)
        if ready == STRICT_HOLD_INTERVALS:
            current = replace(current, phase="refinement" if state.config.include_refinement else "move",
                              recovery_hold_completed_at=measured.control_step_after)
    elif state.phase == "refinement":
        count = min(150, state.refinement_strict_count + 1) if measured.strict_valid else 0
        ready = min(150, state.refinement_ready_count + 1) if measured.strict_valid and measured.quiet_for_handoff else 0
        current = replace(current, refinement_strict_count=count, refinement_ready_count=ready)
        if ready == STRICT_HOLD_INTERVALS:
            current = replace(current, phase="move", refinement_hold_completed_at=measured.control_step_after)
    elif state.phase == "move":
        if current.move_intervals == MOVE_INTERVALS:
            current = replace(current, phase="stop")
    elif state.phase == "stop":
        count = min(STRICT_HOLD_INTERVALS, state.stop_strict_count + 1) if measured.strict_valid else 0
        current = replace(current, stop_strict_count=count)
        if current.stop_intervals == STOP_INTERVALS:
            current = replace(current, phase="complete" if count == STRICT_HOLD_INTERVALS else "failed",
                              failure_reason=None if count == STRICT_HOLD_INTERVALS else "stop_last150_not_strict")
    if current.phase in ("recovery", "refinement") and current.premove_intervals == PREMOVE_LIMIT:
        current = replace(current, phase="failed", failure_reason="shared_premove_budget_exhausted")
    check_state(current)
    return current
