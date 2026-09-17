"""Read-only behavior checks for a NEW uninterrupted .5 m/s diagnostic.

This does not certify source identity, simulator continuity, policy interfaces,
initialization, or video authenticity. Those require the independent live trace.
The strict recovery hold and the old moving/rest thresholds remain separate.
No four-contact or standing-geometry constraint is imposed during movement.
"""

import math
import statistics
import struct

PROTOCOL = "continuous_flow_motion_metrics_v2_quiet_ready"
LEGS = ("FL", "FR", "RL", "RR")
FEET = tuple(leg + "_foot" for leg in LEGS)
JOINTS = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in LEGS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Missing/nonfinite numeric measurement")
    return value


def flag(value):
    require(type(value) in (int, bool) and value in (0, 1), "Expected an explicit measured boolean")
    return bool(value)


def f32(value):
    """Encode scalar constants like the actual float32 tensor comparisons."""
    return struct.unpack("f", struct.pack("f", value))[0]


def percentile(values, percent=95):
    """Linear interpolation, matching the old NumPy default percentile."""
    require(bool(values), "No samples for percentile")
    values = sorted(number(v) for v in values)
    position = (len(values) - 1) * percent / 100
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def measured_stance(row):
    """Recompute geometry/support/strict flags from saved measurements."""
    joint = [number(row[name + "_offset_rad"]) for name in JOINTS]
    geometry = max(abs(value) for value in joint) < f32(.65)
    vertical = 0
    for index, (leg, foot) in enumerate(zip(LEGS, FEET)):
        side = 1 if index % 2 == 0 else -1
        fore = 1 if index < 2 else -1
        x, y = number(row[foot + "_x_b"]), number(row[foot + "_y_b"])
        knee = number(row[leg + "_knee_y_b"])
        geometry = (geometry and side * y > f32(.06) and abs(y) < f32(.30)
                    and side * knee > f32(.04) and fore * x > f32(.08))
        fz = number(row[foot + "_fz"])
        contact = fz > 5.
        require(flag(row[foot + "_vertical_contact"]) == contact, "Current vertical contact disagrees with force")
        vertical += int(contact)
        norm_contact = flag(row[foot + "_contact"])  # CURRENT force-norm contact, not vertical or history union.
        require(norm_contact or abs(fz) <= 5., "Force-norm contact cannot be absent when abs vertical force exceeds5N")
        require(number(row[foot + "_speed_xy"]) >= 0, "Negative foot speed")
        number(row[foot + "_z_w"])
    base_force = number(row["current_base_force_n"])
    require(base_force >= 0, "Negative current base force norm")
    base = base_force > 1.
    require(not base or flag(row["base_contact"]), "Current base contact cannot be absent from force history")
    require(flag(row["stance_geometry_ok"]) == geometry, "Geometry label differs from measured feet/knees/joints")
    require(type(row["current_vertical_foot_contacts"]) is int and
            row["current_vertical_foot_contacts"] == vertical, "Current four-foot count differs")
    require(flag(row["current_base_contact"]) == base, "Current base contact differs from force")
    gravity = row["gravity_b"]
    require(type(gravity) in (list, tuple) and len(gravity) == 3, "Missing measured gravity vector")
    components = [f32(number(value) - target) for value, target in zip(gravity, (0., 0., -1.))]
    squared = [f32(value * value) for value in components]
    error = f32(math.sqrt(f32(f32(squared[0] + squared[1]) + squared[2])))
    # GPU reduction order may differ within a few float32 ULPs. Do not silently
    # widen the strict threshold or claim agreement at that ambiguous boundary.
    require(abs(error - f32(.35)) > 4 * (2 ** -25), "Gravity norm too close to float32 strict boundary for independent audit")
    height = number(row["height"])
    linear = number(row["linear_speed_3d"])
    angular = number(row["angular_speed_3d"])
    require(linear >= 0 and angular >= 0 and number(row["speed"]) >= 0 and number(row["yaw_speed"]) >= 0,
            "Negative norm")
    strict = (geometry and vertical == 4 and not base and f32(.30) < height < f32(.55)
              and error < f32(.35) and linear < .5 and angular < 1.)
    require(type(row["strict_valid_after"]) is bool and row["strict_valid_after"] == strict,
            "Strict hold label differs from measured current state")
    require(type(row["quiet_for_handoff"]) is bool and row["quiet_for_handoff"] == (linear < .06 and angular < .15),
            "Quiet readiness label differs from measured3D speeds")
    return geometry, vertical, base, strict


def phase_stats(rows):
    require(bool(rows), "Missing physical phase")
    stances = [measured_stance(row) for row in rows]
    slips = [number(row[foot + "_speed_xy"]) for row in rows for foot in FEET if flag(row[foot + "_contact"])]
    return {
        "samples": len(rows),
        "height_min": min(number(row["height"]) for row in rows),
        "vx_b_mean": statistics.mean(number(row["vx_b"]) for row in rows),
        "vy_b_mean": statistics.mean(number(row["vy_b"]) for row in rows),
        "yaw_rate_mean": statistics.mean(number(row["wz"]) for row in rows),
        "speed_p95": percentile([row["speed"] for row in rows]),
        "yaw_speed_p95": percentile([row["yaw_speed"] for row in rows]),
        "max_tilt_deg": max(max(abs(number(row["roll_deg"])), abs(number(row["pitch_deg"]))) for row in rows),
        "foot_height_p95": {foot: percentile([row[foot + "_z_w"] for row in rows]) for foot in FEET},
        "contact_slip_mean": statistics.mean(slips) if slips else None,
        "four_feet_norm_contact_fraction": sum(all(flag(row[foot + "_contact"]) for foot in FEET) for row in rows) / len(rows),
        "stance_geometry_fraction": sum(s[0] for s in stances) / len(rows),
        "four_feet_vertical_contact_fraction": sum(s[1] == 4 for s in stances) / len(rows),
        "current_base_clear_fraction": sum(not s[2] for s in stances) / len(rows),
        "stance_geometry_and_support_fraction": sum(s[0] and s[1] == 4 and not s[2] for s in stances) / len(rows),
        "strict_valid_fraction": sum(s[3] for s in stances) / len(rows),
    }


def analyze(rows, *, move_speed=.5):
    """Analyze complete off-refinement preparation/recovery/move/stop records.

Malformed or incomplete evidence raises. Real behavioral failures yield false
checks. A failed/early-stopped runtime must preserve its failure receipt and
must not call this full-sequence analyzer to fabricate missing phases.
    """
    require(type(move_speed) is float and move_speed == .5, "Initial bounded diagnostic supports cmd .5 only")
    require(type(rows) is list and bool(rows), "Need the entire measured episode")
    allowed = ("preparation", "recovery", "move", "stop")
    phase_order = []
    previous = None
    interval = 0
    for index, row in enumerate(rows):
        require(type(row["step"]) is int and row["step"] == index, "Missing/reordered global completed intervals")
        require(abs(number(row["time_s"]) - (index + 1) * .02) < 1e-9, "Noncontinuous physical timestamps")
        phase = row["phase"]
        require(phase in allowed, "Unsupported phase; no silently ignored rows")
        if phase != previous:
            phase_order.append(phase)
            interval = 0
            previous = phase
        interval += 1
        require(type(row["phase_interval"]) is int and row["phase_interval"] == interval,
                "Invalid within-phase completed-interval counter")
        expected = (move_speed, 0., 0.) if phase == "move" else (0., 0., 0.)
        require(tuple(number(row[key]) for key in ("cmd_x", "cmd_y", "cmd_yaw")) == expected,
                "Actual measured command differs from this bounded protocol")
        if phase in ("move", "stop"):
            require(row["actor_role"] == "locomotion", "Motion/stop did not retain locomotion actor")
        elif phase == "recovery":
            require(row["actor_role"] in ("roll", "stand"), "Unexpected recovery actor")
        else:
            require(row["actor_role"] == "nominal_pd", "Undisclosed preparation controller")
        flag(row["done"])
        flag(row["base_contact"])
        measured_stance(row)
    require(phase_order == list(allowed), "Require one uninterrupted preparation->recovery->move->stop sequence")
    grouped = {phase: [row for row in rows if row["phase"] == phase] for phase in allowed}
    require(len(grouped["preparation"]) == 50, "Retained preparation is exactly50control-equivalent intervals")
    require(150 <= len(grouped["recovery"]) <= 550, "Recovery exceeded its finite hold budget")
    require(len(grouped["move"]) == 400 and len(grouped["stop"]) == 300, "Require complete8s motion/6s stop")
    standing = grouped["recovery"][-150:]
    require(all(row["actor_role"] == "stand" for row in standing), "Recovery hold not controlled by standing actor")
    # A strict-valid hold is a behavior check, not assumed from a phase label.
    settled = {"stand": phase_stats(standing), "move": phase_stats(grouped["move"][50:]),
               "stop": phase_stats(grouped["stop"][50:])}
    rest = (settled["stand"], settled["stop"])
    post = standing + grouped["move"] + grouped["stop"]
    moving = settled["move"]
    checks = {
        "recorded_no_done_whole_episode": not any(flag(row["done"]) for row in rows),
        "recovery_last150_strict": all(row["strict_valid_after"] for row in standing),
        "recovery_last150_quiet_ready": all(row["quiet_for_handoff"] for row in standing),
        "stop_last150_strict": all(row["strict_valid_after"] for row in grouped["stop"][-150:]),
        "no_base_contact_after_recovery_hold_start": not any(flag(row["base_contact"]) or flag(row["current_base_contact"]) for row in post),
        "supported_height": all(stats["height_min"] >= .25 for stats in settled.values()),
        "level": all(stats["max_tilt_deg"] < 10 for stats in settled.values()),
        "walk_tracking": abs(moving["vx_b_mean"] - move_speed) < .12,
        "straight_drift": abs(moving["vy_b_mean"]) < .12 and abs(moving["yaw_rate_mean"]) < .15,
        "quiet_stand_stop": all(stats["speed_p95"] < .06 for stats in rest),
        "feet_lift_in_walk": all(z > .04 for z in moving["foot_height_p95"].values()),
        "limited_slip": moving["contact_slip_mean"] is not None and moving["contact_slip_mean"] < .12,
        "four_feet_at_rest": all(stats["four_feet_norm_contact_fraction"] > .95 for stats in rest),
        "normal_stance_geometry_at_rest": all(stats["stance_geometry_fraction"] == 1. for stats in rest),
        "four_vertical_contacts_at_rest": all(stats["four_feet_vertical_contact_fraction"] > .95 for stats in rest),
        "no_current_base_contact_at_rest": all(stats["current_base_clear_fraction"] == 1. for stats in rest),
        "geometry_and_support_at_rest": all(stats["stance_geometry_and_support_fraction"] > .95 for stats in rest),
    }
    return {"protocol": PROTOCOL, "checks": checks, "motion_checks_passed": all(checks.values()),
            "settled_phase_stats": settled, "observed_phase_intervals": {key: len(value) for key, value in grouped.items()},
            "promotion_performed": False,
            "scope": "Bounded cmd .5 development behavior screen only. Stand window is last150 recovery intervals; moving350 and stopping250 intervals exclude first1s. Requires separate live interface/reset/physical-continuity audit. Not trot/run/generalization/hardware acceptance."}
