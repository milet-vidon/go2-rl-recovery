"""Read-only standard-library audit of the fixed two-policy handoff diagnostic.

Internal consistency is not policy acceptance or visual verification. Optional
checkpoint-file hashing verifies bytes only, never model behavior. No simulator,
torch, checkpoint deserialization, CSV, or video is used.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

from audit_recovery_report import (
    AuditError, PROTOCOLS, array, audit_report, boolean, finite_tree, load_report,
    number, ordered_records, require, vector,
)


SUFFIX = "_dual_policy_diagnostic_v1"
TASK = "Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0"
CONTROLLER = "two_policy_fixed_one_way_handoff_diagnostic"
GATE = "tilt<30deg and body angular speed<1rad/s continuously for0.2s; one-way latched"
COUNT_PREFIX = "DUAL-POLICY diagnostic counts, NOT performance of the roll checkpoint alone. "
NOTICE = "Internal consistency only; NOT model acceptance, single-policy recovery, or visual approval."
JOINT_NAMES = {f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
               for kind in ("hip", "thigh", "calf")}


def _sha(value, path):
    require(type(value) is str and re.fullmatch(r"[0-9a-fA-F]{64}", value),
            path, "expected SHA-256 string")
    return value.lower()


def _checkpoint_path(value, path):
    require(type(value) is str and value and "\x00" not in value,
            path, "expected absolute checkpoint path")
    win = PureWindowsPath(value)
    if win.is_absolute():
        require(win.suffix.lower() == ".pt", path, "expected .pt checkpoint")
        return str(win).casefold()
    posix = PurePosixPath(value)
    require(posix.is_absolute() and posix.suffix.lower() == ".pt",
            path, "expected absolute .pt checkpoint")
    return str(posix)


def _check_joint_limits(report, controller):
    limits = controller.get("joint_limits")
    require(type(limits) is dict, "handoff_controller.joint_limits", "expected object")
    names = array(limits.get("joint_names"), 12, "handoff_controller.joint_limits.joint_names")
    require(all(type(name) is str for name in names) and set(names) == JOINT_NAMES,
            "handoff_controller.joint_limits.joint_names", "expected all twelve named Go2 joints")
    require(report.get("joint_names") == names, "joint_names", "joint order differs from limit metadata")
    defaults = vector(limits.get("default_joint_positions_rad"), 12,
                      "handoff_controller.joint_limits.default_joint_positions_rad")
    bounds = array(limits.get("soft_joint_limits_rad"), 12,
                   "handoff_controller.joint_limits.soft_joint_limits_rad")
    outside = False
    for i, bound in enumerate(bounds):
        lo, hi = vector(bound, 2, f"handoff_controller.joint_limits.soft_joint_limits_rad[{i}]")
        require(lo < hi, f"soft_joint_limits_rad[{i}]", "expected lower < upper")
        outside |= not lo <= defaults[i] <= hi
    claimed = boolean(limits.get("default_target_would_be_clamped"), "default_target_would_be_clamped")
    require(claimed == outside, "default_target_would_be_clamped", "flag contradicts saved default/limits")


def _switch_record(record, trial, dt, end_time, path):
    require(type(record) is dict, path, "switch record must be object or null")
    require(number(record.get("trial"), f"{path}.trial", 0, integer=True) == trial,
            f"{path}.trial", "record does not match its trial position")
    time_s = number(record.get("policy_time_s"), f"{path}.policy_time_s", .2)
    require(time_s < end_time, f"{path}.policy_time_s", "switch must precede rollout end")
    require(math.isclose(time_s / dt, round(time_s / dt), rel_tol=0, abs_tol=1e-7),
            f"{path}.policy_time_s", "switch is not on a control-step boundary")
    for key, size in (("root_position_local_m", 3), ("root_quaternion_wxyz", 4),
                      ("root_linear_velocity_w_m_s", 3), ("root_angular_velocity_w_rad_s", 3),
                      ("joint_positions_rad", 12), ("joint_velocities_rad_s", 12)):
        vector(record.get(key), size, f"{path}.{key}")
    quaternion = record["root_quaternion_wxyz"]
    norm = math.hypot(*quaternion)
    require(abs(norm - 1.0) <= 1e-4, f"{path}.root_quaternion_wxyz", "quaternion is not unit length")
    w, x, y, z = (component / norm for component in quaternion)
    cos_up = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    tilt_deg = math.degrees(math.acos(cos_up))
    angular_speed = math.hypot(*record["root_angular_velocity_w_rad_s"])
    # Use the same strict cosine comparison as the gate; acos can round an
    # exactly-threshold angle to 29.999999999999996 degrees.
    require(cos_up > math.cos(math.radians(30.0)), path, "switch snapshot violates tilt <30 degrees")
    require(angular_speed < 1.0, path, "switch snapshot violates angular speed <1 rad/s")
    for key in ("action_linf_jump", "clamped_joint_target_linf_jump_rad", "two_actor_action_linf_difference"):
        number(record.get(key), f"{path}.{key}", 0)
    return {"trial": trial, "policy_time_s": time_s,
            "snapshot_tilt_deg": tilt_deg, "snapshot_angular_speed_rad_s": angular_speed}


def _verify_checkpoint_files(identities):
    verified = []
    for identity in identities:
        path = Path(identity["path"])
        try:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
        except OSError as error:
            raise AuditError(f"{identity['role']} checkpoint: {error}") from error
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                str(path), "checkpoint changed while hashing")
        require(digest.hexdigest() == identity["sha256"], str(path), "checkpoint file SHA-256 mismatch")
        verified.append({**identity, "bytes": after.st_size})
    return verified


def audit_handoff_report(report, *, verify_checkpoint_files=False):
    """Validate recorded facts without mutating report or inferring unseen history."""
    require(type(report) is dict, "report", "expected object")
    finite_tree(report)
    protocol = report.get("protocol_version")
    require(type(protocol) is str and protocol.endswith(SUFFIX), "protocol_version", "not a supported dual diagnostic")
    base_protocol = protocol[:-len(SUFFIX)]
    require(base_protocol in PROTOCOLS, "protocol_version", "unsupported underlying physical protocol")
    require(report.get("task") == TASK, "task", "expected fixed SmithNominal Play")
    require(report.get("controller_type") == CONTROLLER, "controller_type", "incorrect diagnostic controller type")
    require(report.get("single_policy_acceptance_eligible") is False,
            "single_policy_acceptance_eligible", "dual results cannot be single-policy acceptance")
    require(report.get("self_collisions_enabled") is True, "self_collisions_enabled", "fixed physics requires self collisions")
    definition = report.get("success_count_definition")
    require(type(definition) is str and definition.startswith(COUNT_PREFIX),
            "success_count_definition", "missing explicit dual-policy attribution")
    controller = report.get("handoff_controller")
    require(type(controller) is dict, "handoff_controller", "expected object")
    require(controller.get("gate") == GATE, "handoff_controller.gate", "changed gate/hold/latch definition")
    for key in ("state_or_action_history_reset_at_switch", "manual_standing_pd_at_switch", "physics_changed_at_switch"):
        require(controller.get(key) is False, f"handoff_controller.{key}", "must explicitly be false")

    roll_path = _checkpoint_path(controller.get("roll_checkpoint"), "handoff_controller.roll_checkpoint")
    require(roll_path == _checkpoint_path(report.get("checkpoint"), "checkpoint"),
            "handoff_controller.roll_checkpoint", "roll path differs from report checkpoint")
    roll_sha = _sha(controller.get("roll_sha256"), "handoff_controller.roll_sha256")
    require(roll_sha == _sha(report.get("checkpoint_sha256"), "checkpoint_sha256"),
            "handoff_controller.roll_sha256", "roll SHA differs from report checkpoint")
    stand_path = _checkpoint_path(controller.get("stand_checkpoint"), "handoff_controller.stand_checkpoint")
    stand_sha = _sha(controller.get("stand_sha256"), "handoff_controller.stand_sha256")
    require(roll_path != stand_path or roll_sha == stand_sha,
            "handoff_controller", "one checkpoint path cannot have two SHA identities")
    identities = [{"role": "roll", "path": controller["roll_checkpoint"], "sha256": roll_sha},
                  {"role": "stand", "path": controller["stand_checkpoint"], "sha256": stand_sha}]
    _check_joint_limits(report, controller)

    action = report.get("action_representation")
    require(type(action) is dict, "action_representation", "expected object")
    require(action.get("reference") == "nominal" and number(action.get("scale"), "action.scale") == .25,
            "action_representation", "changed nominal action representation")
    require(action.get("target") == "q_reference + scale * action, soft-joint-limit clamped",
            "action_representation.target", "changed target semantics")
    dt = number(action.get("sample_period_s"), "action.sample_period_s", .02, .02)
    number(action.get("held_over_physics_substeps"), "action.held_over_physics_substeps", 4, 4, integer=True)
    results = report.get("results")
    require(type(results) is dict and results, "results", "expected nonempty pose mapping")
    diagnostics = {}
    trial_counts = set()
    for pose, result in results.items():
        path = f"results.{pose}"
        require(type(result) is dict, path, "expected object")
        trials = number(result.get("trials"), f"{path}.trials", 1, integer=True)
        trial_counts.add(trials)
        hold = number(result.get("stable_hold_s"), f"{path}.stable_hold_s", .000001)
        horizon = number(result.get("horizon_s"), f"{path}.horizon_s", .000001)
        final_count = number(result.get("final_valid_stands"), f"{path}.final_valid_stands", 0, trials, integer=True)
        finals = ordered_records(result.get("final_diagnostics"), trials, f"{path}.final_diagnostics")
        info = result.get("handoff_diagnostic")
        require(type(info) is dict, f"{path}.handoff_diagnostic", "missing diagnostic")
        trigger = number(info.get("triggered_trials"), f"{path}.triggered_trials", 0, trials, integer=True)
        untriggered = number(info.get("untriggered_trials"), f"{path}.untriggered_trials", 0, trials, integer=True)
        require(trigger + untriggered == trials, path, "untriggered trials missing from total denominator")
        require(info.get("total_success_denominator_includes_untriggered") is True,
                path, "must explicitly retain untriggered trials in success denominator")
        records = array(info.get("switch_records"), trials, f"{path}.switch_records")
        require(sum(record is not None for record in records) == trigger,
                path, "trigger count does not match switch/non-switch records")
        snapshots = []
        for i, record in enumerate(records):
            if record is not None:
                snapshots.append(_switch_record(record, i, dt, horizon + hold, f"{path}.switch_records[{i}]"))
        final_after = number(info.get("final_valid_after_trigger"), f"{path}.final_valid_after_trigger",
                             0, min(trigger, final_count), integer=True)
        # Both memberships are stored per trial, unlike historical ever-success flags.
        counted = sum(record is not None and
                      number(final.get("stable_hold_s"), f"{path}.final_diagnostics[{i}].stable_hold_s", 0) >= hold
                      for i, (record, final) in enumerate(zip(records, finals)))
        require(final_after == counted, path, "final-after-trigger count contradicts per-trial final holds")
        fractions = array(info.get("clamped_target_fraction_per_joint"), trials, f"{path}.clamped_target_fraction_per_joint")
        for i, fractions_i in enumerate(fractions):
            for j, fraction in enumerate(array(fractions_i, 12, f"{path}.clamp[{i}]")):
                number(fraction, f"{path}.clamp[{i}][{j}]", 0, 1)
        number(info.get("clamp_sample_period_s"), f"{path}.clamp_sample_period_s", dt, dt)
        diagnostics[pose] = {"triggered_trials": trigger, "untriggered_trials": untriggered,
                             "final_valid_after_trigger": final_after, "switch_snapshots": snapshots}
    require(len(trial_counts) == 1, "results", "one evaluator invocation must have one trial count")

    shared = copy.deepcopy(report)
    shared["protocol_version"] = base_protocol
    for key in ("controller_type", "single_policy_acceptance_eligible", "handoff_controller"):
        shared.pop(key)
    shared["success_count_definition"] = definition[len(COUNT_PREFIX):]
    for result in shared["results"].values():
        result.pop("handoff_diagnostic")
    physical = audit_report(shared)
    verified = _verify_checkpoint_files(identities) if verify_checkpoint_files else []
    return {
        "report_internally_valid": True, "notice": NOTICE, "protocol": protocol,
        "single_policy_acceptance_eligible": False,
        "checkpoint_identities": identities, "checkpoint_files_verified": bool(verify_checkpoint_files),
        "verified_checkpoint_files": verified,
        "common_physics_report_checked": True,
        "poses": [{**row, **diagnostics[row["pose"]]} for row in physical["poses"]],
        "limitations": [
            "A switch snapshot cannot prove the prior continuous 0.2-second gate, first eligible switch time, or absence of later physical/history resets.",
            "Historical ever-success trajectories are absent; final holds and switch membership are checked from recorded per-trial fields only.",
            "Final body speeds, missing base contacts and fore/hind foot x are not independently reconstructed.",
            "Action jumps and clamp fractions are checked for recorded consistency, not replayed against actuator history.",
            "Checkpoint file hashing, when requested, verifies bytes only; source, bank, simulator and video evidence are not independently verified.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--verify-checkpoint-files", action="store_true")
    args = parser.parse_args(argv)
    summaries, failed = [], False
    for path in args.reports:
        try:
            summary = audit_handoff_report(load_report(path), verify_checkpoint_files=args.verify_checkpoint_files)
        except (AuditError, OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            summary = {"report_internally_valid": False, "notice": NOTICE, "error": str(error)}
            failed = True
        summaries.append({"report": str(path.resolve()), **summary})
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
