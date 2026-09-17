"""Read-only, standard-library consistency audit; exit 0 is NOT policy acceptance.

Only the explicitly supported four-foot stance_geometry_v1 protocols are parsed.
No model, bank archive, simulator, CSV, or video is loaded. Unreported final body
speeds, base contact and foot x positions cannot be independently revalidated.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys


CRITERION = (
    "gravity error < 0.35, height 0.30-0.55 m, speed < 0.50 m/s, angular speed < 1.00 rad/s, "
    "4 simultaneous foot vertical forces > 5 N, no base contact, feet on correct body sides "
    "(0.06 < signed lateral < 0.30 m), knees on correct sides (>0.04 m), fore/hind feet on "
    "correct ends (>0.08 m), each joint offset <0.65 rad; all continuously held for hold_s"
)
PROTOCOLS = {"stance_geometry_v1", "stance_geometry_v1_state_bank_PD_v1",
             "stance_geometry_v1_presettled_PD_v1"}
NOTICE = "Internal report consistency only; NOT model acceptance, provenance verification, or visual approval."


class AuditError(ValueError):
    pass


def require(condition, path, message):
    if not condition:
        raise AuditError(f"{path}: {message}")


def number(value, path, low=None, high=None, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), path, "expected integer" if integer else "expected number (not bool)")
    require(math.isfinite(value), path, "non-finite number")
    require(low is None or value >= low, path, f"must be >= {low}")
    require(high is None or value <= high, path, f"must be <= {high}")
    return value


def boolean(value, path):
    require(type(value) is bool, path, "expected boolean")
    return value


def array(value, length, path):
    require(type(value) is list and len(value) == length, path, f"expected list of length {length}")
    return value


def vector(value, length, path):
    for i, item in enumerate(array(value, length, path)):
        number(item, f"{path}[{i}]")
    return value


def finite_tree(value, path="report"):
    if type(value) in (float, int):
        require(math.isfinite(value), path, "non-finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            finite_tree(item, f"{path}[{i}]")


def ordered_records(value, trials, path):
    records = array(value, trials, path)
    for i, record in enumerate(records):
        require(type(record) is dict, f"{path}[{i}]", "expected object")
        number(record.get("trial"), f"{path}[{i}].trial", 0, trials - 1, integer=True)
        require(record["trial"] == i, f"{path}[{i}].trial", "trial order must be 0..trials-1")
    return records


def start_state(state, path):
    for key, size in (("projected_gravity_b", 3), ("root_position_w_m", 3),
                      ("root_quaternion_wxyz", 4), ("root_linear_velocity_w_m_s", 3),
                      ("root_angular_velocity_w_rad_s", 3), ("joint_positions_rad", 12)):
        vector(state.get(key), size, f"{path}.{key}")
    for key in ("geometry_ok", "settled", "eligible_settled_fallen_recovery", "contacts_fresh_since_pose_write"):
        boolean(state.get(key), f"{path}.{key}")
    for key in ("gravity_error", "max_joint_speed_rad_s", "quiet_supported_window_s"):
        number(state.get(key), f"{path}.{key}", 0)
    number(state.get("height_m"), f"{path}.height_m")
    number(state.get("tilt_from_upright_deg"), f"{path}.tilt_from_upright_deg", 0, 180)
    fresh = state["contacts_fresh_since_pose_write"]
    if not fresh:
        for key in ("foot_vertical_forces_N", "vertical_foot_contacts", "base_contact", "any_body_contact",
                    "standing_at_policy_start", "fallen_at_policy_start"):
            require(key in state and state[key] is None, f"{path}.{key}", "stale contacts/classification must be null")
        require(not state["settled"] and not state["eligible_settled_fallen_recovery"], path, "unfresh contacts cannot prove settled recovery")
    else:
        for key in ("base_contact", "any_body_contact", "standing_at_policy_start", "fallen_at_policy_start"):
            boolean(state.get(key), f"{path}.{key}")
        forces = vector(state.get("foot_vertical_forces_N"), 4, f"{path}.foot_vertical_forces_N")
        contacts = number(state.get("vertical_foot_contacts"), f"{path}.vertical_foot_contacts", 0, 4, integer=True)
        require(contacts == sum(force > 5 for force in forces), path, "current foot force/count mismatch")
        require(state["settled"] == (state["quiet_supported_window_s"] + 1e-9 >= .25), path, "settled/window mismatch")
        fallen = state["any_body_contact"] and (state["tilt_from_upright_deg"] >= 60 or (state["base_contact"] and state["height_m"] < .26))
        require(state["fallen_at_policy_start"] == fallen, path, "fallen classification contradicts reported contact/tilt/height")
        eligible = fallen and state["settled"] and not state["standing_at_policy_start"]
        require(state["eligible_settled_fallen_recovery"] == eligible, path, "eligibility flags disagree")
        if state["settled"]:
            require(state["any_body_contact"] and math.hypot(*state["root_linear_velocity_w_m_s"]) < .10
                    and math.hypot(*state["root_angular_velocity_w_rad_s"]) < .20
                    and state["max_joint_speed_rad_s"] < .50, path, "settled state contradicts reported current motion/support")
        if state["standing_at_policy_start"]:
            require(contacts == 4 and not state["base_contact"] and state["geometry_ok"]
                    and .30 < state["height_m"] < .55 and state["gravity_error"] < .35
                    and math.hypot(*state["root_linear_velocity_w_m_s"]) < .50
                    and math.hypot(*state["root_angular_velocity_w_rad_s"]) < 1, path, "standing-start flag contradicts reported state")


def final_state(state, horizon, hold, path):
    geometry = boolean(state.get("geometry_ok"), f"{path}.geometry_ok")
    duration = number(state.get("stable_hold_s"), f"{path}.stable_hold_s", 0, horizon + hold + 1e-8)
    contacts = number(state.get("vertical_foot_contacts"), f"{path}.vertical_foot_contacts", 0, 4, integer=True)
    height = number(state.get("height_m"), f"{path}.height_m")
    gravity = number(state.get("gravity_error"), f"{path}.gravity_error", 0)
    offset = number(state.get("max_joint_offset_rad"), f"{path}.max_joint_offset_rad", 0)
    feet = vector(state.get("feet_y_b"), 4, f"{path}.feet_y_b")
    knees = vector(state.get("knees_y_b"), 4, f"{path}.knees_y_b")
    if geometry:
        require(offset < .65 and all(.06 < sign * y < .30 for sign, y in zip((1, -1, 1, -1), feet))
                and all(sign * y > .04 for sign, y in zip((1, -1, 1, -1), knees)), path, "geometry=true contradicts reported joint/lateral geometry")
    if duration > 0:
        require(geometry and contacts == 4 and .30 < height < .55 and gravity < .35,
                path, "positive current stand hold contradicts reported support/geometry/height/gravity")
    # Check additional facts only when actually recorded; their absence is a limitation.
    if "base_contact" in state:
        boolean(state["base_contact"], f"{path}.base_contact")
        require(not (duration > 0 and state["base_contact"]), path, "positive hold with base contact")
    return duration >= hold


def rate(value, numerator, denominator, path):
    if denominator == 0:
        require(value is None, path, "must be null when no eligible trials")
    else:
        number(value, path, 0, 1)
        require(math.isclose(value, numerator / denominator, rel_tol=1e-6, abs_tol=1e-7), path, "rate/count mismatch")


def timing(value, successes, horizon, path):
    if successes == 0:
        require(value is None, path, "must be null with zero successes")
    else:
        number(value, path, 0, horizon + 1e-8)


def bank_selection(result, starts, release, finals, pose, path):
    selection = result.get("state_bank_selection")
    require(type(selection) is dict, path, "missing bank selection")
    n = result["trials"]
    ids = array(selection.get("selected_state_ids"), n, f"{path}.selected_state_ids")
    for value in ids:
        number(value, f"{path}.selected_state_ids", 0, integer=True)
    unique = number(selection.get("selected_unique_state_count"), f"{path}.selected_unique_state_count", 1, n, integer=True)
    available = number(selection.get("available_unique_states"), f"{path}.available_unique_states", 1, integer=True)
    require(unique == len(set(ids)) == min(n, available), path, "unique source count/full pool coverage mismatch")
    require(boolean(selection.get("sampling_with_replacement"), f"{path}.sampling_with_replacement") == (n > available), path, "replacement flag mismatch")
    require(len(set(ids[:min(n, available)])) == min(n, available), path, "duplicate sources before completing nonreplacement prefix")
    actual = array(selection.get("selected_actual_pose_classes"), n, f"{path}.selected_actual_pose_classes")
    requested = array(selection.get("selected_requested_pose_classes"), n, f"{path}.selected_requested_pose_classes")
    labels = ("left", "right") if pose == "side" else ("back",)
    require(all(label in labels for label in actual), path, "selected source class incompatible with requested pose")
    counts = selection.get("actual_pose_class_counts")
    require(type(counts) is dict and set(counts) == set(labels), path, "wrong source class count keys")
    for label in labels:
        number(counts[label], f"{path}.actual_pose_class_counts.{label}", 0, n, integer=True)
        require(counts[label] == actual.count(label), path, "source class count mismatch")
    for records in (starts, release, finals):
        for i, record in enumerate(records):
            number(record.get("bank_state_id"), f"{path}.trial[{i}].bank_state_id", 0, integer=True)
        require([record["bank_state_id"] for record in records] == ids, path, "bank state IDs/order mismatch")
    for records in (starts, release):
        for i, record in enumerate(records):
            require(record.get("bank_saved_pose_class") == actual[i] and record.get("bank_requested_pose_class") == requested[i], path, "saved/requested provenance mismatch")
            gravity = record["projected_gravity_b"]
            axis = max(range(3), key=lambda j: abs(gravity[j]))
            current = ("left" if gravity[1] > 0 else "right") if axis == 1 else ("back" if axis == 2 and gravity[2] > 0 else "other")
            require(record.get("actual_pose_class") == current, path, "actual class contradicts recorded gravity")
    eligible_unique = len({record["bank_state_id"] for record in starts if record["eligible_settled_fallen_recovery"]})
    number(result.get("settled_fallen_unique_state_count"), f"{path}.settled_fallen_unique_state_count", 0, unique, integer=True)
    require(result["settled_fallen_unique_state_count"] == eligible_unique, path, "eligible unique source count mismatch")
    return ids, unique


def audit_report(report):
    """Return a summary or raise AuditError; a valid all-failure report is accepted."""
    require(type(report) is dict, "report", "expected object")
    finite_tree(report)
    protocol = report.get("protocol_version")
    require(protocol in PROTOCOLS, "protocol_version", "unsupported protocol; do not silently audit legacy/diagnostic criteria")
    require(report.get("criterion") == CRITERION, "criterion", "unsupported or changed four-foot criterion")
    for key in ("acceptance_eligible", "self_collisions_enabled"):
        boolean(report.get(key), key)
    require(report.get("policy_action_mode") == "deterministic_mean" and report["acceptance_eligible"], "policy_action_mode", "this CLI supports deterministic reports only")
    require(report.get("video_view") in (None, "front", "side", "oblique"), "video_view", "invalid camera label")
    require(isinstance(report.get("checkpoint_sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", report["checkpoint_sha256"]), "checkpoint_sha256", "expected SHA-256 string (not verified against file)")
    number(report.get("seed"), "seed", integer=True)
    bank_mode = protocol.endswith("_state_bank_PD_v1")
    results = report.get("results")
    require(type(results) is dict and results, "results", "expected nonempty pose mapping")
    allowed = {"side", "upside_down"} if bank_mode else {"upright", "side", "fore_aft", "upside_down", "random"}
    require(set(results) <= allowed, "results", "unsupported pose")
    bank = report.get("state_bank")
    if bank_mode:
        require(type(bank) is dict and bank.get("schema_version") == "nominal_pd_fallen_v1", "state_bank", "unsupported bank schema")
        require(bank.get("split") in ("train", "heldout"), "state_bank.split", "invalid split")
        require(boolean(bank.get("diagnostic_train_split_only"), "state_bank.diagnostic_train_split_only") == (bank["split"] != "heldout"), "state_bank", "split diagnostic flag mismatch")
        require(report.get("start_protocol_id") == f"state_bank_{bank['split']}_nominal_pose_PD", "start_protocol_id", "bank split protocol mismatch")
        require(report.get("angle_deg") is None, "angle_deg", "bank starts do not use controlled-drop angles")
        require(report.get("settle_controller") == "direct nominal-position PD, not policy zero action; not zero torque", "settle_controller", "unsupported bank handover")
        number(report.get("settle_requested_s"), "settle_requested_s", 1)
        number(report.get("settle_actual_s"), "settle_actual_s", 1)
        number(report.get("settle_control_steps"), "settle_control_steps", 1, integer=True)
    elif protocol == "stance_geometry_v1_presettled_PD_v1":
        require(bank is None, "state_bank", "pre-settled controlled drops must not declare a state bank")
        require(report.get("start_protocol_id") == "pre_settled_nominal_pose_PD",
                "start_protocol_id", "expected explicit pre-settled nominal-PD protocol")
        number(report.get("angle_deg"), "angle_deg", 0)
        require(report.get("settle_controller") == "direct nominal-position PD, not policy zero action; not zero torque",
                "settle_controller", "unsupported pre-settling controller")
        require(report.get("settle_execution") == (
            "physics/action substeps only; termination checked each control step; auto-reset "
            "prohibited; no learned policy, reward computation, curriculum or interval events"),
            "settle_execution", "unsupported pre-settling execution or automatic reset")
        requested = number(report.get("settle_requested_s"), "settle_requested_s", 0)
        actual = number(report.get("settle_actual_s"), "settle_actual_s", 0)
        steps = number(report.get("settle_control_steps"), "settle_control_steps", 1, integer=True)
        require(requested > 0 and actual > 0, "settle_requested_s/settle_actual_s",
                "pre-settled protocol requires positive preparation duration")
        # Match the evaluator's ceiling rule at the observed 50-Hz control rate.
        sample_period = .02
        action = report.get("action_representation")
        if action is not None:
            require(type(action) is dict, "action_representation", "expected object")
            if "sample_period_s" in action:
                supplied_period = number(action["sample_period_s"], "action_representation.sample_period_s")
                require(supplied_period == sample_period, "action_representation.sample_period_s",
                        "pre-settled protocol requires the observed 0.02-s control period")
        require(steps == math.ceil(requested / sample_period), "settle_control_steps",
                "must equal ceil(settle_requested_s / 0.02)")
        require(math.isclose(actual, steps * sample_period, rel_tol=1e-9, abs_tol=1e-12),
                "settle_actual_s", "must equal settle_control_steps * 0.02")
    else:
        require(bank is None and report.get("start_protocol_id") == "controlled_drop", "start_protocol_id", "expected controlled drop without bank")
        number(report.get("angle_deg"), "angle_deg", 0)
        for key in ("settle_requested_s", "settle_actual_s", "settle_control_steps"):
            number(report.get(key), key, 0, 0, integer=key.endswith("steps"))
    rows, all_ids = [], []
    for pose, result in results.items():
        path = f"results.{pose}"
        require(type(result) is dict, path, "expected object")
        n = number(result.get("trials"), f"{path}.trials", 1, integer=True)
        hold = number(result.get("stable_hold_s"), f"{path}.stable_hold_s", .000001)
        horizon = number(result.get("horizon_s"), f"{path}.horizon_s", .000001)
        keys = ("successes", "legacy_contact_height_successes", "final_valid_stands", "final_geometry_passes", "standing_starts_not_fallen_recovery", "settled_fallen_trials", "settled_fallen_recovery_successes", "settled_fallen_final_valid_stands")
        for key in keys:
            number(result.get(key), f"{path}.{key}", 0, n, integer=True)
        starts = ordered_records(result.get("policy_start_state"), n, f"{path}.policy_start_state")
        release = ordered_records(result.get("release_state_before_settling"), n, f"{path}.release_state_before_settling")
        finals = ordered_records(result.get("final_diagnostics"), n, f"{path}.final_diagnostics")
        for collection, records in (("policy_start_state", starts), ("release_state_before_settling", release)):
            for i, state in enumerate(records):
                start_state(state, f"{path}.{collection}[{i}]")
        valid = [final_state(state, horizon, hold, f"{path}.final_diagnostics[{i}]") for i, state in enumerate(finals)]
        eligible = [state["eligible_settled_fallen_recovery"] for state in starts]
        require(result["final_valid_stands"] == sum(valid), path, "final_valid_stands/hold mismatch")
        require(result["final_geometry_passes"] == sum(state["geometry_ok"] for state in finals), path, "final geometry count mismatch")
        require(result["final_valid_stands"] <= result["successes"] <= result["legacy_contact_height_successes"], path, "final/ever/legacy success subset mismatch")
        require(result["final_valid_stands"] <= result["final_geometry_passes"], path, "final valid stands exceed geometry passes")
        require(result["settled_fallen_trials"] == sum(eligible), path, "eligible trial count mismatch")
        require(result["standing_starts_not_fallen_recovery"] == sum(state["standing_at_policy_start"] is True for state in starts), path, "standing-start count mismatch")
        efinal = sum(v and e for v, e in zip(valid, eligible))
        esuccess = result["settled_fallen_recovery_successes"]
        require(result["settled_fallen_final_valid_stands"] == efinal, path, "eligible final valid count mismatch")
        require(efinal <= esuccess <= min(sum(eligible), result["successes"]), path, "eligible success subset mismatch")
        require(result["successes"] - esuccess <= n - sum(eligible), path, "too many successes assigned to ineligible starts")
        rate(result.get("success_rate"), result["successes"], n, f"{path}.success_rate")
        rate(result.get("settled_fallen_recovery_rate"), esuccess, sum(eligible), f"{path}.settled_fallen_recovery_rate")
        for key in ("median_recovery_s", "p90_recovery_s"):
            timing(result.get(key), result["successes"], horizon, f"{path}.{key}")
        unique = None
        if bank_mode:
            ids, unique = bank_selection(result, starts, release, finals, pose, path)
            all_ids.extend(ids)
            timing(result.get("settled_fallen_median_recovery_s"), esuccess, horizon, f"{path}.settled_fallen_median_recovery_s")
        rows.append({"pose": pose, "trials": n, "ever_valid_hold": result["successes"], "final_valid_hold": sum(valid), "final_geometry": result["final_geometry_passes"], "eligible_settled_fallen": sum(eligible), "eligible_ever_valid_hold": esuccess, "eligible_final_valid_hold": efinal, "unique_source_states": unique})
    if bank_mode:
        require(bank.get("selected_state_ids") == all_ids, "state_bank.selected_state_ids", "global/per-pose source order mismatch")
        for state_id in bank["selected_state_ids"]:
            number(state_id, "state_bank.selected_state_ids", 0, integer=True)
        number(bank.get("selected_unique_state_count"), "state_bank.selected_unique_state_count", 1, integer=True)
        require(bank["selected_unique_state_count"] == len(set(all_ids)), "state_bank", "global unique source count mismatch")
        number(bank.get("available_split_states"), "state_bank.available_split_states", len(set(all_ids)), integer=True)
    return {"report_internally_valid": True, "notice": NOTICE, "protocol": protocol, "checkpoint_sha256": report["checkpoint_sha256"], "bank_split": bank["split"] if bank_mode else None, "poses": rows,
            "limitations": ["Historical per-trial success flags are absent; ever-success subset counts are checked, not recomputed from trajectories.", "Final body speeds and fore/hind foot x are not revalidated. Final base contact is checked only if supplied; absent facts are not inferred.", "No source/model/bank hashes, videos, or simulator physics are independently verified."]}


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "JSON", f"duplicate key {key!r}")
        obj[key] = value
    return obj


def load_report(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args(argv)
    summaries, failed = [], False
    for path in args.reports:
        try:
            summary = audit_report(load_report(path))
        except (AuditError, OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            summary = {"report_internally_valid": False, "notice": NOTICE, "error": str(error)}
            failed = True
        summaries.append({"report": str(path.resolve()), **summary})
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
