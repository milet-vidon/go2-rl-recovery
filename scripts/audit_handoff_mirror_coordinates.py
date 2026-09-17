"""Read-only stdlib audit of real/virtual mirror coordinates; never acceptance.

CLI: --report NEW [--baseline OFF_OR_HARD]. JSON to stdout only; no file writes.
Partial-selected cohort aggregate changes are explicitly classified, not silently
compared as if selected trajectories were required to remain identical.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
ENTRY_SHA = "cf2fbbdceae0e460727a969bf080e1e230ed716f00f914f47381a167df50ba95"
MATH_SHA = "9e3edf7cd402c745242fa41e8dcf3d4ccb3818cb474344cff52c104c00426d91"
BASELINE_SHA = "c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a"
P = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
SIGNS = (-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)
THRESHOLD = -math.cos(math.radians(30))
NORMALIZED_Y_TOLERANCE = 2e-7
AGGREGATES = {"successes", "legacy_contact_height_successes", "final_valid_stands", "final_geometry_passes",
              "settled_fallen_recovery_successes", "settled_fallen_final_valid_stands", "settled_fallen_recovery_rate",
              "success_rate", "median_recovery_s", "p90_recovery_s", "settled_fallen_median_recovery_s"}
UNCHANGED_RESULTS = {"trials", "release_state_before_settling", "policy_start_state", "standing_starts_not_fallen_recovery",
                     "settled_fallen_trials", "stable_hold_s", "horizon_s", "state_bank_selection", "settled_fallen_unique_state_count"}
HANDOFF_AGGREGATES = {"triggered_trials", "untriggered_trials", "final_valid_after_trigger"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def finite(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            finite(child, f"{path}[{i}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite {path}")


def exact_subset(old, new, path="$", counter=None):
    """Every baseline dictionary field, every list element/type/value: zero tolerance."""
    require(type(old) is type(new), f"Type mismatch {path}")
    if isinstance(old, dict):
        for key, value in old.items():
            require(key in new, f"Missing common field {path}.{key}")
            exact_subset(value, new[key], f"{path}.{key}", counter)
    elif isinstance(old, list):
        require(len(old) == len(new), f"Length mismatch {path}")
        for i, (a, b) in enumerate(zip(old, new)):
            exact_subset(a, b, f"{path}[{i}]", counter)
    else:
        require(old == new, f"Value mismatch {path}: {old!r} != {new!r}")
        if counter is not None:
            counter[0] += 1


def vector(value, size, label):
    require(isinstance(value, list) and len(value) == size, f"Expected {size}-vector {label}")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value), f"Invalid numeric vector {label}")
    return value


def joint_reflection(value):
    vector(value, 12, "joint reflection")
    return [SIGNS[i] * value[P[i]] for i in range(12)]


def observation_reflection(value):
    vector(value, 48, "observation reflection")
    signs = (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1)
    return ([s * x for s, x in zip(signs, value[:12])]
            + joint_reflection(value[12:24]) + joint_reflection(value[24:36]) + joint_reflection(value[36:48]))


def duplicate_reject(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key {key}")
        result[key] = value
    return result


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=duplicate_reject)
    finite(value)
    return value


def load_trace(report_path, report):
    mirror = report.get("mirror_experiment")
    meta = mirror if mirror is not None else report["transition_experiment"]
    key = "mirror_trace" if mirror is not None else "transition_trace"
    name = meta[key + "_file"]
    require(isinstance(name, str) and Path(name).name == name and ":" not in name and "\\" not in name,
            "Trace must be a local sibling basename")
    path = report_path.parent / name
    require(path.resolve().parent == report_path.parent.resolve(), "Trace escapes report directory")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == meta[key + "_sha256"], "Trace SHA mismatch")
    return read_json(path), digest


def verify_frozen_source():
    entry = ROOT / "scripts" / "evaluate_handoff_mirror.py"
    helper = ROOT / "src" / "go2_recovery" / "roll_mirror_math.py"
    require(hashlib.sha256(entry.read_bytes()).hexdigest() == ENTRY_SHA, "Frozen mirror entry changed")
    require(hashlib.sha256(helper.read_bytes()).hexdigest() == MATH_SHA, "Frozen mirror math changed")
    calls = [node for node in ast.walk(ast.parse(entry.read_text(encoding="utf-8")))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "stand_policy"]
    require(len(calls) == 1 and len(calls[0].args) == 1 and isinstance(calls[0].args[0], ast.Name)
            and calls[0].args[0].id == "obs", "Standing actor no longer consumes original obs")


def audit_selection(start, selection, mode, trial):
    require(start["trial"] == selection["trial"] == trial, "Selection/start trial mismatch")
    gravity = vector(start["projected_gravity_b"], 3, "actual policy-start gravity")
    exact_subset(gravity, selection["real_policy_start_projected_gravity_b"], "selection.gravity")
    require(start["contacts_fresh_since_pose_write"] is True, "Policy-start contacts not fresh")
    require(all(type(start[key]) is bool for key in ("settled", "any_body_contact", "base_contact", "fallen_at_policy_start",
                                                     "eligible_settled_fallen_recovery")), "Non-boolean actual-start flags")
    settled = start["quiet_supported_window_s"] + 1e-9 >= .25
    require(start["settled"] is settled, "Recorded settled flag mismatch")
    fallen = bool(start["any_body_contact"] and (start["tilt_from_upright_deg"] >= 60
                  or start["base_contact"] and start["height_m"] < .26))
    require(start["fallen_at_policy_start"] is fallen, "Recorded fallen flag mismatch")
    require(type(start["standing_at_policy_start"]) is bool, "Missing measured standing-start flag")
    eligible = fallen and settled and not start["standing_at_policy_start"]
    require(start["eligible_settled_fallen_recovery"] is eligible
            and selection["eligible_settled_fallen_at_policy_start"] is eligible, "Eligibility mismatch")
    norm = math.sqrt(sum(x*x for x in gravity))
    require(norm > 0, "Zero actual start gravity")
    recomputed = gravity[1] / norm
    logged = selection["normalized_policy_start_gravity_y"]
    require(type(logged) is float and abs(recomputed - logged) <= NORMALIZED_Y_TOLERANCE,
            "Normalized start gravity exceeds float32 audit tolerance")
    require(selection["strict_gravity_y_threshold"] == THRESHOLD, "Changed side threshold")
    # Runtime comparison casts the scalar threshold to float32. Near-boundary
    # records are not certified using tolerant sign inference.
    require(abs(recomputed - THRESHOLD) > NORMALIZED_Y_TOLERANCE, "Ambiguous float32 selection boundary")
    selected = mode == "initial_right" and eligible and logged < f32(THRESHOLD)
    require(selection["selected"] is selected and selection["mask_latched_for_episode"] is True,
            "Fixed mirror selection inconsistent with actual start")
    return selected, abs(recomputed - logged)


def audit_coordinate_record(record, selected, label):
    require(record["roll_mirror_selected"] is selected, f"Changed fixed mask {label}")
    real = vector(record["real_policy_observation"], 48, label + ".real48")
    exact_subset(real, record["policy_observation"], label + ".legacy48")
    virtual = observation_reflection(real) if selected else real
    exact_subset(virtual, record["roll_policy_input_observation"], label + ".virtual48")
    model = vector(record["roll_actor_output_model_raw_action"], 12, label + ".model12")
    physical = joint_reflection(model) if selected else model
    exact_subset(physical, record["roll_actor_output_physical_raw_action"], label + ".physical12")
    exact_subset(physical, record["roll_actor_raw_action"], label + ".legacy_roll12")
    vector(record["stand_actor_raw_action"], 12, label + ".stand12")
    exact_subset(real[36:], record["previous_raw_action"], label + ".real_history")
    vector(record["previous_previous_raw_action"], 12, label + ".history2")
    exact_subset(real[9:12], record["velocity_command_b"], label + ".real_command")


def audit_report(report, trace):
    finite(report); finite(trace)
    require(report["protocol_version"] == "handoff_mirror_experimental_v1", "Not mirror protocol")
    require(trace["schema"] == "handoff_mirror_neighborhood_v1", "Not mirror trace")
    require(report["controller_type"] == "experimental_roll_only_initial_side_mirror_hard_handoff", "Mirror controller identity")
    require(all(report[key] is False for key in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible")),
            "Mirror report wrongly eligible for acceptance/collection")
    require(trace["acceptance_eligible"] is False and trace["training_collection_eligible"] is False, "Trace acceptance flag")
    meta = report["mirror_experiment"]
    require(meta["mode"] in ("off", "initial_right"), "Unknown mirror mode")
    require(meta["evaluator_source_sha256"] == ENTRY_SHA and meta["math_source_sha256"] == MATH_SHA, "Mirror source identity")
    require(meta["baseline_transition_evaluator_sha256"] == BASELINE_SHA, "Baseline source identity")
    for key in ("fixed_mask_per_episode", "roll_only", "standing_uses_real_observation", "gate_uses_real_state",
                "real_policy48_asserted_against_runtime_every_step"):
        require(meta[key] is True, f"Missing guard {key}")
    for key in ("physics_or_history_mutated", "ramp_enabled", "retry_enabled", "startup_selection_changed", "physical_asset_symmetry_proven"):
        require(meta[key] is False, f"Changed factor {key}")
    for key, expected in (("step_dt", .02), ("horizon_s", 8.0), ("hold_s", 3.0), ("min_contacts", 4), ("policy_control_steps", 550)):
        require(meta[key] == expected, f"Changed budget {key}")
    require(report["transition_experiment"]["mode"] == "hard" and report["transition_experiment"]["ramp_seconds"] == 0,
            "Mirror introduced non-hard transition")
    for key, value in trace["mirror_experiment"].items():
        exact_subset(value, meta[key], "trace.mirror_experiment." + key)
    exact_subset(report["joint_names"], trace["joint_names"], "trace.joint_names")
    require(set(report["results"]) == set(trace["poses"]), "Trace pose denominator mismatch")
    default = report["handoff_controller"]["joint_limits"]["default_joint_positions_rad"]
    limits = report["handoff_controller"]["joint_limits"]["soft_joint_limits_rad"]
    native_names = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
    exact_subset(native_names, report["joint_names"], "native joint order")
    exact_subset(default, joint_reflection(default), "nominal mirror commutation")
    for name in ("soft_joint_limits_rad", "hard_joint_limits_rad"):
        bounds = report["handoff_controller"]["joint_limits"][name]
        require(len(bounds) == 12, "Malformed joint limits")
        for i, partner in enumerate(P):
            vector(bounds[i], 2, name)
            mirrored = sorted(SIGNS[i]*x for x in bounds[partner])
            exact_subset(bounds[i], mirrored, name + ".mirror")
    summary = {}; masks = {}; normalized_error = 0.0; row_count = switch_count = 0
    for pose, result in report["results"].items():
        n = result["trials"]
        require(type(n) is int and n in (1, 2, 20), "Invalid complete denominator")
        trails = trace["poses"][pose]; starts = result["policy_start_state"]
        selections = result["mirror_diagnostic"]["selection_records"]
        switches = result["handoff_diagnostic"]["switch_records"]
        require(all(len(x) == n for x in (trails, starts, selections, switches, result["final_diagnostics"])), "Missing per-trial rows")
        chosen = []; triggered = 0
        for i, (trial, start, selection, switch) in enumerate(zip(trails, starts, selections, switches)):
            require(trial["trial"] == i, "Missing/reordered trace trial")
            exact_subset(selection, trial["mirror_selection"], "trace.selection")
            selected, error = audit_selection(start, selection, meta["mode"], i)
            chosen.append(selected); normalized_error = max(normalized_error, error)
            rows = trial["rows"]; require(bool(rows), "Empty trial trace")
            step = trial["switch_step"]
            if trial["triggered"]:
                require(type(step) is int and 0 <= step < 550 and switch is not None, "Missing first switch")
                require(trial["untriggered_tail_only"] is False, "Triggered trace mislabeled as tail-only")
                expected_steps = list(range(max(0, step-50), min(550, step+51)))
                events = trial["events"]
                require(len(events) == 1 and events[0]["from"] == "roll" and events[0]["to"] == "stand"
                        and events[0]["step"] == step and events[0]["alpha"] == 1, "Invalid hard transition event")
                triggered += 1; switch_count += 1
                audit_coordinate_record(switch, selected, f"{pose}.{i}.switch")
            else:
                require(step is None and switch is None and not trial["events"] and trial["untriggered_tail_only"] is True,
                        "Untriggered trial was silently dropped/changed")
                expected_steps = list(range(500, 550))
            require([x["step"] for x in rows] == expected_steps, "Missing/duplicated neighborhood control steps")
            previous = None; edges = []
            for row in rows:
                label = f"{pose}.{i}.step{row['step']}"; row_count += 1
                require(row["trial"] == i and row["policy_time_s"] == row["step"]*.02, "Row trial/time mismatch")
                audit_coordinate_record(row, selected, label)
                switched = step is not None and row["step"] >= step
                require(row["switched"] is switched and row["phase"] == ("stand" if switched else "roll")
                        and row["alpha"] == (1 if switched else 0), "Non-hard phase/alpha")
                edge = step is not None and row["step"] == step
                require(row["just_switched"] is edge, "Incorrect gate edge")
                if edge: edges.append(row)
                issued = row["stand_actor_raw_action"] if switched else row["roll_actor_output_physical_raw_action"]
                exact_subset(issued, row["issued_raw_action"], label + ".issued")
                exact_subset(issued, row["actual_raw_action"], label + ".actual_raw")
                exact_subset(row["previous_raw_action"], row["actual_previous_raw_action"], label + ".advanced_history")
                expected = [min(limits[j][1], max(limits[j][0], f32(default[j] + f32(.25*issued[j])))) for j in range(12)]
                exact_subset(expected, row["expected_executed_joint_target_rad"], label + ".float32_clamp")
                exact_subset(expected, row["actual_executed_joint_target_rad"], label + ".actual_target")
                delta = [f32(expected[j] - row["previous_executed_joint_target_rad"][j]) for j in range(12)]
                exact_subset(delta, row["joint_target_delta_rad"], label + ".delta")
                require(row["issued_action_target_history_assertions_passed"] is True, "Missing per-step live assertion")
                if previous is not None:
                    exact_subset(previous["issued_raw_action"], row["previous_raw_action"], label + ".history1_continuity")
                    exact_subset(previous["previous_raw_action"], row["previous_previous_raw_action"], label + ".history2_continuity")
                    exact_subset(previous["actual_executed_joint_target_rad"], row["previous_executed_joint_target_rad"], label + ".target_continuity")
                    exact_subset(previous["after"], row["before"], label + ".physical_continuity")
                previous = row
            if switch is not None:
                require(len(edges) == 1, "Switch not covered exactly once")
                edge = edges[0]
                for key in ("policy_time_s", "policy_observation", "previous_raw_action", "previous_previous_raw_action",
                            "velocity_command_b", "previous_executed_joint_target_rad", "stand_actor_raw_action", "roll_actor_raw_action",
                            "real_policy_observation", "roll_policy_input_observation", "roll_actor_output_model_raw_action",
                            "roll_actor_output_physical_raw_action", "roll_mirror_selected"):
                    exact_subset(switch[key], edge[key], "switch-neighborhood." + key)
                for key in ("root_position_local_m", "root_quaternion_wxyz", "root_linear_velocity_w_m_s",
                            "root_angular_velocity_w_rad_s", "joint_positions_rad", "joint_velocities_rad_s"):
                    exact_subset(switch[key], edge["before"][key], "switch-physical." + key)
        require(result["mirror_diagnostic"]["selected_trials"] == sum(chosen)
                and result["mirror_diagnostic"]["total_trials"] == n
                and result["mirror_diagnostic"]["all_trials_in_success_denominator"] is True, "Mirror denominator mismatch")
        handoff = result["handoff_diagnostic"]
        require(handoff["triggered_trials"] == triggered and handoff["untriggered_trials"] == n-triggered
                and handoff["total_success_denominator_includes_untriggered"] is True, "Handoff denominator mismatch")
        final = sum(x["stable_hold_s"] >= 3 for x in result["final_diagnostics"])
        require(result["final_valid_stands"] == final and result["final_geometry_passes"] == sum(x["geometry_ok"] for x in result["final_diagnostics"]),
                "Final report counts inconsistent")
        require(type(result["successes"]) is int and final <= result["successes"] <= n, "Invalid acquisition/final counts")
        masks[pose] = chosen
        summary[pose] = {"trials": n, "selected": sum(chosen), "unselected": n-sum(chosen),
                         "successes": result["successes"], "final_valid_stands": final, "trace_trials": len(trails)}
    return {"poses": summary, "stored_rows_checked": row_count, "switch_records_checked": switch_count,
            "max_normalized_y_float64_vs_logged_float32_error": normalized_error}, masks


def compare_unselected(baseline, candidate, baseline_trace, candidate_trace, masks):
    require(baseline["protocol_version"] in ("handoff_transition_experimental_v1", "handoff_mirror_experimental_v1"), "Unsupported baseline")
    expected_controller = ("experimental_dual_policy_handoff_transition" if baseline["protocol_version"] == "handoff_transition_experimental_v1"
                           else "experimental_roll_only_initial_side_mirror_hard_handoff")
    require(baseline["controller_type"] == expected_controller, "Unrecognized baseline controller label")
    if "mirror_experiment" in baseline:
        require(baseline["mirror_experiment"]["mode"] == "off", "Baseline must be off, never another on run")
        require(baseline["mirror_experiment"]["evaluator_source_sha256"] == ENTRY_SHA
                and baseline["mirror_experiment"]["math_source_sha256"] == MATH_SHA, "Baseline mirror source identity")
    else:
        require(baseline["transition_experiment"]["evaluator_source_sha256"] == BASELINE_SHA, "Baseline hard source identity")
    require(baseline["transition_experiment"]["mode"] == "hard", "Baseline must be hard")
    count = [0]; waived = {}; per_trial = 0
    for key, value in baseline.items():
        require(key in candidate, "Missing baseline field " + key)
        if key in ("protocol_version", "controller_type"):
            continue  # Known independently validated experimental labels only.
        if key == "results":
            continue
        if key in ("transition_experiment", "mirror_experiment"):
            allowed = ({"evaluator_source_sha256", "transition_trace_file", "transition_trace_sha256"}
                       if key == "transition_experiment" else {"mode", "mirror_trace_sha256"})
            for child, item in value.items():
                require(child in candidate[key], f"Missing baseline {key}.{child}")
                if child not in allowed:
                    exact_subset(item, candidate[key][child], f"$.{key}.{child}", count)
        else:
            exact_subset(value, candidate[key], "$." + key, count)
    require(set(baseline["results"]) == set(candidate["results"]), "Different baseline pose set")
    for pose, old in baseline["results"].items():
        new = candidate["results"][pose]; mask = masks[pose]
        selected = any(mask); keep = [i for i, value in enumerate(mask) if not value]
        if not selected:
            exact_subset(old, new, "$.results." + pose, count)
        else:
            waived[pose] = []
            for key, value in old.items():
                require(key in new, f"Missing result common field {pose}.{key}")
                if key in UNCHANGED_RESULTS:
                    exact_subset(value, new[key], f"$.results.{pose}.{key}", count)
                elif key in AGGREGATES:
                    waived[pose].append(key)
                elif key == "final_diagnostics":
                    require(len(value) == len(new[key]) == len(mask), "Final trial denominator changed")
                    for i in keep: exact_subset(value[i], new[key][i], f"{pose}.{key}.{i}", count)
                elif key == "handoff_diagnostic":
                    for child, item in value.items():
                        require(child in new[key], "Missing handoff common field " + child)
                        if child in HANDOFF_AGGREGATES:
                            waived[pose].append(key + "." + child)
                        elif child in ("switch_records", "clamped_target_fraction_per_joint"):
                            require(len(item) == len(new[key][child]) == len(mask), "Handoff trial denominator changed")
                            for i in keep: exact_subset(item[i], new[key][child][i], f"{pose}.{key}.{child}.{i}", count)
                        else:
                            exact_subset(item, new[key][child], f"{pose}.{key}.{child}", count)
                elif key == "mirror_diagnostic":
                    require(set(value) == {"selected_trials", "total_trials", "selection_records", "all_trials_in_success_denominator"},
                            "Unclassified baseline mirror fields")
                    exact_subset(value["total_trials"], new[key]["total_trials"], f"{pose}.mirror.total", count)
                    exact_subset(value["all_trials_in_success_denominator"], new[key]["all_trials_in_success_denominator"], f"{pose}.mirror.denominator", count)
                    for i in keep: exact_subset(value["selection_records"][i], new[key]["selection_records"][i], f"{pose}.mirror.selection.{i}", count)
                    waived[pose].append("mirror_diagnostic.selected_trials")
                else:
                    raise ValueError(f"Unclassified common result field: {pose}.{key}")
        # Starts of selected AND unselected rows must be unchanged.
        for key in ("release_state_before_settling", "policy_start_state"):
            exact_subset(old[key], new[key], f"{pose}.all_start_states.{key}", count)
        old_trials = baseline_trace["poses"][pose]; new_trials = candidate_trace["poses"][pose]
        require(len(old_trials) == len(new_trials) == len(mask), "Baseline trace denominator mismatch")
        for i in keep:
            exact_subset(old_trials[i], new_trials[i], f"{pose}.unselected_trace.{i}", count)
            per_trial += 1
    return {"exact_unselected_common_fields": True, "unselected_trials_compared": per_trial,
            "common_scalar_values_compared": count[0], "selected_cohort_aggregate_fields_not_expected_equal": waived,
            "unknown_common_result_fields_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    output = {"schema": "handoff_mirror_coordinate_audit_v1", "audit_pass": False,
              "acceptance_eligible": False, "training_collection_eligible": False, "promotion_performed": False}
    try:
        for path in (args.report, args.baseline):
            if path is not None:
                require(path.resolve().drive.upper() == "E:" and path.is_file(), "Only existing E-drive reports allowed")
        verify_frozen_source()
        report = read_json(args.report); trace, digest = load_trace(args.report, report)
        details, masks = audit_report(report, trace)
        output.update(details, report=str(args.report.resolve()), trace_sha256=digest,
                      frozen_source_verified=True,
                      normalized_y_audit_tolerance=NORMALIZED_Y_TOLERANCE,
                      standing_input_evidence="Real48 record plus verified frozen stand_policy(obs) call; no separately logged standing-input48 exists.",
                      limitations=["Coordinate/provenance audit is not recovery, visual, hardware or dynamics-symmetry acceptance.",
                                   "Stored trace covers neighborhoods, not all550 steps; every-step live assertions are source/report evidence, not full-trajectory replay.",
                                   "Old reports lack per-trial ever-success/legacy-success timelines; selected-cohort aggregate changes cannot be attributed per trial from a1s neighborhood."])
        if args.baseline:
            baseline = read_json(args.baseline); base_trace, _ = load_trace(args.baseline, baseline)
            output["baseline_comparison"] = compare_unselected(baseline, report, base_trace, trace, masks)
        output["audit_pass"] = True
    except (ValueError, KeyError, TypeError, IndexError, OSError, OverflowError) as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if output["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
