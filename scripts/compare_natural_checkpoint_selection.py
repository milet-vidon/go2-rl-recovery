"""Independent all20 development comparison for retrospective checkpoint selection.

Never re-labels a selection report as the final200 candidate. Validates the full
formal receipt and selected actor, authenticates exact measured starts, and then
reports separate geometry, asymmetry, timing and stored-neighborhood effort.
There is deliberately no overall success/promote flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import compare_natural_handoff_candidate as helpers
import compare_handoff_combined as interfaces
import evaluate_natural_checkpoint_selection as selection


HELPER_SHA = "c3c89878bc1a0c0b0c8fc71cf6df03a1144885252eaca2c7ddd302fed1bc847d"
INTERFACE_SHA = "07f0e3f3c9f2cea5cf56afb4987032328b240309c42336092add6896c2b9c7eb"
PROTOCOL = "natural_stance_checkpoint_selection_all20_comparison_v1"
require = helpers.require
sha = helpers.sha
read = selection.candidate._load


def load_verified_trace(report_path, report):
    filename = report["transition_experiment"]["transition_trace_file"]
    require(Path(filename).name == filename, "Trace must be local to its actual report")
    trace_path = report_path.parent / filename
    trace_sha = sha(trace_path)
    trace = read(trace_path)
    require(trace["schema"] == "handoff_combined_neighborhood_v1", "Wrong all-trial trace schema")
    for group, digest_key in (("transition_experiment", "transition_trace_sha256"),
                               ("startup_experiment", "transition_trace_sha256"),
                               ("mirror_experiment", "mirror_trace_sha256"),
                               ("combined_experiment", "trace_sha256")):
        require(report[group][digest_key] == trace_sha, "Conflicting report trace SHA: " + group)
        expected = {k: v for k, v in report[group].items() if k != digest_key}
        require(expected == trace[group], "Trace/report metadata differs: " + group)
    return trace, {"path": str(trace_path), "sha256": trace_sha}


def final_geometry(result):
    rows = result["final_diagnostics"]
    require(type(result["trials"]) is int and result["trials"] == 20 and len(rows) == 20,
            "Need all20 final records; do not cherry-pick successful trials")
    for i, row in enumerate(rows):
        require(type(row["trial"]) is int and row["trial"] == i, "Final trial order changed")
        require(type(row["geometry_ok"]) is bool, "Invalid final geometry boolean")
        for field in ("feet_y_b", "knees_y_b"):
            require(len(row[field]) == 4 and all(type(v) in (float, int) and math.isfinite(v) for v in row[field]),
                    "Incomplete/nonfinite final lateral geometry")
        require(type(row["vertical_foot_contacts"]) is int and 0 <= row["vertical_foot_contacts"] <= 4,
                "Invalid final current contact count")
    for field in ("successes", "final_valid_stands", "final_geometry_passes"):
        require(type(result[field]) is int and 0 <= result[field] <= 20, "Invalid count: " + field)
    require(sum(r["geometry_ok"] for r in rows) == result["final_geometry_passes"], "Geometry count mismatch")
    ending_hold_trials = [r["trial"] for r in rows if r["stable_hold_s"] >= result["stable_hold_s"]]
    require(len(ending_hold_trials) == result["final_valid_stands"] <= result["successes"], "Ending hold/count mismatch")
    require(result["final_valid_stands"] <= result["final_geometry_passes"], "Ending stands exceed normal geometry")
    return {"trials": 20, "successes_at_any_time": result["successes"],
            "final_valid_stands": result["final_valid_stands"],
            "final_geometry_passes": result["final_geometry_passes"],
            "ending_valid_hold_trials": ending_hold_trials,
            "final_current_four_contact_trials": [r["trial"] for r in rows if r["vertical_foot_contacts"] == 4],
            **{field: helpers.distribution([r[field] for r in rows]) for field in
               ("height_m", "gravity_error", "max_joint_offset_rad", "stable_hold_s")},
            "per_trial": [{key: row[key] for key in ("trial", "geometry_ok", "height_m", "gravity_error",
                         "max_joint_offset_rad", "vertical_foot_contacts", "feet_y_b", "knees_y_b", "stable_hold_s")}
                          for row in rows],
            "scope": "Actual last control boundary and continuous hold counter; no unrecorded speed/contact/xyz reconstruction"}


def compare(candidate_path):
    candidate_path = Path(candidate_path).resolve()
    require(candidate_path.drive.upper() == "E:", "Read the actual E-drive selection report")
    require(sha(helpers.__file__) == HELPER_SHA and sha(interfaces.__file__) == INTERFACE_SHA,
            "Frozen measurement/interface helper changed")
    summary_path = helpers.BASE / "summary.json"
    require(sha(summary_path) == helpers.BASE_SUMMARY_SHA, "Original combined baseline summary changed")
    current = read(candidate_path)
    require(current["protocol_version"] == selection.PROTOCOL and
            current["controller_type"] == "retrospective_natural_stance_checkpoint_in_combined_recovery",
            "Not an independently labelled retrospective-selection report")
    require(current["seed"] == 20260918 and len(current["results"]) == 1, "Exactly one fixed development pose")
    pose = next(iter(current["results"]))
    require(pose in ("upright", "side", "upside_down"), "Unexpected pose")
    for flag in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        require(current[flag] is False, "Diagnostic boundary changed: " + flag)
    require(current["startup_experiment"]["mode"] == "supported" and
            current["mirror_experiment"]["mode"] == "initial_right", "Require unchanged combined factors")
    identity = current["checkpoint_selection"]
    verified = selection.selection_receipt(identity["original_training_receipt"], identity["checkpoint"])
    require(identity == verified, "Selection identity differs from independently verified actual artifacts")
    require(current["candidate_training"] == verified["original_formal_training"],
            "Original formal200/final3746 evidence changed or replaced by selected partial checkpoint")
    controller = current["handoff_controller"]
    require(Path(controller["stand_checkpoint"]).resolve() == Path(verified["checkpoint"]).resolve() and
            controller["stand_sha256"] == verified["checkpoint_sha256"] == sha(controller["stand_checkpoint"]),
            "Actual selected stand actor differs")
    require(controller["roll_sha256"] == selection.candidate.base.ROLL_SHA == sha(controller["roll_checkpoint"]),
            "Frozen roll actor changed")
    generated_sha = hashlib.sha256(selection.build_source(verified["checkpoint_sha256"]).encode()).hexdigest()
    require(sha(candidate_path.parent / "combined_generated.py") == generated_sha ==
            current["startup_experiment"]["generated_source_sha256"] ==
            current["combined_experiment"]["generated_source_sha256"], "Unexpected generated selection evaluator")
    require(current["startup_experiment"]["adapter_sha256"] ==
            current["combined_experiment"]["adapter_sha256"] == sha(selection.__file__), "Selection adapter identity changed")
    entries = [row for row in read(summary_path)["rows"] if row["startup"] == "supported" and
               row["mirror"] == "initial_right" and row["pose"] == pose]
    require(len(entries) == 1, "No unique unchanged combined baseline")
    baseline_path = Path(entries[0]["report"]).resolve()
    require(sha(baseline_path).lower() == entries[0]["report_sha256"].lower(), "Actual baseline report changed")
    baseline = read(baseline_path)
    require(baseline["protocol_version"] == selection.candidate.base.PROTOCOL, "Wrong baseline controller protocol")
    require(baseline["handoff_controller"]["stand_sha256"] == selection.candidate.PARENT_SHA,
            "Baseline must remain original stand3547")
    for field in ("mode", "ramp_seconds", "step_dt", "horizon_s", "hold_s", "min_contacts", "hard_zero_semantics"):
        require(baseline["transition_experiment"][field] == current["transition_experiment"][field],
                "Transition gate/hold semantics changed: " + field)
    for field in ("task", "seed", "criterion", "self_collisions_enabled", "settle_requested_s", "settle_actual_s",
                  "settle_control_steps", "policy_action_mode", "checkpoint_sha256", "start_protocol_id"):
        require(baseline[field] == current[field], "Physical/acceptance protocol changed: " + field)
    old, new = baseline["results"][pose], current["results"][pose]
    for field in ("trials", "release_state_before_settling", "policy_start_state", "horizon_s", "stable_hold_s"):
        require(old[field] == new[field], "Measured starts/denominator/hold changed: " + field)
    if pose != "upright":
        require(baseline["state_bank"] == current["state_bank"] and
                old["state_bank_selection"] == new["state_bank_selection"], "Actual bank states/order changed")
    audits, metrics, geometry, trace_provenance, timing = {}, {}, {}, {}, {}
    for label, report, path in (("baseline", baseline, baseline_path), ("selection", current, candidate_path)):
        trace, provenance = load_verified_trace(path, report)
        audits[label] = helpers.audit_interfaces(report, trace, pose)
        result = report["results"][pose]
        geometry[label] = final_geometry(result)
        metrics[label] = helpers.measures(result, trace["poses"][pose])
        trace_provenance[label] = provenance
        timing[label] = {key: result[key] for key in ("successes", "final_valid_stands", "median_recovery_s", "p90_recovery_s")}
        timing[label]["genuine_gate_switch_times_s_by_trial"] = [
            None if trial["switch_step"] is None else trial["switch_step"] * report["transition_experiment"]["step_dt"]
            for trial in trace["poses"][pose]]
        timing[label]["scope"] = "Reported recovery quantiles are conditional on successful trials; genuine gate times are not recovery completion times"
    bias = {}
    for key in ("front_lateral_bias_m", "hind_lateral_bias_m", "paired_lateral_bias_m"):
        first, second = metrics["baseline"][key], metrics["selection"][key]
        deltas = [b - a for a, b in zip(first["values_in_record_order"], second["values_in_record_order"])]
        bias[key] = {"baseline_mean": first["mean"], "selection_mean": second["mean"],
                     "mean_delta_selection_minus_baseline": second["mean"] - first["mean"],
                     "relative_mean_reduction": None if first["mean"] == 0 else 1 - second["mean"] / first["mean"],
                     "lower_bias_trial_count": sum(value < 0 for value in deltas),
                     "higher_bias_trial_count": sum(value > 0 for value in deltas),
                     "per_trial_deltas": deltas}
    return {"protocol": PROTOCOL, "pose": pose, "selection_kind": "retrospective_development_selection",
            "selected_checkpoint": verified, "candidate_report": str(candidate_path), "candidate_report_sha256": sha(candidate_path),
            "baseline_report": str(baseline_path), "baseline_report_sha256": sha(baseline_path),
            "baseline_summary_sha256": helpers.BASE_SUMMARY_SHA, "trace_provenance": trace_provenance,
            "exact_measured_starts_verified": True, "unchanged_criteria_verified": True, "interface_audits": audits,
            "geometry": geometry, "metrics": metrics, "measured_lateral_bias_changes": bias, "timing": timing,
            "baseline_final_valid_stands": old["final_valid_stands"], "selection_final_valid_stands": new["final_valid_stands"],
            "ending_holds_retained": old["final_valid_stands"] == new["final_valid_stands"] == 20,
            "promotion_performed": False, "additional_training_updates": 0, "independent_generalization_proven": False,
            "limits": "No unified success/promotion score. All20 final samples include failures; lower mean lateral bias cannot compensate for lost holds. This is retrospective selection on repeated development states. Only final lateral geometry, not whole-hold/full-xyz/animal-like rolling; torque/action changes are stored gate-neighborhood summaries, not trajectory or substep peaks. Recovery quantiles have success-conditioned denominators. No walk/run integration."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.resolve().drive.upper() == "E:", "Keep new output on E:")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.candidate)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key not in ("metrics", "geometry")}, indent=2, allow_nan=False))
