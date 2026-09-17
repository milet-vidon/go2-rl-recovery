"""Read-only comparison of all-trial final stance, not a natural-motion certificate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from compare_handoff_combined import audit_interfaces
from evaluate_natural_handoff_candidate import PROTOCOL, candidate_receipt, build_source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evaluations/20260917-handoff-combined-v2"
BASE_SUMMARY_SHA = "62db006b67ebac5fa20ef977d3f4a3320afcc8b1c9605358b9307d73ccb5e03a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def distribution(values):
    require(values and all(math.isfinite(x) for x in values), "Missing/nonfinite observations")
    ordered = sorted(values)
    pos = .9 * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo),
            "max": max(values), "values_in_record_order": values}


def measures(result, trials):
    require(result["trials"] == 20 and len(result["final_diagnostics"]) == 20, "Need all20 trials")
    out = {}
    for axle, i in (("front", 0), ("hind", 2)):
        out[f"{axle}_lateral_bias_m"] = distribution([
            abs(row["feet_y_b"][i] + row["feet_y_b"][i + 1]) for row in result["final_diagnostics"]])
        out[f"{axle}_width_m"] = distribution([
            row["feet_y_b"][i] - row["feet_y_b"][i + 1] for row in result["final_diagnostics"]])
    out["paired_lateral_bias_m"] = distribution([
        (abs(row["feet_y_b"][0] + row["feet_y_b"][1]) +
         abs(row["feet_y_b"][2] + row["feet_y_b"][3])) / 2 for row in result["final_diagnostics"]])
    # Existing logger contains gate neighborhoods, not full all-trial trajectories.
    # Summarize exactly those stored stand-active samples and disclose the scope.
    rows = [row for trial in trials for row in trial["rows"] if row["stand_active"]]
    out["stored_stand_active_target_delta_abs_max_rad"] = distribution([
        max(abs(x) for x in row["joint_target_delta_rad"]) for row in rows])
    out["stored_stand_active_torque_abs_max_nm"] = distribution([
        row["after"]["applied_torque_joint_abs_max_at_control_boundary_nm"] for row in rows])
    out["stored_sample_scope"] = "Stand-active samples within existing gate neighborhoods, or last1s for startup; not full trajectories/terminal3s"
    return out


def compare(candidate_path):
    summary = BASE / "summary.json"
    require(sha(summary) == BASE_SUMMARY_SHA, "Baseline summary changed")
    candidate = read(candidate_path)
    require(candidate["protocol_version"] == PROTOCOL, "Not a natural candidate")
    require(len(candidate["results"]) == 1, "Exactly one pose")
    pose = next(iter(candidate["results"]))
    require(candidate["startup_experiment"]["mode"] == "supported" and
            candidate["mirror_experiment"]["mode"] == "initial_right", "Require both existing inference factors")
    identity = candidate["candidate_training"]
    verified = candidate_receipt(identity["training_receipt"], identity["checkpoint"])
    require(verified == identity, "Candidate identity differs from actual receipt")
    require(candidate["handoff_controller"]["stand_sha256"] == verified["checkpoint_sha256"], "Wrong stand actor")
    expected_source_sha = hashlib.sha256(build_source(verified["checkpoint_sha256"]).encode()).hexdigest()
    require(sha(candidate_path.parent / "combined_generated.py") == expected_source_sha ==
            candidate["startup_experiment"]["generated_source_sha256"], "Unexpected generated evaluator")
    entries = [r for r in read(summary)["rows"] if r["startup"] == "supported" and
               r["mirror"] == "initial_right" and r["pose"] == pose]
    require(len(entries) == 1, "No unique baseline")
    baseline_path = Path(entries[0]["report"])
    require(sha(baseline_path).lower() == entries[0]["report_sha256"].lower(), "Baseline report changed")
    baseline = read(baseline_path)
    for field in ("mode", "ramp_seconds", "step_dt", "horizon_s", "hold_s", "min_contacts"):
        require(baseline["transition_experiment"][field] == candidate["transition_experiment"][field],
                "Transition protocol changed: " + field)
    for field in ("task", "seed", "criterion", "self_collisions_enabled", "settle_requested_s", "settle_actual_s",
                  "settle_control_steps", "policy_action_mode", "checkpoint_sha256", "start_protocol_id"):
        require(baseline[field] == candidate[field], "Protocol changed: " + field)
    old, new = baseline["results"][pose], candidate["results"][pose]
    for field in ("release_state_before_settling", "policy_start_state", "horizon_s", "stable_hold_s"):
        require(old[field] == new[field], "Starts/protocol changed: " + field)
    if pose != "upright":
        require(baseline["state_bank"] == candidate["state_bank"], "Bank distribution changed")
    evidence, metrics = {}, {}
    for label, report, path in (("baseline", baseline, baseline_path), ("candidate", candidate, candidate_path)):
        trace_path = path.parent / report["transition_experiment"]["transition_trace_file"]
        require(sha(trace_path) == report["transition_experiment"]["transition_trace_sha256"], "Trace changed")
        trace = read(trace_path)
        evidence[label] = audit_interfaces(report, trace, pose)
        metrics[label] = measures(report["results"][pose], trace["poses"][pose])
    ratios = {k: 1 - metrics["candidate"][k]["mean"] / metrics["baseline"][k]["mean"]
              for k in ("front_lateral_bias_m", "hind_lateral_bias_m", "paired_lateral_bias_m")}
    return {"protocol": "natural_stance_all20_development_comparison_v1", "pose": pose,
            "candidate_report": str(candidate_path), "candidate_report_sha256": sha(candidate_path),
            "baseline_report": str(baseline_path), "baseline_report_sha256": sha(baseline_path),
            "interface_audits": evidence, "metrics": metrics, "relative_mean_bias_reductions": ratios,
            "baseline_final_valid_stands": old["final_valid_stands"], "candidate_final_valid_stands": new["final_valid_stands"],
            "ending_holds_retained": old["final_valid_stands"] == new["final_valid_stands"] == 20,
            "paired_final_lateral_bias_target20percent_met": ratios["paired_lateral_bias_m"] >= .2,
            "recovery_time_s": {k: {"baseline": old[k], "candidate": new[k]} for k in ("median_recovery_s", "p90_recovery_s")},
            "promotion_performed": False,
            "limits": "Repeated development states; final lateral geometry only, not full xyz/whole-hold symmetry or animal-like rolling certification. Stored-neighborhood torque is not peak substep torque. No walk/run integration."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.resolve().drive.upper() == "E:", "Keep outputs on E:")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.candidate)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
