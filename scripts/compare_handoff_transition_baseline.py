"""CPU/stdlib exact report comparison; never imports a simulator entry point.

Additional dictionary metadata is allowed, but every baseline result field,
list length/order, scalar type and numeric value must remain exactly equal.
This checks reports, not policy quality or independent simulation provenance.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


ORIGINAL_SHA256 = "4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f"
ORIGINAL_ENTRY = Path(__file__).with_name("evaluate_go2_recovery.py")
PROTOCOL = "handoff_transition_experimental_v1"
BASELINE_PROTOCOLS = {
    "stance_geometry_v1_state_bank_PD_v1_dual_policy_diagnostic_v1",
    "stance_geometry_v1_presettled_PD_v1_dual_policy_diagnostic_v1",
}
BASELINE_CONTROLLER = "two_policy_fixed_one_way_handoff_diagnostic"
EXPERIMENT_CONTROLLER = "experimental_dual_policy_handoff_transition"
NOTICE = ("Report equivalence only; NOT model acceptance, training-data approval, "
          "hardware/visual approval, or proof that independent simulator runs occurred. "
          "Synthetic cloned reports can test this comparator but cannot prove simulator equivalence.")
METADATA_EXCEPTIONS = {"protocol_version", "controller_type", "acceptance_eligible"}
REQUIRED_METADATA = {
    "angle_deg", "checkpoint", "checkpoint_sha256", "task", "seed",
    "start_protocol", "start_protocol_id", "settle_requested_s", "settle_actual_s",
    "settle_control_steps", "settle_controller", "settle_execution",
    "settled_definition", "fallen_start_definition", "success_count_definition",
    "time_definition", "protocol_version", "joint_names", "video_view",
    "self_collisions_enabled", "criterion", "diagnostic_trial", "policy_action_mode",
    "acceptance_eligible", "controller_type", "single_policy_acceptance_eligible",
    "handoff_controller", "action_representation", "trace_csv", "results",
}


class ComparisonError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def finite_json(value, path="$", depth=0):
    require(depth < 100, f"{path}: excessive JSON depth")
    if type(value) is dict:
        require(all(type(key) is str for key in value), f"{path}: non-string object key")
        for key, item in value.items():
            finite_json(item, f"{path}.{key}", depth + 1)
    elif type(value) is list:
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]", depth + 1)
    elif type(value) is float:
        require(math.isfinite(value), f"{path}: nonfinite number")
    else:
        require(value is None or type(value) in (str, int, bool), f"{path}: unsupported JSON type")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_report(path):
    raw = Path(path).read_bytes()
    report = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    finite_json(report)
    require(type(report) is dict, "Report must be a JSON object")
    return report, hashlib.sha256(raw).hexdigest()


def _validate_identity(report, label):
    for key in ("checkpoint_sha256",):
        require(type(report.get(key)) is str and re.fullmatch(r"[0-9a-fA-F]{64}", report[key]),
                f"{label}.{key}: expected SHA256")
    controller = report.get("handoff_controller")
    require(type(controller) is dict, f"{label}.handoff_controller: missing object")
    for key in ("roll_sha256", "stand_sha256"):
        require(type(controller.get(key)) is str and re.fullmatch(r"[0-9a-fA-F]{64}", controller[key]),
                f"{label}.handoff_controller.{key}: expected SHA256")
    require(type(report.get("checkpoint")) is str and report["checkpoint"], f"{label}: missing checkpoint")
    require(controller.get("roll_checkpoint") == report["checkpoint"] and
            controller["roll_sha256"] == report["checkpoint_sha256"], f"{label}: roll identity contradiction")
    require(type(controller.get("stand_checkpoint")) is str and controller["stand_checkpoint"],
            f"{label}: missing stand checkpoint")


def compare_reports(baseline, candidate):
    """Return exact-subset differences without modifying either input object."""
    finite_json(baseline)
    finite_json(candidate)
    require(type(baseline) is dict and type(candidate) is dict, "Both reports must be objects")
    missing = REQUIRED_METADATA - baseline.keys()
    require(not missing, f"Baseline missing required metadata: {sorted(missing)}")
    require(baseline["protocol_version"] in BASELINE_PROTOCOLS, "Not a supported original dual-policy baseline")
    require(baseline["controller_type"] == BASELINE_CONTROLLER, "Baseline controller is not original one-way dual")
    require(type(baseline["acceptance_eligible"]) is bool, "Baseline acceptance flag is not boolean")
    require(baseline["single_policy_acceptance_eligible"] is False, "Baseline must not claim single-policy acceptance")
    _validate_identity(baseline, "baseline")
    _validate_identity(candidate, "candidate")
    require(candidate.get("protocol_version") == PROTOCOL, "Candidate is not the experimental protocol")
    require(candidate.get("baseline_metric_protocol_version") == baseline["protocol_version"],
            "Candidate's recorded baseline protocol does not match")
    require(candidate.get("controller_type") == EXPERIMENT_CONTROLLER, "Wrong experimental controller label")
    for key in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        require(candidate.get(key) is False, f"Candidate.{key} must be explicit false")
    experiment = candidate.get("transition_experiment")
    require(type(experiment) is dict, "Missing transition_experiment")
    require(experiment.get("mode") == "hard", "Only hard mode can match the original baseline")
    require(type(experiment.get("ramp_seconds")) in (int, float) and experiment["ramp_seconds"] == 0,
            "Hard comparison requires numeric ramp_seconds=0, not boolean")
    require(experiment.get("baseline_evaluator_sha256") == ORIGINAL_SHA256,
            "Wrong frozen original evaluator SHA256")
    require(("state_bank" in baseline) == ("state_bank" in candidate), "Bank/non-bank start metadata changed")
    require(type(baseline["results"]) is dict and baseline["results"], "Baseline has no result buckets")
    require(type(candidate.get("results")) is dict, "Candidate has no result buckets")
    require(set(baseline["results"]) == set(candidate["results"]), "Pose bucket set changed")
    for pose, result in baseline["results"].items():
        require(type(result) is dict and type(result.get("trials")) is int and result["trials"] > 0,
                f"Baseline.results.{pose}: invalid trial count")
        for key in ("release_state_before_settling", "policy_start_state", "final_diagnostics"):
            require(type(result.get(key)) is list and len(result[key]) == result["trials"],
                    f"Baseline.results.{pose}.{key}: inconsistent denominator")
        require(type(result.get("handoff_diagnostic")) is dict and
                type(result["handoff_diagnostic"].get("switch_records")) is list and
                len(result["handoff_diagnostic"]["switch_records"]) == result["trials"],
                f"Baseline.results.{pose}: inconsistent switch-record denominator")

    mismatches = []
    mismatch_count = 0
    checked = {"result_values": 0, "metadata_values": 0}

    def mismatch(path, reason, old, new):
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < 100:
            def compact(value):
                return {"type": type(value).__name__, "length": len(value)} if type(value) in (list, dict) else value
            mismatches.append({"path": path, "reason": reason, "baseline": compact(old), "candidate": compact(new)})

    def compare(old, new, path, group):
        if type(old) is not type(new):
            mismatch(path, "type differs (no numeric coercion)", old, new)
        elif type(old) is dict:
            for key, item in old.items():
                if key not in new:
                    mismatch(f"{path}.{key}", "baseline key missing", item, None)
                else:
                    compare(item, new[key], f"{path}.{key}", group)
        elif type(old) is list:
            if len(old) != len(new):
                mismatch(path, "list length differs", old, new)
            for index, (left, right) in enumerate(zip(old, new)):
                compare(left, right, f"{path}[{index}]", group)
        else:
            checked[group] += 1
            if old != new:
                mismatch(path, "exact value differs; tolerance=0", old, new)

    compare(baseline["results"], candidate["results"], "$.results", "result_values")
    for key, value in baseline.items():
        if key in METADATA_EXCEPTIONS or key == "results":
            continue
        if key not in candidate:
            mismatch(f"$.{key}", "baseline metadata missing", value, None)
        else:
            compare(value, candidate[key], f"$.{key}", "metadata_values")
    return {
        "schema_version": "handoff_transition_baseline_comparison_v1",
        "exact_baseline_equivalence": mismatch_count == 0,
        "mismatch_count": mismatch_count, "mismatches": mismatches,
        "mismatches_truncated": mismatch_count > len(mismatches), "checked_scalar_values": checked,
        "allowed_top_level_relabels": sorted(METADATA_EXCEPTIONS),
        "baseline_protocol_version": baseline["protocol_version"], "candidate_protocol_version": PROTOCOL,
        "original_evaluator_sha256": ORIGINAL_SHA256, "original_evaluator_file_verified": False,
        "acceptance_eligible": False, "training_collection_eligible": False, "promotion_performed": False,
        "notice": NOTICE,
    }


def output_path(value):
    path = Path(value)
    require(path.is_absolute() and path.drive.upper() == "E:", "Output must be an absolute E-drive JSON path")
    resolved = path.resolve()
    require(resolved.drive.upper() == "E:" and resolved.suffix.lower() == ".json",
            "Output must resolve to an E-drive .json file")
    require(not path.exists() and not path.is_symlink() and not resolved.exists(), "Output exists; refusing overwrite")
    require(resolved.parent.is_dir(), "Output parent must already exist")
    return resolved


def write_output(path, payload):
    # Exclusive create closes the race after the read-only path checks.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New absolute E-drive JSON file; existing parent required")
    args = parser.parse_args(argv)
    destination = None
    try:
        if args.output is not None:
            destination = output_path(args.output)
        actual_original_sha = hashlib.sha256(ORIGINAL_ENTRY.read_bytes()).hexdigest()
        require(actual_original_sha == ORIGINAL_SHA256, "Local frozen original evaluator changed")
        baseline, baseline_sha = load_report(args.baseline)
        candidate, candidate_sha = load_report(args.candidate)
        result = compare_reports(baseline, candidate)
        result.update(original_evaluator_file_verified=True,
                      baseline={"path": str(args.baseline.resolve()), "sha256": baseline_sha},
                      candidate={"path": str(args.candidate.resolve()), "sha256": candidate_sha})
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result = {"schema_version": "handoff_transition_baseline_comparison_v1",
                  "exact_baseline_equivalence": False, "error": str(error), "notice": NOTICE,
                  "acceptance_eligible": False, "training_collection_eligible": False, "promotion_performed": False}
    if destination is not None:
        try:
            write_output(destination, result)
        except (OSError, ValueError) as error:
            result.update(exact_baseline_equivalence=False, output_error=str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["exact_baseline_equivalence"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
