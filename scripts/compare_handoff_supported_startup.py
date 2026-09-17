"""Exact, stdlib-only startup control/retention comparison; no simulator imports.

Off and unselected on cases compare all original hard-result fields, applicable
metadata, trial-0 CSV and all stored hard transition neighborhoods. Selected on
upright compares all common outcomes/actual starts with frozen stand-only3547,
plus its trial-0 CSV only. It never invents stand-only all-trial trajectories.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import compare_handoff_transition_baseline as exact
import evaluate_handoff_supported_startup as entry

sys.path.insert(0, str(entry.ROOT / "src/go2_recovery"))
from supported_startup_math import supported_startup_mask


SOURCE_PINS = {
    entry.TEMPLATE: entry.TEMPLATE_SHA, entry.MATH_PATH: entry.MATH_SHA,
    entry.ROOT / "scripts/evaluate_go2_recovery.py": exact.ORIGINAL_SHA256,
    entry.ROOT / "src/go2_recovery/handoff_transition_math.py":
        "b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e",
    entry.ROOT / "src/go2_recovery/recovery_handoff_math.py":
        "df8a45a5e64ddc05a3062b73f769134fd3612aa4149f88eae51a67252699369b",
}
STAND_REFERENCE_SHA = "1c3728c48cb7b2a8ad36ae11f1aae97c03ad53719bca56ddcda372a3c87ad463"
ARTIFACT_KEYS = {"evaluator_source_sha256", "transition_trace_file", "transition_trace_sha256"}
STAND_METADATA_EXCEPTIONS = {
    "checkpoint", "checkpoint_sha256", "protocol_version", "controller_type",
    "acceptance_eligible", "trace_csv", "success_count_definition",
}


class ExactComparison:
    def __init__(self):
        self.checked = 0
        self.mismatch_count = 0
        self.mismatches = []

    def compare(self, old, new, path):
        if type(old) is not type(new):
            self.fail(path, "type mismatch")
        elif type(old) is dict:
            for key, value in old.items():
                if key not in new:
                    self.fail(path + "." + key, "missing old field")
                else:
                    self.compare(value, new[key], path + "." + key)
        elif type(old) is list:
            if len(old) != len(new):
                self.fail(path, "list length mismatch")
            for i, (left, right) in enumerate(zip(old, new)):
                self.compare(left, right, f"{path}[{i}]")
        else:
            self.checked += 1
            if old != new:
                self.fail(path, "exact scalar mismatch; tolerance=0")

    def fail(self, path, reason):
        self.mismatch_count += 1
        if len(self.mismatches) < 100:
            self.mismatches.append({"path": path, "reason": reason})


def _require_candidate(report):
    exact.require(report.get("protocol_version") == entry.PROTOCOL and
                  report.get("controller_type") == entry.CONTROLLER, "Wrong startup protocol")
    exact.require(report.get("checkpoint_sha256") == entry.ROLL_SHA and
                  report.get("handoff_controller", {}).get("stand_sha256") == entry.STAND_SHA,
                  "Wrong frozen two-actor identities")
    for key in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        exact.require(report.get(key) is False, key + " must be false")
    startup = report["startup_experiment"]
    for key, value in {
        "fixed_mask_per_episode": True, "selected_actor_retained": True,
        "selected_counts_as_handoff": False, "selected_counts_as_success": False,
        "unselected_original_hard_gate": True, "mirror_enabled": False,
        "ramp_enabled": False, "retry_enabled": False, "training_performed": False,
        "physical_state_or_history_mutated": False,
        "stand_actor_initialization_rng_isolated": True,
        "all_trial_first_actions_recorded": True,
        "template_sha256": entry.TEMPLATE_SHA, "math_source_sha256": entry.MATH_SHA,
    }.items():
        exact.require(type(startup.get(key)) is type(value) and startup[key] == value,
                      "Startup contract changed: " + key)
    exact.require(startup.get("mode") in ("off", "supported"), "Invalid startup mode")
    transition = report["transition_experiment"]
    for key, value in {"mode": "hard", "ramp_seconds": 0.0, "ramp_steps": 0,
                       "step_dt": .02, "horizon_s": 8.0, "hold_s": 3.0,
                       "min_contacts": 4, "policy_control_steps": 550}.items():
        exact.require(type(transition.get(key)) is type(value) and transition[key] == value,
                      "Changed physical/timing contract: " + key)
    exact.require(len(report["results"]) == 1, "Exactly one cold pose required")
    pose, result = next(iter(report["results"].items()))
    exact.require(result["trials"] == 20, "Finite comparison requires 20 trials")
    mask, records = supported_startup_mask(result["policy_start_state"])
    sd = result["startup_selection"]
    enabled = startup["mode"] == "supported"
    selected = [enabled and value for value in mask]
    exact.require(sd["selected_trials"] == sum(selected) and
                  sd["selected_genuine_handoffs"] == 0 and
                  len(sd["selection_records"]) == 20, "Startup count/denominator contradiction")
    for i, (rec, actual, wanted) in enumerate(zip(sd["selection_records"], records, selected)):
        exact.require(rec["trial"] == i and rec["selected"] is wanted and
                      rec["actual_initial_actor"] == ("stand" if wanted else "roll") and
                      rec["valid"] is True and rec["inputs"] == actual["inputs"] and
                      rec["conditions"] == actual["conditions"], "Selection not actual fixed predicate")
        if wanted:
            exact.require(result["handoff_diagnostic"]["switch_records"][i] is None,
                          "Startup selected trial falsely has a handoff record")
    return pose, result, selected


def compare_reports(baseline, candidate):
    exact.finite_json(baseline)
    exact.finite_json(candidate)
    pose, result, selected = _require_candidate(candidate)
    exact.require(set(baseline["results"]) == {pose}, "Pose mismatch")
    mode = candidate["startup_experiment"]["mode"]
    stand_reference = mode == "supported" and pose == "upright"
    if stand_reference:
        exact.require(all(selected), "Preregistered upright set must select all 20 or stop for investigation")
        exact.require(baseline["checkpoint_sha256"] == entry.STAND_SHA and
                      "handoff_diagnostic" not in baseline["results"][pose], "Need same-state stand-only reference")
        exceptions = STAND_METADATA_EXCEPTIONS
    else:
        exact.require(not any(selected), "Unselected/off comparison cannot contain selected cases")
        exact.require(baseline["protocol_version"] == exact.PROTOCOL and
                      baseline["checkpoint_sha256"] == entry.ROLL_SHA and
                      baseline["transition_experiment"]["evaluator_source_sha256"] == entry.TEMPLATE_SHA,
                      "Need completed frozen hard baseline")
        exceptions = {"protocol_version", "controller_type", "transition_experiment"}
    compare = ExactComparison()
    compare.compare(baseline["results"], candidate["results"], "results")
    for key, value in baseline.items():
        if key in exceptions or key == "results":
            continue
        if key not in candidate:
            compare.fail(key, "baseline metadata missing")
        else:
            compare.compare(value, candidate[key], key)
    if not stand_reference:
        compare.compare({k: v for k, v in baseline["transition_experiment"].items() if k not in ARTIFACT_KEYS},
                        candidate["transition_experiment"], "transition_experiment")
    return compare, pose, stand_reference


def _csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compare_files(baseline_path, candidate_path):
    baseline, old_sha = exact.load_report(baseline_path)
    candidate, new_sha = exact.load_report(candidate_path)
    for path, digest in SOURCE_PINS.items():
        exact.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Pinned source drift: " + str(path))
    compare, pose, stand_reference = compare_reports(baseline, candidate)
    if stand_reference:
        exact.require(old_sha == STAND_REFERENCE_SHA, "Stand reference report SHA changed")
    startup = candidate["startup_experiment"]
    for path, digest in (
        (entry.ROOT / "scripts/evaluate_handoff_supported_startup.py", startup["adapter_sha256"]),
        (candidate_path.parent / startup["generated_source_file"], startup["generated_source_sha256"]),
        (candidate_path.parent / startup["transition_trace_file"], startup["transition_trace_sha256"]),
    ):
        exact.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Candidate source/trace drift")
    exact.require(startup["generated_source_sha256"] == hashlib.sha256(entry.build_source().encode("utf-8")).hexdigest(),
                  "Stored generated source is not the current pinned finite adapter output")
    compare.compare(_csv_rows(baseline_path.parent / baseline["trace_csv"]),
                    _csv_rows(candidate_path.parent / candidate["trace_csv"]), "trial0_csv")
    trace, _ = exact.load_report(candidate_path.parent / startup["transition_trace_file"])
    exact.require(trace["startup_selection_by_pose"][pose] == candidate["results"][pose]["startup_selection"]["selection_records"],
                  "Report/trace startup decisions disagree")
    first = trace["first_action_by_pose"][pose]
    exact.require(len(first) == 20, "Missing all-trial first actions")
    for i, row in enumerate(first):
        selected = candidate["results"][pose]["startup_selection"]["selection_records"][i]["selected"]
        exact.require(row["trial"] == i and row["step"] == 0 and row["just_switched"] is False and
                      row["switched"] is False and row["gate_steps"] == 0 and
                      row["startup_selected"] is selected and row["stand_active"] is selected,
                      "First action falsely classified as a handoff")
        expected = row["stand_actor_raw_action"] if selected else row["roll_actor_raw_action"]
        exact.require(row["issued_raw_action"] == expected == row["actual_raw_action"] and
                      row["previous_raw_action"] == row["actual_previous_raw_action"] and
                      row["issued_action_target_history_assertions_passed"] is True,
                      "First action/real natural history mismatch")
    exact.require(len(trace["poses"][pose]) == 20, "Incomplete all-trial transition neighborhoods")
    for i, trial in enumerate(trace["poses"][pose]):
        exact.require(trial["trial"] == i and trial["rows"], "Missing or reordered trial neighborhood")
        if candidate["results"][pose]["startup_selection"]["selection_records"][i]["selected"]:
            exact.require(trial["triggered"] is False and trial["switch_step"] is None and trial["events"] == [],
                          "Startup-only trial fabricated a transition event")
            for row in trial["rows"]:
                exact.require(row["phase"] == "startup_stand" and row["startup_selected"] is True and
                              row["stand_active"] is True and row["switched"] is False and
                              row["just_switched"] is False and row["gate_steps"] == 0 and
                              row["issued_raw_action"] == row["stand_actor_raw_action"] == row["actual_raw_action"],
                              "Selected actor not retained, or startup falsely relabeled as handoff")
    if not stand_reference:
        old_trace, _ = exact.load_report(baseline_path.parent / baseline["transition_experiment"]["transition_trace_file"])
        compare.compare(old_trace["poses"], trace["poses"], "all_stored_transition_neighborhoods")
    return {
        "schema": "handoff_supported_startup_exact_comparison_v1",
        "exact_equivalence": compare.mismatch_count == 0,
        "mismatch_count": compare.mismatch_count, "mismatches": compare.mismatches,
        "checked_scalars": compare.checked, "pose": pose,
        "comparison": "selected_vs_stand_only" if stand_reference else "unselected_or_off_vs_hard",
        "all_trial_stand_only_trajectory_equivalence_claimed": False,
        "baseline": {"path": str(baseline_path), "sha256": old_sha},
        "candidate": {"path": str(candidate_path), "sha256": new_sha},
        "acceptance_eligible": False, "promotion_performed": False,
        "notice": "Development diagnostic only. No fresh generalization, integrated locomotion or hardware acceptance.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    destination = exact.output_path(args.output)
    try:
        output = compare_files(args.baseline, args.candidate)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        output = {"exact_equivalence": False, "error": str(exc), "acceptance_eligible": False,
                  "promotion_performed": False}
    exact.write_output(destination, output)
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if output["exact_equivalence"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
