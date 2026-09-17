"""Stdlib-only exact joint-controller comparisons and measured-interface audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import audit_handoff_mirror_coordinates as mirror
from compare_handoff_supported_startup import ExactComparison
import evaluate_handoff_combined as entry

sys.path.insert(0, str(entry.ROOT / "src/go2_recovery"))
from supported_startup_math import supported_startup_mask

ROOT = entry.ROOT
BASELINES = {
    "hard-upright": ("20260917-handoff-transition-v1/hard-upright", "df8a86ddddd89833ed764af7b5fe9d36aac28f57e6226a866c8dd212dc05a60b"),
    "hard-side": ("20260917-handoff-transition-v1/hard-side", "b4b070bd314f45b8fcd6bdea3c89bb5c7a6a894cc25cc6fddd6a20c0af9786bc"),
    "hard-upside_down": ("20260917-handoff-transition-v1/hard-upside_down", "0cfe663c826ec23d9e255a8a2fdea3146f37aa8371d35c0bd538be7d6edc1e22"),
    "startup-upright": ("20260917-supported-startup-v1/supported-upright", "3af752d6b28a40bc566df39405bb9978f66fbbcaf4b4bce018323c0950291b29"),
    "startup-side": ("20260917-supported-startup-v1/supported-side", "88f970ce38adb7265cbe5092ccb82103c6067d39333fc4fa13ea7a9bd99f324c"),
    "startup-upside_down": ("20260917-supported-startup-v1/supported-upside_down", "0187f45126747af226b00b95b1215df931a2b85d0e7230188e1025bfa5bc4b2a"),
    "mirror-upright": ("20260917-handoff-mirror-v1/initial_right-upright", "eedfbdec893688914e8c0af8fc677ccaf70789178c704c60fcf7fcfbea66b31f"),
    "mirror-side": ("20260917-handoff-mirror-v1/initial_right-side", "8ba9ab81b8bd0d94cb6a1ab3e1d5b7f1f9c95165de7d9734932f01afb1576924"),
    "mirror-upside_down": ("20260917-handoff-mirror-v1/initial_right-upside_down", "54dfe9670c4684b72e48100224061af48d13d9094c1f866e3f98a82505520ac0"),
}
GROUP_EXCEPTIONS = {
    "transition_experiment": {"evaluator_source_sha256", "transition_trace_file", "transition_trace_sha256", "hard_zero_semantics"},
    "startup_experiment": {"template_sha256", "adapter_sha256", "generated_source_sha256", "generated_source_file",
                           "transition_trace_file", "transition_trace_sha256", "mirror_enabled"},
    "mirror_experiment": {"evaluator_source_sha256", "mirror_trace_file", "mirror_trace_sha256", "startup_selection_changed"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_key(startup, mirrored, pose):
    mirror.require(startup in ("off", "supported") and mirrored in ("off", "initial_right"), "Unknown mode")
    mirror.require(pose in ("upright", "side", "upside_down"), "Unknown pose")
    if startup == "supported" and (mirrored == "off" or pose == "upright"):
        return "startup-" + pose
    return ("mirror-" if mirrored == "initial_right" else "hard-") + pose


def compare_common(baseline, candidate):
    """Only explicitly different labels/artifact identities may be excluded."""
    compare = ExactComparison()
    for key, value in baseline.items():
        if key in {"protocol_version", "controller_type"}:
            continue
        if key in GROUP_EXCEPTIONS:
            value = {k: v for k, v in value.items() if k not in GROUP_EXCEPTIONS[key]}
        if key not in candidate:
            compare.fail(key, "Missing baseline field")
        else:
            compare.compare(value, candidate[key], key)
    return compare


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_first_action(first, rows, start):
    """Bind independent first evidence to the actual start and duplicate row0."""
    mirror.require(first["step"] == 0, "First action is not step0")
    for row in rows:
        if row["step"] == 0:
            mirror.exact_subset(first, row, "first_vs_stored_step0")
            mirror.exact_subset(row, first, "stored_step0_vs_first")
    before = first["before"]
    for key in ("root_quaternion_wxyz", "root_linear_velocity_w_m_s",
                "root_angular_velocity_w_rad_s", "joint_positions_rad", "height_m"):
        mirror.exact_subset(start[key], before[key], "first_start." + key)
    # These are NOT a common field: start uses acos(-gz), transition uses
    # acos(-gz/||g||). Preserve both frozen definitions; compare raw orientation
    # exactly above and compare each derived field to its own historical record.
    mirror.exact_subset(start["base_contact"], before["current_base_contact"], "first_start.base")
    mirror.exact_subset(start["foot_vertical_forces_N"],
                        [force[2] for force in before["current_foot_forces_w_n"]], "first_start.foot_forces")


def audit_interfaces(report, trace, pose):
    result = report["results"][pose]
    n = result["trials"]
    mirror.require(type(n) is int and n == 20, "Need all20 trials")
    predicate, startup_records = supported_startup_mask(result["policy_start_state"])
    selected_startup = [p and report["startup_experiment"]["mode"] == "supported" for p in predicate]
    selections = result["startup_selection"]["selection_records"]
    mirror.require(len(selections) == n and trace["startup_selection_by_pose"][pose] == selections,
                   "Incomplete/inconsistent startup decisions")
    first = trace["first_action_by_pose"][pose]
    trails = trace["poses"][pose]
    mirror.require(len(first) == len(trails) == n, "Missing first actions/neighborhoods")
    defaults = report["handoff_controller"]["joint_limits"]["default_joint_positions_rad"]
    limits = report["handoff_controller"]["joint_limits"]["soft_joint_limits_rad"]
    total_rows = 0
    mirror_count = 0
    for i, (chosen, actual, recorded) in enumerate(zip(selected_startup, startup_records, selections)):
        mirror.require(recorded["trial"] == i and recorded["selected"] is chosen and recorded["valid"] is True
                       and recorded["inputs"] == actual["inputs"] and recorded["conditions"] == actual["conditions"],
                       "Startup routing differs from measured predicate")
        selected_mirror, _ = mirror.audit_selection(result["policy_start_state"][i],
            result["mirror_diagnostic"]["selection_records"][i], report["mirror_experiment"]["mode"], i)
        mirror_count += selected_mirror
        mirror.require(not (chosen and selected_mirror), "Routing masks overlap")
        trial = trails[i]
        mirror.require(trial["trial"] == i and first[i]["trial"] == i and first[i]["step"] == 0, "Trial ordering")
        if chosen:
            mirror.require(not trial["triggered"] and trial["switch_step"] is None and trial["events"] == []
                           and result["handoff_diagnostic"]["switch_records"][i] is None,
                           "Startup fabricated a handoff")
        switch = trial["switch_step"]
        steps = list(range(500, 550)) if switch is None else list(range(max(0, switch - 50), min(550, switch + 51)))
        mirror.require([r["step"] for r in trial["rows"]] == steps, "Stored neighborhood dropped steps")
        audit_first_action(first[i], trial["rows"], result["policy_start_state"][i])
        previous = None
        for row in [first[i], *trial["rows"]]:
            label = f"{pose}.{i}.{row['step']}"
            mirror.audit_coordinate_record(row, selected_mirror, label)
            switched = switch is not None and row["step"] >= switch
            active = chosen or switched
            mirror.require(row["startup_selected"] is chosen and row["stand_active"] is active
                           and row["switched"] is switched and row["just_switched"] is (switch == row["step"]),
                           "Wrong startup/handoff state")
            if chosen:
                mirror.require(row["gate_steps"] == 0 and row["phase"] == "startup_stand", "False startup gate")
            issued = row["stand_actor_raw_action"] if active else row["roll_actor_raw_action"]
            mirror.exact_subset(issued, row["issued_raw_action"], label + ".issued")
            mirror.exact_subset(issued, row["actual_raw_action"], label + ".actual")
            mirror.exact_subset(row["previous_raw_action"], row["actual_previous_raw_action"], label + ".history")
            expected = [min(limits[j][1], max(limits[j][0], mirror.f32(defaults[j] + mirror.f32(.25 * issued[j])))) for j in range(12)]
            mirror.exact_subset(expected, row["actual_executed_joint_target_rad"], label + ".target")
            mirror.exact_subset(expected, row["expected_executed_joint_target_rad"], label + ".expected_target")
            mirror.require(row["issued_action_target_history_assertions_passed"] is True, "Live assertions missing")
            if previous is not None and previous["step"] + 1 == row["step"]:
                mirror.exact_subset(previous["issued_raw_action"], row["previous_raw_action"], label + ".history_continuity")
                mirror.exact_subset(previous["after"], row["before"], label + ".physical_continuity")
            previous = row
            total_rows += 1
    mirror.require(result["startup_selection"]["selected_trials"] == sum(selected_startup)
                   and result["startup_selection"]["selected_genuine_handoffs"] == 0
                   and result["mirror_diagnostic"]["selected_trials"] == mirror_count, "Routing count mismatch")
    return {"startup_selected": sum(selected_startup), "mirror_selected": mirror_count, "stored_rows_and_first_checked": total_rows}


def compare_files(candidate_path):
    report = mirror.read_json(candidate_path)
    mirror.require(report["protocol_version"] == entry.PROTOCOL and report["controller_type"] == entry.CONTROLLER, "Wrong combined protocol")
    mirror.require(len(report["results"]) == 1 and report["seed"] == 20260918, "Need one fixed development pose")
    for flag in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        mirror.require(report[flag] is False, "Diagnostic flags changed")
    pose = next(iter(report["results"]))
    startup = report["startup_experiment"]
    key = baseline_key(startup["mode"], report["mirror_experiment"]["mode"], pose)
    dirname, digest = BASELINES[key]
    baseline_path = ROOT / "evaluations" / dirname / "model_1999.pt_recovery_metrics.json"
    mirror.require(sha(baseline_path) == digest, "Frozen baseline changed")
    baseline = mirror.read_json(baseline_path)
    mirror.require(report["transition_experiment"]["baseline_hard_zero_semantics"] == baseline["transition_experiment"]["hard_zero_semantics"]
                   and report["transition_experiment"]["hard_zero_semantics"] ==
                   "torch.where((startup_selected | switched)[:,None], stand_action, physical_roll_action)",
                   "Changed action equation is not explicitly identified")
    mirror.verify_frozen_source()
    for path, digest in ((Path(entry.__file__), startup["adapter_sha256"]),
                         (candidate_path.parent / "combined_generated.py", startup["generated_source_sha256"])):
        mirror.require(sha(path) == digest, "Combined source identity changed")
    mirror.require(sha(candidate_path.parent / "combined_generated.py") == hashlib.sha256(entry.build_source().encode()).hexdigest(), "Unexpected generated source")
    mirror.require(startup["mirror_enabled"] is (report["mirror_experiment"]["mode"] == "initial_right"), "Incorrect joint mode label")
    mirror.require(report["mirror_experiment"]["startup_selection_changed"] is (startup["mode"] == "supported"),
                   "Incorrect startup-change label")
    for name in ("roll", "stand"):
        controller = report["handoff_controller"]
        mirror.require(sha(Path(controller[f"{name}_checkpoint"])) == controller[f"{name}_sha256"], "Actor changed")
    trace, trace_sha = mirror.load_trace(candidate_path, report)
    mirror.require(trace["schema"] == "handoff_combined_neighborhood_v1", "Wrong combined trace")
    for group, hash_key in (("transition_experiment", "transition_trace_sha256"),
                            ("startup_experiment", "transition_trace_sha256"),
                            ("mirror_experiment", "mirror_trace_sha256"),
                            ("combined_experiment", "trace_sha256")):
        mirror.require(report[group][hash_key] == trace_sha, "Conflicting trace identity")
        expected = {k: v for k, v in report[group].items() if k != hash_key}
        mirror.require(set(expected) == set(trace[group]), "Report/trace metadata keys differ")
        mirror.exact_subset(expected, trace[group], "report_trace." + group)
    old_trace, _ = mirror.load_trace(baseline_path, baseline)
    audit = audit_interfaces(report, trace, pose)
    comparison = compare_common(baseline, report)
    comparison.compare(csv_rows(baseline_path.parent / baseline["trace_csv"]),
                       csv_rows(candidate_path.parent / report["trace_csv"]), "trial0_csv")
    comparison.compare(old_trace["poses"], trace["poses"], "all_stored_neighborhoods")
    if "first_action_by_pose" in old_trace:
        comparison.compare(old_trace["first_action_by_pose"], trace["first_action_by_pose"], "all_first_actions")
    return {"protocol": "combined_exact_development_audit_v1", "exact_equivalence": comparison.mismatch_count == 0,
        "checked_scalars": comparison.checked, "mismatch_count": comparison.mismatch_count, "mismatches": comparison.mismatches,
        "baseline": {"path": str(baseline_path), "sha256": sha(baseline_path)},
        "candidate": {"path": str(candidate_path), "sha256": sha(candidate_path)},
        "trace_sha256": trace_sha, "interface_audit": audit, "baseline_key": key,
        "final_valid_stands": report["results"][pose]["final_valid_stands"],
        "acceptance_eligible": False, "promotion_performed": False,
        "notice": "Repeated development starts; stored neighborhoods are not every-step full trajectories. No integrated locomotion or fresh-generalization acceptance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing comparison evidence")
    try:
        result = compare_files(args.candidate)
    except (OSError, KeyError, ValueError, TypeError, AssertionError) as error:
        result = {"exact_equivalence": False, "error": str(error), "acceptance_eligible": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["exact_equivalence"] else 1)
