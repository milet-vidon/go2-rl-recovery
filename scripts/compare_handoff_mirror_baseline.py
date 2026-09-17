"""Stdlib-only exact mirror-OFF versus frozen hard-transition report comparison.

Never imports any simulator/torch entry. Mirror ON is deliberately rejected:
the comparison is an interface-equivalence gate, not capability acceptance.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import compare_handoff_transition_baseline as exact


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "handoff_mirror_experimental_v1"
CONTROLLER = "experimental_roll_only_initial_side_mirror_hard_handoff"
ROLL_SHA = "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c"
STAND_SHA = "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"
TRANSITION_SHA = "c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a"
TRANSITION_MATH_SHA = "b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e"
GATE_SHA = "df8a45a5e64ddc05a3062b73f769134fd3612aa4149f88eae51a67252699369b"
SELECTION_RULE = ("eligible_settled_fallen_at_real_policy_start AND normalized_projected_gravity_body_y < "
                  "-cos(30deg); mode off selects none; no requested pose/ID/outcome input")
SOURCE_PINS = {
    ROOT / "scripts/evaluate_go2_recovery.py": exact.ORIGINAL_SHA256,
    ROOT / "scripts/evaluate_handoff_transition.py": TRANSITION_SHA,
    ROOT / "src/go2_recovery/handoff_transition_math.py": TRANSITION_MATH_SHA,
    ROOT / "src/go2_recovery/recovery_handoff_math.py": GATE_SHA,
}
ARTIFACT_EXCEPTIONS = {"evaluator_source_sha256", "transition_trace_file", "transition_trace_sha256"}
NOTICE = ("Exact report equivalence for mirror off only; NOT model/standing/recovery acceptance, "
          "training-data approval, or proof of independent simulator execution. Synthetic cloned "
          "positives test the comparator only. Mirror initial_right is diagnostic, never auto-accepted.")


def _sha(value, label):
    exact.require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), f"{label}: expected lowercase SHA256")


def _require_contract(report, label):
    exact.require(report.get("checkpoint_sha256") == ROLL_SHA, f"{label}: wrong frozen roll checkpoint")
    controller = report.get("handoff_controller", {})
    exact.require(controller.get("roll_sha256") == ROLL_SHA and controller.get("stand_sha256") == STAND_SHA,
                  f"{label}: wrong frozen dual checkpoint identities")
    for key in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        exact.require(report.get(key) is False, f"{label}.{key}: must be explicit false")
    exp = report.get("transition_experiment")
    exact.require(type(exp) is dict, f"{label}: missing transition contract")
    for key, value in {"mode": "hard", "ramp_seconds": 0.0, "ramp_steps": 0,
                       "step_dt": .02, "horizon_s": 8.0, "hold_s": 3.0, "min_contacts": 4,
                       "policy_control_steps": 550, "policy_duration_including_hold_s": 11.0,
                       "baseline_evaluator_sha256": exact.ORIGINAL_SHA256,
                       "math_source_sha256": TRANSITION_MATH_SHA,
                       "unchanged_one_way_gate": True, "state_or_history_reset_at_switch": False,
                       "hidden_pd_at_switch": False, "retry_enabled": False}.items():
        exact.require(type(exp.get(key)) is type(value) and exp[key] == value, f"{label}.transition_experiment.{key}: changed")
    return exp


def compare_reports(baseline, candidate):
    exact.finite_json(baseline)
    exact.finite_json(candidate)
    exact.require(type(baseline) is dict and type(candidate) is dict, "Both reports must be objects")
    old_exp = _require_contract(baseline, "baseline")
    new_exp = _require_contract(candidate, "candidate")
    exact.require(baseline.get("protocol_version") == exact.PROTOCOL and
                  baseline.get("controller_type") == exact.EXPERIMENT_CONTROLLER,
                  "Baseline must be the completed hard-transition protocol")
    exact.require(old_exp.get("evaluator_source_sha256") == TRANSITION_SHA, "Wrong hard evaluator provenance")
    exact.require(candidate.get("protocol_version") == PROTOCOL and candidate.get("controller_type") == CONTROLLER,
                  "Wrong mirror experiment schema/controller")
    exact.require(candidate.get("baseline_transition_protocol_version") == exact.PROTOCOL,
                  "Missing original hard-transition protocol identity")
    mirror = candidate.get("mirror_experiment")
    exact.require(type(mirror) is dict and mirror.get("mode") == "off", "Only mirror OFF may pass hard equivalence")
    contract = {
        "selection_rule": SELECTION_RULE, "normalized_gravity_y_threshold": -math.cos(math.radians(30)),
        "fixed_mask_per_episode": True, "roll_only": True, "standing_uses_real_observation": True,
        "gate_uses_real_state": True, "physics_or_history_mutated": False, "ramp_enabled": False,
        "retry_enabled": False, "startup_selection_changed": False, "step_dt": .02,
        "horizon_s": 8.0, "hold_s": 3.0, "min_contacts": 4, "policy_control_steps": 550,
        "baseline_transition_evaluator_sha256": TRANSITION_SHA,
        "baseline_transition_math_sha256": TRANSITION_MATH_SHA,
    }
    for key, value in contract.items():
        exact.require(type(mirror.get(key)) is type(value) and mirror[key] == value, f"mirror_experiment.{key}: changed")
    for key in ("evaluator_source_sha256", "math_source_sha256", "mirror_trace_sha256"):
        _sha(mirror.get(key), f"mirror_experiment.{key}")
    exact.require(mirror["evaluator_source_sha256"] == new_exp.get("evaluator_source_sha256") and
                  mirror["mirror_trace_sha256"] == new_exp.get("transition_trace_sha256"),
                  "Mirror/transition provenance contradicts itself")
    expected_trace = Path(candidate["checkpoint"]).name + "_handoff_mirror_trace.json"
    exact.require(mirror.get("mirror_trace_file") == expected_trace and
                  new_exp.get("transition_trace_file") == expected_trace, "Mirror trace name/path changed")
    for pose, result in candidate.get("results", {}).items():
        md = result.get("mirror_diagnostic")
        exact.require(type(md) is dict and type(md.get("selected_trials")) is int and md["selected_trials"] == 0 and
                      md.get("total_trials") == result.get("trials") and md.get("all_trials_in_success_denominator") is True,
                      f"{pose}: OFF mask/denominator metadata invalid")
        records = md.get("selection_records")
        exact.require(type(records) is list and len(records) == result["trials"], f"{pose}: missing OFF selection records")
        for i, (selection, start) in enumerate(zip(records, result["policy_start_state"])):
            exact.require(selection.get("trial") == i and selection.get("selected") is False and
                          selection.get("mask_latched_for_episode") is True, f"{pose}/{i}: OFF selected/mask mismatch")
            exact.require(selection.get("real_policy_start_projected_gravity_b") == start["projected_gravity_b"] and
                          selection.get("eligible_settled_fallen_at_policy_start") is start["eligible_settled_fallen_recovery"],
                          f"{pose}/{i}: selection did not describe actual policy start")
        for record in result["handoff_diagnostic"]["switch_records"]:
            if record is None:
                continue
            exact.require(record.get("roll_mirror_selected") is False, "OFF switch was mirrored")
            for key, old_key in (("real_policy_observation", "policy_observation"),
                                 ("roll_policy_input_observation", "policy_observation"),
                                 ("roll_actor_output_model_raw_action", "roll_actor_raw_action"),
                                 ("roll_actor_output_physical_raw_action", "roll_actor_raw_action")):
                exact.require(record.get(key) == record[old_key], f"OFF {key} differs from native data")

    # Reuse the frozen stdlib comparator on COPIES, never alter either report.
    # All original result fields/common metadata remain present; only explicit
    # schema labels and the three necessarily different artifact identities vary.
    old, new = copy.deepcopy(baseline), copy.deepcopy(candidate)
    old["protocol_version"] = old["baseline_metric_protocol_version"]
    old["controller_type"] = exact.BASELINE_CONTROLLER
    new["protocol_version"] = exact.PROTOCOL
    new["controller_type"] = exact.EXPERIMENT_CONTROLLER
    for obj in (old, new):
        for key in ARTIFACT_EXCEPTIONS:
            obj["transition_experiment"].pop(key, None)
    output = exact.compare_reports(old, new)
    output.update(schema_version="handoff_mirror_baseline_comparison_v1",
                  exact_mirror_off_equivalence=output["exact_baseline_equivalence"],
                  baseline_protocol_version=exact.PROTOCOL, candidate_protocol_version=PROTOCOL,
                  allowed_top_level_relabels=["protocol_version", "controller_type"],
                  transition_artifact_identity_exceptions=sorted(ARTIFACT_EXCEPTIONS),
                  source_files_verified=False, notice=NOTICE)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    destination = None
    try:
        if args.output is not None:
            destination = exact.output_path(args.output)
        for path, digest in SOURCE_PINS.items():
            exact.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Pinned source changed: {path}")
        baseline, old_sha = exact.load_report(args.baseline)
        candidate, new_sha = exact.load_report(args.candidate)
        output = compare_reports(baseline, candidate)
        mirror = candidate["mirror_experiment"]
        for path, digest in ((ROOT / "scripts/evaluate_handoff_mirror.py", mirror["evaluator_source_sha256"]),
                             (ROOT / "src/go2_recovery/roll_mirror_math.py", mirror["math_source_sha256"]),
                             (args.candidate.parent / mirror["mirror_trace_file"], mirror["mirror_trace_sha256"])):
            exact.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Candidate source/trace changed: {path}")
        output.update(source_files_verified=True,
                      baseline={"path": str(args.baseline.resolve()), "sha256": old_sha},
                      candidate={"path": str(args.candidate.resolve()), "sha256": new_sha})
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        output = {"schema_version": "handoff_mirror_baseline_comparison_v1",
                  "exact_mirror_off_equivalence": False, "exact_baseline_equivalence": False,
                  "error": str(error), "acceptance_eligible": False, "training_collection_eligible": False,
                  "promotion_performed": False, "notice": NOTICE}
    if destination is not None:
        try:
            exact.write_output(destination, output)
        except (OSError, ValueError) as error:
            output.update(exact_mirror_off_equivalence=False, exact_baseline_equivalence=False, output_error=str(error))
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if output["exact_mirror_off_equivalence"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
