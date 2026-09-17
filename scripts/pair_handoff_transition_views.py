"""Pair frozen hard/ramp diagnostics only after checking BOTH actors and traces."""
import argparse
import hashlib
import json
from pathlib import Path

from pair_recovery_views import pair, validate_reports


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(paths):
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    validate_reports(*reports)
    for report, path in zip(reports, paths):
        if report["protocol_version"] != "handoff_transition_experimental_v1":
            raise ValueError("Only transition diagnostics belong in this entry")
        for flag in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
            if report[flag] is not False:
                raise ValueError("Diagnostic must not claim acceptance or training collection")
        meta = report["transition_experiment"]
        trace = path.parent / meta["transition_trace_file"]
        if trace.parent.resolve() != path.parent.resolve() or sha(trace) != meta["transition_trace_sha256"]:
            raise ValueError("Trace path/hash mismatch")
        if sha(Path(report["checkpoint"])) != report["checkpoint_sha256"]:
            raise ValueError("Roll checkpoint changed")
        if not report["handoff_controller"].get("stand_sha256"):
            raise ValueError("Missing secondary actor identity")
        if sha(Path(report["handoff_controller"]["stand_checkpoint"])) != report["handoff_controller"]["stand_sha256"]:
            raise ValueError("Stand checkpoint changed")
    if reports[0]["handoff_controller"] != reports[1]["handoff_controller"]:
        raise ValueError("Secondary controller differs between views")
    first, second = [dict(r["transition_experiment"]) for r in reports]
    # Independent cameras need not reproduce floating trajectories bit-for-bit.
    for meta in (first, second):
        del meta["transition_trace_sha256"]
    if first != second:
        raise ValueError("Transition mode, equations, timings or source hashes differ")
    for pose in reports[0]["results"]:
        for key in ("release_state_before_settling", "policy_start_state"):
            if reports[0]["results"][pose][key] != reports[1]["results"][pose][key]:
                raise ValueError("Measured starting states differ between views")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_report", type=Path)
    parser.add_argument("oblique_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve old paired evidence; choose a new directory")
    paths = [args.front_report, args.oblique_report]
    reports = validate(paths)
    pair(*paths, args.output)
    metadata_path = args.output / "paired_views.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        diagnostic_only=True,
        selection_note="Preselected left/right development cases; NOT a success-rate or integrated-function demo.",
        handoff_controller=reports[0]["handoff_controller"],
        transition_experiment=reports[0]["transition_experiment"],
        source_report_sha256=[sha(p) for p in paths],
        pairing_source_sha256=sha(Path(__file__)),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
