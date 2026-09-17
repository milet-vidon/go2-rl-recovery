"""Pair full mirror diagnostic replays, checking both actors and measured masks."""
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
        if report["protocol_version"] != "handoff_mirror_experimental_v1":
            raise ValueError("Only mirror diagnostic reports are allowed")
        for flag in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
            if report[flag] is not False:
                raise ValueError("Diagnostic is not acceptance or training collection")
        mirror, transition = report["mirror_experiment"], report["transition_experiment"]
        if transition["mode"] != "hard" or transition["ramp_seconds"] != 0:
            raise ValueError("This experiment must not introduce a ramp")
        trace = path.parent / mirror["mirror_trace_file"]
        if trace.parent.resolve() != path.parent.resolve() or sha(trace) != mirror["mirror_trace_sha256"]:
            raise ValueError("Trace path/hash mismatch")
        if transition["transition_trace_sha256"] != mirror["mirror_trace_sha256"]:
            raise ValueError("Conflicting trace identity")
        if sha(Path(report["checkpoint"])) != report["checkpoint_sha256"]:
            raise ValueError("Roll checkpoint changed")
        controller = report["handoff_controller"]
        if sha(Path(controller["stand_checkpoint"])) != controller["stand_sha256"]:
            raise ValueError("Stand checkpoint changed")
    if reports[0]["handoff_controller"] != reports[1]["handoff_controller"]:
        raise ValueError("Standing controller or handoff differs")
    for group, hashkey in (("mirror_experiment", "mirror_trace_sha256"),
                           ("transition_experiment", "transition_trace_sha256")):
        first, second = [dict(r[group]) for r in reports]
        del first[hashkey], second[hashkey]
        if first != second:
            raise ValueError("Mirror/source/transition contracts differ")
    for pose in reports[0]["results"]:
        for key in ("release_state_before_settling", "policy_start_state", "mirror_diagnostic"):
            if reports[0]["results"][pose][key] != reports[1]["results"][pose][key]:
                raise ValueError("Actual starts or fixed mirror masks differ")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_report", type=Path)
    parser.add_argument("oblique_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing paired evidence")
    paths = [args.front_report, args.oblique_report]
    reports = validate(paths)
    pair(*paths, args.output)
    metadata_path = args.output / "paired_views.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(diagnostic_only=True,
        selection_note="Same preregistered right153 development example as the failed hard baseline, not new generalization or an integrated-function demo.",
        handoff_controller=reports[0]["handoff_controller"],
        mirror_experiment=reports[0]["mirror_experiment"],
        transition_experiment=reports[0]["transition_experiment"],
        source_report_sha256=[sha(p) for p in paths], pairing_source_sha256=sha(Path(__file__)))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
