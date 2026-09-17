"""Read-only rendered/unrendered BOTH-ON equivalence and interface audit."""
import argparse
import json
from pathlib import Path

import compare_handoff_combined as audit
import pair_handoff_combined_demo as pairing


def verify_csv_rows(baseline_rows, video_rows, pose, start):
    pairing.require(len(baseline_rows) == 550 and len(video_rows) == 551, "Wrong rollout CSV length")
    initial = video_rows[0]
    pairing.require(initial["policy_phase"] == "before policy" and initial["pose"] == pose
                    and initial["trial"] == "0" and float(initial["time_s"]) == 0., "Wrong extra initial video row")
    pairing.strict_equal(float(initial["height"]), start["height_m"], "video_initial.height")
    pairing.strict_equal(float(initial["gravity_error"]), start["gravity_error"], "video_initial.gravity")
    # Frozen evaluator writes the pre-action row only when recording video.
    # Align this documented boundary, then require every actual control row exactly.
    pairing.strict_equal(baseline_rows, video_rows[1:], "all550_control_rows")


def verify(report_path):
    report = audit.mirror.read_json(report_path)
    pose = next(iter(report["results"]))
    pairing.require(pose in pairing.POSES and report["video_view"] in ("front", "oblique"), "Wrong video scenario")
    base_path = audit.ROOT / "evaluations/20260917-handoff-combined-v2" / ("s1m1-" + pose) / "model_1999.pt_recovery_metrics.json"
    summary = audit.mirror.read_json(base_path.parent.parent / "summary.json")
    pairing.require(summary["status"] == "completed_development_diagnostic", "Joint screen incomplete")
    matching = [r for r in summary["rows"] if r["startup"] == "supported" and r["mirror"] == "initial_right" and r["pose"] == pose]
    pairing.require(len(matching) == 1 and audit.sha(base_path) == matching[0]["report_sha256"].lower(), "Unrendered reference changed")
    baseline = audit.mirror.read_json(base_path)
    # Camera label is the ONLY excluded field. Neither report file is modified.
    pairing.strict_equal({k: v for k, v in baseline.items() if k != "video_view"},
                         {k: v for k, v in report.items() if k != "video_view"}, "rendered_unrendered")
    pairing.verify_artifacts(report_path, report, pose)
    trace, trace_sha = audit.mirror.load_trace(report_path, report)
    measured = audit.audit_interfaces(report, trace, pose)
    verify_csv_rows(audit.csv_rows(base_path.parent / baseline["trace_csv"]),
                    audit.csv_rows(report_path.parent / report["trace_csv"]), pose,
                    report["results"][pose]["policy_start_state"][0])
    return {"report": str(report_path), "report_sha256": audit.sha(report_path), "baseline": str(base_path),
            "baseline_sha256": audit.sha(base_path), "pose": pose, "view": report["video_view"],
            "report_except_view_exact": True, "all550_control_csv_rows_exact": True,
            "extra_pre_action_video_row_checked": True, "trace_sha256": trace_sha,
            "interface_audit": measured, "final_valid_stands": report["results"][pose]["final_valid_stands"],
            "acceptance_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([verify(path.resolve()) for path in args.reports], indent=2, allow_nan=False))
