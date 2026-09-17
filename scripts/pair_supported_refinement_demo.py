"""Pair actual front/oblique post-recovery refinement replays, never full flow.

Requires an existing current-source ON audit of the same NONVIDEO 20-trial case.
The reference is re-audited, not trusted from its success label. All physical
reports, control CSV and all-trial interfaces must match exactly. Failed outcomes
remain included. No simulator or video dependency is imported at module import.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import audit_supported_refinement as audit
import evaluate_supported_refinement as entry
import pair_handoff_combined_demo as original


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "pair_handoff_combined_demo.py": "f52f2d03ad5fd14b679ed254e7179f4f90100d3fe8be03f989d7c70fdb67a463",
    "audit_supported_refinement.py": "a2d44a311b12ed14bcc7d220ab78389cb67ffab6a413297636994b362a6171b8",
    "evaluate_supported_refinement.py": "bdb03cf09fafa9c0ed5aa14ee8818b4f95b6d145300e33b6e5cac8bd1b0ad0ff",
    "evaluate_supported_refinement.ps1": "53f781916e1921fcc1e1697f8ab5965e8ef5b224594a1d27e86ad4b118d4d97f",
}
PROTOCOL = "supported_refinement_paired_trial0_v1"
POSES = original.POSES
EXPECTED_FRAMES, EXPECTED_FPS = original.EXPECTED_FRAMES, original.EXPECTED_FPS
BANNER_HEIGHT = 96
require, strict_equal, sibling = original.require, original.strict_equal, original.sibling
read_video, validate_video_pair = original.read_video, original.validate_video_pair
read_json, sha = audit.mirror.read_json, audit.sha


def check_source_pins():
    for name, digest in PINS.items():
        require(sha(ROOT / "scripts" / name) == digest, "Reviewed dependency changed: " + name)


def validate_report_pair(front, oblique, reference):
    """No acceptance filtering; only camera view is excluded from full equality."""
    require([front.get("video_view"), oblique.get("video_view"), reference.get("video_view")]
            == ["front", "oblique", None], "Require front/oblique and a NONVIDEO reference")
    physical = lambda report: {k: v for k, v in report.items() if k != "video_view"}
    strict_equal(physical(front), physical(oblique), "front_oblique_all_fields_except_view")
    strict_equal(physical(front), physical(reference), "video_nonvideo_all_fields_except_view")
    require(front["protocol_version"] == entry.PROTOCOL and front["controller_type"] == entry.CONTROLLER,
            "Require the three-actor supported-refinement protocol")
    strict_equal(front["diagnostic_trial"], 0, "video_trial")
    strict_equal(front["seed"], 20260918, "development_seed")
    require(len(front["results"]) == 1, "One pose, never concatenate separate episodes")
    pose = next(iter(front["results"]))
    require(pose in POSES, "Unsupported pose")
    result = front["results"][pose]
    strict_equal(result["trials"], 20, "all20_trial_population")
    for key in ("successes", "final_valid_stands", "final_geometry_passes"):
        require(type(result[key]) is int and 0 <= result[key] <= 20, "Invalid outcome count")
    for key in ("policy_start_state", "release_state_before_settling", "final_diagnostics"):
        require(len(result[key]) == 20, "Missing all-trial evidence")
        strict_equal([row["trial"] for row in result[key]], list(range(20)), "ordered." + key)
    for key in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        require(front[key] is False, "Pairing cannot promote or accept a policy")
    controller, refinement = front["handoff_controller"], front["refinement_experiment"]
    require(controller["roll_sha256"] == front["checkpoint_sha256"] == audit.ROLL_SHA and
            controller["stand_sha256"] == audit.STAND_SHA and
            refinement["refinement_actor_training"]["checkpoint_sha256"] == audit.REFINEMENT_SHA,
            "Require exactly roll1999 / stand3547 / refine3746")
    require(refinement["refinement_actor_training"]["training_receipt_sha256"] == audit.RECEIPT_SHA,
            "Wrong refinement training provenance")
    for key, expected in (("mode", "supported"), ("num_envs", 20), ("qualifying_completed_intervals", 150),
                          ("step_dt", .02), ("full_measured_rows", 11000),
                          ("decision_applies_to_next_action", True), ("training_performed_in_this_run", False),
                          ("switch_time_physical_or_history_reset", False), ("extra_pd_ramp_or_retry", False),
                          ("promotion_performed", False)):
        strict_equal(refinement[key], expected, "refinement." + key)
    require(front["startup_experiment"]["mode"] == "supported" and
            front["mirror_experiment"]["mode"] == "initial_right" and
            front["combined_experiment"]["postrecovery_refinement_mode"] == "supported",
            "Original routing must be supported / initial_right, with refinement ON")
    for key, expected in (("mode", "hard"), ("ramp_seconds", 0.0), ("ramp_steps", 0), ("step_dt", .02),
                          ("horizon_s", 8.0), ("hold_s", 3.0), ("min_contacts", 4), ("policy_control_steps", 550)):
        strict_equal(front["transition_experiment"][key], expected, "transition." + key)
    return pose


def verify_reference(audit_path):
    """Recompute the authoritative audit including its actual OFF control."""
    check_source_pins()
    saved = read_json(audit_path)
    require(saved["protocol"] == audit.PROTOCOL and saved["audit_passed"] is True and
            saved["mode"] == "supported" and saved["auditor_source_sha256"] == PINS["audit_supported_refinement.py"],
            "Require a successful current-source ON audit. Re-audit old-source evidence (including back v2) before pairing; never edit the saved receipt.")
    reference = Path(saved["report"])
    off = Path(saved["off_control"]["report"])
    require(all(path.resolve().drive.upper() == "E:" for path in (reference, off, audit_path)),
            "Keep actual reference evidence on E:")
    report = read_json(reference)
    require(report["video_view"] is None, "Reference must be the separately audited nonvideo experiment")
    recomputed = audit.audit(reference, off)
    strict_equal(saved, recomputed, "reference_audit_recomputed_from_actual_evidence")
    return reference.resolve(), report, recomputed


def csv_rows(path, *, video=False):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Missing/duplicate CSV columns")
        rows = list(reader)
    require(len(rows) == 550 + int(video) and all(None not in row and None not in row.values() for row in rows),
            "Require exactly550 control rows, plus exactlyone pre-action row only for video")
    controls = rows[1:] if video else rows
    for step, row in enumerate(controls):
        require(row["trial"] == "0" and float(row["time_s"]) == (step + 1) * .02,
                "Missing/reordered control CSV interval")
    return rows


def verify_control_csv_rows(nonvideo_rows, video_rows, report, pose):
    """Account explicitly for the frozen recorder's video-only pre-action sample."""
    require(len(nonvideo_rows) == 550 and len(video_rows) == 551, "Wrong documented CSV boundary")
    initial = video_rows[0]
    start = report["results"][pose]["policy_start_state"][0]
    require(initial["pose"] == pose and initial["trial"] == "0" and initial["policy_phase"] == "before policy"
            and initial["handoff_active"] == "False" and float(initial["time_s"]) == 0.
            and float(initial["stable_hold_s"]) == 0., "Invalid pre-action identity, gate or history boundary")
    for field, source in (("height", "height_m"), ("gravity_error", "gravity_error")):
        strict_equal(float(initial[field]), start[source], "pre_action." + field)
    for field, source in (("base_contact", "base_contact"), ("geometry_ok", "geometry_ok")):
        require(initial[field] in ("True", "False") and (initial[field] == "True") is start[source],
                "Pre-action measured flag differs: " + field)
    require(initial["feet_contact"] == str(start["vertical_foot_contacts"]), "Pre-action support count differs")
    if "bank_state_id" in start:
        require(initial["bank_state_id"] == str(start["bank_state_id"]), "Pre-action state-bank identity differs")
    for field, source in (("speed", "root_linear_velocity_w_m_s"), ("angular_speed", "root_angular_velocity_w_rad_s")):
        require(abs(float(initial[field]) - audit.norm3(start[source])) <= 2e-7, "Pre-action norm differs: " + field)
    defaults = report["handoff_controller"]["joint_limits"]["default_joint_positions_rad"]
    delta = []
    for name, q, default in zip(report["joint_names"], start["joint_positions_rad"], defaults):
        strict_equal(float(initial[name]), q, "pre_action." + name)
        value = audit.f32(q - default)
        strict_equal(float(initial[name + "_offset"]), value, "pre_action.offset." + name)
        delta.append(value)
    require(len(delta) == 12 and abs(float(initial["joint_rms"]) - math.sqrt(sum(q*q for q in delta) / 12)) <= 2e-7,
            "Pre-action joint RMS differs")
    feet = [[float(initial[f"{leg}_foot_{axis}_b"]) for axis in "xyz"] for leg in ("FL", "FR", "RL", "RR")]
    knees_y = [float(initial[f"{leg}_knee_y_b"]) for leg in ("FL", "FR", "RL", "RR")]
    require(all(math.isfinite(x) for foot in feet for x in foot) and all(math.isfinite(x) for x in knees_y),
            "Nonfinite pre-action geometry")
    geometry = (all(feet[i][1] * sign > audit.f32(.06) and abs(feet[i][1]) < audit.f32(.30)
                    and knees_y[i] * sign > audit.f32(.04) for i, sign in enumerate((1, -1, 1, -1)))
                and all(feet[i][0] * sign > audit.f32(.08) for i, sign in enumerate((1, 1, -1, -1)))
                and max(abs(x) for x in delta) < audit.f32(.65))
    require(geometry is start["geometry_ok"], "Pre-action foot/knee geometry disagrees with measured start")
    gravity = start["projected_gravity_b"]
    for field, numerator in (("roll_deg", gravity[1]), ("pitch_deg", gravity[0])):
        expected = math.degrees(math.atan2(numerator, -gravity[2]))
        require(abs(float(initial[field]) - expected) <= 1e-4, "Pre-action displayed orientation differs")
    # Only this explicit initial boundary is aligned out. Every actual control
    # sample remains present and exact; no filtering by success or time range.
    strict_equal(nonvideo_rows, video_rows[1:], "all550_actual_control_csv_rows")


def verify_artifacts(path, report, pose, video=False):
    verified_report, verified_pose, _, full_path, provenance = audit.load_verified(path)
    strict_equal(report, verified_report, "report_did_not_change")
    require(verified_pose == pose, "Wrong reference pose")
    csv_path = sibling(path.parent, report["trace_csv"])
    csv_rows(csv_path, video=video)
    neighborhood = sibling(path.parent, report["startup_experiment"]["transition_trace_file"])
    generated = sibling(path.parent, report["startup_experiment"]["generated_source_file"])
    files = [path, csv_path, full_path, neighborhood, generated, path.parent / "invocation.json"]
    invocation = read_json(path.parent / "invocation.json")
    files.extend(Path(item["path"]) for item in invocation["source_and_inputs"])
    result = {"report": str(path), "report_sha256": sha(path), "trace_csv": str(csv_path),
              "trace_csv_sha256": sha(csv_path), "full_trace": str(full_path), "full_trace_sha256": sha(full_path),
              "neighborhood": str(neighborhood), "neighborhood_sha256": sha(neighborhood),
              "generated_source_sha256": sha(generated), "verified_provenance": provenance}
    if video:
        video_path = sibling(path.parent, f"model_1999.pt_{pose}.mp4")
        require(video_path.stat().st_size > 0, "Empty video")
        files.append(video_path)
        result.update(video=str(video_path), video_sha256=sha(video_path))
    return result, files


def compare_alltrial_interfaces(source, reference):
    """All11000 physical records/neighborhoods must remain byte-exact."""
    for key in ("full_trace", "neighborhood"):
        left, right = Path(source[key]), Path(reference[key])
        require(source[key + "_sha256"] == reference[key + "_sha256"], "Different all-trial evidence: " + key)
        require(left.read_bytes() == right.read_bytes(), "Different complete control evidence: " + key)


def compare_control_evidence(source, reference, report, pose):
    compare_alltrial_interfaces(source, reference)
    for evidence in (source, reference):
        require(sha(evidence["trace_csv"]) == evidence["trace_csv_sha256"], "CSV changed during verification")
    verify_control_csv_rows(csv_rows(Path(reference["trace_csv"])),
                            csv_rows(Path(source["trace_csv"]), video=True), report, pose)


def compare_video_csvs(front, oblique):
    """Both video CSVs include the initial boundary: all551 rows are exact."""
    left, right = Path(front["trace_csv"]), Path(oblique["trace_csv"])
    require(front["trace_csv_sha256"] == oblique["trace_csv_sha256"] and left.read_bytes() == right.read_bytes(),
            "Front/oblique CSV including pre-action row differs")
    strict_equal(csv_rows(left, video=True), csv_rows(right, video=True), "front_oblique_all551_rows")


def snapshot(paths):
    unique = sorted({Path(path).resolve() for path in paths})
    return [{"path": str(path), "sha256": sha(path)} for path in unique]


def check_snapshot(records):
    for record in records:
        require(sha(record["path"]) == record["sha256"], "Evidence/source changed during pairing: " + record["path"])
    check_source_pins()


def encode_pair(sources, properties, target, cv2, ffmpeg):
    """One decoded source frame -> one encoded pair; no time filtering at all."""
    width, height = properties["width"] * 2, properties["height"] + BANNER_HEIGHT
    caps = [cv2.VideoCapture(source["video"]) for source in sources]
    require(all(cap.isOpened() for cap in caps), "Could not reopen both actual videos")
    command = [ffmpeg, "-nostdin", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s:v", f"{width}x{height}", "-r", str(EXPECTED_FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    with (target.parent / "encoder.log").open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for _ in range(EXPECTED_FRAMES):
                decoded = [cap.read() for cap in caps]
                require(all(ok for ok, _ in decoded), "Input ended early; no partial montage accepted")
                frames = [frame for _, frame in decoded]
                require(all(frame.shape == (properties["height"], properties["width"], 3) for frame in frames),
                        "Input shape changed while pairing")
                paired = cv2.copyMakeBorder(cv2.hconcat(frames), BANNER_HEIGHT, 0, 0, 0,
                                           cv2.BORDER_CONSTANT, value=(12, 12, 12))
                cv2.putText(paired, "RECOVERY + POSTURE ONLY | THREE ACTORS | NOT A WALK / TROT / RUN FLOW",
                            (18, 33), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.putText(paired, "FRONT (left) | OBLIQUE (right) | Matched-condition replays, NOT simultaneous cameras",
                            (18, 67), cv2.FONT_HERSHEY_SIMPLEX, .60, (180, 225, 255), 1, cv2.LINE_AA)
                process.stdin.write(paired.tobytes())
            require(all(not cap.read()[0] for cap in caps), "Extra source frames; no silent temporal cuts")
            process.stdin.close()
            require(process.wait() == 0, "Encoder failed; inspect encoder.log")
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        finally:
            for cap in caps:
                cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_report", type=Path)
    parser.add_argument("oblique_report", type=Path)
    parser.add_argument("reference_audit", type=Path, help="Existing authoritative ON audit JSON; reference must be nonvideo")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = [args.front_report.resolve(), args.oblique_report.resolve()]
    audit_path, output = args.reference_audit.resolve(), args.output_dir.resolve()
    require(all(path.drive.upper() == "E:" for path in [*paths, audit_path, output]), "Keep all new and source artifacts on E:")
    require(len(set(paths)) == 2 and paths[0].parent != paths[1].parent, "Require two actual separate replay recordings")
    require(output != Path("E:/") and not output.exists(), "Use a fresh E-drive output directory")
    initial_sources = snapshot([Path(__file__), ROOT / "scripts/test_pair_supported_refinement_demo.py",
                                audit_path, *paths, *(ROOT / "scripts" / name for name in PINS)])
    reference_path, reference_report, reference_audit = verify_reference(audit_path)
    require(reference_path not in paths, "Reference cannot be either video replay")
    reports = [read_json(path) for path in paths]
    pose = validate_report_pair(*reports, reference_report)
    reference, files = verify_artifacts(reference_path, reference_report, pose)
    sources = []
    for path, report in zip(paths, reports):
        source, evidence_files = verify_artifacts(path, report, pose, video=True)
        compare_control_evidence(source, reference, report, pose)
        sources.append(source)
        files.extend(evidence_files)
    compare_video_csvs(*sources)
    off_path = Path(reference_audit["off_control"]["report"])
    off_report = read_json(off_path)
    _, off_files = verify_artifacts(off_path, off_report, pose)
    files.extend(off_files)
    evidence_snapshot = snapshot(files)
    import cv2
    import imageio_ffmpeg
    properties = [read_video(Path(source["video"]), cv2) for source in sources]
    validate_video_pair(*properties)
    require(properties[0]["width"] >= 800, "Too narrow for the unambiguous on-video disclosure banner")
    check_snapshot(initial_sources)
    check_snapshot(evidence_snapshot)
    output.mkdir(parents=True, exist_ok=False)
    target = output / f"supported_refinement_{pose}_front_oblique.mp4"
    encode_pair(sources, properties[0], target, cv2, imageio_ffmpeg.get_ffmpeg_exe())
    actual = read_video(target, cv2)
    strict_equal(actual, {**properties[0], "width": 2 * properties[0]["width"],
                         "height": properties[0]["height"] + BANNER_HEIGHT}, "complete_encoded_pair")
    check_snapshot(evidence_snapshot)
    check_snapshot(initial_sources)
    result = reference_report["results"][pose]
    metadata = {"protocol": PROTOCOL, "pose": pose, "trials": 20, "video_trial": 0,
                "sources": sources, "source_properties": properties, "output": str(target),
                "output_sha256": sha(target), "output_properties": actual,
                "duration_s": actual["frames"] / actual["fps"], "banner_added_pixels": BANNER_HEIGHT,
                "source_snapshot": initial_sources, "evidence_snapshot": evidence_snapshot,
                "reference_audit": str(audit_path), "reference_audit_sha256": sha(audit_path),
                "nonvideo_reference_audited_again_in_this_process": True, "reference": reference,
                "all_report_fields_except_view_exact": True, "all550_control_csv_rows_vs_nonvideo_exact": True,
                "front_oblique_all551_csv_rows_exact": True, "extra_pre_action_video_row_checked_against_actual_start": True,
                "all11000_control_interfaces_byte_exact": True, "all20_starts_and_results_exact": True,
                "successes": result["successes"], "final_valid_stands": result["final_valid_stands"],
                "refinement_final_valid_holds": reference_report["refinement_experiment"]["poses"][pose]["final_refinement_valid_holds"],
                "acceptance_eligible": False, "single_policy_acceptance_eligible": False,
                "training_collection_eligible": False, "promotion_performed": False,
                "handoff_controller": reference_report["handoff_controller"],
                "refinement_actor_training": reference_report["refinement_experiment"]["refinement_actor_training"],
                "note": "Front left, oblique right: independent matched-condition replays, NOT simultaneous cameras. Three actors: roll1999, stand3547, refine3746. Full276 source frames at25fps; no temporal cuts, speed change, spatial crop or input resize. Only trial0 of each20-environment batch is filmed after1s nominal-PD preparation. This single-pose zero-command posture stage is NOT a continuous recovery/walk/trot/run workflow or naturalness certificate. Failure outcomes are never excluded."}
    with (output / "paired_metadata.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"video": str(target), "verified": actual, "sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
