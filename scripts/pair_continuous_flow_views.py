"""Pair complete, identical physical replays; never certify a new physical result.

Only already-passed v3 sensor-identity front/oblique recordings at cmd .5 qualify.
No simulator imports, input writes, frame selection, cropping or acceleration.
Video dependencies are imported only by build(), after the evidence checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "continuous_recovery_flow_diagnostic_v3_sensor_identity"
METRICS_PROTOCOL = "continuous_flow_motion_metrics_v2_quiet_ready"
ENTRY_SHA = "82570a633b29a15b1b2f29e6f6d5437ec51c9846e963621efd4cf3b93548d9dc"
ACTOR_SHA = {"roll": "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c",
             "stand": "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"}
LOCOMOTION_SHA = {"control3947": "3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f",
                  "balanced4246": "882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc"}
FPS, SOURCE_W, SOURCE_H, BAR_H = 50, 960, 540, 144
OUTPUT_W, OUTPUT_H = 2 * SOURCE_W, SOURCE_H + BAR_H
VIDEO = "continuous_flow.mp4"
REPORT = "continuous_flow_report.json"
REQUIRED_ARTIFACTS = {VIDEO, "measured_rows.json", "video_frames.json", "full_trace.jsonl",
                      "continuous_flow.csv", "initial_state.json", "invocation.json",
                      "actual_config.json", "extracted_helpers.py", "physical_configuration_comparison.json"}
CHECKS = set("recorded_no_done_whole_episode recovery_last150_strict recovery_last150_quiet_ready "
             "stop_last150_strict no_base_contact_after_recovery_hold_start supported_height level "
             "walk_tracking straight_drift quiet_stand_stop feet_lift_in_walk limited_slip "
             "four_feet_at_rest normal_stance_geometry_at_rest four_vertical_contacts_at_rest "
             "no_current_base_contact_at_rest geometry_and_support_at_rest".split())
BANNER = (
    "MATCHED REPLAYS / CONTINUOUS RECOVERY-WALK-STOP / CMD 0.5 / NOT TROT OR FAST RUN",
    "LEFT: FRONT | RIGHT: OBLIQUE | Separate cold-start replays, NOT simultaneous cameras",
    "Includes initial 1s nominal-PD preparation | Bank-initialized fall, NOT a natural fall during locomotion",
    "All original frames retained at 50 fps / native 960x540 each | Simulation diagnostic only; no promotion",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sha_ok(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite evidence")
    elif type(value) is dict:
        for item in value.values():
            finite_tree(item)
    elif type(value) is list:
        for item in value:
            finite_tree(item)


def canonical(value):
    finite_tree(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def exact(a, b, message):
    # Unlike Python ==, this distinguishes 1, 1.0 and true and preserves all floats.
    require(canonical(a) == canonical(b), message)


def read_json(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=object_pairs)
    finite_tree(value)
    return value


def validate_rows_schedule(rows):
    """Structural check only; NEVER returns a physical acceptance result."""
    require(type(rows) is list and 900 <= len(rows) <= 1300, "Incomplete bounded episode")
    order, counts = [], {}
    for index, row in enumerate(rows):
        require(type(row) is dict and type(row.get("step")) is int and row["step"] == index, "Missing/reordered row")
        require(type(row.get("time_s")) is float and row["time_s"] == (index + 1) * .02, "Wrong completed time")
        phase = row.get("phase")
        require(phase in ("preparation", "recovery", "move", "stop"), "Unknown phase")
        if not order or order[-1] != phase:
            order.append(phase)
        counts[phase] = counts.get(phase, 0) + 1
        require(type(row.get("phase_interval")) is int and row["phase_interval"] == counts[phase], "Wrong phase index")
        wanted = [.5, 0., 0.] if phase == "move" else [0., 0., 0.]
        exact([row.get(k) for k in ("cmd_x", "cmd_y", "cmd_yaw")], wanted, "Wrong measured command")
    require(order == ["preparation", "recovery", "move", "stop"], "Interrupted/reordered phase sequence")
    require(counts["preparation"] == 50 and 150 <= counts["recovery"] <= 550 and
            counts["move"] == 400 and counts["stop"] == 300, "Incomplete physical phase lengths")
    return counts


def validate_ledger(frames, rows):
    require(type(frames) is list and len(frames) == len(rows) + 1, "Need t0 plus EVERY completed interval")
    for index, frame in enumerate(frames):
        require(type(frame) is dict and set(frame) == {"frame", "time_s", "completed_interval_step",
                "rendered_preencoding_bgr_sha256"}, "Wrong frame ledger schema")
        require(type(frame["frame"]) is int and frame["frame"] == index and
                type(frame["completed_interval_step"]) is int and frame["completed_interval_step"] == index - 1,
                "Missing/repeated/reordered video frame")
        expected_time = 0. if index == 0 else rows[index - 1]["time_s"]
        require(type(frame["time_s"]) is float and frame["time_s"] == expected_time, "Frame/physical time mismatch")
        require(sha_ok(frame["rendered_preencoding_bgr_sha256"]), "Invalid pre-encoding frame digest")


def validate_sensor_topology(report):
    """Bind the separately recorded sensor and articulation name/index orders."""
    topology = report.get("contact_sensor_topology")
    keys = {"body_names", "num_bodies", "foot_names", "foot_ids", "base_names", "base_ids",
            "articulation_body_names", "foot_articulation_ids"}
    require(type(topology) is dict and set(topology) == keys, "Missing explicit sensor topology")
    exact(topology, report.get("contact_sensor_topology_after"), "Sensor topology changed inside episode")
    sensor, asset = topology["body_names"], topology["articulation_body_names"]
    for names in (sensor, asset):
        require(type(names) is list and bool(names) and all(type(name) is str and name for name in names) and
                len(set(names)) == len(names), "Invalid/duplicate body names")
    require(type(topology["num_bodies"]) is int and topology["num_bodies"] == len(sensor) == len(asset) and
            set(sensor) == set(asset), "Sensor/articulation body sets or counts differ")
    exact(topology["foot_names"], [leg + "_foot" for leg in ("FL", "FR", "RL", "RR")], "Wrong named feet")
    exact(topology["base_names"], ["base"], "Wrong named base")
    for labels, indices, names in ((topology["foot_names"], topology["foot_ids"], sensor),
            (topology["base_names"], topology["base_ids"], sensor),
            (topology["foot_names"], topology["foot_articulation_ids"], asset)):
        require(type(indices) is list and len(indices) == len(labels) and
                all(type(index) is int and 0 <= index < len(names) for index in indices) and
                len(set(indices)) == len(indices), "Invalid named sensor/articulation indices")
        exact([names[index] for index in indices], labels, "Names do not resolve at recorded indices")
    for key in ("physical_interface_before", "physical_interface_after"):
        interfaces = report.get(key)
        require(type(interfaces) is dict and set(interfaces) == {"roll", "stand", "locomotion"},
                "Missing actor interfaces for sensor identity")
        for interface in interfaces.values():
            require(type(interface) is dict, "Invalid physical interface")
            exact(interface.get("body_names"), asset, "Articulation topology differs from physical interface")
    return topology


def validate_report_header(report, view):
    """Metadata rejection checks, NOT proof that any synthetic fixture is physical."""
    require(type(report) is dict and view in ("front", "oblique"), "Wrong report/view")
    require(report.get("protocol_version") == PROTOCOL and report.get("passed") is True and
            report.get("status") == "diagnostic_passed_not_promoted" and report.get("smoke") is False,
            "Only an actually passed full v3 diagnostic is eligible")
    for key in ("live_interface_continuity_passed", "source_hashes_unchanged_at_end", "mirror_classified_once"):
        require(report.get(key) is True, "Missing live evidence: " + key)
    for key in ("post_initialization_reset_allowed", "training_performed", "promotion_performed"):
        require(report.get(key) is False, "Invalid acceptance/reset scope: " + key)
    require(not any(k.startswith("failure") or k == "source_verification_error" for k in report), "Failure evidence present")
    require(report.get("reset_guard_attempts") == [] and report.get("refinement") == "off" and
            report.get("video_view") == view and type(report.get("seed")) is int and
            report.get("pose") in ("side", "upside_down") and type(report.get("num_envs")) is int and report["num_envs"] == 1,
            "Only bank-initialized single-env, refinement-OFF matched views")
    exact([report.get(k) for k in ("control_dt", "physics_dt", "decimation", "preparation_nominal_pd_intervals")],
          [.02, .005, 4, 50], "Changed timing/PD protocol")
    exact(report.get("command_schedule"), {"preparation": [0.,0.,0.], "recovery": [0.,0.,0.],
          "move": [.5,0.,0.], "stop": [0.,0.,0.]}, "Wrong move-speed/schedule")
    require(report.get("policy_action_mode") == "deterministic_mean", "Stochastic replay")
    require(report.get("task") == "Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0", "Wrong physical task")
    exact(report.get("initialization"), {"wrapper_initial_reset": True, "additional_env_reset": False,
          "placement_before_recording": True, "no_policy_history_reset_after_pd": True}, "Wrong initialization/history")
    actors = report.get("actors")
    require(type(actors) is dict and set(actors) == {"roll", "stand", "locomotion"} and
            all(sha_ok(v) for v in actors.values()), "Missing actor identities")
    selection = report.get("locomotion_selection")
    keys = {"path", "sha", "selected_model", "interface_path", "actual_passed_interface_path",
            "screen_report_path", "screen_protocol", "prerequisite", "verified_inputs"}
    require(type(selection) is dict and keys <= set(selection) and selection["sha"] == actors["locomotion"] and
            selection["selected_model"] in ("control3947", "balanced4246") and
            selection["interface_path"] == selection["actual_passed_interface_path"], "Wrong selected actor/interface bundle")
    exact(actors, dict(ACTOR_SHA, locomotion=LOCOMOTION_SHA[selection["selected_model"]]), "Actor differs from frozen entry contract")
    if selection["selected_model"] == "balanced4246":
        limits = selection.get("selection_limitations", {})
        require(limits.get("screen_passed") == 13 and limits.get("screen_total") == 21 and
                limits.get("gait_selection_passed") is False and limits.get("promotion_performed") is False and
                selection.get("completed_screen_identity_audits") == 42 and
                type(selection.get("training_verification")) is dict and bool(selection["training_verification"]),
                "Do not relabel rejected wider/gait selection")
    require(type(report.get("source_and_inputs")) is dict and bool(report["source_and_inputs"]) and
            all(type(k) is str and sha_ok(v) for k,v in report["source_and_inputs"].items()), "No valid source inventory")
    require(type(selection["verified_inputs"]) is dict and bool(selection["verified_inputs"]), "No selected evidence inventory")
    require(set(actors.values()) <= set(report["source_and_inputs"].values()), "Actors absent from verified inventory")
    for key in ("physical_interface_before", "physical_interface_after", "physical_config_after_initialization",
                "physical_config_after_episode", "startup_selection", "release_state", "mirror_joint_contract"):
        require(key in report and bool(report[key]), "Missing physical/startup evidence: " + key)
    exact(report["physical_interface_before"], report["physical_interface_after"], "Interface changed inside episode")
    validate_sensor_topology(report)
    exact(report["physical_config_after_initialization"], report["physical_config_after_episode"], "Physics changed inside episode")
    bank, starts = report.get("bank_selection"), report.get("policy_start_state")
    require(type(bank) is dict and bank.get("selected_unique_state_count") == 1 and
            type(bank.get("selected_state_ids")) is list and len(bank["selected_state_ids"]) == 1 and
            type(starts) is list and len(starts) == 1, "Missing real bank start")
    require(starts[0].get("bank_state_id") == bank["selected_state_ids"][0] and
            starts[0].get("eligible_settled_fallen_recovery") is True and starts[0].get("fallen_at_policy_start") is True,
            "Not eligible measured bank-initialized fallen recovery")
    behavior = report.get("behavior", {})
    require(behavior.get("protocol") == METRICS_PROTOCOL and behavior.get("motion_checks_passed") is True and
            set(behavior.get("checks", {})) == CHECKS and all(v is True for v in behavior["checks"].values()),
            "Missing/failed physical behavior checks")
    state = report.get("flow_state", {})
    require(state.get("phase") == "complete" and state.get("failure_reason") is None and
            state.get("initial_control_step") == 50 and state.get("stop_strict_count") == 150 and
            state.get("recovery_ready_count") == 150 and state.get("recovery_strict_count") == 150,
            "Incomplete quiet-ready flow")
    exact(state.get("config"), {"move_speed": .5, "include_refinement": False}, "Changed flow selection")


def verify_artifacts(directory, inventory):
    require(type(inventory) is dict and REQUIRED_ARTIFACTS <= set(inventory), "Incomplete video artifact inventory")
    result = {}
    for name, expected in inventory.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None and Path(name).name == name and
                name not in (".", "..", REPORT) and sha_ok(expected), "Unsafe artifact name/digest")
        path = (directory / name).resolve(strict=True)
        require(path.parent == directory and path.is_file(), "Artifact escapes source directory")
        require(digest(path) == expected, "Artifact changed: " + name)
        result[str(path)] = expected
    return result


def load_case(directory, view):
    """Read an existing VIDEO case; not a formal-training acceptance API."""
    directory = Path(directory).resolve(strict=True)
    report_path = directory / REPORT
    report_sha = digest(report_path)
    report = read_json(report_path)
    require(digest(report_path) == report_sha, "Report changed while reading")
    validate_report_header(report, view)
    inventory = verify_artifacts(directory, report.get("artifact_sha256"))
    inventory[str(report_path)] = report_sha
    for name, expected in report["source_and_inputs"].items():
        path = Path(name).resolve(strict=True)
        require(path.is_relative_to(ROOT.parent) and sha_ok(expected) and digest(path) == expected,
                "Source/input changed or outside workspace: " + name)
        inventory[str(path)] = expected
    entry = ROOT / "scripts/evaluate_continuous_recovery_flow.py"
    require(report["source_and_inputs"].get(str(entry.resolve())) == ENTRY_SHA and digest(entry) == ENTRY_SHA,
            "Unknown frozen v3 entry")
    selection = report["locomotion_selection"]
    for name, expected in selection["verified_inputs"].items():
        require(report["source_and_inputs"].get(name) == expected, "Selection evidence not in source inventory")
    require(report["source_and_inputs"].get(str(Path(selection["path"]).resolve())) == selection["sha"], "Wrong model path binding")
    rows, frames = read_json(directory / "measured_rows.json"), read_json(directory / "video_frames.json")
    counts = validate_rows_schedule(rows)
    validate_ledger(frames, rows)
    require(type(report.get("actual_completed_intervals")) is int and report["actual_completed_intervals"] == len(rows) and
            type(report.get("frame_count_written")) is int and report["frame_count_written"] == len(frames), "Wrong actual counters")
    state = report["flow_state"]
    require(state.get("completed_intervals") == len(rows) - 50 and state.get("premove_intervals") == counts["recovery"] and
            state.get("move_intervals") == 400 and state.get("stop_intervals") == 300, "Wrong flow interval ledger")
    from continuous_flow_motion_metrics import analyze
    metrics_path = ROOT / "scripts/continuous_flow_motion_metrics.py"
    require(report["source_and_inputs"].get(str(metrics_path.resolve())) == digest(metrics_path), "Unbound behavior implementation")
    exact(analyze(rows, move_speed=.5), report["behavior"], "Recomputed actual full-row behavior differs")
    invocation = read_json(directory / "invocation.json")
    args, evidence = invocation["args"], invocation["evidence"]
    require(args.get("video") is True and args.get("smoke") is False and args.get("view") == view and
            Path(args["output_dir"]).resolve() == directory and args.get("pose") == report["pose"] and
            args.get("seed") == report["seed"] and args.get("speed") == .5 and args.get("refinement") == "off" and
            args.get("locomotion_model") == selection["selected_model"], "Actual invocation differs")
    for role, arg in (("roll", "checkpoint"), ("stand", "stand_checkpoint")):
        require(report["source_and_inputs"].get(str(Path(args[arg]).resolve())) == report["actors"][role], "Wrong actor path")
    require(evidence.get("protocol") == PROTOCOL, "Wrong invocation protocol")
    exact(evidence["source_and_inputs"], report["source_and_inputs"], "Invocation source inventory differs")
    exact(evidence["locomotion"], selection, "Invocation selection differs")
    exact(evidence["prerequisite"], report["prerequisite"], "Invocation prerequisite differs")
    verify_unchanged(inventory)
    return {"directory": directory, "report": report, "rows": rows, "frames": frames,
            "invocation": invocation, "inventory": inventory, "video": directory / VIDEO}


def match_cases(front, oblique):
    require(front["directory"] != oblique["directory"], "Need two independent source directories")
    exact(front["rows"], oblique["rows"], "Full measured trajectories are not exactly identical")
    # Only these fields are view-dependent; all other present AND missing keys must match.
    excluded = {"video_view", "artifact_sha256"}
    exact({k:v for k,v in front["report"].items() if k not in excluded},
          {k:v for k,v in oblique["report"].items() if k not in excluded}, "Replay physical/provenance conditions differ")
    for case in (front, oblique):
        validate_report_header(case["report"], "front" if case is front else "oblique")
    args = [{k:v for k,v in c["invocation"]["args"].items() if k not in {"view", "output_dir"}} for c in (front, oblique)]
    exact(*args, "Invocation differs beyond camera/output")
    exact(front["invocation"]["evidence"], oblique["invocation"]["evidence"], "Evidence bundle differs")
    return len(front["rows"]) + 1


def probe_video(path, expected_frames, width, height, cv2):
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(), "Video cannot be opened: " + str(path))
        require(cap.get(cv2.CAP_PROP_FPS) == FPS and cap.get(cv2.CAP_PROP_FRAME_COUNT) == expected_frames and
                cap.get(cv2.CAP_PROP_FRAME_WIDTH) == width and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == height,
                "Video rate/size/frame-count mismatch")
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape == (height, width, 3), "Decoded frame dimensions changed")
            decoded += 1
        require(decoded == expected_frames, "Truncated/extra decoded video frames")
        return decoded
    finally:
        cap.release()


def verify_unchanged(inventory):
    for name, expected in inventory.items():
        require(digest(name) == expected, "Input/source changed during pairing: " + name)


def build(front_dir, oblique_dir, output_dir, validate_only=False):
    output_dir = Path(output_dir).resolve()
    require(output_dir.drive.upper() == "E:" and output_dir.is_relative_to(ROOT / "evaluations") and
            not output_dir.exists(), "Use a NEW E-drive evaluations output; never overwrite")
    front, oblique = load_case(front_dir, "front"), load_case(oblique_dir, "oblique")
    for case in (front, oblique):
        require(not output_dir.is_relative_to(case["directory"]), "Output cannot alter an input directory")
    expected_frames = match_cases(front, oblique)
    for name in ("OPENCV_FFMPEG_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import cv2
    import imageio_ffmpeg
    import numpy as np
    cv2.setNumThreads(1)
    for case in (front, oblique):
        probe_video(case["video"], expected_frames, SOURCE_W, SOURCE_H, cv2)
        verify_unchanged(case["inventory"])
    if validate_only:
        return {"metadata_and_media_validated": True, "frames_per_view": expected_frames,
                "output_created": False, "new_physical_acceptance": False}
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "continuous_flow_front_oblique.mp4"
    bar = np.zeros((BAR_H, OUTPUT_W, 3), dtype=np.uint8)
    for i, line in enumerate(BANNER):
        cv2.putText(bar, line, (18, 29 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, .64, (245,245,245), 1, cv2.LINE_AA)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUTPUT_W}x{OUTPUT_H}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    caps = [cv2.VideoCapture(str(case["video"])) for case in (front, oblique)]
    try:
        for index in range(expected_frames):
            decoded = [cap.read() for cap in caps]
            require(all(ok and frame.shape == (SOURCE_H, SOURCE_W, 3) for ok, frame in decoded),
                    "Source decode changed at frame " + str(index))
            joined = np.vstack((bar, np.hstack((decoded[0][1], decoded[1][1]))))
            encoder.stdin.write(joined.tobytes())
        require(not any(cap.read()[0] for cap in caps), "Extra source frames during encoding")
        encoder.stdin.close()
        require(encoder.wait() == 0, "Encoder failed; partial output retained without success receipt")
    except BaseException:
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
        encoder.wait()
        raise
    finally:
        for cap in caps:
            cap.release()
    decoded = probe_video(output, expected_frames, OUTPUT_W, OUTPUT_H, cv2)
    for case in (front, oblique):
        verify_unchanged(case["inventory"])
    receipt = {"protocol": "matched_continuous_flow_video_pair_v1", "pairing_completed": True,
               "new_physical_acceptance": False, "promotion_performed": False, "simultaneous_cameras": False,
               "initial_fall": "bank initialized before recorded nominal-PD preparation, not a natural fall during walking",
               "source_report_passed": True, "source_protocol": PROTOCOL, "captions": list(BANNER),
               "actors": front["report"]["actors"], "locomotion_selection": front["report"]["locomotion_selection"],
               "measured_rows_exact_match": True, "rows": len(front["rows"]),
               "canonical_rows_sha256": hashlib.sha256(canonical(front["rows"]).encode()).hexdigest(),
               "sources": [{"view": v, "directory": str(c["directory"]), "verified_sha256": c["inventory"]}
                           for v,c in (("front",front),("oblique",oblique))],
               "composition": {"left": "front", "right": "oblique", "source_size": [SOURCE_W,SOURCE_H],
                               "banner_height": BAR_H, "cropped": False, "resized": False, "speed_factor": 1,
                               "extra_title_frames": 0, "frame_mapping": "output[k] = front[k] beside oblique[k], including t0"},
               "output": {"path": str(output), "sha256": digest(output), "frames": decoded, "fps": FPS,
                          "width": OUTPUT_W, "height": OUTPUT_H, "duration_s": decoded/FPS,
                          "physical_endpoint_s": len(front["rows"])/FPS, "codec": "H.264 yuv420p CRF18"},
               "limits": "Lossy re-encoding preserves every decoded source frame and native geometry, not pixel hashes. "
                         "Recorded pre-encoding frame hashes are artifact-bound but cannot be recovered from lossy MP4. "
                         "This pairer rechecks full behavior rows; it is not an independent simulator/full-interface audit.",
               "builder_sha256": digest(__file__)}
    with (output_dir / "pairing_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--oblique", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    result = build(options.front, options.oblique, options.output_dir, options.validate_only)
    print(json.dumps({"pairing_completed": result.get("pairing_completed", False),
                      "output": result.get("output"), "new_physical_acceptance": False}, indent=2))
