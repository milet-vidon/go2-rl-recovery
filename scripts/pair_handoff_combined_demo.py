"""Pair full trial-0 front/oblique videos from matching 20-environment diagnostics.

One pose per invocation. Failed outcomes are retained, not filtered. The output
is two independent matched-condition replays, never a continuous full workflow.
Video dependencies are imported only inside main; validation helpers are pure.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import evaluate_handoff_combined as entry


POSES = ("upright", "side", "upside_down")
TRACE_GROUPS = {
    "transition_experiment": ("transition_trace_file", "transition_trace_sha256"),
    "startup_experiment": ("transition_trace_file", "transition_trace_sha256"),
    "mirror_experiment": ("mirror_trace_file", "mirror_trace_sha256"),
    "combined_experiment": ("trace_file", "trace_sha256"),
}
EXPECTED_FRAMES = 276
EXPECTED_FPS = 25.0
TRANSITION_MATH_SHA = "b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_equal(left, right, label="value"):
    require(type(left) is type(right), label + ": types differ")
    if type(left) is dict:
        require(left.keys() == right.keys(), label + ": keys differ")
        for key in left:
            strict_equal(left[key], right[key], label + "." + key)
    elif type(left) is list:
        require(len(left) == len(right), label + ": lengths differ")
        for i, (a, b) in enumerate(zip(left, right)):
            strict_equal(a, b, f"{label}[{i}]")
    else:
        if type(left) is float:
            require(math.isfinite(left) and math.isfinite(right), label + ": nonfinite value")
        require(left == right, label + ": exact values differ")


def validate_report_pair(front, oblique):
    """Pure validation; intentionally does not require successful recovery."""
    require([front.get("video_view"), oblique.get("video_view")] == ["front", "oblique"], "Pass front then oblique")
    strict_equal({k: v for k, v in front.items() if k != "video_view"},
                 {k: v for k, v in oblique.items() if k != "video_view"}, "reports_except_view")
    require(front["protocol_version"] == entry.PROTOCOL and front["controller_type"] == entry.CONTROLLER,
            "Wrong combined protocol/controller")
    require(type(front["diagnostic_trial"]) is int and front["diagnostic_trial"] == 0,
            "Only trial 0 is filmed")
    require(type(front["seed"]) is int and front["seed"] == 20260918, "Wrong development seed")
    require(len(front["results"]) == 1, "One independent pose per pair")
    pose = next(iter(front["results"]))
    require(pose in POSES, "Unsupported pose")
    result = front["results"][pose]
    require(type(result["trials"]) is int and result["trials"] == 20, "Require 20-environment evaluation")
    for count in ("successes", "final_valid_stands", "final_geometry_passes"):
        require(type(result[count]) is int and 0 <= result[count] <= 20, "Invalid outcome count")
    for collection in ("policy_start_state", "release_state_before_settling", "final_diagnostics"):
        require(len(result[collection]) == 20 and [r["trial"] for r in result[collection]] == list(range(20)),
                "Incomplete or reordered actual trial records")
    for key in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible"):
        require(front[key] is False, "Pairing must not claim acceptance or collection")
    startup, mirror, combined = (front[k] for k in ("startup_experiment", "mirror_experiment", "combined_experiment"))
    require(startup["mode"] == "supported" and startup["mirror_enabled"] is True and
            mirror["mode"] == "initial_right" and mirror["startup_selection_changed"] is True and
            combined["startup_mode"] == "supported" and combined["mirror_mode"] == "initial_right",
            "Only the explicit BOTH ON controller is supported")
    transition = front["transition_experiment"]
    for key, expected in {"mode": "hard", "ramp_seconds": 0.0, "ramp_steps": 0, "step_dt": .02,
                          "horizon_s": 8.0, "hold_s": 3.0, "min_contacts": 4, "policy_control_steps": 550}.items():
        strict_equal(transition[key], expected, "transition." + key)
    for group in (startup, mirror):
        require(group["ramp_enabled"] is False and group["retry_enabled"] is False, "Extra factor enabled")
    controller = front["handoff_controller"]
    require(controller["roll_sha256"] == entry.ROLL_SHA == front["checkpoint_sha256"] and
            controller["stand_sha256"] == entry.STAND_SHA, "Wrong frozen actors")
    names = [front[group][keys[0]] for group, keys in TRACE_GROUPS.items()]
    hashes = [front[group][keys[1]] for group, keys in TRACE_GROUPS.items()]
    require(names == ["model_1999.pt_handoff_combined_trace.json"] * 4 and len(set(hashes)) == 1,
            "Conflicting trace identity across four groups")
    require(startup["generated_source_file"] == combined["generated_source_file"] == "combined_generated.py" and
            startup["generated_source_sha256"] == combined["generated_source_sha256"], "Conflicting generated source")
    return pose


def validate_video_pair(left, right):
    """Pure timing/shape guard; equal but truncated inputs must fail."""
    strict_equal(left, right, "source_video_properties")
    require(type(left["frames"]) is int and left["frames"] == EXPECTED_FRAMES and
            type(left["fps"]) is float and left["fps"] == EXPECTED_FPS, "Require full 276-frame, 25-fps recordings")
    for key in ("width", "height"):
        require(type(left[key]) is int and left[key] > 0 and left[key] % 2 == 0, "Invalid frame dimensions")


def sibling(parent, name):
    require(type(name) is str and Path(name).name == name and ":" not in name and "\\" not in name,
            "Artifact must be a local sibling filename")
    path = (parent / name).resolve()
    require(path.parent == parent.resolve() and path.is_file(), "Missing/escaping sibling artifact")
    return path


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def verify_artifacts(path, report, pose):
    controller, startup, mirror, combined = (report[k] for k in (
        "handoff_controller", "startup_experiment", "mirror_experiment", "combined_experiment"))
    for name in ("roll", "stand"):
        actor = Path(controller[f"{name}_checkpoint"]).resolve()
        require(actor.drive.upper() == "E:" and sha(actor) == controller[f"{name}_sha256"], "Actor file changed")
    adapter_sha = sha(Path(entry.__file__))
    require(startup["adapter_sha256"] == combined["adapter_sha256"] ==
            mirror["evaluator_source_sha256"] == report["transition_experiment"]["evaluator_source_sha256"] == adapter_sha,
            "Adapter source identity changed")
    require(combined["template_sha256"] == entry.TEMPLATE_SHA == sha(entry.TEMPLATE), "Mirror template changed")
    require(startup["startup_reference_adapter_sha256"] == combined["startup_reference_adapter_sha256"] ==
            entry.STARTUP_REFERENCE_SHA == sha(entry.STARTUP_REFERENCE), "Startup reference changed")
    require(startup["math_source_sha256"] == combined["startup_math_sha256"] == entry.MATH_SHA == sha(entry.MATH_PATH),
            "Startup predicate changed")
    require(mirror["math_source_sha256"] == combined["mirror_math_sha256"] == entry.MIRROR_MATH_SHA == sha(entry.MIRROR_MATH_PATH),
            "Mirror mapping changed")
    require(report["transition_experiment"]["math_source_sha256"] == TRANSITION_MATH_SHA ==
            sha(entry.ROOT / "src/go2_recovery/handoff_transition_math.py"), "Transition math changed")
    generated = sibling(path.parent, "combined_generated.py")
    require(sha(generated) == startup["generated_source_sha256"] ==
            hashlib.sha256(entry.build_source().encode("utf-8")).hexdigest(), "Generated source changed")
    trace_path = sibling(path.parent, startup["transition_trace_file"])
    trace = read_json(trace_path)
    require(trace["schema"] == "handoff_combined_neighborhood_v1", "Wrong trace schema")
    for group, (_, hash_key) in TRACE_GROUPS.items():
        require(sha(trace_path) == report[group][hash_key], "Trace SHA mismatch")
        strict_equal({k: v for k, v in report[group].items() if k != hash_key}, trace[group], "trace." + group)
    require(set(trace["poses"]) == {pose} and len(trace["poses"][pose]) == 20 and
            len(trace["first_action_by_pose"][pose]) == 20, "Incomplete trace population")
    strict_equal(trace["startup_selection_by_pose"][pose],
                 report["results"][pose]["startup_selection"]["selection_records"], "trace.startup_decisions")
    csv_path = sibling(path.parent, report["trace_csv"])
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 551, "Require initial frame and complete 550-step trial-0 CSV")
    video = sibling(path.parent, f"model_1999.pt_{pose}.mp4")
    return {"report": str(path), "report_sha256": sha(path), "video": str(video), "video_sha256": sha(video),
            "trace_csv_sha256": sha(csv_path), "transition_trace_sha256": sha(trace_path),
            "generated_source_sha256": sha(generated), "adapter_sha256": adapter_sha}


def read_video(path, cv2):
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), f"Cannot decode {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape[:2] == (height, width), "Frame dimensions changed")
            count += 1
        require(math.isfinite(fps) and fps > 0 and count > 0 and count == advertised, "Incomplete video")
        return {"frames": count, "fps": fps, "width": width, "height": height}
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_report", type=Path)
    parser.add_argument("oblique_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = [args.front_report.resolve(), args.oblique_report.resolve()]
    output = args.output_dir.resolve()
    require(all(p.drive.upper() == "E:" for p in [*paths, output]), "All artifacts must remain on E:")
    require(output != Path("E:/") and not output.exists(), "Use a fresh E-drive output directory; preserve all evidence")
    reports = [read_json(p) for p in paths]
    pose = validate_report_pair(*reports)
    sources = [verify_artifacts(p, r, pose) for p, r in zip(paths, reports)]
    require(sources[0]["trace_csv_sha256"] == sources[1]["trace_csv_sha256"], "Trial-0 trajectories differ")
    import cv2
    import imageio_ffmpeg
    properties = [read_video(Path(s["video"]), cv2) for s in sources]
    validate_video_pair(*properties)
    output.mkdir(parents=True, exist_ok=False)
    target = output / f"combined_{pose}_front_oblique.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-n", "-i", sources[0]["video"], "-i", sources[1]["video"],
               "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an", "-c:v", "libx264",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True, capture_output=True)
    actual = read_video(target, cv2)
    strict_equal(actual, {**properties[0], "width": 2 * properties[0]["width"]}, "encoded_output")
    result = reports[0]["results"][pose]
    metadata = {"protocol": "combined_paired_trial0_v1", "pose": pose, "trials": 20, "video_trial": 0,
        "sources": sources, "source_properties": properties, "output": str(target), "output_sha256": sha(target),
        "output_properties": actual, "duration_s": actual["frames"] / actual["fps"],
        "pairing_script_sha256": sha(Path(__file__)), "all_report_fields_except_view_exact": True,
        "trial0_csv_exact": True, "successes": result["successes"], "final_valid_stands": result["final_valid_stands"],
        "acceptance_eligible": False, "training_collection_eligible": False, "promotion_performed": False,
        "handoff_controller": reports[0]["handoff_controller"], "combined_experiment": reports[0]["combined_experiment"],
        "note": "Left front, right oblique: independent matched-condition replays, NOT simultaneous cameras. Complete 276 frames at25fps, no trimming, cropping or speed change. Only trial0 of each20-environment development batch is filmed, after1s nominal-PD preparation. This single-pose experiment is NOT a continuous recovery/walk/run workflow; separate poses must be labeled separate experiments. Failed outcomes are retained honestly; pairing is not behavioral acceptance."}
    with (output / "paired_metadata.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"video": str(target), "verified": actual, "sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
