"""Pair complete trial-0 views from verified 20-environment startup replays."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), f"Cannot decode {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape[:2] == dimensions[::-1], "Frame dimensions changed")
            count += 1
        require(fps > 0 and count > 0 and count == advertised, "Incomplete video")
        return {"frames": count, "fps": fps, "width": dimensions[0], "height": dimensions[1]}
    finally:
        cap.release()


def validate(front, oblique):
    paths = [front, oblique]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    require([r["video_view"] for r in reports] == ["front", "oblique"], "Incorrect view order")
    require({k: v for k, v in reports[0].items() if k != "video_view"} ==
            {k: v for k, v in reports[1].items() if k != "video_view"},
            "Reports differ beyond the camera view")
    sources = []
    for report, path in zip(reports, paths):
        require(report["protocol_version"] == "handoff_supported_startup_experimental_v1",
                "Wrong experimental protocol")
        require(report["diagnostic_trial"] == 0 and set(report["results"]) == {"upright"},
                "This demo records only upright trial 0")
        require(report["seed"] == 20260918, "Unexpected development seed")
        result = report["results"]["upright"]
        require(result["trials"] == result["final_valid_stands"] == 20, "Expected 20-trial result")
        startup = report["startup_experiment"]
        require(startup["mode"] == "supported" and not startup["mirror_enabled"], "Wrong selector")
        require(not report["acceptance_eligible"] and not report["training_collection_eligible"],
                "Only diagnostic evidence may be paired")
        controller = report["handoff_controller"]
        require(controller["roll_sha256"] ==
                "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c", "Wrong roll actor")
        require(controller["stand_sha256"] ==
                "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb", "Wrong stand actor")
        for name in ("roll", "stand"):
            require(sha(Path(controller[f"{name}_checkpoint"])) == controller[f"{name}_sha256"],
                    "Checkpoint file changed")
        require(report["checkpoint_sha256"] == controller["roll_sha256"], "Roll identity mismatch")
        for file_key, hash_key in (("transition_trace_file", "transition_trace_sha256"),
                                   ("generated_source_file", "generated_source_sha256")):
            artifact = (path.parent / startup[file_key]).resolve()
            require(artifact.parent == path.parent.resolve(), "Artifact escaped source directory")
            require(sha(artifact) == startup[hash_key], "Diagnostic source/trace hash mismatch")
        video = path.parent / "model_1999.pt_upright.mp4"
        csv_path = path.parent / report["trace_csv"]
        sources.append({"report": str(path), "report_sha256": sha(path),
                        "video": str(video), "video_sha256": sha(video),
                        "trace_csv_sha256": sha(csv_path),
                        "transition_trace_sha256": startup["transition_trace_sha256"],
                        "generated_source_sha256": startup["generated_source_sha256"]})
    require(sources[0]["trace_csv_sha256"] == sources[1]["trace_csv_sha256"],
            "Actual trial-0 trajectory CSV differs")
    return reports, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_report", type=Path)
    parser.add_argument("oblique_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.front_report, args.oblique_report)]
    output = args.output_dir.resolve()
    require(all(p.drive.upper() == "E:" for p in [*paths, output]), "Keep artifacts on E:")
    if output.exists():
        raise FileExistsError("Use a fresh output directory; preserve previous evidence")
    reports, sources = validate(*paths)
    properties = [read_video(Path(s["video"])) for s in sources]
    require(properties[0] == properties[1], "Source frame count/rate/size mismatch")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "supported_startup_front_oblique.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-i", sources[0]["video"],
                    "-i", sources[1]["video"], "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                    "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(target)], check=True, capture_output=True)
    actual = read_video(target)
    require(actual == {**properties[0], "width": 2 * properties[0]["width"]},
            "Output changed duration/rate or lost frames")
    metadata = {"protocol": "supported_startup_paired_trial0_v1", "sources": sources,
                "source_properties": properties, "output": str(target), "output_sha256": sha(target),
                "output_properties": actual, "duration_s": actual["frames"] / actual["fps"],
                "pairing_script_sha256": sha(Path(__file__)),
                "all_report_fields_except_view_exact": True, "trial0_csv_exact": True,
                "handoff_controller": reports[0]["handoff_controller"],
                "startup_experiment": reports[0]["startup_experiment"],
                "note": "Left front, right oblique: independent matched-condition replays, NOT simultaneous cameras. Full untrimmed frames. Only trial 0 of each 20-environment development batch is filmed, after unchanged 1s nominal-PD preparation. This is ordinary supported startup, NOT fallen recovery, a continuous walk/recover/run demonstration, or promotion."}
    (output / "paired_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(target), "verified": actual, "sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
