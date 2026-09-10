"""Pair complete front/oblique recovery replays and encode browser-compatible H.264."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def pair(front_report, oblique_report, output):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (front_report, oblique_report)]
    for key in ("checkpoint_sha256", "task", "seed", "angle_deg", "protocol_version", "criterion"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Mismatched replay metadata: {key}")
    if [report["video_view"] for report in reports] != ["front", "oblique"]:
        raise ValueError("Pass front and oblique reports, in that order.")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for pose in reports[0]["results"]:
        inputs = []
        properties = []
        for path, report in zip((front_report, oblique_report), reports):
            if report["results"][pose]["trials"] != 1:
                raise ValueError("Paired videos must use single-environment replays.")
            video = path.parent / f"{Path(report['checkpoint']).name}_{pose}.mp4"
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot read {video}")
            properties.append(tuple(cap.get(prop) for prop in (
                cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            inputs.append(video)
        if properties[0] != properties[1] or properties[0][0] <= 0:
            raise ValueError(f"Unequal durations or image heights: {properties}")
        target = output / f"{Path(reports[0]['checkpoint']).stem}_{pose}_front_oblique.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(inputs[0]), "-i", str(inputs[1]),
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(target)], check=True, capture_output=True)
        artifacts[pose] = {"video": target.name, "frames": properties[0][0], "fps": properties[0][1]}
        print(target)
    metadata = {
        "note": "Left: front; right: oblique. Independent same-seed replays, full duration without time cropping.",
        "checkpoint_sha256": reports[0]["checkpoint_sha256"],
        "source_reports": [Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()
                           for path in (front_report, oblique_report)],
        "artifacts": artifacts,
    }
    (output / "paired_views.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_report", type=Path)
    parser.add_argument("oblique_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pair(args.front_report, args.oblique_report, args.output)
