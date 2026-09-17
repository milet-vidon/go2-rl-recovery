"""CPU-only, complete native-size locomotion replay pair, diagnostic only.

The two views are independent matched replays, not synchronized cameras.
No source cropping, resizing, frame selection, simulator or policy mutation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

# Bound decoder/OpenCV worker use before importing either library.
os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import imageio_ffmpeg
import numpy as np

from build_full_scenario_review import (
    BAR_HEIGHT, FPS, HEIGHT, SOURCE_HEIGHT, SOURCE_WIDTH, WIDTH,
    compose_frame, draw_text_lines, probe_complete_video, require,
    sha256, source_record, validate_locomotion_reports,
)


ROOT = Path(__file__).resolve().parents[1]
STEM = "model_3947_stand_walk_stop"
EXPECTED_SHA = "aaa3fcdd796bfb99feb95a4940889c8fd4ebeff61282aac0b258ec856b1d7604"
DEFAULT_FRONT = ROOT / "evaluations/20260917-speed3947-video-target10-front"
DEFAULT_OBLIQUE = ROOT / "evaluations/20260917-speed3947-video-target10-oblique"
DEFAULT_OUTPUT = ROOT / "evaluations/20260917-speed3947-diagnostic-paired"


def probe_paired_output(path, expected_frames):
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(), "Cannot decode paired output")
        require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
                and cap.get(cv2.CAP_PROP_FPS) == FPS
                and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == WIDTH
                and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == HEIGHT,
                "Paired output dimensions/rate/frame count mismatch")
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape == (HEIGHT, WIDTH, 3), "Output frame dimensions changed")
            decoded += 1
        require(decoded == expected_frames, f"Incomplete output decode: {decoded}/{expected_frames}")
        return decoded
    finally:
        cap.release()


def build(front_dir, oblique_dir, output_dir, validate_only=False):
    cv2.setNumThreads(1)
    output_dir = Path(output_dir).resolve()
    require(output_dir.drive.upper() == "E:", "Output must remain on E:")
    require(not output_dir.exists(), f"Refusing existing output directory: {output_dir}")
    directories = [Path(front_dir).resolve(strict=True), Path(oblique_dir).resolve(strict=True)]
    report_paths = [directory / f"{STEM}.json" for directory in directories]
    videos = [directory / f"{STEM}.mp4" for directory in directories]
    reports = [json.loads(path.read_text(encoding="utf-8-sig")) for path in report_paths]
    validate_locomotion_reports(*reports)
    first = reports[0]
    require(first["checkpoint_sha256"] == EXPECTED_SHA, "Not the frozen completed speed3947 experiment")
    require(first["seed"] == 20260909, "Unexpected recorded replay seed")
    require(first["protocol"] == {"stand_s": 4., "walk_s": 8., "stop_s": 6.,
                                  "walk_speed": 1., "lateral_speed": 0., "yaw_rate": 0.,
                                  "push_delta_vy": 0.}, "Unexpected complete stand/walk/stop command")
    require(first["passed"] is False and first["acceptance"]["walk_tracking"] is False,
            "This output is explicitly the non-accepted 1.0-m/s diagnostic")
    checkpoints = [Path(report["checkpoint"]).resolve(strict=True) for report in reports]
    for checkpoint in checkpoints:
        require(checkpoint.name == "model_3947.pt" and sha256(checkpoint) == EXPECTED_SHA,
                "Checkpoint file does not match report/model identity")
    sources = []
    expected_frames = 18 * FPS
    for view, report_path, video in zip(("front", "oblique"), report_paths, videos):
        report_record = source_record(report_path, output_dir)
        video_record = source_record(video, output_dir)
        video_record.update(probe_complete_video(video, expected_frames))
        require(sha256(video) == video_record["sha256"], "Source changed while validating")
        sources.append({"view": view, "report": report_record, "video": video_record})
    failed = [key for key, passed in first["acceptance"].items() if not passed]
    walk = first["settled_phase_stats"]["walk"]
    lines = [
        "DIAGNOSTIC | model 3947 | target 1.0 m/s | NOT ACCEPTED",
        f"Complete stand 4s / walk 8s / stop 6s | measured mean vx {walk['vx_b_mean']:.3f} m/s | failed: {', '.join(failed)}",
        "LEFT: FRONT  |  RIGHT: OBLIQUE  |  Independent matched replays, NOT synchronized cameras",
        "All 450 source frames retained at native 960x540 per view / 25 fps. No trimming or resizing.",
        f"SHA256 {EXPECTED_SHA[:16]} | seed 20260909 | Simulation only; no fall-recovery or continuous controller claim.",
        "Single-seed visual diagnostic is NOT the multi-seed acceptance result; baseline recommendations unchanged.",
    ]
    bar = np.zeros((BAR_HEIGHT, WIDTH, 3), dtype=np.uint8)
    draw_text_lines(bar, lines, 22, 25, (100, 155, 255), scale=.62)
    if validate_only:
        print("Validated both complete 450-frame source videos and report/model identities; no output created.")
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model_3947_target10_diagnostic_front_oblique.mp4"
    manifest_path = output.with_suffix(".json")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    caps = [cv2.VideoCapture(str(path)) for path in videos]
    try:
        for index in range(expected_frames):
            frames = [cap.read() for cap in caps]
            require(all(ok for ok, _ in frames), f"Source decode failed at paired frame {index}")
            encoder.stdin.write(compose_frame(frames[0][1], frames[1][1], bar).tobytes())
        require(not any(cap.read()[0] for cap in caps), "Unexpected extra source frames")
        encoder.stdin.close()
        require(encoder.wait() == 0, "Encoder failed; partial output retained, no manifest")
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
    decoded = probe_paired_output(output, expected_frames)
    for source in sources:
        for kind in ("report", "video"):
            record = source[kind]
            require(sha256(output_dir / record["path"]) == record["sha256"], "Source changed during encoding")
    require(all(sha256(path) == EXPECTED_SHA for path in checkpoints), "Checkpoint changed during encoding")
    manifest = {
        "schema_version": "go2_locomotion_diagnostic_pair_v1", "diagnostic_only": True,
        "acceptance_eligible": False, "promotion_performed": False,
        "note": "Independent matched replays, not synchronized cameras. Complete 18-s footage; no cropping, resizing or trimming. Simulation only.",
        "checkpoint_sha256": EXPECTED_SHA, "checkpoints": [source_record(p, output_dir) for p in checkpoints],
        "task": first["task"], "seed": first["seed"], "protocol": first["protocol"],
        "report_passed": first["passed"], "failed_report_criteria": failed,
        "acceptance": first["acceptance"], "captions": lines, "sources": sources,
        "composition": {"left": "front", "right": "oblique", "source_width": SOURCE_WIDTH,
                        "source_height": SOURCE_HEIGHT, "caption_height": BAR_HEIGHT,
                        "original_render_regions_preserved": True, "added_title_card_frames": 0},
        "output": {**source_record(output, output_dir), "frames": expected_frames, "decoded_frames": decoded,
                   "fps": FPS, "duration_s": expected_frames / FPS, "width": WIDTH, "height": HEIGHT,
                   "codec": "H.264", "pixel_format": "yuv420p", "audio": False},
        "builder_sources": [source_record(Path(__file__), output_dir),
                            source_record(Path(__file__).with_name("build_full_scenario_review.py"), output_dir)],
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)
    print(manifest_path)
    print(f"Verified {decoded} complete output frames / {decoded / FPS}s; two CPU encoder threads, no simulator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, default=DEFAULT_FRONT)
    parser.add_argument("--oblique", type=Path, default=DEFAULT_OBLIQUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    build(args.front, args.oblique, args.output_dir, args.validate_only)
