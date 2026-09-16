"""CPU-only, fail-closed assembly of complete, independently replayed Go2 videos.

No simulator imports. This review is eight separately evaluated scenarios, NOT a
continuous policy-switching demonstration. Inputs are never trimmed or resized.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from pair_recovery_views import validate_reports


VIEWS = ("front", "oblique")
SOURCE_WIDTH, SOURCE_HEIGHT, FPS = 960, 540, 25
WIDTH, BAR_HEIGHT = SOURCE_WIDTH * 2, 160
HEIGHT = SOURCE_HEIGHT + BAR_HEIGHT
CARD_SECONDS = 2
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "evaluations/20260916-target128x2000-video-review"
LOCOMOTION_TASK = "Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0"
ARM_TASKS = {arm: f"Isaac-Recovery-Bank-{name}Target-Flat-Unitree-Go2-Play-v0"
             for arm, name in (("nominal", "Nominal"), ("current", "Current"))}
LOCOMOTION_STEM = "natural_stop_diagonal_model2250_stand_walk_stop"
# This is deliberately a review of this exact experiment, not arbitrary files
# named model_1999.pt. Recovery hashes were verified against the completed
# 20260916-target128x2000-{nominal,current}-heldout reports and checkpoint bytes.
EXPECTED_CHECKPOINTS = {
    "retained_locomotion2250": "33be609b3daab4a2f773d70aeeb41bc42c5cbc843780efc963ed04d5dd368417",
    "nominal": "58f3231a1959a3d5fa82142e9dcfb2aee2b7cc86f1575e3cee25e36c9efc6ca2",
    "current": "5479e23fd0617026a2d913eb509356cc5c8d2324a2601dd0e26b4556e9cb8710",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path, output_dir):
    path = Path(path).resolve(strict=True)
    return {"path": Path(os.path.relpath(path, output_dir.resolve())).as_posix(),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_checkpoint_identity(report, group):
    require(group in EXPECTED_CHECKPOINTS and report.get("checkpoint_sha256") == EXPECTED_CHECKPOINTS[group],
            f"Checkpoint is not the specified completed experiment/baseline: {group}")


def validate_recovery_result(report, pose):
    """Cross-check single-trial aggregates against the actual evaluator schema.

The final report does not contain base force, body speeds or foot x positions.
Do not invent those measurements: the evaluator's continuous hold counter and
geometry flag cover them; consistency of all available final fields is checked.
"""
    bank = pose != "upright"
    expected_protocol = "stance_geometry_v1_state_bank_PD_v1" if bank else "stance_geometry_v1"
    require(report.get("protocol_version") == expected_protocol, "Unexpected recovery geometry protocol")
    require(report.get("acceptance_eligible") is True, "Non-acceptance recovery report")
    result = report["results"][pose]
    require(type(result.get("trials")) is int and result["trials"] == 1, "Expected one recovery trial")
    counts = ("successes", "final_valid_stands", "final_geometry_passes", "standing_starts_not_fallen_recovery",
              "settled_fallen_trials", "settled_fallen_recovery_successes", "settled_fallen_final_valid_stands")
    for key in counts:
        require(type(result.get(key)) is int and result[key] in (0, 1), f"Invalid single-trial count: {key}")
    require(result.get("stable_hold_s") == 3 and result.get("horizon_s") == 8,
            "Expected unchanged 3-second hold and 8-second horizon")
    for key in ("policy_start_state", "final_diagnostics"):
        require(isinstance(result.get(key), list) and len(result[key]) == 1,
                f"Missing single-trial record: {key}")
        require(type(result[key][0].get("trial")) is int and result[key][0]["trial"] == 0,
                f"Invalid trial index: {key}")
    start, final = result["policy_start_state"][0], result["final_diagnostics"][0]
    require(type(start.get("settled")) is bool and type(start.get("eligible_settled_fallen_recovery")) is bool,
            "Invalid settled/eligible start flags")
    eligible = (start["settled"] is True and start.get("fallen_at_policy_start") is True
                and start.get("standing_at_policy_start") is False)
    require(start["eligible_settled_fallen_recovery"] is eligible,
            "Start eligibility contradicts settled/fallen/standing flags")
    require(result["settled_fallen_trials"] == int(eligible), "Fallen trial count contradicts start record")
    require(result["standing_starts_not_fallen_recovery"] == int(start.get("standing_at_policy_start") is True),
            "Standing-start count contradicts start record")
    if bank:
        require(eligible and start.get("contacts_fresh_since_pose_write") is True,
                "Bank review must start from a confirmed settled fallen state with fresh contacts")
        require(start.get("any_body_contact") is True and start.get("quiet_supported_window_s", 0) >= 0.25,
                "Bank start lacks the measured settled contact window")
        require(final.get("bank_state_id") == start.get("bank_state_id"), "Final/start bank state ID mismatch")
    for key, aggregate in (("settled_fallen_recovery_successes", "successes"),
                           ("settled_fallen_final_valid_stands", "final_valid_stands")):
        require(result[key] == (result[aggregate] if eligible else 0), f"Fallen count contradicts trial: {key}")

    def finite_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    require(type(final.get("geometry_ok")) is bool, "Missing final geometry flag")
    for key in ("stable_hold_s", "height_m", "gravity_error", "max_joint_offset_rad"):
        require(finite_number(final.get(key)), f"Missing/nonfinite final measurement: {key}")
    for key in ("feet_y_b", "knees_y_b"):
        require(isinstance(final.get(key), list) and len(final[key]) == 4
                and all(finite_number(v) for v in final[key]), f"Invalid final geometry measurements: {key}")
    require(type(final.get("vertical_foot_contacts")) is int and 0 <= final["vertical_foot_contacts"] <= 4,
            "Invalid final current vertical support count")
    require(final["gravity_error"] >= 0 and final["max_joint_offset_rad"] >= 0
            and 0 <= final["stable_hold_s"] <= result["horizon_s"] + result["stable_hold_s"],
            "Invalid negative/out-of-horizon final measurement")
    geometry_fields_ok = (all(0.06 < sign * y < 0.30 for sign, y in zip((1, -1, 1, -1), final["feet_y_b"]))
                          and all(sign * y > 0.04 for sign, y in zip((1, -1, 1, -1), final["knees_y_b"]))
                          and final["max_joint_offset_rad"] < 0.65)
    require(not final["geometry_ok"] or geometry_fields_ok, "Final geometry flag contradicts measured limbs")
    require(result["final_geometry_passes"] == int(final["geometry_ok"]), "Final geometry aggregate mismatch")
    # Any positive counter means the complete strict predicate held at the last step.
    final_available_checks = (final["geometry_ok"] and geometry_fields_ok
                              and final["vertical_foot_contacts"] == 4
                              and 0.30 < final["height_m"] < 0.55 and final["gravity_error"] < 0.35)
    require(final["stable_hold_s"] == 0 or final_available_checks,
            "Positive final hold contradicts geometry/height/tilt/current four-foot support")
    held = int(final["stable_hold_s"] >= result["stable_hold_s"])
    require(result["final_valid_stands"] == held, "Final-valid count contradicts recorded continuous hold")
    require(result["final_valid_stands"] <= result["successes"], "Final success cannot exceed any-time success")


def validate_locomotion_reports(front, oblique):
    """Compare protocol and discrete outcomes, not floating-point trajectories."""
    keys = ("checkpoint_sha256", "task", "seed", "protocol_version", "rest_geometry_protocol",
            "protocol", "criteria", "rest_stance_criterion", "acceptance", "passed")
    for key in keys:
        require(key in front and key in oblique and front[key] == oblique[key],
                f"Mismatched or missing locomotion report field: {key}")
    require([r.get("video_view") for r in (front, oblique)] == list(VIEWS),
            "Locomotion inputs must be front then oblique")
    require(front["task"] == LOCOMOTION_TASK, "Unexpected locomotion task")
    require(front["protocol_version"] == "stand_walk_stop_stance_geometry_v2"
            and front["rest_geometry_protocol"] == "stance_geometry_v1", "Outdated locomotion protocol")
    acceptance = front["acceptance"]
    require(isinstance(acceptance, dict) and bool(acceptance)
            and all(type(value) is bool for value in acceptance.values()), "Invalid acceptance values")
    for key in ("no_reset", "normal_stance_geometry_at_rest", "four_vertical_contacts_at_rest",
                "no_current_base_contact_at_rest", "geometry_and_support_at_rest"):
        require(key in acceptance, f"Missing stance acceptance check: {key}")
    require(type(front["passed"]) is bool and front["passed"] == all(acceptance.values()),
            "Locomotion passed flag contradicts acceptance")


def probe_complete_video(path, expected_frames, allow_initial_frame=False):
    """Decode the whole source before publishing any output, detecting truncation."""
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(), f"Cannot open video: {path}")
        metadata = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        allowed_counts = (expected_frames, expected_frames + 1) if allow_initial_frame else (expected_frames,)
        require(metadata["frames"] in allowed_counts and metadata["fps"] == float(FPS)
                and metadata["width"] == SOURCE_WIDTH and metadata["height"] == SOURCE_HEIGHT,
                f"Unexpected full source dimensions/duration: {path}: {metadata}")
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape == (SOURCE_HEIGHT, SOURCE_WIDTH, 3), f"Changing dimensions: {path}")
            decoded += 1
        require(decoded == metadata["frames"],
                f"Truncated source: {path}: decoded {decoded}/{metadata['frames']}")
        metadata["decoded_frames"] = decoded
        metadata["duration_s"] = decoded / FPS
        metadata["protocol_frames"] = expected_frames
        metadata["recorded_initial_frames"] = decoded - expected_frames
        metadata["initial_frame_note"] = (
            "Recorder writes one pre-policy state before its complete policy loop; retained without trimming."
            if decoded > expected_frames else "No separately recorded initial frame.")
        return metadata
    finally:
        cap.release()


def chapter_specs(root):
    chapters = []
    for scenario in ("normal", "push"):
        name = f"retained_locomotion2250_{scenario}"
        chapters.append({"id": name, "kind": "locomotion", "scenario": scenario,
                         "title": f"Retained baseline 2250: stand / walk / stop ({scenario})",
                         "protocol_duration_s": 18, "directories": [root / f"{name}_{view}" for view in VIEWS]})
    for arm in ("nominal", "current"):
        for pose in ("upright", "side", "upside_down"):
            group = "upright" if pose == "upright" else "heldout"
            chapters.append({"id": f"{arm}_{pose}", "kind": "recovery", "arm": arm, "pose": pose,
                             "title": f"New {arm} model 1999: {pose}", "protocol_duration_s": 11,
                             "directories": [root / f"{arm}-{group}-{view}" for view in VIEWS]})
    return chapters


def prepare_chapters(root, output_dir):
    """All reports, full videos and model hashes are verified before output opens."""
    chapters = chapter_specs(root)
    cached_reports, report_hashes, cached_models, arm_hashes = {}, {}, {}, {}
    for chapter in chapters:
        locomotion = chapter["kind"] == "locomotion"
        report_name = f"{LOCOMOTION_STEM}.json" if locomotion else "model_1999.pt_recovery_metrics.json"
        video_name = f"{LOCOMOTION_STEM}.mp4" if locomotion else f"model_1999.pt_{chapter['pose']}.mp4"
        paths = [directory / report_name for directory in chapter["directories"]]
        videos = [directory / video_name for directory in chapter["directories"]]
        for path in paths + videos:
            require(path.is_file(), f"Missing required source (no output created): {path}")
        reports = []
        for path in paths:
            if path not in cached_reports:
                raw = path.read_bytes()
                cached_reports[path] = json.loads(raw.decode("utf-8-sig"))
                report_hashes[path] = hashlib.sha256(raw).hexdigest()
            reports.append(cached_reports[path])
        first, second = reports
        if locomotion:
            validate_locomotion_reports(first, second)
            protocol = first["protocol"]
            require([protocol.get(k) for k in ("stand_s", "walk_s", "stop_s")] == [4, 8, 6],
                    "Expected complete 4/8/6-second locomotion protocol")
            require((protocol["push_delta_vy"] == 0) == (chapter["scenario"] == "normal"),
                    "Push condition disagrees with chapter label")
            chapter["outcome"] = {"passed": first["passed"], "acceptance": first["acceptance"]}
            chapter["display_passed"] = first["passed"]
        else:
            validate_reports(first, second)
            for report in reports:
                validate_recovery_result(report, chapter["pose"])
            require([r.get("video_view") for r in reports] == list(VIEWS),
                    "Recovery inputs must be front then oblique")
            require(first["task"] == ARM_TASKS[chapter["arm"]], "Wrong action-reference task")
            require(first.get("policy_action_mode") == "deterministic_mean", "Diagnostic policy cannot be accepted")
            result = first["results"][chapter["pose"]]
            require(result["trials"] == 1, "Review videos require one trial")
            require(result["horizon_s"] == 8 and result["stable_hold_s"] == 3,
                    "Expected complete 8-second horizon plus 3-second hold")
            for key in ("successes", "final_valid_stands"):
                require(type(result.get(key)) is int and result[key] in (0, 1), f"Invalid {key}")
            chapter["outcome"] = {key: result[key] for key in (
                "trials", "successes", "final_valid_stands", "stable_hold_s", "horizon_s")}
            chapter["display_passed"] = result["successes"] == 1 and result["final_valid_stands"] == 1
            if chapter["pose"] != "upright":
                require(first.get("acceptance_eligible") is True, "Ineligible bank replay")
                require(first["state_bank"]["split"] == "heldout", "Not a heldout replay")
                selected = result["state_bank_selection"]["selected_state_ids"]
                require(len(selected) == 1, "Expected one bank state")
                chapter["bank_state_id"] = selected[0]
                for key in ("settled_fallen_trials", "settled_fallen_recovery_successes",
                            "settled_fallen_final_valid_stands"):
                    chapter["outcome"][key] = result[key]
                require(result["settled_fallen_trials"] == 1, "Bank video did not start settled and fallen")
                chapter["handover"] = {key: first[key] for key in (
                    "settle_controller", "settle_actual_s", "settle_control_steps", "start_protocol_id")}
                chapter["handover"]["included_in_video"] = False
        require(first["seed"] == 20260918, "Unexpected review seed")
        model_group = "retained_locomotion2250" if locomotion else chapter["arm"]
        for report in reports:
            validate_checkpoint_identity(report, model_group)
        checkpoint_paths = [Path(r["checkpoint"]) for r in reports]
        for checkpoint in checkpoint_paths:
            require(checkpoint.is_file(), f"Checkpoint needed to verify hash: {checkpoint}")
            if checkpoint not in cached_models:
                cached_models[checkpoint] = sha256(checkpoint)
            require(cached_models[checkpoint] == first["checkpoint_sha256"], "Checkpoint/report SHA mismatch")
        expected_stem = "natural_stop_diagonal_model2250" if locomotion else "model_1999"
        require(all(p.stem == expected_stem for p in checkpoint_paths), "Unexpected model filename")
        if model_group in arm_hashes:
            require(arm_hashes[model_group] == first["checkpoint_sha256"], "Model changed between scenarios")
        arm_hashes[model_group] = first["checkpoint_sha256"]
        chapter.update(checkpoint_sha256=first["checkpoint_sha256"], task=first["task"], seed=first["seed"],
                       protocol_version=first["protocol_version"],
                       checkpoints=[source_record(p, output_dir) for p in checkpoint_paths])
        chapter["sources"] = []
        for view, path, video in zip(VIEWS, paths, videos):
            report_record = source_record(path, output_dir)
            require(report_record["sha256"] == report_hashes[path], f"Report changed during validation: {path}")
            video_record = source_record(video, output_dir)
            video_record.update(probe_complete_video(video, chapter["protocol_duration_s"] * FPS,
                                                     allow_initial_frame=not locomotion))
            require(sha256(video) == video_record["sha256"], f"Video changed during validation: {video}")
            chapter["sources"].append({"view": view, "report": report_record, "video": video_record})
        require(chapter["sources"][0]["video"]["frames"] == chapter["sources"][1]["video"]["frames"],
                f"Unequal paired source frame counts: {chapter['id']}")
        chapter["frames"] = chapter["sources"][0]["video"]["frames"]
        chapter["duration_s"] = chapter["frames"] / FPS
        chapter["video_paths"] = videos
        del chapter["directories"]
    return chapters


def label_lines(chapter, index):
    outcome = chapter["outcome"]
    if chapter["kind"] == "locomotion":
        status = f"passed={str(outcome['passed']).lower()} | OLD RETAINED BASELINE, not newly trained"
    else:
        status = (f"successes={outcome['successes']}/{outcome['trials']} | "
                  f"final_valid_stands={outcome['final_valid_stands']}/{outcome['trials']}")
    return [f"{index + 1}/8  {chapter['title']}  |  {'PASS' if chapter['display_passed'] else 'FAIL'}",
            f"SHA256 {chapter['checkpoint_sha256'][:12]} | seed={chapter['seed']} | {status}",
            f"bank_state_id={chapter.get('bank_state_id', 'not applicable')} | full source duration, no trimming",
            "LEFT FRONT / RIGHT OBLIQUE: independent matched replays, NOT synchronized cameras",
            "Separate policies/scenarios, NOT continuous closed loop. Simulation only.",
            "Recovery: nominal-PD handover precedes policy time and is NOT in the recovery footage."
            if "handover" in chapter else "Upright/locomotion starts are NOT genuine settled-fall recovery."]


def draw_text_lines(canvas, lines, y, step, color, scale=0.62):
    for line in lines:
        # Keep labels on the added surface, never on original rendered pixels.
        require(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= WIDTH - 40,
                f"Caption too wide; refusing to hide metadata: {line}")
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += step


def compose_frame(front, oblique, bar):
    require(front.shape == oblique.shape == (SOURCE_HEIGHT, SOURCE_WIDTH, 3), "Source frame changed size")
    return np.concatenate((bar, np.concatenate((front, oblique), axis=1)), axis=0)


def encode_review(chapters, output):
    total_frames = sum(CARD_SECONDS * FPS + c["frames"] for c in chapters)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    cursor = 0
    # Child stderr is inherited (not a possibly deadlocking, unread pipe).
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, chapter in enumerate(chapters):
            lines = label_lines(chapter, index)
            color = (100, 225, 100) if chapter["display_passed"] else (110, 150, 255)
            bar = np.zeros((BAR_HEIGHT, WIDTH, 3), dtype=np.uint8)
            draw_text_lines(bar, lines, 22, 25, color)
            card = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            draw_text_lines(card, lines, 150, 55, color, scale=0.72)
            chapter["timeline"] = {"card_start_frame": cursor, "card_start_s": cursor / FPS}
            for _ in range(CARD_SECONDS * FPS):
                process.stdin.write(card.tobytes())
                cursor += 1
            chapter["timeline"].update(footage_start_frame=cursor, footage_start_s=cursor / FPS)
            caps = [cv2.VideoCapture(str(path)) for path in chapter["video_paths"]]
            try:
                for frame_index in range(chapter["frames"]):
                    frames = [cap.read() for cap in caps]
                    require(all(ok for ok, _ in frames), f"Source decode failed at {chapter['id']} frame {frame_index}")
                    process.stdin.write(compose_frame(frames[0][1], frames[1][1], bar).tobytes())
                    cursor += 1
                require(not any(cap.read()[0] for cap in caps), f"Unexpected extra source frames: {chapter['id']}")
            finally:
                for cap in caps:
                    cap.release()
            chapter["timeline"].update(end_frame_exclusive=cursor, end_s=cursor / FPS)
            print(f"Encoded {chapter['id']}: {cursor}/{total_frames} output frames", flush=True)
        process.stdin.close()
        require(process.wait() == 0, "H.264 encoder failed; partial video retained without manifest")
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    require(cursor == total_frames, "Output frame count mismatch")
    cap = cv2.VideoCapture(str(output))
    try:
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        require(actual_frames == cursor and cap.get(cv2.CAP_PROP_FPS) == FPS,
                "Encoded output metadata mismatch")
    finally:
        cap.release()
    return cursor


def build(root, output, validate_only=False):
    output = Path(output).resolve()
    require(output.suffix.lower() == ".mp4", "Output must use .mp4")
    manifest_path = output.with_suffix(".json")
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    chapters = prepare_chapters(Path(root).resolve(), output.parent)
    # Validate captions before opening the encoder as well.
    for index, chapter in enumerate(chapters):
        draw_text_lines(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
                        label_lines(chapter, index), 150, 55, (255, 255, 255), scale=0.72)
    if validate_only:
        print("All eight chapters validated; no outputs created.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = encode_review(chapters, output)
    # Do not publish a trusted manifest if files changed after the preflight.
    for chapter in chapters:
        for source in chapter["sources"]:
            for kind in ("report", "video"):
                record = source[kind]
                require(sha256(output.parent / record["path"]) == record["sha256"],
                        f"Source changed during encoding: {record['path']}")
        del chapter["video_paths"]
    manifest = {"schema_version": "go2_full_scenario_review_v1", "chapters": chapters,
                "note": "Independent matched-condition front/oblique replays, NOT synchronized cameras. "
                        "Different policies shown in separate scenarios, NOT a continuous closed loop. "
                        "All source frames retained at native resolution and frame rate, no trimming. "
                        "Separately recorded recovery initial frames are retained; encoded duration may exceed "
                        "the policy protocol duration by one frame. "
                        "Recovery nominal-PD handover is not in the footage. Failed outcomes are retained. "
                        "A one-trial video is not the multi-trial acceptance result. Simulation only.",
                "output": {**source_record(output, output.parent), "width": WIDTH, "height": HEIGHT,
                           "frames": frames, "fps": FPS, "duration_s": frames / FPS,
                           "codec": "H.264", "pixel_format": "yuv420p", "audio": False,
                           "added_caption_height": BAR_HEIGHT, "chapter_card_seconds": CARD_SECONDS}}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Fully decode/validate all sources; write nothing")
    args = parser.parse_args()
    build(args.root, args.output, args.validate_only)
