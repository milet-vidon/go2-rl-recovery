"""Pair complete front/oblique recovery replays and encode browser-compatible H.264."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def validate_reports(left, right):
    """Match independent replay conditions/outcomes, not floating-point trajectories."""
    missing = object()

    def same_field(a, b, key, path, required=False):
        x, y = a.get(key, missing), b.get(key, missing)
        if (required and x is missing) or x != y:
            raise ValueError(f"Mismatched replay metadata: {path}.{key}; independent replays are not synchronized cameras")

    for key in ('checkpoint_sha256', 'task', 'seed', 'angle_deg', 'protocol_version', 'criterion'):
        same_field(left, right, key, 'report', required=True)
    bank_mode = 'state_bank' in left or 'state_bank' in right
    for key in ('start_protocol_id', 'settle_requested_s', 'settle_actual_s', 'settle_control_steps',
                'settle_controller', 'policy_action_mode', 'acceptance_eligible', 'action_representation',
                'self_collisions_enabled'):
        same_field(left, right, key, 'report', required=bank_mode and key in (
            'settle_actual_s', 'settle_control_steps', 'settle_controller', 'policy_action_mode', 'acceptance_eligible'))
    if bank_mode:
        if not isinstance(left.get('state_bank'), dict) or not isinstance(right.get('state_bank'), dict):
            raise ValueError('Both bank replays require state_bank provenance')
        for key in ('sha256', 'schema_version', 'split', 'selected_state_ids'):
            same_field(left['state_bank'], right['state_bank'], key, 'state_bank', required=True)
        for key in ('fixed_material', 'physics', 'selected_unique_state_count', 'diagnostic_train_split_only'):
            same_field(left['state_bank'], right['state_bank'], key, 'state_bank')
    if set(left['results']) != set(right['results']):
        raise ValueError('Mismatched replay pose sets')
    start_keys = ('trial', 'settled', 'standing_at_policy_start', 'fallen_at_policy_start',
                  'eligible_settled_fallen_recovery', 'classification', 'bank_state_id',
                  'bank_saved_pose_class', 'bank_requested_pose_class', 'actual_pose_class')
    if bank_mode:
        for report in (left, right):
            flat_ids = [state_id for result in report['results'].values()
                        for state_id in result.get('state_bank_selection', {}).get('selected_state_ids', [])]
            if flat_ids != report['state_bank']['selected_state_ids']:
                raise ValueError('Global bank source order disagrees with per-pose selections')
    for pose, first in left['results'].items():
        second = right['results'][pose]
        path = f'results.{pose}'
        same_field(first, second, 'trials', path, required=True)
        if type(first['trials']) is not int or first['trials'] < 1:
            raise ValueError('Trials must be positive integers')
        for key in ('successes', 'final_valid_stands', 'legacy_contact_height_successes', 'final_geometry_passes',
                    'standing_starts_not_fallen_recovery', 'settled_fallen_trials',
                    'settled_fallen_recovery_successes', 'settled_fallen_final_valid_stands',
                    'stable_hold_s', 'horizon_s', 'state_bank_selection'):
            same_field(first, second, key, path, required=bank_mode and key in (
                'successes', 'final_valid_stands', 'settled_fallen_trials', 'settled_fallen_recovery_successes',
                'settled_fallen_final_valid_stands', 'stable_hold_s', 'horizon_s', 'state_bank_selection'))
        for collection in ('policy_start_state', 'final_diagnostics'):
            a, b = first.get(collection, missing), second.get(collection, missing)
            if a is missing or b is missing:
                if a is not b or bank_mode:
                    raise ValueError(f'Missing {path}.{collection}')
                continue
            if not isinstance(a, list) or not isinstance(b, list) or len(a) != first['trials'] or len(b) != first['trials']:
                raise ValueError(f'Invalid {path}.{collection} trial records')
            for i, (x, y) in enumerate(zip(a, b)):
                if collection == 'policy_start_state':
                    for key in start_keys:
                        same_field(x, y, key, f'{path}.{collection}[{i}]', required=bank_mode)
                else:
                    for key in ('trial', 'geometry_ok', 'base_contact', 'vertical_foot_contacts'):
                        same_field(x, y, key, f'{path}.{collection}[{i}]')
                    same_field({'has_hold': 'stable_hold_s' in x}, {'has_hold': 'stable_hold_s' in y}, 'has_hold', path)
                    if 'stable_hold_s' in x:
                        if (x['stable_hold_s'] >= first['stable_hold_s']) != (y['stable_hold_s'] >= second['stable_hold_s']):
                            raise ValueError(f'Different final held-stand outcome in {path}, trial {i}')
        if bank_mode:
            selection = first['state_bank_selection']
            ids = selection['selected_state_ids']
            if len(ids) != first['trials'] or len(set(ids)) != selection['selected_unique_state_count']:
                raise ValueError(f'Invalid source selection in {path}')
            for result in (first, second):
                if [r['bank_state_id'] for r in result['policy_start_state']] != ids:
                    raise ValueError(f'Start records disagree with selected source order in {path}')
                if [r['trial'] for r in result['policy_start_state']] != list(range(first['trials'])):
                    raise ValueError(f'Invalid trial order in {path}')


def pair(front_report, oblique_report, output):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (front_report, oblique_report)]
    validate_reports(*reports)
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
        if target.exists():
            raise FileExistsError(f"Preserve the existing video; use a fresh output: {target}")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(inputs[0]), "-i", str(inputs[1]),
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(target)], check=True, capture_output=True)
        artifacts[pose] = {"video": target.name, "frames": properties[0][0], "fps": properties[0][1]}
        print(target)
    metadata = {
        "note": "Left: front; right: oblique. Independent matched-condition replays, NOT synchronized cameras; full duration without time cropping. Conditions, source order, start eligibility and discrete outcomes checked; physical trajectories need not be bitwise identical.",
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
