"""Copy a rollout's raw measurements and report with portable artifact links."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def archive(report_path, destination, label, checkpoint=None, video=None):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    source_csv = Path(report['artifacts']['csv'])
    if not source_csv.is_absolute():
        source_csv = report_path.parent / source_csv
    target_csv = destination / f'{label}.csv'
    shutil.copy2(source_csv, target_csv)
    report['source_checkpoint'] = report['checkpoint']
    report['checkpoint'] = os.path.relpath(checkpoint, destination).replace('\\', '/') if checkpoint else None
    report['artifacts'] = {
        'csv': target_csv.name,
        'video': os.path.relpath(video, destination).replace('\\', '/') if video else None,
    }
    with target_csv.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    stop_start = report['protocol']['stand_s'] + report['protocol']['walk_s']
    dt = float(rows[1]['time_s']) - float(rows[0]['time_s'])
    hold_steps = round(1.0 / dt)
    stop_rows = [r for r in rows if r['phase'] == 'stop']
    quiet = [float(r['speed']) < 0.06 and float(r['height']) >= 0.25
             and max(abs(float(r['roll_deg'])), abs(float(r['pitch_deg']))) < 10
             and all(int(r[f'{foot}_foot_contact']) for foot in ('FL', 'FR', 'RL', 'RR'))
             for r in stop_rows]
    first_hold = next((i for i in range(len(quiet) - hold_steps + 1) if all(quiet[i:i + hold_steps])), None)
    report['stop_settling'] = {
        'criterion': 'First 1 s hold: speed <0.06 m/s, height >=0.25 m, tilt <10 deg, four contacts',
        'hold_begin_s_after_stop': None if first_hold is None else float(stop_rows[first_hold]['time_s']) - stop_start,
    }
    output = destination / f'{label}.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    s = report['settled_phase_stats']
    print(json.dumps({'label': label, 'seed': report['seed'], 'passed': report['passed'],
                      'failed_checks': [k for k, v in report['acceptance'].items() if not v],
                      'height_stand_walk_stop': [s[p]['height_mean'] for p in ('stand', 'walk', 'stop')],
                      'walk_vx': s['walk'].get('vx_b_mean', s['walk']['vx_mean']),
                      'diagonal_support': s['walk'].get('diagonal_support_fraction'),
                      'stop_speed_p95': s['stop']['speed_p95'], 'stop_settling': report['stop_settling']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    archive(args.report, args.destination, args.label, args.checkpoint, args.video)
