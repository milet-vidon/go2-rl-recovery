"""Encode full rollouts and plot recorded stand/walk/stop measurements."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build(report_path, destination, label):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    def artifact_path(value):
        path = Path(value)
        return path if path.is_absolute() else report_path.parent / path
    with artifact_path(report['artifacts']['csv']).open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    destination.mkdir(parents=True, exist_ok=True)
    t = np.array([float(r['time_s']) for r in rows])
    values = lambda key: np.array([float(r[key]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    axes[0].plot(t, values('height'), label='Base link height')
    axes[0].axhline(0.25, color='firebrick', linestyle='--', label='Minimum screening height')
    axes[0].set_ylabel('Height (m)')
    axes[1].plot(t, values('vx_b') if 'vx_b' in rows[0] else values('vx'), label='Measured forward velocity')
    axes[1].plot(t, values('cmd_x'), '--', label='Command')
    axes[1].set_ylabel('Velocity (m/s)')
    for name in ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'):
        axes[2].plot(t, values(f'{name}_z_w'), label=name.replace('_foot', ''))
    axes[2].set_ylabel('Foot center z (m)')
    axes[2].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.axvline(report['protocol']['stand_s'], color='gray', linewidth=0.7)
        ax.axvline(report['protocol']['stand_s'] + report['protocol']['walk_s'], color='gray', linewidth=0.7)
        for r in rows:
            if float(r.get('push_delta_vy', 0)):
                ax.axvline(float(r['time_s']), color='darkorange', linewidth=0.6)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper right', ncols=2)
    fig.suptitle(label + ' | stand 0-4s, walk 4-12s, stop 12-18s')
    fig.savefig(destination / f'{label}_metrics.png', dpi=150)
    plt.close(fig)
    video = report['artifacts'].get('video')
    if video:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(artifact_path(video)), '-an', '-c:v', 'libx264',
                        '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(destination / f'{label}.mp4')], check=True, capture_output=True)
    print(json.dumps({'label': label, 'metrics': str(destination / f'{label}_metrics.png'), 'video': str(destination / f'{label}.mp4') if video else None}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    build(args.report, args.destination, args.label)
