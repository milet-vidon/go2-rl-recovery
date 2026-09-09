"""Evaluate a Go2 recovery checkpoint from deterministic pose classes and optionally record video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


POSE_CLASSES = ("upright", "side", "fore_aft", "upside_down", "random")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Recovery-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--output_dir", required=True, type=Path)
parser.add_argument("--trials", type=int, default=100, help="Independent trials for each pose class.")
parser.add_argument("--horizon_s", type=float, default=8.0)
parser.add_argument("--hold_s", type=float, default=3.0, help="Continuous stable stand duration required for success.")
parser.add_argument("--seed", type=int, default=20260908)
parser.add_argument("--video_pose", choices=(*POSE_CLASSES, "all"))
parser.add_argument("--poses", nargs="+", choices=POSE_CLASSES, default=POSE_CLASSES)
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument("--video_resolution", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--angle_deg", type=float, default=90.0, help="Roll/pitch angle used for side and fore-aft starts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_pose:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_math import (  # noqa: E402
    reset_clearance_height,
    upright_error_squared,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def _uniform(shape: tuple[int, ...], low: float, high: float, *, device: str, generator: torch.Generator) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, device=device, generator=generator)


def _pose_quaternions(pose_class: str, count: int, device: str, generator: torch.Generator, angle_deg: float = 90.0) -> torch.Tensor:
    roll = torch.zeros(count, device=device)
    pitch = torch.zeros(count, device=device)
    yaw = _uniform((count,), -torch.pi, torch.pi, device=device, generator=generator)
    noise = lambda: _uniform((count,), -0.20, 0.20, device=device, generator=generator)
    signs = torch.where(
        torch.rand((count,), device=device, generator=generator) < 0.5,
        torch.tensor(-1.0, device=device),
        torch.tensor(1.0, device=device),
    )
    if pose_class == "upright":
        roll = _uniform((count,), -0.25, 0.25, device=device, generator=generator)
        pitch = _uniform((count,), -0.25, 0.25, device=device, generator=generator)
    elif pose_class == "side":
        roll = signs * torch.tensor(angle_deg * torch.pi / 180.0, device=device) + noise()
        pitch = noise()
    elif pose_class == "fore_aft":
        roll = noise()
        pitch = signs * torch.tensor(angle_deg * torch.pi / 180.0, device=device) + noise()
    elif pose_class == "upside_down":
        roll = signs * torch.pi + noise()
        pitch = noise()
    elif pose_class == "random":
        quaternion = torch.randn((count, 4), device=device, generator=generator)
        return torch.nn.functional.normalize(quaternion, dim=-1)
    else:
        raise ValueError(f"Unsupported pose class: {pose_class}")
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def _set_pose_class(env, pose_class: str, generator: torch.Generator) -> None:
    asset = env.scene["robot"]
    count = env.num_envs
    device = env.device
    env_ids = torch.arange(count, device=device)
    orientations = _pose_quaternions(pose_class, count, device, generator, args_cli.angle_deg)
    if pose_class == "upright":
        height = _uniform((count,), 0.38, 0.42, device=device, generator=generator)
    else:
        height = reset_clearance_height(
            math_utils.matrix_from_quat(orientations),
            _uniform((count,), 0.015, 0.035, device=device, generator=generator),
        )
    position = torch.zeros((count, 3), device=device)
    position[:, 2] = height
    position += env.scene.env_origins
    asset.write_root_pose_to_sim(torch.cat((position, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((count, 6), device=device), env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    env.episode_length_buf.zero_()


def _stable_stand(env, foot_ids: list[int]) -> torch.Tensor:
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    gravity_error = torch.sqrt(upright_error_squared(asset.data.projected_gravity_b))
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    linear_speed = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_w, dim=1)
    forces = sensor.data.net_forces_w_history[:, :, foot_ids].norm(dim=-1).amax(dim=1)
    supported = (forces > 5.0).sum(dim=1) >= 2
    return (
        (gravity_error < 0.35)
        & (height > 0.30)
        & (height < 0.55)
        & (linear_speed < 0.50)
        & (angular_speed < 1.00)
        & supported
    )


def _trace_row(env, foot_ids, pose_class, step, stable_steps):
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    gravity = asset.data.projected_gravity_b[0]
    base_id = sensor.find_bodies("base")[0]
    return {
        "pose": pose_class, "time_s": (step + 1) * env.step_dt, "trial": 0,
        "height": float(asset.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]),
        "roll_deg": float(torch.atan2(gravity[1], -gravity[2]) * 180 / torch.pi),
        "pitch_deg": float(torch.atan2(gravity[0], -gravity[2]) * 180 / torch.pi),
        "gravity_error": float(torch.sqrt(upright_error_squared(gravity))),
        "speed": float(asset.data.root_lin_vel_w[0].norm()),
        "angular_speed": float(asset.data.root_ang_vel_w[0].norm()),
        "feet_contact": int((sensor.data.net_forces_w[0, foot_ids].norm(dim=-1) > 5).sum()),
        "base_contact": bool((sensor.data.net_forces_w[0, base_id].norm(dim=-1) > 1).any()),
        "joint_rms": float((asset.data.joint_pos[0] - asset.data.default_joint_pos[0]).square().mean().sqrt()),
        "stable_hold_s": float(stable_steps[0]) * env.step_dt,
    }


def _open_video(path: Path, frame: np.ndarray, fps: int):
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _annotate(frame: np.ndarray, pose_class: str, step: int, dt: float, stable_s: float, success: bool) -> np.ndarray:
    image = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
    labels = (
        f"Go2 self-recovery | pose: {pose_class}",
        f"time: {step * dt:4.2f} s | stable hold: {stable_s:4.2f} s",
        "SUCCESS: held stable stand" if success else "recovering",
        f"{args_cli.checkpoint.parent.name} / {args_cli.checkpoint.name}",
        "Simulation | controlled-drop starts | no hardware validation",
    )
    for row, text in enumerate(labels):
        color = (80, 220, 80) if row == 2 and success else (245, 245, 245)
        font_scale = 0.48 if row >= 3 else 0.70
        cv2.putText(image, text, (20, 38 + row * 32), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1 if row >= 3 else 2, cv2.LINE_AA)
    return image


def main() -> None:
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.trials < 1 or args_cli.horizon_s <= 0 or args_cli.hold_s <= 0:
        raise ValueError("Trials, horizon and hold time must be positive.")
    if args_cli.video_fps not in (10, 25, 50):
        raise ValueError("Use 10, 25 or 50 fps so video timing exactly divides the 50 Hz control loop.")
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    eval_task = args_cli.task
    env_cfg = load_cfg_from_registry(eval_task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(eval_task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.trials
    env_cfg.scene.env_spacing = 2.0
    env_cfg.episode_length_s = args_cli.horizon_s + args_cli.hold_s + 1.0
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (2.3, 2.3, 1.35)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.30)
    env_cfg.viewer.resolution = tuple(args_cli.video_resolution)
    env = gym.make(eval_task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_pose else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(str(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    policy_nn = runner.alg.policy
    foot_ids = env.unwrapped.scene.sensors["contact_forces"].find_bodies(
        ("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True
    )[0]
    if len(foot_ids) != 4:
        raise RuntimeError(f"Expected four Go2 feet, found ids={foot_ids}")

    generator = torch.Generator(device=env.device).manual_seed(args_cli.seed)
    steps = round((args_cli.horizon_s + args_cli.hold_s) / env.unwrapped.step_dt)
    hold_steps = round(args_cli.hold_s / env.unwrapped.step_dt)
    results: dict[str, dict] = {}
    trace = []
    checkpoint_name = args_cli.checkpoint.name

    try:
        for pose_class in args_cli.poses:
            env.reset()
            _set_pose_class(env.unwrapped, pose_class, generator)
            obs = env.get_observations()
            stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
            recovered_at = torch.full((env.num_envs,), float("nan"), device=env.device)
            video_writer = None
            video_path = args_cli.output_dir / f"{checkpoint_name}_{pose_class}.mp4"
            for step in range(steps):
                # no_grad keeps Isaac Lab's mutable simulator buffers writable across
                # deterministic pose resets; inference_mode would mark them immutable.
                with torch.no_grad():
                    obs, _, dones, _ = env.step(policy(obs))
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(dones)
                stable = _stable_stand(env.unwrapped, foot_ids)
                stable_steps = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))
                trace.append(_trace_row(env.unwrapped, foot_ids, pose_class, step, stable_steps))
                if dones.any():
                    raise RuntimeError("Unexpected auto-reset would invalidate trial measurements.")
                newly_recovered = torch.isnan(recovered_at) & (stable_steps >= hold_steps)
                recovered_at[newly_recovered] = (step + 1 - hold_steps) * env.unwrapped.step_dt
                if args_cli.video_pose in (pose_class, "all") and step % max(1, round(1.0 / (args_cli.video_fps * env.unwrapped.step_dt))) == 0:
                    frame = env.unwrapped.render()
                    if frame is None:
                        raise RuntimeError("No RGB frame returned. Run with --enable_cameras.")
                    if video_writer is None:
                        video_writer = _open_video(video_path, frame, args_cli.video_fps)
                    video_writer.write(
                        _annotate(
                            frame,
                            pose_class,
                            step,
                            env.unwrapped.step_dt,
                            stable_steps[0].item() * env.unwrapped.step_dt,
                            not torch.isnan(recovered_at[0]).item(),
                        )
                    )
            if video_writer is not None:
                video_writer.release()
            success = ~torch.isnan(recovered_at)
            successful_times = recovered_at[success].cpu().numpy()
            results[pose_class] = {
                "trials": int(env.num_envs),
                "successes": int(success.sum().item()),
                "success_rate": float(success.float().mean().item()),
                "median_recovery_s": float(np.median(successful_times)) if len(successful_times) else None,
                "p90_recovery_s": float(np.percentile(successful_times, 90)) if len(successful_times) else None,
                "stable_hold_s": args_cli.hold_s,
                "horizon_s": args_cli.horizon_s,
            }
            print(f"{pose_class}: {results[pose_class]}")
    finally:
        env.close()

    report = {
        "angle_deg": args_cli.angle_deg,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args_cli.checkpoint.read_bytes()).hexdigest(),
        "task": eval_task,
        "seed": args_cli.seed,
        "start_protocol": "controlled drop from conservative default-pose envelope, not pre-settled fallen poses",
        "time_definition": "onset of the first stable interval held for hold_s; simulation includes hold_s after horizon_s",
        "criterion": "gravity error < 0.35, root height 0.30-0.55 m, linear speed < 0.50 m/s, angular speed < 1.00 rad/s, at least two foot contacts > 5 N, continuously held for hold_s",
        "results": results,
        "diagnostic_trial": 0,
    }
    trace_path = args_cli.output_dir / f"{checkpoint_name}_recovery_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    report["trace_csv"] = trace_path.name
    report_path = args_cli.output_dir / f"{checkpoint_name}_recovery_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        sys.stdout.flush()
        simulation_app.close()
