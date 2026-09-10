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
parser.add_argument("--min_contacts", type=int, default=4, help="Required foot contacts for a valid final stand.")
parser.add_argument("--seed", type=int, default=20260908)
parser.add_argument("--video_pose", choices=(*POSE_CLASSES, "all"))
parser.add_argument("--poses", nargs="+", choices=POSE_CLASSES, default=POSE_CLASSES)
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument("--video_resolution", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--angle_deg", type=float, default=90.0, help="Roll/pitch angle used for side and fore-aft starts.")
parser.add_argument("--view", choices=("oblique", "front", "side"), default="oblique")
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
    normal_stance_geometry,
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


def _stable_stand(env, foot_ids: list[int], min_contacts: int) -> torch.Tensor:
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    gravity_error = torch.sqrt(upright_error_squared(asset.data.projected_gravity_b))
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    linear_speed = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_w, dim=1)
    forces = sensor.data.net_forces_w_history[:, :, foot_ids].norm(dim=-1).amax(dim=1)
    supported = (forces > 5.0).sum(dim=1) >= min_contacts
    return (
        (gravity_error < 0.35)
        & (height > 0.30)
        & (height < 0.55)
        & (linear_speed < 0.50)
        & (angular_speed < 1.00)
        & supported
    )


def _stance_geometry(env):
    asset = env.scene["robot"]
    feet = asset.find_bodies(("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)[0]
    knees = asset.find_bodies(("FL_calf", "FR_calf", "RL_calf", "RR_calf"), preserve_order=True)[0]
    q = asset.data.root_quat_w[:, None, :].expand(-1, 4, -1)
    foot_b = math_utils.quat_apply_inverse(q, asset.data.body_pos_w[:, feet] - asset.data.root_pos_w[:, None])
    knee_b = math_utils.quat_apply_inverse(q, asset.data.body_pos_w[:, knees] - asset.data.root_pos_w[:, None])
    delta = asset.data.joint_pos - asset.data.default_joint_pos
    valid = normal_stance_geometry(foot_b, knee_b, delta)
    return valid, foot_b, knee_b, delta


def _trace_row(env, foot_ids, pose_class, step, stable_steps):
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    gravity = asset.data.projected_gravity_b[0]
    base_id = sensor.find_bodies("base")[0]
    geometry_ok, foot_b, knee_b, delta = _stance_geometry(env)
    row = {
        "pose": pose_class, "time_s": (step + 1) * env.step_dt, "trial": 0,
        "height": float(asset.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]),
        "roll_deg": float(torch.atan2(gravity[1], -gravity[2]) * 180 / torch.pi),
        "pitch_deg": float(torch.atan2(gravity[0], -gravity[2]) * 180 / torch.pi),
        "gravity_error": float(torch.sqrt(upright_error_squared(gravity))),
        "speed": float(asset.data.root_lin_vel_w[0].norm()),
        "angular_speed": float(asset.data.root_ang_vel_w[0].norm()),
        "feet_contact": int((sensor.data.net_forces_w[0, foot_ids, 2] > 5).sum()),
        "base_contact": bool((sensor.data.net_forces_w[0, base_id].norm(dim=-1) > 1).any()),
        "joint_rms": float((asset.data.joint_pos[0] - asset.data.default_joint_pos[0]).square().mean().sqrt()),
        "stable_hold_s": float(stable_steps[0]) * env.step_dt,
        "geometry_ok": bool(geometry_ok[0]),
    }
    for i, name in enumerate(("FL", "FR", "RL", "RR")):
        for j, axis in enumerate("xyz"):
            row[f"{name}_foot_{axis}_b"] = float(foot_b[0, i, j])
        row[f"{name}_knee_y_b"] = float(knee_b[0, i, 1])
    for i, name in enumerate(asset.joint_names):
        row[name] = float(asset.data.joint_pos[0, i])
        row[name + "_offset"] = float(delta[0, i])
    return row


def _open_video(path: Path, frame: np.ndarray, fps: int):
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _annotate(frame: np.ndarray, pose_class: str, step: int, dt: float, stable_s: float, row: dict) -> np.ndarray:
    image = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
    labels = (
        f"Go2 self-recovery | pose: {pose_class}",
        f"time: {(step + 1) * dt:4.2f} s | valid stand hold: {stable_s:4.2f} s",
        ("LEG GEOMETRY OK" if row["geometry_ok"] else "INVALID LEG GEOMETRY")
        + (f" | valid stand held {args_cli.hold_s:g}s" if stable_s >= args_cli.hold_s else " | stand hold pending"),
        f"{args_cli.checkpoint.parent.name} / {args_cli.checkpoint.name}",
        "Simulation | controlled-drop starts | no hardware validation",
    )
    for i, text in enumerate(labels):
        color = ((80, 220, 80) if row['geometry_ok'] else (60, 80, 255)) if i == 2 else (245, 245, 245)
        font_scale = 0.48 if i >= 3 else 0.70
        cv2.putText(image, text, (20, 38 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1 if i >= 3 else 2, cv2.LINE_AA)
    cv2.putText(image, f"Front feet body y (m): FL {row['FL_foot_y_b']:+.3f} / FR {row['FR_foot_y_b']:+.3f}",
                (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .60, (245, 245, 245), 2, cv2.LINE_AA)
    return image


def _diagnostic_scene(env):
    from isaaclab import sim as sim_utils
    from pxr import UsdShade
    stage = sim_utils.get_current_stage()
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.22, 0.27), roughness=0.9)
    material.func('/World/RecoveryFloorMaterial', material)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/ground')).Bind(
        UsdShade.Material(stage.GetPrimAtPath('/World/RecoveryFloorMaterial')),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    light = sim_utils.DistantLightCfg(intensity=1800.0)
    light.func('/World/RecoverySun', light, orientation=(0.8805, 0.2798, 0.3647, 0.1159))


def _camera(env):
    a = env.scene["robot"].data
    # Front and side are defined by the robot yaw, not by its randomized world yaw.
    heading = math_utils.yaw_quat(a.root_quat_w[:1])
    offset = {"front": (1.25, 0.0, 0.5), "side": (0.0, 1.45, 0.55), "oblique": (1.05, 1.05, 0.60)}[args_cli.view]
    eye = a.root_pos_w[0] + math_utils.quat_apply(heading, a.root_pos_w.new_tensor([offset]))[0]
    target = a.root_pos_w[0].clone()
    target[2] -= 0.08
    env.sim.set_camera_view(eye=eye.cpu().tolist(), target=target.cpu().tolist())


def main() -> None:
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.trials < 1 or args_cli.horizon_s <= 0 or args_cli.hold_s <= 0:
        raise ValueError("Trials, horizon and hold time must be positive.")
    if args_cli.min_contacts < 1 or args_cli.min_contacts > 4:
        raise ValueError("min_contacts must be between 1 and 4.")
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
    if args_cli.video_pose:
        _diagnostic_scene(env.unwrapped)
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
            if args_cli.video_pose in (pose_class, "all"):
                _camera(env.unwrapped)
                for _ in range(4):
                    # Initialize the RGB render product before recording time zero.
                    env.unwrapped.render()
            stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
            recovered_at = torch.full((env.num_envs,), float("nan"), device=env.device)
            legacy_steps = torch.zeros_like(stable_steps)
            legacy_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            video_writer = None
            video_path = args_cli.output_dir / f"{checkpoint_name}_{pose_class}.mp4"
            for step in range(steps):
                # no_grad keeps Isaac Lab's mutable simulator buffers writable across
                # deterministic pose resets; inference_mode would mark them immutable.
                with torch.no_grad():
                    obs, _, dones, _ = env.step(policy(obs))
                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(dones)
                stable = _stable_stand(env.unwrapped, foot_ids, args_cli.min_contacts)
                legacy_steps = torch.where(stable, legacy_steps + 1, 0)
                legacy_success |= legacy_steps >= hold_steps
                geometry_ok, foot_b, knee_b, joint_delta = _stance_geometry(env.unwrapped)
                base_ids = env.unwrapped.scene.sensors["contact_forces"].find_bodies("base")[0]
                base_contact = (env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w[:, base_ids].norm(dim=-1) > 1).any(1)
                sensor_data = env.unwrapped.scene.sensors["contact_forces"].data
                current_support = (sensor_data.net_forces_w[:, foot_ids, 2] > 5).sum(1) >= args_cli.min_contacts
                stable &= geometry_ok & ~base_contact & current_support
                stable_steps = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))
                trace.append(_trace_row(env.unwrapped, foot_ids, pose_class, step, stable_steps))
                if dones.any():
                    raise RuntimeError("Unexpected auto-reset would invalidate trial measurements.")
                newly_recovered = torch.isnan(recovered_at) & (stable_steps >= hold_steps)
                recovered_at[newly_recovered] = (step + 1 - hold_steps) * env.unwrapped.step_dt
                if args_cli.video_pose in (pose_class, "all") and step % max(1, round(1.0 / (args_cli.video_fps * env.unwrapped.step_dt))) == 0:
                    _camera(env.unwrapped)
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
                            trace[-1],
                        )
                    )
            if video_writer is not None:
                video_writer.release()
            success = ~torch.isnan(recovered_at)
            successful_times = recovered_at[success].cpu().numpy()
            results[pose_class] = {
                "trials": int(env.num_envs),
                "successes": int(success.sum().item()),
                "legacy_contact_height_successes": int(legacy_success.sum().item()),
                "final_valid_stands": int((stable_steps >= hold_steps).sum().item()),
                "final_geometry_passes": int(geometry_ok.sum().item()),
                "final_diagnostics": [
                    {
                        "trial": i,
                        "geometry_ok": bool(geometry_ok[i]),
                        "stable_hold_s": float(stable_steps[i]) * env.unwrapped.step_dt,
                        "height_m": float(env.unwrapped.scene["robot"].data.root_pos_w[i, 2]
                                          - env.unwrapped.scene.env_origins[i, 2]),
                        "gravity_error": float(torch.sqrt(upright_error_squared(
                            env.unwrapped.scene["robot"].data.projected_gravity_b[i]))),
                        "vertical_foot_contacts": int((sensor_data.net_forces_w[i, foot_ids, 2] > 5).sum()),
                        "feet_y_b": foot_b[i, :, 1].cpu().tolist(),
                        "knees_y_b": knee_b[i, :, 1].cpu().tolist(),
                        "max_joint_offset_rad": float(joint_delta[i].abs().max()),
                    }
                    for i in range(env.num_envs)
                ],
                "success_rate": float(success.float().mean().item()),
                "median_recovery_s": float(np.median(successful_times)) if len(successful_times) else None,
                "p90_recovery_s": float(np.percentile(successful_times, 90)) if len(successful_times) else None,
                "stable_hold_s": args_cli.hold_s,
                "horizon_s": args_cli.horizon_s,
            }
            summary = {key: value for key, value in results[pose_class].items() if key != "final_diagnostics"}
            print(f"{pose_class}: {summary}")
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
        "protocol_version": "stance_geometry_v1",
        "video_view": args_cli.view if args_cli.video_pose else None,
        "self_collisions_enabled": env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
        "criterion": f"gravity error < 0.35, height 0.30-0.55 m, speed < 0.50 m/s, angular speed < 1.00 rad/s, {args_cli.min_contacts} simultaneous foot vertical forces > 5 N, no base contact, feet on correct body sides (0.06 < signed lateral < 0.30 m), knees on correct sides (>0.04 m), fore/hind feet on correct ends (>0.08 m), each joint offset <0.65 rad; all continuously held for hold_s",
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
