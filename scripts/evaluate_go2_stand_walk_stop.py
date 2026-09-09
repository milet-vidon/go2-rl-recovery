"""Evaluate a Go2 policy with a fixed stand -> walk -> stop command sequence.

This diagnostic intentionally removes command resampling, heading control, pushes, and
fall termination.  A policy is therefore judged on the same three phases every run,
and a fall remains visible in the video instead of being hidden by an automatic reset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--checkpoint", type=Path, help="RSL-RL checkpoint; omit with --zero_action")
parser.add_argument("--zero_action", action="store_true", help="Use the actuator default pose with zero policy actions")
parser.add_argument("--push_speed", type=float, default=0.0, help="Apply signed lateral delta-v (m/s) midway through each phase")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--output_dir", required=True, type=Path)
parser.add_argument("--stand_s", type=float, default=4.0)
parser.add_argument("--walk_s", type=float, default=8.0)
parser.add_argument("--stop_s", type=float, default=6.0)
parser.add_argument("--walk_speed", type=float, default=0.5)
parser.add_argument("--lateral_speed", type=float, default=0.0)
parser.add_argument("--yaw_rate", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=20260909)
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--real_time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_fps not in (10, 25, 50):
    raise ValueError("--video_fps must divide the 50 Hz control loop: use 10, 25 or 50")
if min(args_cli.stand_s, args_cli.walk_s, args_cli.stop_s) <= 0:
    raise ValueError("All phase durations must be positive")
args_cli.enable_cameras = not args_cli.no_video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
import isaaclab.sim as sim_utils


def _configure_diagnostic(env_cfg, total_s: float) -> None:
    """Make command and reset behavior deterministic for a single diagnostic rollout."""
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.0
    env_cfg.episode_length_s = total_s + 2.0
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (1.0, 1.7, 0.5)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.04)
    env_cfg.viewer.resolution = (args_cli.width, args_cli.height)

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    command_cfg.debug_vis = False
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.ranges.heading = None

    # Keep the initial pose and model randomization, but eliminate interval pushes and
    # base-contact auto-resets so the exact behavior is visible and measurable.
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.terminations.base_contact = None

    # Start above the plane at the task's default root height. Use the standard
    # reset functions explicitly, so recovery tasks cannot interpret 1.0 as an offset.
    env_cfg.events.reset_base.func = mdp.reset_root_state_uniform
    env_cfg.events.reset_base.params = {}
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.func = mdp.reset_joints_by_scale
    env_cfg.events.reset_robot_joints.params = {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}


def _set_command(term, value: tuple[float, float, float]) -> None:
    term.vel_command_b.fill_(0.0)
    term.vel_command_b[0, :3] = torch.as_tensor(value, device=term.device)
    term.is_standing_env.fill_(False)
    term.is_heading_env.fill_(False)


def _phase(t: float) -> tuple[str, tuple[float, float, float]]:
    if t < args_cli.stand_s:
        return "stand", (0.0, 0.0, 0.0)
    if t < args_cli.stand_s + args_cli.walk_s:
        return "walk", (args_cli.walk_speed, args_cli.lateral_speed, args_cli.yaw_rate)
    return "stop", (0.0, 0.0, 0.0)


def _open_video(path: Path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps, (args_cli.width, args_cli.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _diagnostic_floor():
    """A visible grid and shadows; no colliders or physical parameters changed."""
    from pxr import UsdShade
    stage = sim_utils.get_current_stage()
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.23, 0.27), roughness=0.9)
    material.func('/World/DiagnosticFloorMaterial', material)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/ground')).Bind(
        UsdShade.Material(stage.GetPrimAtPath('/World/DiagnosticFloorMaterial')),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    line_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.50, 0.56), roughness=0.9)
    for index in range(41):
        coordinate = (index - 20) * 0.5
        for axis in (0, 1):
            size = (0.006, 20.0, 0.002) if axis == 0 else (20.0, 0.006, 0.002)
            position = (coordinate, 0, 0.002) if axis == 0 else (0, coordinate, 0.002)
            line = sim_utils.CuboidCfg(size=size, visual_material=line_material)
            line.func(f'/World/DiagnosticGrid_{axis}_{index}', line, translation=position)
    light = sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.97, 0.92))
    light.func('/World/DiagnosticSun', light, orientation=(0.8805, 0.2798, 0.3647, 0.1159))


def _annotate(frame: np.ndarray, phase: str, t: float, row: dict[str, float]) -> np.ndarray:
    image = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
    lines = (
        f"Go2 stand-walk-stop | {phase}",
        f"t={t:5.2f}s cmd=({row['cmd_x']:+.2f}, {row['cmd_y']:+.2f}, {row['cmd_yaw']:+.2f})",
        f"height={row['height']:.3f}m roll={row['roll_deg']:+.1f} pitch={row['pitch_deg']:+.1f}",
        f"speed={row['speed']:.2f}m/s contacts={int(row['foot_contacts'])} base_contact={int(row['base_contact'])}",
        f"lateral delta-v={args_cli.push_speed:.2f} m/s | automatic fall reset OFF",
    )
    for i, line in enumerate(lines):
        color = (80, 220, 80) if i == 0 and phase == "stop" else (245, 245, 245)
        cv2.putText(image, line, (18, 34 + 27 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return image


def main() -> None:
    if args_cli.zero_action and args_cli.checkpoint is not None:
        raise ValueError("--zero_action and --checkpoint are mutually exclusive")
    if not args_cli.zero_action and (args_cli.checkpoint is None or not args_cli.checkpoint.is_file()):
        raise FileNotFoundError(args_cli.checkpoint)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    total_s = args_cli.stand_s + args_cli.walk_s + args_cli.stop_s

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    _configure_diagnostic(env_cfg, total_s)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None if args_cli.no_video else "rgb_array")
    if not args_cli.no_video:
        _diagnostic_floor()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.zero_action:
        policy = lambda obs: torch.zeros((1, env.unwrapped.action_manager.total_action_dim), device=env.device)
        policy_nn = None
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
        runner.load(str(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        policy_nn = runner.alg.policy
    term = env.unwrapped.command_manager.get_term("base_velocity")
    asset = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = contact_sensor.find_bodies(("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)[0]
    base_ids = contact_sensor.find_bodies(("base",), preserve_order=True)[0]
    foot_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    foot_body_ids = asset.find_bodies(foot_names, preserve_order=True)[0]
    if len(foot_ids) != 4 or len(base_ids) != 1:
        raise RuntimeError(f"Unexpected sensor bodies: feet={foot_ids}, base={base_ids}")

    env.reset()
    _set_command(term, (0.0, 0.0, 0.0))
    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    steps = round(total_s / dt)
    sample_every = max(1, round(1.0 / (args_cli.video_fps * dt)))
    result_stem = "zero_action" if args_cli.zero_action else args_cli.checkpoint.stem
    video_path = args_cli.output_dir / f"{result_stem}_stand_walk_stop.mp4"
    csv_path = args_cli.output_dir / f"{result_stem}_stand_walk_stop.csv"
    json_path = args_cli.output_dir / f"{result_stem}_stand_walk_stop.json"
    writer = None if args_cli.no_video else _open_video(video_path)
    fields = [
        "step", "time_s", "phase", "cmd_x", "cmd_y", "cmd_yaw", "height", "x", "y",
        "speed", "yaw_speed", "roll_deg", "pitch_deg", "foot_contacts", "base_contact", "done",
        "vx", "vy", "vx_b", "vy_b", "wz", "push_delta_vy", "joint_rms_deviation",
    ]
    fields += [f"{name}_{quantity}" for name in foot_names for quantity in ("x_b", "y_b", "z_w", "fz", "contact", "speed_xy")]
    push_steps = {round(t / dt): sign * args_cli.push_speed for t, sign in (
        (args_cli.stand_s / 2, 1), (args_cli.stand_s + args_cli.walk_s / 2, -1),
        (args_cli.stand_s + args_cli.walk_s + args_cli.stop_s / 2, 1))}
    geometry = {"body_names": asset.body_names, "joint_names": asset.joint_names,
                "default_joint_positions": asset.data.default_joint_pos[0].tolist(),
                "default_root_position": asset.data.default_root_state[0, :3].tolist(),
                "ground_asset": GroundPlaneCfg().usd_path, "robot_asset": env_cfg.scene.robot.spawn.usd_path,
                "ground_env_override": os.getenv("ISAACLAB_GROUND_USD")}
    rows: list[dict[str, object]] = []
    max_roll = max_pitch = max_tilt = 0.0
    min_height = float("inf")
    phase_stats: dict[str, dict[str, float]] = {}
    start_wall = time.time()
    try:
        for step in range(steps):
            t = step * dt
            phase, command = _phase(t)
            _set_command(term, command)
            push_delta = push_steps.get(step, 0.0)
            if push_delta:
                velocity = asset.data.root_vel_w.clone()
                velocity[:, 1] += push_delta
                asset.write_root_velocity_to_sim(velocity)
                env.unwrapped.scene.update(dt)
            # The command term is part of the policy observation. Recompute it after
            # each phase change so the next action sees exactly the requested command.
            obs = env.get_observations()
            with torch.no_grad():
                obs, _, dones, _ = env.step(policy(obs))
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

            root_pos = asset.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]
            gravity = asset.data.projected_gravity_b[0]
            # For a level quadruped, gravity x/y give a stable tilt measure without
            # relying on Euler-angle conventions near pi.
            roll = float(torch.atan2(gravity[1], -gravity[2]).item() * 180.0 / np.pi)
            pitch = float(torch.atan2(gravity[0], -gravity[2]).item() * 180.0 / np.pi)
            tilt = float(torch.linalg.norm(gravity[:2]).item())
            lin_vel = asset.data.root_lin_vel_w[0]
            ang_vel = asset.data.root_ang_vel_w[0]
            foot_forces = contact_sensor.data.net_forces_w_history[0, :, foot_ids].norm(dim=-1).amax(dim=0)
            base_force = contact_sensor.data.net_forces_w_history[0, :, base_ids].norm(dim=-1).amax().item()
            row = {
                "step": step + 1, "time_s": (step + 1) * dt, "phase": phase,
                "cmd_x": float(term.command[0, 0].item()), "cmd_y": float(term.command[0, 1].item()),
                "cmd_yaw": float(term.command[0, 2].item()), "height": float(root_pos[2].item()),
                "x": float(root_pos[0].item()), "y": float(root_pos[1].item()),
                "speed": float(torch.linalg.norm(lin_vel[:2]).item()),
                "yaw_speed": float(abs(ang_vel[2].item())), "roll_deg": roll, "pitch_deg": pitch,
                "foot_contacts": int((foot_forces > 5.0).sum().item()), "base_contact": int(base_force > 1.0),
                "done": int(dones[0].item()),
                "vx": float(lin_vel[0]), "vy": float(lin_vel[1]), "push_delta_vy": push_delta,
                "wz": float(ang_vel[2]),
                "vx_b": float(asset.data.root_lin_vel_b[0, 0]), "vy_b": float(asset.data.root_lin_vel_b[0, 1]),
                "joint_rms_deviation": float((asset.data.joint_pos[0] - asset.data.default_joint_pos[0]).square().mean().sqrt()),
            }
            feet_world = asset.data.body_pos_w[0, foot_body_ids]
            feet_body = quat_apply_inverse(asset.data.root_quat_w[0].expand(4, -1), feet_world - asset.data.root_pos_w[0])
            for i, name in enumerate(foot_names):
                row.update({f"{name}_x_b": float(feet_body[i, 0]), f"{name}_y_b": float(feet_body[i, 1]),
                            f"{name}_z_w": float(feet_world[i, 2]),
                            f"{name}_fz": float(contact_sensor.data.net_forces_w[0, foot_ids[i], 2]),
                            f"{name}_speed_xy": float(asset.data.body_lin_vel_w[0, foot_body_ids[i], :2].norm()),
                            f"{name}_contact": int(contact_sensor.data.net_forces_w[0, foot_ids[i]].norm() > 5.0)})
            rows.append(row)
            max_roll = max(max_roll, abs(roll)); max_pitch = max(max_pitch, abs(pitch)); max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, row["height"])
            stats = phase_stats.setdefault(phase, {"count": 0.0, "height_sum": 0.0, "contacts_sum": 0.0, "speed_sum": 0.0})
            stats["count"] += 1.0; stats["height_sum"] += row["height"]; stats["contacts_sum"] += row["foot_contacts"]; stats["speed_sum"] += row["speed"]
            if writer is not None and step % sample_every == 0:
                frame = env.unwrapped.render()
                if frame is None:
                    raise RuntimeError("No RGB frame returned")
                writer.write(_annotate(frame, phase, t, row))
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.time() - start_wall - step * dt)))
    finally:
        if writer is not None:
            writer.release()
        env.close()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        output = csv.DictWriter(handle, fieldnames=fields)
        output.writeheader(); output.writerows(rows)
    for stats in phase_stats.values():
        count = max(stats["count"], 1.0)
        stats["mean_height"] = stats.pop("height_sum") / count
        stats["mean_contacts"] = stats.pop("contacts_sum") / count
        stats["mean_speed"] = stats.pop("speed_sum") / count
    settled = {}
    for name, begin in (("stand", 0), ("walk", args_cli.stand_s), ("stop", args_cli.stand_s + args_cli.walk_s)):
        samples = [r for r in rows if r['phase'] == name and r['time_s'] > begin + 1.0]
        settled[name] = {"height_mean": float(np.mean([r['height'] for r in samples])),
                         "height_min": min(r['height'] for r in samples),
                         "vx_mean": float(np.mean([r['vx'] for r in samples])),
                         "vx_b_mean": float(np.mean([r['vx_b'] for r in samples])),
                         "vy_b_mean": float(np.mean([r['vy_b'] for r in samples])),
                         "yaw_rate_mean": float(np.mean([r['wz'] for r in samples])),
                         "yaw_speed_p95": float(np.percentile([r['yaw_speed'] for r in samples], 95)),
                         "speed_p95": float(np.percentile([r['speed'] for r in samples], 95)),
                         "max_tilt_deg": max(max(abs(r['roll_deg']), abs(r['pitch_deg'])) for r in samples),
                         "four_feet_contact_fraction": float(np.mean([all(r[f'{n}_contact'] for n in foot_names) for r in samples])),
                         "foot_height_p95": {n: float(np.percentile([r[f'{n}_z_w'] for r in samples], 95)) for n in foot_names},
                         "diagonal_support_fraction": float(np.mean([
                             r['FL_foot_contact'] == r['RR_foot_contact'] and r['FR_foot_contact'] == r['RL_foot_contact']
                             and r['FL_foot_contact'] != r['FR_foot_contact'] for r in samples])),
                         "contact_slip_mean": float(np.mean([r[f'{n}_speed_xy'] for r in samples for n in foot_names if r[f'{n}_contact']]))}
    acceptance = {
        "no_reset": sum(r['done'] for r in rows) == 0,
        "no_base_contact": sum(r['base_contact'] for r in rows) == 0,
        "supported_height": all(s['height_min'] >= 0.25 for s in settled.values()),
        "level": all(s['max_tilt_deg'] < (20 if args_cli.push_speed else 10) for s in settled.values()),
        "walk_tracking": abs(settled['walk']['vx_b_mean'] - args_cli.walk_speed) < 0.12,
        "quiet_stand_stop": all(settled[p]['speed_p95'] < (0.20 if args_cli.push_speed else 0.06) for p in ('stand', 'stop')),
        "feet_lift_in_walk": all(z > 0.04 for z in settled['walk']['foot_height_p95'].values()),
        "limited_slip": settled['walk']['contact_slip_mean'] < 0.12,
        "four_feet_at_rest": all(settled[p]['four_feet_contact_fraction'] > 0.95 for p in ('stand', 'stop')),
    }
    # Non-default commands add axis-specific checks; historical straight-line
    # screens remain unchanged so old/new checkpoint comparisons stay valid.
    if args_cli.lateral_speed or args_cli.yaw_rate:
        acceptance['lateral_tracking'] = abs(settled['walk']['vy_b_mean'] - args_cli.lateral_speed) < 0.12
        acceptance['yaw_tracking'] = abs(settled['walk']['yaw_rate_mean'] - args_cli.yaw_rate) < 0.15
        acceptance['quiet_yaw'] = all(settled[p]['yaw_speed_p95'] < 0.15 for p in ('stand', 'stop'))
    report = {
        "checkpoint": None if args_cli.checkpoint is None else str(args_cli.checkpoint.resolve()),
        "checkpoint_sha256": None if args_cli.checkpoint is None else hashlib.sha256(args_cli.checkpoint.read_bytes()).hexdigest(),
        "task": args_cli.task, "seed": args_cli.seed, "geometry": geometry,
        "protocol": {"stand_s": args_cli.stand_s, "walk_s": args_cli.walk_s, "stop_s": args_cli.stop_s, "walk_speed": args_cli.walk_speed, "lateral_speed": args_cli.lateral_speed, "yaw_rate": args_cli.yaw_rate, "push_delta_vy": args_cli.push_speed},
        "criteria": "diagnostic only: inspect stable height, tilt, foot support, command tracking, and no hidden resets",
        "global": {"min_height": min_height, "max_abs_roll_deg": max_roll, "max_abs_pitch_deg": max_pitch, "max_gravity_xy": max_tilt, "steps": steps},
        "phase_stats": phase_stats, "settled_phase_stats": settled, "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "artifacts": {"video": None if args_cli.no_video else str(video_path), "csv": str(csv_path)},
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
