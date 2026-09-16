"""Evaluate a Go2 recovery checkpoint from deterministic pose classes and optionally record video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
parser.add_argument("--settle_s", type=float, default=0.0,
                    help="Optional pre-policy gravity settling with zero action / nominal-pose PD, NOT passive zero torque.")
parser.add_argument("--state_bank_path", type=Path,
                    help="Optional validated states.npz or directory; only the fixed-physics Bank Play task is allowed.")
parser.add_argument("--state_bank_split", choices=("heldout", "train"), default="heldout")
parser.add_argument("--stochastic_diagnostic", action="store_true",
                    help="BackExplore sampling diagnostic ONLY; never a model acceptance result.")
parser.add_argument("--min_contacts", type=int, default=4, help="Required foot contacts for a valid final stand.")
parser.add_argument("--seed", type=int, default=20260908)
parser.add_argument("--video_pose", choices=(*POSE_CLASSES, "all"))
parser.add_argument("--poses", nargs="+", choices=POSE_CLASSES, default=None)
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument("--video_resolution", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--angle_deg", type=float, default=90.0, help="Roll/pitch angle used for side and fore-aft starts.")
parser.add_argument("--view", choices=("oblique", "front", "side"), default="oblique")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.poses is None:
    args_cli.poses = ("side", "upside_down") if args_cli.state_bank_path else POSE_CLASSES
diagnostic_task = "Isaac-Recovery-Bank-BackExplore-Flat-Unitree-Go2-Play-v0"
target_tasks = tuple(f"Isaac-Recovery-Bank-{v}-Flat-Unitree-Go2-Play-v0"
                     for v in ("NominalTarget", "CurrentTarget"))
if args_cli.stochastic_diagnostic and (args_cli.task != diagnostic_task or not args_cli.state_bank_path):
    parser.error("Stochastic diagnostics require BackExplore Play and an explicit state bank")
if args_cli.state_bank_path:
    allowed_tasks = (diagnostic_task,) if args_cli.stochastic_diagnostic else (
        "Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0", *target_tasks)
    if args_cli.task not in allowed_tasks:
        parser.error(f"This state-bank action mode requires one of {allowed_tasks}")
    if not math.isfinite(args_cli.settle_s) or args_cli.settle_s < 1.0:
        parser.error("State-bank replay requires --settle_s >= 1 for nominal-PD handover validation")
    if any(pose not in ("side", "upside_down") for pose in args_cli.poses):
        parser.error("State-bank poses must be side (actual left/right) or upside_down (actual back)")
    # Process-local only. Bank cfg must not load/sample its TRAIN reset bank while
    # the evaluator supplies the explicitly selected split and fixed state IDs.
    os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = "1"
elif args_cli.task == "Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0":
    parser.error("Bank Play evaluation requires explicit --state_bank_path")
elif args_cli.task in target_tasks:
    # Deterministic controlled-drop evaluation owns its starts, never the train bank.
    os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = "1"
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
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_math import (  # noqa: E402
    reset_clearance_height,
    normal_stance_geometry,
    upright_error_squared,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_control_targets import (  # noqa: E402
    ControlStepJointPositionAction,
)


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


def _start_snapshot(env, foot_ids, *, contacts_fresh: bool, quiet_steps=None) -> list[dict]:
    """Record actual per-trial states; pose labels alone do not prove fallen starts."""
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    base_ids = sensor.find_bodies("base")[0]
    gravity = asset.data.projected_gravity_b
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    tilt = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0)) * (180.0 / torch.pi)
    forces = sensor.data.net_forces_w
    vertical_contacts = (forces[:, foot_ids, 2] > 5.0).sum(1)
    base_contact = (forces[:, base_ids].norm(dim=-1) > 1.0).any(1)
    grounded = (forces.norm(dim=-1) > 1.0).any(1)
    geometry_ok, _, _, _ = _stance_geometry(env)
    standing = (_stable_stand(env, foot_ids, args_cli.min_contacts) & geometry_ok
                & ~base_contact & (vertical_contacts >= args_cli.min_contacts))
    # Conservative operational definition, not a claim of passive / natural falls.
    fallen = grounded & ((tilt >= 60.0) | (base_contact & (height < 0.26)))
    records = []
    for i in range(env.num_envs):
        quiet_duration = 0.0 if quiet_steps is None else float(quiet_steps[i]) * env.step_dt
        settled = contacts_fresh and quiet_duration + 1e-9 >= 0.25
        standing_i = bool(standing[i]) if contacts_fresh else None
        fallen_i = bool(fallen[i]) if contacts_fresh else None
        if not contacts_fresh:
            classification = "controlled_drop_unsettled_contacts_not_yet_valid"
        elif standing_i:
            classification = "standing_at_policy_start_not_fallen_recovery"
        elif fallen_i and settled:
            classification = "settled_fallen_under_nominal_pose_PD"
        elif fallen_i:
            classification = "fallen_but_not_settled"
        else:
            classification = "other_nonstanding_start_not_confirmed_fallen"
        records.append({
            "trial": i,
            "projected_gravity_b": gravity[i].cpu().tolist(),
            "tilt_from_upright_deg": float(tilt[i]),
            "gravity_error": float(torch.sqrt(upright_error_squared(gravity[i]))),
            "height_m": float(height[i]),
            "root_position_w_m": asset.data.root_pos_w[i].cpu().tolist(),
            "root_quaternion_wxyz": asset.data.root_quat_w[i].cpu().tolist(),
            "root_linear_velocity_w_m_s": asset.data.root_lin_vel_w[i].cpu().tolist(),
            "root_angular_velocity_w_rad_s": asset.data.root_ang_vel_w[i].cpu().tolist(),
            "joint_positions_rad": asset.data.joint_pos[i].cpu().tolist(),
            "max_joint_speed_rad_s": float(asset.data.joint_vel[i].abs().max()),
            "contacts_fresh_since_pose_write": contacts_fresh,
            "foot_vertical_forces_N": forces[i, foot_ids, 2].cpu().tolist() if contacts_fresh else None,
            "vertical_foot_contacts": int(vertical_contacts[i]) if contacts_fresh else None,
            "base_contact": bool(base_contact[i]) if contacts_fresh else None,
            "any_body_contact": bool(grounded[i]) if contacts_fresh else None,
            "geometry_ok": bool(geometry_ok[i]),
            "quiet_supported_window_s": quiet_duration,
            "settled": settled,
            "standing_at_policy_start": standing_i,
            "fallen_at_policy_start": fallen_i,
            "eligible_settled_fallen_recovery": bool(fallen_i and settled and not standing_i),
            "classification": classification,
        })
    return records


def _settle_nominal_pose(env, settle_steps: int) -> torch.Tensor:
    """Step physics only: no policy, reward, curriculum, interval events, or auto-reset."""
    terms = env.action_manager.active_terms
    if len(terms) != 1:
        raise RuntimeError("PD settling requires exactly one default-offset joint-position action term.")
    term = env.action_manager.get_term(terms[0])
    sampled_target = type(term) is ControlStepJointPositionAction
    if ((not sampled_target and (type(term) is not JointPositionAction or not term.cfg.use_default_offset))
            or term.cfg.asset_name != "robot" or term.cfg.clip is not None
            or term.action_dim != env.scene["robot"].num_joints):
        raise RuntimeError("Zero action is not verified as the complete nominal joint-position PD target.")
    zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    quiet_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    with torch.no_grad():
        for step in range(settle_steps):
            if not sampled_target:
                env.action_manager.process_action(zeros)
            # Mirror the public action/physics APIs used by ManagerBasedRLEnv.step,
            # but never call env.step: it automatically resets done environments.
            for _ in range(env.cfg.decimation):
                env._sim_step_counter += 1
                if sampled_target:
                    # Handover is explicitly the SAME nominal-pose PD in both arms,
                    # independent of what a zero action means to the new policy.
                    asset = env.scene["robot"]
                    asset.set_joint_position_target(asset.data.default_joint_pos)
                else:
                    env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                if rendering and env._sim_step_counter % env.cfg.sim.render_interval == 0:
                    env.sim.render()
                env.scene.update(dt=env.physics_dt)
            env.episode_length_buf += 1
            dones = env.termination_manager.compute()
            if dones.any():
                indices = dones.nonzero(as_tuple=False).flatten().cpu().tolist()
                raise RuntimeError(f"Termination during PD settling at step {step + 1}, trials {indices}; "
                                   "no automatic reset was performed; evaluation aborted.")
            asset = env.scene["robot"]
            grounded = (env.scene.sensors["contact_forces"].data.net_forces_w.norm(dim=-1) > 1.0).any(1)
            quiet = (grounded & (asset.data.root_lin_vel_w.norm(dim=1) < 0.10)
                     & (asset.data.root_ang_vel_w.norm(dim=1) < 0.20)
                     & (asset.data.joint_vel.abs().amax(dim=1) < 0.50))
            quiet_steps = torch.where(quiet, quiet_steps + 1, torch.zeros_like(quiet_steps))
    return quiet_steps


def _reset_policy_history(env, policy_nn) -> None:
    """Clear inference history without resetting the settled physical state."""
    env.action_manager.reset()
    env.observation_manager.reset()
    env.termination_manager.reset()
    env.episode_length_buf.zero_()
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(torch.ones(env.num_envs, device=env.device, dtype=torch.bool))


def _select_bank_states(bank, pose_class: str, count: int, seed: int):
    """CPU-local RNG, actual pose labels, and full pool coverage before repeats.

    Per-class seeding makes selected IDs independent of checkpoint and evaluation
    pose order. Stored requested labels never control selection.
    """
    labels = {"side": ("left", "right"), "upside_down": ("back",)}
    if pose_class not in labels or count < 1:
        raise ValueError("Bank selection needs side/upside_down and positive trial count")
    id_values = bank["state_id"].cpu().tolist()
    candidates = [i for i, label in enumerate(bank["pose_class"]) if label in labels[pose_class]]
    if not candidates:
        raise ValueError(f"Selected state-bank split has no actual {labels[pose_class]} states")
    candidates.sort(key=lambda index: id_values[index])
    pool = torch.tensor(candidates, dtype=torch.long)
    salt = 104729 if pose_class == "side" else 130363
    generator = torch.Generator(device="cpu").manual_seed((int(seed) + salt) % (2**63 - 1))
    selected = pool[torch.randperm(len(pool), generator=generator)[:count]]
    if count > len(pool):
        extra = pool[torch.randint(len(pool), (count - len(pool),), generator=generator)]
        selected = torch.cat((selected, extra))
    selected_list = selected.tolist()
    selected_ids = [id_values[i] for i in selected_list]
    actual_labels = [bank["pose_class"][i] for i in selected_list]
    info = {
        "available_unique_states": len(pool), "selected_state_ids": selected_ids,
        "selected_unique_state_count": len(set(selected_ids)),
        "sampling_with_replacement": count > len(pool),
        "selected_actual_pose_classes": actual_labels,
        "selected_requested_pose_classes": [bank["requested_pose_class"][i] for i in selected_list],
        "actual_pose_class_counts": {label: actual_labels.count(label) for label in labels[pose_class]},
        "yaw_augmentation_rad": 0.0,
    }
    return selected.to(device=bank["state_id"].device), info


def _validate_bank_environment(bank, env) -> None:
    """Compare bank provenance against the fixed-physics Bank Play environment."""
    cfg, metadata = env.cfg, bank["metadata"]
    if not cfg.events.reset_base.params.get("collection_mode") or cfg.events.reset_robot_joints is not None:
        raise ValueError("State-bank evaluation requires collection mode and no later joint reset")
    if cfg.events.add_base_mass is not None or cfg.events.base_com is not None:
        raise ValueError("State-bank comparison must disable mass/CoM randomization")
    if not cfg.scene.robot.spawn.articulation_props.enabled_self_collisions:
        raise ValueError("State-bank comparison requires self collisions")
    if metadata["body_names"] != list(env.scene["robot"].body_names):
        raise ValueError("State-bank body names/order differ from target robot")
    physics = metadata.get("physics", {})
    if "dt" in physics and not math.isclose(float(physics["dt"]), env.physics_dt, abs_tol=1e-9):
        raise ValueError("State-bank physics dt differs from target environment")
    if "decimation" in physics and int(physics["decimation"]) != cfg.decimation:
        raise ValueError("State-bank control decimation differs from target environment")
    material = metadata.get("material", {})
    for name, expected in (("static_friction", 0.8), ("dynamic_friction", 0.6), ("restitution", 0.0)):
        target = cfg.events.physics_material.params[f"{name}_range"]
        if tuple(target) != (expected, expected):
            raise ValueError(f"Bank task has unexpected fixed {name}")
        if name in material and not math.isclose(float(material[name]), expected, abs_tol=1e-9):
            raise ValueError(f"Bank manifest {name} differs from target environment")


def _set_bank_states(env, bank, selected, transform_bank_root) -> None:
    """Restore complete measured state, changing only the environment origin."""
    asset = env.scene["robot"]
    ids = torch.arange(env.num_envs, device=env.device)
    pose, velocity = transform_bank_root(
        bank["root_pose_local"][selected], bank["root_velocity_w"][selected],
        env.scene.env_origins[ids], 0.0,
    )
    asset.write_joint_state_to_sim(bank["joint_pos"][selected], bank["joint_vel"][selected], env_ids=ids)
    asset.write_root_pose_to_sim(pose, env_ids=ids)  # Root LINK pose.
    asset.write_root_velocity_to_sim(velocity, env_ids=ids)  # Root CoM world velocity.
    asset.set_joint_position_target(asset.data.default_joint_pos[ids], env_ids=ids)
    asset.set_joint_velocity_target(torch.zeros_like(bank["joint_vel"][selected]), env_ids=ids)
    asset.set_joint_effort_target(torch.zeros_like(bank["joint_vel"][selected]), env_ids=ids)
    env.action_manager.reset(ids)
    env.scene.sensors["contact_forces"].reset(ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    env.episode_length_buf.zero_()


def _attach_bank_provenance(records, selection) -> None:
    for i, state in enumerate(records):
        state["bank_state_id"] = selection["selected_state_ids"][i]
        state["bank_saved_pose_class"] = selection["selected_actual_pose_classes"][i]
        state["bank_requested_pose_class"] = selection["selected_requested_pose_classes"][i]
        gravity = state["projected_gravity_b"]
        axis = max(range(3), key=lambda j: abs(gravity[j]))
        state["actual_pose_class"] = ("left" if gravity[1] > 0 else "right") if axis == 1 else (
            "back" if axis == 2 and gravity[2] > 0 else "other")


def _trace_row(env, foot_ids, pose_class, step, stable_steps):
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    gravity = asset.data.projected_gravity_b[0]
    base_id = sensor.find_bodies("base")[0]
    geometry_ok, foot_b, knee_b, delta = _stance_geometry(env)
    # Copy trial-zero trace data in two blocks rather than synchronizing for
    # every scalar. Keep integer counters separate from float32 pose data.
    values = torch.cat((
        torch.stack((
            asset.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2],
            torch.atan2(gravity[1], -gravity[2]) * 180 / torch.pi,
            torch.atan2(gravity[0], -gravity[2]) * 180 / torch.pi,
            torch.sqrt(upright_error_squared(gravity)),
            asset.data.root_lin_vel_w[0].norm(),
            asset.data.root_ang_vel_w[0].norm(),
            (asset.data.joint_pos[0] - asset.data.default_joint_pos[0]).square().mean().sqrt(),
        )),
        torch.cat((foot_b[0], knee_b[0, :, 1:2]), dim=1).reshape(-1),
        torch.stack((asset.data.joint_pos[0], delta[0]), dim=1).reshape(-1),
    )).detach().cpu().tolist()
    counts = torch.stack((
        (sensor.data.net_forces_w[0, foot_ids, 2] > 5).sum(),
        (sensor.data.net_forces_w[0, base_id].norm(dim=-1) > 1).any(),
        stable_steps[0],
        geometry_ok[0],
    )).to(torch.int64).detach().cpu().tolist()
    row = {
        "pose": pose_class, "time_s": (step + 1) * env.step_dt, "trial": 0,
        "height": float(values[0]),
        "roll_deg": float(values[1]),
        "pitch_deg": float(values[2]),
        "gravity_error": float(values[3]),
        "speed": float(values[4]),
        "angular_speed": float(values[5]),
        "feet_contact": int(counts[0]),
        "base_contact": bool(counts[1]),
        "joint_rms": float(values[6]),
        "stable_hold_s": float(counts[2]) * env.step_dt,
        "geometry_ok": bool(counts[3]),
    }
    for i, name in enumerate(("FL", "FR", "RL", "RR")):
        for j, axis in enumerate("xyz"):
            row[f"{name}_foot_{axis}_b"] = float(values[7 + 4 * i + j])
        row[f"{name}_knee_y_b"] = float(values[7 + 4 * i + 3])
    for i, name in enumerate(asset.joint_names):
        row[name] = float(values[23 + 2 * i])
        row[name + "_offset"] = float(values[23 + 2 * i + 1])
    return row


def _open_video(path: Path, frame: np.ndarray, fps: int):
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _annotate(frame: np.ndarray, pose_class: str, step: int, dt: float, stable_s: float, row: dict,
              start_classification: str | None = None) -> np.ndarray:
    image = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
    labels = (
        (f"STOCHASTIC DIAGNOSTIC - NOT ACCEPTANCE | {pose_class}"
         if getattr(args_cli, "stochastic_diagnostic", False) else f"Go2 self-recovery | pose: {pose_class}"),
        f"time: {(step + 1) * dt:4.2f} s | valid stand hold: {stable_s:4.2f} s",
        ("LEG SHAPE ONLY: OK" if row["geometry_ok"] else "INVALID LEG GEOMETRY")
        + (f" | valid stand held {args_cli.hold_s:g}s" if stable_s >= args_cli.hold_s else " | stand hold pending"),
        f"{args_cli.checkpoint.parent.name} / {args_cli.checkpoint.name}",
        ("Simulation | pre-settled nominal-pose PD (NOT zero torque)" if args_cli.settle_s > 0
         else "Simulation | controlled-drop starts | no hardware validation"),
    )
    if args_cli.state_bank_path:
        labels = labels[:-1] + (f"Simulation | state-bank {args_cli.state_bank_split} | nominal PD handover",)
    if start_classification is not None:
        labels += (f"Actual policy start: {start_classification}",)
    for i, text in enumerate(labels):
        # Leg shape in body coordinates can pass while upside down. Green is
        # reserved for a CURRENT completed, continuous valid-standing hold.
        status_color = ((80, 220, 80) if stable_s >= args_cli.hold_s
                        else ((0, 190, 255) if row['geometry_ok'] else (60, 80, 255)))
        if getattr(args_cli, "stochastic_diagnostic", False):
            status_color = (0, 190, 255)  # Never render a green acceptance marker.
        color = status_color if i == 2 else (245, 245, 245)
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
    offset = {"front": (1.65, 0.0, 0.65), "side": (0.0, 1.75, 0.65), "oblique": (1.30, 1.30, 0.75)}[args_cli.view]
    # Back-down feet are ABOVE the torso: root_height-0.08 aims into the
    # floor and clips them. Frame the actual articulated body bounds instead.
    bodies = a.body_pos_w[0]
    target = (bodies.amin(dim=0) + bodies.amax(dim=0)) * 0.5
    target[2] += 0.08  # Leave the robot below the diagnostic text overlay.
    eye = target + math_utils.quat_apply(heading, a.root_pos_w.new_tensor([offset]))[0]
    env.sim.set_camera_view(eye=eye.cpu().tolist(), target=target.cpu().tolist())


def _select_policy_action(runner, device, stochastic_diagnostic=False):
    inference = runner.get_inference_policy(device=device)
    return runner.alg.policy.act if stochastic_diagnostic else inference


def _mark_action_mode(report, stochastic_diagnostic=False):
    report["policy_action_mode"] = "stochastic_diagnostic" if stochastic_diagnostic else "deterministic_mean"
    report["acceptance_eligible"] = not stochastic_diagnostic
    if stochastic_diagnostic:
        report["protocol_version"] += "_stochastic_diagnostic_NOT_ACCEPTANCE"
        report["success_count_definition"] = (
            "STOCHASTIC TRAINING DIAGNOSTIC ONLY: counts below cannot accept or promote a model. "
            + report["success_count_definition"])


def _new_motion_diagnostic(env):
    a = env.scene["robot"].data
    return {"q_min": a.joint_pos.clone(), "q_max": a.joint_pos.clone(),
            "joint_speed_peak": torch.zeros_like(a.joint_pos),
            "applied_torque_peak": torch.zeros_like(a.joint_pos),
            "std_min": torch.full_like(a.joint_pos, float("inf")),
            "std_max": torch.zeros_like(a.joint_pos),
            "min_tilt_deg": torch.full_like(a.root_pos_w[:, 2], 180.0),
            "max_height_m": a.root_pos_w[:, 2] - env.scene.env_origins[:, 2]}


def _update_motion_diagnostic(env, motion, action_std):
    # Read-only control-boundary samples; these are NOT maxima over every
    # physics substep, and cannot prove that substep saturation never occurred.
    a = env.scene["robot"].data
    motion["q_min"] = torch.minimum(motion["q_min"], a.joint_pos)
    motion["q_max"] = torch.maximum(motion["q_max"], a.joint_pos)
    motion["joint_speed_peak"] = torch.maximum(motion["joint_speed_peak"], a.joint_vel.abs())
    motion["applied_torque_peak"] = torch.maximum(motion["applied_torque_peak"], a.applied_torque.abs())
    motion["std_min"] = torch.minimum(motion["std_min"], action_std)
    motion["std_max"] = torch.maximum(motion["std_max"], action_std)
    gravity = a.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-gravity[:, 2] / gravity.norm(dim=-1).clamp_min(1e-6)).clamp(-1, 1)))
    motion["min_tilt_deg"] = torch.minimum(motion["min_tilt_deg"], tilt)
    motion["max_height_m"] = torch.maximum(motion["max_height_m"], a.root_pos_w[:, 2] - env.scene.env_origins[:, 2])


def _motion_diagnostic_report(motion, step_dt=0.02):
    return {"sample_rate_hz": 1.0 / step_dt, "acceptance_eligible": False,
            "measurement": "policy control-boundary samples only; peaks may miss physics-substep extrema",
            "joint_span_rad": (motion["q_max"] - motion["q_min"]).cpu().tolist(),
            **{key: value.cpu().tolist() for key, value in motion.items() if key not in ("q_min", "q_max")}}


def main() -> None:
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.trials < 1 or args_cli.horizon_s <= 0 or args_cli.hold_s <= 0:
        raise ValueError("Trials, horizon and hold time must be positive.")
    if not math.isfinite(args_cli.settle_s) or not 0.0 <= args_cli.settle_s <= 60.0:
        raise ValueError("settle_s must be finite and between 0 and 60 seconds.")
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
    env_cfg.episode_length_s = args_cli.settle_s + args_cli.horizon_s + args_cli.hold_s + 1.0
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # _camera owns yaw-relative framing each recorded frame. asset_root's
    # post-render callback would overwrite it with the default wide view.
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (2.3, 2.3, 1.35)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.30)
    env_cfg.viewer.resolution = tuple(args_cli.video_resolution)
    env = gym.make(eval_task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_pose else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(str(args_cli.checkpoint))
    policy = _select_policy_action(runner, env.device, args_cli.stochastic_diagnostic)
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
    settle_steps = math.ceil(args_cli.settle_s / env.unwrapped.step_dt)
    settle_actual_s = settle_steps * env.unwrapped.step_dt
    joint_names = list(env.unwrapped.scene["robot"].joint_names)
    results: dict[str, dict] = {}
    trace = []
    checkpoint_name = args_cli.checkpoint.name
    bank = None

    try:
        if args_cli.state_bank_path:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_state_bank import (
                load_recovery_state_bank, transform_bank_root, validate_bank_joint_limits,
            )
            bank = load_recovery_state_bank(args_cli.state_bank_path, joint_names, env.device,
                                            split=args_cli.state_bank_split)
            limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0]
            validate_bank_joint_limits(bank, limits[:, 0], limits[:, 1])
            _validate_bank_environment(bank, env.unwrapped)
        for pose_class in args_cli.poses:
            env.reset()
            selection = None
            if bank is None:
                _set_pose_class(env.unwrapped, pose_class, generator)
            else:
                selected, selection = _select_bank_states(bank, pose_class, env.num_envs, args_cli.seed)
                _set_bank_states(env.unwrapped, bank, selected, transform_bank_root)
            release_state = _start_snapshot(env.unwrapped, foot_ids, contacts_fresh=False)
            if settle_steps:
                quiet_steps = _settle_nominal_pose(env.unwrapped, settle_steps)
                policy_start_state = _start_snapshot(
                    env.unwrapped, foot_ids, contacts_fresh=True, quiet_steps=quiet_steps)
                _reset_policy_history(env.unwrapped, policy_nn)
            else:
                # Preserve the historical default path: no additional physics step
                # or history reset, so existing controlled-drop runs remain reproducible.
                policy_start_state = release_state
            if selection is not None:
                _attach_bank_provenance(release_state, selection)
                _attach_bank_provenance(policy_start_state, selection)
            eligible_fallen = torch.tensor(
                [state["eligible_settled_fallen_recovery"] for state in policy_start_state],
                device=env.device, dtype=torch.bool)
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
            start_label = policy_start_state[0]["classification"] if settle_steps else None
            if selection is not None:
                start_label = f"ID {selection['selected_state_ids'][0]} | {start_label}"
            if settle_steps and args_cli.video_pose in (pose_class, "all"):
                # Show the actual pre-policy state before the first learned action.
                _camera(env.unwrapped)
                env.unwrapped.sim.render()
                env.unwrapped.sim.render()
                frame = env.unwrapped.render()
                if frame is None:
                    raise RuntimeError("No RGB frame returned for the pre-policy state.")
                video_writer = _open_video(video_path, frame, args_cli.video_fps)
                initial_row = _trace_row(env.unwrapped, foot_ids, pose_class, -1, stable_steps)
                if selection is not None:
                    initial_row["bank_state_id"] = selection["selected_state_ids"][0]
                trace.append(initial_row)
                video_writer.write(_annotate(frame, pose_class, -1, env.unwrapped.step_dt, 0.0,
                                             initial_row, start_label))
            motion = _new_motion_diagnostic(env.unwrapped) if args_cli.stochastic_diagnostic else None
            for step in range(steps):
                # no_grad keeps Isaac Lab's mutable simulator buffers writable across
                # deterministic pose resets; inference_mode would mark them immutable.
                with torch.no_grad():
                    obs, _, dones, _ = env.step(policy(obs))
                if motion is not None:
                    _update_motion_diagnostic(env.unwrapped, motion, policy_nn.action_std)
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
                if selection is not None:
                    trace[-1]["bank_state_id"] = selection["selected_state_ids"][0]
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
                            start_label,
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
                "release_state_before_settling": release_state,
                "policy_start_state": policy_start_state,
                "standing_starts_not_fallen_recovery": sum(
                    state["standing_at_policy_start"] is True for state in policy_start_state),
                "settled_fallen_trials": int(eligible_fallen.sum().item()),
                "settled_fallen_recovery_successes": int((success & eligible_fallen).sum().item()),
                "settled_fallen_final_valid_stands": int(
                    ((stable_steps >= hold_steps) & eligible_fallen).sum().item()),
                "settled_fallen_recovery_rate": (float(success[eligible_fallen].float().mean().item())
                                                 if eligible_fallen.any() else None),
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
            if motion is not None:
                results[pose_class]["stochastic_motion_diagnostic"] = _motion_diagnostic_report(motion, env.unwrapped.step_dt)
            if selection is not None:
                results[pose_class]["state_bank_selection"] = selection
                eligible_ids = {state["bank_state_id"] for state in policy_start_state
                                if state["eligible_settled_fallen_recovery"]}
                results[pose_class]["settled_fallen_unique_state_count"] = len(eligible_ids)
                eligible_times = recovered_at[success & eligible_fallen].cpu().numpy()
                results[pose_class]["settled_fallen_median_recovery_s"] = (
                    float(np.median(eligible_times)) if len(eligible_times) else None)
                for i, diagnostic in enumerate(results[pose_class]["final_diagnostics"]):
                    diagnostic["bank_state_id"] = selection["selected_state_ids"][i]
            summary = {key: value for key, value in results[pose_class].items()
                       if key not in ("final_diagnostics", "release_state_before_settling", "policy_start_state")}
            print(f"{pose_class}: {summary}")
    finally:
        env.close()

    report = {
        "angle_deg": args_cli.angle_deg,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args_cli.checkpoint.read_bytes()).hexdigest(),
        "task": eval_task,
        "seed": args_cli.seed,
        "start_protocol": ("pre-settled nominal-pose PD: gravity acts with zero policy action commanding default "
                           "joint positions; NOT passive zero-torque settling; actual fallen/standing status recorded per trial"
                           if settle_steps else
                           "controlled drop from conservative default-pose envelope, not pre-settled fallen poses"),
        "start_protocol_id": "pre_settled_nominal_pose_PD" if settle_steps else "controlled_drop",
        "settle_requested_s": args_cli.settle_s,
        "settle_actual_s": settle_actual_s,
        "settle_control_steps": settle_steps,
        "settle_controller": "zero action, default joint-position PD; not zero torque" if settle_steps else None,
        "settle_execution": ("physics/action substeps only; termination checked each control step; auto-reset "
                             "prohibited; no learned policy, reward computation, curriculum or interval events"
                             if settle_steps else None),
        "settled_definition": "at least 0.25 s continuous any-body contact >1 N, root speed <0.10 m/s, "
                              "angular speed <0.20 rad/s, every joint speed <0.50 rad/s before policy starts",
        "fallen_start_definition": "current any-body contact >1 N and either tilt from upright >=60 deg "
                                   "or base contact >1 N with height <0.26 m; eligible only if settled and not already standing",
        "success_count_definition": "successes counts valid stand acquisition/retention from all starts; "
                                    "only settled_fallen_recovery_successes counts confirmed settled-fallen starts; "
                                    "already-standing starts are never counted as fallen recovery",
        "time_definition": "policy time starts AFTER optional settling; onset of the first stable interval held "
                           "for hold_s; policy simulation includes hold_s after horizon_s; settling is excluded",
        "protocol_version": "stance_geometry_v1_presettled_PD_v1" if settle_steps else "stance_geometry_v1",
        "joint_names": joint_names,
        "video_view": args_cli.view if args_cli.video_pose else None,
        "self_collisions_enabled": env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
        "criterion": f"gravity error < 0.35, height 0.30-0.55 m, speed < 0.50 m/s, angular speed < 1.00 rad/s, {args_cli.min_contacts} simultaneous foot vertical forces > 5 N, no base contact, feet on correct body sides (0.06 < signed lateral < 0.30 m), knees on correct sides (>0.04 m), fore/hind feet on correct ends (>0.08 m), each joint offset <0.65 rad; all continuously held for hold_s",
        "results": results,
        "diagnostic_trial": 0,
    }
    if bank is not None:
        selected_ids = [state_id for result in results.values()
                        for state_id in result["state_bank_selection"]["selected_state_ids"]]
        report.update({
            "angle_deg": None,
            "start_protocol_id": f"state_bank_{args_cli.state_bank_split}_nominal_pose_PD",
            "start_protocol": "Complete measured fallen root-link pose, CoM world velocity and joint state "
                              "replayed from the explicit bank split; only environment-origin translation, "
                              "no yaw augmentation, root lifting or new joint noise; at least 1 s nominal-PD "
                              "handover precedes policy time; actual settled-fallen eligibility rechecked",
            "protocol_version": "stance_geometry_v1_state_bank_PD_v1",
            "state_bank": {
                "path": bank["archive_path"], "manifest_path": bank["manifest_path"],
                "sha256": bank["archive_sha256"], "schema_version": bank["metadata"]["schema_version"],
                "split": args_cli.state_bank_split, "available_split_states": len(bank["state_id"]),
                "selected_state_ids": selected_ids, "selected_unique_state_count": len(set(selected_ids)),
                "selection_protocol": "per-class CPU seed; actual class, sorted IDs, random permutation "
                                      "without replacement then extra draws only if trials exceed available states",
                "comparison_requirement": "Compare checkpoints under the same Bank Play task, bank SHA-256, "
                                          "heldout split, selected IDs/order, seed, trial count and handover duration; "
                                          "repeated draws are not additional independent source states",
                "fixed_material": bank["metadata"].get("material"),
                "physics": bank["metadata"].get("physics"),
                "diagnostic_train_split_only": args_cli.state_bank_split != "heldout",
            },
        })
    _mark_action_mode(report, args_cli.stochastic_diagnostic)
    if args_cli.task in target_tasks:
        reference = env_cfg.actions.joint_pos.reference
        report["action_representation"] = {
            "reference": reference, "scale": env_cfg.actions.joint_pos.scale,
            "target": "q_reference + scale * action, soft-joint-limit clamped",
            "sample_period_s": env_cfg.sim.dt * env_cfg.decimation,
            "held_over_physics_substeps": env_cfg.decimation,
            "pre_policy_handover": "direct nominal-position PD, not policy zero action",
        }
        if "state_bank" in report:
            report["state_bank"]["comparison_requirement"] += (
                "; action-reference pairs intentionally differ in action semantics only; compare recorded action_representation")
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
