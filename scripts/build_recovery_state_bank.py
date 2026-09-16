"""Collect and cold-replay-check stationary fallen Go2 states under nominal-pose PD.

Run only in a separate collection process, never from a training reset callback.
No learned policy is used. Zero action means default joint-position PD, NOT zero torque.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--settle_s", type=float, default=2.0)
    parser.add_argument("--max_settle_s", type=float, default=5.0)
    parser.add_argument("--replay_s", type=float, default=1.0)
    parser.add_argument("--quiet_hold_s", type=float, default=0.5)
    parser.add_argument("--heldout_every", type=int, default=5,
                        help="Every Nth whole generation batch is held out; never split yaw variants.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for name in ("settle_s", "max_settle_s", "replay_s", "quiet_hold_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if not args.quiet_hold_s <= args.settle_s <= args.max_settle_s <= 60:
        parser.error("Require quiet_hold_s <= settle_s <= max_settle_s <= 60")
    if not args.quiet_hold_s <= args.replay_s <= 60:
        parser.error("Require quiet_hold_s <= replay_s <= 60")
    if args.num_envs < 1 or args.batches < 1 or args.heldout_every < 2:
        parser.error("num_envs/batches must be positive and heldout_every >= 2")
    if args.output_dir.resolve().drive.upper() != "E:":
        parser.error("All generated state-bank artifacts must be stored on E:")
    if any((args.output_dir / name).exists() for name in ("states.npz", "manifest.json")):
        parser.error("Refusing to overwrite an existing bank: choose a new output_dir")
    if os.environ.get("ISAACLAB_RECOVERY_BANK_COLLECTION") != "1":
        parser.error("Set ISAACLAB_RECOVERY_BANK_COLLECTION=1, or use build_recovery_state_bank.ps1")
    return args


args_cli = _arguments()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_math import (  # noqa: E402
    reset_clearance_height,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_state_bank import (  # noqa: E402
    validate_recovery_state_bank,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


THRESHOLDS = {
    "root_speed_m_s": 0.10,
    "root_angular_speed_rad_s": 0.20,
    "joint_speed_rad_s": 0.50,
    "contact_force_N": 1.0,
    "minimum_fallen_tilt_deg": 60.0,
    "maximum_fallen_height_m": 0.40,
    "replay_height_drift_m": 0.02,
    "replay_xy_drift_m": 0.03,
    "replay_orientation_drift_deg": 5.0,
    "replay_joint_drift_rad": 0.10,
    "replay_peak_root_speed_m_s": 0.35,
    "replay_peak_angular_speed_rad_s": 1.0,
    "replay_peak_joint_speed_rad_s": 4.0,
    "replay_peak_body_force_to_weight": 8.0,
}
REQUESTED = ("left", "right", "back", "random")
ACTUAL = ("other", "left", "right", "back")


def _uniform(shape, low, high, env, generator):
    return low + (high - low) * torch.rand(shape, device=env.device, generator=generator)


def _validate_collection_task(env):
    cfg = env.cfg
    if cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is not True:
        raise RuntimeError("Collection requires enabled_self_collisions=True")
    for name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
        if getattr(cfg.events, name, None) is not None:
            raise RuntimeError(f"Collection task must disable event {name}; do not silently change training physics")
    material = getattr(cfg.events, "physics_material", None)
    expected = {"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0)}
    if material is None or any(tuple(material.params.get(key, ())) != value for key, value in expected.items()):
        raise RuntimeError("Collection requires fixed friction 0.8/0.6 and restitution 0.0")
    terms = env.action_manager.active_terms
    if len(terms) != 1:
        raise RuntimeError("Expected exactly one default-offset joint-position action term")
    term = env.action_manager.get_term(terms[0])
    if (type(term) is not JointPositionAction or not term.cfg.use_default_offset or term.cfg.clip is not None
            or term.cfg.asset_name != "robot" or term.action_dim != 12):
        raise RuntimeError("Zero action must be a verified complete 12-joint nominal-pose PD target")
    if env.scene["robot"].num_joints != 12:
        raise RuntimeError("The state-bank schema requires the 12 actuated Go2 joints")


def _write_state(env, root_pose_local, root_velocity_w, joint_pos, joint_vel):
    """Write a whole offline batch without advancing physics; never called in training."""
    robot = env.scene["robot"]
    ids = torch.arange(env.num_envs, device=env.device)
    pose_w = root_pose_local.clone()
    pose_w[:, :3] += env.scene.env_origins
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    robot.write_root_pose_to_sim(pose_w, env_ids=ids)
    robot.write_root_velocity_to_sim(root_velocity_w, env_ids=ids)
    robot.set_joint_position_target(robot.data.default_joint_pos, env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=ids)
    robot.set_joint_effort_target(torch.zeros_like(joint_vel), env_ids=ids)
    env.action_manager.reset()
    env.observation_manager.reset()
    env.termination_manager.reset()
    env.episode_length_buf.zero_()
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)


def _physics_step(env, zero_actions):
    """No env.step(), automatic reset, reward, curriculum or interval events."""
    env.action_manager.process_action(zero_actions)
    rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if rendering and env._sim_step_counter % env.cfg.sim.render_interval == 0:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    dones = env.termination_manager.compute()
    if dones.any():
        raise RuntimeError(f"Unexpected done in offline physics phase, env ids {dones.nonzero().flatten().tolist()}; "
                           "no automatic reset was performed")


def _measure(env):
    a = env.scene["robot"].data
    force = env.scene.sensors["contact_forces"].data.net_forces_w
    gravity = a.projected_gravity_b
    height = a.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    speed = a.root_lin_vel_w.norm(dim=1)
    angular_speed = a.root_ang_vel_w.norm(dim=1)
    joint_speed = a.joint_vel.abs().amax(dim=1)
    grounded = (force.norm(dim=-1) > THRESHOLDS["contact_force_N"]).any(1)
    finite = (torch.isfinite(a.root_state_w).all(1) & torch.isfinite(a.joint_pos).all(1)
              & torch.isfinite(a.joint_vel).all(1) & torch.isfinite(force).all((1, 2)))
    within_limits = ((a.joint_pos >= a.soft_joint_pos_limits[..., 0])
                     & (a.joint_pos <= a.soft_joint_pos_limits[..., 1])).all(1)
    dominant = gravity.abs().argmax(1)
    actual = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    actual[(dominant == 1) & (gravity[:, 1] > 0.5)] = 1  # Left side points toward ground.
    actual[(dominant == 1) & (gravity[:, 1] < -0.5)] = 2
    actual[(dominant == 2) & (gravity[:, 2] > 0.5)] = 3
    tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
    in_cell = ((a.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).abs() < 0.8).all(1)
    fallen = (grounded & (actual > 0) & (tilt >= THRESHOLDS["minimum_fallen_tilt_deg"])
              & (height > 0.02) & (height < THRESHOLDS["maximum_fallen_height_m"]))
    quiet = (finite & within_limits & in_cell & fallen & (speed < THRESHOLDS["root_speed_m_s"])
             & (angular_speed < THRESHOLDS["root_angular_speed_rad_s"])
             & (joint_speed < THRESHOLDS["joint_speed_rad_s"]))
    # PhysX mass metadata may remain on CPU while contact forces are on CUDA.
    # Collection forbids mass randomization, so transfer this denominator once.
    weight = getattr(env, "_go2_bank_fixed_weight", None)
    if weight is None:
        weight = (a.default_mass.sum(1).to(device=force.device, dtype=force.dtype) * 9.81).clamp_min(1e-6)
        env._go2_bank_fixed_weight = weight
    force_to_weight = force.norm(dim=-1).amax(1) / weight
    return {"finite": finite, "within_limits": within_limits, "in_cell": in_cell,
            "grounded": grounded, "fallen": fallen, "quiet": quiet, "actual": actual,
            "height": height, "tilt": tilt, "speed": speed, "angular_speed": angular_speed,
            "joint_speed": joint_speed, "force_to_weight": force_to_weight}


def _state_tensors(env):
    a = env.scene["robot"].data
    return {"root_pose_local": torch.cat((a.root_pos_w - env.scene.env_origins, a.root_quat_w), dim=1).clone(),
            "root_velocity_w": torch.cat((a.root_lin_vel_w, a.root_ang_vel_w), dim=1).clone(),
            "joint_pos": a.joint_pos.clone(), "joint_vel": a.joint_vel.clone()}


def _initial_batch(env, batch, generator):
    n = env.num_envs
    requested = (torch.arange(n, device=env.device) + batch * n) % len(REQUESTED)
    roll = _uniform((n,), -0.10, 0.10, env, generator)
    pitch = _uniform((n,), -0.10, 0.10, env, generator)
    yaw = _uniform((n,), -math.pi, math.pi, env, generator)
    roll += torch.where(requested == 0, -math.pi / 2, 0.0)
    roll += torch.where(requested == 1, math.pi / 2, 0.0)
    roll += torch.where(requested == 2, math.pi, 0.0)
    orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    random_q = torch.nn.functional.normalize(torch.randn((n, 4), device=env.device, generator=generator), dim=1)
    orientation[requested == 3] = random_q[requested == 3]
    height = reset_clearance_height(math_utils.matrix_from_quat(orientation),
                                    _uniform((n,), 0.015, 0.035, env, generator))
    root_pose = torch.cat((torch.zeros((n, 2), device=env.device), height[:, None], orientation), dim=1)
    qpos = env.scene["robot"].data.default_joint_pos + _uniform((n, 12), -0.10, 0.10, env, generator)
    limits = env.scene["robot"].data.soft_joint_pos_limits
    qpos = qpos.clamp(limits[..., 0], limits[..., 1])
    return requested, {"root_pose_local": root_pose,
                       "root_velocity_w": torch.zeros((n, 6), device=env.device),
                       "joint_pos": qpos, "joint_vel": torch.zeros_like(qpos)}


def _snapshot_diagnostic(env, measure, i, quiet_steps, actual_s):
    a = env.scene["robot"].data
    sensor = env.scene.sensors["contact_forces"]
    base = sensor.find_bodies("base")[0]
    return {"settle_actual_s": actual_s, "quiet_hold_s": float(quiet_steps[i]) * env.step_dt,
            "projected_gravity_b": a.projected_gravity_b[i].cpu().tolist(),
            "height_m": float(measure["height"][i]), "tilt_deg": float(measure["tilt"][i]),
            "root_speed_m_s": float(measure["speed"][i]),
            "root_angular_speed_rad_s": float(measure["angular_speed"][i]),
            "max_joint_speed_rad_s": float(measure["joint_speed"][i]),
            "body_normal_contact_forces_w_N": sensor.data.net_forces_w[i].cpu().tolist(),
            "base_contact": bool((sensor.data.net_forces_w[i, base].norm(dim=-1) > 1).any()),
            "grounded": bool(measure["grounded"][i]), "fallen": bool(measure["fallen"][i]),
            "actual_pose_class": ACTUAL[int(measure["actual"][i])]}


def _collect_batch(env, initial, zero_actions):
    env.reset()
    _write_state(env, **initial)
    quiet_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    captured = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    saved = {name: tensor.clone() for name, tensor in initial.items()}
    diagnostics = {}
    required = math.ceil(args_cli.quiet_hold_s / env.step_dt)
    minimum = math.ceil(args_cli.settle_s / env.step_dt)
    maximum = math.ceil(args_cli.max_settle_s / env.step_dt)
    for step in range(maximum):
        _physics_step(env, zero_actions)
        measured = _measure(env)
        quiet_steps = torch.where(measured["quiet"], quiet_steps + 1, torch.zeros_like(quiet_steps))
        new = (~captured & (quiet_steps >= required)) if step + 1 >= minimum else torch.zeros_like(captured)
        if new.any():
            snapshot = _state_tensors(env)
            for name in saved:
                saved[name][new] = snapshot[name][new]
            for i in new.nonzero().flatten().cpu().tolist():
                diagnostics[i] = _snapshot_diagnostic(env, measured, i, quiet_steps, (step + 1) * env.step_dt)
            captured |= new
        if captured.all():
            break
    rejected = {}
    for i in (~captured).nonzero().flatten().cpu().tolist():
        if not bool(measured["finite"][i]):
            reason = "nonfinite_physics_state"
        elif not bool(measured["within_limits"][i]):
            reason = "outside_soft_joint_limits"
        elif not bool(measured["in_cell"][i]):
            reason = "outside_collection_cell"
        elif not bool(measured["fallen"][i]):
            reason = "not_confirmed_left_right_or_back_fallen"
        else:
            reason = "did_not_reach_continuous_quiet_hold"
        rejected[i] = reason
    return saved, captured, diagnostics, rejected


def _cold_replay(env, saved, candidate_mask, source_diagnostics, zero_actions):
    # env.reset resets sensors, actuator state, action/observation managers. The
    # following full-batch restore is deliberately outside training reset terms.
    env.reset()
    replay = {name: tensor.clone() for name, tensor in saved.items()}
    filler = ~candidate_mask
    replay["root_pose_local"][filler] = replay["root_pose_local"].new_tensor([0, 0, 0.4, 1, 0, 0, 0])
    replay["root_velocity_w"][filler] = 0
    replay["joint_pos"][filler] = env.scene["robot"].data.default_joint_pos[filler]
    replay["joint_vel"][filler] = 0
    _write_state(env, **replay)
    source_classes = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    for i, diagnostic in source_diagnostics.items():
        source_classes[i] = ACTUAL.index(diagnostic["actual_pose_class"])
    good = candidate_mask.clone()
    quiet_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    peak_names = ("speed", "angular_speed", "joint_speed", "force_to_weight", "height_drift",
                  "xy_drift", "orientation_drift", "joint_drift")
    peaks = {name: torch.zeros(env.num_envs, device=env.device) for name in peak_names}
    count = math.ceil(args_cli.replay_s / env.step_dt)
    for _ in range(count):
        _physics_step(env, zero_actions)
        measured = _measure(env)
        a = env.scene["robot"].data
        measured["height_drift"] = (measured["height"] - saved["root_pose_local"][:, 2]).abs()
        measured["xy_drift"] = ((a.root_pos_w - env.scene.env_origins)[:, :2]
                                - saved["root_pose_local"][:, :2]).norm(dim=1)
        dot = (a.root_quat_w * saved["root_pose_local"][:, 3:]).sum(1).abs().clamp(0, 1)
        measured["orientation_drift"] = torch.rad2deg(2 * torch.acos(dot))
        measured["joint_drift"] = (a.joint_pos - saved["joint_pos"]).abs().amax(1)
        for name in peaks:
            peaks[name] = torch.maximum(peaks[name], measured[name])
        quiet_steps = torch.where(measured["quiet"], quiet_steps + 1, torch.zeros_like(quiet_steps))
        good &= measured["finite"] & measured["within_limits"] & measured["in_cell"]
        good &= measured["actual"] == source_classes
    limits = {"speed": "replay_peak_root_speed_m_s", "angular_speed": "replay_peak_angular_speed_rad_s",
              "joint_speed": "replay_peak_joint_speed_rad_s", "force_to_weight": "replay_peak_body_force_to_weight",
              "height_drift": "replay_height_drift_m", "xy_drift": "replay_xy_drift_m",
              "orientation_drift": "replay_orientation_drift_deg", "joint_drift": "replay_joint_drift_rad"}
    for name, threshold in limits.items():
        good &= peaks[name] <= THRESHOLDS[threshold]
    good &= quiet_steps >= math.ceil(args_cli.quiet_hold_s / env.step_dt)
    diagnostics, rejected = {}, {}
    for i in candidate_mask.nonzero().flatten().cpu().tolist():
        if bool(good[i]):
            diagnostics[i] = {"passed": True, "duration_s": count * env.step_dt,
                              "final_quiet_hold_s": float(quiet_steps[i]) * env.step_dt,
                              "final_actual_pose_class": ACTUAL[int(measured["actual"][i])],
                              **{f"peak_{name}": float(value[i]) for name, value in peaks.items()}}
        else:
            reasons = [f"exceeded_{limits[name]}" for name in peaks
                       if not bool(torch.isfinite(peaks[name][i])) or float(peaks[name][i]) > THRESHOLDS[limits[name]]]
            if int(quiet_steps[i]) < math.ceil(args_cli.quiet_hold_s / env.step_dt):
                reasons.append("final_quiet_hold_failed")
            if int(measured["actual"][i]) != int(source_classes[i]):
                reasons.append("actual_pose_class_changed")
            rejected[i] = ";".join(reasons) or "nonfinite_limits_cell_or_transient_pose_change"
    return good, diagnostics, rejected


def _safe_config_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_config_value(item) for item in value]
    return str(value)


def _provenance(env):
    cfg = env.cfg
    a = env.scene["robot"].data
    usd = Path(str(cfg.scene.robot.spawn.usd_path))
    return {"joint_names": list(env.scene["robot"].joint_names),
            "body_names": list(env.scene["robot"].body_names),
            "contact_body_names": list(env.scene.sensors["contact_forces"].body_names),
            "default_joint_pos_rad": a.default_joint_pos[0].cpu().tolist(),
            "joint_default_pos": a.default_joint_pos[0].cpu().tolist(),
            "body_mass_kg": a.default_mass[0].cpu().tolist(),
            "physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt, "decimation": cfg.decimation,
            "physics": {"dt": env.physics_dt, "decimation": cfg.decimation},
            "terrain": {"type": cfg.scene.terrain.terrain_type,
                        "physics_material": _safe_config_value(cfg.scene.terrain.physics_material.to_dict())},
            "env_spacing_m": cfg.scene.env_spacing,
            "robot_usd": str(usd),
            "robot_usd_root_sha256": hashlib.sha256(usd.read_bytes()).hexdigest() if usd.is_file() else None,
            "robot_asset_hash_scope": "root USD only; referenced dependencies are not recursively hashed",
            "articulation_properties": _safe_config_value(cfg.scene.robot.spawn.articulation_props.to_dict()),
            "rigid_body_properties": _safe_config_value(cfg.scene.robot.spawn.rigid_props.to_dict()),
            "material": {"static_friction": 0.8, "dynamic_friction": 0.6, "restitution": 0.0},
            "actuators": {name: {key: _safe_config_value(getattr(actuator, key, None))
                                   for key in ("stiffness", "damping", "effort_limit", "effort_limit_sim",
                                               "saturation_effort", "velocity_limit", "friction")}
                          for name, actuator in cfg.scene.robot.actuators.items()}}


def main():
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 3.0
    cfg.episode_length_s = args_cli.max_settle_s + args_cli.replay_s + 2.0
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    accepted, rejected_records, batch_statistics = [], [], []
    total_candidates = args_cli.num_envs * args_cli.batches
    try:
        _validate_collection_task(env)
        provenance = _provenance(env)
        generator = torch.Generator(device=env.device).manual_seed(args_cli.seed)
        zero_actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        with torch.no_grad():
            for batch in range(args_cli.batches):
                requested, initial = _initial_batch(env, batch, generator)
                saved, captured, source_diagnostics, source_rejected = _collect_batch(env, initial, zero_actions)
                split = int((batch + 1) % args_cli.heldout_every == 0)
                if captured.any():
                    good, replay_diagnostics, replay_rejected = _cold_replay(
                        env, saved, captured, source_diagnostics, zero_actions)
                else:
                    good = captured
                    replay_diagnostics, replay_rejected = {}, {}
                for i in range(env.num_envs):
                    state_id = batch * env.num_envs + i
                    common = {"state_id": state_id, "source_seed": args_cli.seed, "source_batch": batch, "source_env": i,
                              "requested_pose_class": REQUESTED[int(requested[i])], "split": split}
                    if bool(good[i]):
                        accepted.append({**common, "pose_class": source_diagnostics[i]["actual_pose_class"],
                                         "arrays": {name: value[i].cpu().numpy().copy() for name, value in saved.items()},
                                         "collection": source_diagnostics[i], "replay": replay_diagnostics[i]})
                    else:
                        rejected_records.append({**common,
                                                 "phase": "replay" if bool(captured[i]) else "collection",
                                                 "reason": replay_rejected.get(i, source_rejected.get(i, "unknown"))})
                row = {"batch": batch, "split": split, "candidates": env.num_envs,
                       "settled_candidates": int(captured.sum()), "cold_replay_accepted": int(good.sum())}
                batch_statistics.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()

    shape = {"root_pose_local": (7,), "root_velocity_w": (6,), "joint_pos": (12,), "joint_vel": (12,)}
    arrays = {name: (np.stack([item["arrays"][name] for item in accepted]).astype(np.float32)
                     if accepted else np.empty((0, *tail), dtype=np.float32)) for name, tail in shape.items()}
    for name in ("pose_class", "requested_pose_class"):
        arrays[name] = np.asarray([item[name] for item in accepted], dtype="U64")
    arrays["state_id"] = np.asarray([item["state_id"] for item in accepted], dtype=np.int64)
    arrays["split"] = np.asarray([item["split"] for item in accepted], dtype=np.int8)
    if any(not np.isfinite(arrays[name]).all() for name in shape):
        raise RuntimeError("Refusing to save a bank with non-finite state arrays")
    bank_path = args_cli.output_dir / "states.npz"
    np.savez_compressed(bank_path, **arrays)
    # Explicitly prove that the serialized file does not require pickle.
    with np.load(bank_path, allow_pickle=False) as check:
        for name, expected in arrays.items():
            if not np.array_equal(check[name], expected):
                raise RuntimeError(f"NPZ readback mismatch for {name}")
    statistics = {"generated_candidates": total_candidates, "accepted_states": len(accepted),
                  "rejected_states": len(rejected_records),
                  "train_states": sum(item["split"] == 0 for item in accepted),
                  "heldout_states": sum(item["split"] == 1 for item in accepted),
                  "actual_pose_counts": dict(Counter(item["pose_class"] for item in accepted)),
                  "rejection_reasons": dict(Counter(item["reason"] for item in rejected_records))}
    thresholds = {**THRESHOLDS, "quiet_hold_s": args_cli.quiet_hold_s}
    usable = statistics["train_states"] > 0 and statistics["heldout_states"] > 0
    manifest = {"schema_version": "nominal_pd_fallen_v1", "controller": "nominal_pose_PD",
                "controller_description": "zero action commands default joint positions with the existing PD/DC-motor model; NOT passive zero torque",
                "self_collisions_enabled": True,
                "status": "validated" if usable else ("insufficient_splits" if accepted else "no_accepted_states"),
                "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args_cli.seed,
                "task": args_cli.task, "sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                **provenance,
                "frames": {"root_pose_local": "root LINK position minus source env origin; quaternion wxyz in world axes",
                           "root_velocity_w": "root CoM linear xyz and angular xyz velocity, both in world axes",
                           "joint_order": "joint_names", "pose_class": "dominant gravity_b axis: +y left, -y right, +z back"},
                "collection_parameters": {"num_envs": args_cli.num_envs, "batches": args_cli.batches,
                                          "settle_s": args_cli.settle_s, "max_settle_s": args_cli.max_settle_s,
                                          "replay_s": args_cli.replay_s, "quiet_hold_s": args_cli.quiet_hold_s},
                "split_definition": f"whole batch held out when (batch_index + 1) modulo {args_cli.heldout_every} equals 0; no yaw-variant splitting",
                "replay_validation": {"thresholds": thresholds, "statistics": statistics,
                                      "sampling": "diagnostics at each control step; no mesh-depth test",
                                      "states": [{key: value for key, value in item.items() if key != "arrays"}
                                                 for item in accepted]},
                "batch_statistics": batch_statistics, "rejected_candidates": rejected_records,
                "limitations": ["nominal-pose PD settled states, not passive or arbitrary natural-fall states",
                                "collected on a plane with fixed physical parameters; terrain/material changes require new validation",
                                "cold replay resets sensors/actuators/managers and restores measured state including small velocities",
                                "collision-enabled dynamics and bounded cold replay are checked; no mesh-level penetration-depth proof",
                                "state arrays store original accepted pre-replay states, not manually corrected poses"]}
    schema_error = None
    if usable:
        try:
            validate_recovery_state_bank(arrays, manifest, provenance["joint_names"],
                                         archive_sha256=manifest["sha256"])
        except ValueError as exc:
            schema_error = str(exc)
            manifest["status"] = "schema_validation_failed"
            manifest["schema_validation_error"] = schema_error
    (args_cli.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(statistics, indent=2), flush=True)
    print(f"Bank: {bank_path}", flush=True)
    if not accepted:
        raise RuntimeError("No states passed collection and cold-replay validation; diagnostic artifacts were saved")
    if not usable:
        raise RuntimeError("Bank needs nonempty train and heldout splits; increase batches or reduce heldout_every; diagnostics saved")
    if schema_error is not None:
        raise RuntimeError(f"State-bank schema validation failed: {schema_error}")


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
