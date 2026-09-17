"""One real physical episode for the separately scoped continuous diagnostic.

Imported only after AppLauncher. No simulator is created at module import.
Initial placement is before t=0; all subsequent nominal-PD and policy intervals
are recorded. No old evaluator main, pose loop, reset-after-PD or spliced scene.
"""
from __future__ import annotations

import csv
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import continuous_flow_phase as flow
import evaluate_continuous_recovery_flow as entry


class ResetForbidden(RuntimeError):
    pass


class NoResetGuard:
    """Raise BEFORE the environment can transplant terminal physical state."""
    def __init__(self, env):
        self.env, self.attempts, self.original = env, [], {}
        for name in ("reset", "_reset_idx"):
            self.original[name] = getattr(env, name)
            def reject(*args, _name=name, **kwargs):
                self.attempts.append({"method": _name, "sim_step": int(env._sim_step_counter)})
                raise ResetForbidden("Post-initialization reset forbidden: " + _name)
            setattr(env, name, reject)

    def restore_for_close(self):
        for name, value in self.original.items():
            setattr(self.env, name, value)


def physical_index(sim_step, origin):
    if type(sim_step) is not int or type(origin) is not int or sim_step < origin or (sim_step - origin) % 4:
        raise ValueError("Physical counter must advance by complete four-substep intervals")
    return (sim_step - origin) // 4


def safe_json(value):
    """Invalid numeric evidence stays explicit, never becomes a success value."""
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_number": repr(value)}
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    return value


def finite_evidence(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_evidence(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_evidence(v) for v in value)
    return True


def unchanged_between_intervals(previous_after, next_before):
    # Only an explicit real command update is permitted between intervals.
    keys = set(previous_after) | set(next_before)
    return all(previous_after.get(key) == next_before.get(key) for key in keys if key != "velocity_command_b")


def contact_sensor_topology(sensor, asset):
    """Read actual sensor and articulation orders separately; never equate them."""
    feet, base = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], ["base"]
    foot_ids, foot_names = sensor.find_bodies(feet, preserve_order=True)
    base_ids, base_names = sensor.find_bodies(base, preserve_order=True)
    asset_ids, asset_names = asset.find_bodies(feet, preserve_order=True)
    sensor_names, robot_names = list(sensor.body_names), list(asset.body_names)
    if (len(set(sensor_names)) != len(sensor_names) or len(set(robot_names)) != len(robot_names)
            or set(sensor_names) != set(robot_names) or len(sensor_names) != sensor.num_bodies
            or list(foot_names) != feet or list(base_names) != base or list(asset_names) != feet
            or [sensor_names[i] for i in foot_ids] != feet or [sensor_names[i] for i in base_ids] != base
            or [robot_names[i] for i in asset_ids] != feet):
        raise RuntimeError("Actual sensor/articulation topology is incomplete or misordered")
    return {"body_names": sensor_names, "num_bodies": int(sensor.num_bodies),
            "foot_names": feet, "foot_ids": list(foot_ids), "base_names": base, "base_ids": list(base_ids),
            "articulation_body_names": robot_names, "foot_articulation_ids": list(asset_ids)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(safe_json(value), handle, indent=2, allow_nan=False)
        handle.flush()


def snapshot(env):
    a = env.scene["robot"].data
    sensor = env.scene.sensors["contact_forces"]
    tensors = {"root_pos_w": a.root_pos_w, "root_quat_w": a.root_quat_w,
               "root_lin_vel_w": a.root_lin_vel_w, "root_ang_vel_w": a.root_ang_vel_w,
               "joint_pos": a.joint_pos, "joint_vel": a.joint_vel,
               "projected_gravity_b": a.projected_gravity_b,
               "actual_raw_action": env.action_manager.action, "actual_previous_raw_action": env.action_manager.prev_action,
               "executed_target": a.joint_pos_target,
               "velocity_command_b": env.command_manager.get_command("base_velocity"),
               "current_forces_w": sensor.data.net_forces_w,
               "force_history_w": sensor.data.net_forces_w_history,
               "applied_torque_at_control_boundary": a.applied_torque}
    result = {key: value.detach().cpu().tolist() for key, value in tensors.items()}
    result.update(sim_step=int(env._sim_step_counter), common_step=int(env.common_step_counter),
                  episode_step=int(env.episode_length_buf[0]))
    return result


def measurement(env, helpers, foot_ids, base_ids, foot_body_ids, step, phase, phase_interval, role, done):
    import torch
    a, sensor = env.scene["robot"].data, env.scene.sensors["contact_forces"]
    geometry, feet_b, knees_b, delta = helpers["_stance_geometry"](env)
    g = a.projected_gravity_b[0]
    local = a.root_pos_w[0] - env.scene.env_origins[0]
    current = sensor.data.net_forces_w[0]
    base_force = current[base_ids].norm(dim=-1).amax()
    vertical = current[foot_ids, 2] > 5.
    history_feet = sensor.data.net_forces_w_history[0, :, foot_ids].norm(dim=-1).amax(dim=0)
    history_base = sensor.data.net_forces_w_history[0, :, base_ids].norm(dim=-1).amax()
    linear = a.root_lin_vel_w[0].norm()
    angular = a.root_ang_vel_w[0].norm()
    strict = bool(geometry[0] & vertical.all() & (base_force <= 1.) & (local[2] > .30) & (local[2] < .55)
                  & ((g - g.new_tensor((0., 0., -1.))).norm() < .35) & (linear < .5) & (angular < 1.))
    command = env.command_manager.get_command("base_velocity")[0].tolist()
    row = {"step": step, "time_s": (step + 1) * .02, "phase": phase, "phase_interval": phase_interval,
           "actor_role": role, "cmd_x": command[0], "cmd_y": command[1], "cmd_yaw": command[2],
           "height": float(local[2]), "x": float(local[0]), "y": float(local[1]),
           "speed": float(a.root_lin_vel_w[0, :2].norm()), "yaw_speed": float(a.root_ang_vel_w[0, 2].abs()),
           "roll_deg": float(torch.atan2(g[1], -g[2]) * 180 / torch.pi),
           "pitch_deg": float(torch.atan2(g[0], -g[2]) * 180 / torch.pi),
           "foot_contacts": int((history_feet > 5.).sum()), "base_contact": int(history_base > 1.), "done": int(done),
           "vx": float(a.root_lin_vel_w[0, 0]), "vy": float(a.root_lin_vel_w[0, 1]),
           "vx_b": float(a.root_lin_vel_b[0, 0]), "vy_b": float(a.root_lin_vel_b[0, 1]),
           "wz": float(a.root_ang_vel_w[0, 2]), "push_delta_vy": 0.,
           "joint_rms_deviation": float(delta[0].square().mean().sqrt()), "stance_geometry_ok": int(geometry[0]),
           "current_vertical_foot_contacts": int(vertical.sum()), "current_base_contact": int(base_force > 1.),
           "current_base_force_n": float(base_force), "max_joint_offset_rad": float(delta[0].abs().amax()),
           "gravity_b": g.tolist(), "linear_speed_3d": float(linear), "angular_speed_3d": float(angular),
           "strict_valid_after": strict}
    for i, leg in enumerate(("FL", "FR", "RL", "RR")):
        foot = leg + "_foot"
        row.update({foot + "_x_b": float(feet_b[0, i, 0]), foot + "_y_b": float(feet_b[0, i, 1]),
                    foot + "_z_w": float(a.body_pos_w[0, foot_body_ids[i], 2]),
                    foot + "_fz": float(current[foot_ids[i], 2]),
                    foot + "_contact": int(current[foot_ids[i]].norm() > 5.),
                    foot + "_vertical_contact": int(vertical[i]),
                    foot + "_speed_xy": float(a.body_lin_vel_w[0, foot_body_ids[i], :2].norm()),
                    leg + "_knee_y_b": float(knees_b[0, i, 1])})
    row.update({name + "_offset_rad": float(delta[0, i]) for i, name in enumerate(env.scene["robot"].joint_names)})
    tilt = torch.acos((-g[2]).clamp(-1., 1.)) * 180 / torch.pi
    row["fallen_after"] = bool((current.norm(dim=-1) > 1.).any() & ((tilt >= 60.) | ((base_force > 1.) & (local[2] < .26))))
    row["quiet_for_handoff"] = float(linear) < .06 and float(angular) < .15
    return row


def set_command(term, command):
    import torch
    if term.vel_command_b.dtype != torch.float32:
        raise RuntimeError("Exact command transport requires actual float32 manager")
    term.vel_command_b.fill_(0.)
    term.vel_command_b[0, :3] = term.vel_command_b.new_tensor(command)
    term.is_standing_env.fill_(False)
    term.is_heading_env.fill_(False)
    if tuple(term.command[0].tolist()) != command:
        raise RuntimeError("Real command manager did not receive exact phase command")


def assert_continuity(before, after, phase, origin):
    if after["sim_step"] != before["sim_step"] + 4 or physical_index(after["sim_step"], origin) != physical_index(before["sim_step"], origin) + 1:
        raise RuntimeError("Missing, duplicated or reset physics interval")
    if after["episode_step"] != before["episode_step"] + 1:
        raise RuntimeError("Episode counter was reset or skipped")
    increment = 0 if phase == "preparation" else 1
    if after["common_step"] != before["common_step"] + increment:
        raise RuntimeError("Unexpected manager vs direct-PD counter semantics")


def expected_initialized_config(before, env_regex_ns):
    """Only InteractiveScene's exact constructor namespace expansion is allowed.

    This is not a dynamics normalization. All remaining fields, including USD
    paths, collisions, actuators, events and dt, must remain byte-value equal.
    """
    if before["robot"]["prim_path"] != "{ENV_REGEX_NS}/Robot" or env_regex_ns != "/World/envs/env_.*":
        raise RuntimeError("Unexpected original robot namespace contract")
    expected = copy.deepcopy(before)
    expected["robot"]["prim_path"] = before["robot"]["prim_path"].format(ENV_REGEX_NS=env_regex_ns)
    return expected


def run(args, evidence):
    import cv2
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils import math as math_utils
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_math import (
        normal_stance_geometry, upright_error_squared, reset_clearance_height)
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_state_bank import (
        load_recovery_state_bank, validate_bank_joint_limits, transform_bank_root)
    import evaluate_locomotion_recovery_physics as retention
    from continuous_flow_motion_metrics import analyze
    sys.path.insert(0, str(entry.ROOT / "src/go2_recovery"))
    from recovery_handoff_math import update_handoff_gate
    from supported_startup_math import supported_startup_mask
    from roll_mirror_math import initial_right_mask, roll_input_copy, physical_roll_action, validate_mirror_joint_contract

    helpers = {"torch": torch, "math": math, "math_utils": math_utils, "args_cli": args,
               "normal_stance_geometry": normal_stance_geometry, "upright_error_squared": upright_error_squared,
               "reset_clearance_height": reset_clearance_height}
    exec(compile(evidence["helpers_source"], "<continuous-frozen-helpers>", "exec"), helpers)
    (args.output_dir / "extracted_helpers.py").write_text(evidence["helpers_source"], encoding="utf-8")
    env, guard, writer, trace_file = None, None, None, None
    rows, frames, failure, state = [], [], None, None
    locomotion = evidence["locomotion"]
    report = {"protocol_version": entry.PROTOCOL, "scope": "Single real-episode .5 command development diagnostic; not promotion, trot/run or deployment",
              "passed": False, "status": "initializing", "smoke": args.smoke, "refinement": "off", "actors": dict(entry.MODEL_SHA, locomotion=locomotion["sha"]),
              "locomotion_selection": locomotion,
              "handoff_readiness": "150 consecutive original-strict AND3Dlinear<.06m/s AND3Dangular<.15rad/s; stronger separate readiness, no old gate change",
              "task": entry.TASK, "num_envs": 1, "pose": args.pose, "seed": args.seed,
              "control_dt": .02, "physics_dt": .005, "decimation": 4,
              "command_schedule": {"preparation": [0.,0.,0.], "recovery": [0.,0.,0.], "move": [.5,0.,0.], "stop": [0.,0.,0.]},
              "policy_action_mode": "deterministic_mean", "training_performed": False, "promotion_performed": False,
              "source_and_inputs": evidence["source_and_inputs"], "prerequisite": evidence["prerequisite"],
              "post_initialization_reset_allowed": False, "preparation_nominal_pd_intervals": 50,
              "behavior": None, "live_interface_continuity_passed": False, "video_view": args.view if args.video else None}
    pending = None
    try:
        cfg = load_cfg_from_registry(entry.TASK, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(entry.TASK, "rsl_rl_cfg_entry_point")
        loco_cfg = load_cfg_from_registry(retention.TASK, "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs, cfg.scene.env_spacing = 1, 2.
        cfg.episode_length_s = 28.  # 1 s preparation + at most11+8+6 s, plus2s margin.
        cfg.seed, cfg.sim.device = args.seed, args.device
        cfg.viewer.origin_type, cfg.viewer.asset_name = "world", "robot"
        cfg.viewer.resolution = (960, 540)
        cfg.viewer.eye, cfg.viewer.lookat = (2.3, 2.3, 1.35), (0., 0., .30)
        cmd = cfg.commands.base_velocity
        cmd.heading_command, cmd.rel_heading_envs, cmd.rel_standing_envs = False, 0., 0.
        cmd.resampling_time_range, cmd.debug_vis = (1e6, 1e6), False
        cmd.ranges.lin_vel_x = cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0., 0.)
        cmd.ranges.heading = None
        if cfg.events.push_robot is not None or cfg.events.base_external_force_torque is not None or cfg.terminations.base_contact is not None:
            raise RuntimeError("Unexpected physical pushes or base auto-reset in fixed Smith Play")
        if agent_cfg.clip_actions is not None or loco_cfg.clip_actions is not None:
            raise RuntimeError("Raw action clipping not allowed")
        physical_cfg = retention.physical_config(cfg)
        screened_interface = json.loads(Path(locomotion["interface_path"]).read_text())
        reference = screened_interface["config"]["after"]
        # Retain the actual pre-environment comparison even if a guard rejects it.
        # Path spelling is matched at launch; no physical field is normalized away.
        write_json(args.output_dir / "physical_configuration_comparison.json",
                   {"actual": physical_cfg, "screened_reference": reference})
        for name in ("robot", "terrain_physics_material", "decimation", "action", "events"):
            if physical_cfg[name] != reference[name]:
                raise RuntimeError("Not measured common recovery physics: " + name)
        for field in ("device", "render", "render_interval"):
            reference["sim"].pop(field, None)
        actual_sim = dict(physical_cfg["sim"])
        for field in ("device", "render", "render_interval"):
            actual_sim.pop(field, None)
        if actual_sim != reference["sim"]:
            raise RuntimeError("Unscreened simulation physics")
        write_json(args.output_dir / "actual_config.json", {"env": retention.serializable(cfg),
                   "recovery_agent": retention.serializable(agent_cfg), "locomotion_agent": retention.serializable(loco_cfg)})
        env = RslRlVecEnvWrapper(gym.make(entry.TASK, cfg=cfg, render_mode="rgb_array" if args.video else None), clip_actions=None)
        raw = env.unwrapped
        report["initialization"] = {"wrapper_initial_reset": True, "additional_env_reset": False,
                                    "placement_before_recording": True, "no_policy_history_reset_after_pd": True}
        runners = {}
        runners["roll"] = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runners["roll"].load(str(args.checkpoint), load_optimizer=False)
        device = torch.device(env.device)
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            for role, config, path in (("stand", agent_cfg, args.stand_checkpoint), ("locomotion", loco_cfg, locomotion["path"])):
                runners[role] = OnPolicyRunner(env, config.to_dict(), log_dir=None, device=args.device)
                runners[role].load(str(path), load_optimizer=False)
        policies = {role: runner.get_inference_policy(device=env.device) for role, runner in runners.items()}
        interfaces = {role: retention.validate_interface(raw, runner.alg.policy, "recovery") for role, runner in runners.items()}
        if any(value != screened_interface["interface"] for value in interfaces.values()):
            raise RuntimeError("Actual controller/mass/CoM/material interface differs from passed .5 screen")
        a, asset = raw.scene["robot"].data, raw.scene["robot"]
        report["mirror_joint_contract"] = validate_mirror_joint_contract(asset.joint_names, a.default_joint_pos, a.soft_joint_pos_limits, a.joint_pos_limits)
        report["physical_interface_before"] = interfaces
        initialized_config = retention.physical_config(cfg)
        report["physical_config_after_initialization"] = initialized_config
        if initialized_config != expected_initialized_config(physical_cfg, raw.scene.env_regex_ns):
            raise RuntimeError("Unexplained configuration change during environment initialization")
        sensor = raw.scene.sensors["contact_forces"]
        topology = contact_sensor_topology(sensor, asset)
        report["contact_sensor_topology"] = topology
        foot_ids, base_ids, foot_body_ids = topology["foot_ids"], topology["base_ids"], topology["foot_articulation_ids"]
        term = raw.command_manager.get_term("base_velocity")
        selection = None
        if args.pose == "upright":
            helpers["_set_pose_class"](raw, args.pose, torch.Generator(device=env.device).manual_seed(args.seed))
        else:
            bank = load_recovery_state_bank(entry.BANK, list(asset.joint_names), env.device, split="heldout")
            validate_bank_joint_limits(bank, a.soft_joint_pos_limits[0, :, 0], a.soft_joint_pos_limits[0, :, 1])
            helpers["_validate_bank_environment"](bank, raw)
            selected, selection = helpers["_select_bank_states"](bank, args.pose, 1, args.seed)
            helpers["_set_bank_states"](raw, bank, selected, transform_bank_root)
        set_command(term, flow.ZERO_COMMAND)
        if bool(raw.action_manager.action.any()) or bool(raw.action_manager.prev_action.any()):
            raise RuntimeError("Initial native action history not zero")
        release = helpers["_start_snapshot"](raw, foot_ids, contacts_fresh=False)
        if selection:
            helpers["_attach_bank_provenance"](release, selection)
        report["release_state"] = release
        report["bank_selection"] = selection
        guard = NoResetGuard(raw)
        origin = int(raw._sim_step_counter)
        report["simstep_origin"] = origin
        report["termination_terms"] = list(raw.termination_manager.active_terms)
        if set(raw.termination_manager.active_terms) != {"time_out"}:
            raise RuntimeError("Unexpected live termination term")
        trace_file = (args.output_dir / "full_trace.jsonl").open("x", encoding="utf-8")

        def record_frame(row):
            if not args.video:
                return
            helpers["_camera"](raw)
            rgb = raw.render()
            if rgb is None:
                raise RuntimeError("Missing actual camera frame")
            bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
            lines = ("CONTINUOUS .5 COMMAND DIAGNOSTIC / 3 ACTORS / NOT TROT-RUN",
                     f"t={row['time_s']:.2f}s {row['phase']} actor={row['actor_role']} view={args.view}",
                     f"height={row.get('height', 0.):.3f}m strict={row.get('strict_valid_after', False)} cmd={row.get('cmd_x', 0.):.2f}",
                     "Includes initial 1s nominal-PD preparation; no post-preparation reset")
            for index, text in enumerate(lines):
                cv2.putText(bgr, text, (12, 24 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX, .44, (245,245,245), 1, cv2.LINE_AA)
            writer.write(bgr)
            frames.append({"frame": len(frames), "time_s": row["time_s"], "completed_interval_step": row["step"],
                           "rendered_preencoding_bgr_sha256": hashlib.sha256(bgr.tobytes()).hexdigest()})

        if args.video:
            helpers["_diagnostic_scene"](raw)
            for _ in range(4):
                helpers["_camera"](raw)
                raw.render()
            if int(raw._sim_step_counter) != origin:
                raise RuntimeError("Render warmup stepped physics")
            writer = cv2.VideoWriter(str(args.output_dir / "continuous_flow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50, (960,540))
            if not writer.isOpened():
                raise RuntimeError("Cannot open video writer")
        write_json(args.output_dir / "initial_state.json", snapshot(raw))
        record_frame({"time_s": 0., "step": -1, "phase": "initialization_complete", "actor_role": "none"})

        previous_after = snapshot(raw)

        def append_interval(before, phase, number, role, detail, done=False):
            nonlocal previous_after
            after = snapshot(raw)
            index = physical_index(after["sim_step"], origin) - 1
            row = measurement(raw, helpers, foot_ids, base_ids, foot_body_ids, index, phase, number, role, done)
            row.update(simstep_before=before["sim_step"], simstep_after=after["sim_step"],
                       common_step_before=before["common_step"], common_step_after=after["common_step"],
                       episode_step_before=before["episode_step"], episode_step_after=after["episode_step"])
            rows.append(row)
            trace_file.write(json.dumps(safe_json({"row": row, "before": before, "after": after, "interface": detail}), allow_nan=False) + "\n")
            trace_file.flush()
            record_frame(row)
            if not unchanged_between_intervals(previous_after, before):
                raise RuntimeError("Physical/action/history state changed between completed intervals")
            previous_after = after
            assert_continuity(before, after, phase, origin)
            if not finite_evidence(before) or not finite_evidence(after) or not finite_evidence(row):
                raise RuntimeError("Nonfinite measured physical/interface state")
            if before["velocity_command_b"] != after["velocity_command_b"]:
                raise RuntimeError("Command manager changed command inside actual interval")
            return row

        quiet = torch.zeros(1, device=env.device, dtype=torch.int32)
        with torch.no_grad():
            for prep in range(50):
                before = snapshot(raw)
                pending = (before, "preparation", prep+1, "nominal_pd", {"controller": "direct_nominal_PD", "issued_policy_action": None})
                for _ in range(4):
                    raw._sim_step_counter += 1
                    asset.set_joint_position_target(a.default_joint_pos)
                    raw.scene.write_data_to_sim()
                    raw.sim.step(render=False)
                    raw.scene.update(dt=raw.physics_dt)
                raw.episode_length_buf += 1
                dones = raw.termination_manager.compute()
                row = append_interval(*pending, done=bool(dones.any()))
                pending = None
                if bool(dones.any()):
                    raise RuntimeError("Termination during recorded PD; no reset performed")
                if raw.action_manager.action.tolist() != before["actual_raw_action"] or raw.action_manager.prev_action.tolist() != before["actual_previous_raw_action"]:
                    raise RuntimeError("Direct PD changed native action history")
                if not torch.equal(a.joint_pos_target, a.default_joint_pos):
                    raise RuntimeError("Preparation target is not nominal")
                grounded = (sensor.data.net_forces_w.norm(dim=-1) > 1.).any(1)
                valid_quiet = grounded & (a.root_lin_vel_w.norm(dim=1) < .10) & (a.root_ang_vel_w.norm(dim=1) < .20) & (a.joint_vel.abs().amax(dim=1) < .50)
                quiet = torch.where(valid_quiet, quiet+1, torch.zeros_like(quiet))
            actual_start = helpers["_start_snapshot"](raw, foot_ids, contacts_fresh=True, quiet_steps=quiet)
            if selection:
                helpers["_attach_bank_provenance"](actual_start, selection)
            selected_start, startup_records = supported_startup_mask(actual_start)
            startup = torch.tensor(selected_start, device=env.device, dtype=torch.bool)
            eligible = torch.tensor([r["eligible_settled_fallen_recovery"] for r in actual_start], device=env.device, dtype=torch.bool)
            mirror = initial_right_mask(a.projected_gravity_b.clone(), eligible, "initial_right")
            if bool((startup & (mirror | eligible)).any()):
                raise RuntimeError("Startup/fallen/mirror overlap")
            if args.pose != "upright" and not bool(eligible[0]):
                raise RuntimeError("Requested bank start is not actually eligible settled-fallen; do not count as recovery")
            report.update(policy_start_state=actual_start, startup_selection=startup_records,
                          mirror_selected=mirror.tolist(), mirror_classified_once=True)
            state = flow.initial_state(flow.FlowConfig(.5, False), actual_control_step=physical_index(int(raw._sim_step_counter), origin))
            gate = torch.zeros(1, dtype=torch.int32, device=env.device)
            switched = torch.zeros(1, dtype=torch.bool, device=env.device)
            phase_previous, phase_interval, policy_steps = None, 0, 0
            while state.phase not in flow.TERMINAL:
                just_switched = torch.zeros_like(switched)
                if state.phase == "recovery":
                    if policy_steps > 0:
                        next_gate, next_switched, next_edge = update_handoff_gate(a.projected_gravity_b, a.root_ang_vel_b, gate, switched, .02)
                        gate = torch.where(startup, gate, next_gate)
                        switched = torch.where(startup, switched, next_switched)
                        just_switched = torch.where(startup, just_switched, next_edge)
                    if bool((startup & (switched | just_switched | (gate != 0))).any()):
                        raise RuntimeError("Startup mislabeled as genuine roll-to-stand gate")
                    plan = flow.next_action(state, recovery_stand_active=bool((startup | switched)[0]))
                else:
                    plan = flow.next_action(state)
                phase_interval = phase_interval + 1 if phase_previous == plan.phase else 1
                phase_previous = plan.phase
                set_command(term, plan.real_command)
                obs = env.get_observations()
                record = retention.begin_step(raw, obs, plan.absolute_control_step, "recovery")
                before = snapshot(raw)
                if not finite_evidence(before) or not bool(torch.isfinite(obs["policy"]).all()):
                    raise RuntimeError("Nonfinite actual pre-action state")
                native = obs["policy"].clone()
                detail = {"plan": asdict(plan), "state_before": asdict(state), "real_policy48": native[0].tolist(),
                          "old_gate_count": int(gate[0]), "old_gate_switched": bool(switched[0]),
                          "old_gate_just_switched": bool(just_switched[0]), "startup_selected": bool(startup[0]),
                          "mirror_selected": bool(mirror[0])}
                if plan.phase == "recovery":
                    roll_obs = roll_input_copy(obs, mirror)
                    model_action = policies["roll"](roll_obs)
                    roll_action = physical_roll_action(model_action, mirror)
                    stand_action = policies["stand"](obs)
                    action = torch.where((startup | switched)[:,None], stand_action, roll_action)
                    detail.update(roll_virtual48=roll_obs["policy"][0].tolist(), roll_model_action=model_action[0].tolist(),
                                  roll_physical_action=roll_action[0].tolist(), stand_action=stand_action[0].tolist())
                else:
                    action = policies["locomotion"](obs)
                if not torch.equal(native, obs["policy"]) or tuple(action.shape) != (1,12) or not bool(torch.isfinite(action).all()):
                    raise RuntimeError("Invalid actor output or mutated actual policy observation")
                detail["issued_raw_action"] = action[0].tolist()
                pending = (before, plan.phase, phase_interval, plan.actor_role, detail)
                _, _, dones, _ = env.step(action)
                # Persist the actual full interval before any assertion can abort it.
                detail["retention_interface"] = retention.end_step(raw, record, action, dones)
                row = append_interval(*pending, done=bool(dones.any()))
                pending = None
                measured = flow.CompletedInterval(plan.absolute_control_step,
                    physical_index(int(raw._sim_step_counter), origin), tuple(before["velocity_command_b"][0]),
                    tuple(term.command[0].tolist()), row["strict_valid_after"], row["fallen_after"], bool(dones.any()), True, row["quiet_for_handoff"])
                state = flow.after_completed_interval(state, plan, measured)
                policy_steps += 1
                if args.smoke and policy_steps == 2:
                    break
        report["flow_state"] = asdict(state)
        if state.phase == "failed":
            raise RuntimeError("Continuous sequence failed: " + str(state.failure_reason))
        if guard.attempts:
            raise RuntimeError("Reset was attempted")
        after_interface = {role: retention.validate_interface(raw, runner.alg.policy, "recovery") for role, runner in runners.items()}
        report["physical_interface_after"] = after_interface
        report["contact_sensor_topology_after"] = contact_sensor_topology(sensor, asset)
        if report["contact_sensor_topology_after"] != topology:
            raise RuntimeError("Sensor/articulation ordering changed during continuous episode")
        report["physical_config_after_episode"] = retention.physical_config(cfg)
        if interfaces != after_interface or report["physical_config_after_episode"] != initialized_config:
            raise RuntimeError("Physical/interface configuration changed during continuous episode")
        entry.verify_inventory(evidence["source_and_inputs"])
        report["live_interface_continuity_passed"] = True
        if args.smoke:
            if len(rows) != 52 or state.completed_intervals != 2 or len([r for r in rows if r["phase"] == "preparation"]) != 50:
                raise RuntimeError("Smoke did not complete its exact50+2 measured intervals")
            report["status"] = "smoke_complete_not_full_flow"
        else:
            if state.phase != "complete":
                raise RuntimeError("Full continuous schedule did not complete")
            report["behavior"] = analyze(rows, move_speed=.5)
            report["passed"] = report["behavior"]["motion_checks_passed"] is True
            report["status"] = "diagnostic_passed_not_promoted" if report["passed"] else "behavior_failed"
    except Exception as exc:
        failure = exc
        report["status"], report["passed"] = "failed", False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if state is not None:
            report["flow_state"] = asdict(state)
        if env is not None:
            try:
                raw = env.unwrapped
                report["failure_physical_state"] = snapshot(raw)
                if pending is not None and int(raw._sim_step_counter) == pending[0]["sim_step"] + 4:
                    if not rows or rows[-1]["simstep_after"] != int(raw._sim_step_counter):
                        append_interval(*pending, done=bool(raw.termination_manager.dones.any()))
            except Exception as record_error:
                report["failure_record_error"] = str(record_error)
    finally:
        if trace_file is not None:
            trace_file.close()
        if writer is not None:
            writer.release()
        report["reset_guard_attempts"] = [] if guard is None else guard.attempts
        report["actual_completed_intervals"] = len(rows)
        report["frame_count_written"] = len(frames)
        try:
            entry.verify_inventory(evidence["source_and_inputs"])
            report["source_hashes_unchanged_at_end"] = True
        except Exception as source_error:
            failure = failure or source_error
            report.update(passed=False, status="failed", source_hashes_unchanged_at_end=False,
                          source_verification_error=str(source_error))
        write_json(args.output_dir / "measured_rows.json", rows)
        write_json(args.output_dir / "video_frames.json", frames)
        if rows:
            with (args.output_dir / "continuous_flow.csv").open("x", encoding="utf-8", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                csv_writer.writeheader()
                for row in rows:
                    csv_writer.writerow({k: json.dumps(v) if isinstance(v, (list,dict)) else v for k,v in row.items()})
        report["artifact_sha256"] = {p.name: entry.sha(p) for p in args.output_dir.iterdir() if p.is_file() and p.name != "simulator.log"}
        write_json(args.output_dir / "continuous_flow_report.json", report)
        if guard is not None:
            guard.restore_for_close()
        if env is not None:
            env.close()
    print(json.dumps({"status": report["status"], "passed": report["passed"], "rows": len(rows), "output": str(args.output_dir)}))
    if failure is not None:
        raise RuntimeError("Continuous diagnostic failed; complete available evidence saved") from failure
