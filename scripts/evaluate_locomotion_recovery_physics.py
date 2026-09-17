"""Frozen control3947 retention screen, with explicit original/recovery physics.

The old evaluator is SHA pinned and transformed in memory. Original mode changes
no configuration, action, observation, reset or acceptance; added evidence is
read-only. Recovery mode changes only self collision, the nominal soft-clamped
action and fixed randomization to the SmithNominal reference. It deliberately
keeps the original upright reset (Bank reset is incompatible with this screen).
This is not a recovery/locomotion integration or a trained candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/evaluate_go2_stand_walk_stop.py"
TEMPLATE_SHA = "bbba369177d95bc24163bdd2b5ba96506f5db7d4b1ce38875f8a8885c49e22e4"
CONTROL = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/2026-09-17_03-10-21_20260917-speedretention128x300/model_3947.pt")
CONTROL_SHA = "3dfbe03e0ac332fee688c970b420167affcdb6e8cacee16e40fc12120751df3f"
TASK = "Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0"
REFERENCE_TASK = "Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0"
PROTOCOL = "locomotion_recovery_physics_retention_v1"
OBS_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
NATIVE_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR"))
DEFAULT_Q = (.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5)
FROZEN_SOURCES = {
    "recovery_env_cfg.py": "45836efa79a4aded4b9a8cb3f46ca5ac0a0d9c9a0f7d4609884ff65059a7865f",
    "recovery_control_targets.py": "5220033d6b0f76bc313527dce017bb37f67f19a192034a15ac2a9eb7abd6ecac",
}
INSTALLED = Path("E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Frozen anchor count is not one: {old[:100]!r}")
    return source.replace(old, new, 1)


def validate_cli(args):
    if args.task != TASK or args.zero_action:
        raise ValueError("This isolated screen requires frozen control3947 and NaturalStop Play")
    if args.checkpoint is None or sha(args.checkpoint) != CONTROL_SHA:
        raise ValueError("Checkpoint is not the frozen original-distribution control3947")
    if (args.stand_s, args.walk_s, args.stop_s) != (4., 8., 6.):
        raise ValueError("The retained protocol is exactly 4 s stand / 8 s walk / 6 s stop")
    if not all(math.isfinite(v) for v in (args.walk_speed, args.lateral_speed, args.yaw_rate, args.push_speed)):
        raise ValueError("Commands and push must be finite")
    if args.physics_mode not in ("original", "recovery"):
        raise ValueError("An explicit physics mode is required")
    if args.output_dir.resolve().drive.upper() != "E:":
        raise ValueError("All new outputs must stay on E:")
    for name in ("retention_generated.py", "retention_interface.json", f"{args.checkpoint.stem}_stand_walk_stop.json",
                 f"{args.checkpoint.stem}_stand_walk_stop.csv", f"{args.checkpoint.stem}_stand_walk_stop.mp4"):
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)


def serializable(value):
    """Stable config evidence; no repr addresses or simulator imports."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"configuration_nonfinite_number": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, slice):
        return {"configuration_slice": [value.start, value.stop, value.step]}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "to_dict"):
        return serializable(value.to_dict())
    raise TypeError(f"Unsupported config evidence type: {type(value)}")


def physical_config(cfg):
    return serializable({
        "robot": cfg.scene.robot, "sim": cfg.sim,
        "terrain_physics_material": cfg.scene.terrain.physics_material,
        "decimation": cfg.decimation, "action": cfg.actions.joint_pos,
        "events": {name: getattr(cfg.events, name) for name in
                   ("add_base_mass", "base_com", "physics_material", "push_robot", "base_external_force_torque")},
        "reset_base": cfg.events.reset_base, "reset_robot_joints": cfg.events.reset_robot_joints,
    })


def configure_physics(cfg, mode, reference=None):
    """Mutate only an explicit recovery copy; reject unsupported reset topology."""
    if mode not in ("original", "recovery"):
        raise ValueError("Unknown physics mode")
    if cfg.events.reset_base is None or cfg.events.reset_robot_joints is None:
        raise ValueError("This screen must keep both ordinary upright reset terms")
    if cfg.events.reset_base.func.__name__ != "reset_root_state_uniform" or cfg.events.reset_robot_joints.func.__name__ != "reset_joints_by_scale":
        raise ValueError("The original diagnostic reset functions were not installed")
    before = physical_config(cfg)
    if mode == "original":
        return {"before": before, "after": physical_config(cfg), "reference_checked": False,
                "reset_protocol": "Original upright reset; no bank states, settling or additional history reset"}
    if reference is None:
        raise ValueError("Recovery mode requires an actual SmithNominal configuration")
    ref = physical_config(reference)
    if reference.events.reset_robot_joints is not None or reference.events.reset_base.func.__name__ != "RecoveryBankReset":
        raise ValueError("Smith reference reset topology changed; never copy it into this screen")
    if reference.events.add_base_mass is not None or reference.events.base_com is not None:
        raise ValueError("Smith reference has unexpected mass/CoM randomization")
    if not reference.scene.robot.spawn.articulation_props.enabled_self_collisions:
        raise ValueError("Smith reference must use self collisions")
    for name, expected in (("static_friction", .8), ("dynamic_friction", .6), ("restitution", 0.)):
        if tuple(reference.events.physics_material.params[name + "_range"]) != (expected, expected):
            raise ValueError("Smith reference material changed")
    action = reference.actions.joint_pos
    if (action.class_type.__name__ != "ControlStepJointPositionAction" or action.reference != "nominal"
            or action.scale != .25 or action.offset != 0. or action.use_default_offset or action.clip is not None):
        raise ValueError("Smith reference action semantics changed")
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.actions.joint_pos = copy.deepcopy(action)
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.physics_material = copy.deepcopy(reference.events.physics_material)
    after = physical_config(cfg)
    for field in ("robot", "terrain_physics_material", "decimation", "action", "events"):
        if after[field] != ref[field]:
            raise ValueError(f"Recovery/reference physical config differs: {field}")
    # Viewer/device are runtime-only; all dynamics-bearing simulation fields stay exact.
    actual_sim, reference_sim = copy.deepcopy(after["sim"]), copy.deepcopy(ref["sim"])
    for field in ("device", "render", "render_interval"):
        actual_sim.pop(field, None)
        reference_sim.pop(field, None)
    if actual_sim != reference_sim:
        raise ValueError("Recovery/reference simulation dynamics config differs")
    for field in ("reset_base", "reset_robot_joints"):
        if after[field] != before[field]:
            raise ValueError("Physics adapter changed the original upright reset")
    return {"before": before, "after": after, "reference_checked": True,
            "reference": ref, "reference_task": REFERENCE_TASK,
            "reset_protocol": "Original upright reset retained, NOT Smith bank reset; no added settle/history reset"}


def configure_runtime(cfg, mode, load_cfg):
    for name, expected in FROZEN_SOURCES.items():
        for path in (ROOT / "src/go2_recovery" / name, INSTALLED / name):
            if sha(path) != expected:
                raise ValueError(f"Frozen reference source changed: {path}")
    reference = None
    if mode == "recovery":
        # Only instantiate the reference config. No bank, environment, reset or actor is run.
        previous = os.environ.get("ISAACLAB_RECOVERY_BANK_COLLECTION")
        os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = "1"
        try:
            reference = load_cfg(REFERENCE_TASK, "env_cfg_entry_point")
        finally:
            if previous is None:
                del os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"]
            else:
                os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = previous
    return configure_physics(cfg, mode, reference)


def validate_interface(env, actor, mode):
    import torch
    manager, cfg = env.observation_manager, env.cfg.observations.policy
    if tuple(manager.active_terms["policy"]) != OBS_TERMS or tuple(manager.group_obs_dim["policy"]) != (48,):
        raise RuntimeError("Native seven-term policy48 observation layout changed")
    if list(manager.group_obs_term_dim["policy"]) != [(3,), (3,), (3,), (3,), (12,), (12,), (12,)]:
        raise RuntimeError("Policy48 term dimensions changed")
    if cfg.enable_corruption or not cfg.concatenate_terms:
        raise RuntimeError("Require uncorrupted concatenated native observations")
    funcs = ("base_lin_vel", "base_ang_vel", "projected_gravity", "generated_commands", "joint_pos_rel", "joint_vel_rel", "last_action")
    for name, expected in zip(OBS_TERMS, funcs):
        term = getattr(cfg, name)
        if term.func.__name__ != expected or term.clip is not None or term.modifiers or term.history_length != 0:
            raise RuntimeError(f"Observation semantics changed: {name}")
        if term.scale is not None and not bool(torch.as_tensor(term.scale).eq(1).all()):
            raise RuntimeError(f"Observation scaling changed: {name}")
    if actor.actor_obs_normalization or not isinstance(actor.actor_obs_normalizer, torch.nn.Identity):
        raise RuntimeError("Frozen control requires identity observation normalization")
    if tuple(actor.obs_groups["policy"]) != ("policy",) or actor.is_recurrent:
        raise RuntimeError("Require unchanged feed-forward actor consuming real policy48")
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    if tuple(robot.joint_names) != NATIVE_JOINT_NAMES or tuple(term._joint_names) != NATIVE_JOINT_NAMES:
        raise RuntimeError("Native joint/action order changed")
    data = robot.data
    if not torch.equal(data.default_joint_pos[0], data.default_joint_pos.new_tensor(DEFAULT_Q)) or bool(data.default_joint_vel.any()):
        raise RuntimeError("Nominal joint pose or velocity changed")
    expected_class = "ControlStepJointPositionAction" if mode == "recovery" else "JointPositionAction"
    if type(term).__name__ != expected_class or term.cfg.scale != .25 or term.cfg.clip is not None:
        raise RuntimeError("Action implementation/scale/clipping changed")
    if mode == "original" and (not term.cfg.use_default_offset or not torch.equal(term._offset, data.default_joint_pos)):
        raise RuntimeError("Original action must use default-pose offset")
    if mode == "recovery" and (term.cfg.reference != "nominal" or term.cfg.use_default_offset or term.cfg.offset != 0.):
        raise RuntimeError("Recovery action is not nominal soft clamp")
    if env.step_dt != .02 or env.cfg.decimation != 4 or env.physics_dt != .005:
        raise RuntimeError("Require original 50 Hz control / 200 Hz physics")
    view = robot.root_physx_view
    return {"observation_terms": list(OBS_TERMS), "observation_dim": 48,
            "native_joint_names": list(robot.joint_names), "default_joint_positions": data.default_joint_pos[0].tolist(),
            "default_joint_velocities": data.default_joint_vel[0].tolist(),
            "soft_joint_limits": data.soft_joint_pos_limits[0].tolist(),
            "hard_joint_limits": data.joint_pos_limits[0].tolist(),
            "action_class": type(term).__name__, "scale": .25,
            "action_formula": "soft_clamp(default_q + .25 * real_raw_action)" if mode == "recovery" else "default_q + .25 * real_raw_action",
            "history": "Native actual action manager action/prev_action only; checked before/after every step; never reset by adapter",
            "actor_normalization": "identity", "step_dt": env.step_dt, "physics_dt": env.physics_dt,
            "body_names": list(robot.body_names), "actual_masses": view.get_masses().tolist(),
            "actual_coms": view.get_coms().tolist(), "actual_material_properties": view.get_material_properties().tolist()}


def begin_step(env, obs, step, mode):
    """Inspect before policy inference, without replacing inputs or changing buffers."""
    import torch
    a, manager = env.scene["robot"].data, env.action_manager
    measured = torch.cat((a.root_lin_vel_b, a.root_ang_vel_b, a.projected_gravity_b,
                          env.command_manager.get_command("base_velocity"), a.joint_pos - a.default_joint_pos,
                          a.joint_vel - a.default_joint_vel, manager.action), dim=-1)
    if not torch.equal(obs["policy"], measured) or tuple(measured.shape) != (1, 48):
        raise RuntimeError("Actual policy48 differs from real native measurements/history")
    if step == 0 and (bool(manager.action.any()) or bool(manager.prev_action.any())):
        raise RuntimeError("Original environment reset did not provide zero initial action history")
    return {"step": step, "physics_mode": mode, "observation": obs["policy"].clone(),
            "previous_raw_action": manager.action.clone(), "previous_previous_raw_action": manager.prev_action.clone(),
            "previous_target": a.joint_pos_target.clone(), "joint_pos": a.joint_pos.clone(), "joint_vel": a.joint_vel.clone()}


def end_step(env, record, action, dones):
    """Exact read-after-step checks; any auto-reset invalidates retention evidence."""
    import torch
    a, manager = env.scene["robot"].data, env.action_manager
    if tuple(action.shape) != (1, 12) or not bool(torch.isfinite(action).all()):
        raise RuntimeError("Actor produced invalid native action12")
    if bool(dones.any()):
        raise RuntimeError("Unexpected auto-reset: no valid uninterrupted retention evidence")
    if not torch.equal(manager.action, action) or not torch.equal(manager.prev_action, record["previous_raw_action"]):
        raise RuntimeError("Actual execution/history differs from unchanged policy action")
    expected = a.default_joint_pos + action * .25
    if record["physics_mode"] == "recovery":
        limits = a.soft_joint_pos_limits
        expected = torch.clamp(expected, min=limits[:, :, 0], max=limits[:, :, 1])
    if not torch.equal(a.joint_pos_target, expected):
        raise RuntimeError("Executed position target differs from declared original/recovery semantics")
    record.update(raw_action=action.clone(), actual_action=manager.action.clone(), actual_prev_action=manager.prev_action.clone(),
                  executed_target=a.joint_pos_target.clone(), expected_target=expected, done=dones.clone())
    return {key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value for key, value in record.items()}


def straight_drift_acceptance(walk, lateral_speed, yaw_rate, push_speed):
    """Keep the old chain's independent gate for straight unperturbed cases."""
    if lateral_speed != 0. or yaw_rate != 0. or push_speed != 0.:
        return None
    return abs(walk["vy_b_mean"]) < .12 and abs(walk["yaw_rate_mean"]) < .15


def finish_report(report, args, config, interface, records, generated):
    if len(records) != 900 or report["global"]["steps"] != 900:
        raise RuntimeError("Incomplete 18 s / 900-step retention evidence")
    trace_path = args.output_dir / "retention_interface.json"
    trace = {"schema": PROTOCOL, "physics_mode": args.physics_mode, "config": config,
             "interface": interface, "steps": records}
    with trace_path.open("x", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, allow_nan=False)
    with (args.output_dir / "retention_generated.py").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(generated)
    report["baseline_metric_protocol_version"] = report["protocol_version"]
    report["protocol_version"] = PROTOCOL
    # Existing acceptance/passed stay byte-for-value unchanged. The old chain
    # reported straightness separately; retain that additional gate explicitly.
    drift_ok = straight_drift_acceptance(report["settled_phase_stats"]["walk"],
                                         args.lateral_speed, args.yaw_rate, args.push_speed)
    report["retention_acceptance"] = {"original_checks": report["passed"], "straight_drift": drift_ok}
    report["retention_passed"] = report["passed"] and drift_ok is not False
    report["retention_experiment"] = {"physics_mode": args.physics_mode, "frozen_control_sha256": CONTROL_SHA,
        "template_sha256": TEMPLATE_SHA, "adapter_sha256": sha(__file__),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "trace_path": str(trace_path.resolve()), "trace_sha256": sha(trace_path),
        "source_hashes": FROZEN_SOURCES, "read_only_interface_steps": len(records),
        "no_added_reset_or_history_write": True, "reference_checked": config["reference_checked"],
        "claim": "Frozen locomotion retention only; not fallen recovery, fast running or uninterrupted full-flow acceptance"}


def build_source():
    if sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError("Original retention evaluator changed")
    source = TEMPLATE.read_text(encoding="utf-8")
    edits = [
        ('parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")',
         'parser.add_argument("--task", default=_retention_adapter.TASK)\nparser.add_argument("--physics_mode", required=True, choices=("original", "recovery"))'),
        ('args_cli = parser.parse_args()', 'args_cli = parser.parse_args()\n_retention_adapter.validate_cli(args_cli)'),
        ('    _configure_diagnostic(env_cfg, total_s)',
         '    _configure_diagnostic(env_cfg, total_s)\n    retention_config = _retention_adapter.configure_runtime(env_cfg, args_cli.physics_mode, load_cfg_from_registry)'),
        ('    dt = env.unwrapped.step_dt',
         '    retention_interface = _retention_adapter.validate_interface(env.unwrapped, policy_nn, args_cli.physics_mode)\n    retention_records = []\n    dt = env.unwrapped.step_dt'),
        ('            with torch.no_grad():\n                obs, _, dones, _ = env.step(policy(obs))',
         '''            retention_record = _retention_adapter.begin_step(env.unwrapped, obs, step, args_cli.physics_mode)
            with torch.no_grad():
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
            retention_records.append(_retention_adapter.end_step(env.unwrapped, retention_record, action, dones))'''),
        ('    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")',
         '''    _retention_adapter.finish_report(report, args_cli, retention_config, retention_interface, retention_records, _retention_generated_source)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")'''),
    ]
    for old, new in edits:
        source = replace_once(source, old, new)
    compile(source, str(TEMPLATE) + "::<retention-adapter>", "exec")
    return source


if __name__ == "__main__":
    source = build_source()
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
                 "_retention_adapter": sys.modules[__name__], "_retention_generated_source": source}
    exec(compile(source, str(Path(__file__).resolve()) + "::<generated>", "exec"), namespace)
