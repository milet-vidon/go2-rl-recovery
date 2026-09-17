"""Three single-factor physics diagnostics, NOT complete recovery physics.

Frozen control3947; exactly vx=.8, seed20260909, 4/8/6 seconds. All original
acceptance plus the existing independent straight-drift check remain unchanged.
Only new files are written. This entry neither trains nor integrates recovery.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import evaluate_locomotion_recovery_physics as retained


ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = "cc2437ffb0d84da97eda5906333e54070f2d065ea475a148b6be7f1c6cfea4c8"
PROTOCOL = "locomotion_single_factor_physics_ablation_v1"
TASK, CONTROL, CONTROL_SHA = retained.TASK, retained.CONTROL, retained.CONTROL_SHA
VARIANTS = ("self_collision", "soft_clamp", "fixed_randomization")
ACTION_SEMANTICS = {
    "self_collision": "original_nominal_unclamped",
    "soft_clamp": "nominal_soft_clamped",
    "fixed_randomization": "original_nominal_unclamped",
}
CHANGES = {
    "self_collision": ["robot.spawn.articulation_props.enabled_self_collisions"],
    "soft_clamp": ["action"],
    "fixed_randomization": ["events.add_base_mass", "events.base_com", "events.physics_material"],
}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_helper():
    require(retained.sha(retained.__file__) == HELPER_SHA, "Reviewed retention helper changed")


def action_mode(variant):
    require(variant in VARIANTS, "Unknown isolated physics variant")
    # This string selects a private helper branch only; it is never serialized
    # as the experiment's physics label. Partial variants are not recovery mode.
    return "recovery" if variant == "soft_clamp" else "original"


def validate_cli(args):
    verify_helper()
    require(args.task == TASK and not args.zero_action, "Require frozen NaturalStop Play/control3947")
    require(args.checkpoint is not None and retained.sha(args.checkpoint) == CONTROL_SHA,
            "Frozen control3947 checkpoint changed")
    require(args.physics_mode in VARIANTS, "Only the three explicit isolated variants are permitted")
    require((args.stand_s, args.walk_s, args.stop_s) == (4., 8., 6.), "Use unchanged 4/8/6 s phases")
    require((args.walk_speed, args.lateral_speed, args.yaw_rate, args.push_speed) == (.8, 0., 0., 0.),
            "This preregistered diagnostic is exactly vx=.8, no lateral/yaw command or push")
    require(type(args.seed) is int and args.seed == 20260909, "Use the fixed development seed20260909")
    require(args.output_dir.resolve().drive.upper() == "E:", "All outputs stay on E:")
    for name in ("ablation_generated.py", "ablation_interface.json", "model_3947_stand_walk_stop.json",
                 "model_3947_stand_walk_stop.csv", "model_3947_stand_walk_stop.mp4"):
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)


def configure_physics(cfg, variant, reference):
    require(variant in VARIANTS, "Unknown isolated physics variant")
    before = retained.physical_config(cfg)
    # Reuse the already checked reference contract on a disposable config copy.
    # This validates shared physics and the full Smith reference without applying
    # its complete changes to the actual experiment or mutating the reference.
    reference_before = retained.physical_config(reference)
    probe = retained.configure_physics(copy.deepcopy(cfg), "recovery", reference)
    require(reference_before == retained.physical_config(reference), "Reference unexpectedly mutated")
    require(before["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] is False,
            "Single-factor baseline must start with original self-collisions OFF")
    require(cfg.actions.joint_pos.class_type.__name__ == "JointPositionAction" and
            cfg.actions.joint_pos.use_default_offset and cfg.actions.joint_pos.scale == .25 and
            cfg.actions.joint_pos.clip is None, "Original action contract changed")
    expected = copy.deepcopy(before)
    if variant == "self_collision":
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        expected["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] = True
    elif variant == "soft_clamp":
        cfg.actions.joint_pos = copy.deepcopy(reference.actions.joint_pos)
        expected["action"] = copy.deepcopy(reference_before["action"])
    else:
        cfg.events.add_base_mass = None
        cfg.events.base_com = None
        cfg.events.physics_material = copy.deepcopy(reference.events.physics_material)
        for key in ("add_base_mass", "base_com", "physics_material"):
            expected["events"][key] = copy.deepcopy(reference_before["events"][key])
    after = retained.physical_config(cfg)
    require(after == expected, "Undeclared extra physics change in isolated variant")
    require(after != before, "Chosen factor did not change the baseline")
    return {"physics_variant": variant, "action_semantics": ACTION_SEMANTICS[variant],
            "before": before, "after": after, "reference_checked": probe["reference_checked"],
            "full_recovery_reference": reference_before,
            "full_recovery_physics": False, "single_factor_relative_to_original": True,
            "declared_changed_fields": CHANGES[variant],
            "reset_protocol": "Original upright reset retained exactly; no bank states, extra settling, or history reset",
            "factor_scope": "fixed_randomization is one grouped mass/CoM/material intervention, not three independently isolated factors"}


def configure_runtime(cfg, variant, load_cfg):
    verify_helper()
    # Check all existing local/installed source pins with no mutation.
    retained.configure_runtime(cfg, "original", load_cfg)
    previous = os.environ.get("ISAACLAB_RECOVERY_BANK_COLLECTION")
    os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = "1"
    try:
        reference = load_cfg(retained.REFERENCE_TASK, "env_cfg_entry_point")
    finally:
        if previous is None:
            del os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"]
        else:
            os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = previous
    return configure_physics(cfg, variant, reference)


def validate_interface(env, actor, variant):
    result = retained.validate_interface(env, actor, action_mode(variant))
    result.update(physics_variant=variant, action_semantics=ACTION_SEMANTICS[variant],
                  full_recovery_physics=False)
    return result


def begin_step(env, obs, step, variant):
    record = retained.begin_step(env, obs, step, action_mode(variant))
    del record["physics_mode"]
    record.update(physics_variant=variant, action_semantics=ACTION_SEMANTICS[variant])
    return record


def end_step(env, record, action, dones):
    variant = record["physics_variant"]
    require(record["action_semantics"] == ACTION_SEMANTICS[variant], "Action semantic label changed")
    internal = dict(record, physics_mode=action_mode(variant))
    result = retained.end_step(env, internal, action, dones)
    del result["physics_mode"]
    return result


def finish_report(report, args, config, interface, records, generated):
    require(len(records) == 900 and report["global"]["steps"] == 900, "Incomplete 18 s/900-step evidence")
    require(all(row["physics_variant"] == args.physics_mode and
                row["action_semantics"] == ACTION_SEMANTICS[args.physics_mode] and
                "physics_mode" not in row for row in records), "Mislabeled single-factor trace")
    trace_path = args.output_dir / "ablation_interface.json"
    trace = {"schema": PROTOCOL, "physics_variant": args.physics_mode,
             "action_semantics": ACTION_SEMANTICS[args.physics_mode], "full_recovery_physics": False,
             "config": config, "interface": interface, "steps": records}
    with trace_path.open("x", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, allow_nan=False)
    with (args.output_dir / "ablation_generated.py").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(generated)
    report["baseline_metric_protocol_version"] = report["protocol_version"]
    report["protocol_version"] = PROTOCOL
    drift_ok = retained.straight_drift_acceptance(report["settled_phase_stats"]["walk"],
                                                 args.lateral_speed, args.yaw_rate, args.push_speed)
    report["retention_acceptance"] = {"original_checks": report["passed"], "straight_drift": drift_ok}
    report["retention_passed"] = report["passed"] and drift_ok is not False
    report["ablation_experiment"] = {
        "physics_variant": args.physics_mode, "action_semantics": ACTION_SEMANTICS[args.physics_mode],
        "full_recovery_physics": False, "single_factor_relative_to_original": True,
        "declared_changed_fields": CHANGES[args.physics_mode],
        "frozen_control_sha256": CONTROL_SHA, "template_sha256": retained.TEMPLATE_SHA,
        "retention_helper_sha256": HELPER_SHA, "adapter_sha256": retained.sha(__file__),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "trace_path": str(trace_path.resolve()), "trace_sha256": retained.sha(trace_path),
        "source_hashes": retained.FROZEN_SOURCES, "read_only_interface_steps": len(records),
        "no_added_reset_or_history_write": True, "reference_checked": config["reference_checked"],
        "training_performed": False, "promotion_performed": False,
        "claim": "Single-factor frozen-policy cause isolation only. Not complete recovery physics, continuous integration, independent-seed generalization, or fast-running acceptance."}


def build_source():
    verify_helper()
    source = retained.build_source()
    source = retained.replace_once(source,
        'parser.add_argument("--physics_mode", required=True, choices=("original", "recovery"))',
        'parser.add_argument("--physics_mode", required=True, choices=("self_collision", "soft_clamp", "fixed_randomization"))')
    compile(source, str(__file__) + "::<single-factor>", "exec")
    return source


if __name__ == "__main__":
    source = build_source()
    exec(compile(source, str(Path(__file__).resolve()) + "::<generated>", "exec"),
         {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
          "_retention_adapter": sys.modules[__name__], "_retention_generated_source": source})
