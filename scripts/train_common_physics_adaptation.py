"""Finite full-state control3947 continuation in the combined recovery physics.

Keep the original TRAIN commands/rewards/noise/upright reset/push/PPO. Change
only self-collisions, nominal soft-clamped targets, and fixed mass/CoM/material.
Smoke16x2 and formal128x100 independently resume3947. No sampler/gait changes,
automatic continuation, promotion, hardware execution or fast-running claim.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from check_speed_tracking_weight_block import (
    ROOT, RUN_ROOT, PARENT, PARENT_CONFIG, PARENT_RUN, PARENT_SHA,
    provenance, parent_metadata, checkpoint_metadata, number_is, read, sha, require,
)
from check_smith_standweight_config import differences, _log_path


PROTOCOL = "bounded_common_physics_adaptation_training_v1"
TASK = "Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0"
OFFICIAL = Path("E:/IsaacLab/repo/scripts/reinforcement_learning/rsl_rl/train.py")
OFFICIAL_SHA = "7586436b3318c1eb705600aef07bdd54f55d5c35943b266563972adc0c7a22cb"
REFERENCE_ENV = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_handoff_stand/20260917-handoff-stand128x100/params/env.yaml")
REFERENCE_ENV_SHA = "929f1644ffc694c70ea093e232a158e417c289333a591d5ff1757a8805584206"
EXTRA_FROZEN = {
    Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery/2026-09-16_19-02-40_20260916-smithnominal128x2000/model_1999.pt"):
        "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c",
    Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_handoff_stand/20260917-handoff-stand128x100/model_3547.pt"):
        "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb",
}
GUARD_PINS = {
    ROOT / "scripts/check_speed_tracking_weight_block.py": "2fc3d87e69f73ecc6110aa9d8cd396e016a54983f6021016cfca1850b5fb73f7",
    ROOT / "scripts/check_smith_standweight_config.py": "79f572ee2b1d9bc86864734dd6ecb4dfc2cc4acefc040de6a961f3340ce3b581",
}
BUDGETS = {(16, 2), (128, 100)}
PHYSICAL_FIELDS = ["scene.robot.spawn.articulation_props.enabled_self_collisions", "actions.joint_pos",
                   "events.add_base_mass", "events.base_com", "events.physics_material"]
STATE = {}


def arguments_ok(num_envs, updates, run_name):
    require(type(num_envs) is int and type(updates) is int and (num_envs, updates) in BUDGETS,
            "Only independent smoke16x2 or one formal128x100 block")
    require(isinstance(run_name, str) and re.fullmatch(r"20260917-commonphysics[A-Za-z0-9_-]+", run_name),
            "Use a unique 20260917-commonphysics... run tag")


def reference_document():
    require(sha(REFERENCE_ENV) == REFERENCE_ENV_SHA, "Frozen nominal-physics YAML changed")
    reference = read(REFERENCE_ENV)
    action = reference["actions"]["joint_pos"]
    require(action["class_type"].endswith("recovery_control_targets:ControlStepJointPositionAction") and
            action["reference"] == "nominal" and action["scale"] == "0.25" and
            action["offset"] == "0.0" and action["use_default_offset"] == "false" and action["clip"] == "null",
            "Reference action semantics changed")
    require(reference["scene"]["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] == "true",
            "Reference self-collisions must be on")
    require(reference["events"]["add_base_mass"] == reference["events"]["base_com"] == "null",
            "Reference mass/CoM must be fixed")
    material = reference["events"]["physics_material"]["params"]
    for key, values in (("static_friction_range", ["0.8", "0.8"]),
                        ("dynamic_friction_range", ["0.6", "0.6"]), ("restitution_range", ["0.0", "0.0"])):
        require(material[key] == values, "Reference material changed")
    return reference


def expected_environment(parent, candidate, reference, num_envs):
    expected = copy.deepcopy(parent)
    expected["scene"]["num_envs"] = expected["scene"]["terrain"]["num_envs"] = str(num_envs)
    _log_path(candidate["log_dir"], "env.log_dir")
    _log_path(candidate["sim"]["log_dir"], "env.sim.log_dir", allow_null=True)
    expected["log_dir"], expected["sim"]["log_dir"] = candidate["log_dir"], candidate["sim"]["log_dir"]
    expected["scene"]["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] = "true"
    expected["actions"]["joint_pos"] = copy.deepcopy(reference["actions"]["joint_pos"])
    for key in ("add_base_mass", "base_com", "physics_material"):
        expected["events"][key] = copy.deepcopy(reference["events"][key])
    return expected


def actual_yaml_changes(parent, actual):
    """Record actual differences, not merely a broad allowlist of intended edits."""
    result = []
    for path, before, after in differences(parent, actual, "env"):
        relative = path.removeprefix("env.")
        physical = any(relative == field or relative.startswith(field + ".") or
                       relative.startswith(field + "[") for field in PHYSICAL_FIELDS)
        result.append({"path": relative, "before": before, "after": after,
                       "kind": "declared_physics" if physical else "run_metadata"})
    return result


def check_documents(env, agent, parent_env, parent_agent, reference, num_envs, updates, run_name):
    arguments_ok(num_envs, updates, run_name)
    require(parent_agent["run_name"] == "20260917-speedretention128x300" and
            parent_agent["max_iterations"] == "300" and parent_agent["seed"] == "42" and
            parent_agent["resume"] == "true" and parent_agent["load_checkpoint"] == "model_3648.pt",
            "Wrong original parent TRAIN configuration")
    command = parent_env["commands"]["base_velocity"]
    require(command["ranges"]["lin_vel_x"] == ["-0.5", "1.0"] and command["rel_standing_envs"] == "0.3",
            "Parent moving/standing command distribution changed")
    require(parent_env["observations"]["policy"]["enable_corruption"] == "true" and
            parent_env["events"]["push_robot"] != "null", "Require original noisy/pushed TRAIN configuration, not Play")
    expected = expected_environment(parent_env, env, reference, num_envs)
    delta = differences(expected, env, "env")
    require(not delta, "Undeclared reward/command/reset/noise/push/physics change: " + str(delta))
    expected_agent = copy.deepcopy(parent_agent)
    expected_agent.update(run_name=run_name, max_iterations=str(updates), load_run=PARENT_RUN,
                          load_checkpoint="model_3947.pt")
    rate = agent["algorithm"]["learning_rate"]
    require(number_is(rate, Decimal("0.00001")), "Initial scalar LR must match actual parent LR1e-5")
    expected_agent["algorithm"]["learning_rate"] = rate
    delta = differences(expected_agent, agent, "agent")
    require(not delta, "Undeclared PPO/seed/resume change: " + str(delta))
    changes = actual_yaml_changes(parent_env, env)
    def nested(document, dotted):
        for key in dotted.split("."):
            document = document[key]
        return document
    unchanged = [field for field in PHYSICAL_FIELDS if nested(parent_env, field) == nested(env, field)]
    return {"full_actual_yaml_checked": True, "only_declared_physics_and_run_metadata_changed": True,
            "allowed_physics_fields": PHYSICAL_FIELDS, "original_reward_command_noise_reset_push_ppo_preserved": True,
            "training_task_not_play": True, "actual_yaml_differences": changes,
            "actual_physics_changed_paths": [row["path"] for row in changes if row["kind"] == "declared_physics"],
            "declared_physics_fields_actually_unchanged": unchanged,
            "physical_delta_note": "Pinned TRAIN parent already has fixed material .8/.6/0 and no CoM event. Those are enforced invariants, not new interventions. Actual changes are self-collision, action implementation/soft clamp, and removal of base-mass randomization."}


def check_final(metadata, parent, updates):
    require(type(updates) is int and updates in (2, 100), "Invalid finite update budget")
    require(metadata["iter"] == 3947 + updates - 1 and metadata["adam_steps"] == [79120 + 20 * updates] * 17,
            "Actual iteration/Adam steps do not match bounded continuation")
    require(metadata["model_shapes"] == parent["model_shapes"] and
            metadata["adam_state_shapes"] == parent["adam_state_shapes"], "Actor/critic/std/Adam layout changed")
    group, original = copy.deepcopy(metadata["optimizer_group"]), copy.deepcopy(parent["optimizer_group"])
    group.pop("lr")
    original.pop("lr")
    require(group == original, "Adam options or parameter order changed")
    require(metadata["all_tensors_finite"] is True and metadata["tensor_count"] == 68,
            "Incomplete finite model/optimizer check")
    require(metadata["sha256"] != parent["sha256"], "Candidate checkpoint is unchanged parent")


def all_sources():
    for path, expected in {OFFICIAL: OFFICIAL_SHA, REFERENCE_ENV: REFERENCE_ENV_SHA,
                           **GUARD_PINS, **EXTRA_FROZEN}.items():
        require(sha(path) == expected, "Pinned source/reference/model drift: " + str(path))
    sources = provenance()
    paths = [OFFICIAL, REFERENCE_ENV, *GUARD_PINS, *EXTRA_FROZEN, Path(__file__).resolve(),
             ROOT / "scripts/run_common_physics_adaptation.ps1", ROOT / "scripts/test_common_physics_adaptation.py"]
    for path in paths:
        sources.append({"path": str(path.resolve()), "sha256": sha(path)})
    # Identical repeated references are harmless; contradictory aliases are not.
    unique = {}
    for row in sources:
        key = str(Path(row["path"]).resolve())
        require(key not in unique or unique[key] == row["sha256"].lower(), "Contradictory source pin")
        unique[key] = row["sha256"].lower()
    return [{"path": key, "sha256": value} for key, value in sorted(unique.items())]


def verify_sources():
    for row in STATE["sources"]:
        require(sha(row["path"]) == row["sha256"], "Training source/model changed: " + row["path"])
    require(all_sources() == STATE["sources"], "Source/model inventory changed")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def json_metadata(value):
    """Match receipt transport (tuple -> list), without loosening value checks."""
    return json.loads(json.dumps(value, allow_nan=False))


def verify_smoke(path, sources):
    path = Path(path).resolve()
    require(path.drive.upper() == "E:" and path.name == "common_physics_training_result.json" and
            path.parent.parent == RUN_ROOT.resolve(), "Require actual isolated E-drive smoke result")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(receipt.get("protocol") == PROTOCOL and receipt.get("completion_verified") is True and
            receipt.get("num_envs") == 16 and receipt.get("updates") == 2 and receipt.get("actual_environment_steps") == 768,
            "Formal training requires successful smoke16x2 first")
    require(receipt.get("source_snapshot") == sources and receipt.get("parent", {}).get("sha256") == PARENT_SHA,
            "Smoke used different source or parent")
    checkpoint = Path(receipt["checkpoint"]).resolve()
    require(checkpoint.parent == path.parent and checkpoint.name == "model_3948.pt", "Smoke checkpoint path mismatch")
    actual = checkpoint_metadata(checkpoint)
    require(json_metadata(actual) == receipt["checkpoint_metadata"], "Smoke checkpoint metadata/bytes changed")
    check_final(actual, parent_metadata(), 2)
    before = json.loads((path.parent / "common_physics_pretrain_guard.json").read_text(encoding="utf-8"))
    require(before.get("passed") is True and before.get("parent_model_and_adam_exact") is True and
            before.get("config_guard", {}).get("full_actual_yaml_checked") is True and
            before.get("parent_sha256") == PARENT_SHA and before.get("initial_iteration") == 3947 and
            before.get("initial_adam_step") == 79120 and before.get("initial_scalar_and_optimizer_lr") == 1e-5,
            "Smoke pre-learning guard missing")
    check_documents(read(path.parent / "params/env.yaml"), read(path.parent / "params/agent.yaml"),
                    read(PARENT_CONFIG / "env.yaml"), read(PARENT_CONFIG / "agent.yaml"), reference_document(),
                    16, 2, receipt["run_name"])
    return {"path": str(path), "sha256": sha(path), "checkpoint_sha256": actual["sha256"], "smoke_only_not_quality": True}


def apply_physics(cfg, reference):
    """Edit only the five declared fields on TRAIN config; no Play replacement."""
    before = copy.deepcopy(cfg.to_dict())
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.actions.joint_pos = copy.deepcopy(reference.actions.joint_pos)
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.physics_material = copy.deepcopy(reference.events.physics_material)
    expected = copy.deepcopy(before)
    expected["scene"]["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] = True
    expected["actions"]["joint_pos"] = reference.actions.joint_pos.to_dict()
    expected["events"]["add_base_mass"] = expected["events"]["base_com"] = None
    expected["events"]["physics_material"] = reference.events.physics_material.to_dict()
    require(cfg.to_dict() == expected, "Runtime edit changed more than the declared physics fields")
    require(cfg.observations.policy.enable_corruption and cfg.events.push_robot is not None,
            "Original training noise/pushes were accidentally disabled")


def configure(env_cfg, agent_cfg, log_dir):
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_env_cfg import UnitreeGo2RecoveryBankSmithNominalEnvCfg
    args = STATE["args"]
    require(agent_cfg.run_name == args.run_name and agent_cfg.logger == "tensorboard", "Wrong run/external logger")
    path = Path(log_dir).resolve()
    require(path.parent == RUN_ROOT.resolve() and not path.exists(), "Use a new isolated E-drive run")
    verify_sources()
    previous = os.environ.get("ISAACLAB_RECOVERY_BANK_COLLECTION")
    os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = "1"
    try:
        reference = UnitreeGo2RecoveryBankSmithNominalEnvCfg()
    finally:
        if previous is None:
            del os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"]
        else:
            os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = previous
    apply_physics(env_cfg, reference)
    path.mkdir(parents=True, exist_ok=False)
    STATE["log_dir"] = path
    with (path / "training_source_generated.py").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(STATE["generated"])
    write_new(path / "common_physics_invocation.json", {
        "protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": STATE["parent"], "parent_path": str(PARENT), "run_name": args.run_name,
        "num_envs": args.num_envs, "updates": args.max_iterations, "source_snapshot": STATE["sources"],
        "argv": STATE["argv"], "smoke_evidence": STATE["smoke"],
        "generated_sha256": sha(path / "training_source_generated.py"),
        "allowed_physics_fields": PHYSICAL_FIELDS, "quality_accepted": False, "promotion_performed": False})


def equal_tree(actual, expected, label="state"):
    import torch
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()), "Non-exact parent tensor: " + label)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "State keys differ: " + label)
        for key in expected:
            equal_tree(actual[key], expected[key], label + "/" + str(key))
    elif isinstance(expected, (tuple, list)):
        require(type(actual) is type(expected) and len(actual) == len(expected), "State length/type differs")
        for i, value in enumerate(expected):
            equal_tree(actual[i], value, label + "/" + str(i))
    else:
        require(actual == expected, "Parent scalar differs: " + label)


def verify_before(runner, env, log_dir):
    import torch
    args = STATE["args"]
    path = Path(log_dir)
    guard = check_documents(read(path / "params/env.yaml"), read(path / "params/agent.yaml"),
        read(PARENT_CONFIG / "env.yaml"), read(PARENT_CONFIG / "agent.yaml"), reference_document(),
        args.num_envs, args.max_iterations, args.run_name)
    saved = torch.load(PARENT, map_location="cpu", weights_only=True)
    equal_tree(runner.alg.policy.state_dict(), saved["model_state_dict"], "actor_critic_std")
    equal_tree(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"], "Adam")
    require(runner.current_learning_iteration == 3947 and runner.alg.learning_rate == 1e-5 and
            runner.alg.optimizer.param_groups[0]["lr"] == 1e-5 and runner.tot_timesteps == 0,
            "Incorrect initial iteration/LR or already-used runner")
    STATE["runner"] = runner
    STATE["initial_common_steps"] = int(env.unwrapped.common_step_counter)
    verify_sources()
    write_new(path / "common_physics_pretrain_guard.json", {
        "passed": True, "parent_model_and_adam_exact": True, "config_guard": guard,
        "parent_sha256": PARENT_SHA, "initial_iteration": 3947, "initial_adam_step": 79120,
        "initial_scalar_and_optimizer_lr": 1e-5, "initial_common_steps": STATE["initial_common_steps"],
        "actual_yaml": [{"path": str(path / "params" / name), "sha256": sha(path / "params" / name)}
                        for name in ("env.yaml", "agent.yaml")], "quality_accepted": False})


def finish(env, log_dir):
    args, runner = STATE["args"], STATE["runner"]
    path = Path(log_dir)
    checkpoint = path / f"model_{3947 + args.max_iterations - 1}.pt"
    metadata = checkpoint_metadata(checkpoint)
    check_final(metadata, STATE["parent"], args.max_iterations)
    require(runner.current_learning_iteration == metadata["iter"], "Runner/final iteration differs")
    actual_steps = runner.tot_timesteps
    expected_steps = args.num_envs * args.max_iterations * 24
    require(type(actual_steps) is int and actual_steps == expected_steps, "Actual environment-step budget differs")
    require(int(env.unwrapped.common_step_counter) - STATE["initial_common_steps"] == 24 * args.max_iterations,
            "Actual control-step count differs")
    guard = check_documents(read(path / "params/env.yaml"), read(path / "params/agent.yaml"),
        read(PARENT_CONFIG / "env.yaml"), read(PARENT_CONFIG / "agent.yaml"), reference_document(),
        args.num_envs, args.max_iterations, args.run_name)
    verify_sources()
    # Adam moments/std may adapt, but never silently discard the full-state parent.
    write_new(path / "common_physics_training_result.json", {
        "protocol": PROTOCOL, "completion_verified": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name, "parent": STATE["parent"], "parent_path": str(PARENT),
        "num_envs": args.num_envs, "updates": args.max_iterations, "actual_environment_steps": actual_steps,
        "actual_control_steps": int(env.unwrapped.common_step_counter) - STATE["initial_common_steps"],
        "checkpoint": str(checkpoint), "checkpoint_metadata": metadata, "config_guard": guard,
        "source_snapshot": STATE["sources"], "smoke_evidence": STATE["smoke"],
        "actual_yaml": [{"path": str(path / "params" / name), "sha256": sha(path / "params" / name)}
                        for name in ("env.yaml", "agent.yaml")],
        "generated_sha256": sha(path / "training_source_generated.py"),
        "quality_accepted": False, "promotion_performed": False, "hardware_execution": False,
        "note": "Finite shared-physics adaptation only. Preserve commands/rewards/noise/reset/push/PPO. New recovery-physics .5/.8 stand/stop/turn/push plus original-physics regression are required; no fast-running or full-flow claim."})
    print("COMMON_PHYSICS_FINITE_COMPLETE " + str(checkpoint), flush=True)


def build_source(source):
    replacements = [
        ("    # create isaac environment\n", "    _common_configure(env_cfg, agent_cfg, log_dir)\n\n    # create isaac environment\n"),
        ("    # run training\n", "    _common_verify_before(runner, env, log_dir)\n\n    # run training\n"),
        ("    # close the simulator\n", "    _common_finish(env, log_dir)\n\n    # close the simulator\n"),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, "Official trainer anchor drift")
        source = source.replace(old, new, 1)
    compile(source, str(OFFICIAL) + "::<common-physics>", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--max_iterations", type=int, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--smoke_receipt", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args, remaining = parser.parse_known_args()
    arguments_ok(args.num_envs, args.max_iterations, args.run_name)
    require(args.task == TASK, "Use the original noisy/pushed TRAIN task")
    require(Path.cwd().resolve() == Path("E:/IsaacLab/repo").resolve(), "Official trainer must run in the E-drive Isaac checkout")
    sources, parent = all_sources(), parent_metadata()
    reference_document()
    if args.max_iterations == 100:
        require(args.smoke_receipt is not None, "Formal block requires the completed same-source smoke receipt")
        smoke = verify_smoke(args.smoke_receipt, sources)
    else:
        require(args.smoke_receipt is None, "Smoke always starts independently from original3947")
        smoke = None
    generated = build_source(OFFICIAL.read_text(encoding="utf-8"))
    STATE.update(args=args, sources=sources, parent=parent, generated=generated, smoke=smoke, argv=list(sys.argv))
    if args.preflight_only:
        print(json.dumps({"protocol": PROTOCOL, "preflight_pass": True, "parent": parent,
              "num_envs": args.num_envs, "updates": args.max_iterations, "smoke_evidence": smoke,
              "expected_final_iteration": 3947 + args.max_iterations - 1,
              "expected_adam_step": 79120 + 20 * args.max_iterations,
              "expected_environment_steps": args.num_envs * 24 * args.max_iterations,
              "source_snapshot": sources, "simulator_started": False, "quality_accepted": False}, indent=2))
        return
    # Remove only this adapter's private arguments before the immutable trainer parses argv.
    sys.argv = [sys.argv[0], "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
                "--run_name", args.run_name, "--task", args.task, *remaining]
    sys.path.insert(0, str(OFFICIAL.parent))
    exec(compile(generated, str(OFFICIAL), "exec"), {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
         "_common_configure": configure, "_common_verify_before": verify_before, "_common_finish": finish})


if __name__ == "__main__":
    main()
