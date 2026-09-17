"""Finite speed-sampler experiment layered onto immutable official trainer."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

from check_speed_tracking_weight_block import (
    ROOT, PARENT, PARENT_CONFIG, PARENT_RUN, provenance, parent_metadata,
    checkpoint_metadata, check_documents, check_final, read, sha, require,
)

OFFICIAL = Path("E:/IsaacLab/repo/scripts/reinforcement_learning/rsl_rl/train.py")
OFFICIAL_SHA = "7586436b3318c1eb705600aef07bdd54f55d5c35943b266563972adc0c7a22cb"
NEW_SOURCES = [Path(__file__).resolve(), ROOT / "scripts/speed_gated_command.py",
               ROOT / "src/go2_recovery/speed_gate_math.py"]
STATE = {}


def check_sources():
    for record in STATE["sources"]:
        require(sha(Path(record["path"])) == record["sha256"].lower(), "Frozen source/model drift: " + record["path"])


def configure(env_cfg, agent_cfg, log_dir):
    from speed_gated_command import GatedVelocityCommand
    require(agent_cfg.run_name == STATE["args"].run_name, "Run tag changed")
    path = Path(log_dir).resolve()
    require(path.drive.upper() == "E:" and not path.exists(), "Use a new E-drive output")
    check_sources()
    env_cfg.commands.base_velocity.class_type = GatedVelocityCommand
    path.mkdir(parents=True, exist_ok=False)
    STATE["log_dir"] = path
    (path / "training_source_generated.py").write_text(STATE["generated"], encoding="utf-8")
    (path / "speed_experiment_invocation.json").write_text(json.dumps({
        "protocol": "bounded_speed_gate_training_v1", "sources": STATE["sources"],
        "parent": str(PARENT), "parent_sha256": STATE["parent"]["sha256"],
        "argv": sys.argv, "finite_updates": STATE["args"].max_iterations,
        "num_envs": STATE["args"].num_envs, "quality_accepted": False,
        "generated_sha256": sha(path / "training_source_generated.py")}, indent=2), encoding="utf-8")


def verify_before(runner, env, log_dir):
    import torch
    args = STATE["args"]
    config = Path(log_dir) / "params"
    document = read(config / "env.yaml")
    require(document["commands"]["base_velocity"]["class_type"] == "speed_gated_command:GatedVelocityCommand",
            "New sampler was not installed in actual configuration")
    document = copy.deepcopy(document)
    document["commands"]["base_velocity"]["class_type"] = "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand"
    check_documents(document, read(config / "agent.yaml"), read(PARENT_CONFIG / "env.yaml"),
                    read(PARENT_CONFIG / "agent.yaml"), "A", args.num_envs, args.max_iterations, args.run_name)
    saved = torch.load(PARENT, map_location=runner.device, weights_only=False)
    current = runner.alg.policy.state_dict()
    require(set(current) == set(saved["model_state_dict"]), "Parent architecture mismatch")
    require(all(torch.equal(current[k], saved["model_state_dict"][k]) for k in current), "Parent actor/critic/std not resumed exactly")
    actual_optim = runner.alg.optimizer.state_dict()
    parent_optim = saved["optimizer_state_dict"]
    require(actual_optim["param_groups"] == parent_optim["param_groups"], "Parent optimizer groups differ")
    require(set(actual_optim["state"]) == set(parent_optim["state"]), "Parent Adam state missing")
    for index, values in actual_optim["state"].items():
        require(set(values) == set(parent_optim["state"][index]), "Parent Adam keys differ")
        for key, value in values.items():
            # Adam step counters may intentionally remain on CPU while moments
            # are on CUDA. Compare exact values without changing runner state.
            require(torch.equal(value.detach().cpu(), parent_optim["state"][index][key].detach().cpu()),
                    "Parent Adam moment/counter not resumed")
    require(runner.alg.learning_rate == 1e-5 and runner.current_learning_iteration == 3947, "Wrong resume LR/iteration")
    check_sources()
    (Path(log_dir) / "speed_pretrain_guard.json").write_text(json.dumps({
        "passed": True, "full_config_checked": True, "parent_model_and_adam_exact": True,
        "initial_cap": env.unwrapped.command_manager.get_term("base_velocity").course.snapshot()["cap"],
        "quality_accepted": False}, indent=2), encoding="utf-8")


def finish(env, log_dir):
    args = STATE["args"]
    course = env.unwrapped.command_manager.get_term("base_velocity")
    course.save_course()
    checkpoint = Path(log_dir) / f"model_{3947 + args.max_iterations - 1}.pt"
    metadata = checkpoint_metadata(checkpoint)
    check_final(metadata, STATE["parent"], args.max_iterations)
    check_sources()
    result = {"status": "finite_training_complete", "checkpoint": str(checkpoint), "checkpoint_metadata": metadata,
        "course": course.course.snapshot(), "num_envs": args.num_envs, "updates": args.max_iterations,
        "steps": args.num_envs * args.max_iterations * 24, "quality_accepted": False,
        "note": "Training only. No auto-extension or promotion. Full18 behavioral regression required."}
    (Path(log_dir) / "speed_training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SPEED_GATE_FINITE_COMPLETE " + str(checkpoint), flush=True)


def build_source(source):
    replacements = [
        ("    # create isaac environment\n", "    _speed_configure(env_cfg, agent_cfg, log_dir)\n\n    # create isaac environment\n"),
        ("    # run training\n", "    _speed_verify_before(runner, env, log_dir)\n\n    # run training\n"),
        ("    # close the simulator\n", "    _speed_finish(env, log_dir)\n\n    # close the simulator\n"),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, "Official trainer anchor drift")
        source = source.replace(old, new)
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--max_iterations", type=int, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--task", required=True)
    args, _ = parser.parse_known_args()
    require((args.num_envs, args.max_iterations) in {(16, 2), (128, 50)}, "Only finite smoke/formal budgets")
    require(re.fullmatch(r"20260917-speedgate[\w-]+", args.run_name) is not None, "Unexpected run name")
    require(args.task == "Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0", "Wrong parent task")
    require(sha(OFFICIAL) == OFFICIAL_SHA, "Official trainer source changed")
    sources = provenance() + [{"path": str(p), "sha256": sha(p)} for p in [OFFICIAL, *NEW_SOURCES]]
    generated = build_source(OFFICIAL.read_text(encoding="utf-8"))
    STATE.update(args=args, sources=sources, parent=parent_metadata(), generated=generated)
    sys.path.insert(0, str(OFFICIAL.parent))
    exec(compile(generated, str(OFFICIAL), "exec"), {
        "__name__": "__main__", "__file__": str(Path(__file__).resolve()),
        "_speed_configure": configure, "_speed_verify_before": verify_before, "_speed_finish": finish})
