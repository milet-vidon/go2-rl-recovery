"""Read-only full-schema and optional checkpoint guard for the speed pilot.

YAML tags remain data. Torch is imported only for explicit CPU checkpoint checks.
This verifies provenance/finite tensors, NOT policy quality or real-world safety.
"""

import argparse
import copy
import hashlib
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from check_smith_standweight_config import differences, read, require, _log_path


ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT / "configs/natural-robust-push-20260909"
PARENT_RUN = "2026-09-09_15-50-37_natural_robust_push_20260909"
PARENT_CHECKPOINT = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat") / PARENT_RUN / "model_3648.pt"
PARENT_SHA = "475f07bf0b05101dd4157212b4e3fd48a04731d452d7a349fe467c508dbaaac8"


def check(candidate_dir, num_envs, iterations, run_name):
    require(type(num_envs) is int and type(iterations) is int
            and (num_envs, iterations) in {(16, 2), (128, 300)},
            "Only 16-env/2-update smoke or 128-env/300-update formal budgets are allowed")
    require(isinstance(run_name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_name),
            "Invalid expected run name")
    parent_env, parent_agent = read(PARENT_DIR / "env.yaml"), read(PARENT_DIR / "agent.yaml")
    env, agent = read(Path(candidate_dir) / "env.yaml"), read(Path(candidate_dir) / "agent.yaml")
    for label, obj in (("parent env", parent_env), ("parent agent", parent_agent),
                       ("candidate env", env), ("candidate agent", agent)):
        require(isinstance(obj, dict), f"{label} must be a YAML mapping")
    require(parent_agent.get("run_name") == "natural_robust_push_20260909"
            and parent_agent.get("experiment_name") == "unitree_go2_flat"
            and parent_agent.get("resume") == "true"
            and parent_agent.get("load_run") == "2026-09-09_15-15-13_natural_robust_20260909"
            and parent_agent.get("load_checkpoint") == "model_2849.pt"
            and parent_agent.get("max_iterations") == "800" and parent_agent.get("seed") == "42",
            "Unexpected archived parent identity")
    command = parent_env["commands"]["base_velocity"]
    require(command["ranges"]["lin_vel_x"] == ["-0.5", "1.0"]
            and command["rel_standing_envs"] == "0.3", "Parent speed range/standing mixture changed")
    expected_env = copy.deepcopy(parent_env)
    expected_env["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = ["-0.5", "1.2"]
    expected_env["scene"]["num_envs"] = str(num_envs)
    expected_env["scene"]["terrain"]["num_envs"] = str(num_envs)
    _log_path(env["log_dir"], "env.log_dir")
    _log_path(env["sim"]["log_dir"], "env.sim.log_dir", allow_null=True)
    expected_env["log_dir"] = env["log_dir"]
    expected_env["sim"]["log_dir"] = env["sim"]["log_dir"]
    delta = differences(expected_env, env, "env")
    require(not delta, f"Unexpected environment/command/reward/push change: {delta}")
    expected_agent = copy.deepcopy(parent_agent)
    expected_agent.update(run_name=run_name, max_iterations=str(iterations),
                          load_run=PARENT_RUN, load_checkpoint="model_3648.pt")
    # The historical config is 3e-4 but its saved adaptive optimizer LR is 1e-5.
    # Explicitly synchronize the runner's LR scalar to that saved state.
    learning_rate = agent["algorithm"]["learning_rate"]
    try:
        correct_rate = Decimal(learning_rate) == Decimal("0.00001")
    except (InvalidOperation, TypeError, ValueError):
        correct_rate = False
    require(correct_rate, "Adaptive LR scalar must explicitly match saved parent LR 1e-5")
    expected_agent["algorithm"]["learning_rate"] = learning_rate
    delta = differences(expected_agent, agent, "agent")
    require(not delta, f"Unexpected PPO/resume configuration change: {delta}")
    print(f"Speed config passed: {num_envs} envs/{iterations} extra updates; only x command upper bound 1.0 -> 1.2 m/s. "
          "Standing 30%, commands/rewards/pushes/physics/PPO otherwise unchanged; LR scalar explicitly synchronized to saved 1e-5.")


def check_checkpoint(path, iterations):
    import torch

    path = Path(path)
    require(path.name == f"model_{3648 + iterations - 1}.pt", "Unexpected final checkpoint filename")
    require(hashlib.sha256(PARENT_CHECKPOINT.read_bytes()).hexdigest() == PARENT_SHA,
            "Frozen parent checkpoint hash mismatch")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint.get("iter") == 3648 + iterations - 1, "Wrong final checkpoint iteration")
    require(set(checkpoint["model_state_dict"]) == set(parent["model_state_dict"]), "Model schema changed")
    tensor_count = 0
    def finite_tree(value, prefix):
        nonlocal tensor_count
        if torch.is_tensor(value):
            require(torch.isfinite(value).all().item(), f"Nonfinite tensor: {prefix}")
            tensor_count += 1
        elif isinstance(value, dict):
            for key, item in value.items():
                finite_tree(item, f"{prefix}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                finite_tree(item, f"{prefix}[{i}]")
        elif type(value) in (int, float):
            require(math.isfinite(value), f"Nonfinite scalar: {prefix}")
    finite_tree(checkpoint, "checkpoint")
    require(tensor_count > 0, "Checkpoint contains no tensors")
    for key, tensor in parent["model_state_dict"].items():
        require(checkpoint["model_state_dict"][key].shape == tensor.shape, f"Model tensor shape changed: {key}")
    old_optimizer, optimizer = parent["optimizer_state_dict"], checkpoint["optimizer_state_dict"]
    require(set(optimizer["state"]) == set(old_optimizer["state"]), "Optimizer parameter-state keys changed")
    require(len(optimizer["param_groups"]) == len(old_optimizer["param_groups"]), "Optimizer group count changed")
    for old_group, group in zip(old_optimizer["param_groups"], optimizer["param_groups"]):
        require(group["params"] == old_group["params"], "Optimizer parameter grouping changed")
    for key, old_state in old_optimizer["state"].items():
        state = optimizer["state"][key]
        require(set(state) == set(old_state), f"Optimizer schema changed: {key}")
        require(float(state["step"]) == float(old_state["step"]) + iterations * 20,
                f"Optimizer update count does not prove full-state continuation: {key}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Checkpoint finite/provenance check passed: {tensor_count} tensors, final iter {checkpoint['iter']}, SHA256 {digest}")
    print("Full-model/optimizer resume with saved-LR scalar synchronization; iteration label repeats and RNG/simulator reset. No quality acceptance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    check(args.candidate_dir, args.num_envs, args.iterations, args.run_name)
    if args.checkpoint is not None:
        check_checkpoint(args.checkpoint, args.iterations)
