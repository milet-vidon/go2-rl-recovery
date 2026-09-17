"""Read-only full-schema guard for the original-command continuation control.

The failed expanded-range pilot and this control share the same parent, optimizer
and synchronized LR. Only the command-range intervention is absent here.
"""

import argparse
import copy
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

from check_smith_standweight_config import differences, read, require, _log_path
from check_speed_curriculum_config import PARENT_DIR, PARENT_RUN, check_checkpoint


def check(candidate_dir, num_envs, iterations, run_name):
    require(type(num_envs) is int and type(iterations) is int
            and (num_envs, iterations) in {(16, 2), (128, 300)},
            "Only 16-env/2-update smoke or 128-env/300-update formal control is allowed")
    require(isinstance(run_name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_name),
            "Invalid expected run name")
    parent_env, parent_agent = read(PARENT_DIR / "env.yaml"), read(PARENT_DIR / "agent.yaml")
    env, agent = read(Path(candidate_dir) / "env.yaml"), read(Path(candidate_dir) / "agent.yaml")
    for name, document in (("parent env", parent_env), ("parent agent", parent_agent),
                           ("candidate env", env), ("candidate agent", agent)):
        require(isinstance(document, dict), f"{name} must contain a YAML mapping")
    require(parent_agent.get("run_name") == "natural_robust_push_20260909"
            and parent_agent.get("experiment_name") == "unitree_go2_flat"
            and parent_agent.get("resume") == "true"
            and parent_agent.get("load_run") == "2026-09-09_15-15-13_natural_robust_20260909"
            and parent_agent.get("load_checkpoint") == "model_2849.pt"
            and parent_agent.get("max_iterations") == "800" and parent_agent.get("seed") == "42",
            "Archived parent identity changed")
    command = parent_env["commands"]["base_velocity"]
    require(command["ranges"]["lin_vel_x"] == ["-0.5", "1.0"]
            and command["rel_standing_envs"] == "0.3", "Parent command range/standing mixture changed")
    expected_env = copy.deepcopy(parent_env)
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
    rate = agent["algorithm"]["learning_rate"]
    try:
        correct_rate = Decimal(rate) == Decimal("0.00001")
    except (InvalidOperation, TypeError, ValueError):
        correct_rate = False
    require(correct_rate, "Adaptive LR scalar must match saved parent LR 1e-5")
    expected_agent["algorithm"]["learning_rate"] = rate
    delta = differences(expected_agent, agent, "agent")
    require(not delta, f"Unexpected PPO/full-resume configuration change: {delta}")
    print(f"Original-distribution control config passed: {num_envs} envs/{iterations} extra updates; "
          "unchanged vx[-0.5,1.0],30% standing, rewards/pushes/physics/observations/actions/PPO; "
          "only LR scalar synchronized to saved parent1e-5 and run metadata changed. No quality acceptance.")


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
