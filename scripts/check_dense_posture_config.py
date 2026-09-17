"""Read-only full-schema guard for the isolated dense-posture continuation.

Serialized YAML tags are only data. This imports no simulator and writes no files.
Standard RSL-RL resume restores model/optimizer but is not an exact uninterrupted
resume: the scalar adaptive learning rate is initialized from the configuration.
"""

import argparse
import copy
import re
from pathlib import Path

from check_smith_standweight_config import differences, read, require, _log_path


ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT / "configs/20260916-smithnominal128x2000"
PARENT_RUN = "2026-09-16_19-02-40_20260916-smithnominal128x2000"
PARENT_CHECKPOINT = "model_1999.pt"
ALLOWED_BUDGETS = {(16, 2), (128, 600)}
DENSE_REWARD = {
    "func": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_supported_mdp:SupportedPostureReward",
    "params": {},
    "weight": "6.0",
}


def check(candidate_dir, num_envs, iterations, run_name):
    require(type(num_envs) is int and type(iterations) is int
            and (num_envs, iterations) in ALLOWED_BUDGETS,
            "Only 16-env/2-iteration smoke or 128-env/600-iteration formal budgets are allowed")
    require(isinstance(run_name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_name),
            "Expected run name must contain only letters, digits, underscore or hyphen")
    candidate_dir = Path(candidate_dir)
    parent_env, parent_agent = read(PARENT_DIR / "env.yaml"), read(PARENT_DIR / "agent.yaml")
    env, agent = read(candidate_dir / "env.yaml"), read(candidate_dir / "agent.yaml")
    for label, obj in (("parent env", parent_env), ("parent agent", parent_agent),
                       ("candidate env", env), ("candidate agent", agent)):
        require(isinstance(obj, dict), f"{label} must be a YAML mapping")
    require(parent_agent.get("run_name") == "20260916-smithnominal128x2000"
            and parent_agent.get("resume") == "true"
            and parent_agent.get("load_run") == "bootstrap_fresh42_target_v1"
            and parent_agent.get("load_checkpoint") == "model_0.pt"
            and parent_agent.get("seed") == "42"
            and parent_agent.get("max_iterations") == "2000",
            "Archived parent identity/bootstrap changed")
    require("dense_posture" not in parent_env["rewards"], "Parent unexpectedly has dense posture shaping")
    for component in ("roll", "stand"):
        require(parent_env["rewards"][f"smith_{component}"] == {
            "func": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_smith_mdp:SmithRecoveryReward",
            "params": {"mode": component, "target_height": "0.32"}, "weight": "10.0"},
            "Archived parent Smith reward changed")

    expected_env = copy.deepcopy(parent_env)
    expected_env["scene"]["num_envs"] = str(num_envs)
    expected_env["scene"]["terrain"]["num_envs"] = str(num_envs)
    expected_env["rewards"]["dense_posture"] = copy.deepcopy(DENSE_REWARD)
    _log_path(env["log_dir"], "env.log_dir")
    _log_path(env["sim"]["log_dir"], "env.sim.log_dir", allow_null=True)
    expected_env["log_dir"] = env["log_dir"]
    expected_env["sim"]["log_dir"] = env["sim"]["log_dir"]
    env_diff = differences(expected_env, env, "env")
    require(not env_diff, f"Unexpected environment/reward change: {env_diff}")

    expected_agent = copy.deepcopy(parent_agent)
    expected_agent.update(run_name=run_name, max_iterations=str(iterations),
                          load_run=PARENT_RUN, load_checkpoint=PARENT_CHECKPOINT)
    agent_diff = differences(expected_agent, agent, "agent")
    require(not agent_diff, f"Unexpected PPO/resume configuration change: {agent_diff}")
    print(f"Dense-posture config passed: {num_envs} environments, {iterations} additional updates, "
          f"parent1999 -> model_{1999 + iterations - 1}.pt; only dense_posture reward added at weight 6. "
          "Standard full-model/optimizer resume, not exact uninterrupted LR/RNG/simulator continuation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    check(args.candidate_dir, args.num_envs, args.iterations, args.run_name)
