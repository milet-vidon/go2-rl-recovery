"""Read-only check: reward-only Go2 pilot matches its completed nominal control.

BaseLoader treats YAML tags as data and never constructs Python objects.
"""
import argparse
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REMOVED = ("upright_and_height", "orientation_progress", "recovery_success", "stable_stand",
           "static_stance", "low_height_support", "low_height_lift", "conditional_stand_posture",
           "uncrossed_stand", "crossed_limbs")


def read(path):
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted(set(a) | set(b))
                for item in differences(a.get(key, "<MISSING>"), b.get(key, "<MISSING>"), f"{prefix}.{key}")]
    return [] if a == b else [(prefix, a, b)]


def check(candidate_dir, num_envs, iterations):
    baseline = ROOT / "configs/20260916-target128x2000-nominal"
    control, candidate = read(baseline / "env.yaml"), read(Path(candidate_dir) / "env.yaml")
    assert candidate["scene"]["num_envs"] == str(num_envs), "Wrong environment count"
    if "num_envs" in candidate["scene"]["terrain"]:
        assert candidate["scene"]["terrain"]["num_envs"] == str(num_envs), "Wrong terrain environment count"
    expected_rewards = copy.deepcopy(control["rewards"])
    for name in REMOVED:
        expected_rewards[name] = "null"
    for component in ("roll", "stand"):
        expected_rewards[f"smith_{component}"] = {
            "func": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_smith_mdp:SmithRecoveryReward",
            "params": {"mode": component, "target_height": "0.32"}, "weight": "10.0"}
    reward_diff = differences(expected_rewards, candidate["rewards"], "rewards")
    assert not reward_diff, f"Unexpected reward configuration: {reward_diff}"
    del control["rewards"], candidate["rewards"]
    for env in (control, candidate):
        env["scene"]["num_envs"] = "<run-count>"
        # Terrain copies the environment count after environment construction.
        if "num_envs" in env["scene"]["terrain"]:
            env["scene"]["terrain"]["num_envs"] = "<run-count>"
        env["sim"]["log_dir"] = "<runtime-log-dir>"
        env["log_dir"] = "<runtime-log-dir>"
    env_diff = differences(control, candidate, "env")
    assert not env_diff, f"Non-reward environment changed: {env_diff}"
    control_agent, candidate_agent = read(baseline / "agent.yaml"), read(Path(candidate_dir) / "agent.yaml")
    assert candidate_agent["max_iterations"] == str(iterations), "Wrong training budget"
    for agent in (control_agent, candidate_agent):
        agent["run_name"] = "<run-name>"
        agent["max_iterations"] = "<run-budget>"
    agent_diff = differences(control_agent, candidate_agent, "agent")
    assert not agent_diff, f"PPO/bootstrap changed: {agent_diff}"
    print(f"Reward-only config check passed: {num_envs} envs, {iterations} iterations; physics/actions/bank/PPO/bootstrap unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    check(args.candidate_dir, args.num_envs, args.iterations)
