"""Read-only full-schema guard for the shutdown-interrupted w30 continuation."""

import copy
import sys
from pathlib import Path

from check_smith_standweight_config import check, differences, read, require, _log_path


PARENT_RUN = "2026-09-16_20-41-29_20260916-smithstandpair128x1000_w30"
PARENT = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_recovery") / PARENT_RUN / "params"
TAG = "20260916-smithstandresume2900_w30"


def verify(candidate):
    check(PARENT, 128, 1000, 30)
    candidate = Path(candidate)
    old_env, old_agent = read(PARENT / "env.yaml"), read(PARENT / "agent.yaml")
    env, agent = read(candidate / "env.yaml"), read(candidate / "agent.yaml")
    expected_env = copy.deepcopy(old_env)
    _log_path(env["log_dir"], "env.log_dir")
    _log_path(env["sim"]["log_dir"], "env.sim.log_dir", allow_null=True)
    expected_env["log_dir"] = env["log_dir"]
    expected_env["sim"]["log_dir"] = env["sim"]["log_dir"]
    require(not differences(expected_env, env), f"Environment changed: {differences(expected_env, env)}")
    expected_agent = copy.deepcopy(old_agent)
    expected_agent.update(run_name=TAG, max_iterations="98", load_run=PARENT_RUN,
                          load_checkpoint="model_2900.pt")
    require(not differences(expected_agent, agent), f"Agent changed: {differences(expected_agent, agent)}")
    print("Resume config passed: unchanged w30 physics/rewards/PPO; full model2900 resume, 98 new updates from2901 to2998.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_smith_resume2900_config.py PARAMS_DIRECTORY")
    verify(sys.argv[1])
