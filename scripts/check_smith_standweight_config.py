"""Read-only full-schema guard for the paired Smith stand-weight continuation.

Only YAML BaseLoader is used: serialized Python tags are data, never executed.
The guard does not import torch, Isaac Lab, or a simulator and writes no files.
"""

import argparse
import copy
from decimal import Decimal, InvalidOperation
from pathlib import Path, PureWindowsPath

import yaml


ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT / "configs/20260916-smithnominal128x2000"
PARENT_RUN = "2026-09-16_19-02-40_20260916-smithnominal128x2000"
PARENT_CHECKPOINT = "model_1999.pt"
ALLOWED_BUDGETS = {(16, 2), (128, 1000)}
_MISSING = object()


def read(path):
    """Read tagged Isaac Lab YAML without constructing tagged Python objects."""
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def require(condition, message):
    # Explicit exceptions keep the guard effective under python -O as well.
    if not condition:
        raise AssertionError(message)


def differences(expected, actual, prefix=""):
    """Compare the complete mapping/list schema, including missing keys."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [item for key in sorted(set(expected) | set(actual))
                for item in differences(expected.get(key, _MISSING), actual.get(key, _MISSING),
                                        f"{prefix}.{key}")]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [(prefix + ".length", len(expected), len(actual))]
        return [item for i, (a, b) in enumerate(zip(expected, actual))
                for item in differences(a, b, f"{prefix}[{i}]")]
    if expected == actual:
        return []
    a = "<MISSING>" if expected is _MISSING else expected
    b = "<MISSING>" if actual is _MISSING else actual
    return [(prefix, a, b)]


def _stand_weight(value, expected):
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        return False
    return result.is_finite() and result == expected


def _log_path(value, label, allow_null=False):
    require(isinstance(value, str), f"{label} must be a YAML scalar")
    if allow_null and value == "null":
        return
    path = PureWindowsPath(value)
    require(path.is_absolute() and path.drive.upper() == "E:" and ".." not in path.parts,
            f"{label} must remain an absolute E: path")


def check(candidate_dir, num_envs, iterations, stand_weight):
    require(type(num_envs) is int and type(iterations) is int
            and (num_envs, iterations) in ALLOWED_BUDGETS,
            "Only 16-env/2-iteration smoke or 128-env/1000-iteration formal budgets are allowed")
    require(type(stand_weight) is int and stand_weight in (10, 30),
            "stand_weight must be exactly 10 or 30")
    candidate_dir = Path(candidate_dir)
    parent_env, parent_agent = read(PARENT_DIR / "env.yaml"), read(PARENT_DIR / "agent.yaml")
    candidate_env, candidate_agent = read(candidate_dir / "env.yaml"), read(candidate_dir / "agent.yaml")
    for name, document in (("parent env", parent_env), ("parent agent", parent_agent),
                           ("candidate env", candidate_env), ("candidate agent", candidate_agent)):
        require(isinstance(document, dict), f"{name} must contain a YAML mapping")

    # Fail closed if the archived parent no longer identifies the intended run.
    require(parent_agent.get("run_name") == "20260916-smithnominal128x2000"
            and parent_agent.get("resume") == "true"
            and parent_agent.get("load_run") == "bootstrap_fresh42_target_v1"
            and parent_agent.get("load_checkpoint") == "model_0.pt"
            and parent_agent.get("seed") == "42"
            and parent_agent.get("max_iterations") == "2000",
            "Unexpected parent agent identity/bootstrap")
    for component in ("roll", "stand"):
        require(parent_env["rewards"][f"smith_{component}"]["weight"] == "10.0",
                "Unexpected parent Smith reward weight")

    expected_env = copy.deepcopy(parent_env)
    require(candidate_env["scene"]["num_envs"] == str(num_envs), "Wrong environment count")
    require(candidate_env["scene"]["terrain"]["num_envs"] == str(num_envs),
            "Wrong terrain environment count")
    expected_env["scene"]["num_envs"] = str(num_envs)
    expected_env["scene"]["terrain"]["num_envs"] = str(num_envs)
    weight = candidate_env["rewards"]["smith_stand"]["weight"]
    require(_stand_weight(weight, stand_weight), "Wrong smith_stand weight")
    # Only this scalar is variable; the mode, function and target remain exact.
    expected_env["rewards"]["smith_stand"]["weight"] = weight
    _log_path(candidate_env["log_dir"], "env.log_dir")
    _log_path(candidate_env["sim"]["log_dir"], "env.sim.log_dir", allow_null=True)
    expected_env["log_dir"] = candidate_env["log_dir"]
    expected_env["sim"]["log_dir"] = candidate_env["sim"]["log_dir"]
    env_diff = differences(expected_env, candidate_env, "env")
    require(not env_diff, f"Unexpected environment/reward change: {env_diff}")

    require(candidate_agent.get("resume") == "true", "resume must remain true")
    require(candidate_agent.get("load_run") == PARENT_RUN, "Wrong parent load_run")
    # Isaac Lab maps CLI --checkpoint to the saved load_checkpoint field.
    require(candidate_agent.get("load_checkpoint") == PARENT_CHECKPOINT, "Wrong parent load_checkpoint")
    require(candidate_agent.get("max_iterations") == str(iterations), "Wrong additional training budget")
    run_name = candidate_agent.get("run_name")
    require(isinstance(run_name, str) and bool(run_name.strip()) and run_name != "null",
            "run_name must be a nonempty YAML scalar")
    expected_agent = copy.deepcopy(parent_agent)
    expected_agent.update(run_name=run_name, max_iterations=str(iterations),
                          load_run=PARENT_RUN, load_checkpoint=PARENT_CHECKPOINT)
    agent_diff = differences(expected_agent, candidate_agent, "agent")
    require(not agent_diff, f"Unexpected PPO/resume configuration change: {agent_diff}")
    print(f"Smith stand-weight config check passed: {num_envs} envs, {iterations} additional iterations, "
          f"roll=10/stand={stand_weight}; exact parent 1999 resume; all other physics/actions/bank/PPO unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--stand-weight", type=int, choices=(10, 30), required=True)
    args = parser.parse_args()
    check(args.candidate_dir, args.num_envs, args.iterations, args.stand_weight)
