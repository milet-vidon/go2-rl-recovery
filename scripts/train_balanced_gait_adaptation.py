"""Matched, finite common-physics control/timing-variance experiment.

Both arms independently resume full control3947; smoke16x2 then128x300.
The only treatment is a command-gated completed-duration variance term.
Existing physics/old rewards/commands/noise/push/PPO and source guards remain.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/train_common_physics_adaptation.py"
TEMPLATE_SHA = "fa6551d82994273ad8d0ea1cd2234cb0743454066af43706a1187a87db21387f"
PROTOCOL = "balanced_gait_common_physics_training_v1"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen adapter anchor changed: " + old[:100])
    return source.replace(old, new, 1)


def program(arm):
    if arm not in ("control", "balanced"):
        raise ValueError("Select control or balanced")
    if hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Frozen adaptation trainer changed")
    source = TEMPLATE.read_text(encoding="utf-8")
    weight = "0.0" if arm == "control" else "-10.0"
    changes = [
        ('PROTOCOL = "bounded_common_physics_adaptation_training_v1"',
         f'PROTOCOL = "{PROTOCOL}"\nARM = "{arm}"\nGAIT_WEIGHT = {weight}'),
        ('BUDGETS = {(16, 2), (128, 100)}', 'BUDGETS = {(16, 2), (128, 300)}'),
        ('r"20260917-commonphysics[A-Za-z0-9_-]+"', f'r"20260917-gait-{arm}-[A-Za-z0-9_-]+"'),
        ('"Only independent smoke16x2 or one formal128x100 block"',
         '"Only independent smoke16x2 or one formal128x300 block"'),
        ('updates in (2, 100)', 'updates in (2, 300)'),
        ('if args.max_iterations == 100:', 'if args.max_iterations == 300:'),
        ('    return expected\n\n\ndef actual_yaml_changes',
         '''    expected["rewards"]["balanced_duration"] = {
        "func": "balanced_gait_reward:balanced_gait_duration_reward", "params": {}, "weight": str(GAIT_WEIGHT)}
    return expected


def actual_yaml_changes'''),
        ('    require(cfg.observations.policy.enable_corruption and cfg.events.push_robot is not None,',
         '''    from isaaclab.managers import RewardTermCfg
    from balanced_gait_reward import balanced_gait_duration_reward
    cfg.rewards.balanced_duration = RewardTermCfg(func=balanced_gait_duration_reward, weight=GAIT_WEIGHT)
    require(cfg.observations.policy.enable_corruption and cfg.events.push_robot is not None,'''),
        ('ROOT / "scripts/run_common_physics_adaptation.ps1", ROOT / "scripts/test_common_physics_adaptation.py"',
         '''ROOT / "scripts/run_balanced_gait_adaptation.ps1", ROOT / "scripts/test_balanced_gait_adaptation.py",
             ROOT / "scripts/train_common_physics_adaptation.py", ROOT / "scripts/run_common_physics_adaptation.ps1",
             ROOT / "scripts/balanced_gait_reward.py", ROOT / "scripts/test_balanced_gait_reward.py",
             ROOT / "scripts/balanced_gait_exposure.py", ROOT / "scripts/test_balanced_gait_exposure.py"'''),
        ('"training_task_not_play": True, "actual_yaml_differences": changes,',
         '"training_task_not_play": True, "arm": ARM, "balanced_duration_weight": GAIT_WEIGHT, "actual_yaml_differences": changes,'),
        ('"kind": "declared_physics" if physical else "run_metadata"',
         '"kind": "declared_physics" if physical else ("declared_reward" if relative.startswith("rewards.balanced_duration") else "run_metadata")'),
        ('    STATE["runner"] = runner',
         '''    from balanced_gait_exposure import DurationExposure
    STATE["exposure"] = DurationExposure(env, path / "gait_exposure.jsonl")
    STATE["runner"] = runner'''),
        ('    # Adam moments/std may adapt, but never silently discard the full-state parent.',
         '''    exposure = STATE["exposure"].finish()
    write_new(path / "gait_exposure_summary.json", exposure)
    # Adam moments/std may adapt, but never silently discard the full-state parent.'''),
        ('"quality_accepted": False, "promotion_performed": False, "hardware_execution": False,',
         '"arm": ARM, "balanced_duration_weight": GAIT_WEIGHT, "gait_exposure": exposure,\n        "quality_accepted": False, "promotion_performed": False, "hardware_execution": False,'),
        ('"note": "Finite shared-physics adaptation only. Preserve commands/rewards/noise/reset/push/PPO. New recovery-physics .5/.8 stand/stop/turn/push plus original-physics regression are required; no fast-running or full-flow claim."',
         '"note": "Matched independent common-physics128x300 control/timing-variance experiment. Original rewards and command/noise/reset/push/PPO retained; only added term weight differs between arms. Final checkpoint only; no automatic continuation/promotion, gait or full-flow claim."'),
    ]
    for old, new in changes:
        source = replace_once(source, old, new)
    compile(source, str(Path(__file__)) + "::<matched-gait>", "exec")
    return source


def adapter(arm):
    result = types.ModuleType("_matched_gait_" + arm)
    result.__file__ = str(Path(__file__).resolve())
    exec(compile(program(arm), result.__file__, "exec"), result.__dict__)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", required=True, choices=("control", "balanced"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    adapter(args.arm).main()


if __name__ == "__main__":
    main()
