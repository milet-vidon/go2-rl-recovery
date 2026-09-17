"""Matched300-update final4246 evaluator; unchanged900-step retention physics.

The frozen100-update candidate verifier is privately specialized for one arm
and its exact final budget. The underlying simulator source and acceptance
thresholds are not edited. Pending trainer review fails closed before use.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/evaluate_common_physics_candidate.py"
TEMPLATE_SHA = "f62f2230742b868c71f68b9241ef2ea9a9099a699c233303a5c4983051857ce1"
TRAINER = ROOT / "scripts/train_balanced_gait_adaptation.py"
TRAINER_SHA = "574e9cac8a5413293df898b6802fe5c0d1a3da9d34a00c0bb084b7e396dc79c5"
PROTOCOL = "trained_balanced_gait_candidate_retention_v1"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen candidate anchor changed: " + old[:100])
    return source.replace(old, new, 1)


def program(arm):
    if arm not in ("control", "balanced"):
        raise ValueError("Select control or balanced arm")
    if hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Frozen candidate template changed")
    if len(TRAINER_SHA) != 64 or any(c not in "0123456789abcdef" for c in TRAINER_SHA):
        raise ValueError("Balanced trainer SHA is pending review; evaluation is disabled")
    if hashlib.sha256(TRAINER.read_bytes()).hexdigest() != TRAINER_SHA:
        raise ValueError("Reviewed balanced trainer changed")
    source = TEMPLATE.read_text(encoding="utf-8")
    weight = 0.0 if arm == "control" else -10.0
    changes = [
        ('import train_common_physics_adaptation as training',
         f'import train_balanced_gait_adaptation as matched_training\ntraining = matched_training.adapter("{arm}")\nARM = "{arm}"\nGAIT_WEIGHT = {weight}'),
        ('TRAINER_SHA = "fa6551d82994273ad8d0ea1cd2234cb0743454066af43706a1187a87db21387f"',
         f'TRAINER_SHA = "{TRAINER_SHA}"'),
        ('PROTOCOL = "trained_common_physics_candidate_retention_v1"', f'PROTOCOL = "{PROTOCOL}"'),
        ('checkpoint == path.parent / "model_4046.pt"', 'checkpoint == path.parent / "model_4246.pt"'),
        ('Only final formal4046 is allowed, not a smoke/selection', 'Only final formal4246 is allowed, not a smoke/selection'),
        ('("updates", 100), ("actual_environment_steps", 307200)', '("updates", 300), ("actual_environment_steps", 921600)'),
        ('("actual_control_steps", 2400)', '("actual_control_steps", 7200)'),
        ('    sources, parent = training.all_sources(), training.parent_metadata()',
         '''    require(receipt.get("arm") == ARM and type(receipt.get("balanced_duration_weight")) in (int, float)
            and receipt["balanced_duration_weight"] == GAIT_WEIGHT, "Training arm/weight mismatch")
    sources, parent = training.all_sources(), training.parent_metadata()'''),
        ('training.check_final(actual, parent, 100)', 'training.check_final(actual, parent, 300)'),
        ('    smoke = receipt["smoke_evidence"]',
         '''    smoke = receipt["smoke_evidence"]
    smoke_receipt = read_json(smoke["path"])
    require(smoke_receipt.get("arm") == ARM and type(smoke_receipt.get("balanced_duration_weight")) in (int, float)
            and smoke_receipt["balanced_duration_weight"] == GAIT_WEIGHT, "Smoke arm/weight mismatch")'''),
        ('        128, 100, receipt["run_name"])', '        128, 300, receipt["run_name"])'),
        ('"parent_control_sha256": training.PARENT_SHA, "formal_updates_verified": 100,',
         '"parent_control_sha256": training.PARENT_SHA, "formal_updates_verified": 300,\n            "arm": ARM, "balanced_duration_weight": GAIT_WEIGHT,'),
        ('"actual_environment_steps": 307200, "actual_iteration": actual["iter"],',
         '"actual_environment_steps": 921600, "actual_control_steps": 7200, "actual_iteration": actual["iter"],'),
        ('types.ModuleType("_common_physics_candidate_adapter")', 'types.ModuleType("_balanced_gait_candidate_adapter_" + ARM)'),
    ]
    for old, new in changes:
        source = replace_once(source, old, new)
    compile(source, str(Path(__file__)) + "::<matched-gait-candidate>", "exec")
    return source


def adapter(arm):
    module = types.ModuleType("_balanced_gait_candidate_" + arm)
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(program(arm), module.__file__, "exec"), module.__dict__)
    return module


def candidate_receipt(path, checkpoint, arm):
    return adapter(arm).candidate_receipt(path, checkpoint)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", required=True, choices=("control", "balanced"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    adapter(args.arm).main()


if __name__ == "__main__":
    main()
