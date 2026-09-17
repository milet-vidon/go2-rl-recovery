"""Final4545 slip-weight research screen; frozen physical metrics, no promotion.

Both arms require their actual same-arm full300-update receipt and independent
smoke, not an old balanced21-case summary. The simulator/metric source remains
byte-identical to the established retention evaluator. Common18/original15
case selection is declared in the separate finite launcher.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import types

import evaluate_balanced_gait_candidate as balanced
import train_commonphysics_slip_adaptation as slip

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SHA = "e42f2f34f1968e1bf99f6c1dc13aa1b53463fa22893db485b23dcad75df4c8fe"
TRAINER_SHA = "fb053b54539ec37c8c5047a3b2c27a3cee4f42d83d32ff401979957f3a7fbf60"
PROTOCOL = "trained_commonphysics_slip_candidate_retention_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def program(arm):
    require(arm in slip.WEIGHTS, "Select slip arm A or B")
    require(slip.sha(balanced.__file__) == TEMPLATE_SHA, "Frozen balanced evaluator changed")
    require(slip.sha(slip.__file__) == TRAINER_SHA, "Frozen slip trainer changed")
    source = balanced.program("balanced")
    changes = [
        ('import train_balanced_gait_adaptation as matched_training\ntraining = matched_training.adapter("balanced")\nARM = "balanced"\nGAIT_WEIGHT = -10.0',
         f'training = _verified_slip_training\nARM = "{arm}"\nGAIT_WEIGHT = -10.0\nSLIP_WEIGHT = {slip.WEIGHTS[arm]}'),
        (f'TRAINER_SHA = "{balanced.TRAINER_SHA}"', f'TRAINER_SHA = "{TRAINER_SHA}"'),
        (f'PROTOCOL = "{balanced.PROTOCOL}"', f'PROTOCOL = "{PROTOCOL}"'),
        ('checkpoint == path.parent / "model_4246.pt"', 'checkpoint == path.parent / "model_4545.pt"'),
        ('Only final formal4246 is allowed, not a smoke/selection', 'Only final formal4545 is allowed, not a smoke/selection'),
        ('before["initial_iteration"] == 3947', 'before["initial_iteration"] == 4246'),
        ('before["initial_adam_step"] == 79120', 'before["initial_adam_step"] == 85120'),
        ('before["initial_scalar_and_optimizer_lr"] == 1e-5', 'before["initial_scalar_and_optimizer_lr"] == training.PARENT_LR'),
        ('    return {"training_receipt": str(path),',
         '    _verify_slip_ledgers(path, receipt, training, _continuous_proof)\n    return {"training_receipt": str(path),'),
        ('"arm": ARM, "balanced_duration_weight": GAIT_WEIGHT,',
         '"arm": ARM, "balanced_duration_weight": GAIT_WEIGHT, "foot_slip_weight": SLIP_WEIGHT,\n'
         '            "parent_training_receipt_sha256": matched_parent_receipt_sha,\n'
         '            "continuous_prerequisite": copy.deepcopy(_continuous_proof),\n'
         '            "research_candidate_not_natural_gait": True,'),
        ('types.ModuleType("_balanced_gait_candidate_adapter_" + ARM)',
         'types.ModuleType("_slip_candidate_adapter_" + ARM)'),
    ]
    for old, new in changes:
        source = slip.replace_once(source, old, new)
    compile(source, str(Path(__file__)) + "::<slip-candidate>", "exec")
    return source


def verify_slip_ledgers(path, receipt, training, continuous):
    """Bind initial, pre-learning, final, smoke and parent evidence together."""
    parent_proof = training.STATE["parent_training_verification"]
    ledger_paths = [path, path.parent / "common_physics_invocation.json",
                    path.parent / "common_physics_pretrain_guard.json",
                    Path(receipt["smoke_evidence"]["path"])]
    for ledger_path in ledger_paths:
        ledger = slip.read_json(ledger_path)
        require(ledger.get("arm") == training.ARM and
                type(ledger.get("foot_slip_weight")) in (float, int) and
                ledger["foot_slip_weight"] == slip.WEIGHTS[training.ARM] and
                type(ledger.get("balanced_duration_weight")) in (float, int) and
                ledger["balanced_duration_weight"] == -10. and
                ledger.get("parent_training_receipt_sha256") == slip.PARENT_RECEIPT_SHA and
                ledger.get("parent_training_verification") == parent_proof and
                ledger.get("continuous_prerequisite") == continuous and
                ledger.get("research_candidate_not_natural_gait") is True,
                "Actual initial/final/smoke slip lineage or continuous proof differs")
    require(slip.formal_prerequisite(continuous["report_path"]) == continuous,
            "Actual continuous prerequisite changed during candidate verification")


def adapter(arm, continuous):
    require(continuous is not None, "Candidate requires actual continuous v3 prerequisite")
    # No simulator is started here; full parent checkpoints/configs are read.
    training = slip.adapter(arm, continuous)
    result = types.ModuleType("_slip_candidate_" + arm)
    result.__file__ = str(Path(__file__).resolve())
    result.__dict__.update(_verified_slip_training=training,
                           _continuous_proof=copy.deepcopy(continuous),
                           _verify_slip_ledgers=verify_slip_ledgers,
                           matched_parent_receipt_sha=slip.PARENT_RECEIPT_SHA)
    exec(compile(program(arm), result.__file__, "exec"), result.__dict__)
    return result


def candidate_receipt(path, checkpoint, arm):
    require(arm in slip.WEIGHTS, "Select slip arm A or B")
    path = Path(path).resolve()
    require(path.drive.upper() == "E:" and path.name == "common_physics_training_result.json",
            "Require an actual E-drive slip training receipt")
    ledger = slip.read_json(path)
    require(ledger.get("protocol") == slip.PROTOCOL and ledger.get("arm") == arm and
            ledger.get("completion_verified") is True,
            "Wrong/incomplete slip arm training receipt")
    saved = ledger.get("continuous_prerequisite")
    require(type(saved) is dict and "report_path" in saved, "Missing actual continuous prerequisite")
    continuous = slip.formal_prerequisite(saved["report_path"])
    require(saved == continuous, "Saved continuous proof differs from freshly audited evidence")
    return adapter(arm, continuous).candidate_receipt(path, checkpoint)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=tuple(slip.WEIGHTS), required=True)
    parser.add_argument("--candidate_training_result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args, remaining = parser.parse_known_args()
    receipt = candidate_receipt(args.candidate_training_result, args.checkpoint, args.arm)
    if args.verify_only:
        import json
        print(json.dumps(receipt, indent=2, allow_nan=False))
        return
    entry = adapter(args.arm, receipt["continuous_prerequisite"])
    physical = entry.make_adapter(receipt)
    generated = physical.build_source()
    require(generated == entry.frozen.build_source(), "Slip adapter changed physical evaluation code")
    sys.argv = [sys.argv[0], "--checkpoint", str(args.checkpoint), *remaining]
    exec(compile(generated, str(Path(__file__)) + "::<slip-retention>", "exec"),
         {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
          "_retention_adapter": physical, "_retention_generated_source": generated})


if __name__ == "__main__":
    main()
