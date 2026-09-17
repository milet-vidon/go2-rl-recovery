"""Finite matched slip-weight research continuation; never model promotion.

Both arms independently resume balanced4246 (a REJECTED broader-gait candidate)
including actor/critic/std/Adam. A=-.5 and B=-1 existing foot_slip; balanced
duration remains -10. Smoke16x2, formal128x300, seed42, final checkpoint only.

Formal execution requires the pinned independent continuous prerequisite
validator to freshly audit the actual new full report. The old21-case screen
or a video pairing receipt cannot unlock this guard. No automatic run follows.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/train_balanced_gait_adaptation.py"
TEMPLATE_SHA = "574e9cac8a5413293df898b6802fe5c0d1a3da9d34a00c0bb084b7e396dc79c5"
VERIFIER = ROOT / "scripts/evaluate_balanced_gait_candidate.py"
VERIFIER_SHA = "e42f2f34f1968e1bf99f6c1dc13aa1b53463fa22893db485b23dcad75df4c8fe"
PARENT_RUN = "2026-09-17_16-33-35_20260917-gait-balanced-128x300-first"
PARENT_DIR = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat") / PARENT_RUN
PARENT = PARENT_DIR / "model_4246.pt"
PARENT_SHA = "882baefd00193c4b71bef2a47b5cb7382c37dd9e87b48f7f5863eaa364baebdc"
PARENT_RECEIPT = PARENT_DIR / "common_physics_training_result.json"
PARENT_RECEIPT_SHA = "299b45ea7c89e10107c4273e1d4867e546c3017bb8e18ee48cc9664427c91b9e"
PARENT_CONFIG_SHA = {"env.yaml": "7345a2df4932258222c4cd4d8848d65623d9fb6844f082063db6e18cda7f273c",
                     "agent.yaml": "dac3c38f41352b65eb627b1f0a0a2e3ac479455a4bbd55f623d0ababf91ffd4e"}
PROTOCOL = "matched_commonphysics_slip_training_v1"
WEIGHTS = {"A": -0.5, "B": -1.0}

# A separate full-episode evidence guard is required, not the paired-video tool.
# Frozen after root reviewed16CPU tests and actual1004-interval v3 audit PASS.
# This is a bounded training prerequisite, not natural-gait/full-user acceptance.
# Changing it invalidates all same-source smoke snapshots.
CONTINUOUS_GATE = ROOT / "scripts/audit_continuous_flow.py"
CONTINUOUS_GATE_SHA = "a23733f3cb2c4ebb08e360a0ace0ae182f3095f1a0f7e0904db732a07bb93f7d"
CONTINUOUS_PROTOCOL = "continuous_recovery_flow_diagnostic_v3_sensor_identity"
REFINER = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_handoff_stand/20260917-natural-stand128x200-first/model_3746.pt")
REFINER_SHA = "f40772c228996400a9813ad244b48509737dabf2c31af001144de34283433b9e"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field: " + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=bad)


def replace_once(source, old, new):
    require(source.count(old) == 1, "Frozen slip adapter anchor changed: " + old[:100])
    return source.replace(old, new, 1)


def validate_parent():
    """Re-run the actual full formal/smoke/config/source verifier, not a label."""
    for path, expected in ((TEMPLATE, TEMPLATE_SHA), (VERIFIER, VERIFIER_SHA),
                           (PARENT, PARENT_SHA), (PARENT_RECEIPT, PARENT_RECEIPT_SHA)):
        require(sha(path) == expected, "Frozen balanced parent/source changed: " + str(path))
    for name, expected in PARENT_CONFIG_SHA.items():
        require(sha(PARENT_DIR / "params" / name) == expected, "Balanced parent config changed: " + name)
    from evaluate_balanced_gait_candidate import candidate_receipt
    from check_speed_tracking_weight_block import checkpoint_metadata
    proof = candidate_receipt(PARENT_RECEIPT, PARENT, "balanced")
    metadata = checkpoint_metadata(PARENT)
    require(metadata["iter"] == 4246 and metadata["adam_steps"] == [85120] * 17,
            "Parent must be actual balanced4246 with all17 Adam85120 states")
    require(metadata["all_tensors_finite"] is True and metadata["tensor_count"] == 68,
            "Incomplete parent state")
    rate = metadata["optimizer_group"]["lr"]
    require(type(rate) in (float, int) and Decimal(str(rate)).is_finite() and rate > 0,
            "Invalid actual parent optimizer LR")
    return metadata, proof


def formal_prerequisite(path):
    """Freshly audit full continuous evidence and bind all immutable inputs.

    Required API: audit(path) returns a JSON-serializable trace-audit ledger
    with report path/hash, audit_passed/training_prerequisite_eligible,
    source/artifact inventory, interval count and exact recomputed metrics.
    That validator must independently check the actual full rows/trace and
    source/actor interfaces; report.passed or a paired video is insufficient.
    """
    require(path is not None, "Formal requires an explicit new continuous.5 report")
    path = Path(path).resolve()
    require(path.drive.upper() == "E:" and path.name == "continuous_flow_report.json" and path.is_file(),
            "Require the actual E-drive continuous.5 report")
    require(isinstance(CONTINUOUS_GATE_SHA, str) and len(CONTINUOUS_GATE_SHA) == 64 and
            all(c in "0123456789abcdef" for c in CONTINUOUS_GATE_SHA),
            "FORMAL_DISABLED: continuous v3 full-trace prerequisite guard is not yet reviewed/pinned")
    require(CONTINUOUS_GATE.is_file() and sha(CONTINUOUS_GATE) == CONTINUOUS_GATE_SHA,
            "Reviewed continuous prerequisite guard changed/missing")
    report = read_json(path)
    require(report.get("protocol_version") == CONTINUOUS_PROTOCOL and report.get("passed") is True and
            report.get("status") == "diagnostic_passed_not_promoted" and report.get("smoke") is False and
            report.get("num_envs") == 1 and report.get("refinement") == "off" and
            report.get("flow_state", {}).get("phase") == "complete" and
            report.get("actors", {}).get("locomotion") == PARENT_SHA and
            report.get("command_schedule", {}).get("move") == [.5, 0., 0.] and
            report.get("live_interface_continuity_passed") is True and
            report.get("source_hashes_unchanged_at_end") is True and report.get("reset_guard_attempts") == [],
            "Not the newly passed real balanced4246 continuous.5 v3 diagnostic")
    spec = importlib.util.spec_from_file_location("_slip_continuous_gate", CONTINUOUS_GATE)
    require(spec is not None and spec.loader is not None, "Cannot load reviewed continuous validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    proof = module.audit(path)
    require(isinstance(proof, dict) and proof.get("protocol") == "continuous_flow_trace_audit_v1" and
            proof.get("audit_passed") is True and proof.get("training_prerequisite_eligible") is True and
            proof.get("explicit_sensor_identity_verified") is True and
            proof.get("metrics_recomputed_exact") is True and proof.get("checkpoint_sha256") == PARENT_SHA and
            proof.get("selected_model") == "balanced4246" and proof.get("promotion_performed") is False and
            proof.get("report_sha256") == sha(path) and Path(proof["report_path"]).resolve() == path and
            proof.get("source_sha256") == CONTINUOUS_GATE_SHA,
            "Continuous prerequisite did not verify actual new v3 sensor-identity evidence")
    continuous_source_records(proof)
    return json.loads(json.dumps(proof, allow_nan=False))


def continuous_source_records(proof):
    """Expand every audited input/artifact into the training start/end ledger.

    A proof's nested digest map is not merely explanatory receipt text: each
    file is checked here and then included in all_sources/verify_sources. No
    extra mutation or waiver is permitted between smoke, formal start or end.
    """
    inventory = proof.get("source_and_artifacts_sha256")
    require(type(inventory) is dict and inventory, "Missing actual continuous source/artifact inventory")
    required = {str(Path(proof["report_path"]).resolve()): proof["report_sha256"],
                str(CONTINUOUS_GATE.resolve()): CONTINUOUS_GATE_SHA}
    canonical, aliases = {}, set()
    for name, digest in inventory.items():
        require(type(name) is str and type(digest) is str and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest), "Invalid continuous inventory item")
        path = Path(name)
        require(path.is_absolute() and path.drive.upper() == "E:" and path.resolve().drive.upper() == "E:",
                "Continuous evidence inventory must remain on E:")
        key = str(path.resolve())
        require(key.casefold() not in aliases, "Duplicate/aliased continuous inventory path")
        aliases.add(key.casefold())
        require(path.is_file() and sha(path) == digest, "Audited continuous source/artifact drift: " + key)
        canonical[key] = digest
    require(all(canonical.get(path) == digest for path, digest in required.items()),
            "Continuous inventory omits its actual report or pinned auditor")
    return [{"path": key, "sha256": digest} for key, digest in sorted(canonical.items())]


def program(arm):
    require(arm in WEIGHTS, "Arm must be A or B")
    require(sha(TEMPLATE) == TEMPLATE_SHA, "Frozen balanced trainer changed")
    import train_balanced_gait_adaptation as balanced
    source = balanced.program("balanced")
    changes = [
        ('PROTOCOL = "balanced_gait_common_physics_training_v1"\nARM = "balanced"\nGAIT_WEIGHT = -10.0',
         f'PROTOCOL = "{PROTOCOL}"\nARM = "{arm}"\nGAIT_WEIGHT = -10.0\nSLIP_WEIGHT = {WEIGHTS[arm]}'),
        ('r"20260917-gait-balanced-[A-Za-z0-9_-]+"', f'r"20260917-slip-{arm}-[A-Za-z0-9_-]+"'),
        ('parent_agent["run_name"] == "20260917-speedretention128x300"',
         'parent_agent["run_name"] == "20260917-gait-balanced-128x300-first"'),
        ('parent_agent["load_checkpoint"] == "model_3648.pt"', 'parent_agent["load_checkpoint"] == "model_3947.pt"'),
        ('load_checkpoint="model_3947.pt")', 'load_checkpoint="model_4246.pt")'),
        ('number_is(rate, Decimal("0.00001"))', 'number_is(rate, Decimal(str(PARENT_LR)))'),
        ('"Initial scalar LR must match actual parent LR1e-5"', '"Initial scalar LR must match actual balanced4246 optimizer"'),
        ('    return expected\n\n\ndef actual_yaml_changes',
         '    expected["rewards"]["foot_slip"]["weight"] = str(SLIP_WEIGHT)\n    return expected\n\n\ndef actual_yaml_changes'),
        ('    require(cfg.observations.policy.enable_corruption and cfg.events.push_robot is not None,',
         '    require(cfg.rewards.foot_slip.weight == -0.5, "Original foot_slip weight changed")\n    cfg.rewards.foot_slip.weight = SLIP_WEIGHT\n    require(cfg.observations.policy.enable_corruption and cfg.events.push_robot is not None,'),
        ('metadata["iter"] == 3947 + updates - 1 and metadata["adam_steps"] == [79120 + 20 * updates] * 17',
         'metadata["iter"] == 4246 + updates - 1 and metadata["adam_steps"] == [85120 + 20 * updates] * 17'),
        ('checkpoint.name == "model_3948.pt"', 'checkpoint.name == "model_4247.pt"'),
        ('before.get("initial_iteration") == 3947', 'before.get("initial_iteration") == 4246'),
        ('before.get("initial_adam_step") == 79120', 'before.get("initial_adam_step") == 85120'),
        ('before.get("initial_scalar_and_optimizer_lr") == 1e-5', 'before.get("initial_scalar_and_optimizer_lr") == PARENT_LR'),
        ('runner.current_learning_iteration == 3947 and runner.alg.learning_rate == 1e-5',
         'runner.current_learning_iteration == 4246 and runner.alg.learning_rate == PARENT_LR'),
        ('runner.alg.optimizer.param_groups[0]["lr"] == 1e-5', 'runner.alg.optimizer.param_groups[0]["lr"] == PARENT_LR'),
        ('"initial_iteration": 3947, "initial_adam_step": 79120', '"initial_iteration": 4246, "initial_adam_step": 85120'),
        ('"initial_scalar_and_optimizer_lr": 1e-5', '"initial_scalar_and_optimizer_lr": PARENT_LR'),
        ('f"model_{3947 + args.max_iterations - 1}.pt"', 'f"model_{4246 + args.max_iterations - 1}.pt"'),
        ('"expected_final_iteration": 3947 + args.max_iterations - 1', '"expected_final_iteration": 4246 + args.max_iterations - 1'),
        ('"expected_adam_step": 79120 + 20 * args.max_iterations', '"expected_adam_step": 85120 + 20 * args.max_iterations'),
        ('"Smoke always starts independently from original3947"', '"Smoke independently starts from balanced4246, never from smoke weights"'),
        ('    if args.max_iterations == 300:\n',
         '''    if args.max_iterations == 300:
        require(STATE.get("continuous_prerequisite") is not None, "Formal cannot bypass the new continuous prerequisite")
        require(_slip_verify_continuous(STATE["continuous_prerequisite"]["report_path"]) == STATE["continuous_prerequisite"],
                "Formal continuous evidence changed")
'''),
        ('"only_declared_physics_and_run_metadata_changed": True', '"only_declared_slip_reward_and_run_metadata_changed": True'),
        ('"original_reward_command_noise_reset_push_ppo_preserved": True',
         '"other_rewards_command_noise_reset_push_ppo_preserved": True'),
        ('"balanced_duration_weight": GAIT_WEIGHT, "actual_yaml_differences": changes,',
         '"balanced_duration_weight": GAIT_WEIGHT, "foot_slip_weight": SLIP_WEIGHT, "actual_yaml_differences": changes,'),
        ('relative.startswith("rewards.balanced_duration")',
         '(relative.startswith("rewards.balanced_duration") or relative == "rewards.foot_slip.weight")'),
        ('"Pinned TRAIN parent already has fixed material .8/.6/0 and no CoM event. Those are enforced invariants, not new interventions. Actual changes are self-collision, action implementation/soft clamp, and removal of base-mass randomization."',
         '"Parent already uses common physics and balanced_duration=-10. All actual physics/other reward differences are forbidden; treatment changes existing foot_slip only."'),
        ('"note": "Matched independent common-physics128x300 control/timing-variance experiment. Original rewards and command/noise/reset/push/PPO retained; only added term weight differs between arms. Final checkpoint only; no automatic continuation/promotion, gait or full-flow claim."',
         '"note": "Matched finite slip-weight research from rejected broader-gait balanced4246. A=-.5/B=-1 existing foot_slip; all other common physics/rewards unchanged. Final only; no automatic continuation/promotion, natural gait, fast-run or full-flow acceptance."'),
    ]
    for old, new in changes:
        source = replace_once(source, old, new)
    compile(source, str(Path(__file__)) + "::<slip-adapter>", "exec")
    return source


def adapter(arm, continuous=None):
    """Private specialization; old modules, files and their guards are untouched."""
    # Detach caller-owned dictionaries before the experiment freezes evidence.
    continuous = json.loads(json.dumps(continuous, allow_nan=False))
    result = types.ModuleType("_matched_slip_" + arm)
    result.__file__ = str(Path(__file__).resolve())
    exec(compile(program(arm), result.__file__, "exec"), result.__dict__)
    parent, proof = validate_parent()
    result.PARENT, result.PARENT_SHA, result.PARENT_RUN = PARENT, PARENT_SHA, PARENT_RUN
    result.PARENT_CONFIG = PARENT_DIR / "params"
    result.PARENT_LR = parent["optimizer_group"]["lr"]
    result.parent_metadata = lambda: validate_parent()[0]
    result._slip_verify_continuous = formal_prerequisite
    result.STATE.update(parent_training_verification=proof, continuous_prerequisite=continuous)
    old_documents = result.check_documents
    def checked_documents(*args):
        guard = old_documents(*args)
        require(not guard["actual_physics_changed_paths"], "Slip continuation cannot change parent physics")
        reward_changes = [row for row in guard["actual_yaml_differences"] if row["path"].startswith("rewards.")]
        expected_paths = [] if arm == "A" else ["rewards.foot_slip.weight"]
        require([row["path"] for row in reward_changes] == expected_paths, "More than existing slip scalar changed")
        return guard
    result.check_documents = checked_documents
    old_sources = result.all_sources
    def sources():
        validate_parent()
        inventory = old_sources()
        additions = [TEMPLATE, VERIFIER, PARENT, PARENT_RECEIPT,
                     REFINER,
                     ROOT / "scripts/run_commonphysics_slip_adaptation.ps1",
                     ROOT / "scripts/test_commonphysics_slip_adaptation.py",
                     *[PARENT_DIR / "params" / name for name in PARENT_CONFIG_SHA]]
        if continuous is not None:
            refreshed = formal_prerequisite(continuous["report_path"])
            require(refreshed == continuous, "Continuous prerequisite changed after selection")
            inventory.extend(continuous_source_records(continuous))
        require(sha(REFINER) == REFINER_SHA, "Frozen posture refiner changed")
        indexed = {}
        for row in inventory:
            key, digest = str(Path(row["path"]).resolve()), row["sha256"]
            require(key not in indexed or indexed[key] == digest, "Conflicting continuous/base source digest")
            indexed[key] = digest
        for path in additions:
            key, digest = str(path.resolve()), sha(path)
            require(key not in indexed or indexed[key] == digest, "Conflicting source digest")
            indexed[key] = digest
        return [{"path": key, "sha256": value} for key, value in sorted(indexed.items())]
    result.all_sources = sources
    old_write = result.write_new
    def write(path, value):
        value = copy.deepcopy(value)
        if Path(path).name in ("common_physics_invocation.json", "common_physics_pretrain_guard.json",
                               "common_physics_training_result.json"):
            value.update(arm=arm, foot_slip_weight=WEIGHTS[arm], balanced_duration_weight=-10.,
                         parent_training_verification=proof, parent_training_receipt_sha256=PARENT_RECEIPT_SHA,
                         continuous_prerequisite=continuous, research_candidate_not_natural_gait=True)
        old_write(path, value)
    result.write_new = write
    old_smoke = result.verify_smoke
    def smoke(path, snapshot):
        verified = old_smoke(path, snapshot)
        ledger = read_json(path)
        require(ledger.get("arm") == arm and ledger.get("foot_slip_weight") == WEIGHTS[arm] and
                ledger.get("balanced_duration_weight") == -10. and
                ledger.get("parent_training_receipt_sha256") == PARENT_RECEIPT_SHA and
                ledger.get("parent_training_verification") == proof and
                ledger.get("continuous_prerequisite") == continuous,
                "Independent smoke arm/parent/continuous prerequisite differs")
        return verified
    result.verify_smoke = smoke
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", required=True, choices=tuple(WEIGHTS))
    parser.add_argument("--continuous_report", type=Path)
    parser.add_argument("--max_iterations", type=int, required=True)
    args, remaining = parser.parse_known_args()
    # A valid full-flow prerequisite can be supplied to smoke so its exact
    # evidence/source inventory matches formal. Ungated smoke cannot unlock it.
    continuous = formal_prerequisite(args.continuous_report) if args.continuous_report is not None or args.max_iterations == 300 else None
    sys.argv = [sys.argv[0], "--max_iterations", str(args.max_iterations), *remaining]
    adapter(args.arm, continuous).main()


if __name__ == "__main__":
    main()
