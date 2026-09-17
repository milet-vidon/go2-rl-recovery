"""Evaluate only the verified final combined-physics100-update candidate.

The frozen retention evaluator's dynamics, reset, commands and metrics are
unchanged. A process-local adapter changes only the authenticated checkpoint
identity and explicitly labelled candidate provenance. No baseline is edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import evaluate_locomotion_recovery_physics as frozen
import train_common_physics_adaptation as training

BASE_SHA = "cc2437ffb0d84da97eda5906333e54070f2d065ea475a148b6be7f1c6cfea4c8"
TRAINER_SHA = "fa6551d82994273ad8d0ea1cd2234cb0743454066af43706a1187a87db21387f"
PROTOCOL = "trained_common_physics_candidate_retention_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def json_value(value):
    """JSON transport changes tuples to lists; preserve every numeric value."""
    return json.loads(json.dumps(value, allow_nan=False))


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject, object_pairs_hook=unique)


def candidate_receipt(path, checkpoint):
    """Verify actual final model/config/source/stage artifacts, not a success label."""
    path, checkpoint = Path(path).resolve(), Path(checkpoint).resolve()
    require(frozen.sha(frozen.__file__) == BASE_SHA, "Frozen retention adapter changed")
    require(frozen.sha(training.__file__) == TRAINER_SHA, "Reviewed training adapter changed")
    require(path.drive.upper() == "E:" and path.parent.parent == training.RUN_ROOT.resolve() and
            path.name == "common_physics_training_result.json", "Require isolated E-drive training result")
    require(checkpoint == path.parent / "model_4046.pt", "Only final formal4046 is allowed, not a smoke/selection")
    receipt = read_json(path)
    require(receipt.get("protocol") == training.PROTOCOL and receipt.get("completion_verified") is True,
            "Incomplete or wrong training protocol")
    for name, expected in (("num_envs", 128), ("updates", 100), ("actual_environment_steps", 307200),
                           ("actual_control_steps", 2400)):
        require(type(receipt.get(name)) is int and receipt[name] == expected, "Wrong formal budget: " + name)
    for name in ("quality_accepted", "promotion_performed", "hardware_execution"):
        require(receipt.get(name) is False, "Training cannot supply behavior acceptance: " + name)
    require(Path(receipt["checkpoint"]).resolve() == checkpoint and
            Path(receipt["parent_path"]).resolve() == training.PARENT.resolve(), "Training lineage paths differ")
    sources, parent = training.all_sources(), training.parent_metadata()
    require(receipt["source_snapshot"] == sources and receipt["parent"] == json_value(parent),
            "Actual parent/source inventory differs from training receipt")
    actual = training.checkpoint_metadata(checkpoint)
    require(json_value(actual) == receipt["checkpoint_metadata"], "Actual candidate bytes/state differ from receipt")
    training.check_final(actual, parent, 100)
    smoke = receipt["smoke_evidence"]
    require(training.verify_smoke(smoke["path"], sources) == smoke, "Independent same-source smoke evidence differs")
    guard = training.check_documents(training.read(path.parent / "params/env.yaml"),
        training.read(path.parent / "params/agent.yaml"), training.read(training.PARENT_CONFIG / "env.yaml"),
        training.read(training.PARENT_CONFIG / "agent.yaml"), training.reference_document(),
        128, 100, receipt["run_name"])
    require(guard == receipt["config_guard"], "Actual full training config differs")
    expected_yaml = [{"path": str(path.parent / "params" / name),
                      "sha256": frozen.sha(path.parent / "params" / name)} for name in ("env.yaml", "agent.yaml")]
    require(receipt["actual_yaml"] == expected_yaml, "Training YAML bytes differ")
    before = read_json(path.parent / "common_physics_pretrain_guard.json")
    require(before.get("passed") is True and before.get("parent_model_and_adam_exact") is True and
            before["config_guard"] == guard and before["actual_yaml"] == expected_yaml and
            before["parent_sha256"] == training.PARENT_SHA and before["initial_iteration"] == 3947 and
            before["initial_adam_step"] == 79120 and before["initial_scalar_and_optimizer_lr"] == 1e-5,
            "Missing/mismatched exact-load pre-learning guard")
    invocation = read_json(path.parent / "common_physics_invocation.json")
    for name in ("protocol", "parent", "parent_path", "run_name", "num_envs", "updates", "source_snapshot", "smoke_evidence"):
        require(invocation[name] == receipt[name], "Initial/final training ledger differs: " + name)
    generated = training.build_source(training.OFFICIAL.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(generated.encode()).hexdigest()
    require(frozen.sha(path.parent / "training_source_generated.py") == expected_sha ==
            receipt["generated_sha256"] == invocation["generated_sha256"], "Actual generated trainer differs")
    return {"training_receipt": str(path), "training_receipt_sha256": frozen.sha(path),
            "checkpoint": str(checkpoint), "checkpoint_sha256": actual["sha256"],
            "parent_control_sha256": training.PARENT_SHA, "formal_updates_verified": 100,
            "actual_environment_steps": 307200, "actual_iteration": actual["iter"],
            "actual_adam_steps": actual["adam_steps"], "training_source_sha256": TRAINER_SHA,
            "model_and_optimizer_all_finite": actual["all_tensors_finite"],
            "quality_accepted": False, "promotion_performed": False,
            "scope": "Source/config/full checkpoint consistency; behavior must be independently evaluated"}


def make_adapter(receipt):
    """A private module copy; frozen module globals and physical functions stay intact."""
    require(frozen.sha(frozen.__file__) == BASE_SHA, "Frozen retention adapter changed")
    adapter = types.ModuleType("_common_physics_candidate_adapter")
    source = Path(frozen.__file__).read_text(encoding="utf-8")
    # Not __main__: importing the existing pure helper declarations starts no sim.
    adapter.__file__ = str(Path(frozen.__file__).resolve())
    exec(compile(source, adapter.__file__, "exec"), adapter.__dict__)
    adapter.CONTROL = Path(receipt["checkpoint"])
    adapter.CONTROL_SHA = receipt["checkpoint_sha256"]
    adapter.PROTOCOL = PROTOCOL
    original_finish = adapter.finish_report
    def finish(report, args, config, interface, records, generated):
        original_finish(report, args, config, interface, records, generated)
        evidence = report["retention_experiment"]
        require(evidence.pop("frozen_control_sha256") == receipt["checkpoint_sha256"], "Wrong issued actor")
        evidence.update(candidate_checkpoint_sha256=receipt["checkpoint_sha256"],
                        parent_control_sha256=training.PARENT_SHA, frozen_adapter_sha256=BASE_SHA,
                        adapter_sha256=frozen.sha(__file__),
                        claim="Trained candidate screen only; no fallen recovery, fast-running or full-flow acceptance")
        report["candidate_training"] = copy.deepcopy(receipt)
    adapter.finish_report = finish
    return adapter


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate_training_result", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args, remaining = parser.parse_known_args()
    receipt = candidate_receipt(args.candidate_training_result, args.checkpoint)
    if args.verify_only:
        print(json.dumps(receipt, indent=2))
        return
    adapter = make_adapter(receipt)
    generated = adapter.build_source()
    require(generated == frozen.build_source(), "Candidate adapter changed physical evaluator code")
    sys.argv = [sys.argv[0], "--checkpoint", str(args.checkpoint), *remaining]
    exec(compile(generated, str(Path(__file__)) + "::<candidate-retention>", "exec"),
         {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
          "_retention_adapter": adapter, "_retention_generated_source": generated})


if __name__ == "__main__":
    main()
