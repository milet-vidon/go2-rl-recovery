"""Receipt-bound evaluation of a NEW natural-stance actor in frozen combined control.

Only the standing checkpoint changes; original combined entry and actors stay
untouched. New candidate outcomes cannot be presented as original exact matches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import evaluate_handoff_combined as base

BASE_SHA = "0788beb2e8d4457f6c286d9aff80638b269e080a9b75167f9f7977456380bed0"
PROTOCOL = "natural_stance_candidate_combined_v1"
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN_ROOT = WORKSPACE / "repo/logs/rsl_rl/unitree_go2_handoff_stand"
PARENT = RUN_ROOT / "20260917-handoff-stand128x100/model_3547.pt"
PARENT_SHA = "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"
TRAINER_PINS = {
    "train_handoff_stand.py": "aee036d6cb700fb953f5627b7d37b93bc3bdbda9756591d4efb20f5869fbc88c",
    "train_natural_handoff_stand.py": "734e1643f3cbac5b17667db3e866a71041d3352e63378211ddb8103251e57c85",
    "natural_stance_reward.py": "b314df837ae5e396a9308bc3a276cf3364fabc30e4242687824a0b6a9aeac3f6",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "Duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant,
                       object_pairs_hook=unique_object)
    _require(isinstance(value, dict), "Receipt must be a JSON object")
    return value


def _receipt_header(receipt, path, stand_path, *, allow_smoke=False):
    """Pure checks; receipt claims alone never authenticate training or a model."""
    run = Path(receipt["run_dir"]).resolve()
    _require(run.parent == RUN_ROOT.resolve(), "Candidate must be in the isolated handoff run root")
    _require(path == run / "training_result.json" and stand_path.parent == run,
             "Receipt/checkpoint must belong to the same actual run")
    _require(receipt["protocol"] == "supported_symmetry_handoff_training_v1", "Wrong training protocol")
    for field in ("preflight_pass", "completion_verified", "loaded_full_parent_exact", "report_written_before_shutdown"):
        _require(receipt.get(field) is True, "Missing verified training flag: " + field)
    for field in ("promotion_performed", "hardware_execution", "recovery_acceptance_eligible", "training_ready"):
        _require(receipt.get(field) is False, "Unexpected training boundary: " + field)
    _require(type(receipt.get("intended_process_exit_code")) is int and receipt["intended_process_exit_code"] == 0,
             "Training did not complete with exit code zero")
    mode = receipt.get("mode")
    _require(mode == "formal" or (allow_smoke and mode == "smoke"),
             "Only formal training is accepted; smoke requires explicit diagnostic opt-in")
    n, updates = (128, 200) if mode == "formal" else (16, 2)
    iteration, adam_step, steps = 3547 + updates - 1, 71120 + 20 * updates, n * 24 * updates
    for key, value in {"num_envs": n, "updates": updates, "verified_new_updates": updates,
                       "expected_final_iteration": iteration, "expected_final_adam_step": adam_step,
                       "expected_environment_steps": steps, "verified_environment_steps": steps}.items():
        _require(type(receipt.get(key)) is int and receipt[key] == value, "Wrong fixed training budget: " + key)
    _require(Path(receipt["final_checkpoint"]).resolve() == stand_path and
             stand_path.name == f"model_{iteration}.pt", "Evaluate only the verified final candidate")
    parent = receipt["parent"]
    _require(Path(parent["path"]).resolve() == PARENT.resolve() and parent["sha256"] == PARENT_SHA,
             "Candidate must resume the frozen full stand3547 parent")
    _require(receipt.get("initial_algorithm_scalar_learning_rate") == 1e-5, "Wrong resumed scalar LR")
    guard = receipt["config_guard"]
    for field in ("aligned_original_rewards_preserved", "observations_terminations_commands_curriculum_equal",
                  "smith_robot_physx_dt_decimation_material_actions_equal"):
        _require(guard.get(field) is True, "Training configuration guard failed: " + field)
    _require(guard.get("supported_symmetry_weight") == 2.0 and
             guard.get("ordinary_branch_is_already_settled_standing") is False,
             "Unexpected natural reward or reset claim")
    return n, updates, iteration, adam_step


def _summary_matches(record, summary):
    for key, value in summary.items():
        _require(record.get(key) == value and (not isinstance(value, bool) or record.get(key) is value),
                 "Actual checkpoint contradicts receipt field: " + key)


def candidate_receipt(path, stand_path, *, allow_smoke=False):
    path, stand_path = Path(path).resolve(), Path(stand_path).resolve()
    if path.drive.upper() != "E:" or stand_path.drive.upper() != "E:":
        raise ValueError("All candidate artifacts stay on E:")
    receipt = _load(path)
    n, updates, iteration, adam_step = _receipt_header(receipt, path, stand_path, allow_smoke=allow_smoke)
    # Authenticate the actual, reviewed executable sources, not a self-authored JSON label.
    for name, expected in TRAINER_PINS.items():
        _require(_digest(ROOT / "scripts" / name) == expected, "Reviewed training source changed: " + name)
    import train_handoff_stand as training
    import torch
    expected_sources = {str(Path(row["path"]).resolve()): row["sha256"] for row in training.snapshot_sources()}
    for name in ("train_natural_handoff_stand.py", "natural_stance_reward.py"):
        expected_sources[str((ROOT / "scripts" / name).resolve())] = TRAINER_PINS[name]
    actual_sources = {str(Path(row["path"]).resolve()): row["sha256"] for row in receipt["source_snapshot"]}
    _require(len(actual_sources) == len(receipt["source_snapshot"]) and actual_sources == expected_sources,
             "Incomplete, stale, or changed training source snapshot")
    _require(receipt["frozen_models"] == training.verify_frozen_models(), "Protected model inventory changed")
    _require(receipt["installed_dependency_evidence"] == training.verify_installed_reference(),
             "Installed dependency evidence changed")
    # The independently written preflight and exact-load receipt must agree with the final record.
    for name in ("preflight.json", "before_learning.json"):
        stage = _load(path.parent / name)
        required_stage = {"protocol", "preflight_pass", "mode", "num_envs", "updates", "seed", "run_dir",
                          "dataset", "input_receipt", "subset_receipt", "parent", "source_snapshot", "frozen_models",
                          "installed_dependency_evidence", "expected_final_iteration", "expected_final_adam_step",
                          "expected_environment_steps", "promotion_performed", "hardware_execution",
                          "recovery_acceptance_eligible", "training_ready", "raw_dataset_manifest_unchanged"}
        _require(required_stage.issubset(stage), "Incomplete independent stage ledger: " + name)
        if name == "before_learning.json":
            _require(stage.get("completion_verified") is False and stage.get("loaded_full_parent_exact") is True,
                     "Missing pre-learning full-parent-load evidence")
            _require({"config_guard", "initial_algorithm_scalar_learning_rate"}.issubset(stage),
                     "Missing pre-learning configuration/LR evidence")
        for key, value in stage.items():
            if key != "completion_verified":
                _require(receipt.get(key) == value, "Training stage ledger differs: " + name + "/" + key)
        _require(stage.get("source_snapshot") == receipt["source_snapshot"] and stage.get("parent") == receipt["parent"],
                 "Training stage source/parent evidence is incomplete")
    arrays, manifest, selected, dataset = training.load_raw_collection(training.DEFAULT_DATASET, None, all_samples=True)
    _require(len(selected) == 79 and manifest["total_source_trials"] == 80 and dataset == receipt["dataset"],
             "Candidate does not use the frozen 79/80 TRAIN collection")
    _require(bool(training.np.isin(arrays["source_train_state_id"], training.train_ids_from_bank(manifest)).all()),
             "Heldout state in training collection")
    for key, expected, validator in (
        ("input_receipt", training.DEFAULT_INPUT_RECEIPT, lambda r: training.validate_input_receipt(r, arrays, dataset)),
        ("subset_receipt", training.DEFAULT_SUBSET_RECEIPT, lambda r: training.validate_subset_receipt(r, dataset)),
    ):
        evidence = receipt[key]
        _require(Path(evidence["path"]).resolve() == expected.resolve() and evidence["sha256"] == _digest(expected),
                 "Training input/reset receipt changed: " + key)
        validator(_load(expected))
    _require(_digest(PARENT) == PARENT_SHA, "Frozen parent bytes changed")
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    _summary_matches(receipt["parent"], training.checkpoint_summary(parent, 3547, 71120))
    matching = [r for r in receipt["checkpoints"] if Path(r["path"]).resolve() == stand_path]
    actual_sha = _digest(stand_path)
    _require(len(matching) == 1 and matching[0]["sha256"] == actual_sha and actual_sha != PARENT_SHA,
             "Candidate checkpoint identity validation failed")
    checkpoint = torch.load(stand_path, map_location="cpu", weights_only=True)
    _summary_matches(matching[0], training.checkpoint_summary(checkpoint, iteration, adam_step))
    _require(checkpoint["model_state_dict"].keys() == parent["model_state_dict"].keys(),
             "Candidate model architecture keys differ from full parent")
    for key, value in checkpoint["model_state_dict"].items():
        original = parent["model_state_dict"][key]
        _require(value.shape == original.shape and value.dtype == original.dtype,
                 "Candidate tensor layout differs from parent: " + key)
    for key, state in checkpoint["optimizer_state_dict"]["state"].items():
        original = parent["optimizer_state_dict"]["state"].get(key)
        _require(original is not None, "Candidate Adam parameter IDs differ")
        for field in ("exp_avg", "exp_avg_sq"):
            _require(state[field].shape == original[field].shape and state[field].dtype == original[field].dtype,
                     "Candidate Adam moment layout differs")
    _require(any(not torch.equal(value, parent["model_state_dict"][key])
                 for key, value in checkpoint["model_state_dict"].items()), "Candidate weights never changed")
    expected_yaml = {str((path.parent / "params" / name).resolve()) for name in ("env.yaml", "agent.yaml")}
    yaml_rows = receipt["actual_yaml"]
    _require(len(yaml_rows) == 2 and {str(Path(row["path"]).resolve()) for row in yaml_rows} == expected_yaml,
             "Missing actual training YAML artifacts")
    for row in yaml_rows:
        _require(_digest(row["path"]) == row["sha256"], "Actual training YAML changed")
    return {"training_receipt": str(path), "training_receipt_sha256": _digest(path),
            "checkpoint": str(stand_path), "checkpoint_sha256": actual_sha,
            "parent_checkpoint_sha256": PARENT_SHA, "mode": receipt["mode"],
            "verified_updates": updates, "verified_environment_steps": n * 24 * updates,
            "actual_checkpoint_iteration": iteration, "actual_adam_step": adam_step,
            "smoke_diagnostic_only": receipt["mode"] == "smoke",
            "formal_training_verified": receipt["mode"] == "formal",
            "recovery_acceptance_proven": False, "promotion_performed": False,
            "verification_boundary": "Source/artifact/CPU-checkpoint and independent stage-ledger consistency; not a signed attestation or recovery success proof."}


def build_source(stand_sha):
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Original combined adapter changed")
    if len(stand_sha) != 64 or any(c not in '0123456789abcdef' for c in stand_sha):
        raise ValueError("Invalid candidate SHA")
    source = base.build_source()
    source = base._replace_once(source, f'STAND_CHECKPOINT_SHA256 = "{base.STAND_SHA}"', f'STAND_CHECKPOINT_SHA256 = "{stand_sha}"')
    source = base._replace_once(source, '    _write_json_new(report_path, report)',
        '    report["candidate_training"] = CANDIDATE_RECEIPT\n    _write_json_new(report_path, report)')
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate_training_result", required=True, type=Path)
    parser.add_argument("--stand_checkpoint", required=True, type=Path)
    parser.add_argument("--allow_smoke_diagnostic", action="store_true")
    args, remaining = parser.parse_known_args()
    receipt = candidate_receipt(args.candidate_training_result, args.stand_checkpoint,
                                allow_smoke=args.allow_smoke_diagnostic)
    source = build_source(receipt["checkpoint_sha256"])
    sys.argv = [sys.argv[0], *remaining, "--stand_checkpoint", str(args.stand_checkpoint)]
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
        "COMBINED_PROTOCOL": PROTOCOL, "COMBINED_CONTROLLER": "experimental_trained_stance_candidate_in_combined_recovery",
        "COMBINED_TEMPLATE_SHA": base.TEMPLATE_SHA, "COMBINED_STARTUP_REFERENCE_SHA": base.STARTUP_REFERENCE_SHA,
        "COMBINED_STARTUP_MATH_SHA": base.STARTUP_MATH_SHA, "COMBINED_MIRROR_MATH_SHA": base.MIRROR_MATH_SHA,
        "COMBINED_INVALID_JSON": base.invalid_diagnostic_json, "COMBINED_GENERATED_SOURCE": source,
        "COMBINED_GENERATED_SHA": hashlib.sha256(source.encode()).hexdigest(), "CANDIDATE_RECEIPT": receipt}
    exec(compile(source, str(Path(__file__)) + "::<natural-candidate>", "exec"), namespace)
