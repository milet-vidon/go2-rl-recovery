"""Bounded retrospective selection of two existing natural-stance checkpoints.

Authenticate the completed formal200 run and its real final3746 first. Only then
verify an already recorded model3600/3700 from that exact run, independently on
CPU. Never rewrites a training receipt or calls a partial checkpoint a new/final
training run. Reusing development states for selection is not generalization.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import evaluate_natural_handoff_candidate as candidate


PROTOCOL = "natural_stance_checkpoint_selection_v1"
CANDIDATE_SHA = "a97f8592fc61dc2fe2e07fcd2a669eb040b64219e2348602d6d7aa015f0d53ee"
RUN = candidate.RUN_ROOT / "20260917-natural-stand128x200-first"
RECEIPT = RUN / "training_result.json"
RECEIPT_SHA = "531fde006c9f164f7535f0937f7a2828935eb868ed2f39cc00991fe49bdbb94f"
FINAL = RUN / "model_3746.pt"
FINAL_SHA = "f40772c228996400a9813ad244b48509737dabf2c31af001144de34283433b9e"
ALLOWED = {
    3600: "a291f537c03dd61d5091134469cf05754dbae5aeade4ad138cb6a2b05e779208",
    3700: "e7990d5f24a7f8da1dacf1c37c6d4b4db89d59977c57efb4f7c0ccb98ec346ea",
}
require = candidate._require
digest = candidate._digest


def selection_header(receipt, receipt_path, selected_path):
    """Pure catalogue checks; not a replacement for actual full-run verification."""
    receipt_path, selected_path = Path(receipt_path).resolve(), Path(selected_path).resolve()
    require(receipt_path == RECEIPT.resolve() and Path(receipt["run_dir"]).resolve() == RUN.resolve(),
            "Only the original completed formal200 run may be selected")
    require(Path(receipt["final_checkpoint"]).resolve() == FINAL.resolve(), "Original final3746 receipt changed")
    candidate._receipt_header(receipt, receipt_path, FINAL.resolve())
    allowed_paths = {(RUN / f"model_{i}.pt").resolve(): i for i in ALLOWED}
    require(selected_path in allowed_paths, "Only existing checkpoints3600/3700 are allowed; no scan/final/other-run selection")
    iteration = allowed_paths[selected_path]
    effective = iteration - 3547 + 1
    expected_adam = 71120 + 20 * effective
    rows = [row for row in receipt["checkpoints"] if Path(row["path"]).resolve() == selected_path]
    require(len(rows) == 1, "Selected checkpoint must appear exactly once in original receipt")
    row = rows[0]
    require(row["sha256"] == ALLOWED[iteration], "Selected SHA differs from original bounded catalogue")
    require(type(row["iteration"]) is int and row["iteration"] == iteration,
            "Selected receipt iteration mismatch")
    require(row["model_tensor_count"] == 17 and row["adam_state_count"] == 17 and
            row["all_model_and_adam_tensors_finite"] is True,
            "Incomplete selected checkpoint evidence")
    require(row["adam_steps"] == [float(expected_adam)] * 17, "Selected receipt Adam step mismatch")
    return iteration, effective, expected_adam, row


def selection_receipt(receipt_path, selected_path):
    receipt_path, selected_path = Path(receipt_path).resolve(), Path(selected_path).resolve()
    require(digest(candidate.__file__) == CANDIDATE_SHA, "Frozen full-candidate validator changed")
    require(receipt_path == RECEIPT.resolve() and digest(receipt_path) == RECEIPT_SHA,
            "Original formal200 receipt bytes/path changed")
    require(digest(FINAL) == FINAL_SHA, "Actual final3746 bytes changed")
    # Crucial: the existing validator authenticates full final3746, its source,
    # 79/80 TRAIN dataset, preflight/before-learning ledgers, parent, YAML and Adam.
    # No argument/receipt mutation is made to persuade it to accept a partial run.
    formal = candidate.candidate_receipt(receipt_path, FINAL, allow_smoke=False)
    require(formal["formal_training_verified"] is True and formal["verified_updates"] == 200 and
            formal["actual_checkpoint_iteration"] == 3746 and formal["actual_adam_step"] == 75120 and
            formal["checkpoint_sha256"] == FINAL_SHA, "Original full formal200 verification failed")
    receipt = candidate._load(receipt_path)
    iteration, effective, adam, row = selection_header(receipt, receipt_path, selected_path)
    require(digest(selected_path) == row["sha256"], "Actual selected checkpoint SHA differs from original receipt")
    import torch
    import train_handoff_stand as training
    checkpoint = torch.load(selected_path, map_location="cpu", weights_only=True)
    require(type(checkpoint.get("iter")) is int, "Actual selected iteration must be an integer")
    actual = training.checkpoint_summary(checkpoint, iteration, adam)
    candidate._summary_matches(row, actual)
    parent = torch.load(candidate.PARENT, map_location="cpu", weights_only=True)
    require(checkpoint["model_state_dict"].keys() == parent["model_state_dict"].keys(),
            "Selected actor/critic/std architecture differs from parent")
    for key, value in checkpoint["model_state_dict"].items():
        old = parent["model_state_dict"][key]
        require(value.shape == old.shape and value.dtype == old.dtype, "Selected tensor layout changed: " + key)
    for key, state in checkpoint["optimizer_state_dict"]["state"].items():
        old = parent["optimizer_state_dict"]["state"].get(key)
        require(old is not None, "Selected Adam parameter IDs changed")
        for field in ("exp_avg", "exp_avg_sq"):
            require(state[field].shape == old[field].shape and state[field].dtype == old[field].dtype,
                    "Selected Adam moment layout changed")
    require(any(not torch.equal(value, parent["model_state_dict"][key])
                for key, value in checkpoint["model_state_dict"].items()), "Selected weights did not change")
    require(digest(receipt_path) == RECEIPT_SHA and digest(selected_path) == ALLOWED[iteration],
            "Receipt or selected model changed during verification")
    return {"protocol": PROTOCOL, "selection_kind": "retrospective_development_selection",
            "retrospective_development_selection": True,
            "checkpoint": str(selected_path), "checkpoint_sha256": ALLOWED[iteration],
            "actual_checkpoint_iteration": iteration, "actual_adam_step": adam,
            "effective_updates": effective, "effective_environment_steps": 128 * 24 * effective,
            "model_tensor_count": actual["model_tensor_count"], "adam_state_count": actual["adam_state_count"],
            "all_model_and_adam_tensors_finite": actual["all_model_and_adam_tensors_finite"],
            "original_training_receipt": str(receipt_path), "original_training_receipt_sha256": RECEIPT_SHA,
            "original_verified_updates": 200, "original_final_checkpoint": str(FINAL.resolve()),
            "original_final_checkpoint_sha256": FINAL_SHA, "original_formal_training": formal,
            "bounded_catalogue_iterations": list(ALLOWED), "additional_training_updates": 0,
            "receipt_rewritten": False, "selected_is_final_checkpoint": False,
            "promotion_performed": False, "recovery_acceptance_proven": False,
            "independent_generalization_proven": False,
            "boundary": "Existing early checkpoints selected retrospectively after final3746 regression; repeatedly used development starts, not fresh heldout acceptance. No automatic scan or promotion."}


def build_source(stand_sha):
    require(digest(candidate.__file__) == CANDIDATE_SHA, "Frozen full-candidate validator changed")
    require(stand_sha in ALLOWED.values(), "Only the two recorded checkpoint hashes may be generated")
    source = candidate.build_source(stand_sha)
    source = candidate.base._replace_once(source, '    _write_json_new(report_path, report)',
        '    report["checkpoint_selection"] = CHECKPOINT_SELECTION\n    _write_json_new(report_path, report)')
    return source


def execution_namespace(selection, source):
    base = candidate.base
    return {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
        "COMBINED_PROTOCOL": PROTOCOL,
        "COMBINED_CONTROLLER": "retrospective_natural_stance_checkpoint_in_combined_recovery",
        "COMBINED_TEMPLATE_SHA": base.TEMPLATE_SHA, "COMBINED_STARTUP_REFERENCE_SHA": base.STARTUP_REFERENCE_SHA,
        "COMBINED_STARTUP_MATH_SHA": base.STARTUP_MATH_SHA, "COMBINED_MIRROR_MATH_SHA": base.MIRROR_MATH_SHA,
        "COMBINED_INVALID_JSON": base.invalid_diagnostic_json, "COMBINED_GENERATED_SOURCE": source,
        "COMBINED_GENERATED_SHA": hashlib.sha256(source.encode()).hexdigest(),
        # Keep the actual full200/final3746 evidence intact and separately labelled.
        "CANDIDATE_RECEIPT": selection["original_formal_training"], "CHECKPOINT_SELECTION": selection}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate_training_result", required=True, type=Path)
    parser.add_argument("--stand_checkpoint", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    receipt = selection_receipt(args.candidate_training_result, args.stand_checkpoint)
    source = build_source(receipt["checkpoint_sha256"])
    sys.argv = [sys.argv[0], *remaining, "--stand_checkpoint", str(args.stand_checkpoint)]
    exec(compile(source, str(Path(__file__)) + "::<retrospective-selection>", "exec"), execution_namespace(receipt, source))
