"""CPU-only guard tests; no simulator, physics claim, or artifact modification."""
import ast
import copy
from pathlib import Path
import unittest
from unittest import mock

import evaluate_natural_handoff_candidate as candidate


SMOKE = candidate.RUN_ROOT / "20260917-natural-stand16x2-smoke"


class CandidateGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = candidate._load(SMOKE / "training_result.json")

    def formal_header(self):
        value = copy.deepcopy(self.smoke)
        run = candidate.RUN_ROOT / "header-test-formal-never-created"
        value.update(mode="formal", num_envs=128, updates=200, verified_new_updates=200,
                     expected_final_iteration=3746, expected_final_adam_step=75120,
                     expected_environment_steps=614400, verified_environment_steps=614400,
                     run_dir=str(run), final_checkpoint=str(run / "model_3746.pt"))
        return value, run / "training_result.json", run / "model_3746.pt"

    def test_formal_header_fixed_budget(self):
        value, path, model = self.formal_header()
        self.assertEqual(candidate._receipt_header(value, path, model), (128, 200, 3746, 75120))

    def test_budget_and_completion_mutations_rejected(self):
        changes = {"updates": 100, "num_envs": 16, "verified_new_updates": 199,
                   "verified_environment_steps": 614399, "expected_final_iteration": 3747,
                   "expected_final_adam_step": 75119, "completion_verified": False,
                   "loaded_full_parent_exact": 1, "intended_process_exit_code": False,
                   "promotion_performed": True, "initial_algorithm_scalar_learning_rate": 2.25e-5}
        for key, bad in changes.items():
            with self.subTest(key=key):
                value, path, model = self.formal_header()
                value[key] = bad
                with self.assertRaises(ValueError):
                    candidate._receipt_header(value, path, model)

    def test_wrong_parent_and_reward_rejected(self):
        for group, key, bad in (("parent", "sha256", "0" * 64),
                                ("parent", "path", str(candidate.PARENT.with_name("model_3448.pt"))),
                                ("config_guard", "supported_symmetry_weight", 20.0),
                                ("config_guard", "smith_robot_physx_dt_decimation_material_actions_equal", False)):
            with self.subTest(key=key):
                value, path, model = self.formal_header()
                value[group][key] = bad
                with self.assertRaises(ValueError):
                    candidate._receipt_header(value, path, model)

    def test_receipt_and_checkpoint_cannot_be_cross_run(self):
        value, path, model = self.formal_header()
        for bad_path, bad_model in ((path.with_name("pretend.json"), model),
                                    (path, model.parent.parent / "model_3746.pt"),
                                    (path, model.with_name("model_3745.pt"))):
            with self.assertRaises(ValueError):
                candidate._receipt_header(value, bad_path, bad_model)

    def test_smoke_requires_explicit_diagnostic_permission(self):
        path, model = SMOKE / "training_result.json", SMOKE / "model_3548.pt"
        with self.assertRaises(ValueError):
            candidate._receipt_header(self.smoke, path, model)
        self.assertEqual(candidate._receipt_header(self.smoke, path, model, allow_smoke=True),
                         (16, 2, 3548, 71160))

    def test_actual_smoke_artifacts_checkpoint_and_evidence(self):
        result = candidate.candidate_receipt(SMOKE / "training_result.json", SMOKE / "model_3548.pt",
                                             allow_smoke=True)
        self.assertTrue(result["smoke_diagnostic_only"])
        self.assertFalse(result["formal_training_verified"])
        self.assertFalse(result["recovery_acceptance_proven"])
        self.assertEqual(result["actual_adam_step"], 71160)

    def test_forged_receipt_source_snapshot_rejected(self):
        original = candidate._load
        bad = copy.deepcopy(self.smoke)
        bad["source_snapshot"] = bad["source_snapshot"][:-1]
        def changed(path):
            return bad if Path(path) == SMOKE / "training_result.json" else original(path)
        with mock.patch.object(candidate, "_load", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "snapshot"):
                candidate.candidate_receipt(SMOKE / "training_result.json", SMOKE / "model_3548.pt", allow_smoke=True)

    def test_truncated_preflight_ledger_rejected(self):
        original = candidate._load
        def changed(path):
            value = original(path)
            if Path(path) == SMOKE / "preflight.json":
                del value["updates"]
            return value
        with mock.patch.object(candidate, "_load", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Incomplete independent"):
                candidate.candidate_receipt(SMOKE / "training_result.json", SMOKE / "model_3548.pt", allow_smoke=True)

    def test_duplicate_json_keys_and_nonfinite_json_rejected(self):
        for payload in ('{"completion_verified": false, "completion_verified": true}', '{"score": NaN}'):
            with mock.patch.object(Path, "read_text", return_value=payload):
                with self.assertRaises(ValueError):
                    candidate._load(SMOKE / "training_result.json")

    def test_forged_receipt_checkpoint_summary_rejected(self):
        original = candidate._load
        bad = copy.deepcopy(self.smoke)
        bad["checkpoints"][0]["adam_steps"][0] += 1
        def changed(path):
            return bad if Path(path) == SMOKE / "training_result.json" else original(path)
        with mock.patch.object(candidate, "_load", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Actual checkpoint"):
                candidate.candidate_receipt(SMOKE / "training_result.json", SMOKE / "model_3548.pt", allow_smoke=True)

    def test_actual_checkpoint_wrong_adam_rejected_even_with_finite_receipt(self):
        import torch
        original = torch.load
        def changed(path, **kwargs):
            state = original(path, **kwargs)
            if Path(path) == SMOKE / "model_3548.pt":
                first = next(iter(state["optimizer_state_dict"]["state"].values()))
                first["step"] += 1
            return state
        with mock.patch.object(torch, "load", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Adam step"):
                candidate.candidate_receipt(SMOKE / "training_result.json", SMOKE / "model_3548.pt", allow_smoke=True)

    def test_actual_checkpoint_nan_rejected_even_with_finite_receipt(self):
        import torch
        original = torch.load
        def changed(path, **kwargs):
            state = original(path, **kwargs)
            if Path(path) == SMOKE / "model_3548.pt":
                state["model_state_dict"]["std"][0] = float("nan")
            return state
        with mock.patch.object(torch, "load", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Nonfinite model"):
                candidate.candidate_receipt(SMOKE / "training_result.json", SMOKE / "model_3548.pt", allow_smoke=True)

    def test_generated_program_only_two_reviewed_changes(self):
        new_sha = "a" * 64
        baseline = candidate.base.build_source()
        actual = candidate.build_source(new_sha)
        expected = baseline.replace(f'STAND_CHECKPOINT_SHA256 = "{candidate.base.STAND_SHA}"',
                                    f'STAND_CHECKPOINT_SHA256 = "{new_sha}"', 1)
        expected = expected.replace('    _write_json_new(report_path, report)',
                                    '    report["candidate_training"] = CANDIDATE_RECEIPT\n    _write_json_new(report_path, report)', 1)
        self.assertEqual(actual, expected)
        ast.parse(actual)
        self.assertIn(candidate.base.ROLL_SHA, actual)

    def test_bad_hash_and_changed_original_adapter_rejected(self):
        for bad in ("a" * 63, "A" * 64, "g" * 64):
            with self.assertRaises(ValueError):
                candidate.build_source(bad)
        with mock.patch.object(candidate, "BASE_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Original combined"):
                candidate.build_source("a" * 64)


if __name__ == "__main__":
    unittest.main()
