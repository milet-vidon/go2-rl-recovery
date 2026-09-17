"""CPU-only bounded-selection guard tests; never launches a simulator or trains."""
import ast
import copy
from pathlib import Path
import unittest
from unittest import mock

import evaluate_natural_checkpoint_selection as selection


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = selection.candidate._load(selection.RECEIPT)

    def test_actual_formal200_and_selected3600_independently_verified(self):
        result = selection.selection_receipt(selection.RECEIPT, selection.RUN / "model_3600.pt")
        self.assertEqual(result["effective_updates"], 54)
        self.assertEqual(result["actual_adam_step"], 72200)
        self.assertEqual(result["model_tensor_count"], 17)
        self.assertEqual(result["original_formal_training"]["verified_updates"], 200)
        self.assertEqual(result["original_formal_training"]["actual_checkpoint_iteration"], 3746)
        self.assertEqual(result["original_formal_training"]["actual_adam_step"], 75120)
        self.assertFalse(result["selected_is_final_checkpoint"])
        self.assertFalse(result["receipt_rewritten"])
        self.assertEqual(result["additional_training_updates"], 0)
        self.assertTrue(result["retrospective_development_selection"])
        self.assertFalse(result["recovery_acceptance_proven"])

    def test_catalogue_exactly_two_effective_budgets(self):
        for iteration, updates, adam in ((3600, 54, 72200), (3700, 154, 74200)):
            actual = selection.selection_header(self.receipt, selection.RECEIPT,
                                                selection.RUN / f"model_{iteration}.pt")
            self.assertEqual(actual[:3], (iteration, updates, adam))
        self.assertEqual(set(selection.ALLOWED), {3600, 3700})

    def test_wrong_run_final_or_unlisted_partial_rejected(self):
        for path in (selection.FINAL, selection.RUN / "model_3650.pt", selection.candidate.PARENT,
                     selection.RUN.parent / "other-run/model_3600.pt"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    selection.selection_header(self.receipt, selection.RECEIPT, path)

    def test_original_formal_budget_never_rewritten_as54(self):
        for key, value in (("updates", 54), ("verified_new_updates", 54),
                           ("expected_final_iteration", 3600), ("expected_final_adam_step", 72200),
                           ("final_checkpoint", str(selection.RUN / "model_3600.pt"))):
            altered = copy.deepcopy(self.receipt)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                selection.selection_header(altered, selection.RECEIPT, selection.RUN / "model_3600.pt")

    def test_selected_receipt_row_must_be_unique_present_and_true(self):
        for case in ("missing", "duplicate", "sha", "iteration", "adam", "finite", "tensor_count"):
            altered = copy.deepcopy(self.receipt)
            row = next(r for r in altered["checkpoints"] if r["iteration"] == 3600)
            if case == "missing":
                altered["checkpoints"].remove(row)
            elif case == "duplicate":
                altered["checkpoints"].append(copy.deepcopy(row))
            elif case == "sha":
                row["sha256"] = "0" * 64
            elif case == "iteration":
                row["iteration"] = 3601
            elif case == "adam":
                row["adam_steps"][0] += 1
            elif case == "finite":
                row["all_model_and_adam_tensors_finite"] = 1
            else:
                row["model_tensor_count"] = 16
            with self.subTest(case=case), self.assertRaises(ValueError):
                selection.selection_header(altered, selection.RECEIPT, selection.RUN / "model_3600.pt")

    def test_full_final_validation_happens_before_selected_load(self):
        import torch
        with mock.patch.object(selection.candidate, "candidate_receipt", side_effect=ValueError("final failed")) as full:
            with mock.patch.object(torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "final failed"):
                    selection.selection_receipt(selection.RECEIPT, selection.RUN / "model_3600.pt")
                full.assert_called_once_with(selection.RECEIPT.resolve(), selection.FINAL, allow_smoke=False)
                loader.assert_not_called()

    def test_actual_selected_iter_adam_and_nan_rejected(self):
        import torch
        original = torch.load
        for case in ("iteration", "adam", "nan", "layout"):
            def altered(path, **kwargs):
                state = original(path, **kwargs)
                if Path(path) == selection.RUN / "model_3600.pt":
                    if case == "iteration":
                        state["iter"] = 3601
                    elif case == "adam":
                        next(iter(state["optimizer_state_dict"]["state"].values()))["step"] += 1
                    elif case == "nan":
                        state["model_state_dict"]["std"][0] = float("nan")
                    else:
                        del state["model_state_dict"]["std"]
                return state
            with self.subTest(case=case), mock.patch.object(torch, "load", side_effect=altered):
                with self.assertRaises(ValueError):
                    selection.selection_receipt(selection.RECEIPT, selection.RUN / "model_3600.pt")

    def test_receipt_validator_and_selected_sha_are_pinned(self):
        for key in ("CANDIDATE_SHA", "RECEIPT_SHA", "FINAL_SHA"):
            with self.subTest(key=key), mock.patch.object(selection, key, "0" * 64):
                with self.assertRaises(ValueError):
                    selection.selection_receipt(selection.RECEIPT, selection.RUN / "model_3600.pt")

    def test_generated_controller_only_stand_sha_and_evidence_change(self):
        chosen = selection.ALLOWED[3600]
        baseline = selection.candidate.build_source(chosen)
        expected = baseline.replace('    _write_json_new(report_path, report)',
            '    report["checkpoint_selection"] = CHECKPOINT_SELECTION\n    _write_json_new(report_path, report)', 1)
        actual = selection.build_source(chosen)
        self.assertEqual(actual, expected)
        ast.parse(actual)
        self.assertIn(selection.candidate.base.ROLL_SHA, actual)
        self.assertIn('report["candidate_training"] = CANDIDATE_RECEIPT', actual)

    def test_unknown_sha_cannot_generate_unbounded_candidate(self):
        for value in ("a" * 64, selection.FINAL_SHA, selection.candidate.PARENT_SHA):
            with self.assertRaises(ValueError):
                selection.build_source(value)

    def test_original_training_evidence_is_separate_from_selection(self):
        formal = {"checkpoint_sha256": selection.FINAL_SHA, "verified_updates": 200}
        selected = {"original_formal_training": formal, "checkpoint_sha256": selection.ALLOWED[3600],
                    "effective_updates": 54}
        namespace = selection.execution_namespace(selected, "synthetic source only")
        self.assertIs(namespace["CANDIDATE_RECEIPT"], formal)
        self.assertIs(namespace["CHECKPOINT_SELECTION"], selected)
        self.assertEqual(namespace["COMBINED_PROTOCOL"], selection.PROTOCOL)
        self.assertNotEqual(namespace["CANDIDATE_RECEIPT"]["checkpoint_sha256"], selected["checkpoint_sha256"])

    def test_header_checks_do_not_mutate_original_receipt(self):
        before = copy.deepcopy(self.receipt)
        selection.selection_header(self.receipt, selection.RECEIPT, selection.RUN / "model_3600.pt")
        self.assertEqual(before, self.receipt)


if __name__ == "__main__":
    unittest.main()
