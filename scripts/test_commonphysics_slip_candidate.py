"""CPU-only source/negative-ledger checks, never physical or gait acceptance."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import evaluate_commonphysics_slip_candidate as shell


PROOF = {"protocol": "continuous_flow_trace_audit_v1", "report_path": "E:/unit-test-only/continuous_flow_report.json",
         "audit_passed": True, "explicit_sensor_identity_verified": True}


def private_entry(arm="B"):
    training = SimpleNamespace(__file__=shell.slip.__file__, ARM=arm,
        PROTOCOL=shell.slip.PROTOCOL, PARENT=shell.slip.PARENT, PARENT_SHA=shell.slip.PARENT_SHA,
        RUN_ROOT=shell.slip.PARENT_DIR.parent, PARENT_LR=1e-5,
        STATE={"parent_training_verification": {"fixture": "actual-parent-proof"}})
    with mock.patch.object(shell.slip, "adapter", return_value=training):
        return shell.adapter(arm, PROOF)


class SlipSourceTests(unittest.TestCase):
    def test_both_programs_compile_without_simulator(self):
        for arm in ("A", "B"):
            source = shell.program(arm)
            compile(source, "<test-private-slip>", "exec")
            self.assertIn('model_4545.pt', source)
            self.assertIn('before["initial_iteration"] == 4246', source)
            self.assertIn('before["initial_adam_step"] == 85120', source)
            self.assertIn('training.check_final(actual, parent, 300)', source)

    def test_wrong_arm_or_frozen_source_drift_refused(self):
        with self.assertRaisesRegex(ValueError, "arm"):
            shell.program("balanced")
        for name in ("TEMPLATE_SHA", "TRAINER_SHA"):
            with mock.patch.object(shell, name, "0" * 64), self.assertRaises(ValueError):
                shell.program("A")

    def test_arm_generated_code_only_declared_scalar_and_identity_differ(self):
        a, b = shell.program("A"), shell.program("B")
        self.assertEqual(a.replace('ARM = "A"', 'ARM = "B"').replace('SLIP_WEIGHT = -0.5', 'SLIP_WEIGHT = -1.0'), b)

    def test_underlying_simulator_and_physical_functions_unchanged(self):
        entry = private_entry()
        old = (entry.frozen.CONTROL, entry.frozen.CONTROL_SHA, entry.frozen.PROTOCOL)
        case = entry.make_adapter({"checkpoint": "E:/unit-test-only/model_4545.pt", "checkpoint_sha256": "a" * 64})
        self.assertEqual(case.build_source(), entry.frozen.build_source())
        self.assertEqual((entry.frozen.CONTROL, entry.frozen.CONTROL_SHA, entry.frozen.PROTOCOL), old)
        for name in ("configure_physics", "configure_runtime", "validate_interface", "begin_step", "end_step",
                     "straight_drift_acceptance", "validate_cli"):
            self.assertEqual(getattr(case, name).__code__.co_code, getattr(entry.frozen, name).__code__.co_code)

    def test_strict_drift_boundary_is_not_relaxed(self):
        entry = private_entry()
        for value, expected in ((.119999, True), (.12, False), (-.12, False)):
            self.assertIs(entry.frozen.straight_drift_acceptance({"vy_b_mean": value, "yaw_rate_mean": 0}, 0, 0, 0), expected)

    def test_full_formal_evidence_guards_still_present(self):
        source = shell.program("A")
        for needle in ("training.verify_smoke", "training.all_sources()", "training.parent_metadata()",
                       "training.check_documents", "checkpoint_metadata", "parent_model_and_adam_exact",
                       "initial_scalar_and_optimizer_lr", "actual_yaml", "generated_sha256",
                       "_verify_slip_ledgers", "Smoke arm/weight mismatch"):
            self.assertIn(needle, source)


class SlipReceiptNegativeTests(unittest.TestCase):
    def setUp(self):
        self.entry = private_entry()
        self.run = self.entry.training.RUN_ROOT / "unit-test-only-do-not-create-slip-B"
        self.path = self.run / "common_physics_training_result.json"
        self.checkpoint = self.run / "model_4545.pt"
        self.receipt = {"protocol": shell.slip.PROTOCOL, "arm": "B", "completion_verified": True,
            "foot_slip_weight": -1., "balanced_duration_weight": -10., "num_envs": 128, "updates": 300,
            "actual_environment_steps": 921600, "actual_control_steps": 7200, "quality_accepted": False,
            "promotion_performed": False, "hardware_execution": False, "checkpoint": str(self.checkpoint),
            "parent_path": str(shell.slip.PARENT)}

    def verify(self, value=None, checkpoint=None):
        with mock.patch.object(self.entry, "read_json", return_value=value or self.receipt):
            return self.entry.candidate_receipt(self.path, checkpoint or self.checkpoint)

    def test_only_predeclared_final4545(self):
        for name in ("model_4246.pt", "model_4247.pt", "model_4544.pt"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "final formal4545"):
                self.verify(checkpoint=self.run / name)

    def test_each_budget_including_bool_fails(self):
        for name in ("num_envs", "updates", "actual_environment_steps", "actual_control_steps"):
            for wrong in (1, True, str(self.receipt[name])):
                value = copy.deepcopy(self.receipt)
                value[name] = wrong
                with self.subTest(name=name, wrong=wrong), self.assertRaisesRegex(ValueError, "budget"):
                    self.verify(value)

    def test_failure_promotion_and_wrong_arm_cannot_be_hidden(self):
        for name, wrong in (("completion_verified", False), ("quality_accepted", True),
                            ("promotion_performed", True), ("hardware_execution", True),
                            ("arm", "A"), ("balanced_duration_weight", True), ("balanced_duration_weight", 0)):
            value = copy.deepcopy(self.receipt)
            value[name] = wrong
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.verify(value)

    def test_old3947_parent_is_rejected(self):
        value = copy.deepcopy(self.receipt)
        value["parent_path"] = str(self.run / "model_3947.pt")
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.verify(value)

    def test_drifted_actual_inventory_cannot_be_trusted(self):
        value = copy.deepcopy(self.receipt)
        value.update(source_snapshot=[], parent={})
        with mock.patch.object(self.entry.training, "all_sources", return_value=[{"path": "actual"}], create=True), \
             mock.patch.object(self.entry.training, "parent_metadata", return_value={}, create=True), \
             self.assertRaisesRegex(ValueError, "inventory"):
            self.verify(value)

    def test_old_protocol_or_cross_arm_rejected_before_audit(self):
        value = copy.deepcopy(self.receipt)
        value["protocol"] = "balanced_gait_common_physics_training_v1"
        with mock.patch.object(shell.slip, "read_json", return_value=value), \
             mock.patch.object(shell.slip, "formal_prerequisite") as audit, self.assertRaises(ValueError):
            shell.candidate_receipt(self.path, self.checkpoint, "B")
        audit.assert_not_called()

    def test_cached_or_missing_continuous_proof_cannot_unlock(self):
        value = copy.deepcopy(self.receipt)
        with mock.patch.object(shell.slip, "read_json", return_value=value), self.assertRaisesRegex(ValueError, "Missing"):
            shell.candidate_receipt(self.path, self.checkpoint, "B")
        value["continuous_prerequisite"] = copy.deepcopy(PROOF)
        with mock.patch.object(shell.slip, "read_json", return_value=value), \
             mock.patch.object(shell.slip, "formal_prerequisite", return_value={"changed": True}) as audit, \
             self.assertRaisesRegex(ValueError, "freshly audited"):
            shell.candidate_receipt(self.path, self.checkpoint, "B")
        audit.assert_called_once_with(PROOF["report_path"])


class SlipLedgerTests(unittest.TestCase):
    def setUp(self):
        self.training = private_entry().training
        self.path = self.training.RUN_ROOT / "unit-test-only" / "common_physics_training_result.json"
        self.receipt = {"smoke_evidence": {"path": "E:/unit-test-only/smoke/common_physics_training_result.json"}}
        self.ledger = {"arm": "B", "foot_slip_weight": -1., "balanced_duration_weight": -10.,
            "parent_training_receipt_sha256": shell.slip.PARENT_RECEIPT_SHA,
            "parent_training_verification": self.training.STATE["parent_training_verification"],
            "continuous_prerequisite": copy.deepcopy(PROOF), "research_candidate_not_natural_gait": True}

    def verify(self, ledgers):
        with mock.patch.object(shell.slip, "read_json", side_effect=ledgers), \
             mock.patch.object(shell.slip, "formal_prerequisite", return_value=copy.deepcopy(PROOF)):
            shell.verify_slip_ledgers(self.path, self.receipt, self.training, PROOF)

    def test_all_four_ledgers_are_checked(self):
        self.verify([copy.deepcopy(self.ledger) for _ in range(4)])
        for index in range(4):
            ledgers = [copy.deepcopy(self.ledger) for _ in range(4)]
            ledgers[index]["foot_slip_weight"] = -.5
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "lineage"):
                self.verify(ledgers)

    def test_forged_flags_parent_or_continuous_rejected(self):
        for name, wrong in (("foot_slip_weight", True), ("balanced_duration_weight", True),
                            ("parent_training_receipt_sha256", "0" * 64),
                            ("parent_training_verification", {}), ("continuous_prerequisite", {}),
                            ("research_candidate_not_natural_gait", False)):
            ledgers = [copy.deepcopy(self.ledger) for _ in range(4)]
            ledgers[0][name] = wrong
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "lineage"):
                self.verify(ledgers)


class SlipLauncherDeclarationTests(unittest.TestCase):
    def test_full33_new_cases_and_duplicate_trajectory_disclosure(self):
        text = (shell.ROOT / "scripts/run_commonphysics_slip_candidate_screen.ps1").read_text()
        self.assertIn("$taskSeeds=@(20260909,20260910,20260911)", text)
        self.assertIn("$taskSummary.rows.Count -ne 33", text)
        self.assertIn("common18_passed", text)
        self.assertIn("original15_retained", text)
        self.assertIn("distinct_observed_trajectories", text)
        self.assertIn("independent_generalization_claimed=$false", text)
        self.assertIn("['full_flow_passed']=$false", text)
        self.assertIn("['natural_gait_accepted']=$false", text)

    def test_case_wrapper_preserves_old_simulation_and_new_final_identity(self):
        text = (shell.ROOT / "scripts/evaluate_commonphysics_slip_candidate.ps1").read_text()
        self.assertIn("7cb9ba878f6bb7bfb89edd059aac2350453a70e6dbe21ba4090848dbe2e2751c", text)
        self.assertIn("Replace-SlipCase '4246' '4545' 6", text)
        self.assertIn("foot_slip_weight", text)
        self.assertIn("if($SourceOnly){return $taskSource}", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
