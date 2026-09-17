"""Bounded CPU adapter/negative-receipt tests; not simulator acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import evaluate_common_physics_candidate as entry


class CandidateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.receipt = {"checkpoint": "E:/IsaacLab/tmp/fixture/model_4046.pt",
                        "checkpoint_sha256": "a" * 64, "formal_updates_verified": 100}

    def test_source_exact_and_original_module_globals_unchanged(self):
        old = (entry.frozen.CONTROL, entry.frozen.CONTROL_SHA, entry.frozen.PROTOCOL)
        adapter = entry.make_adapter(self.receipt)
        self.assertEqual(adapter.build_source(), entry.frozen.build_source())
        self.assertEqual((entry.frozen.CONTROL, entry.frozen.CONTROL_SHA, entry.frozen.PROTOCOL), old)
        self.assertNotEqual(adapter.PROTOCOL, entry.frozen.PROTOCOL)
        self.assertEqual(adapter.CONTROL_SHA, self.receipt["checkpoint_sha256"])
        for name in ("configure_physics", "configure_runtime", "validate_interface", "begin_step", "end_step",
                     "straight_drift_acceptance", "validate_cli"):
            self.assertEqual(getattr(adapter, name).__code__.co_code, getattr(entry.frozen, name).__code__.co_code)

    def test_source_mutation_refused(self):
        with mock.patch.object(entry.frozen, "sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "adapter changed"):
                entry.make_adapter(self.receipt)

    def test_report_labels_candidate_not_frozen_control_and_keeps_failure(self):
        with tempfile.TemporaryDirectory(dir=entry.frozen.ROOT.parent / "tmp", prefix="common-candidate-test-") as directory:
            adapter = entry.make_adapter(self.receipt)
            args = SimpleNamespace(output_dir=Path(directory), physics_mode="recovery",
                                   lateral_speed=0., yaw_rate=0., push_speed=0.)
            report = {"global": {"steps": 900}, "protocol_version": "old_fixture", "passed": True,
                      "settled_phase_stats": {"walk": {"vy_b_mean": -.126406, "yaw_rate_mean": .025}}}
            adapter.finish_report(report, args, {"reference_checked": True}, {}, [{}] * 900, "fixture-source")
            self.assertTrue(report["passed"])
            self.assertFalse(report["retention_passed"])
            self.assertEqual(report["candidate_training"], self.receipt)
            self.assertNotIn("frozen_control_sha256", report["retention_experiment"])
            self.assertEqual(report["retention_experiment"]["candidate_checkpoint_sha256"], "a" * 64)
            trace = json.loads((Path(directory) / "retention_interface.json").read_text())
            self.assertEqual(trace["schema"], entry.PROTOCOL)
            self.assertEqual(len(trace["steps"]), 900)

    def test_strict_drift_boundary_not_relaxed(self):
        adapter = entry.make_adapter(self.receipt)
        for vy, expected in ((.119999, True), (.12, False), (-.12, False), (-.126406, False)):
            self.assertIs(adapter.straight_drift_acceptance({"vy_b_mean": vy, "yaw_rate_mean": 0}, 0, 0, 0), expected)

    def test_wrong_candidate_hash_fails_cli(self):
        adapter = entry.make_adapter(self.receipt)
        args = SimpleNamespace(task=adapter.TASK, zero_action=False, checkpoint=entry.training.PARENT)
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            adapter.validate_cli(args)


class CandidateReceiptNegativeTests(unittest.TestCase):
    def setUp(self):
        self.run = entry.training.RUN_ROOT / "20260917-commonphysics128x100-cpu-fixture"
        self.path, self.checkpoint = self.run / "common_physics_training_result.json", self.run / "model_4046.pt"
        self.receipt = {"protocol": entry.training.PROTOCOL, "completion_verified": True,
                        "num_envs": 128, "updates": 100, "actual_environment_steps": 307200,
                        "actual_control_steps": 2400, "quality_accepted": False, "promotion_performed": False,
                        "hardware_execution": False, "checkpoint": str(self.checkpoint),
                        "parent_path": str(entry.training.PARENT)}

    def verify(self, receipt=None, path=None, checkpoint=None):
        def digest(value):
            return entry.BASE_SHA if str(value) == str(entry.frozen.__file__) else entry.TRAINER_SHA
        with mock.patch.object(entry.frozen, "sha", side_effect=digest), \
             mock.patch.object(entry, "read_json", return_value=receipt or self.receipt):
            return entry.candidate_receipt(path or self.path, checkpoint or self.checkpoint)

    def test_reject_smoke_or_partial_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "final formal4046"):
            self.verify(checkpoint=self.run / "model_3948.pt")

    def test_each_budget_and_boolean_counter_rejected(self):
        for name in ("num_envs", "updates", "actual_environment_steps", "actual_control_steps"):
            for wrong in (1, True, str(self.receipt[name])):
                value = copy.deepcopy(self.receipt)
                value[name] = wrong
                with self.subTest(name=name, value=wrong), self.assertRaisesRegex(ValueError, "budget"):
                    self.verify(value)

    def test_incomplete_and_behavior_claims_rejected(self):
        for name, wrong in (("completion_verified", False), ("quality_accepted", True),
                            ("promotion_performed", True), ("hardware_execution", True)):
            value = copy.deepcopy(self.receipt)
            value[name] = wrong
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.verify(value)

    def test_wrong_parent_rejected(self):
        value = copy.deepcopy(self.receipt)
        value["parent_path"] = str(self.run / "model_3996.pt")
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.verify(value)

    def test_c_drive_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, "E-drive"):
            self.verify(path="C:/bad/common_physics_training_result.json")

    def test_source_inventory_drift_rejected(self):
        value = copy.deepcopy(self.receipt)
        value.update(source_snapshot=[], parent={})
        with mock.patch.object(entry.training, "all_sources", return_value=[{"path": "expected", "sha256": "x"}]), \
             mock.patch.object(entry.training, "parent_metadata", return_value={}):
            with self.assertRaisesRegex(ValueError, "inventory"):
                self.verify(value)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for text in ('{"updates":100,"updates":2}', '{"updates":NaN}'):
            with mock.patch.object(Path, "read_text", return_value=text), self.assertRaises(ValueError):
                entry.read_json(self.path)

    def test_checkpoint_tuple_json_transport_preserves_values(self):
        parent = entry.training.parent_metadata()
        transported = json.loads(json.dumps(parent))
        self.assertEqual(entry.json_value(parent), transported)
        self.assertEqual(parent["optimizer_group"]["betas"], (.9, .999))
        changed = copy.deepcopy(transported)
        changed["optimizer_group"]["betas"][0] = .8
        self.assertNotEqual(entry.json_value(parent), changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
