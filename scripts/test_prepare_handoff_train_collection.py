"""Pure CPU/read-only regression checks for the RAW handoff collection exporter."""

import copy
import io
import json
from pathlib import Path
import unittest

import numpy as np

from audit_recovery_report import AuditError, load_report
import prepare_handoff_train_collection as exporter


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "evaluations/20260917-handoff-train-smoke/model_1999.pt_recovery_metrics.json"


class RawHandoffCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual = load_report(SMOKE)
        cls.index, _ = exporter.read_source_bank(cls.actual["state_bank"])
        cls.collector = ROOT / "scripts/evaluate_go2_recovery.py"
        cls.collector_sha = exporter.sha256(cls.collector)

    def setUp(self):
        self.report = copy.deepcopy(self.actual)

    def record(self):
        return self.report["results"]["side"]["handoff_diagnostic"]["switch_records"][0]

    def rejected(self):
        with self.assertRaises((ValueError, KeyError, AuditError)):
            exporter.extract_report(self.report, self.index)

    def test_real_smoke_preserves_observations_and_source(self):
        before = copy.deepcopy(self.report)
        samples, ledgers = exporter.extract_report(self.report, self.index)
        self.assertEqual(len(samples), 2)
        self.assertEqual(ledgers[0]["total_trials"], 2)
        self.assertEqual(self.report, before)
        for sample, original in zip(samples, self.report["results"]["side"]["handoff_diagnostic"]["switch_records"]):
            for field in exporter.VECTOR_FIELDS:
                np.testing.assert_array_equal(sample[field], original[field])
            self.assertEqual(self.index[sample["source_train_state_id"]][0], 0)

    def test_declared_heldout_split_rejected(self):
        self.report["state_bank"]["split"] = "heldout"
        self.report["state_bank"]["diagnostic_train_split_only"] = False
        self.rejected()

    def test_actual_source_split_overrides_report_label(self):
        source_id = self.report["results"]["side"]["state_bank_selection"]["selected_state_ids"][0]
        changed_index = dict(self.index)
        changed_index[source_id] = (1, changed_index[source_id][1])
        with self.assertRaisesRegex(ValueError, "heldout"):
            exporter.extract_report(self.report, changed_index)

    def test_missing_required_new_or_observation_field_rejected(self):
        for key in ("policy_observation", "previous_previous_raw_action", "velocity_command_b"):
            with self.subTest(key=key):
                self.report = copy.deepcopy(self.actual)
                self.record().pop(key)
                self.rejected()

    def test_missing_hard_limits_rejected(self):
        self.report["handoff_controller"]["joint_limits"].pop("hard_joint_limits_rad")
        self.rejected()

    def test_nonfinite_and_dimension_rejected(self):
        for value in ([0.0] * 11, [float("nan")] * 12, [float("inf")] * 12, ["x"] * 12):
            with self.subTest(value=value[0]):
                self.report = copy.deepcopy(self.actual)
                self.record()["previous_previous_raw_action"] = value
                self.rejected()

    def test_raw_history_and_target_mismatches_rejected(self):
        for key in ("previous_raw_action", "previous_executed_joint_target_rad", "velocity_command_b"):
            with self.subTest(key=key):
                self.report = copy.deepcopy(self.actual)
                self.record()[key][0] += .3
                self.rejected()

    def test_root_velocity_observation_mismatch_rejected(self):
        self.record()["root_linear_velocity_w_m_s"][0] += 1
        self.rejected()

    def test_unknown_source_id_and_source_pose_rejected(self):
        source_id = self.report["results"]["side"]["state_bank_selection"]["selected_state_ids"][0]
        missing = dict(self.index)
        del missing[source_id]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            exporter.extract_report(self.report, missing)
        wrong_pose = dict(self.index)
        wrong_pose[source_id] = (0, "back")
        with self.assertRaisesRegex(ValueError, "pose"):
            exporter.extract_report(self.report, wrong_pose)

    def test_actual_limit_exceedances_are_recorded_not_clipped(self):
        samples, _ = exporter.extract_report(self.report, self.index)
        self.assertTrue(any(np.any(x["soft_limit_excess_rad"] > 0) for x in samples))
        self.assertTrue(any(np.any(x["hard_limit_excess_rad"] > 0) for x in samples))
        for sample, record in zip(samples, self.report["results"]["side"]["handoff_diagnostic"]["switch_records"]):
            np.testing.assert_array_equal(sample["joint_positions_rad"], record["joint_positions_rad"])

    def test_untriggered_trial_is_preserved_and_source_checked(self):
        result = self.report["results"]["side"]
        diagnostic = result["handoff_diagnostic"]
        diagnostic["switch_records"][1] = None
        diagnostic["triggered_trials"] = 1
        diagnostic["untriggered_trials"] = 1
        diagnostic["final_valid_after_trigger"] = sum(
            record is not None and final["stable_hold_s"] >= result["stable_hold_s"]
            for record, final in zip(diagnostic["switch_records"], result["final_diagnostics"]))
        samples, ledgers = exporter.extract_report(self.report, self.index)
        self.assertEqual(len(samples), 1)
        self.assertEqual(ledgers[0]["total_trials"], 2)
        self.assertEqual(ledgers[0]["untriggered_trials"], 1)
        self.assertIsNone(ledgers[0]["trials"][1]["switch_time_s"])
        invalid_index = dict(self.index)
        source_id = result["state_bank_selection"]["selected_state_ids"][1]
        invalid_index[source_id] = (1, invalid_index[source_id][1])
        with self.assertRaisesRegex(ValueError, "heldout"):
            exporter.extract_report(self.report, invalid_index)

    def test_wrong_collector_sha_rejected(self):
        with self.assertRaisesRegex(ValueError, "Collector"):
            exporter.prepare_collection([SMOKE], self.collector, "0" * 64)

    def test_source_bank_hash_and_false_train_flag_rejected(self):
        for field, value in (("sha256", "0" * 64), ("diagnostic_train_split_only", False)):
            bank = copy.deepcopy(self.actual["state_bank"])
            bank[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                exporter.read_source_bank(bank)

    def test_duplicate_report_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            exporter.prepare_collection([SMOKE, SMOKE], self.collector, self.collector_sha)

    def test_existing_and_non_e_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            exporter.output_path(ROOT)
        with self.assertRaisesRegex(ValueError, "E:"):
            exporter.output_path(Path("C:/forbidden-handoff-export"))

    def test_prepare_is_raw_not_trainable_and_npz_roundtrip_in_memory(self):
        arrays, manifest = exporter.prepare_collection([SMOKE], self.collector, self.collector_sha)
        self.assertEqual(manifest["schema_version"], "handoff_observation_collection_v1")
        for key in ("training_ready", "replay_validated", "legacy_bank_loader_compatible", "acceptance_eligible"):
            self.assertIs(manifest[key], False)
        self.assertEqual(manifest["total_source_trials"], 2)
        self.assertEqual(manifest["source_reports"][0]["recorded_acceptance_eligible"], self.actual["acceptance_eligible"])
        json.dumps(manifest, allow_nan=False)
        payload = io.BytesIO()
        np.savez_compressed(payload, **arrays)
        payload.seek(0)
        with np.load(payload, allow_pickle=False) as restored:
            self.assertEqual(set(restored.files), set(arrays))
            for key, value in arrays.items():
                np.testing.assert_array_equal(restored[key], value)


if __name__ == "__main__":
    unittest.main()
