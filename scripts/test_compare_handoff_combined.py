"""Synthetic comparator mutations of real baselines; not simulator evidence."""
import copy
import json
from pathlib import Path
import unittest

import compare_handoff_combined as comparator


class CombinedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory, _ = comparator.BASELINES["startup-upright"]
        cls.report = json.loads((comparator.ROOT / "evaluations" / directory / "model_1999.pt_recovery_metrics.json").read_text(encoding="utf-8"))

    def test_all_twelve_reference_mappings(self):
        for pose in ("upright", "side", "upside_down"):
            self.assertEqual(comparator.baseline_key("off", "off", pose), "hard-" + pose)
            self.assertEqual(comparator.baseline_key("supported", "off", pose), "startup-" + pose)
            self.assertEqual(comparator.baseline_key("off", "initial_right", pose), "mirror-" + pose)
            self.assertEqual(comparator.baseline_key("supported", "initial_right", pose),
                             ("startup-" if pose == "upright" else "mirror-") + pose)

    def test_all_historical_report_hashes(self):
        for directory, digest in comparator.BASELINES.values():
            path = comparator.ROOT / "evaluations" / directory / "model_1999.pt_recovery_metrics.json"
            self.assertEqual(comparator.sha(path), digest)

    def test_exact_synthetic_copy(self):
        result = comparator.compare_common(self.report, copy.deepcopy(self.report))
        self.assertEqual(result.mismatch_count, 0)
        self.assertGreater(result.checked, 1000)

    def test_only_declared_metadata_exceptions(self):
        changed = copy.deepcopy(self.report)
        changed["protocol_version"] = "synthetic_only"
        changed["controller_type"] = "synthetic_only"
        for group, fields in comparator.GROUP_EXCEPTIONS.items():
            if group in changed:
                for field in fields:
                    changed[group][field] = "synthetic_identity_change"
        self.assertEqual(comparator.compare_common(self.report, changed).mismatch_count, 0)

    def test_metrics_physics_and_modes_must_not_change(self):
        cases = [(["seed"], 0), (["criterion"], "relaxed"),
                 (["results", "upright", "final_valid_stands"], 19),
                 (["results", "upright", "trials"], 19),
                 (["transition_experiment", "hold_s"], 2.0),
                 (["transition_experiment", "hidden_pd_at_switch"], True),
                 (["startup_experiment", "selected_counts_as_handoff"], True),
                 (["startup_experiment", "mode"], "off"),
                 (["handoff_controller", "stand_sha256"], "wrong"),
                 (["self_collisions_enabled"], False)]
        for keys, value in cases:
            with self.subTest(keys=keys):
                candidate = copy.deepcopy(self.report)
                node = candidate
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                self.assertGreater(comparator.compare_common(self.report, candidate).mismatch_count, 0)

    def test_missing_initial_state_and_small_float_change(self):
        candidate = copy.deepcopy(self.report)
        candidate["results"]["upright"]["policy_start_state"].pop()
        self.assertGreater(comparator.compare_common(self.report, candidate).mismatch_count, 0)
        candidate = copy.deepcopy(self.report)
        candidate["results"]["upright"]["final_diagnostics"][0]["height_m"] += 1e-12
        self.assertGreater(comparator.compare_common(self.report, candidate).mismatch_count, 0)

    def test_csv_and_neighborhood_scalar_exactness(self):
        result = comparator.ExactComparison()
        result.compare([{"actual_raw_action": [0.1], "step": 0}],
                       [{"actual_raw_action": [0.100000000001], "step": 0}], "synthetic_row")
        self.assertEqual(result.mismatch_count, 1)

    def test_inputs_unchanged(self):
        before = copy.deepcopy(self.report)
        comparator.compare_common(self.report, self.report)
        self.assertEqual(before, self.report)

    def test_first_action_matches_start_and_duplicate(self):
        directory, _ = comparator.BASELINES["startup-upright"]
        path = comparator.ROOT / "evaluations" / directory / self.report["transition_experiment"]["transition_trace_file"]
        trace = json.loads(path.read_text(encoding="utf-8"))
        for first_row, start_row in zip(trace["first_action_by_pose"]["upright"],
                                         self.report["results"]["upright"]["policy_start_state"]):
            comparator.audit_first_action(first_row, [], start_row)
        first = trace["first_action_by_pose"]["upright"][0]
        start = self.report["results"]["upright"]["policy_start_state"][0]
        duplicate = copy.deepcopy(first)
        comparator.audit_first_action(first, [duplicate], start)
        changed = copy.deepcopy(first)
        changed["issued_raw_action"][0] += .001
        with self.assertRaises(ValueError):
            comparator.audit_first_action(changed, [duplicate], start)
        changed = copy.deepcopy(first)
        changed["before"]["joint_positions_rad"][0] += .001
        with self.assertRaises(ValueError):
            comparator.audit_first_action(changed, [], start)


if __name__ == "__main__":
    unittest.main(verbosity=2)
