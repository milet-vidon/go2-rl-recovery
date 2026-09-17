"""Stdlib-only coordinate-audit regressions; real reports are read, never edited.

Synthetic on-mode fixtures verify audit logic only, never mirror-policy success.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_handoff_mirror_coordinates.py")
spec = importlib.util.spec_from_file_location("mirror_coordinate_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ROOT = SCRIPT.parents[1]
OFF_PATH = ROOT / "evaluations" / "20260917-handoff-mirror-v1" / "off-side" / "model_1999.pt_recovery_metrics.json"
HARD_PATH = ROOT / "evaluations" / "20260917-handoff-transition-v1" / "hard-side" / "model_1999.pt_recovery_metrics.json"


def synthetic_on(base, trace):
    """Coordinate-consistent mock only; no actor/network/physics execution."""
    result, output = copy.deepcopy(base), copy.deepcopy(trace)
    result["mirror_experiment"]["mode"] = output["mirror_experiment"]["mode"] = "initial_right"
    selected_count = 0
    for pose, values in result["results"].items():
        for i, start in enumerate(values["policy_start_state"]):
            selection = values["mirror_diagnostic"]["selection_records"][i]
            selected = bool(start["eligible_settled_fallen_recovery"] and selection["normalized_policy_start_gravity_y"] < audit.f32(audit.THRESHOLD))
            selection["selected"] = selected
            trial = output["poses"][pose][i]
            trial["mirror_selection"]["selected"] = selected
            selected_count += selected
            records = trial["rows"] + ([values["handoff_diagnostic"]["switch_records"][i]] if trial["triggered"] else [])
            for record in records:
                record["roll_mirror_selected"] = selected
                record["roll_policy_input_observation"] = (audit.observation_reflection(record["real_policy_observation"])
                                                            if selected else record["real_policy_observation"].copy())
                # Choose synthetic model outputs whose inverse reflection equals
                # the old physical rollout; this is NOT a model inference claim.
                physical = record["roll_actor_output_physical_raw_action"]
                record["roll_actor_output_model_raw_action"] = audit.joint_reflection(physical) if selected else physical.copy()
        values["mirror_diagnostic"]["selected_trials"] = sum(x["selected"] for x in values["mirror_diagnostic"]["selection_records"])
    return result, output, selected_count


class CoordinateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_bytes = OFF_PATH.read_bytes()
        cls.report = audit.read_json(OFF_PATH)
        cls.trace, cls.digest = audit.load_trace(OFF_PATH, cls.report)
        cls.hard = audit.read_json(HARD_PATH)
        cls.hard_trace, _ = audit.load_trace(HARD_PATH, cls.hard)

    def test_real_off_report_and_original_sources(self):
        audit.verify_frozen_source()
        details, masks = audit.audit_report(self.report, self.trace)
        self.assertEqual(details["poses"]["side"]["selected"], 0)
        self.assertEqual(details["poses"]["side"]["trials"], 20)
        comparison = audit.compare_unselected(self.hard, self.report, self.hard_trace, self.trace, masks)
        self.assertTrue(comparison["exact_unselected_common_fields"])
        self.assertEqual(comparison["unselected_trials_compared"], 20)

    def test_synthetic_on_coordinates_and_unselected_full_common_fields(self):
        report, trace, selected = synthetic_on(self.report, self.trace)
        details, masks = audit.audit_report(report, trace)
        self.assertEqual(selected, 11)
        self.assertEqual(details["poses"]["side"]["selected"], 11)
        comparison = audit.compare_unselected(self.report, report, self.trace, trace, masks)
        self.assertEqual(comparison["unselected_trials_compared"], 9)
        self.assertIn("successes", comparison["selected_cohort_aggregate_fields_not_expected_equal"]["side"])

    def test_selected_model_output_reflection_error_rejected(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        trace["poses"]["side"][3]["rows"][0]["roll_actor_output_model_raw_action"][0] += .001
        with self.assertRaisesRegex(ValueError, "physical12"):
            audit.audit_report(report, trace)

    def test_polar_axial_last_action_mapping_errors_rejected(self):
        for index in (1, 3, 5, 7, 10, 11, 12, 24, 36):
            report, trace, _ = synthetic_on(self.report, self.trace)
            trace["poses"]["side"][3]["rows"][0]["roll_policy_input_observation"][index] += .01
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "virtual48"):
                audit.audit_report(report, trace)

    def test_virtual_must_not_replace_real_legacy_observation(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        row = trace["poses"]["side"][3]["rows"][0]
        row["policy_observation"] = row["roll_policy_input_observation"].copy()
        with self.assertRaisesRegex(ValueError, "legacy48"):
            audit.audit_report(report, trace)

    def test_mask_must_match_actual_start_not_requested_pose_or_id(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        selection = report["results"]["side"]["mirror_diagnostic"]["selection_records"][0]
        selection["selected"] = True
        trace["poses"]["side"][0]["mirror_selection"]["selected"] = True
        with self.assertRaisesRegex(ValueError, "actual start"):
            audit.audit_report(report, trace)

    def test_mask_cannot_change_mid_episode(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        trace["poses"]["side"][3]["rows"][1]["roll_mirror_selected"] = False
        with self.assertRaisesRegex(ValueError, "fixed mask"):
            audit.audit_report(report, trace)

    def test_normalized_y_tolerance_does_not_relax_selection(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        selection = report["results"]["side"]["mirror_diagnostic"]["selection_records"][3]
        selection["normalized_policy_start_gravity_y"] += 1e-3
        trace["poses"]["side"][3]["mirror_selection"] = copy.deepcopy(selection)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            audit.audit_report(report, trace)

    def test_missing_trial_row_or_switch_rejected(self):
        for kind in ("trial", "row", "switch"):
            report, trace, _ = synthetic_on(self.report, self.trace)
            if kind == "trial": trace["poses"]["side"].pop()
            elif kind == "row": trace["poses"]["side"][0]["rows"].pop(2)
            else: report["results"]["side"]["handoff_diagnostic"]["switch_records"][0] = None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.audit_report(report, trace)

    def test_actual_history_or_target_corruption_rejected(self):
        for field in ("actual_raw_action", "actual_previous_raw_action", "actual_executed_joint_target_rad", "joint_target_delta_rad"):
            report, trace, _ = synthetic_on(self.report, self.trace)
            trace["poses"]["side"][3]["rows"][0][field][0] += .001
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.audit_report(report, trace)

    def test_switch_coordinates_checked_separately(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        report["results"]["side"]["handoff_diagnostic"]["switch_records"][3]["roll_policy_input_observation"][1] += .001
        with self.assertRaisesRegex(ValueError, "virtual48"):
            audit.audit_report(report, trace)

    def test_unselected_tiny_physical_result_action_and_metadata_changes_fail(self):
        for kind in ("final", "trace", "metadata"):
            report, trace, _ = synthetic_on(self.report, self.trace)
            _, masks = audit.audit_report(report, trace)
            if kind == "final": report["results"]["side"]["final_diagnostics"][0]["height_m"] += 1e-15
            elif kind == "trace": trace["poses"]["side"][0]["rows"][0]["before"]["height_m"] += 1e-15
            else: report["criterion"] += " changed"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.compare_unselected(self.report, report, self.trace, trace, masks)

    def test_selected_start_must_still_be_exact(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        _, masks = audit.audit_report(report, trace)
        report["results"]["side"]["policy_start_state"][3]["height_m"] += 1e-15
        with self.assertRaises(ValueError):
            audit.compare_unselected(self.report, report, self.trace, trace, masks)

    def test_unknown_common_result_field_is_not_silently_ignored(self):
        report, trace, _ = synthetic_on(self.report, self.trace)
        _, masks = audit.audit_report(report, trace)
        base = copy.deepcopy(self.report)
        base["results"]["side"]["future_unknown_field"] = 1
        report["results"]["side"]["future_unknown_field"] = 1
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            audit.compare_unselected(base, report, self.trace, trace, masks)

    def test_entire_unselected_pose_compares_all_aggregate_fields(self):
        _, masks = audit.audit_report(self.report, self.trace)
        report = copy.deepcopy(self.report)
        report["results"]["side"]["successes"] += 1
        with self.assertRaises(ValueError):
            audit.compare_unselected(self.report, report, self.trace, self.trace, masks)

    def test_duplicate_nonfinite_missing_and_type_change_rejected(self):
        with self.assertRaises(ValueError): audit.duplicate_reject([("a", 1), ("a", 2)])
        with self.assertRaises(ValueError): audit.finite({"a": float("inf")})
        with self.assertRaises(ValueError): audit.exact_subset({"a": 1}, {})
        with self.assertRaises(ValueError): audit.exact_subset(1, 1.0)
        self.assertEqual(hashlib.sha256(OFF_PATH.read_bytes()).digest(), hashlib.sha256(self.original_bytes).digest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
