"""Stdlib CPU tests on synthetic clones, NOT independent mirror simulator proof."""

import copy
import hashlib
import math
from pathlib import Path
import unittest

import compare_handoff_mirror_baseline as audit


def fixture(old):
    report = copy.deepcopy(old)
    report.update(protocol_version=audit.PROTOCOL, controller_type=audit.CONTROLLER,
                  baseline_transition_protocol_version=audit.exact.PROTOCOL)
    trace = Path(report["checkpoint"]).name + "_handoff_mirror_trace.json"
    report["transition_experiment"].update(evaluator_source_sha256="a" * 64,
                                          transition_trace_file=trace, transition_trace_sha256="b" * 64)
    report["mirror_experiment"] = {
        "mode": "off", "selection_rule": audit.SELECTION_RULE,
        "normalized_gravity_y_threshold": -math.cos(math.radians(30)),
        "fixed_mask_per_episode": True, "roll_only": True, "standing_uses_real_observation": True,
        "gate_uses_real_state": True, "physics_or_history_mutated": False, "ramp_enabled": False,
        "retry_enabled": False, "startup_selection_changed": False, "step_dt": .02,
        "horizon_s": 8.0, "hold_s": 3.0, "min_contacts": 4, "policy_control_steps": 550,
        "baseline_transition_evaluator_sha256": audit.TRANSITION_SHA,
        "baseline_transition_math_sha256": audit.TRANSITION_MATH_SHA,
        "evaluator_source_sha256": "a" * 64, "math_source_sha256": "c" * 64,
        "mirror_trace_file": trace, "mirror_trace_sha256": "b" * 64,
    }
    for result in report["results"].values():
        selections = [{"trial": i, "selected": False, "mask_latched_for_episode": True,
                       "real_policy_start_projected_gravity_b": copy.deepcopy(start["projected_gravity_b"]),
                       "eligible_settled_fallen_at_policy_start": start["eligible_settled_fallen_recovery"]}
                      for i, start in enumerate(result["policy_start_state"])]
        result["mirror_diagnostic"] = {"selected_trials": 0, "total_trials": result["trials"],
            "all_trials_in_success_denominator": True, "selection_records": selections}
        for row in result["handoff_diagnostic"]["switch_records"]:
            if row is not None:
                row.update(roll_mirror_selected=False,
                    real_policy_observation=copy.deepcopy(row["policy_observation"]),
                    roll_policy_input_observation=copy.deepcopy(row["policy_observation"]),
                    roll_actor_output_model_raw_action=copy.deepcopy(row["roll_actor_raw_action"]),
                    roll_actor_output_physical_raw_action=copy.deepcopy(row["roll_actor_raw_action"]))
    return report


class MirrorBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = [audit.ROOT / f"evaluations/20260917-handoff-transition-v1/hard-{pose}/model_1999.pt_recovery_metrics.json"
                     for pose in ("side", "upside_down", "upright")]
        cls.baselines = [audit.exact.load_report(path)[0] for path in cls.paths]

    def setUp(self):
        self.old = copy.deepcopy(self.baselines[0])
        self.new = fixture(self.old)

    def rejected(self):
        try:
            result = audit.compare_reports(self.old, self.new)
        except audit.exact.ComparisonError:
            return
        self.assertFalse(result["exact_mirror_off_equivalence"])

    def test_three_actual_hard_baselines_synthetic_positive_only(self):
        self.assertEqual([next(iter(x["results"].values()))["final_valid_stands"] for x in self.baselines], [9, 20, 12])
        for old in self.baselines:
            new = fixture(old)
            before = copy.deepcopy((old, new))
            result = audit.compare_reports(old, new)
            self.assertTrue(result["exact_mirror_off_equivalence"])
            self.assertFalse(result["acceptance_eligible"])
            self.assertFalse(result["source_files_verified"])
            self.assertIn("Synthetic", result["notice"])
            self.assertEqual((old, new), before)

    def test_actual_frozen_sources_match(self):
        for path, digest in audit.SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_new_metadata_allowed_without_loosening_existing_fields(self):
        self.new["results"]["side"]["extra"] = {"diagnostic_only": True}
        self.assertTrue(audit.compare_reports(self.old, self.new)["exact_mirror_off_equivalence"])
        self.new["results"]["side"]["final_diagnostics"][3]["height_m"] += 1e-12
        self.rejected()

    def test_result_order_missing_fields_and_scalar_types(self):
        for change in ("order", "missing", "int", "tiny"):
            self.new = fixture(self.old)
            r = self.new["results"]["side"]
            if change == "order": r["policy_start_state"].reverse()
            if change == "missing": del r["handoff_diagnostic"]["switch_records"][0]["previous_raw_action"]
            if change == "int": r["trials"] = 20.0
            if change == "tiny": r["policy_start_state"][0]["height_m"] += 1e-12
            self.rejected()

    def test_old_metadata_remains_exact(self):
        for key, value in (("seed", 1), ("criterion", "weakened"), ("self_collisions_enabled", False)):
            self.new = fixture(self.old)
            self.new[key] = value
            self.rejected()
        self.new = fixture(self.old)
        self.new["transition_experiment"]["actual_joint_target_asserted_every_step_atol_rad"] = .01
        self.rejected()

    def test_wrong_fixed_models_or_hard_semantics(self):
        for label, key, value in (("handoff_controller", "stand_sha256", "d" * 64),
                                  ("transition_experiment", "ramp_seconds", .2),
                                  ("transition_experiment", "retry_enabled", True),
                                  ("transition_experiment", "math_source_sha256", "d" * 64)):
            self.new = fixture(self.old)
            self.new[label][key] = value
            self.rejected()

    def test_mirror_on_never_passes_off_comparator(self):
        self.new["mirror_experiment"]["mode"] = "initial_right"
        self.rejected()

    def test_wrong_selector_rule_and_mutation_claims_rejected(self):
        for key, value in (("normalized_gravity_y_threshold", -.5), ("selection_rule", "requested side pose"),
                           ("startup_selection_changed", True), ("standing_uses_real_observation", False),
                           ("physics_or_history_mutated", True)):
            self.new = fixture(self.old)
            self.new["mirror_experiment"][key] = value
            self.rejected()

    def test_off_added_coordinates_must_equal_real_values(self):
        for key in ("real_policy_observation", "roll_policy_input_observation",
                    "roll_actor_output_model_raw_action", "roll_actor_output_physical_raw_action"):
            self.new = fixture(self.old)
            self.new["results"]["side"]["handoff_diagnostic"]["switch_records"][0][key][0] += 1e-12
            self.rejected()

    def test_off_selection_and_trace_identity_fail_closed(self):
        self.new["results"]["side"]["mirror_diagnostic"]["selection_records"][0]["selected"] = True
        self.rejected()
        self.new = fixture(self.old)
        self.new["mirror_experiment"]["mirror_trace_file"] = "../escaped.json"
        self.rejected()
        self.new = fixture(self.old)
        self.new["extra"] = float("nan")
        self.rejected()


if __name__ == "__main__":
    unittest.main(verbosity=2)
