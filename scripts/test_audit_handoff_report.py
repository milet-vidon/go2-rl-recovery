"""Standard-library tests; fixtures and optional real reports are read-only."""

import copy
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import audit_handoff_report as audit
from test_audit_recovery_report import fixture as physical_fixture, success_fixture


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "evaluations/20260916-handoff2998-to3448-smoke/model_2998.pt_recovery_metrics.json"


def fixture(trigger=True, final_valid=False, bank=True):
    report = success_fixture() if final_valid else physical_fixture(bank)
    names = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf")
             for leg in ("FL", "FR", "RL", "RR")]
    report.update(protocol_version=report["protocol_version"] + audit.SUFFIX,
                  task=audit.TASK, controller_type=audit.CONTROLLER,
                  single_policy_acceptance_eligible=False,
                  success_count_definition=audit.COUNT_PREFIX + "Recorded valid holds from all starts.",
                  checkpoint="E:/fixtures/roll.pt", joint_names=names,
                  action_representation={"reference": "nominal", "scale": .25,
                                         "target": "q_reference + scale * action, soft-joint-limit clamped",
                                         "sample_period_s": .02, "held_over_physics_substeps": 4})
    report["handoff_controller"] = {
        "roll_checkpoint": report["checkpoint"], "roll_sha256": report["checkpoint_sha256"],
        "stand_checkpoint": "E:/fixtures/stand.pt", "stand_sha256": "b" * 64,
        "gate": audit.GATE, "state_or_action_history_reset_at_switch": False,
        "manual_standing_pd_at_switch": False, "physics_changed_at_switch": False,
        "joint_limits": {"joint_names": names, "default_joint_positions_rad": [0] * 12,
                         "soft_joint_limits_rad": [[-3, 3] for _ in range(12)],
                         "default_target_would_be_clamped": False},
    }
    record = {"trial": 0, "policy_time_s": .2, "root_position_local_m": [0, 0, .16],
              "root_quaternion_wxyz": [1, 0, 0, 0], "root_linear_velocity_w_m_s": [0, 0, 0],
              "root_angular_velocity_w_rad_s": [0, 0, 0], "joint_positions_rad": [0] * 12,
              "joint_velocities_rad_s": [0] * 12, "action_linf_jump": 2.,
              "clamped_joint_target_linf_jump_rad": .5, "two_actor_action_linf_difference": 2.}
    report["results"]["side"]["handoff_diagnostic"] = {
        "triggered_trials": int(trigger), "untriggered_trials": int(not trigger),
        "final_valid_after_trigger": int(trigger and final_valid),
        "total_success_denominator_includes_untriggered": True,
        "switch_records": [record if trigger else None],
        "clamped_target_fraction_per_joint": [[0.] * 12], "clamp_sample_period_s": .02,
    }
    return report


class HandoffAuditTests(unittest.TestCase):
    def reject(self, path, value, report=None):
        report = copy.deepcopy(fixture() if report is None else report)
        item = report
        for key in path[:-1]:
            item = item[key]
        item[path[-1]] = value
        with self.assertRaises(audit.AuditError, msg=str(path)):
            audit.audit_handoff_report(report)

    def test_success_and_failure_are_consistency_not_acceptance(self):
        for trigger, final in ((False, False), (True, False), (True, True)):
            output = audit.audit_handoff_report(fixture(trigger, final))
            self.assertTrue(output["report_internally_valid"])
            self.assertFalse(output["single_policy_acceptance_eligible"])
            self.assertFalse(output["checkpoint_files_verified"])
            self.assertIn("NOT model acceptance", output["notice"])
            self.assertEqual(output["poses"][0]["final_valid_after_trigger"], int(trigger and final))

    def test_controlled_drop_supported_without_rewriting_input(self):
        report = fixture(bank=False)
        saved = copy.deepcopy(report)
        self.assertTrue(audit.audit_handoff_report(report)["common_physics_report_checked"])
        self.assertEqual(report, saved)

    def test_dual_metadata_and_disabled_mutations(self):
        cases = ((["protocol_version"], "stance_geometry_v1"),
                 (["controller_type"], "single_policy"), (["task"], "another task"),
                 (["single_policy_acceptance_eligible"], True),
                 (["single_policy_acceptance_eligible"], 0),
                 (["self_collisions_enabled"], False),
                 (["success_count_definition"], "Success"),
                 (["handoff_controller", "gate"], audit.GATE.replace("0.2s", "0.18s")))
        for path, value in cases:
            self.reject(path, value)
        for key in ("state_or_action_history_reset_at_switch", "manual_standing_pd_at_switch", "physics_changed_at_switch"):
            for value in (True, None, 0):
                self.reject(["handoff_controller", key], value)

    def test_checkpoint_identity_mutations(self):
        for path, value in ((["checkpoint_sha256"], "c" * 64),
                            (["handoff_controller", "roll_sha256"], "c" * 64),
                            (["handoff_controller", "roll_checkpoint"], "E:/different.pt"),
                            (["handoff_controller", "stand_sha256"], "bad"),
                            (["handoff_controller", "stand_checkpoint"], "relative.pt"),
                            (["handoff_controller", "stand_checkpoint"], "E:/fixtures/roll.pt")):
            self.reject(path, value)

    def test_omitted_untriggered_denominator_rejected(self):
        report = fixture(trigger=False)
        self.reject(["results", "side", "handoff_diagnostic", "untriggered_trials"], 0, report)
        self.reject(["results", "side", "handoff_diagnostic", "total_success_denominator_includes_untriggered"], False, report)

    def test_trigger_count_record_and_trial_mismatches(self):
        prefix = ["results", "side", "handoff_diagnostic"]
        for tail, value in ((["triggered_trials"], 0), (["triggered_trials"], True),
                            (["switch_records"], []), (["switch_records", 0], None),
                            (["switch_records", 0, "trial"], 1), (["switch_records", 0, "trial"], False)):
            self.reject(prefix + tail, value)

    def test_switch_time_bounds_and_grid(self):
        for value in (.18, .201, 11., 12., -1., math.nan, math.inf, True):
            self.reject(["results", "side", "handoff_diagnostic", "switch_records", 0, "policy_time_s"], value)
        for value in (.2, 3.14, 10.98):
            report = fixture()
            report["results"]["side"]["handoff_diagnostic"]["switch_records"][0]["policy_time_s"] = value
            self.assertTrue(audit.audit_handoff_report(report)["report_internally_valid"])

    def test_quaternion_normalization_and_tilt(self):
        key = ["results", "side", "handoff_diagnostic", "switch_records", 0, "root_quaternion_wxyz"]
        for value in ([0, 0, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0], [1, math.nan, 0, 0]):
            self.reject(key, value)
        for degrees in (30., 30.001, 90.):
            half = math.radians(degrees / 2)
            self.reject(key, [math.cos(half), math.sin(half), 0, 0])
        report = fixture()
        half = math.radians(29.99 / 2)
        report["results"]["side"]["handoff_diagnostic"]["switch_records"][0]["root_quaternion_wxyz"] = [math.cos(half), math.sin(half), 0, 0]
        self.assertTrue(audit.audit_handoff_report(report)["report_internally_valid"])

    def test_angular_speed_norm_and_nonfinite_state(self):
        prefix = ["results", "side", "handoff_diagnostic", "switch_records", 0]
        for value in ([1., 0, 0], [.8, .8, 0], [0, 0, math.inf]):
            self.reject(prefix + ["root_angular_velocity_w_rad_s"], value)
        for key in ("root_position_local_m", "root_linear_velocity_w_m_s", "joint_positions_rad", "joint_velocities_rad_s"):
            self.reject(prefix + [key, 0], math.nan)
        for key in ("action_linf_jump", "clamped_joint_target_linf_jump_rad", "two_actor_action_linf_difference"):
            self.reject(prefix + [key], -1)

    def test_clamp_array_shape_range_and_sample_period(self):
        prefix = ["results", "side", "handoff_diagnostic"]
        for value in ([], [[0] * 11], [[0] * 12] * 2):
            self.reject(prefix + ["clamped_target_fraction_per_joint"], value)
        for value in (-.01, 1.01, True, math.nan):
            self.reject(prefix + ["clamped_target_fraction_per_joint", 0, 0], value)
        self.reject(prefix + ["clamp_sample_period_s"], .01)

    def test_final_after_trigger_exact_recorded_intersection(self):
        key = ["results", "side", "handoff_diagnostic", "final_valid_after_trigger"]
        self.reject(key, 1)
        self.reject(key, 0, fixture(final_valid=True))
        self.reject(key, 1, fixture(trigger=False))

    def test_joint_limit_metadata(self):
        prefix = ["handoff_controller", "joint_limits"]
        self.reject(prefix + ["joint_names", 0], "unknown_joint")
        self.reject(["joint_names"], list(reversed(fixture()["joint_names"])))
        self.reject(prefix + ["default_target_would_be_clamped"], True)
        self.reject(prefix + ["soft_joint_limits_rad", 0], [1, -1])
        self.reject(prefix + ["default_joint_positions_rad", 0], math.nan)

    def test_action_representation_mutations(self):
        for key, value in (("reference", "current"), ("scale", .5), ("sample_period_s", .01),
                           ("held_over_physics_substeps", 2), ("target", "unclamped")):
            self.reject(["action_representation", key], value)

    def test_original_physical_auditor_is_still_enforced(self):
        self.reject(["results", "side", "success_rate"], .5)
        self.reject(["criterion"], "relaxed criterion")
        report = fixture()
        with patch.object(audit, "audit_report", wraps=audit.audit_report) as delegated:
            audit.audit_handoff_report(report)
        shared = delegated.call_args.args[0]
        self.assertIsNot(shared, report)
        self.assertNotIn("handoff_controller", shared)
        self.assertNotIn("handoff_diagnostic", shared["results"]["side"])
        self.assertFalse(shared["protocol_version"].endswith(audit.SUFFIX))
        self.assertTrue(report["protocol_version"].endswith(audit.SUFFIX))

    def test_nonfinite_unknown_field_is_rejected(self):
        report = fixture()
        report["unknown"] = {"nested": [math.nan]}
        with self.assertRaises(audit.AuditError):
            audit.audit_handoff_report(report)

    def test_duplicate_json_rejected_without_writing_fixture(self):
        with patch.object(Path, "read_text", return_value='{"results": {}, "results": {}}'):
            with self.assertRaises(audit.AuditError):
                audit.load_report(Path("unused.json"))

    def test_cli_failure_exit_without_creating_or_changing_file(self):
        report = fixture()
        report["single_policy_acceptance_eligible"] = True
        with patch.object(audit, "load_report", return_value=report), redirect_stdout(io.StringIO()) as output:
            code = audit.main(["unused.json"])
        self.assertEqual(code, 1)
        self.assertIn("NOT model acceptance", output.getvalue())

    def test_actual_smoke_read_only(self):
        if not SMOKE.is_file():
            self.skipTest("Optional real diagnostic report absent")
        before = SMOKE.read_bytes()
        report = audit.load_report(SMOKE)
        saved = copy.deepcopy(report)
        output = audit.audit_handoff_report(report, verify_checkpoint_files=True)
        self.assertTrue(output["checkpoint_files_verified"])
        self.assertEqual(len(output["verified_checkpoint_files"]), 2)
        rows = {row["pose"]: row for row in output["poses"]}
        self.assertEqual((rows["side"]["triggered_trials"], rows["side"]["final_valid_after_trigger"]), (2, 2))
        self.assertEqual((rows["upside_down"]["triggered_trials"], rows["upside_down"]["untriggered_trials"]), (0, 2))
        self.assertEqual(report, saved)
        self.assertEqual(SMOKE.read_bytes(), before)

    def test_actual_smoke_changed_stand_hash_fails_file_verification(self):
        if not SMOKE.is_file():
            self.skipTest("Optional real diagnostic report absent")
        report = audit.load_report(SMOKE)
        report["handoff_controller"]["stand_sha256"] = "c" * 64
        with self.assertRaises(audit.AuditError):
            audit.audit_handoff_report(report, verify_checkpoint_files=True)

    def test_real_cli_success_is_not_visual_or_policy_pass(self):
        if not SMOKE.is_file():
            self.skipTest("Optional real diagnostic report absent")
        before = SMOKE.read_bytes()
        with redirect_stdout(io.StringIO()) as output:
            code = audit.main([str(SMOKE), "--verify-checkpoint-files"])
        self.assertEqual(code, 0)
        self.assertIn("NOT model acceptance", output.getvalue())
        self.assertIn("visual approval", output.getvalue())
        self.assertEqual(SMOKE.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
