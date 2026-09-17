"""Read-only regression tests for explicit nominal-PD pre-settled reports."""

import copy
from pathlib import Path
import unittest

from audit_recovery_report import AuditError, CRITERION, audit_report, load_report
from test_audit_recovery_report import fixture


ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ("physics/action substeps only; termination checked each control step; auto-reset "
             "prohibited; no learned policy, reward computation, curriculum or interval events")


def presettled_fixture():
    report = fixture(bank=False)
    report.update(protocol_version="stance_geometry_v1_presettled_PD_v1",
                  start_protocol_id="pre_settled_nominal_pose_PD", state_bank=None,
                  settle_controller="direct nominal-position PD, not policy zero action; not zero torque",
                  settle_execution=EXECUTION, settle_requested_s=1., settle_actual_s=1.,
                  settle_control_steps=50, action_representation={"sample_period_s": .02})
    return report


class PresettledAuditTests(unittest.TestCase):
    def reject(self, keys, value, report=None):
        candidate = copy.deepcopy(presettled_fixture() if report is None else report)
        target = candidate
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        with self.assertRaises(AuditError, msg=f"{keys}: {value!r}"):
            audit_report(candidate)

    def test_explicit_protocol_valid_failure_is_not_policy_acceptance(self):
        report = presettled_fixture()
        before = copy.deepcopy(report)
        result = audit_report(report)
        self.assertTrue(result["report_internally_valid"])
        self.assertEqual(result["protocol"], "stance_geometry_v1_presettled_PD_v1")
        self.assertIn("NOT model acceptance", result["notice"])
        self.assertEqual(result["poses"][0]["final_valid_hold"], 0)
        self.assertEqual(report, before)

    def test_protocol_identity_and_bank_rejected(self):
        for keys, value in (
            (("start_protocol_id",), "controlled_drop"),
            (("start_protocol_id",), "state_bank_heldout_nominal_pose_PD"),
            (("state_bank",), {}),
            (("state_bank",), {"schema_version": "nominal_pd_fallen_v1"}),
            (("protocol_version",), "stance_geometry_v1"),
            (("protocol_version",), "stance_geometry_v1_state_bank_PD_v1"),
            (("protocol_version",), "stance_geometry_v1_presettled_PD_v2"),
        ):
            with self.subTest(keys=keys, value=value):
                self.reject(keys, value)

    def test_controller_and_no_reset_execution_required(self):
        for value in (None, "zero action, default joint-position PD; not zero torque",
                      "zero torque", "learned policy"):
            self.reject(("settle_controller",), value)
        for value in (None, "", EXECUTION.replace("auto-reset prohibited", "auto-reset allowed"),
                      EXECUTION.replace("no learned policy", "learned policy")):
            self.reject(("settle_execution",), value)

    def test_angle_is_finite_nonnegative_number(self):
        for value in (None, True, "30", -1, float("nan"), float("inf"), float("-inf")):
            self.reject(("angle_deg",), value)

    def test_durations_strictly_positive_finite_numbers(self):
        for field in ("settle_requested_s", "settle_actual_s"):
            for value in (0, -1, None, True, "1", float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    self.reject((field,), value)

    def test_steps_strictly_positive_integer(self):
        for value in (0, -1, None, True, 50., "50", float("nan"), float("inf"), 49, 51):
            self.reject(("settle_control_steps",), value)

    def test_ceiling_and_actual_duration(self):
        for requested, steps, actual in ((.001, 1, .02), (.02, 1, .02), (.021, 2, .04),
                                         (1.001, 51, 1.02)):
            report = presettled_fixture()
            report.update(settle_requested_s=requested, settle_control_steps=steps, settle_actual_s=actual)
            self.assertTrue(audit_report(report)["report_internally_valid"])
            self.reject(("settle_control_steps",), steps + 1, report)
            self.reject(("settle_actual_s",), actual + .001, report)
        report = presettled_fixture()
        report.update(settle_requested_s=1.001, settle_control_steps=50, settle_actual_s=1.)
        with self.assertRaises(AuditError):
            audit_report(report)

    def test_action_period_if_supplied_is_observed_twenty_ms(self):
        for value in (None, 0, -.02, .01, .0200001, True, ".02", float("nan"), float("inf")):
            self.reject(("action_representation", "sample_period_s"), value)
        for value in ([], "unknown", False):
            self.reject(("action_representation",), value)
        for omit_action in (False, True):
            report = presettled_fixture()
            if omit_action:
                del report["action_representation"]
            else:
                del report["action_representation"]["sample_period_s"]
            self.assertTrue(audit_report(report)["report_internally_valid"])

    def test_acceptance_geometry_and_counts_not_weakened(self):
        self.reject(("criterion",), CRITERION.replace("4 simultaneous", "2 simultaneous"))
        self.reject(("results", "side", "final_valid_stands"), 1)
        self.reject(("results", "side", "final_diagnostics", 0, "stable_hold_s"), .02)

    def test_old_protocol_behavior_unchanged(self):
        for bank in (False, True):
            report = fixture(bank)
            self.assertTrue(audit_report(report)["report_internally_valid"])
        for field in ("settle_requested_s", "settle_actual_s", "settle_control_steps"):
            self.reject((field,), 1, fixture(bank=False))

    def test_real_reports_read_only(self):
        paths = [ROOT / "evaluations/20260917-denseposture-smoke-upright-pd1/model_2000.pt_recovery_metrics.json",
                 ROOT / "evaluations/20260916-smith-stand-compat3448-pd1/recovery_aligned_model3448.pt_recovery_metrics.json"]
        existing = [path for path in paths if path.is_file()]
        if not existing:
            self.skipTest("Optional real pre-settled report fixtures are absent")
        for path in existing:
            with self.subTest(report=path):
                before = path.read_bytes()
                report = load_report(path)
                summary = audit_report(report)
                self.assertEqual(summary["protocol"], "stance_geometry_v1_presettled_PD_v1")
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(summary["poses"][0]["eligible_settled_fallen"], 0)
        print(f"Read-only pre-settled fixtures: {len(existing)} reports")


if __name__ == "__main__":
    unittest.main()
