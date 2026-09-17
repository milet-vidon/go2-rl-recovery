"""CPU/stdlib identity checks; fixtures derive from actual4046 report schema.

No simulator/PyTorch import. The4246 identity substitutions below are explicit
test fixtures, not fabricated experiment evidence or accepted performance.
"""
import copy
import csv
import io
from pathlib import Path
import unittest
from unittest import mock

import audit_balanced_gait_screen_identity as audit

BASE = audit.ROOT / "evaluations/20260917-commonphysics4046-first"


def fixture():
    summary = audit.read_json(BASE / "summary.json")
    row = copy.deepcopy(next(r for r in summary["rows"] if r["physics"] == "recovery" and r["case"] == "normal"))
    report = audit.read_json(row["report"])
    invocation = audit.read_json(Path(row["report"]).parent / "invocation.json")
    checkpoint = Path("E:/IsaacLab/tmp/identity-fixture/model_4246.pt")
    receipt = checkpoint.parent / "common_physics_training_result.json"
    checkpoint_sha, receipt_sha = "a"*64, "b"*64
    row.update(arm="balanced", reused_existing=False)
    report.update(protocol_version=audit.REPORT_PROTOCOL, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha)
    proof = report["candidate_training"]
    proof.update(arm="balanced", balanced_duration_weight=-10.0, formal_updates_verified=300,
                 actual_environment_steps=921600, actual_control_steps=7200, actual_iteration=4246,
                 actual_adam_steps=[85120]*17, training_receipt=str(receipt), training_receipt_sha256=receipt_sha,
                 checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha)
    invocation.update(schema="trained_balanced_gait_candidate_invocation_v1", arm="balanced",
                      checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha)
    argv = invocation["args"]
    argv.extend(["--arm", "balanced"])
    for key, value in (("--checkpoint", str(checkpoint)), ("--candidate_training_result", str(receipt))):
        argv[argv.index(key)+1] = value
    return row, report, invocation, checkpoint, checkpoint_sha, receipt, receipt_sha


def validate(value):
    row, report, invocation, checkpoint, checkpoint_sha, receipt, receipt_sha = value
    return audit.validate_case(row, report, invocation, "balanced", checkpoint, checkpoint_sha, receipt, receipt_sha)


class ScreenIdentityTests(unittest.TestCase):
    def test_schema_derived_valid_binding_preserves_behavior_failure(self):
        value = fixture()
        self.assertIs(value[0]["passed"], False)  # Real4046 normal case fails lift.
        validate(value)  # Evidence audit is not a behavior acceptance gate.

    def test_actual_report_seed_not_summary_label_must_match(self):
        value = fixture(); value[1]["seed"] = 20260910
        with self.assertRaisesRegex(ValueError, "report seed"): validate(value)

    def test_swapping_successful_speed_report_cannot_fake_another_case(self):
        value = fixture(); value[1]["protocol"]["walk_speed"] = .8
        with self.assertRaisesRegex(ValueError, "actual command"): validate(value)

    def test_actual_physics_and_arm_are_bound(self):
        for target, key, wrong in (("retention_experiment", "physics_mode", "original"),
                                   ("candidate_training", "arm", "control"),
                                   ("candidate_training", "balanced_duration_weight", 0.0)):
            value = fixture(); value[1][target][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError): validate(value)

    def test_actual_receipt_path_and_sha_are_bound(self):
        for key, wrong in (("training_receipt", "E:/IsaacLab/tmp/other.json"),
                           ("training_receipt_sha256", "c"*64), ("checkpoint_sha256", "d"*64)):
            value = fixture(); value[1]["candidate_training"][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError): validate(value)

    def test_formal_budget_and_optimizer_ledger_bound(self):
        for key, wrong in (("formal_updates_verified",100),("actual_environment_steps",307200),
                           ("actual_control_steps",2400),("actual_iteration",4046),
                           ("actual_adam_steps",[81120]*17)):
            value = fixture(); value[1]["candidate_training"][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError): validate(value)

    def test_actual_invocation_commands_and_arm_bound(self):
        for key, wrong in (("--walk_speed","0.8"),("--seed","20260910"),("--arm","control"),
                           ("--physics_mode","original"),("--push_speed","0.5")):
            value = fixture(); args = value[2]["args"]; args[args.index(key)+1] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError): validate(value)

    def test_duplicate_cli_argument_rejected(self):
        value = fixture(); value[2]["args"] += ["--arm", "balanced"]
        with self.assertRaisesRegex(ValueError, "duplicated"): validate(value)

    def test_nonboolean_pass_or_suppressed_failure_rejected(self):
        value = fixture(); value[0]["passed"] = 0
        with self.assertRaisesRegex(ValueError, "passed label"): validate(value)
        value = fixture(); value[0]["failed_criteria"] = []
        with self.assertRaisesRegex(ValueError, "failure list"): validate(value)

    def test_changed_drift_threshold_label_rejected(self):
        value = fixture(); value[1]["retention_acceptance"]["straight_drift"] = False
        with self.assertRaisesRegex(ValueError, "retention gate"): validate(value)

    def summary(self):
        rows = [{"physics":p,"case":c,"seed":s,"arm":"balanced","passed":False,"reused_existing":False,
                 "report":f"E:/IsaacLab/tmp/test/{p}-{c}-{s}/model_4246_stand_walk_stop.json",
                 "invocation":f"E:/IsaacLab/tmp/test/{p}-{c}-{s}/invocation.json"}
                for p,c,s in sorted(audit.expected_cases())]
        return {"protocol":"balanced_gait_candidate21_screen_v1","status":"completed_screening_only",
                "arm":"balanced","rows":rows,"promotion_performed":False,"training_performed":False,"reused_cases":0}

    def test_exact21_identity_set_with_all_failures_is_valid_evidence(self):
        audit.validate_summary(self.summary(), "balanced")

    def test_duplicate_actual_report_rejected_despite_unique21_row_labels(self):
        summary = self.summary(); summary["rows"][1]["report"] = summary["rows"][0]["report"]
        with self.assertRaisesRegex(ValueError, "Duplicate actual report"): audit.validate_summary(summary,"balanced")

    def test_duplicate_invocation_and_case_identity_rejected(self):
        summary = self.summary(); summary["rows"][1]["invocation"] = summary["rows"][0]["invocation"]
        with self.assertRaisesRegex(ValueError, "Duplicate actual invocation"): audit.validate_summary(summary,"balanced")
        summary = self.summary(); summary["rows"][1] = copy.deepcopy(summary["rows"][0])
        with self.assertRaisesRegex(ValueError, "21-case"): audit.validate_summary(summary,"balanced")

    def test_json_duplicate_and_nonfinite_rejected(self):
        for value in ('{"arm":"control","arm":"balanced"}', '{"x":NaN}'):
            with mock.patch.object(Path,"read_text",return_value=value), self.assertRaises(ValueError):
                audit.read_json(Path("unused.json"))


class ActualTrajectorySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = BASE / "recovery-normal-seed20260909"
        cls.csv_path = folder / "model_4046_stand_walk_stop.csv"
        cls.trace = audit.read_json(folder / "retention_interface.json")
        # Identity alias only for this test; all actual900 measurements unchanged.
        cls.trace["schema"] = audit.REPORT_PROTOCOL

    def test_real_4046_900_step_timeline_and_command_schema(self):
        audit.validate_trajectory(self.csv_path, self.trace, "recovery", "normal")

    def test_observed_command_mismatch_rejected(self):
        trace = copy.deepcopy(self.trace); trace["steps"][200]["observation"][0][9] = .8
        with self.assertRaisesRegex(ValueError, "observed command"):
            audit.validate_trajectory(self.csv_path, trace, "recovery", "normal")

    def test_csv_wrong_command_rejected_without_rewriting_source(self):
        with self.csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[200]["cmd_x"] = "0.8"
        output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with mock.patch.object(Path,"open",return_value=io.StringIO(output.getvalue())):
            with self.assertRaisesRegex(ValueError, "CSV actual command"):
                audit.validate_trajectory(self.csv_path, self.trace, "recovery", "normal")

    def test_missing_actual_trace_interval_rejected(self):
        trace = copy.deepcopy(self.trace); trace["steps"].pop()
        with self.assertRaisesRegex(ValueError, "900-step"):
            audit.validate_trajectory(self.csv_path, trace, "recovery", "normal")

    def reset_fixture(self, csv_done="1", trace_done=1):
        with self.csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[10]["done"] = csv_done
        output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        trace = copy.deepcopy(self.trace); trace["steps"][10]["done"] = [trace_done]
        return output.getvalue(), trace

    def test_recorded_reset_behavior_failure_keeps_identity_valid(self):
        text, trace = self.reset_fixture()
        with mock.patch.object(Path,"open",return_value=io.StringIO(text)):
            measured = audit.validate_trajectory(self.csv_path, trace, "recovery", "normal")
        self.assertEqual(measured["done_intervals"], 1)
        audit.validate_reset_check({"acceptance":{"no_reset":False}}, measured)

    def test_forged_no_reset_pass_rejected_against_actual_done(self):
        with self.assertRaisesRegex(ValueError, "actual measured done"):
            audit.validate_reset_check({"acceptance":{"no_reset":True}}, {"done_intervals":1})
        with self.assertRaisesRegex(ValueError, "actual measured done"):
            audit.validate_reset_check({"acceptance":{"no_reset":0}}, {"done_intervals":1})

    def test_done_disagreement_and_invalid_type_rejected(self):
        for csv_done, trace_done in (("0",1),("1",0),("1",1.0),("0",2),("false",False)):
            text, trace = self.reset_fixture(csv_done, trace_done)
            with self.subTest(csv=csv_done, trace=trace_done), mock.patch.object(Path,"open",return_value=io.StringIO(text)):
                with self.assertRaisesRegex(ValueError, "done"):
                    audit.validate_trajectory(self.csv_path, trace, "recovery", "normal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
