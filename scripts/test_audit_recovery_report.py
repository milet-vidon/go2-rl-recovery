"""Standard-library regression tests; real report fixtures are read-only/optional."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from audit_recovery_report import AuditError, CRITERION, audit_report, load_report, main


ROOT = Path(__file__).resolve().parents[1]


def fixture(bank=True):
    start = {
        "trial": 0, "projected_gravity_b": [0, 1, 0], "root_position_w_m": [0, 0, .15],
        "root_quaternion_wxyz": [1, 0, 0, 0], "root_linear_velocity_w_m_s": [0, 0, 0],
        "root_angular_velocity_w_rad_s": [0, 0, 0], "joint_positions_rad": [0] * 12,
        "geometry_ok": True, "settled": True, "eligible_settled_fallen_recovery": True,
        "contacts_fresh_since_pose_write": True, "gravity_error": 1.414, "max_joint_speed_rad_s": 0,
        "quiet_supported_window_s": 1, "height_m": .15, "tilt_from_upright_deg": 90,
        "base_contact": False, "any_body_contact": True, "standing_at_policy_start": False,
        "fallen_at_policy_start": True, "foot_vertical_forces_N": [0] * 4, "vertical_foot_contacts": 0,
        "bank_state_id": 12, "bank_saved_pose_class": "left", "bank_requested_pose_class": "left",
        "actual_pose_class": "left", "classification": "settled_fallen_under_nominal_pose_PD",
    }
    release = copy.deepcopy(start)
    release.update(contacts_fresh_since_pose_write=False, settled=False, quiet_supported_window_s=0,
                   eligible_settled_fallen_recovery=False,
                   classification="controlled_drop_unsettled_contacts_not_yet_valid")
    for key in ("base_contact", "any_body_contact", "standing_at_policy_start", "fallen_at_policy_start",
                "foot_vertical_forces_N", "vertical_foot_contacts"):
        release[key] = None
    final = {"trial": 0, "geometry_ok": True, "stable_hold_s": 0, "height_m": .15,
             "gravity_error": 1.414, "vertical_foot_contacts": 0, "feet_y_b": [.17, -.17, .17, -.17],
             "knees_y_b": [.15, -.15, .15, -.15], "max_joint_offset_rad": .1, "bank_state_id": 12}
    result = {"trials": 1, "successes": 0, "legacy_contact_height_successes": 0,
              "final_valid_stands": 0, "final_geometry_passes": 1,
              "standing_starts_not_fallen_recovery": 0, "settled_fallen_trials": 1,
              "settled_fallen_recovery_successes": 0, "settled_fallen_final_valid_stands": 0,
              "settled_fallen_recovery_rate": 0, "success_rate": 0, "median_recovery_s": None,
              "p90_recovery_s": None, "stable_hold_s": 3, "horizon_s": 8,
              "policy_start_state": [start], "release_state_before_settling": [release],
              "final_diagnostics": [final], "settled_fallen_unique_state_count": 1,
              "settled_fallen_median_recovery_s": None,
              "state_bank_selection": {"selected_state_ids": [12], "selected_unique_state_count": 1,
                  "available_unique_states": 1, "sampling_with_replacement": False,
                  "selected_actual_pose_classes": ["left"], "selected_requested_pose_classes": ["left"],
                  "actual_pose_class_counts": {"left": 1, "right": 0}}}
    report = {"protocol_version": "stance_geometry_v1_state_bank_PD_v1", "criterion": CRITERION,
              "acceptance_eligible": True, "self_collisions_enabled": True, "policy_action_mode": "deterministic_mean",
              "video_view": "front", "checkpoint_sha256": "a" * 64, "seed": 20260918,
              "start_protocol_id": "state_bank_heldout_nominal_pose_PD", "angle_deg": None,
              "settle_controller": "direct nominal-position PD, not policy zero action; not zero torque",
              "settle_requested_s": 1, "settle_actual_s": 1, "settle_control_steps": 50,
              "state_bank": {"schema_version": "nominal_pd_fallen_v1", "split": "heldout",
                             "diagnostic_train_split_only": False, "selected_state_ids": [12],
                             "selected_unique_state_count": 1, "available_split_states": 1},
              "results": {"side": result}}
    if not bank:
        report.update(protocol_version="stance_geometry_v1", start_protocol_id="controlled_drop", angle_deg=30,
                      settle_requested_s=0, settle_actual_s=0, settle_control_steps=0, settle_controller=None)
        del report["state_bank"]
        result["policy_start_state"] = [copy.deepcopy(release)]
        result.update(settled_fallen_trials=0, settled_fallen_recovery_rate=None)
        for key in ("state_bank_selection", "settled_fallen_unique_state_count", "settled_fallen_median_recovery_s"):
            del result[key]
        for record in (*result["policy_start_state"], release, final):
            for key in tuple(record):
                if key.startswith("bank_") or key == "actual_pose_class":
                    del record[key]
    return report


def success_fixture(final_valid=True):
    report = fixture()
    result = report["results"]["side"]
    result.update(successes=1, legacy_contact_height_successes=1, success_rate=1,
                  settled_fallen_recovery_successes=1, settled_fallen_recovery_rate=1,
                  median_recovery_s=2, p90_recovery_s=2, settled_fallen_median_recovery_s=2)
    if final_valid:
        result.update(final_valid_stands=1, settled_fallen_final_valid_stands=1)
        result["final_diagnostics"][0].update(stable_hold_s=3, height_m=.32, gravity_error=0, vertical_foot_contacts=4)
    return report


class AuditTests(unittest.TestCase):
    def mutate_rejected(self, report, path, value):
        report = copy.deepcopy(report)
        item = report
        for key in path[:-1]:
            item = item[key]
        item[path[-1]] = value
        with self.assertRaises(AuditError, msg=str(path)):
            audit_report(report)

    def test_valid_failure_is_not_policy_pass(self):
        for bank in (False, True):
            output = audit_report(fixture(bank))
            self.assertTrue(output["report_internally_valid"])
            self.assertIn("NOT model acceptance", output["notice"])
            self.assertEqual(output["poses"][0]["final_valid_hold"], 0)

    def test_ever_success_may_fail_at_finish(self):
        row = audit_report(success_fixture(False))["poses"][0]
        self.assertEqual((row["ever_valid_hold"], row["final_valid_hold"]), (1, 0))

    def test_valid_success(self):
        self.assertEqual(audit_report(success_fixture())["poses"][0]["final_valid_hold"], 1)

    def test_metadata_mutations(self):
        for path, value in ((["protocol_version"], "legacy"), (["criterion"], CRITERION.replace("4 simultaneous", "2 simultaneous")),
                            (["acceptance_eligible"], 1), (["self_collisions_enabled"], "true"),
                            (["seed"], True), (["checkpoint_sha256"], "wrong"),
                            (["state_bank", "diagnostic_train_split_only"], True),
                            (["state_bank", "selected_state_ids"], [13]),
                            (["state_bank", "selected_unique_state_count"], 2)):
            self.mutate_rejected(fixture(), path, value)

    def test_count_and_rate_mutations(self):
        for key, value in (("successes", True), ("successes", 1.0), ("successes", -1), ("successes", 2),
                           ("final_valid_stands", 1), ("final_geometry_passes", 0), ("success_rate", .5),
                           ("settled_fallen_trials", 0), ("settled_fallen_final_valid_stands", 1),
                           ("settled_fallen_recovery_successes", 1), ("standing_starts_not_fallen_recovery", 1),
                           ("median_recovery_s", 1), ("settled_fallen_unique_state_count", 0)):
            self.mutate_rejected(fixture(), ["results", "side", key], value)

    def test_order_and_length(self):
        for collection in ("policy_start_state", "release_state_before_settling", "final_diagnostics"):
            self.mutate_rejected(fixture(), ["results", "side", collection], [])
            self.mutate_rejected(fixture(), ["results", "side", collection, 0, "trial"], True)
            self.mutate_rejected(fixture(), ["results", "side", collection, 0, "trial"], 1)

    def test_start_mutations(self):
        for key, value in (("geometry_ok", 1), ("settled", "true"), ("eligible_settled_fallen_recovery", False),
                           ("quiet_supported_window_s", 0), ("any_body_contact", False),
                           ("fallen_at_policy_start", False), ("standing_at_policy_start", True),
                           ("vertical_foot_contacts", 1), ("bank_state_id", 13), ("actual_pose_class", "back"),
                           ("root_linear_velocity_w_m_s", [1, 0, 0])):
            self.mutate_rejected(fixture(), ["results", "side", "policy_start_state", 0, key], value)

    def test_positive_hold_necessary_conditions(self):
        for key, value in (("geometry_ok", False), ("vertical_foot_contacts", 3), ("height_m", .3),
                           ("height_m", .55), ("gravity_error", .35), ("max_joint_offset_rad", .65),
                           ("feet_y_b", [-.17, .17, .17, -.17]), ("knees_y_b", [0, -.15, .15, -.15]),
                           ("base_contact", True)):
            self.mutate_rejected(success_fixture(), ["results", "side", "final_diagnostics", 0, key], value)

    def test_short_positive_hold_still_requires_valid_state(self):
        self.mutate_rejected(fixture(), ["results", "side", "final_diagnostics", 0, "stable_hold_s"], .02)

    def test_geometry_false_may_reflect_unreported_x(self):
        report = fixture()
        result = report["results"]["side"]
        result["final_geometry_passes"] = 0
        result["final_diagnostics"][0]["geometry_ok"] = False
        self.assertTrue(audit_report(report)["report_internally_valid"])

    def test_nonfinite_anywhere_and_boolean_numeric(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, "0"):
            self.mutate_rejected(fixture(), ["results", "side", "final_diagnostics", 0, "height_m"], value)
        report = fixture()
        report["extra"] = {"nested": [float("nan")]}
        with self.assertRaises(AuditError):
            audit_report(report)

    def test_bank_selection_mutations(self):
        for key, value in (("selected_state_ids", [True]), ("selected_unique_state_count", True),
                           ("available_unique_states", 0), ("sampling_with_replacement", True),
                           ("selected_actual_pose_classes", ["back"]), ("selected_requested_pose_classes", ["back"]),
                           ("actual_pose_class_counts", {"left": 0, "right": 1})):
            self.mutate_rejected(fixture(), ["results", "side", "state_bank_selection", key], value)

    def test_missing_unreported_final_quantities_are_allowed(self):
        report = success_fixture()
        final = report["results"]["side"]["final_diagnostics"][0]
        self.assertNotIn("base_contact", final)
        self.assertNotIn("feet_x_b", final)
        self.assertTrue(audit_report(report)["report_internally_valid"])

    def test_load_rejects_duplicate_keys(self):
        # Temporary files are test-only, not evaluator input/output mutations.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"results": {}, "results": {}}', encoding="utf-8")
            with self.assertRaises(AuditError):
                load_report(path)

    def test_cli_zero_means_valid_failure_and_never_changes_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            text = json.dumps(fixture())
            path.write_text(text, encoding="utf-8")
            with redirect_stdout(io.StringIO()) as output:
                code = main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn("NOT model acceptance", output.getvalue())
            self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_existing_reports_read_only(self):
        paths = [ROOT / f"evaluations/20260916-target128x2000-{arm}-{protocol}/model_1999.pt_recovery_metrics.json"
                 for arm in ("nominal", "current") for protocol in ("heldout", "angle30", "angle45")]
        paths += [ROOT / f"evaluations/20260916-smith-smoke-{protocol}/model_1.pt_recovery_metrics.json"
                  for protocol in ("heldout", "angle30")]
        paths += [ROOT / f"evaluations/20260916-target128x2000-video-review/nominal-{scenario}-{view}/model_1999.pt_recovery_metrics.json"
                  for scenario in ("heldout", "upright") for view in ("front", "oblique")]
        existing = [path for path in paths if path.is_file()]
        if not existing:
            self.skipTest("Optional real report fixtures are absent")
        groups = 0
        for path in existing:
            with self.subTest(report=path.name, directory=path.parent.name):
                before = path.read_bytes()
                groups += len(audit_report(load_report(path))["poses"])
                self.assertEqual(path.read_bytes(), before)
        print(f"Read-only real fixtures: {len(existing)} reports, {groups} pose groups")


if __name__ == "__main__":
    unittest.main()
