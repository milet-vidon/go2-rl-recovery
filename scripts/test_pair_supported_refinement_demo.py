"""CPU-only contract tests; in-memory video-report views are NOT real recordings."""

import copy
import sys
import unittest
from unittest.mock import patch

import pair_supported_refinement_demo as pairer


class RefinementPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Existing NONVIDEO evidence only. Future video paths/files are not fabricated.
        path = pairer.ROOT / "evaluations/20260917-supported-refinement-v1/supported-side/model_1999.pt_recovery_metrics.json"
        cls.reference = pairer.read_json(path)

    def fixture(self):
        reference = copy.deepcopy(self.reference)
        front, oblique = copy.deepcopy(reference), copy.deepcopy(reference)
        front["video_view"], oblique["video_view"] = "front", "oblique"
        return front, oblique, reference

    def test_actual_nonvideo_contract_with_inmemory_views(self):
        self.assertEqual(pairer.validate_report_pair(*self.fixture()), "side")

    def test_no_video_torch_or_sim_imported(self):
        for name in ("cv2", "imageio_ffmpeg", "torch", "isaaclab.app"):
            self.assertNotIn(name, sys.modules)

    def test_failed_outcomes_are_not_filtered(self):
        reports = self.fixture()
        for report in reports:
            for key in ("successes", "final_valid_stands", "final_geometry_passes"):
                report["results"]["side"][key] = 0
            report["refinement_experiment"]["poses"]["side"]["final_refinement_valid_holds"] = 0
        self.assertEqual(pairer.validate_report_pair(*reports), "side")

    def test_refuse_reference_video_wrong_view_or_population(self):
        for index, field, value in ((2, "video_view", "front"), (0, "video_view", "side"),
                                     (0, "seed", 1), (0, "diagnostic_trial", 1)):
            reports = self.fixture()
            reports[index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                pairer.validate_report_pair(*reports)
        reports = self.fixture()
        for report in reports:
            report["results"]["side"]["trials"] = 1
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(*reports)

    def test_all20_starts_finals_and_controller_must_match_reference(self):
        for collection in ("policy_start_state", "release_state_before_settling", "final_diagnostics"):
            reports = self.fixture()
            for report in reports[:2]:
                report["results"]["side"][collection][19]["height_m"] += .01
            with self.subTest(collection=collection), self.assertRaises(ValueError):
                pairer.validate_report_pair(*reports)
        reports = self.fixture()
        for report in reports[:2]:
            report["handoff_controller"]["joint_limits"]["soft_joint_limits_rad"][0][0] -= .01
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(*reports)

    def test_artifact_provenance_is_not_blanket_excluded(self):
        for group, key in (("refinement_experiment", "trace_sha256"),
                           ("startup_experiment", "generated_source_sha256")):
            reports = self.fixture()
            reports[0][group][key] = "f" * 64
            with self.subTest(group=group), self.assertRaises(ValueError):
                pairer.validate_report_pair(*reports)

    def test_legacy_protocol_off_mode_wrong_three_actors_rejected(self):
        for group, key, value in ((None, "protocol_version", pairer.original.entry.PROTOCOL),
                                  ("refinement_experiment", "mode", "off"),
                                  ("handoff_controller", "roll_sha256", "f" * 64),
                                  ("handoff_controller", "stand_sha256", "f" * 64)):
            reports = self.fixture()
            for report in reports:
                (report if group is None else report[group])[key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                pairer.validate_report_pair(*reports)
        reports = self.fixture()
        for report in reports:
            report["refinement_experiment"]["refinement_actor_training"]["checkpoint_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(*reports)

    def test_no_gate_relaxation_reset_ramp_or_promotion(self):
        for group, key, value in (("refinement_experiment", "qualifying_completed_intervals", 149),
                                  ("refinement_experiment", "switch_time_physical_or_history_reset", True),
                                  ("refinement_experiment", "extra_pd_ramp_or_retry", True),
                                  ("transition_experiment", "min_contacts", 2),
                                  ("transition_experiment", "ramp_seconds", .2),
                                  (None, "acceptance_eligible", True)):
            reports = self.fixture()
            for report in reports:
                (report if group is None else report[group])[key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                pairer.validate_report_pair(*reports)

    def test_strict_types_and_no_missing_extra_report_fields(self):
        for value in (True, 1.0):
            with self.assertRaises(ValueError):
                pairer.strict_equal(1, value)
        for change in ("extra", "missing"):
            reports = self.fixture()
            if change == "extra":
                reports[0]["hidden_change"] = True
            else:
                del reports[0]["self_collisions_enabled"]
            with self.assertRaises(ValueError):
                pairer.validate_report_pair(*reports)

    def test_full_video_only_no_equal_truncation(self):
        full = {"frames": 276, "fps": 25.0, "width": 960, "height": 540}
        pairer.validate_video_pair(full, copy.deepcopy(full))
        for key, value in (("frames", 275), ("fps", 50.), ("width", 0), ("height", 541)):
            changed = {**full, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                pairer.validate_video_pair(changed, copy.deepcopy(changed))

    def test_frozen_source_pins_current(self):
        pairer.check_source_pins()

    def test_reference_audit_is_recomputed_not_trusted_from_label(self):
        path = pairer.ROOT / "evaluations/20260917-supported-refinement-v1/supported-side/full_trajectory_audit_v1.json"
        saved = pairer.read_json(path)
        # A mock isolates this transport/validation regression from the costly
        # independent full audit. Production verify_reference always reruns it.
        with patch.object(pairer.audit, "audit", return_value=copy.deepcopy(saved)) as run:
            reference, report, _ = pairer.verify_reference(path)
            self.assertEqual(str(reference), saved["report"])
            self.assertIsNone(report["video_view"])
            run.assert_called_once_with(pairer.Path(saved["report"]), pairer.Path(saved["off_control"]["report"]))
        changed = copy.deepcopy(saved)
        changed["measured"]["independent_final_strict_holds"] -= 1
        with patch.object(pairer.audit, "audit", return_value=changed), self.assertRaises(ValueError):
            pairer.verify_reference(path)

    def test_obsolete_changed_source_audit_is_not_authoritative(self):
        for version in (1, 2):
            path = pairer.ROOT / f"evaluations/20260917-supported-refinement-v1/supported-upside_down/full_trajectory_audit_v{version}.json"
            with patch.object(pairer.audit, "audit") as run, self.assertRaises(ValueError):
                pairer.verify_reference(path)
            run.assert_not_called()

    def test_actual_control_evidence_exact_and_any_digest_change_rejected(self):
        directory = pairer.ROOT / "evaluations/20260917-supported-refinement-v1/supported-side"
        source = {}
        for key, name in (("trace_csv", self.reference["trace_csv"]),
                          ("full_trace", self.reference["refinement_experiment"]["trace_file"]),
                          ("neighborhood", self.reference["startup_experiment"]["transition_trace_file"])):
            source[key] = str(directory / name)
            source[key + "_sha256"] = pairer.sha(directory / name)
        pairer.compare_alltrial_interfaces(source, copy.deepcopy(source))
        self.assertEqual(len(pairer.csv_rows(pairer.Path(source["trace_csv"]))), 550)
        with self.assertRaises(ValueError):
            pairer.csv_rows(pairer.Path(source["trace_csv"]), video=True)
        for key in ("full_trace", "neighborhood"):
            changed = copy.deepcopy(source)
            changed[key + "_sha256"] = "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                pairer.compare_alltrial_interfaces(changed, source)

    def test_actual_frozen_recorder_preaction_boundary_and_mutations(self):
        # Real OLD combined recordings exercise the same frozen recorder's CSV
        # boundary. They are not presented as new refinement recordings.
        directory = pairer.ROOT / "evaluations/20260917-handoff-combined-v2/s1m1-side"
        report = pairer.read_json(directory / "model_1999.pt_recovery_metrics.json")
        controls = pairer.csv_rows(directory / report["trace_csv"])
        video_path = pairer.ROOT / "evaluations/20260917-combined20-side-front/model_1999.pt_recovery_trace.csv"
        video = pairer.csv_rows(video_path, video=True)
        pairer.verify_control_csv_rows(controls, video, report, "side")
        changes = ((0, "height", ".9"), (0, "time_s", ".02"), (0, "feet_contact", "4"),
                   (0, "FL_hip_joint", ".9"), (0, "FL_hip_joint_offset", ".9"),
                   (0, "FL_foot_y_b", "-.15"), (0, "handoff_active", "True"),
                   (0, "policy_phase", "roll"), (1, "height", ".9"), (550, "height", ".9"))
        for index, key, value in changes:
            changed = copy.deepcopy(video)
            changed[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                pairer.verify_control_csv_rows(controls, changed, report, "side")
        with self.assertRaises(ValueError):
            pairer.verify_control_csv_rows(controls, video[1:], report, "side")

    def test_front_oblique_include_exact_extra_initial_row(self):
        sources = []
        for view in ("front", "oblique"):
            path = pairer.ROOT / f"evaluations/20260917-combined20-side-{view}/model_1999.pt_recovery_trace.csv"
            sources.append({"trace_csv": str(path), "trace_csv_sha256": pairer.sha(path)})
        pairer.compare_video_csvs(*sources)
        sources[1]["trace_csv_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            pairer.compare_video_csvs(*sources)


if __name__ == "__main__":
    unittest.main(verbosity=2)
