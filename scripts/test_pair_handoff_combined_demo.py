"""Pure in-memory pairing contract tests; no video or simulator dependencies."""
import copy
import sys
import unittest

import pair_handoff_combined_demo as pairer


def fixture(pose="upright", final=20):
    trace = "model_1999.pt_handoff_combined_trace.json"
    digest = "a" * 64
    start = [{"trial": i} for i in range(20)]
    report = {
        "video_view": "front", "protocol_version": pairer.entry.PROTOCOL,
        "controller_type": pairer.entry.CONTROLLER, "diagnostic_trial": 0, "seed": 20260918,
        "acceptance_eligible": False, "single_policy_acceptance_eligible": False, "training_collection_eligible": False,
        "checkpoint_sha256": pairer.entry.ROLL_SHA,
        "handoff_controller": {"roll_sha256": pairer.entry.ROLL_SHA, "stand_sha256": pairer.entry.STAND_SHA},
        "results": {pose: {"trials": 20, "successes": final, "final_valid_stands": final, "final_geometry_passes": 20,
                            "policy_start_state": copy.deepcopy(start), "release_state_before_settling": copy.deepcopy(start),
                            "final_diagnostics": copy.deepcopy(start)}},
        "transition_experiment": {"mode": "hard", "ramp_seconds": 0.0, "ramp_steps": 0, "step_dt": .02,
                                  "horizon_s": 8.0, "hold_s": 3.0, "min_contacts": 4, "policy_control_steps": 550},
        "startup_experiment": {"mode": "supported", "mirror_enabled": True, "ramp_enabled": False, "retry_enabled": False,
                               "generated_source_file": "combined_generated.py", "generated_source_sha256": digest},
        "mirror_experiment": {"mode": "initial_right", "startup_selection_changed": True, "ramp_enabled": False, "retry_enabled": False},
        "combined_experiment": {"startup_mode": "supported", "mirror_mode": "initial_right",
                                "generated_source_file": "combined_generated.py", "generated_source_sha256": digest},
    }
    for group, (name, hashed) in pairer.TRACE_GROUPS.items():
        report[group].update({name: trace, hashed: digest})
    other = copy.deepcopy(report)
    other["video_view"] = "oblique"
    return report, other


class PairTests(unittest.TestCase):
    def test_three_poses_and_failed_outcomes_are_preserved(self):
        for pose in pairer.POSES:
            for final in (0, 9, 20):
                self.assertEqual(pairer.validate_report_pair(*fixture(pose, final)), pose)

    def test_dependency_import_is_light(self):
        for name in ("cv2", "imageio_ffmpeg", "torch", "isaaclab.app"):
            self.assertNotIn(name, sys.modules)

    def test_strict_type_not_bool_numeric_coercion(self):
        for value in (True, 1.0):
            with self.assertRaises(ValueError):
                pairer.strict_equal(1, value)
        with self.assertRaises(ValueError):
            pairer.strict_equal(float("nan"), float("nan"))

    def test_different_physical_start_rejected(self):
        front, oblique = fixture()
        oblique["results"]["upright"]["policy_start_state"][0]["height_m"] = .4
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(front, oblique)

    def test_wrong_view_trial_population_and_seed_rejected(self):
        for key, value in (("diagnostic_trial", 1), ("seed", 7), ("video_view", "side")):
            front, oblique = fixture()
            front[key] = value
            oblique[key] = value
            with self.assertRaises(ValueError):
                pairer.validate_report_pair(front, oblique)
        front, oblique = fixture()
        for report in (front, oblique):
            report["results"]["upright"]["trials"] = 1
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(front, oblique)

    def test_multiple_pose_report_rejected(self):
        front, oblique = fixture()
        for report in (front, oblique):
            report["results"]["side"] = copy.deepcopy(report["results"]["upright"])
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(front, oblique)

    def test_wrong_factor_actor_acceptance_or_trace_rejected(self):
        changes = (("startup_experiment", "mode", "off"),
                   ("mirror_experiment", "mode", "off"),
                   ("mirror_experiment", "startup_selection_changed", False),
                   ("transition_experiment", "ramp_seconds", .2),
                   ("handoff_controller", "stand_sha256", "b" * 64),
                   ("combined_experiment", "trace_sha256", "b" * 64))
        for group, key, value in changes:
            front, oblique = fixture()
            for report in (front, oblique):
                report[group][key] = value
            with self.assertRaises(ValueError):
                pairer.validate_report_pair(front, oblique)
        front, oblique = fixture()
        front["acceptance_eligible"] = oblique["acceptance_eligible"] = True
        with self.assertRaises(ValueError):
            pairer.validate_report_pair(front, oblique)

    def test_complete_video_only(self):
        full = {"frames": 276, "fps": 25.0, "width": 960, "height": 540}
        pairer.validate_video_pair(full, copy.deepcopy(full))
        for key, value in (("frames", 275), ("fps", 50.0), ("width", 0), ("height", 541)):
            bad = {**full, key: value}
            with self.assertRaises(ValueError):
                pairer.validate_video_pair(bad, copy.deepcopy(bad))
        with self.assertRaises(ValueError):
            pairer.validate_video_pair(full, {**full, "frames": 275})


if __name__ == "__main__":
    unittest.main(verbosity=2)
