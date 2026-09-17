"""Synthetic checks only; never substitute these for a physical rollout."""
import copy
import unittest

from continuous_flow_motion_metrics import analyze, measured_stance, percentile, f32, FEET, LEGS, JOINTS


def example():
    rows = []
    for phase, count in (("preparation", 50), ("recovery", 150), ("move", 400), ("stop", 300)):
        for interval in range(1, count + 1):
            moving = phase == "move"
            row = {"step": len(rows), "time_s": (len(rows) + 1)*.02, "phase": phase,
                   "phase_interval": interval, "actor_role": "nominal_pd" if phase == "preparation" else
                   "stand" if phase == "recovery" else "locomotion", "cmd_x": .5 if moving else 0.,
                   "cmd_y": 0., "cmd_yaw": 0., "height": .32, "gravity_b": [0., 0., -1.],
                   "linear_speed_3d": .5 if moving else 0., "angular_speed_3d": 0.,
                   "vx_b": .5 if moving else 0., "vy_b": 0., "wz": 0.,
                   "speed": .5 if moving else 0., "yaw_speed": 0., "roll_deg": 0., "pitch_deg": 0.,
                   "stance_geometry_ok": 1, "current_base_force_n": 0., "current_base_contact": 0,
                   "current_vertical_foot_contacts": 2 if moving else 4, "base_contact": 0, "done": 0,
                   "strict_valid_after": not moving, "quiet_for_handoff": not moving}
            for index, (leg, foot) in enumerate(zip(LEGS, FEET)):
                side = 1. if index % 2 == 0 else -1.
                contact = not moving or (index in (0, 3)) == ((interval // 10) % 2 == 0)
                row.update({foot + "_x_b": .2 if index < 2 else -.2, foot + "_y_b": side * .16,
                            leg + "_knee_y_b": side * .1, foot + "_fz": 20. if contact else 0.,
                            foot + "_vertical_contact": int(contact), foot + "_contact": int(contact),
                            foot + "_speed_xy": .01 if contact else .7,
                            foot + "_z_w": .025 if contact else .07})
            for joint in JOINTS:
                row[joint + "_offset_rad"] = 0.
            rows.append(row)
    return rows


class MotionTests(unittest.TestCase):
    def test_quiet_handoff_measurements_are_independent_of_strict_geometry(self):
        rows = example()
        rows[100].update(linear_speed_3d=.07, speed=.07, quiet_for_handoff=False)
        result = analyze(rows)
        self.assertTrue(result["checks"]["recovery_last150_strict"])
        self.assertFalse(result["checks"]["recovery_last150_quiet_ready"])
        self.assertFalse(result["motion_checks_passed"])

    def test_forged_quiet_and_exact_threshold_rejected(self):
        for key, value in (("linear_speed_3d", .06), ("angular_speed_3d", .15)):
            row = example()[100]
            row[key] = value
            with self.assertRaisesRegex(ValueError, "Quiet readiness"):
                measured_stance(row)
            row["quiet_for_handoff"] = False
            self.assertTrue(measured_stance(row)[3])

    def test_synthetic_complete_and_no_four_contact_motion_gate(self):
        report = analyze(example())
        self.assertTrue(report["motion_checks_passed"])
        self.assertEqual(report["settled_phase_stats"]["move"]["samples"], 350)
        self.assertEqual(report["settled_phase_stats"]["stop"]["samples"], 250)
        self.assertFalse(report["promotion_performed"])

    def test_no_reset_masked_in_preparation(self):
        rows = example()
        rows[1]["done"] = 1
        self.assertFalse(analyze(rows)["checks"]["recorded_no_done_whole_episode"])

    def test_initial_base_contact_allowed(self):
        rows = example()
        rows[1].update(base_contact=1, current_base_force_n=20., current_base_contact=1, strict_valid_after=False)
        self.assertTrue(analyze(rows)["motion_checks_passed"])

    def test_contact_union_cannot_fake_four_current_feet(self):
        row = example()[-1]
        row["FL_foot_fz"] = 0.
        with self.assertRaisesRegex(ValueError, "vertical contact"):
            measured_stance(row)

    def test_crossed_front_foot_cannot_keep_geometry_label(self):
        row = example()[-1]
        row["FL_foot_y_b"] = -.15
        with self.assertRaisesRegex(ValueError, "Geometry label"):
            measured_stance(row)

    def test_actual_late_bad_stance_fails_without_hidden_tail_cut(self):
        rows = example()
        rows[-1].update(FL_foot_y_b=-.15, stance_geometry_ok=0, strict_valid_after=False)
        result = analyze(rows)
        self.assertFalse(result["checks"]["stop_last150_strict"])
        self.assertFalse(result["checks"]["normal_stance_geometry_at_rest"])

    def test_tracking_regression_is_failure_not_error(self):
        rows = example()
        for row in rows:
            if row["phase"] == "move":
                row["vx_b"] = .3
        self.assertFalse(analyze(rows)["checks"]["walk_tracking"])

    def test_command_counter_and_actor_corruption_rejected(self):
        for key, value in (("cmd_x", .8), ("phase_interval", 1), ("actor_role", "stand"), ("step", 201)):
            rows = example()
            rows[220][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze(rows)

    def test_missing_and_extra_samples_rejected(self):
        for rows in (example()[:-1], example()[1:], example() + [copy.deepcopy(example()[-1])]):
            with self.assertRaises(ValueError):
                analyze(rows)

    def test_empty_contact_slip_not_nan_or_success(self):
        rows = example()
        for row in rows:
            if row["phase"] == "move":
                for foot in FEET:
                    row[foot + "_contact"] = 0
                    row[foot + "_vertical_contact"] = 0
                    row[foot + "_fz"] = 0.
                row["current_vertical_foot_contacts"] = 0
        result = analyze(rows)
        self.assertIsNone(result["settled_phase_stats"]["move"]["contact_slip_mean"])
        self.assertFalse(result["checks"]["limited_slip"])

    def test_percentile_linear(self):
        self.assertAlmostEqual(percentile([0., 10.]), 9.5)
        self.assertEqual(percentile([1.] * 150), 1.)
        with self.assertRaises(ValueError):
            percentile([float("nan")])

    def test_float32_joint_and_height_strict_boundaries(self):
        row = example()[-1]
        row.update(FL_hip_joint_offset_rad=f32(.65), stance_geometry_ok=0, strict_valid_after=False)
        self.assertFalse(measured_stance(row)[0])
        row = example()[-1]
        row.update(height=f32(.30), strict_valid_after=False)
        self.assertFalse(measured_stance(row)[3])

    def test_current_base_cannot_hide_behind_false_history(self):
        row = example()[250]
        row.update(current_base_force_n=20., current_base_contact=1, base_contact=0)
        with self.assertRaisesRegex(ValueError, "force history"):
            measured_stance(row)


if __name__ == "__main__":
    unittest.main()
