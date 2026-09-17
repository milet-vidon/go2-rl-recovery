"""CPU-only independent audit regressions, including actual measured first rows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import audit_supported_refinement as audit


class FullTraceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = audit.ROOT / "evaluations/20260917-supported-refinement-v1/off-upside_down"
        cls.report = audit.mirror.read_json(root / "model_1999.pt_recovery_metrics.json")
        with (root / "supported_refinement_full_trace.jsonl").open() as stream:
            cls.first = json.loads(next(stream))["rows"][0]
            cls.second = json.loads(next(stream))["rows"][0]
        cls.defaults = cls.report["handoff_controller"]["joint_limits"]["default_joint_positions_rad"]
        cls.limits = cls.report["handoff_controller"]["joint_limits"]["soft_joint_limits_rad"]

    def normal_row(self):
        return {"after": {"joint_positions_rad": self.defaults, "height_m": .32,
                          "root_linear_velocity_w_m_s": [0., 0., 0.],
                          "root_angular_velocity_w_rad_s": [0., 0., 0.],
                          "current_foot_forces_w_n": [[0., 0., 20.]] * 4,
                          "current_base_forces_w_n": [[0., 0., 0.]], "current_base_contact": False},
                "projected_gravity_b_after": [0., 0., -1.],
                "feet_b_after_m": [[.2, .15, -.3], [.2, -.15, -.3], [-.2, .15, -.3], [-.2, -.15, -.3]],
                "knees_b_after_m": [[.1, .1, 0.], [.1, -.1, 0.], [-.1, .1, 0.], [-.1, -.1, 0.]]}

    def test_normal_measured_stance(self):
        self.assertTrue(audit.strict_standing(self.normal_row(), self.defaults)["valid"])

    def test_crossed_foot_knee_large_joint_wrong_front_hind_rejected(self):
        for kind in ("foot", "knee", "joint", "front_hind"):
            row = self.normal_row()
            row = copy.deepcopy(row)
            if kind == "foot": row["feet_b_after_m"][0][1] = -.15
            if kind == "knee": row["knees_b_after_m"][0][1] = -.1
            if kind == "joint": row["after"]["joint_positions_rad"][0] += .7
            if kind == "front_hind": row["feet_b_after_m"][0][0] = -.2
            with self.subTest(kind=kind):
                self.assertFalse(audit.strict_standing(row, self.defaults)["valid"])

    def test_simultaneous_vertical_support_not_force_magnitude(self):
        for force in ([20., 0., 0.], [0., 0., 5.], [0., 0., -20.]):
            row = copy.deepcopy(self.normal_row())
            row["after"]["current_foot_forces_w_n"][0] = force
            self.assertFalse(audit.strict_standing(row, self.defaults)["valid"])

    def test_base_contact_invalidates_stance_and_inconsistent_flag_rejected(self):
        row = copy.deepcopy(self.normal_row())
        row["after"]["current_base_forces_w_n"] = [[2., 0., 0.]]
        with self.assertRaises(ValueError): audit.strict_standing(row, self.defaults)
        row["after"]["current_base_contact"] = True
        self.assertFalse(audit.strict_standing(row, self.defaults)["valid"])

    def test_gravity_height_motion_fail(self):
        for field, value in (("height_m", .1), ("root_linear_velocity_w_m_s", [.6, 0, 0]),
                             ("root_angular_velocity_w_rad_s", [0, 0, 1.1])):
            row = copy.deepcopy(self.normal_row())
            row["after"][field] = value
            self.assertFalse(audit.strict_standing(row, self.defaults)["valid"])
        row = self.normal_row()
        row["projected_gravity_b_after"] = [0., 0., 1.]
        self.assertFalse(audit.strict_standing(row, self.defaults)["valid"])

    def test_gate_exact150_completed_and_no_unlatch(self):
        count, latched = 0, False
        for step in range(150):
            self.assertFalse(latched)
            count, latched = audit.gate_after(True, True, count, latched)
        self.assertEqual((count, latched), (150, True))
        self.assertEqual(audit.gate_after(False, True, count, latched), (0, True))
        self.assertEqual(audit.gate_after(True, False, 149, False), (0, False))

    def test_gate_interruption_resets_count_not_physics(self):
        self.assertEqual(audit.gate_after(False, True, 149, False), (0, False))
        with self.assertRaises(ValueError): audit.gate_after(True, True, 150, False)

    def test_actual_two_interval_action_target_history(self):
        audit.issued_and_history(self.first, None, self.defaults, self.limits, False)
        audit.issued_and_history(self.second, self.first, self.defaults, self.limits, False)

    def test_actual_action_target_history_mutations_rejected(self):
        for field in ("actual_raw_action", "previous_raw_action", "actual_executed_joint_target_rad"):
            row = copy.deepcopy(self.second)
            row[field][0] += .01
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.issued_and_history(row, self.first, self.defaults, self.limits, False)

    def test_prefix_has_exact_full_interval_and_real_edge_state(self):
        self.assertEqual(audit.prefix_exact(self.first, self.first, None), "full_interval")
        self.assertEqual(audit.prefix_exact(self.first, self.first, 0), "edge_pre_action")
        changed = copy.deepcopy(self.first)
        changed["before"]["height_m"] += .01
        with self.assertRaises(ValueError): audit.prefix_exact(changed, self.first, 0)

    def test_stream_rejects_missing_extra_reordered_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT.parent / "tmp", prefix="refine-audit-") as temporary:
            path = Path(temporary) / "fixture.jsonl"
            for payload in ([{"pose": "upside_down", "step": 1, "rows": [self.first]}],
                            [{"pose": "upside_down", "step": 0, "rows": []}], []):
                path.write_text("".join(json.dumps(x) + "\n" for x in payload), encoding="utf-8")
                with self.assertRaises(ValueError): list(audit.stream_intervals(path, "upside_down", n=1, steps=1))

    def test_all_source_pins_are_actual64char_hashes(self):
        for name, expected in audit.PINS.items():
            self.assertEqual(len(expected), 64)
            self.assertEqual(audit.sha(audit.ROOT / "scripts" / name), expected)

    def test_on_cannot_self_report_changed_defaults_or_limits(self):
        for field in ("default_joint_positions_rad", "soft_joint_limits_rad"):
            changed = copy.deepcopy(self.report)
            values = changed["handoff_controller"]["joint_limits"][field]
            if isinstance(values[0], list): values[0][0] -= .1
            else: values[0] += .1
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.baseline_protocol(self.report, changed, "upside_down")
        changed = copy.deepcopy(self.report)
        changed["handoff_controller"]["invented_gate_or_reset"] = True
        with self.assertRaises(ValueError):
            audit.baseline_protocol(self.report, changed, "upside_down")

    def test_actual_ordinary_upright_has_no_bank_not_a_fallen_reset(self):
        path = audit.ROOT / "evaluations/20260917-supported-refinement-v1/off-upright/model_1999.pt_recovery_metrics.json"
        upright = audit.mirror.read_json(path)
        self.assertNotIn("state_bank", upright)
        audit.baseline_protocol(upright, upright, "upright")
        changed = copy.deepcopy(upright)
        changed["state_bank"] = self.report["state_bank"]
        with self.assertRaises(ValueError): audit.baseline_protocol(upright, changed, "upright")


if __name__ == "__main__":
    unittest.main(verbosity=2)
