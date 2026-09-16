"""Simulator-free regressions for the stand/stop geometry screening additions."""

import ast
import importlib.util
from pathlib import Path
import unittest

import torch


_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("recovery_math", _ROOT / "src/go2_recovery/recovery_math.py")
_MATH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MATH)
_TREE = ast.parse((_ROOT / "scripts/evaluate_go2_stand_walk_stop.py").read_text(encoding="utf-8"))
_FUNCTIONS = [n for n in _TREE.body if isinstance(n, ast.FunctionDef)
              and n.name in ("_stance_summary", "_rest_stance_acceptance")]
_SCOPE = {}
exec(compile(ast.Module(body=_FUNCTIONS, type_ignores=[]), "<stance_screen>", "exec"), _SCOPE)
summarize = _SCOPE["_stance_summary"]
accept = _SCOPE["_rest_stance_acceptance"]


def normal_row():
    return {"stance_geometry_ok": 1, "current_vertical_foot_contacts": 4,
            "current_base_contact": 0, "max_joint_offset_rad": 0.1,
            # Historical force-norm/history-union support deliberately stays true.
            "foot_contacts": 4, "base_contact": 0}


def phase_summaries(stand_rows, stop_rows=None):
    return {"stand": summarize(stand_rows),
            "stop": summarize(stand_rows if stop_rows is None else stop_rows),
            # Walking may lift feet or leave the static joint envelope normally.
            "walk": summarize([dict(normal_row(), stance_geometry_ok=0,
                                     current_vertical_foot_contacts=2)])}


class RestStanceTests(unittest.TestCase):
    def test_normal_rest_passes_and_walk_is_not_constrained(self):
        self.assertTrue(all(accept(phase_summaries([normal_row()])).values()))

    def test_crossing_knee_joint_and_fore_hind_errors_fail(self):
        feet = torch.tensor([[[.20, .14, -.31], [.20, -.14, -.31],
                              [-.20, .14, -.31], [-.20, -.14, -.31]]])
        for defect in ("crossed_feet", "crossed_knee", "one_joint", "wrong_end"):
            with self.subTest(defect=defect):
                altered_feet, knees, joints = feet.clone(), feet.clone(), torch.zeros(1, 12)
                if defect == "crossed_feet":
                    altered_feet[0, :2, 1] *= -1
                elif defect == "crossed_knee":
                    knees[0, 0, 1] = -.08
                elif defect == "one_joint":
                    joints[0, 0] = .9
                else:
                    altered_feet[0, 1, 0] = -.1
                rows = [normal_row() for _ in range(100)]
                rows[50]["stance_geometry_ok"] = int(_MATH.normal_stance_geometry(altered_feet, knees, joints)[0])
                # A single invalid rest frame must fail even with four contacts.
                result = accept(phase_summaries(rows))
                self.assertFalse(result["normal_stance_geometry_at_rest"])
                self.assertTrue(result["four_vertical_contacts_at_rest"])

    def test_missing_current_vertical_support_cannot_use_legacy_contacts(self):
        rows = [dict(normal_row(), current_vertical_foot_contacts=3) for _ in range(100)]
        result = accept(phase_summaries(rows))
        self.assertFalse(result["four_vertical_contacts_at_rest"])
        self.assertFalse(result["geometry_and_support_at_rest"])

    def test_support_fraction_is_strictly_above_95_percent(self):
        rows = [normal_row() for _ in range(100)]
        for row in rows[:5]:
            row["current_vertical_foot_contacts"] = 3
        self.assertFalse(accept(phase_summaries(rows))["four_vertical_contacts_at_rest"])
        rows[4]["current_vertical_foot_contacts"] = 4
        self.assertTrue(accept(phase_summaries(rows))["four_vertical_contacts_at_rest"])

    def test_single_current_base_contact_fails_only_stop_too(self):
        stop_rows = [normal_row() for _ in range(100)]
        stop_rows[50]["current_base_contact"] = 1
        result = accept(phase_summaries([normal_row()], stop_rows))
        self.assertFalse(result["no_current_base_contact_at_rest"])

    def test_summary_records_auditable_counts_and_maximum_joint(self):
        rows = [normal_row(), dict(normal_row(), max_joint_offset_rad=.6)]
        stats = summarize(rows)
        self.assertEqual(stats["sample_count"], 2)
        self.assertEqual(stats["stance_geometry_invalid_samples"], 0)
        self.assertEqual(stats["max_joint_offset_rad"], .6)

    def test_empty_window_is_not_accepted(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
