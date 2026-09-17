import unittest
from pathlib import Path
import compare_balanced_gait_screens as analysis


class GaitComparisonTests(unittest.TestCase):
    def test_complete_cycles_exclude_initial_and_terminal_partial(self):
        flags=([False]*6+[True]*6)*3
        records=analysis.cycles(flags,[i*.02 for i in range(len(flags))])
        self.assertEqual(len(records),2)
        for row in records:
            self.assertAlmostEqual(row["period_s"],.24)
            self.assertAlmostEqual(row["stance_s"],.12)
            self.assertAlmostEqual(row["swing_s"],.12)

    def test_single_frame_chatter_does_not_create_cycle(self):
        flags=[False,False,True,False,True,False,False,True,False,False]
        self.assertEqual(analysis.cycles(flags,[i*.02 for i in range(len(flags))]),[])

    def test_constant_stance_or_flight_no_complete_cycles(self):
        for flag in (True,False):
            self.assertEqual(analysis.cycles([flag]*350,[i*.02 for i in range(350)]),[])

    def test_actual_4046_force_duty_not_misread_boolean_text(self):
        path=Path(__file__).resolve().parents[1]/"evaluations/20260917-commonphysics4046-first/recovery-normal-seed20260909/model_4046_stand_walk_stop.csv"
        result=analysis.measure(path)
        self.assertEqual(result["samples"],350)
        self.assertAlmostEqual(result["duty"]["FL"],291/350)
        self.assertAlmostEqual(result["diagonal_duty_gap"],.53)
        self.assertEqual(result["csv_sha256"],"7cc6d23f70b935b690e6c6ff5b8200ffd08a09fa46c957435307acdc07e3ee5f")


if __name__ == "__main__":
    unittest.main(verbosity=2)
