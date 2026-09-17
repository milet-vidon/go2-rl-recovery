"""Pure tests of the frozen evaluator's extra pre-action video CSV boundary."""
import copy
import unittest
from verify_handoff_combined_replays import verify_csv_rows


class ReplayBoundaryTests(unittest.TestCase):
    def test_exact_boundary_and_mutations(self):
        rows = [{"time_s": str((i + 1) * .02), "action": str(i)} for i in range(550)]
        first = {"policy_phase": "before policy", "pose": "side", "trial": "0", "time_s": "0.0",
                 "height": "0.15", "gravity_error": "1.5"}
        video = [first, *copy.deepcopy(rows)]
        start = {"height_m": .15, "gravity_error": 1.5}
        verify_csv_rows(rows, video, "side", start)
        for index, key, value in ((0, "height", "0.16"), (0, "time_s", ".02"),
                                  (1, "action", "wrong"), (550, "action", "wrong")):
            changed = copy.deepcopy(video)
            changed[index][key] = value
            with self.assertRaises(ValueError):
                verify_csv_rows(rows, changed, "side", start)
        with self.assertRaises(ValueError):
            verify_csv_rows(rows, video[:-1], "side", start)


if __name__ == "__main__":
    unittest.main()
