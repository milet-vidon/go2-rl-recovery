"""CPU-only tests; does not import Isaac Lab, torch, or launch a simulator."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import unittest


_PATH = Path(__file__).resolve().parents[1] / "src" / "go2_recovery" / "speed_gate_math.py"
_SPEC = importlib.util.spec_from_file_location("speed_gate_math_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
GateCurriculum = _MODULE.GateCurriculum
SpeedGateInputError = _MODULE.SpeedGateInputError


def window(cap=0.8, success=True, **changes):
    record = {
        "cap_at_sample": cap,
        "vx": cap,
        "mean_linear_reward": 0.9 if success else 0.8,
        "mean_angular_reward": 0.9,
        "duration_s": 3.5,
        "terminated": False,
        "standing": False,
    }
    record.update(changes)
    return record


class GateCurriculumTests(unittest.TestCase):
    def test_initial_and_empty(self):
        gate = GateCurriculum()
        before = gate.snapshot()
        self.assertEqual(before["cap"], 0.8)
        self.assertEqual(before["eligible_count"], 0)
        self.assertEqual(before["rejected_count"], 0)
        self.assertEqual(gate.observe([]), before)

    def test_nonoverlapping_window_waits_for_32(self):
        gate = GateCurriculum()
        state = gate.observe([window() for _ in range(31)])
        self.assertEqual(state["pending_count"], 31)
        self.assertEqual(state["decisions"], [])
        state = gate.observe([window()])
        self.assertEqual(state["cap"], 0.9)
        self.assertEqual(state["pending_count"], 0)
        self.assertEqual(state["pending_successes"], 0)
        self.assertEqual(state["decisions"][0]["count"], 32)

    def test_command_band_inclusive_boundaries(self):
        gate = GateCurriculum()
        state = gate.observe([
            window(vx=0.6), window(vx=0.8),
            window(vx=math.nextafter(0.6, -math.inf)),
            window(vx=math.nextafter(0.8, math.inf)),
        ])
        self.assertEqual(state["eligible_count"], 2)
        self.assertEqual(state["rejected_by_reason"]["outside_command_band"], 2)

    def test_duration_boundary(self):
        state = GateCurriculum().observe([
            window(duration_s=math.nextafter(3.5, -math.inf)),
            window(duration_s=3.5), window(duration_s=4),
            window(duration_s=-1),
        ])
        self.assertEqual(state["eligible_count"], 2)
        self.assertEqual(state["rejected_by_reason"]["short_duration"], 2)

    def test_both_rewards_strictly_above_threshold(self):
        state = GateCurriculum().observe([
            window(mean_linear_reward=0.8),
            window(mean_angular_reward=0.8),
            window(mean_linear_reward=math.nextafter(0.8, math.inf),
                   mean_angular_reward=math.nextafter(0.8, math.inf)),
            window(mean_linear_reward=1, mean_angular_reward=0.7),
        ])
        self.assertEqual(state["pending_count"], 4)
        self.assertEqual(state["pending_successes"], 1)

    def test_mixed_resets_standing_short_windows_do_not_count(self):
        gate = GateCurriculum()
        state = gate.observe([
            window(terminated=True), window(standing=True),
            window(duration_s=0), window(),
            window(terminated=True, standing=True), window(success=False),
        ])
        self.assertEqual(state["eligible_count"], 2)
        self.assertEqual(state["pending_successes"], 1)
        self.assertEqual(state["rejected_count"], 4)
        self.assertEqual(state["rejected_by_reason"]["standing"], 2)
        self.assertEqual(state["rejected_by_reason"]["terminated"], 1)

    def test_25_successes_do_not_advance(self):
        gate = GateCurriculum()
        state = gate.observe([window() for _ in range(25)] +
                             [window(success=False) for _ in range(7)])
        self.assertEqual(state["cap"], 0.8)
        self.assertFalse(state["decisions"][0]["threshold_met"])
        self.assertEqual(state["decisions"][0]["successes"], 25)
        self.assertEqual(state["pending_count"], 0)

    def test_26_successes_advance(self):
        state = GateCurriculum().observe([window() for _ in range(26)] +
                                         [window(success=False) for _ in range(6)])
        self.assertEqual(state["cap"], 0.9)
        self.assertEqual(state["decisions"][0]["success_rate"], 26 / 32)
        self.assertTrue(state["decisions"][0]["promoted"])

    def test_failure_then_success_does_not_reuse_windows(self):
        gate = GateCurriculum()
        gate.observe([window(success=False) for _ in range(32)])
        state = gate.observe([window() for _ in range(31)])
        self.assertEqual(state["cap"], 0.8)
        self.assertEqual(len(state["decisions"]), 1)
        state = gate.observe([window()])
        self.assertEqual([x["successes"] for x in state["decisions"]], [0, 32])
        self.assertEqual(state["eligible_count"], 64)

    def test_two_levels_and_no_advance_above_one(self):
        gate = GateCurriculum()
        for cap, lower in ((0.8, 0.6), (0.9, 0.7), (1.0, 0.8)):
            gate.observe([window(cap=cap, vx=lower) for _ in range(32)])
        state = gate.snapshot()
        self.assertEqual(state["cap"], 1.0)
        self.assertEqual([d["cap_after"] for d in state["decisions"]], [0.9, 1.0, 1.0])
        self.assertEqual([d["promoted"] for d in state["decisions"]], [True, True, False])
        self.assertEqual(state["eligible_count"], 96)

    def test_remaining_old_cap_is_discarded_after_promotion(self):
        gate = GateCurriculum()
        state = gate.observe([window() for _ in range(50)])
        self.assertEqual(state["cap"], 0.9)
        self.assertEqual(state["eligible_count"], 32)
        self.assertEqual(state["rejected_by_reason"]["different_cap"], 18)
        self.assertEqual(state["pending_count"], 0)
        state = gate.observe([window(cap=0.8, vx=0.8), window(cap=0.9, vx=0.8)])
        self.assertEqual(state["pending_count"], 1)

    def test_exact_cap_match_and_future_cap_rejected(self):
        state = GateCurriculum().observe([
            window(cap=math.nextafter(0.8, math.inf), vx=0.8),
            window(cap=0.9, vx=0.8), window(cap=0.7, vx=0.7),
        ])
        self.assertEqual(state["eligible_count"], 0)
        self.assertEqual(state["rejected_by_reason"]["different_cap"], 3)

    def test_all_numeric_fields_reject_nonfinite_and_wrong_types(self):
        for field in GateCurriculum.NUMERIC_FIELDS:
            for invalid in (math.nan, math.inf, -math.inf, True, "0.8", None, 10 ** 1000):
                with self.subTest(field=field, invalid=repr(invalid)[:30]):
                    gate = GateCurriculum()
                    before = gate.snapshot()
                    with self.assertRaises(SpeedGateInputError):
                        gate.observe([window(), window(**{field: invalid})])
                    self.assertEqual(gate.snapshot(), before)

    def test_bool_fields_require_actual_bools(self):
        for field in GateCurriculum.BOOL_FIELDS:
            for invalid in (0, 1, "false", None, math.nan):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(SpeedGateInputError):
                        GateCurriculum().observe([window(**{field: invalid})])

    def test_missing_fields_are_invalid(self):
        for field in GateCurriculum.NUMERIC_FIELDS + GateCurriculum.BOOL_FIELDS:
            record = window()
            del record[field]
            with self.subTest(field=field):
                with self.assertRaises(SpeedGateInputError):
                    GateCurriculum().observe([record])

    def test_batch_and_record_types(self):
        for invalid in (None, 1, "text", b"bytes", window(), [None], ["text"]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(SpeedGateInputError):
                    GateCurriculum().observe(invalid)

    def test_invalid_rejected_record_still_aborts_atomically(self):
        gate = GateCurriculum()
        gate.observe([window() for _ in range(31)])
        before = gate.snapshot()
        with self.assertRaises(SpeedGateInputError):
            gate.observe([window(), window(standing=True, mean_linear_reward=math.nan)])
        self.assertEqual(gate.snapshot(), before)

    def test_inputs_and_snapshots_are_independent(self):
        records = [window(extra={"leave": [1, 2]}) for _ in range(32)]
        before = deepcopy(records)
        gate = GateCurriculum()
        result = gate.observe(records)
        self.assertEqual(records, before)
        result["cap"] = 7
        result["decisions"][0]["count"] = 100
        result["rejected_by_reason"]["standing"] = 100
        snapshot = gate.snapshot()
        self.assertEqual(snapshot["cap"], 0.9)
        self.assertEqual(snapshot["decisions"][0]["count"], 32)
        self.assertEqual(snapshot["rejected_by_reason"]["standing"], 0)

    def test_batch_partition_does_not_change_decisions_or_counters(self):
        records = ([window(success=False) for _ in range(32)] +
                   [window(standing=True), window(terminated=True)] +
                   [window() for _ in range(40)] +
                   [window(cap=0.9) for _ in range(39)] +
                   [window(cap=1.0) for _ in range(35)])
        expected = GateCurriculum().observe(records)
        for chunk_size in (1, 2, 7, 17, 31, 32, 33, 64, len(records)):
            gate = GateCurriculum()
            for start in range(0, len(records), chunk_size):
                gate.observe(records[start:start + chunk_size])
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(gate.snapshot(), expected)
        self.assertEqual(expected["cap"], 1.0)
        self.assertEqual(expected["pending_count"], 3)
        self.assertEqual(expected["rejected_count"], 17)

    def test_generator_input_and_counter_conservation(self):
        records = [window(), window(success=False), window(standing=True),
                   window(terminated=True), window(cap=0.9)]
        state = GateCurriculum().observe(record for record in records)
        self.assertEqual(state["eligible_count"] + state["rejected_count"], len(records))
        self.assertEqual(sum(state["rejected_by_reason"].values()), state["rejected_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
