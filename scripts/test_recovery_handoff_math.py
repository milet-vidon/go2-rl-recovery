"""CPU tests for the diagnostic handoff gate; no simulator imports."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).parents[1] / "src/go2_recovery/recovery_handoff_math.py"
SPEC = importlib.util.spec_from_file_location("recovery_handoff_math", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
update_handoff_gate = MODULE.update_handoff_gate


def state(n=1, dtype=torch.float64, counter_dtype=torch.int32):
    gravity = torch.zeros(n, 3, dtype=dtype)
    gravity[:, 2] = -1.0
    return (gravity, torch.zeros_like(gravity),
            torch.zeros(n, dtype=counter_dtype), torch.zeros(n, dtype=torch.bool))


class HandoffGateTests(unittest.TestCase):
    def test_public_api_supports_positional_and_named_step_dt(self):
        gravity, angular, counter, switched = state()
        counter.fill_(9)
        positional = update_handoff_gate(gravity, angular, counter, switched, .02)
        named = update_handoff_gate(gravity=gravity, angular_velocity=angular,
                                    previous_steps=counter, switched=switched, step_dt=.02)
        for first, second in zip(positional, named):
            self.assertTrue(torch.equal(first, second))

    def test_nine_steps_do_not_trigger_tenth_does_once(self):
        gravity, angular, counter, switched = state()
        for step in range(1, 13):
            counter, switched, edge = update_handoff_gate(gravity, angular, counter, switched)
            self.assertEqual(counter.item(), min(step, 10))
            self.assertEqual(switched.item(), step >= 10)
            self.assertEqual(edge.item(), step == 10)

    def test_leaving_gate_clears_hold_and_requires_ten_new_samples(self):
        gravity, angular, counter, switched = state()
        counter.fill_(9)
        angular[0, 0] = 1.0
        counter, switched, edge = update_handoff_gate(gravity, angular, counter, switched)
        self.assertEqual(counter.item(), 0)
        self.assertFalse(switched.item() or edge.item())
        angular.zero_()
        for _ in range(9):
            counter, switched, edge = update_handoff_gate(gravity, angular, counter, switched)
            self.assertFalse(switched.item() or edge.item())
        self.assertTrue(update_handoff_gate(gravity, angular, counter, switched)[2].item())

    def test_environments_have_independent_counters_and_latches(self):
        gravity, angular, counter, switched = state(4)
        counter[:] = torch.tensor([9, 8, 9, 9])
        angular[2, 1] = 1.01
        switched[3] = True
        counts, latches, edges = update_handoff_gate(gravity, angular, counter, switched)
        self.assertEqual(counts.tolist(), [10, 9, 0, 10])
        self.assertEqual(latches.tolist(), [True, False, False, True])
        self.assertEqual(edges.tolist(), [True, False, False, False])

    def test_latch_never_switches_back_even_with_invalid_state(self):
        gravity, angular, counter, switched = state(3)
        counter.fill_(10)
        switched.fill_(True)
        gravity[0, 2] = 1.0
        gravity[1].zero_()
        gravity[2, 0] = math.nan
        counts, latches, edges = update_handoff_gate(gravity, angular, counter, switched)
        self.assertEqual(counts.tolist(), [0, 0, 0])
        self.assertTrue(latches.all().item())
        self.assertFalse(edges.any().item())

    def test_strict_tilt_threshold_and_up_direction(self):
        gravity, angular, counter, switched = state(6)
        for index, degrees in enumerate((0.0, 29.999, 30.0, 30.001, 90.0, 180.0)):
            theta = math.radians(degrees)
            gravity[index] = torch.tensor([math.sin(theta), 0.0, -math.cos(theta)], dtype=gravity.dtype)
        counter.fill_(9)
        counts, _, edges = update_handoff_gate(gravity, angular, counter, switched)
        self.assertEqual(counts.tolist(), [10, 10, 0, 0, 0, 0])
        self.assertEqual(edges.tolist(), [True, True, False, False, False, False])

    def test_angular_speed_uses_norm_not_individual_axes(self):
        gravity, angular, counter, switched = state(4)
        angular[:] = torch.tensor([[.999, 0, 0], [1., 0, 0], [.8, .8, 0], [0, 0, -1.1]])
        counter.fill_(9)
        self.assertEqual(update_handoff_gate(gravity, angular, counter, switched)[2].tolist(),
                         [True, False, False, False])

    def test_nonfinite_or_zero_gravity_never_triggers(self):
        gravity, angular, counter, switched = state(7)
        gravity[0].zero_()
        gravity[1, 0] = math.nan
        gravity[2, 2] = -math.inf
        angular[3, 0] = math.nan
        angular[4, 2] = math.inf
        gravity[5].fill_(torch.finfo(gravity.dtype).max)
        angular[6].fill_(torch.finfo(angular.dtype).max)
        counter.fill_(9)
        counts, latches, edges = update_handoff_gate(gravity, angular, counter, switched)
        self.assertEqual(counts.tolist(), [0] * 7)
        self.assertFalse(latches.any().item() or edges.any().item())

    def test_gravity_scale_does_not_change_tilt(self):
        gravity, angular, counter, switched = state(3)
        gravity *= torch.tensor([.1, 1., 9.81], dtype=gravity.dtype)[:, None]
        counter.fill_(9)
        self.assertTrue(update_handoff_gate(gravity, angular, counter, switched)[2].all().item())

    def test_inputs_are_not_mutated_or_returned_by_alias(self):
        values = state(2)
        values[2].fill_(9)
        originals = tuple(value.clone() for value in values)
        outputs = update_handoff_gate(*values)
        for value, original in zip(values, originals):
            self.assertTrue(torch.equal(value, original))
        for output in outputs:
            self.assertTrue(all(output.data_ptr() != value.data_ptr() for value in values))

    def test_counter_saturates_without_integer_overflow(self):
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            values = state(counter_dtype=dtype)
            values[2].fill_(torch.iinfo(dtype).max)
            counts, _, edges = update_handoff_gate(*values)
            self.assertEqual(counts.item(), 10)
            self.assertEqual(counts.dtype, dtype)
            self.assertTrue(edges.item())

    def test_negative_incoming_counter_restarts_conservatively(self):
        values = state()
        values[2].fill_(-4)
        counts, _, edges = update_handoff_gate(*values)
        self.assertEqual(counts.item(), 1)
        self.assertFalse(edges.item())

    def test_nondefault_step_duration_rounds_up_hold_samples(self):
        for dt, previous, expected in ((.03, 5, False), (.03, 6, True), (.025, 7, True), (.2, 0, True)):
            values = state()
            values[2].fill_(previous)
            self.assertEqual(update_handoff_gate(*values, step_dt=dt)[2].item(), expected)

    def test_empty_batches_and_supported_floating_types(self):
        for dtype in (torch.float32, torch.float64):
            for n in (0, 1, 5):
                outputs = update_handoff_gate(*state(n, dtype))
                self.assertTrue(all(value.shape == (n,) for value in outputs))
                self.assertEqual(outputs[0].dtype, torch.int32)
                self.assertEqual(outputs[1].dtype, torch.bool)
                self.assertEqual(outputs[2].dtype, torch.bool)

    def test_invalid_dt_and_metadata_fail(self):
        for dt in (0, -1, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                update_handoff_gate(*state(), step_dt=dt)
        for dt in (True, "0.02", None):
            with self.subTest(dt=dt), self.assertRaises(TypeError):
                update_handoff_gate(*state(), step_dt=dt)
        mutations = ((0, torch.zeros(1, 2), ValueError),
                     (1, torch.zeros(2, 3), ValueError),
                     (1, torch.zeros(1, 3, dtype=torch.float32), ValueError),
                     (2, torch.zeros(1, dtype=torch.float64), TypeError),
                     (3, torch.zeros(1, dtype=torch.int32), TypeError),
                     (2, torch.zeros(1, 1, dtype=torch.int32), ValueError),
                     (0, [[0., 0., -1.]], TypeError))
        for index, replacement, exception in mutations:
            values = list(state())
            values[index] = replacement
            with self.subTest(index=index, replacement=replacement), self.assertRaises(exception):
                update_handoff_gate(*values)
        with self.assertRaises(ValueError):
            update_handoff_gate(*state(counter_dtype=torch.int8), step_dt=.001)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
