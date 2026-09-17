"""CPU-only checks for the completed-interval standing refinement latch."""

from __future__ import annotations

import inspect
import unittest

import torch

from supported_refinement_gate import (
    CONTROL_DT,
    REQUIRED_COMPLETED_INTERVALS,
    REQUIRED_HOLD_SECONDS,
    advance_after_completed_interval as advance,
)


class SupportedRefinementGateTests(unittest.TestCase):
    def state(self, count=0, latched=False):
        return torch.tensor([count], dtype=torch.int64), torch.tensor([latched])

    def test_fixed_protocol(self):
        self.assertEqual(CONTROL_DT, 0.02)
        self.assertEqual(REQUIRED_COMPLETED_INTERVALS, 150)
        self.assertEqual(REQUIRED_HOLD_SECONDS, 3.0)

    def test_149_intervals_do_not_enable(self):
        count, latch = self.state()
        for _ in range(149):
            count, latch, enabled = advance(torch.tensor([True]), count, latch)
            self.assertFalse(bool(latch[0]))
            self.assertFalse(bool(enabled[0]))
        self.assertEqual(count.item(), 149)

    def test_exact_150_enables_once_for_next_action(self):
        count, latch = self.state()
        action_actors = []
        enable_intervals = []
        for completed_interval in range(1, 152):
            # The action is chosen before the completed interval is scored.
            action_actors.append("refine" if latch.item() else "original")
            count, latch, enabled = advance(torch.tensor([True]), count, latch)
            if enabled.item():
                enable_intervals.append(completed_interval)
        self.assertEqual(action_actors[:150], ["original"] * 150)
        self.assertEqual(action_actors[150], "refine")
        self.assertEqual(enable_intervals, [150])
        self.assertEqual(count.item(), 150)

    def test_invalid_at_150_resets_unlatched_count(self):
        count, latch = self.state(149)
        count, latch, enabled = advance(torch.tensor([False]), count, latch)
        self.assertEqual(count.item(), 0)
        self.assertFalse(latch.item())
        self.assertFalse(enabled.item())
        for _ in range(149):
            count, latch, enabled = advance(torch.tensor([True]), count, latch)
        self.assertFalse(latch.item())
        count, latch, enabled = advance(torch.tensor([True]), count, latch)
        self.assertTrue(latch.item())
        self.assertTrue(enabled.item())

    def test_invalid_after_latch_does_not_hide_refinement_regression(self):
        count, latch = self.state(150, True)
        count, latch, enabled = advance(torch.tensor([False]), count, latch)
        self.assertEqual(count.item(), 0)
        self.assertTrue(latch.item())
        self.assertFalse(enabled.item())
        for _ in range(150):
            count, latch, enabled = advance(torch.tensor([True]), count, latch)
            self.assertTrue(latch.item())
            self.assertFalse(enabled.item())

    def test_multiple_environments_are_independent(self):
        count = torch.tensor([0, 148, 149, 150, 149, 3], dtype=torch.int64)
        latch = torch.tensor([False, False, False, True, False, True])
        valid = torch.tensor([True, True, True, True, False, False])
        result = advance(valid, count, latch)
        self.assertEqual(result[0].tolist(), [1, 149, 150, 150, 0, 0])
        self.assertEqual(result[1].tolist(), [False, False, True, True, False, True])
        self.assertEqual(result[2].tolist(), [False, False, True, False, False, False])

    def test_row_permutation_equivariant(self):
        count = torch.tensor([0, 148, 149, 150, 149, 3], dtype=torch.int64)
        latch = torch.tensor([False, False, False, True, False, True])
        valid = torch.tensor([True, True, True, True, False, False])
        permutation = torch.tensor([4, 2, 0, 5, 3, 1])
        expected = advance(valid, count, latch)
        actual = advance(valid[permutation], count[permutation], latch[permutation])
        for want, got in zip(expected, actual):
            self.assertTrue(torch.equal(want[permutation], got))

    def test_inputs_unchanged_and_outputs_do_not_alias(self):
        count = torch.tensor([149, 4, 150], dtype=torch.int64)
        latch = torch.tensor([False, False, True])
        valid = torch.tensor([True, False, False])
        inputs = (valid, count, latch)
        copies = tuple(value.clone() for value in inputs)
        outputs = advance(*inputs)
        for value, before in zip(inputs, copies):
            self.assertTrue(torch.equal(value, before))
        for output in outputs:
            for value in inputs:
                self.assertNotEqual(output.data_ptr(), value.data_ptr())
        outputs[0].fill_(0)
        outputs[1].fill_(False)
        outputs[2].fill_(False)
        for value, before in zip(inputs, copies):
            self.assertTrue(torch.equal(value, before))

    def test_noncontiguous_inputs_are_supported_without_mutation(self):
        count = torch.tensor([149, 8, 3, 9], dtype=torch.int64)[::2]
        valid = torch.tensor([True, False, False, True])[::2]
        latch = torch.tensor([False, True, False, True])[::2]
        result = advance(valid, count, latch)
        self.assertEqual(result[0].tolist(), [150, 0])
        self.assertEqual(count.tolist(), [149, 3])

    def test_rejects_non_tensor_inputs(self):
        count, latch = self.state()
        valid = torch.tensor([True])
        for bad in ([True], True, None, 1):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                advance(bad, count, latch)
        with self.assertRaises(TypeError):
            advance(valid, [0], latch)
        with self.assertRaises(TypeError):
            advance(valid, count, [False])

    def test_rejects_wrong_dtypes(self):
        count, latch = self.state()
        with self.assertRaises(TypeError):
            advance(torch.tensor([1]), count, latch)
        with self.assertRaises(TypeError):
            advance(torch.tensor([True]), count.float(), latch)
        with self.assertRaises(TypeError):
            advance(torch.tensor([True]), count.int(), latch)
        with self.assertRaises(TypeError):
            advance(torch.tensor([True]), count, latch.float())

    def test_rejects_wrong_or_empty_shapes(self):
        for shape in ((), (1, 1), (0,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                advance(
                    torch.zeros(shape, dtype=torch.bool),
                    torch.zeros(shape, dtype=torch.int64),
                    torch.zeros(shape, dtype=torch.bool),
                )
        with self.assertRaises(ValueError):
            advance(torch.tensor([True, False]), *self.state())
        with self.assertRaises(ValueError):
            advance(torch.tensor([True]), torch.tensor([0, 1]), torch.tensor([False]))
        with self.assertRaises(ValueError):
            advance(torch.tensor([True]), torch.tensor([0]), torch.tensor([False, True]))

    def test_rejects_out_of_range_counts(self):
        for bad_count in (-1, 151, 2**63 - 1):
            with self.subTest(count=bad_count), self.assertRaises(ValueError):
                advance(torch.tensor([True]), *self.state(bad_count))

    def test_rejects_impossible_unlatched_150(self):
        with self.assertRaises(ValueError):
            advance(torch.tensor([True]), *self.state(150, False))

    def test_rejects_wrong_or_nonfinite_dt(self):
        for dt in (0.01, 0.020000001, 0.0, -0.02, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                advance(torch.tensor([True]), *self.state(), dt=dt)
        for dt in (True, None, "0.02"):
            with self.subTest(dt=dt), self.assertRaises(TypeError):
                advance(torch.tensor([True]), *self.state(), dt=dt)

    def test_rejects_sparse_and_unmaterialized_storage(self):
        with self.assertRaises(ValueError):
            advance(torch.tensor([True]).to_sparse(), *self.state())
        with self.assertRaises(ValueError):
            advance(torch.empty(1, dtype=torch.bool, device="meta"), *self.state())

    def test_no_reset_or_environment_interface(self):
        self.assertEqual(
            list(inspect.signature(advance).parameters),
            ["strict_valid", "completed_count", "latched", "dt"],
        )
        with self.assertRaises(TypeError):
            advance(torch.tensor([True]), *self.state(), reset=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
