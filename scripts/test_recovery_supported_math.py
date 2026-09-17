"""CPU-only tests of dense shaping; no simulator or task registration needed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


_PATH = Path(__file__).resolve().parents[1] / "src/go2_recovery/recovery_supported_math.py"
_SPEC = importlib.util.spec_from_file_location("recovery_supported_math", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MATH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MATH)
score = _MATH.supported_pose_score
validate_range = _MATH.validate_supported_joint_soft_range


class SupportedPoseScoreTests(unittest.TestCase):
    def inputs(self, n=3):
        return torch.ones(n), torch.zeros(n, 12), torch.ones(12)

    def test_exact_formula_and_extremes(self):
        cos = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, 1.0])
        delta = torch.tensor([0.0, 0.0, 0.25, 0.5, 0.0, 2.0]).unsqueeze(-1).repeat(1, 12)
        actual = score(cos, delta, torch.ones(12))
        expected = torch.tensor([0.0, 0.0, 0.1875, 0.5, 1.0, 0.0])
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))

    def test_joint_error_monotonic_and_sign_symmetric(self):
        magnitude = torch.tensor([0.0, 0.1, 0.3, 0.7, 0.9, 1.0, 2.0])
        delta = magnitude[:, None].repeat(1, 12)
        actual = score(torch.ones(7), delta, torch.ones(12))
        self.assertTrue(bool((actual[:-1] >= actual[1:]).all()))
        torch.testing.assert_close(actual, score(torch.ones(7), -delta, torch.ones(12)))

    def test_uprightness_monotonic(self):
        cos = torch.linspace(-2, 2, 31)
        actual = score(cos, torch.full((31, 12), 0.4), torch.ones(12))
        self.assertTrue(bool((actual[:-1] <= actual[1:]).all()))

    def test_batched_and_unbatched_ranges_match(self):
        cos, delta, ranges = self.inputs()
        delta[:] = 0.2
        torch.testing.assert_close(score(cos, delta, ranges), score(cos, delta, ranges.repeat(3, 1)))

    def test_per_joint_ranges(self):
        cos, delta, _ = self.inputs(1)
        ranges = torch.arange(1, 13, dtype=torch.float32)
        delta[:] = ranges * 0.75
        torch.testing.assert_close(score(cos, delta, ranges), torch.tensor([0.25]))

    def test_gradients_finite_and_large_error_not_exponentially_flat(self):
        cos = torch.tensor([0.4, 0.9], requires_grad=True)
        delta = torch.full((2, 12), 0.9, requires_grad=True)
        actual = score(cos, delta, torch.ones(12))
        actual.sum().backward()
        self.assertTrue(bool(torch.isfinite(cos.grad).all()))
        self.assertTrue(bool(torch.isfinite(delta.grad).all()))
        self.assertTrue(bool((delta.grad.abs() > 0).all()))
        torch.testing.assert_close(delta.grad, -(cos.detach().square()[:, None] / 12).expand_as(delta))

    def test_nan_state_is_not_silently_sanitized(self):
        cos = torch.tensor([float("nan"), 1.0])
        delta = torch.full((2, 12), 0.2)
        delta[1, 0] = float("nan")
        self.assertTrue(bool(torch.isnan(score(cos, delta, torch.ones(12))).all()))

    def test_empty_batches(self):
        for ranges in (torch.ones(12), torch.ones(0, 12)):
            validate_range(ranges)
            actual = score(torch.empty(0), torch.empty(0, 12), ranges)
            self.assertEqual(actual.shape, (0,))

    def test_metadata_errors(self):
        cos, delta, ranges = self.inputs()
        bad_inputs = [
            (cos[:, None], delta, ranges, ValueError),
            (cos, torch.zeros(3, 11), ranges, ValueError),
            (cos, torch.zeros(2, 12), ranges, ValueError),
            (cos, delta, torch.ones(1, 12), ValueError),
            (cos, delta, torch.ones(12, 1), ValueError),
            (cos, delta, torch.ones(2, 3, 12), ValueError),
            (cos, delta.long(), ranges, TypeError),
            (cos, delta, ranges.double(), TypeError),
            (cos.double(), delta, ranges, TypeError),
            (cos, delta.to("meta"), ranges, ValueError),
            ([1.0] * 3, delta, ranges, TypeError),
        ]
        for bad_cos, bad_delta, bad_range, exception in bad_inputs:
            with self.subTest(cos_type=type(bad_cos), delta_shape=bad_delta.shape, range_shape=bad_range.shape):
                with self.assertRaises(exception):
                    score(bad_cos, bad_delta, bad_range)

    def test_init_range_value_validation(self):
        validate_range(torch.ones(12))
        validate_range(torch.ones(3, 12))
        for invalid in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            ranges = torch.ones(12)
            ranges[4] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_range(ranges)
        with self.assertRaises(TypeError):
            validate_range(torch.ones(12, dtype=torch.int32))
        with self.assertRaises(ValueError):
            validate_range(torch.ones(11))

    def test_inputs_are_not_mutated(self):
        cos, delta, ranges = self.inputs()
        originals = [x.clone() for x in (cos, delta, ranges)]
        score(cos, delta, ranges)
        for actual, expected in zip((cos, delta, ranges), originals):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
