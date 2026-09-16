"""CPU-only regression tests for Smith-inspired recovery reward equations."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


_MODULE_PATH = Path(__file__).parents[1] / "src/go2_recovery/recovery_smith_math.py"
_SPEC = importlib.util.spec_from_file_location("recovery_smith_math", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
smith_recovery_terms = _MODULE.smith_recovery_terms


class SmithRecoveryMathTests(unittest.TestCase):
    def inputs(self, n=3, j=12, dtype=torch.float64):
        return [torch.ones(n, dtype=dtype), torch.full((n,), 0.32, dtype=dtype),
                torch.zeros(n, j, dtype=dtype), torch.zeros(n, j, dtype=dtype),
                torch.ones(j, dtype=dtype)]

    def test_inverted_side_and_nominal_stand(self):
        values = self.inputs()
        values[0] = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = smith_recovery_terms(*values)
        torch.testing.assert_close(result["total"], values[0].new_tensor([0.0, 0.125, 1.0]))
        self.assertEqual(result["stand_gate"].tolist(), [False, False, True])

    def test_strict_stand_gate_at_threshold(self):
        for dtype in (torch.float32, torch.float64):
            values = self.inputs(dtype=dtype)
            threshold = torch.tensor(math.cos(0.2 * math.pi), dtype=dtype)
            values[0] = torch.stack((torch.nextafter(threshold, threshold.new_tensor(-math.inf)),
                                     threshold,
                                     torch.nextafter(threshold, threshold.new_tensor(math.inf))))
            result = smith_recovery_terms(*values)
            self.assertEqual(result["stand_gate"].tolist(), [False, False, True])
            self.assertEqual(result["stand"][:2].tolist(), [0.0, 0.0])

    def test_closed_gate_releases_pose_height_and_velocity(self):
        values = self.inputs(n=4)
        values[0] = torch.tensor([-1.0, -0.5, 0.0, 0.8], dtype=torch.float64)
        original = smith_recovery_terms(*values)
        values[1] = torch.tensor([-2.0, 0.0, 2.0, 0.1], dtype=torch.float64)
        values[2].fill_(4.0)
        values[3].fill_(20.0)
        changed = smith_recovery_terms(*values)
        torch.testing.assert_close(changed["total"], original["total"])
        self.assertTrue(torch.equal(changed["stand"], torch.zeros(4, dtype=torch.float64)))

    def test_joint_penalties_use_weighted_sum_not_mean(self):
        values = self.inputs(n=1, j=3)
        values[2][0] = torch.tensor([1.0, -2.0, 3.0])
        values[3][0] = torch.tensor([2.0, -3.0, 4.0])
        values[4] = torch.tensor([1.0, 0.75, 0.5], dtype=torch.float64)
        result = smith_recovery_terms(*values)
        expected_pose = math.exp(-0.6 * (1.0 + 2.25 + 2.25))
        expected_velocity = math.exp(-0.02 * (4.0 + 9.0 + 16.0))
        self.assertAlmostEqual(result["pose"].item(), expected_pose)
        self.assertAlmostEqual(result["velocity"].item(), expected_velocity)
        self.assertAlmostEqual(result["total"].item(),
                               0.5 + 0.5 * (0.2 + 0.6 * expected_pose + 0.2 * expected_velocity))

    def test_joint_permutation_invariant_when_weights_follow_names(self):
        values = self.inputs(n=2, j=3)
        values[2] = torch.tensor([[0.1, 0.6, -0.4], [0.8, -0.2, 0.3]], dtype=torch.float64)
        values[3] = values[2] * 3.0
        values[4] = torch.tensor([1.0, 0.75, 0.5], dtype=torch.float64)
        original = smith_recovery_terms(*values)
        permutation = [2, 0, 1]
        permuted = smith_recovery_terms(values[0], values[1], values[2][:, permutation],
                                       values[3][:, permutation], values[4][permutation])
        torch.testing.assert_close(original["total"], permuted["total"])

    def test_cosine_height_clamp_and_custom_target(self):
        values = self.inputs(n=5)
        values[0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
        values[1] = torch.tensor([-1.0, 0.0, 0.2, 0.4, 0.8], dtype=torch.float64)
        result = smith_recovery_terms(*values, target_height=0.4)
        torch.testing.assert_close(result["roll"], values[0].new_tensor([0.0, 0.0, 0.25, 1.0, 1.0]))
        torch.testing.assert_close(result["height"], values[0].new_tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

    def test_batch_shapes_dtypes_and_empty_batch(self):
        for dtype in (torch.float32, torch.float64):
            for n in (0, 1, 7):
                result = smith_recovery_terms(*self.inputs(n=n, j=5, dtype=dtype))
                self.assertEqual(set(result), {"roll", "stand", "height", "pose", "velocity", "stand_gate", "total"})
                for name, value in result.items():
                    self.assertEqual(value.shape, (n,))
                    self.assertEqual(value.device.type, "cpu")
                    self.assertEqual(value.dtype, torch.bool if name == "stand_gate" else dtype)

    def test_target_height_rejects_invalid_scalars(self):
        for target in (0.0, -0.1, math.nan, math.inf, -math.inf):
            with self.subTest(target=target), self.assertRaises(ValueError):
                smith_recovery_terms(*self.inputs(), target_height=target)
        for target in (True, "0.32", torch.tensor(0.32), None):
            with self.subTest(target=target), self.assertRaises(TypeError):
                smith_recovery_terms(*self.inputs(), target_height=target)

    def test_invalid_input_shapes_and_types(self):
        mutations = [
            (0, torch.ones(3, 1, dtype=torch.float64), ValueError),
            (1, torch.ones(2, dtype=torch.float64), ValueError),
            (2, torch.zeros(3, 12, 1, dtype=torch.float64), ValueError),
            (2, torch.zeros(2, 12, dtype=torch.float64), ValueError),
            (2, torch.zeros(3, 0, dtype=torch.float64), ValueError),
            (3, torch.zeros(3, 11, dtype=torch.float64), ValueError),
            (4, torch.ones(1, 12, dtype=torch.float64), ValueError),
            (4, torch.ones(11, dtype=torch.float64), ValueError),
            (4, torch.ones(12, dtype=torch.float32), ValueError),
            (1, torch.ones(3, dtype=torch.int64), TypeError),
            (0, [1.0, 1.0, 1.0], TypeError),
            (4, torch.ones(12, device="meta", dtype=torch.float64), ValueError),
        ]
        for index, replacement, exception in mutations:
            with self.subTest(index=index, shape=getattr(replacement, "shape", None), exception=exception):
                values = self.inputs()
                values[index] = replacement
                with self.assertRaises(exception):
                    smith_recovery_terms(*values)

    def test_finite_gradients_at_clamp_and_gate_boundaries(self):
        threshold = math.cos(0.2 * math.pi)
        values = self.inputs(n=9, j=3)
        values[0] = torch.tensor([-2.0, -1.0, 0.0, threshold - 1e-6, threshold,
                                  threshold + 1e-6, 1.0, 2.0, 0.95], dtype=torch.float64)
        values[1] = torch.tensor([-0.1, 0.0, 0.1, 0.2, 0.3, 0.32, 0.4, 0.32, 0.2], dtype=torch.float64)
        values[2].fill_(0.2)
        values[3].fill_(0.3)
        values = [value.requires_grad_() for value in values]
        result = smith_recovery_terms(*values)
        result["total"].sum().backward()
        for value in values:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        for value in values[1:4]:
            self.assertTrue(torch.equal(value.grad[:5], torch.zeros_like(value.grad[:5])))
        self.assertEqual(values[0].grad[0].item(), 0.0)
        self.assertEqual(values[0].grad[7].item(), 0.0)
        self.assertGreater(values[1].grad[-1].item(), 0.0)
        self.assertLess(values[2].grad[-1, 0].item(), 0.0)
        self.assertLess(values[3].grad[-1, 0].item(), 0.0)

    def test_gradcheck_away_from_discontinuous_threshold(self):
        values = self.inputs(n=2, j=3)
        values[0] = torch.tensor([0.3, 0.95], dtype=torch.float64)
        values[1].fill_(0.2)
        values[2].fill_(0.1)
        values[3].fill_(0.2)
        values = tuple(value.requires_grad_() for value in values)
        self.assertTrue(torch.autograd.gradcheck(lambda *args: smith_recovery_terms(*args)["total"], values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
