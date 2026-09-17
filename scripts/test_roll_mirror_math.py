"""CPU-only mirror mapping, real/virtual isolation, and frozen-source contracts."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "go2_recovery"))
from roll_mirror_math import (NATIVE_JOINT_NAMES, JOINT_PERMUTATION, JOINT_SIGNS,
    RIGHT_GRAVITY_Y_THRESHOLD, initial_right_mask, physical_roll_action,
    reflect_joint_vector, reflect_policy_observation, roll_input_copy, validate_mirror_joint_contract)
from recovery_math import normal_stance_geometry

ENTRY = ROOT / "scripts" / "evaluate_handoff_mirror.py"
TRANSITION = ROOT / "scripts" / "evaluate_handoff_transition.py"
REPORT_ROOT = ROOT / "evaluations" / "20260917-handoff-transition-v1"


def actual_metadata():
    return json.loads((REPORT_ROOT / "hard-side" / "model_1999.pt_recovery_metrics.json").read_text())


def actual_contract():
    meta = actual_metadata()["handoff_controller"]["joint_limits"]
    return [meta["joint_names"], torch.tensor([meta["default_joint_positions_rad"]]),
            torch.tensor([meta["soft_joint_limits_rad"]]), torch.tensor([meta["hard_joint_limits_rad"]])]


def funcs(path):
    return {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)}


class MirrorMathTests(unittest.TestCase):
    def test_native_mapping_is_go2_not_anymal(self):
        self.assertEqual(NATIVE_JOINT_NAMES, tuple(actual_contract()[0]))
        x = torch.arange(12.).reshape(1, 12)
        self.assertEqual(reflect_joint_vector(x).tolist()[0], [-1, 0, -3, -2, 5, 4, 7, 6, 9, 8, 11, 10])

    def test_joint_and_observation_involutions(self):
        generator = torch.Generator().manual_seed(910)
        for width, transform in [(12, reflect_joint_vector), (48, reflect_policy_observation)]:
            x = torch.randn(31, width, generator=generator)
            original = x.clone()
            y = transform(x)
            self.assertTrue(torch.equal(transform(y), x))
            self.assertTrue(torch.equal(x, original))
            self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_polar_axial_and_command_signs(self):
        x = torch.ones(1, 48)
        y = reflect_policy_observation(x)[0]
        self.assertEqual(y[:12].tolist(), [1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1])
        for start in (12, 24, 36):
            self.assertEqual(y[start:start+12].tolist(), list(JOINT_SIGNS))

    def test_actual_default_and_limits_exact(self):
        receipt = validate_mirror_joint_contract(*actual_contract())
        self.assertTrue(receipt["default_exactly_compatible"])
        self.assertTrue(receipt["soft_limits_exactly_compatible"])
        self.assertTrue(receipt["hard_limits_exactly_compatible"])
        self.assertFalse(receipt["physical_asset_symmetry_proven"])

    def test_contract_rejects_wrong_names_default_and_limits(self):
        for mutation in range(5):
            args = actual_contract()
            if mutation == 0:
                args[0][0], args[0][1] = args[0][1], args[0][0]
            elif mutation == 1:
                args[1][0, 0] += .01
            elif mutation == 2:
                args[2][0, 0, 0] -= .01
            elif mutation == 3:
                args[3][0, 4, 0] += .01
            else:
                args[2][0, 0, 0] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_mirror_joint_contract(*args)

    def test_target_clamp_commutation_including_saturation(self):
        _, default, limits, _ = actual_contract()
        actions = torch.randn(256, 12, generator=torch.Generator().manual_seed(731)) * 50
        target = lambda a: (default + .25 * a).clamp(limits[:, :, 0], limits[:, :, 1])
        self.assertTrue(torch.allclose(target(reflect_joint_vector(actions)), reflect_joint_vector(target(actions)), atol=1e-6, rtol=0))

    def test_real_pose_cone_eligibility_not_pose_label(self):
        gravity = torch.tensor([[0., -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
                                [0, -2, 0], [0, -.8, .6], [0, -1, 0]])
        eligible = torch.tensor([True, True, True, True, True, True, False])
        self.assertEqual(initial_right_mask(gravity, eligible, "initial_right").tolist(),
                         [True, False, False, False, True, False, False])
        self.assertFalse(initial_right_mask(gravity, eligible, "off").any())
        self.assertEqual(RIGHT_GRAVITY_Y_THRESHOLD, -0.8660254037844387)

    def test_fixed_mask_is_independent_of_later_gravity(self):
        gravity = torch.tensor([[0., -1, 0]])
        mask = initial_right_mask(gravity, torch.tensor([True]), "initial_right")
        gravity[:] = torch.tensor([[0., 1, 0]])
        self.assertTrue(mask[0])

    def test_actual_three_baseline_start_masks(self):
        expected = {"side": 11, "upside_down": 0, "upright": 0}
        for pose, count in expected.items():
            report = json.loads((REPORT_ROOT / f"hard-{pose}" / "model_1999.pt_recovery_metrics.json").read_text())
            starts = report["results"][pose]["policy_start_state"]
            g = torch.tensor([x["projected_gravity_b"] for x in starts])
            e = torch.tensor([x["eligible_settled_fallen_recovery"] for x in starts])
            self.assertEqual(int(initial_right_mask(g, e, "initial_right").sum()), count)
            self.assertEqual(int(initial_right_mask(g, e, "off").sum()), 0)

    def test_tensordict_deep_copy_no_real_leaf_mutation(self):
        obs = TensorDict({"policy": torch.arange(96.).reshape(2, 48),
                          "extra": TensorDict({"value": torch.ones(2, 5)}, batch_size=[2])}, batch_size=[2])
        before = obs.clone(recurse=True)
        mirror = roll_input_copy(obs, torch.tensor([True, False]))
        self.assertTrue(torch.equal(mirror["policy"][0], reflect_policy_observation(obs["policy"])[0]))
        self.assertTrue(torch.equal(mirror["policy"][1], obs["policy"][1]))
        self.assertNotEqual(mirror["policy"].data_ptr(), obs["policy"].data_ptr())
        self.assertNotEqual(mirror["extra", "value"].data_ptr(), obs["extra", "value"].data_ptr())
        mirror["policy"].zero_()
        mirror["extra", "value"].zero_()
        self.assertTrue(torch.equal(obs["policy"], before["policy"]))
        self.assertTrue(torch.equal(obs["extra", "value"], before["extra", "value"]))

    def test_off_input_and_output_numerical_identity(self):
        obs = TensorDict({"policy": torch.randn(5, 48)}, batch_size=[5])
        mask = torch.zeros(5, dtype=torch.bool)
        copied = roll_input_copy(obs, mask)
        self.assertTrue(torch.equal(copied["policy"], obs["policy"]))
        action = torch.randn(5, 12)
        self.assertTrue(torch.equal(physical_roll_action(action, mask), action))

    def test_virtual_history_consistent_with_issued_native_action(self):
        model_action = torch.arange(36.).reshape(3, 12)
        selected = torch.tensor([True, False, True])
        physical = physical_roll_action(model_action, selected)
        real = torch.zeros(3, 48)
        real[:, 36:] = physical  # Simulated next real last-action measurement, not a runtime buffer write.
        inputs = roll_input_copy(TensorDict({"policy": real}, batch_size=[3]), selected)
        self.assertTrue(torch.equal(inputs["policy"][:, 36:], model_action))

    def test_geometry_formula_invariance_not_asset_kinematics(self):
        generator = torch.Generator().manual_seed(433)
        feet = torch.randn(128, 4, 3, generator=generator) * .2
        knees = torch.randn(128, 4, 3, generator=generator) * .1
        residual = torch.randn(128, 12, generator=generator) * .2
        feet[0] = torch.tensor([[.2, .16, -.3], [.2, -.16, -.3], [-.2, .16, -.3], [-.2, -.16, -.3]])
        knees[0] = feet[0] / 2
        residual[0] = 0
        reflection = lambda x: x[:, (1, 0, 3, 2)] * x.new_tensor([1, -1, 1])
        original = normal_stance_geometry(feet, knees, residual)
        self.assertTrue(original[0])
        self.assertTrue(torch.equal(original, normal_stance_geometry(reflection(feet), reflection(knees), reflect_joint_vector(residual))))

    def test_invalid_inputs_fail_closed(self):
        for x in [torch.zeros(1, 47), torch.zeros(48), torch.zeros(1, 48, dtype=torch.long), torch.full((1, 48), float("nan"))]:
            with self.assertRaises(ValueError):
                reflect_policy_observation(x)
        for g in [torch.zeros(1, 3), torch.full((1, 3), float("inf"))]:
            with self.assertRaises(ValueError):
                initial_right_mask(g, torch.ones(1, dtype=torch.bool), "initial_right")
        with self.assertRaises(TypeError):
            roll_input_copy({"policy": torch.zeros(1, 48)}, torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            physical_roll_action(torch.zeros(1, 12), torch.ones(1))


class MirrorSourceTests(unittest.TestCase):
    def test_frozen_sources_unchanged(self):
        files = {ROOT / "scripts" / "evaluate_go2_recovery.py": "4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f",
                 TRANSITION: "c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a",
                 ROOT / "src" / "go2_recovery" / "handoff_transition_math.py": "b66167039b4698d66f3cc412b19bacd0b1b2f37279f7ce0856ec6c09b196aa4e"}
        for path, expected in files.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_original_helpers_identical(self):
        old, new = funcs(TRANSITION), funcs(ENTRY)
        for name in ("_set_pose_class", "_start_snapshot", "_settle_nominal_pose", "_reset_policy_history",
                     "_set_bank_states", "_select_bank_states", "_stable_stand", "_stance_geometry",
                     "_transition_physical_rows", "_transition_rows_before"):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)

    def test_gate_and_single_step_calls_preserved(self):
        old, new = funcs(TRANSITION)["main"], funcs(ENTRY)["main"]
        for name in ("update_handoff_gate", "transition_action", "_reset_policy_history", "_settle_nominal_pose"):
            calls = lambda node: [ast.dump(x) for x in ast.walk(node) if isinstance(x, ast.Call)
                                  and isinstance(x.func, ast.Name) and x.func.id == name]
            self.assertEqual(calls(old), calls(new), name)
        mask_calls = [x for x in ast.walk(new) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                      and x.func.id == "initial_right_mask"]
        self.assertEqual(len(mask_calls), 1)
        self.assertEqual([ast.unparse(x) for x in mask_calls[0].args], ["start_gravity", "eligible_fallen", "args_cli.mirror_mode"])

    def test_no_ramp_cli_and_standing_uses_real_obs(self):
        text = ENTRY.read_text(encoding="utf-8")
        self.assertNotIn('parser.add_argument("--ramp_seconds"', text)
        self.assertNotIn('parser.add_argument("--transition_mode"', text)
        self.assertIn('parser.set_defaults(transition_mode="hard", ramp_seconds=0.0)', text)
        self.assertIn('stand_action = stand_policy(obs)', text)
        self.assertIn('roll_model_action = policy(roll_obs)', text)
        self.assertIn('"protocol_version"] = "handoff_mirror_experimental_v1"', text)
        self.assertIn('"physical_asset_symmetry_proven": False', text)
        self.assertIn('"roll_actor_output_model_raw_action"', text)
        self.assertIn('"roll_actor_output_physical_raw_action"', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
