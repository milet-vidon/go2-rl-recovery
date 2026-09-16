"""CPU AST/mock checks for the Go2 Smith reward adapter; no simulator import.

These exercise the real class body with a minimal ManagerTermBase replacement.
They do not claim to replace the Isaac Lab runtime integration smoke test.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_smith_math_adapter_test", ROOT / "src/go2_recovery/recovery_smith_math.py")
assert SPEC and SPEC.loader
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)


class MockManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env


class MockScene(dict):
    pass


def adapter_class():
    source = ROOT / "src/go2_recovery/recovery_smith_mdp.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "SmithRecoveryReward")
    scope = {"ManagerTermBase": MockManagerTermBase,
             "smith_recovery_terms": MATH.smith_recovery_terms}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[node.name]


NAMES = [f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
         for kind in ("hip", "thigh", "calf")]


def make_env(names=None, n=3):
    names = list(NAMES if names is None else names)
    joint_count = len(names)
    default = torch.linspace(-0.8, 0.4, joint_count, dtype=torch.float64).repeat(n, 1)
    root = torch.zeros(n, 3, dtype=torch.float64)
    root[:, 2] = 0.32
    gravity = torch.zeros(n, 3, dtype=torch.float64)
    gravity[:, 2] = -1.0
    data = NS(joint_pos=default.clone(), default_joint_pos=default.clone(),
              joint_vel=torch.zeros_like(default), root_pos_w=root,
              projected_gravity_b=gravity)
    robot = NS(joint_names=names, data=data)
    scene = MockScene(robot=robot)
    scene.env_origins = torch.zeros(n, 3, dtype=torch.float64)
    return NS(scene=scene), data


class SmithAdapterTests(unittest.TestCase):
    def test_named_weights_for_all_twelve_joints(self):
        env, data = make_env()
        term = adapter_class()(NS(), env)
        expected = {"hip": 1.0, "thigh": 0.75, "calf": 0.5}
        for name, weight in zip(env.scene["robot"].joint_names, term.joint_weights):
            self.assertEqual(weight.item(), expected[name.split("_")[1]])
        self.assertEqual(term.joint_weights.shape, (12,))
        self.assertEqual(term.joint_weights.dtype, data.joint_pos.dtype)
        self.assertEqual(term.joint_weights.device, data.joint_pos.device)

    def test_shuffled_joint_names_and_state_preserve_results(self):
        env, data = make_env()
        delta = torch.linspace(-0.5, 0.8, 36, dtype=torch.float64).reshape(3, 12)
        data.joint_pos += delta
        data.joint_vel = delta * 2.0
        data.projected_gravity_b[:, 2] = torch.tensor([-1.0, -0.95, -0.4])
        baseline = adapter_class()(NS(), env)
        permutation = [7, 0, 11, 3, 5, 2, 9, 8, 1, 10, 6, 4]
        shuffled, shuffled_data = make_env([NAMES[i] for i in permutation])
        for attribute in ("joint_pos", "default_joint_pos", "joint_vel"):
            setattr(shuffled_data, attribute, getattr(data, attribute)[:, permutation].clone())
        shuffled_data.projected_gravity_b = data.projected_gravity_b.clone()
        shuffled_term = adapter_class()(NS(), shuffled)
        torch.testing.assert_close(shuffled_term.joint_weights,
                                   baseline.joint_weights[permutation])
        for mode in ("roll", "stand"):
            torch.testing.assert_close(baseline(env, mode), shuffled_term(shuffled, mode))

    def test_unknown_missing_duplicate_and_extra_names_rejected(self):
        cases = {
            "unknown": ["FL_ankle_joint"] + NAMES[1:],
            "missing": NAMES[:-1],
            "duplicate": NAMES[:-1] + [NAMES[0]],
            "extra": NAMES + ["auxiliary_joint"],
            "wrong_leg": [name.replace("FL_", "LF_") for name in NAMES],
        }
        for label, names in cases.items():
            with self.subTest(label=label):
                env, _ = make_env(names)
                with self.assertRaises(ValueError):
                    adapter_class()(NS(), env)

    def test_only_roll_and_stand_modes_are_supported(self):
        env, _ = make_env()
        term = adapter_class()(NS(), env)
        for mode in ("total", "pose", "height", "velocity", "stand_gate", "Roll", "", None, 0):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                term(env, mode)
        for mode in ("roll", "stand"):
            self.assertEqual(term(env, mode).shape, (3,))

    def test_world_height_subtracts_environment_origin(self):
        env, data = make_env()
        term = adapter_class()(NS(), env)
        # Both unsaturated and clamped heights: raw world z would give a
        # different stand reward in at least two environments.
        local_height = torch.tensor([0.08, 0.16, 0.40], dtype=torch.float64)
        data.root_pos_w[:, 2] = local_height
        reference = term(env, "stand")
        env.scene.env_origins = torch.tensor([[2.0, 3.0, 2.0], [-4.0, 5.0, -3.0],
                                             [6.0, -7.0, 1.5]], dtype=torch.float64)
        data.root_pos_w = env.scene.env_origins.clone()
        data.root_pos_w[:, 2] += local_height
        torch.testing.assert_close(term(env, "stand"), reference)
        expected = 0.8 + 0.2 * (local_height / 0.32).clamp(0, 1)
        torch.testing.assert_close(reference, expected)

    def test_projected_gravity_sign_and_default_joint_reference(self):
        env, data = make_env()
        data.projected_gravity_b = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                                               [0.0, 0.0, -1.0]], dtype=torch.float64)
        term = adapter_class()(NS(), env)
        # The default joint pose is deliberately nonzero in this fixture.
        self.assertGreater(data.default_joint_pos.abs().sum().item(), 0)
        torch.testing.assert_close(term(env, "roll"), torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64))
        torch.testing.assert_close(term(env, "stand"), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))

    def test_custom_target_height_is_forwarded(self):
        env, data = make_env()
        term = adapter_class()(NS(), env)
        data.root_pos_w[:, 2] = 0.16
        torch.testing.assert_close(term(env, "stand", target_height=0.64),
                                   torch.full((3,), 0.85, dtype=torch.float64))
        with self.assertRaises(ValueError):
            term(env, "stand", target_height=0.0)

    def test_weighted_adapter_pair_equals_twenty_times_total(self):
        env, data = make_env(n=5)
        data.projected_gravity_b[:, 2] = torch.tensor([1.0, 0.0, -0.81, -0.95, -1.0])
        data.joint_pos += torch.linspace(-0.8, 0.6, 60, dtype=torch.float64).reshape(5, 12)
        data.joint_vel = torch.linspace(-4.0, 5.0, 60, dtype=torch.float64).reshape(5, 12)
        data.root_pos_w[:, 2] = torch.linspace(0.05, 0.4, 5)
        term = adapter_class()(NS(), env)
        expected = MATH.smith_recovery_terms(
            -data.projected_gravity_b[:, 2],
            data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
            data.joint_pos - data.default_joint_pos, data.joint_vel,
            term.joint_weights)["total"]
        weighted = 10.0 * term(env, "roll") + 10.0 * term(env, "stand")
        torch.testing.assert_close(weighted, 20.0 * expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
