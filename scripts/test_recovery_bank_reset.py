"""Simulator-free subset-safety regressions for the fallen-state reset term.

The actual class body is AST-loaded with deliberately narrow mocks. These
tests verify reset ownership and state preservation, NOT PhysX contact replay,
quaternion transforms, fallen-state validation, or learned recovery quality.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


_SOURCE = Path(__file__).parents[1] / "src/go2_recovery/recovery_bank_mdp.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_CLASS_NODES = [node for node in _TREE.body
                if isinstance(node, ast.ClassDef) and node.name == "RecoveryBankReset"]
assert len(_CLASS_NODES) == 1, "Expected the real RecoveryBankReset class"


def forbidden(*args, **kwargs):
    raise AssertionError("A subset reset must not advance simulation or reset global history")


class FakeManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg


class FakeRobot:
    def __init__(self, count=12, joints=12):
        self.joint_names = [f"joint_{i}" for i in range(joints)]
        limits = torch.tensor([-2.0, 2.0]).repeat(count, joints, 1)
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros(count, joints),
            soft_joint_pos_limits=limits,
        )
        # Distinct sentinels make accidental writes to unselected rows visible.
        self.root_pose = torch.arange(count * 7, dtype=torch.float32).reshape(count, 7) + 100
        self.root_velocity = torch.arange(count * 6, dtype=torch.float32).reshape(count, 6) + 200
        self.joint_pos = torch.arange(count * joints, dtype=torch.float32).reshape(count, joints) + 300
        self.joint_vel = torch.arange(count * joints, dtype=torch.float32).reshape(count, joints) + 400
        self.writes = []
        self.reset = forbidden

    def _write(self, name, values, env_ids):
        assert env_ids is not None, "Every write must explicitly name the selected environments"
        assert values.shape[0] == env_ids.numel()
        self.writes.append((name, env_ids.clone()))
        getattr(self, name)[env_ids] = values

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        self._write("joint_pos", joint_pos, env_ids)
        self._write("joint_vel", joint_vel, env_ids)

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self._write("root_pose", pose, env_ids)

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        self._write("root_velocity", velocity, env_ids)


class FakeScene(dict):
    def __init__(self, robot, count):
        super().__init__(robot=robot)
        self.env_origins = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) * 10
        self.update = forbidden
        self.reset = forbidden


def make_env(later_joint_reset=None):
    count = 12
    robot = FakeRobot(count)
    return SimpleNamespace(
        num_envs=count,
        device="cpu",
        physics_dt=0.005,
        cfg=SimpleNamespace(decimation=4, events=SimpleNamespace(reset_robot_joints=later_joint_reset)),
        scene=FakeScene(robot, count),
        common_step_counter=219,
        episode_length_buf=torch.arange(count) + 50,
        reset_buf=torch.arange(count) % 2 == 0,
        reset_terminated=torch.arange(count) % 3 == 0,
        reset_time_outs=torch.arange(count) % 4 == 0,
        action_manager=SimpleNamespace(action=torch.arange(count * 12).reshape(count, 12).float(), reset=forbidden),
        observation_manager=SimpleNamespace(history=torch.arange(count * 24).reshape(count, 2, 12), reset=forbidden),
        sim=SimpleNamespace(step=forbidden, forward=forbidden, reset=forbidden,
                            current_time=1.095, current_time_step_index=219),
        step=forbidden,
        reset=forbidden,
    )


def make_bank():
    count = 6
    return {
        "root_pose_local": torch.tensor([[0.01 * i, -0.02 * i, 0.10 + 0.01 * i, 1., 0., 0., 0.]
                                         for i in range(count)]),
        "root_velocity_w": torch.arange(count * 6).reshape(count, 6).float() * 0.001,
        "joint_pos": torch.arange(count * 12).reshape(count, 12).float() * 0.01 - 0.3,
        "joint_vel": torch.arange(count * 12).reshape(count, 12).float() * 0.002 + 0.001,
        "state_id": torch.tensor([101, 109, 117, 205, 213, 221]),
        "pose_class": ["left", "right", "back", "left", "right", "back"],
        "metadata": {"physics": {"dt": 0.005, "decimation": 4}},
    }


def fake_quat(roll, pitch, yaw):
    quat = torch.zeros((roll.numel(), 4), dtype=roll.dtype, device=roll.device)
    quat[:, 0] = 1
    return quat


def fake_transform(pose, velocity, origins, yaw):
    # Coordinate-transform correctness belongs to the bank helper's own tests.
    transformed = pose.clone()
    transformed[:, :3] += origins
    return transformed, velocity.clone()


def load_class(bank):
    loader = Mock(return_value=bank)
    transform = Mock(side_effect=fake_transform)
    limits_validator = Mock()
    scope = {
        "math": math, "torch": torch, "ManagerTermBase": FakeManagerTermBase,
        "math_utils": SimpleNamespace(
            quat_from_euler_xyz=fake_quat,
            matrix_from_quat=lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1)),
        "reset_clearance_height": lambda rotation, margin: margin + 0.55,
        "load_recovery_state_bank": loader,
        "transform_bank_root": transform,
        "validate_bank_joint_limits": limits_validator,
    }
    exec(compile(ast.Module(body=_CLASS_NODES, type_ignores=[]), str(_SOURCE), "exec"), scope)
    return scope["RecoveryBankReset"], loader, transform, limits_validator


def global_snapshot(env):
    return {
        "common_step_counter": env.common_step_counter,
        "current_time": env.sim.current_time,
        "current_time_step_index": env.sim.current_time_step_index,
        "episode_length_buf": env.episode_length_buf.clone(),
        "reset_buf": env.reset_buf.clone(),
        "reset_terminated": env.reset_terminated.clone(),
        "reset_time_outs": env.reset_time_outs.clone(),
        "actions": env.action_manager.action.clone(),
        "observation_history": env.observation_manager.history.clone(),
        "env_origins": env.scene.env_origins.clone(),
    }


class RecoveryBankResetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260916)
        self.bank = make_bank()
        self.env = make_env()
        self.reset_class, self.loader, self.transform, self.limits_validator = load_class(self.bank)
        self.ids = torch.tensor([1, 7, 11])

    def construct(self, collection=False):
        cfg = SimpleNamespace(params={"collection_mode": collection, "bank_path": "unused-mocked-bank.pt"})
        return self.reset_class(cfg, self.env)

    def reset(self, term, collection=False, start=1., end=1., ids=None):
        term(self.env, self.ids if ids is None else ids,
             bank_path="unused-mocked-bank.pt", collection_mode=collection,
             bank_fraction_start=start, bank_fraction_end=end, curriculum_steps=1000)

    def assert_snapshot_equal(self, before):
        after = global_snapshot(self.env)
        for key, value in before.items():
            with self.subTest(unchanged=key):
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(after[key], value, rtol=0, atol=0)
                else:
                    self.assertEqual(after[key], value)

    def test_noncontiguous_ids_only_change_selected_root_and_joints(self):
        robot = self.env.scene["robot"]
        before = {name: getattr(robot, name).clone()
                  for name in ("root_pose", "root_velocity", "joint_pos", "joint_vel")}
        global_before = global_snapshot(self.env)
        term = self.construct()
        term.last_state_ids[:] = torch.arange(self.env.num_envs) + 1000
        previous_ids = term.last_state_ids.clone()
        self.reset(term)
        other = torch.ones(self.env.num_envs, dtype=torch.bool)
        other[self.ids] = False
        for name, old in before.items():
            torch.testing.assert_close(getattr(robot, name)[other], old[other], rtol=0, atol=0)
            self.assertTrue(torch.all(torch.any(getattr(robot, name)[self.ids] != old[self.ids], dim=1)))
        torch.testing.assert_close(term.last_state_ids[other], previous_ids[other], rtol=0, atol=0)
        self.assertEqual(len(robot.writes), 4)
        for _, actual_ids in robot.writes:
            torch.testing.assert_close(actual_ids, self.ids, rtol=0, atol=0)
        self.assert_snapshot_equal(global_before)

    def test_bank_samples_preserve_saved_joint_positions_and_velocities(self):
        term = self.construct()
        self.reset(term)
        robot = self.env.scene["robot"]
        for env_id in self.ids.tolist():
            source = torch.where(self.bank["state_id"] == term.last_state_ids[env_id])[0]
            self.assertEqual(source.numel(), 1)
            for name in ("joint_pos", "joint_vel"):
                torch.testing.assert_close(getattr(robot, name)[env_id], self.bank[name][source.item()], rtol=0, atol=0)
            expected_pose = self.bank["root_pose_local"][source.item()].clone()
            expected_pose[:3] += self.env.scene.env_origins[env_id]
            torch.testing.assert_close(robot.root_pose[env_id], expected_pose, rtol=0, atol=0)
            torch.testing.assert_close(robot.root_velocity[env_id], self.bank["root_velocity_w"][source.item()], rtol=0, atol=0)
        self.loader.assert_called_once_with("unused-mocked-bank.pt", robot.joint_names, "cpu", split="train")
        self.limits_validator.assert_called_once()

    def test_collection_mode_never_loads_or_samples_a_bank(self):
        self.loader.side_effect = AssertionError("Collection must work before any bank exists")
        term = self.construct(collection=True)
        before = global_snapshot(self.env)
        self.reset(term, collection=True)
        self.loader.assert_not_called()
        self.transform.assert_not_called()
        self.limits_validator.assert_not_called()
        self.assertIsNone(term.bank)
        self.assertTrue(torch.all(term.last_state_ids[self.ids] == -1))
        torch.testing.assert_close(self.env.scene["robot"].joint_vel[self.ids], torch.zeros(3, 12), rtol=0, atol=0)
        self.assert_snapshot_equal(before)

    def test_collection_only_construction_cannot_later_sample_bank(self):
        term = self.construct(collection=True)
        with self.assertRaisesRegex(RuntimeError, "collection-only"):
            self.reset(term, collection=False)
        self.assertEqual(self.env.scene["robot"].writes, [])

    def test_later_joint_reset_term_is_rejected(self):
        self.env = make_env(later_joint_reset=object())
        for collection in (False, True):
            with self.subTest(collection=collection), self.assertRaisesRegex(ValueError, "reset_robot_joints=None"):
                self.construct(collection)
        self.loader.assert_not_called()

    def test_missing_actual_left_right_or_back_class_is_rejected(self):
        original = self.bank["pose_class"]
        for missing in ("left", "right", "back"):
            self.bank["pose_class"] = ["collection_label" if value == missing else value for value in original]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, f"no validated {missing}"):
                self.construct()

    def test_invalid_fractions_fail_before_any_simulation_write(self):
        term = self.construct()
        for start, end in ((-.01, .5), (.5, 1.01), (.8, .2), (float("nan"), .5), (0., float("inf"))):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "Bank fractions"):
                self.reset(term, start=start, end=end)
        self.assertEqual(self.env.scene["robot"].writes, [])

    def test_zero_fraction_uses_rehearsal_not_bank(self):
        term = self.construct()
        self.reset(term, start=0., end=0.)
        self.transform.assert_not_called()
        self.assertTrue(torch.all(term.last_state_ids[self.ids] == -1))

    def test_empty_selection_is_noop(self):
        term = self.construct()
        before = global_snapshot(self.env)
        self.reset(term, ids=torch.tensor([], dtype=torch.long))
        self.assertEqual(self.env.scene["robot"].writes, [])
        self.transform.assert_not_called()
        self.assert_snapshot_equal(before)

    def test_physics_dt_or_decimation_mismatch_is_rejected(self):
        for field, wrong in (("dt", .01), ("decimation", 2)):
            original = self.bank["metadata"]["physics"][field]
            self.bank["metadata"]["physics"][field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs"):
                self.construct()
            self.bank["metadata"]["physics"][field] = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
