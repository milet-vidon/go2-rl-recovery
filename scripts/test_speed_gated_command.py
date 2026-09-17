"""CPU mock lifecycle tests; no Isaac Sim import or training."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch


class FakeUniform:
    def __init__(self, cfg, env):
        self.cfg, self._env, self.num_envs, self.device = cfg, env, env.num_envs, "cpu"
        self.robot = env.robot
        self.vel_command_b = torch.zeros(self.num_envs, 3)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool)

    def reset(self, env_ids):
        self._resample_command(env_ids)
        return {}

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids))
        for i, bounds in enumerate(((-0.5, 1.0), (-0.2, 0.2), (-0.6, 0.6))):
            self.vel_command_b[env_ids, i] = r.uniform_(*bounds)
        self.is_standing_env[env_ids] = r.uniform_(0, 1) <= 0.3


fake_module = ModuleType("isaaclab.envs.mdp.commands.velocity_command")
fake_module.UniformVelocityCommand = FakeUniform
source = Path(__file__).with_name("speed_gated_command.py")
spec = importlib.util.spec_from_file_location("gated_under_test", source)
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
    spec.loader.exec_module(module)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("E:/IsaacLab/artifacts/recovery-20260917") / ("speedgate-cpu-" + uuid.uuid4().hex)
        self.path.mkdir()
        n = 64
        self.cfg = SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(-0.5, 1.0)), heading_command=False, rel_standing_envs=0.3)
        self.env = SimpleNamespace(num_envs=n, step_dt=0.02, common_step_counter=0, cfg=SimpleNamespace(log_dir=str(self.path)),
            robot=SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3))))
        self.command = module.GatedVelocityCommand(self.cfg, self.env)
        self.ids = torch.arange(n)

    def test_channels_and_rng_preserved(self):
        ordinary = FakeUniform(self.cfg, self.env)
        torch.manual_seed(42)
        ordinary._resample_command(self.ids)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        self.command._resample_command(self.ids)
        original = ordinary.vel_command_b
        positive = (original[:, 0] > 0) & ~ordinary.is_standing_env
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertTrue(torch.equal(ordinary.is_standing_env, self.command.is_standing_env))
        self.assertTrue(torch.equal(original[:, 1:], self.command.vel_command_b[:, 1:]))
        self.assertTrue(torch.equal(original[~positive], self.command.vel_command_b[~positive]))
        self.assertTrue(torch.equal(original[positive, 0] * 0.8, self.command.vel_command_b[positive, 0]))

    def test_warmup_exact_and_finite(self):
        for _ in range(25):
            self.command._update_metrics()
        self.assertEqual(int(self.command.samples.sum()), 0)
        self.command._update_metrics()
        self.assertTrue(torch.equal(self.command.samples, torch.ones(64, dtype=torch.long)))
        self.assertTrue(torch.equal(self.command.lin_sum, torch.ones(64, dtype=torch.float64)))
        self.env.robot.data.root_lin_vel_b[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.command._update_metrics()

    def fill_window(self):
        self.command.samples[:] = 175
        self.command.lin_sum[:] = self.command.ang_sum[:] = 175 * 0.9
        self.command.vel_command_b[:, 0] = 0.7
        self.command.is_standing_env[:] = False

    def test_reset_never_promotes_and_clears(self):
        self.fill_window()
        self.command.reset(self.ids)
        self.assertEqual(self.command.course.snapshot()["cap"], 0.8)
        self.assertEqual(self.command.course.snapshot()["eligible_count"], 0)
        self.assertEqual(self.command.discarded_reset_windows, 64)
        self.assertEqual(self.command.windows, [])
        self.assertEqual(int(self.command.samples.sum()), 0)

    def test_natural_complete_windows_promote_only_one_level(self):
        self.fill_window()
        self.command._resample_command(self.ids)
        state = self.command.course.snapshot()
        self.assertEqual(state["cap"], 0.9)
        self.assertEqual(state["eligible_count"], 32)
        self.assertEqual(state["rejected_by_reason"]["different_cap"], 32)
        self.assertEqual(len(self.command.windows), 64)
        self.assertEqual(int(self.command.samples.sum()), 0)
        self.assertTrue(torch.equal(self.command.sample_cap, torch.full((64,), 0.9, dtype=torch.float64)))

    def test_standing_and_short_windows_never_promote(self):
        self.fill_window()
        self.command.is_standing_env[:32] = True
        self.command.samples[32:] = 174
        self.command._resample_command(self.ids)
        self.assertEqual(self.command.course.snapshot()["eligible_count"], 0)

    def test_official_generation_compiles_without_simulator(self):
        import train_speed_gated_curriculum as entry
        source_text = entry.OFFICIAL.read_text(encoding="utf-8")
        ast.parse(entry.build_source(source_text))
        with self.assertRaises(AssertionError):
            entry.build_source(source_text.replace("    # run training\n", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
