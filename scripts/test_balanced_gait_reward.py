"""CPU-only contracts and exact installed-upstream semantics, without Isaac Sim."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from balanced_gait_reward import (
    OFFICIAL_SOURCE_SHA256,
    balanced_gait_duration_reward,
    command_gated_air_time_variance_penalty as reward,
    duration_variance_components as components,
    runtime_duration_variance_components as runtime_components,
)


OFFICIAL = Path("E:/IsaacLab/repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py")


def official_function():
    """Execute only the real pinned pure function AST, never import Isaac Sim."""
    data = OFFICIAL.read_bytes()
    if hashlib.sha256(data).hexdigest() != OFFICIAL_SOURCE_SHA256:
        raise AssertionError("Installed upstream reward source differs from inspected SHA")
    parsed = ast.parse(data, filename=str(OFFICIAL))
    found = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "air_time_variance_penalty"]
    if len(found) != 1:
        raise AssertionError("Expected one actual upstream timing variance function")
    node = copy.deepcopy(found[0])
    node.returns = None
    for argument in node.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(OFFICIAL), "exec"), scope)
    return scope["air_time_variance_penalty"]


def fake_runtime(air, contact, commands, ids=(4, 1, 5, 2)):
    # Sensor-body IDs deliberately differ from both foot order and joint IDs.
    names = ["base", "FR_foot", "RR_foot", "FL_calf", "FL_foot", "RL_foot"]
    data = SimpleNamespace(last_air_time=air, last_contact_time=contact)
    class FakeSensor:
        pass
    sensor = FakeSensor()
    sensor.cfg = SimpleNamespace(track_air_time=True)
    sensor.body_names, sensor.data = names, data
    def get_command(name):
        if name != "base_velocity":
            raise KeyError(name)
        return commands
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}),
                          command_manager=SimpleNamespace(get_command=get_command))
    cfg = SimpleNamespace(name="contact_forces", body_ids=list(ids))
    return env, cfg


class BalancedGaitRewardTests(unittest.TestCase):
    def inputs(self, n=1, dtype=torch.float32):
        air = torch.tensor([[.14, .34, .34, .14]], dtype=dtype).repeat(n, 1)
        contact = torch.tensor([[.34, .14, .14, .34]], dtype=dtype).repeat(n, 1)
        command = torch.tensor([[.5, 0., 0.]], dtype=dtype).repeat(n, 1)
        return air, contact, command

    def test_balanced_is_zero(self):
        air, contact, cmd = self.inputs()
        air.fill_(.24); contact.fill_(.24)
        raw, eligible, gated = components(air, contact, cmd)
        self.assertEqual(raw.tolist(), [0.]); self.assertEqual(gated.tolist(), [0.])
        self.assertEqual(eligible.tolist(), [True])

    def test_unequal_uses_sample_variance_without_weight_or_dt(self):
        raw, eligible, gated = components(*self.inputs(dtype=torch.float64))
        self.assertAlmostEqual(raw.item(), .02666666666666667, places=14)
        self.assertTrue(torch.equal(raw, gated)); self.assertTrue(eligible.item())
        self.assertAlmostEqual((-10 * .02 * gated).item(), -.005333333333333333, places=14)

    def test_stationary_turn_backward_lateral_disable_only_gated_term(self):
        air, contact, cmd = self.inputs(6)
        cmd[:] = torch.tensor([[0., 0., 0.], [.5, 0., .5], [.5, 0., -.5],
                               [-.5, 0., 0.], [.5, .2, 0.], [.5, -.2, 0.]])
        raw, eligible, gated = components(air, contact, cmd)
        self.assertTrue((raw > 0).all()); self.assertFalse(eligible.any())
        self.assertTrue(torch.equal(gated, torch.zeros(6)))

    def test_strict_command_boundaries(self):
        air, contact, cmd = self.inputs(8)
        cmd[:] = torch.tensor([[.1, 0., 0.], [.100001, 0., 0.], [.5, .1, 0.],
                               [.5, -.1, 0.], [.5, 0., .3], [.5, 0., -.3],
                               [.5, .099999, .299999], [.5, -.099999, -.299999]])
        self.assertEqual(components(air, contact, cmd)[1].tolist(),
                         [False, True, False, False, False, False, True, True])

    def test_clamp_is_upper_point_five_without_fabricated_lower_bound(self):
        air, contact, cmd = self.inputs(dtype=torch.float64)
        air[:] = torch.tensor([[0., .25, .5, 4.]], dtype=torch.float64)
        contact.zero_()
        raw = components(air, contact, cmd)[0]
        self.assertTrue(torch.equal(raw, torch.var(torch.tensor([[0., .25, .5, .5]], dtype=torch.float64), dim=1)))

    def test_zero_uninitialized_timers_are_not_fabricated_cycles(self):
        air, contact, cmd = self.inputs()
        air.zero_(); contact.zero_()
        raw, eligible, gated = components(air, contact, cmd)
        self.assertTrue(eligible.item()); self.assertEqual(raw.item(), 0.)
        self.assertEqual(gated.item(), 0.)

    def test_all_foot_permutations_invariant(self):
        import itertools
        values = self.inputs(dtype=torch.float64)
        want = components(*values)
        for permutation in itertools.permutations(range(4)):
            got = components(values[0][:, permutation], values[1][:, permutation], values[2])
            for a, b in zip(want, got):
                torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)

    def test_multiple_environment_independence_and_row_permutation(self):
        air, contact, cmd = self.inputs(4)
        air[1].fill_(.24); contact[1].fill_(.24)
        cmd[2].zero_(); air[3] *= .5
        want = components(air, contact, cmd)
        order = torch.tensor([3, 0, 2, 1])
        got = components(air[order], contact[order], cmd[order])
        for a, b in zip(want, got):
            self.assertTrue(torch.equal(a[order], b))
        for i in range(4):
            for a, b in zip(want, components(air[i:i+1], contact[i:i+1], cmd[i:i+1])):
                self.assertTrue(torch.equal(a[i:i+1], b))

    def test_does_not_mutate_inputs_or_return_input_alias(self):
        inputs = self.inputs(); copies = [v.clone() for v in inputs]
        outputs = components(*inputs)
        for before, value in zip(copies, inputs):
            self.assertTrue(torch.equal(before, value))
        for out in outputs:
            self.assertTrue(all(out.data_ptr() != v.data_ptr() for v in inputs))
        outputs[0].zero_()
        self.assertGreater(outputs[2].item(), 0.)

    def test_invalid_type_dtype_shape_rejected(self):
        for index in range(3):
            for bad in (None, [], torch.ones((1, 4 if index < 2 else 3), dtype=torch.int64),
                        torch.ones((1, 4 if index < 2 else 3), dtype=torch.bool)):
                args = list(self.inputs()); args[index] = bad
                with self.subTest(index=index, bad=repr(bad)):
                    with self.assertRaises((TypeError, ValueError)):
                        components(*args)
        for bad in (torch.ones(4), torch.ones(1, 3), torch.ones(1, 5), torch.ones(0, 4), torch.ones(1, 1, 4)):
            with self.assertRaises(ValueError):
                components(bad, *self.inputs()[1:])

    def test_mismatched_batch_or_dtype_rejected(self):
        air, contact, cmd = self.inputs()
        with self.assertRaises(ValueError): components(air.repeat(2, 1), contact, cmd)
        with self.assertRaises(ValueError): components(air.double(), contact, cmd)
        with self.assertRaises(ValueError): components(air, contact, cmd[:, :2])

    def test_nonfinite_rejected_even_when_ineligible(self):
        for index in range(3):
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                args = list(self.inputs()); args[2].zero_(); args[index][0, 0] = nonfinite
                with self.subTest(index=index, nonfinite=nonfinite):
                    with self.assertRaises(ValueError): components(*args)

    def test_negative_timer_rejected(self):
        for index in (0, 1):
            args = list(self.inputs()); args[index][0, 0] = -.001
            with self.assertRaises(ValueError): components(*args)

    def test_runtime_correct_noncontiguous_sensor_mapping(self):
        air = torch.tensor([[99., .34, .14, 99., .14, .34]])
        contact = torch.tensor([[99., .14, .34, 99., .34, .14]])
        cmd = self.inputs()[2]
        env, cfg = fake_runtime(air, contact, cmd)
        want = components(*self.inputs())
        before = (air.clone(), contact.clone(), cmd.clone(), copy.deepcopy(cfg.body_ids))
        for a, b in zip(want, runtime_components(env, cfg)):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(reward(env, cfg), want[2]))
        self.assertTrue(torch.equal(air, before[0]) and torch.equal(contact, before[1]) and torch.equal(cmd, before[2]))
        self.assertEqual(cfg.body_ids, before[3])
        cfg.body_ids.reverse()
        torch.testing.assert_close(reward(env, cfg), want[2], rtol=1e-6, atol=1e-8)

    def test_runtime_rejects_wrong_sensor_mapping(self):
        air, contact = torch.ones(1, 6), torch.ones(1, 6)
        env, cfg = fake_runtime(air, contact, self.inputs()[2])
        for ids in (slice(None), [0, 1, 2, 3], [4, 1, 5, 5], [4, 1, 5], [4, 1, 5, 6], [True, 1, 5, 2]):
            cfg.body_ids = ids
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError): reward(env, cfg)

    def test_runtime_requires_tracking(self):
        env, cfg = fake_runtime(torch.ones(1, 6), torch.ones(1, 6), self.inputs()[2])
        env.scene.sensors[cfg.name].cfg.track_air_time = False
        with self.assertRaises(ValueError): reward(env, cfg)

    def test_no_parameter_entry_caches_only_metadata_not_timer_values(self):
        air = torch.tensor([[99., .34, .14, 99., .14, .34]])
        contact = torch.tensor([[99., .14, .34, 99., .34, .14]])
        env, cfg = fake_runtime(air, contact, self.inputs()[2])
        sensor = env.scene.sensors[cfg.name]
        before_keys = set(vars(sensor)), set(vars(env)), set(vars(sensor.data))
        self.assertTrue(torch.equal(balanced_gait_duration_reward(env), reward(env, cfg)))
        air[:, cfg.body_ids] = .24; contact[:, cfg.body_ids] = .24
        self.assertEqual(balanced_gait_duration_reward(env).item(), 0.)
        self.assertEqual(before_keys, (set(vars(sensor)), set(vars(env)), set(vars(sensor.data))))

    def test_no_parameter_entry_rejects_missing_or_duplicate_foot_names(self):
        for names in (["base", "FR_foot", "RR_foot", "FL_calf", "FL_foot", "RL_calf"],
                      ["FL_foot", "FR_foot", "RR_foot", "RL_foot", "FL_foot", "base"]):
            env, cfg = fake_runtime(torch.ones(1, 6), torch.ones(1, 6), self.inputs()[2])
            env.scene.sensors[cfg.name].body_names = names
            with self.assertRaises(ValueError): balanced_gait_duration_reward(env)

    def test_exact_installed_official_function_semantics(self):
        upstream = official_function()
        generator = torch.Generator().manual_seed(1947)
        for dtype in (torch.float32, torch.float64):
            air = torch.rand((128, 6), generator=generator, dtype=dtype) * .9
            contact = torch.rand((128, 6), generator=generator, dtype=dtype) * .9
            cmd = self.inputs(128, dtype)[2]
            env, cfg = fake_runtime(air, contact, cmd)
            actual_raw = runtime_components(env, cfg)[0]
            expected_raw = upstream(env, cfg)
            self.assertTrue(torch.equal(actual_raw, expected_raw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
