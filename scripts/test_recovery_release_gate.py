"""Simulator-free single-variable BankControl/BankRelease ablation regressions."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_math", ROOT / "src/go2_recovery/recovery_math.py")
math_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_module)
gate = math_module.stance_penalty_gate


def reward_class():
    tree = ast.parse((ROOT / "src/go2_recovery/recovery_mdp.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UncrossedStanceReward")
    scope = {"torch": torch, "ManagerTermBase": object,
             # Fake test geometry is already expressed in a root identity frame.
             "math_utils": SimpleNamespace(quat_apply_inverse=lambda q, v: v),
             "upright_error_squared": math_module.upright_error_squared,
             "normal_stance_geometry": math_module.normal_stance_geometry,
             "stance_alignment_penalty": math_module.stance_alignment_penalty,
             "stance_penalty_gate": gate}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<reward>", "exec"), scope)
    return scope[node.name]


class CountingSensor:
    def __init__(self, data):
        self.value, self.reads = data, 0

    @property
    def data(self):
        self.reads += 1
        return self.value


def make_reward(count=1):
    feet = torch.tensor([[[.20, .16, -.31], [.20, -.16, -.31],
                          [-.20, .16, -.31], [-.20, -.16, -.31]]]).repeat(count, 1, 1)
    root = torch.zeros(count, 3); root[:, 2] = .33
    quat = torch.zeros(count, 4); quat[:, 0] = 1
    gravity = torch.zeros(count, 3); gravity[:, 2] = -1
    data = SimpleNamespace(root_quat_w=quat, root_pos_w=root,
                           body_pos_w=torch.cat((feet, feet), dim=1) + root[:, None],
                           projected_gravity_b=gravity, joint_pos=torch.full((count, 12), .15),
                           default_joint_pos=torch.zeros(count, 12),
                           root_lin_vel_w=torch.zeros(count, 3), root_ang_vel_w=torch.zeros(count, 3))
    forces = torch.zeros(count, 5, 3); forces[:, :4, 2] = 10
    cls = reward_class()
    reward = cls.__new__(cls)
    reward.robot = SimpleNamespace(data=data)
    reward.sensor = CountingSensor(SimpleNamespace(net_forces_w=forces))
    reward.feet, reward.knees = list(range(4)), list(range(4, 8))
    reward.contact_feet, reward.base = list(range(4)), [4]
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(count, 3)))
    return reward, env


class GateMathTests(unittest.TestCase):
    def test_legacy_gate_is_bitwise_unchanged(self):
        cosine = torch.linspace(-1.5, 1.5, 10001)
        expected = ((cosine - .5) / .4).clamp(0, 1)
        self.assertTrue(torch.equal(gate(cosine), expected))

    def test_fallen_low_or_insufficient_support_are_zero(self):
        cosine = torch.tensor([-.99, .5, .84, .85, 1., 1., 1.])
        heights = torch.tensor([.33, .33, .33, .33, .239, .24, .33])
        forces = torch.full((7, 4), 10.)
        forces[-1] = torch.tensor([10., 5., 0., 0.])
        result = gate(cosine, heights, forces, mode="release")
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_supported_normal_stance_retains_full_penalty(self):
        result = gate(torch.ones(3), torch.tensor([.29, .32, .33]),
                      torch.tensor([[6., 6., 0., 0.], [10., 10., 10., 0.], [8., 8., 8., 8.]]), mode="release")
        self.assertTrue(torch.equal(result, torch.ones(3)))

    def test_ramps_multiply_and_support_requires_strict_force_threshold(self):
        cosine, height = torch.tensor([.9, .9], dtype=torch.float64), torch.tensor([.26, .26], dtype=torch.float64)
        forces = torch.tensor([[5.0001, 5.0001, 0., 0.], [5.0, 5.0001, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(gate(cosine, height, forces, mode="release"), torch.tensor([.25, 0.], dtype=torch.float64))

    def test_continuity_at_both_posture_and_height_thresholds(self):
        epsilon = 1e-7
        forces = torch.full((2, 4), 10., dtype=torch.float64)
        for threshold in (.85, .95):
            result = gate(torch.tensor([threshold-epsilon, threshold+epsilon], dtype=torch.float64),
                          torch.full((2,), .33, dtype=torch.float64), forces, mode="release")
            self.assertLess(float((result[1] - result[0]).abs()), 1e-5)
        for threshold in (.24, .28):
            result = gate(torch.ones(2, dtype=torch.float64),
                          torch.tensor([threshold-epsilon, threshold+epsilon], dtype=torch.float64), forces, mode="release")
            self.assertLess(float((result[1] - result[0]).abs()), 1e-5)

    def test_release_gate_bounded_and_scalar_tensor_supported(self):
        c, h = torch.meshgrid(torch.linspace(-1.1, 1.1, 73), torch.linspace(0, .6, 71), indexing="ij")
        result = gate(c, h, torch.full((*c.shape, 4), 10.), mode="release")
        self.assertTrue(bool(((result >= 0) & (result <= 1)).all()))
        self.assertEqual(gate(torch.tensor(1.), torch.tensor(.33), torch.ones(4)*10, mode="release").item(), 1.)

    def test_unknown_mode_missing_data_and_bad_shape_fail(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            gate(torch.ones(1), mode="typo")
        with self.assertRaisesRegex(ValueError, "requires"):
            gate(torch.ones(1), mode="release")
        with self.assertRaisesRegex(ValueError, "shapes"):
            gate(torch.ones(1), torch.ones(2), torch.ones(1, 4), mode="release")


class RewardIntegrationTests(unittest.TestCase):
    def test_old_penalty_outputs_match_original_formula_no_new_sensor_reads(self):
        reward, env = make_reward(101)
        data = reward.robot.data
        data.projected_gravity_b[:, 2] = torch.linspace(-1, 1, 101)
        data.joint_pos = torch.linspace(-.8, .8, 101*12).reshape(101, 12)
        data.body_pos_w[:, :, 1] += torch.linspace(-.25, .25, 101)[:, None]
        feet = data.body_pos_w[:, reward.feet] - data.root_pos_w[:, None]
        knees = data.body_pos_w[:, reward.knees] - data.root_pos_w[:, None]
        side, fore = torch.tensor([1.,-1.,1.,-1.]), torch.tensor([1.,1.,-1.,-1.])
        original_gate = ((-data.projected_gravity_b[:, 2] - .5) / .4).clamp(0, 1)
        for mode in ("penalty", "aligned_penalty"):
            crossing = ((.09 - feet[:, :, 1]*side).clamp(min=0)/.1).mean(1)
            crossing += ((.05 - knees[:, :, 1]*side).clamp(min=0)/.1).mean(1)
            crossing += ((.10 - feet[:, :, 0]*fore).clamp(min=0)/.2).mean(1)
            if mode == "aligned_penalty":
                crossing += math_module.stance_alignment_penalty(feet)
            expected = original_gate * (crossing + data.joint_pos.abs().mean(1))
            self.assertTrue(torch.equal(reward(env, mode), expected))
        self.assertEqual(reward.sensor.reads, 0)

    def test_release_only_changes_gate_and_normal_stance_penalty_is_equal(self):
        reward, env = make_reward(3)
        data = reward.robot.data
        # Normal, partly-upright/high, and upright but too low.
        data.projected_gravity_b[:, 2] = torch.tensor([-1., -.7, -1.])
        data.root_pos_w[2, 2] = .22
        old = reward(env, "aligned_penalty")
        new = reward(env, "aligned_release_penalty")
        self.assertGreater(old[0].item(), 0.)
        self.assertEqual(new[0].item(), old[0].item())
        self.assertGreater(old[1].item(), 0.)
        self.assertGreater(old[2].item(), 0.)
        self.assertEqual(new[1].item(), 0.)
        self.assertEqual(new[2].item(), 0.)

    def test_terminal_stand_still_requires_valid_geometry_four_feet_and_height(self):
        reward, env = make_reward(4)
        reward.sensor.value.net_forces_w[1, 3, 2] = 0
        reward.robot.data.body_pos_w[2, 0, 1] = -.2
        reward.robot.data.root_pos_w[3, 2] = .23
        result = reward(env, "stand", target_height=.33)
        self.assertGreater(result[0].item(), .99)
        self.assertTrue(torch.equal(result[1:], torch.zeros(3)))

    def test_unknown_reward_mode_fails(self):
        reward, env = make_reward()
        with self.assertRaisesRegex(ValueError, "mode"):
            reward(env, "aligned_release_typo")


class ConfigurationTests(unittest.TestCase):
    def test_control_release_differ_only_by_gate_mode(self):
        class FakeBank:
            def __init__(self):
                self.__post_init__()
            def __post_init__(self):
                self.events = SimpleNamespace(reset_base=SimpleNamespace(params={
                    "bank_fraction_start": .2, "bank_fraction_end": .6, "curriculum_steps": 12000}))
                self.rewards = SimpleNamespace(crossed_limbs=SimpleNamespace(weight=-4., params={"mode":"aligned_penalty"}),
                                               uncrossed_stand=SimpleNamespace(weight=12., params={"mode":"stand", "target_height":.33}),
                                               dof_pos_limits=SimpleNamespace(weight=-1.))
                self.scene = SimpleNamespace(num_envs=512, self_collisions=True)
                self.observations = SimpleNamespace(policy=SimpleNamespace(enable_corruption=True))
        tree = ast.parse((ROOT / "src/go2_recovery/recovery_env_cfg.py").read_text(encoding="utf-8"))
        nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n,ast.ClassDef) and
                 n.name.startswith(("UnitreeGo2RecoveryBankControl", "UnitreeGo2RecoveryBankRelease"))]
        for node in nodes: node.decorator_list = []
        scope = {"UnitreeGo2RecoveryBankEnvCfg": FakeBank}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<configs>", "exec"), scope)
        control = scope["UnitreeGo2RecoveryBankControlEnvCfg"]()
        release = scope["UnitreeGo2RecoveryBankReleaseEnvCfg"]()
        self.assertEqual(control.events.reset_base.params, {"bank_fraction_start":.6, "bank_fraction_end":.6, "curriculum_steps":12000})
        self.assertEqual(release.rewards.crossed_limbs.params["mode"], "aligned_release_penalty")
        release.rewards.crossed_limbs.params["mode"] = "aligned_penalty"
        self.assertEqual(vars(control), vars(release))
        self.assertEqual(FakeBank().events.reset_base.params["bank_fraction_start"], .2)
        for name in ("UnitreeGo2RecoveryBankControlEnvCfg_PLAY", "UnitreeGo2RecoveryBankReleaseEnvCfg_PLAY"):
            cfg = scope[name]()
            self.assertEqual(cfg.scene.num_envs, 1)
            self.assertFalse(cfg.observations.policy.enable_corruption)

    def test_four_registrations_reuse_unchanged_ppo_config(self):
        tree = ast.parse((ROOT / "src/go2_recovery/go2_registry_snapshot.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == "_variant")
        entries = []
        scope = {"gym":SimpleNamespace(register=lambda **kw: entries.append(kw)),
                 "agents":SimpleNamespace(__name__="test.agents"), "__name__":"test.go2"}
        exec(compile(ast.Module(body=[node],type_ignores=[]), "<registry>", "exec"),scope)
        self.assertEqual(len(entries), 4)
        for variant in ("Control", "Release"):
            for suffix in ("", "-Play"):
                name = f"Isaac-Recovery-Bank-{variant}-Flat-Unitree-Go2{suffix}-v0"
                self.assertIn(name, [entry["id"] for entry in entries])
        self.assertTrue(all(entry["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":UnitreeGo2RecoveryPPORunnerCfg") for entry in entries))


if __name__ == "__main__":
    unittest.main()
