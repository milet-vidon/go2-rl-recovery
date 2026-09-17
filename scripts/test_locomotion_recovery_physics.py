"""CPU-only tests for the isolated frozen locomotion retention adapter."""

import ast
from types import SimpleNamespace as NS
import unittest

import torch

import evaluate_locomotion_recovery_physics as entry


class Config(NS):
    def to_dict(self):
        return vars(self)


def reset_root_state_uniform():
    pass


def reset_joints_by_scale():
    pass


def RecoveryBankReset():
    pass


class ControlStepJointPositionAction:
    pass


class JointPositionAction:
    pass


def config(reference=False):
    action = Config(class_type=ControlStepJointPositionAction if reference else JointPositionAction,
                    reference="nominal", scale=.25, offset=0., use_default_offset=not reference, clip=None)
    material = Config(func=reset_root_state_uniform, params={
        "static_friction_range": (.8, .8) if reference else (.6, 1.),
        "dynamic_friction_range": (.6, .6) if reference else (.4, .8), "restitution_range": (0., 0.)})
    return Config(scene=Config(robot=Config(spawn=Config(articulation_props=Config(enabled_self_collisions=reference)),
                                           actuators=Config(kp=25., kd=.5)),
                              terrain=Config(physics_material=Config(static_friction=1.))),
                  sim=Config(dt=.005, gravity=(0., 0., -9.81), device="cpu", render=Config(mode="none"), render_interval=4),
                  decimation=4, actions=Config(joint_pos=action),
                  events=Config(add_base_mass=None if reference else Config(params={"range": (-1., 3.)}),
                                base_com=None, physics_material=material, push_robot=None, base_external_force_torque=None,
                                reset_base=Config(func=RecoveryBankReset if reference else reset_root_state_uniform, params={}),
                                reset_robot_joints=None if reference else Config(func=reset_joints_by_scale, params={})))


def runtime():
    q = torch.tensor([entry.DEFAULT_Q], dtype=torch.float32)
    limits = torch.stack((q - .5, q + .5), dim=-1)
    data = NS(default_joint_pos=q, default_joint_vel=torch.zeros_like(q),
              joint_pos=q.clone(), joint_vel=torch.zeros_like(q), joint_pos_target=q.clone(),
              root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
              projected_gravity_b=torch.tensor([[0., 0., -1.]]), soft_joint_pos_limits=limits)
    manager = NS(action=torch.zeros_like(q), prev_action=torch.zeros_like(q))
    env = NS(scene={"robot": NS(data=data)}, action_manager=manager,
             command_manager=NS(get_command=lambda name: torch.zeros(1, 3)))
    obs = {"policy": torch.cat((data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
                                torch.zeros(1, 3), data.joint_pos - q, data.joint_vel, manager.action), dim=1)}
    return env, obs


class AdapterTests(unittest.TestCase):
    def test_pinned_source_builds_without_simulator(self):
        source = entry.build_source()
        ast.parse(source)
        self.assertIn('--physics_mode', source)
        self.assertEqual(source.count('env.reset()'), 1)
        self.assertNotIn('_reset_policy_history', source)

    def test_old_metric_and_reset_functions_unchanged(self):
        old = ast.parse(entry.TEMPLATE.read_text(encoding="utf-8"))
        new = ast.parse(entry.build_source())
        for name in ("_configure_diagnostic", "_phase", "_stance_summary", "_rest_stance_acceptance"):
            left = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right), name)
        def acceptance(tree):
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in ("acceptance", "legacy_acceptance", "settled") for t in n.targets)]
        self.assertEqual(acceptance(old), acceptance(new))

    def test_actor_called_once_and_no_extra_step(self):
        source = entry.build_source()
        self.assertEqual(source.count('action = policy(obs)'), 1)
        self.assertEqual(source.count('env.step('), 1)
        self.assertNotIn('manager.action.fill_', source)
        self.assertNotIn('action_manager.reset', source)

    def test_bad_anchor_fails_closed(self):
        for source in ("", "xx"):
            with self.assertRaises(ValueError):
                entry.replace_once(source, "x", "y")

    def test_config_slice_and_infinity_are_explicit_json_evidence(self):
        value = entry.serializable({"joint_ids": slice(None), "limit": float("inf")})
        self.assertEqual(value["joint_ids"], {"configuration_slice": [None, None, None]})
        self.assertEqual(value["limit"], {"configuration_nonfinite_number": "inf"})
        entry.json.dumps(value, allow_nan=False)

    def test_original_config_entirely_unchanged(self):
        cfg = config()
        before = entry.serializable(cfg)
        evidence = entry.configure_physics(cfg, "original")
        self.assertEqual(before, entry.serializable(cfg))
        self.assertEqual(evidence["before"], evidence["after"])
        self.assertFalse(evidence["reference_checked"])

    def test_recovery_changes_only_declared_fields(self):
        cfg, ref = config(), config(True)
        before = entry.physical_config(cfg)
        ref_before = entry.serializable(ref)
        result = entry.configure_physics(cfg, "recovery", ref)
        self.assertTrue(result["reference_checked"])
        after = entry.physical_config(cfg)
        self.assertEqual(before["reset_base"], after["reset_base"])
        self.assertEqual(before["reset_robot_joints"], after["reset_robot_joints"])
        self.assertEqual(before["sim"], after["sim"])
        self.assertEqual(ref_before, entry.serializable(ref))
        self.assertIsNot(cfg.actions.joint_pos, ref.actions.joint_pos)
        self.assertIsNot(cfg.events.physics_material, ref.events.physics_material)

    def test_bank_reset_not_accepted_as_ordinary_reset(self):
        with self.assertRaisesRegex(ValueError, 'ordinary upright reset'):
            entry.configure_physics(config(True), "original")

    def test_changed_reference_physics_rejected(self):
        for mutate in (lambda c: setattr(c.sim, "dt", .01),
                       lambda c: setattr(c.scene.robot.actuators, "kp", 35.),
                       lambda c: setattr(c.actions.joint_pos, "reference", "current"),
                       lambda c: setattr(c.events, "add_base_mass", Config(foo=1)),
                       lambda c: setattr(c.scene.robot.spawn.articulation_props, "enabled_self_collisions", False)):
            reference = config(True)
            mutate(reference)
            with self.assertRaises(ValueError):
                entry.configure_physics(config(), "recovery", reference)

    def test_begin_reads_native_observation_without_mutation(self):
        env, obs = runtime()
        before = obs["policy"].clone()
        record = entry.begin_step(env, obs, 0, "original")
        self.assertTrue(torch.equal(obs["policy"], before))
        self.assertTrue(torch.equal(record["observation"], before))
        self.assertNotEqual(record["observation"].data_ptr(), obs["policy"].data_ptr())

    def test_forged_history_or_observation_rejected(self):
        env, obs = runtime()
        obs["policy"][0, 36] = 1.
        with self.assertRaises(RuntimeError):
            entry.begin_step(env, obs, 0, "original")
        env, obs = runtime()
        env.action_manager.prev_action[0, 0] = 1.
        with self.assertRaises(RuntimeError):
            entry.begin_step(env, obs, 0, "original")

    def execute_fake(self, mode, action):
        env, obs = runtime()
        record = entry.begin_step(env, obs, 0, mode)
        manager, data = env.action_manager, env.scene["robot"].data
        manager.prev_action = manager.action.clone()
        manager.action = action.clone()
        data.joint_pos_target = data.default_joint_pos + action * .25
        if mode == "recovery":
            data.joint_pos_target.clamp_(min=data.soft_joint_pos_limits[:, :, 0], max=data.soft_joint_pos_limits[:, :, 1])
        return env, record

    def test_original_does_not_clamp_but_recovery_does(self):
        action = torch.full((1, 12), 8.)
        targets = []
        for mode in ("original", "recovery"):
            env, record = self.execute_fake(mode, action)
            row = entry.end_step(env, record, action, torch.tensor([False]))
            targets.append(row["executed_target"])
            self.assertEqual(row["actual_action"], action.tolist())
        self.assertNotEqual(targets[0], targets[1])

    def test_target_history_action_reset_mutations_rejected(self):
        action = torch.full((1, 12), .2)
        for field in ("target", "history", "action", "done"):
            env, record = self.execute_fake("recovery", action)
            dones = torch.tensor([False])
            if field == "target":
                env.scene["robot"].data.joint_pos_target[0, 0] += .001
            elif field == "history":
                env.action_manager.prev_action[0, 0] += .001
            elif field == "action":
                env.action_manager.action[0, 0] += .001
            else:
                dones[0] = True
            with self.assertRaises(RuntimeError, msg=field):
                entry.end_step(env, record, action, dones)

    def test_straight_drift_uses_both_old_strict_thresholds(self):
        for vy, yaw, expected in ((.119, .149, True), (-.119, -.149, True),
                                  (.12, 0., False), (0., .15, False),
                                  (-.12, 0., False), (0., -.15, False)):
            with self.subTest(vy=vy, yaw=yaw):
                self.assertIs(entry.straight_drift_acceptance(
                    {"vy_b_mean": vy, "yaw_rate_mean": yaw}, 0., 0., 0.), expected)

    def test_straight_drift_is_not_applied_to_push_or_turn_or_lateral(self):
        for lateral, yaw, push in ((0., 0., .5), (0., .5, 0.), (.2, 0., 0.)):
            self.assertIsNone(entry.straight_drift_acceptance(
                {"vy_b_mean": 1., "yaw_rate_mean": 1.}, lateral, yaw, push))


if __name__ == "__main__":
    unittest.main()
