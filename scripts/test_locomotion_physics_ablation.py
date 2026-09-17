"""CPU-only single-factor guards; no simulator and no existing artifact edits."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import torch

import evaluate_locomotion_physics_ablation as entry
from test_locomotion_recovery_physics import config, runtime


class AblationTests(unittest.TestCase):
    def test_generated_program_only_changes_explicit_variant_choices(self):
        old = entry.retained.build_source()
        new = entry.build_source()
        expected = old.replace(
            'parser.add_argument("--physics_mode", required=True, choices=("original", "recovery"))',
            'parser.add_argument("--physics_mode", required=True, choices=("self_collision", "soft_clamp", "fixed_randomization"))', 1)
        self.assertEqual(new, expected)
        ast.parse(new)
        self.assertEqual(new.count("env.reset()"), 1)
        self.assertEqual(new.count("env.step("), 1)
        self.assertEqual(new.count("action = policy(obs)"), 1)
        self.assertNotIn("action_manager.reset", new)

    def test_old_metric_reset_and_schedule_functions_identical(self):
        old = ast.parse(entry.retained.TEMPLATE.read_text(encoding="utf-8"))
        new = ast.parse(entry.build_source())
        for name in ("_configure_diagnostic", "_phase", "_set_command", "_stance_summary", "_rest_stance_acceptance"):
            def function(tree):
                return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name))
            self.assertEqual(function(old), function(new), name)
        def acceptance(tree):
            return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in ("acceptance", "legacy_acceptance", "settled")
                            for target in node.targets)]
        self.assertEqual(acceptance(old), acceptance(new))

    def test_self_collision_only(self):
        cfg, ref = config(), config(True)
        before = entry.retained.physical_config(cfg)
        result = entry.configure_physics(cfg, "self_collision", ref)
        expected = copy.deepcopy(before)
        expected["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] = True
        self.assertEqual(result["after"], expected)
        self.assertEqual(result["action_semantics"], "original_nominal_unclamped")
        self.assertFalse(result["full_recovery_physics"])

    def test_soft_clamp_only(self):
        cfg, ref = config(), config(True)
        before = entry.retained.physical_config(cfg)
        result = entry.configure_physics(cfg, "soft_clamp", ref)
        expected = copy.deepcopy(before)
        expected["action"] = entry.retained.physical_config(ref)["action"]
        self.assertEqual(result["after"], expected)
        self.assertFalse(cfg.scene.robot.spawn.articulation_props.enabled_self_collisions)
        self.assertIsNot(cfg.actions.joint_pos, ref.actions.joint_pos)
        self.assertEqual(result["action_semantics"], "nominal_soft_clamped")

    def test_fixed_randomization_only(self):
        cfg, ref = config(), config(True)
        before = entry.retained.physical_config(cfg)
        result = entry.configure_physics(cfg, "fixed_randomization", ref)
        expected = copy.deepcopy(before)
        for key in ("add_base_mass", "base_com", "physics_material"):
            expected["events"][key] = entry.retained.physical_config(ref)["events"][key]
        self.assertEqual(result["after"], expected)
        self.assertFalse(cfg.scene.robot.spawn.articulation_props.enabled_self_collisions)
        self.assertEqual(result["action_semantics"], "original_nominal_unclamped")

    def test_reference_never_mutated_and_reset_sim_unchanged(self):
        for variant in entry.VARIANTS:
            cfg, ref = config(), config(True)
            before = entry.retained.physical_config(cfg)
            ref_before = entry.retained.serializable(ref)
            result = entry.configure_physics(cfg, variant, ref)
            self.assertEqual(ref_before, entry.retained.serializable(ref))
            for field in ("sim", "reset_base", "reset_robot_joints", "decimation", "terrain_physics_material"):
                self.assertEqual(result["after"][field], before[field])

    def test_unsupported_baseline_or_reference_fails_closed(self):
        for variant in entry.VARIANTS:
            bad = config()
            bad.scene.robot.spawn.articulation_props.enabled_self_collisions = True
            with self.assertRaises(ValueError):
                entry.configure_physics(bad, variant, config(True))
            bad_ref = config(True)
            bad_ref.scene.robot.actuators.kp = 900
            with self.assertRaises(ValueError):
                entry.configure_physics(config(), variant, bad_ref)
        with self.assertRaises(ValueError):
            entry.configure_physics(config(), "recovery", config(True))

    def test_interface_branch_is_action_semantics_not_physics_label(self):
        for variant in entry.VARIANTS:
            with mock.patch.object(entry.retained, "validate_interface", return_value={}) as check:
                result = entry.validate_interface("env", "actor", variant)
                check.assert_called_once_with("env", "actor", "recovery" if variant == "soft_clamp" else "original")
                self.assertEqual(result["physics_variant"], variant)
                self.assertNotIn("physics_mode", result)
                self.assertFalse(result["full_recovery_physics"])

    def execute_fake(self, variant, action):
        env, obs = runtime()
        record = entry.begin_step(env, obs, 0, variant)
        data, manager = env.scene["robot"].data, env.action_manager
        manager.prev_action = manager.action.clone()
        manager.action = action.clone()
        data.joint_pos_target = data.default_joint_pos + action * .25
        if variant == "soft_clamp":
            data.joint_pos_target.clamp_(min=data.soft_joint_pos_limits[:, :, 0], max=data.soft_joint_pos_limits[:, :, 1])
        return env, record

    def test_trace_never_calls_partial_factor_recovery_physics(self):
        targets = {}
        for variant in entry.VARIANTS:
            action = torch.full((1, 12), 8.)
            env, record = self.execute_fake(variant, action)
            result = entry.end_step(env, record, action, torch.tensor([False]))
            self.assertNotIn("physics_mode", record)
            self.assertNotIn("physics_mode", result)
            self.assertEqual(result["physics_variant"], variant)
            self.assertEqual(result["action_semantics"], entry.ACTION_SEMANTICS[variant])
            self.assertEqual(result["raw_action"], action.tolist())
            targets[variant] = result["executed_target"]
        self.assertEqual(targets["self_collision"], targets["fixed_randomization"])
        self.assertNotEqual(targets["self_collision"], targets["soft_clamp"])

    def test_history_command_target_done_mutations_fail(self):
        for variant in entry.VARIANTS:
            for mutation in ("history", "target", "done", "label"):
                action = torch.full((1, 12), .1)
                env, record = self.execute_fake(variant, action)
                done = torch.tensor([False])
                if mutation == "history":
                    env.action_manager.prev_action[0, 0] += .001
                elif mutation == "target":
                    env.scene["robot"].data.joint_pos_target[0, 0] += .001
                elif mutation == "done":
                    done[0] = True
                else:
                    record["action_semantics"] = "recovery"
                with self.assertRaises((RuntimeError, ValueError)):
                    entry.end_step(env, record, action, done)

    def test_cli_rejects_protocol_changes(self):
        args = NS(task=entry.TASK, zero_action=False, checkpoint=entry.CONTROL,
                  physics_mode="self_collision", stand_s=4., walk_s=8., stop_s=6.,
                  walk_speed=.8, lateral_speed=0., yaw_rate=0., push_speed=0., seed=20260909,
                  output_dir=entry.ROOT / "evaluations/never-created-ablation-unit-test")
        entry.validate_cli(args)
        for field, value in (("walk_speed", .81), ("seed", 20260910), ("stop_s", 3.),
                             ("push_speed", .1), ("physics_mode", "recovery")):
            bad = copy.copy(args)
            setattr(bad, field, value)
            with self.assertRaises(ValueError):
                entry.validate_cli(bad)

    def test_finish_keeps_old_acceptance_and_extra_drift(self):
        for variant in entry.VARIANTS:
            report = {"global": {"steps": 900}, "protocol_version": "stand_walk_stop_stance_geometry_v2",
                      "acceptance": {"walk_tracking": True, "no_reset": True}, "passed": True,
                      "settled_phase_stats": {"walk": {"vy_b_mean": -.126406, "yaw_rate_mean": .025564}}}
            old_acceptance = copy.deepcopy(report["acceptance"])
            args = NS(output_dir=entry.ROOT / "evaluations/not-created", physics_mode=variant,
                      lateral_speed=0., yaw_rate=0., push_speed=0.)
            records = [{"physics_variant": variant, "action_semantics": entry.ACTION_SEMANTICS[variant]}] * 900
            with mock.patch.object(Path, "open", mock.mock_open()), mock.patch.object(entry.retained, "sha", return_value="f" * 64):
                entry.finish_report(report, args, {"reference_checked": True}, {}, records, "generated")
            self.assertEqual(report["acceptance"], old_acceptance)
            self.assertTrue(report["passed"])
            self.assertFalse(report["retention_passed"])
            self.assertFalse(report["ablation_experiment"]["full_recovery_physics"])
            self.assertNotIn("retention_experiment", report)


if __name__ == "__main__":
    unittest.main()
