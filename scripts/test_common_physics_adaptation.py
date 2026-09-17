"""Bounded CPU-only guards; no simulator, training, model writes, or temp files."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

import train_common_physics_adaptation as entry


class Config(NS):
    def to_dict(self):
        def value(item):
            if isinstance(item, Config):
                return item.to_dict()
            if isinstance(item, dict):
                return {key: value(one) for key, one in item.items()}
            if isinstance(item, list):
                return [value(one) for one in item]
            return item
        return {key: value(item) for key, item in vars(self).items()}


def runtime_config(reference=False):
    return Config(scene=Config(robot=Config(spawn=Config(articulation_props=Config(enabled_self_collisions=reference)))),
                  actions=Config(joint_pos=Config(reference="nominal", clamp=reference)),
                  events=Config(add_base_mass=None if reference else Config(range=(-1., 3.)), base_com=None,
                                physics_material=Config(static=.8 if reference else (.6, 1.)),
                                push_robot=Config(interval=(4., 8.)), reset_base=Config(uniform=True)),
                  observations=Config(policy=Config(enable_corruption=True)),
                  commands=Config(standing_fraction=.3, speed=(-.5, 1.)), rewards=Config(original=1.5),
                  sim=Config(dt=.005), decimation=4)


class CommonPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_env = entry.read(entry.PARENT_CONFIG / "env.yaml")
        cls.parent_agent = entry.read(entry.PARENT_CONFIG / "agent.yaml")
        cls.reference = entry.reference_document()
        cls.parent = entry.parent_metadata()

    def documents(self, n=128, updates=100):
        env = entry.expected_environment(self.parent_env, self.parent_env, self.reference, n)
        agent = copy.deepcopy(self.parent_agent)
        agent.update(run_name="20260917-commonphysics-unit-test", max_iterations=str(updates),
                     load_run=entry.PARENT_RUN, load_checkpoint="model_3947.pt")
        return env, agent

    def check(self, env, agent, n=128, updates=100):
        return entry.check_documents(env, agent, self.parent_env, self.parent_agent, self.reference,
                                     n, updates, "20260917-commonphysics-unit-test")

    def test_actual_parent_full_state_and_reference_are_pinned(self):
        self.assertEqual(self.parent["sha256"], entry.PARENT_SHA)
        self.assertEqual(self.parent["adam_steps"], [79120] * 17)
        self.assertEqual(self.parent["optimizer_group"]["lr"], 1e-5)
        self.assertEqual(self.parent["tensor_count"], 68)
        sources = entry.all_sources()
        hashes = {row["path"]: row["sha256"] for row in sources}
        for path, digest in entry.EXTRA_FROZEN.items():
            self.assertEqual(hashes[str(path.resolve())], digest)

    def test_two_finite_budgets(self):
        for n, updates in ((16, 2), (128, 100)):
            env, agent = self.documents(n, updates)
            self.assertTrue(self.check(env, agent, n, updates)["full_actual_yaml_checked"])
        for n, updates in ((16, 100), (128, 2), (128, 200), (512, 100)):
            with self.assertRaises(AssertionError):
                entry.arguments_ok(n, updates, "20260917-commonphysics-unit-test")

    def test_actual_physical_delta_distinguishes_unchanged_material_com(self):
        env, agent = self.documents()
        guard = self.check(env, agent)
        self.assertEqual(set(guard["declared_physics_fields_actually_unchanged"]),
                         {"events.base_com", "events.physics_material"})
        paths = guard["actual_physics_changed_paths"]
        self.assertIn("scene.robot.spawn.articulation_props.enabled_self_collisions", paths)
        self.assertIn("events.add_base_mass", paths)
        self.assertTrue(any(path.startswith("actions.joint_pos") for path in paths))
        self.assertFalse(any(path.startswith("events.base_com") or path.startswith("events.physics_material") for path in paths))
        self.assertTrue(all(row["kind"] == "run_metadata" for row in guard["actual_yaml_differences"]
                            if row["path"].endswith("num_envs")))

    def test_full_environment_mutations_rejected(self):
        changes = [
            (("commands", "base_velocity", "ranges", "lin_vel_x"), ["-0.5", "1.2"]),
            (("commands", "base_velocity", "rel_standing_envs"), "0.1"),
            (("commands", "base_velocity", "class_type"), "speed_gated_command:GatedVelocityCommand"),
            (("observations", "policy", "enable_corruption"), "false"),
            (("events", "push_robot"), "null"),
            (("events", "reset_base"), "null"),
            (("rewards", "track_lin_vel_xy_exp", "weight"), "2.0"),
            (("scene", "robot", "spawn", "articulation_props", "enabled_self_collisions"), "false"),
            (("actions", "joint_pos", "reference"), "current"),
            (("actions", "joint_pos", "scale"), "0.5"),
            (("events", "physics_material", "params", "static_friction_range"), [".6", "1.0"]),
            (("sim", "dt"), "0.01"), (("log_dir",), "C:/wrong"),
        ]
        for keys, bad in changes:
            with self.subTest(keys=keys):
                env, agent = self.documents()
                node = env
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = bad
                with self.assertRaises(AssertionError):
                    self.check(env, agent)

    def test_ppo_parent_metadata_mutations_rejected(self):
        for key, value in (("resume", "false"), ("seed", "20260917"),
                           ("load_checkpoint", "model_3996.pt"), ("load_run", "wrong"),
                           ("max_iterations", "200"), ("logger", "wandb")):
            env, agent = self.documents()
            agent[key] = value
            with self.assertRaises(AssertionError):
                self.check(env, agent)
        env, agent = self.documents()
        agent["algorithm"]["learning_rate"] = "0.0003"
        with self.assertRaises(AssertionError):
            self.check(env, agent)

    def test_runtime_edit_does_not_replace_training_config(self):
        cfg, reference = runtime_config(), runtime_config(True)
        before, ref_before = copy.deepcopy(cfg.to_dict()), copy.deepcopy(reference.to_dict())
        entry.apply_physics(cfg, reference)
        after = cfg.to_dict()
        for key in ("commands", "rewards", "observations", "sim", "decimation"):
            self.assertEqual(before[key], after[key])
        for key in ("push_robot", "reset_base"):
            self.assertEqual(before["events"][key], after["events"][key])
        self.assertEqual(reference.to_dict(), ref_before)
        self.assertIsNot(cfg.actions.joint_pos, reference.actions.joint_pos)
        self.assertIsNot(cfg.events.physics_material, reference.events.physics_material)

    def test_runtime_refuses_play_like_config(self):
        for field in ("noise", "push"):
            cfg = runtime_config()
            if field == "noise":
                cfg.observations.policy.enable_corruption = False
            else:
                cfg.events.push_robot = None
            with self.assertRaises(AssertionError):
                entry.apply_physics(cfg, runtime_config(True))

    def test_actual_final_iteration_and_adam_budget_required(self):
        for updates in (2, 100):
            candidate = copy.deepcopy(self.parent)
            candidate.update(iter=3947 + updates - 1, adam_steps=[79120 + updates * 20] * 17, sha256="f" * 64)
            entry.check_final(candidate, self.parent, updates)
            for key, value in (("iter", 4047), ("adam_steps", [79120] * 17), ("tensor_count", 17),
                               ("model_shapes", {}), ("all_tensors_finite", False), ("sha256", self.parent["sha256"])):
                bad = copy.deepcopy(candidate)
                bad[key] = value
                with self.assertRaises(AssertionError):
                    entry.check_final(bad, self.parent, updates)

    def test_full_parent_tensor_comparison_rejects_lost_std_or_adam(self):
        expected = {"model": {"std": torch.ones(12)}, "optimizer": {"state": {0: {
            "step": torch.tensor(79120.), "exp_avg": torch.ones(12), "exp_avg_sq": torch.ones(12)}}}}
        entry.equal_tree(copy.deepcopy(expected), expected)
        for field in ("std", "adam", "step", "dtype"):
            actual = copy.deepcopy(expected)
            if field == "std":
                actual["model"]["std"][0] = 0.
            elif field == "adam":
                actual["optimizer"]["state"][0]["exp_avg"].zero_()
            elif field == "step":
                actual["optimizer"]["state"][0]["step"] += 1
            else:
                actual["model"]["std"] = actual["model"]["std"].double()
            with self.assertRaises(AssertionError):
                entry.equal_tree(actual, expected)

    def test_generated_official_trainer_has_only_three_hooks(self):
        old = entry.OFFICIAL.read_text(encoding="utf-8")
        source = entry.build_source(old)
        ast.parse(source)
        stripped = source.replace("    _common_configure(env_cfg, agent_cfg, log_dir)\n\n", "", 1)
        stripped = stripped.replace("    _common_verify_before(runner, env, log_dir)\n\n", "", 1)
        stripped = stripped.replace("    _common_finish(env, log_dir)\n\n", "", 1)
        self.assertEqual(old, stripped)
        self.assertNotIn("speed_gated", source)
        self.assertEqual(source.count("runner.learn("), 1)
        self.assertEqual(source.count("gym.make("), 1)

    def test_official_anchor_drift_rejected(self):
        with self.assertRaises(AssertionError):
            entry.build_source("no official anchors")

    def test_formal_cannot_use_missing_or_outside_smoke_evidence(self):
        for path in ("C:/bad/common_physics_training_result.json", "E:/IsaacLab/fake.json"):
            with self.assertRaises(AssertionError):
                entry.verify_smoke(path, [])

    def test_actual_first_smoke_metadata_json_transport(self):
        path = Path("E:/IsaacLab/repo/logs/rsl_rl/unitree_go2_flat/"
                    "2026-09-17_15-26-39_20260917-commonphysics16x2-smoke/"
                    "common_physics_training_result.json")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        actual = entry.checkpoint_metadata(Path(receipt["checkpoint"]))
        self.assertEqual(actual["sha256"],
                         "250a62617a8ef24eeed4c99180cfabf5bae3b85c62cd23ef49cd75748d328a1d")
        self.assertIsInstance(actual["optimizer_group"]["betas"], tuple)
        self.assertIsInstance(receipt["checkpoint_metadata"]["optimizer_group"]["betas"], list)
        self.assertNotEqual(actual, receipt["checkpoint_metadata"])
        self.assertEqual(entry.json_metadata(actual), receipt["checkpoint_metadata"])
        # Archive-only regression: supplying its old ledger tests receipt transport,
        # not authorization to run the changed trainer from this old smoke.
        self.assertEqual(entry.verify_smoke(path, receipt["source_snapshot"])["checkpoint_sha256"],
                         actual["sha256"])
        with self.assertRaisesRegex(AssertionError, "different source or parent"):
            entry.verify_smoke(path, entry.all_sources())

    def test_json_transport_keeps_checkpoint_value_mismatches(self):
        canonical = entry.json_metadata(self.parent)
        for field in ("lr", "betas", "adam_steps", "model_shapes", "sha256"):
            changed = copy.deepcopy(self.parent)
            if field == "lr":
                changed["optimizer_group"]["lr"] = 1e-4
            elif field == "betas":
                changed["optimizer_group"]["betas"] = (.8, .999)
            elif field == "adam_steps":
                changed[field][0] += 1
            elif field == "model_shapes":
                changed[field] = {}
            else:
                changed[field] = "f" * 64
            with self.subTest(field=field):
                self.assertNotEqual(entry.json_metadata(changed), canonical)
        with self.assertRaises(ValueError):
            entry.json_metadata({"lr": float("nan")})


if __name__ == "__main__":
    unittest.main()
