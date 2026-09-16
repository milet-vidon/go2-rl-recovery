"""CPU-only full-schema regression tests; no simulator, torch or file writes."""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smith_standweight_guard", ROOT / "scripts/check_smith_standweight_config.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
PARENT = ROOT / "configs/20260916-smithnominal128x2000"
CANDIDATE = ROOT / "configs/_in_memory_standweight_candidate"


def make_candidate(env, agent, num_envs=16, iterations=2, weight=30):
    env, agent = copy.deepcopy(env), copy.deepcopy(agent)
    env["scene"]["num_envs"] = str(num_envs)
    env["scene"]["terrain"]["num_envs"] = str(num_envs)
    env["rewards"]["smith_stand"]["weight"] = f"{weight}.0"
    env["log_dir"] = "E:/IsaacLab/artifacts/in-memory-standweight/run"
    env["sim"]["log_dir"] = "E:/IsaacLab/artifacts/in-memory-standweight/runtime"
    agent["run_name"] = "in_memory_standweight_test"
    agent["max_iterations"] = str(iterations)
    agent["load_run"] = "2026-09-16_19-02-40_20260916-smithnominal128x2000"
    agent["load_checkpoint"] = "model_1999.pt"
    return env, agent


class SmithStandWeightConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_env = GUARD.read(PARENT / "env.yaml")
        cls.parent_agent = GUARD.read(PARENT / "agent.yaml")

    def setUp(self):
        self.env, self.agent = make_candidate(self.parent_env, self.parent_agent)

    def run_guard(self, num_envs=16, iterations=2, weight=30):
        documents = {PARENT / "env.yaml": self.parent_env,
                     PARENT / "agent.yaml": self.parent_agent,
                     CANDIDATE / "env.yaml": self.env,
                     CANDIDATE / "agent.yaml": self.agent}

        def read_copy(path):
            self.assertIn(Path(path), documents)
            return copy.deepcopy(documents[Path(path)])

        with patch.object(GUARD, "read", side_effect=read_copy), contextlib.redirect_stdout(io.StringIO()):
            GUARD.check(CANDIDATE, num_envs, iterations, weight)

    def reject_env(self, path, value):
        node = self.env
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with self.assertRaises((AssertionError, KeyError)):
            self.run_guard()

    def test_both_arms_smoke_and_formal_pass(self):
        for num_envs, iterations in ((16, 2), (128, 1000)):
            for weight in (10, 30):
                with self.subTest(num_envs=num_envs, iterations=iterations, weight=weight):
                    self.env, self.agent = make_candidate(self.parent_env, self.parent_agent,
                                                          num_envs, iterations, weight)
                    self.run_guard(num_envs, iterations, weight)

    def test_parent_is_actual_complete_schema(self):
        self.assertEqual(self.parent_env["scene"]["num_envs"], "128")
        self.assertEqual(self.parent_agent["load_checkpoint"], "model_0.pt")
        self.assertIn("observations", self.parent_env)
        self.assertIn("robot", self.parent_env["scene"])
        self.assertIn("algorithm", self.parent_agent)

    def test_weight_must_match_arm_and_be_finite(self):
        for value in ("10.0", "20", "NaN", "Infinity", "-30", "null"):
            with self.subTest(value=value):
                self.setUp()
                self.env["rewards"]["smith_stand"]["weight"] = value
                with self.assertRaisesRegex(AssertionError, "Wrong smith_stand weight"):
                    self.run_guard()

    def test_unsupported_budgets_and_weights_rejected(self):
        for n, it, w in ((512, 2, 30), (16, 1000, 30), (128, 2, 30),
                         (128, 2000, 30), (16, 2, 20), (16, 2, True)):
            with self.subTest(n=n, it=it, w=w):
                with self.assertRaises(AssertionError):
                    self.run_guard(n, it, w)

    def test_env_and_terrain_counts_checked_separately(self):
        for path in (("scene", "num_envs"), ("scene", "terrain", "num_envs")):
            with self.subTest(path=path):
                self.setUp()
                self.reject_env(path, "128")

    def test_physics_and_actuator_drift_rejected(self):
        for path, value in ((("sim", "dt"), "0.01"),
                            (("decimation",), "8"),
                            (("scene", "robot", "actuators", "base_legs", "effort_limit"), "180.0"),
                            (("scene", "robot", "actuators", "base_legs", "stiffness"), "100.0")):
            with self.subTest(path=path):
                self.setUp()
                self.reject_env(path, value)

    def test_action_reference_scale_and_new_key_rejected(self):
        for key, value in (("reference", "current"), ("scale", "0.5"), ("new_filter", "true")):
            with self.subTest(key=key):
                self.setUp()
                self.reject_env(("actions", "joint_pos", key), value)

    def test_other_rewards_and_stand_formula_rejected(self):
        for path, value in ((("rewards", "smith_roll", "weight"), "30.0"),
                            (("rewards", "smith_stand", "params", "target_height"), "0.4"),
                            (("rewards", "smith_stand", "params", "mode"), "total"),
                            (("rewards", "smith_stand", "func"), "different:Reward"),
                            (("rewards", "dof_pos_limits", "weight"), "0.0"),
                            (("rewards", "action_rate_l2", "weight"), "0.0")):
            with self.subTest(path=path):
                self.setUp()
                self.reject_env(path, value)

    def test_bank_and_observation_noise_drift_rejected(self):
        for path, value in ((("events", "reset_base", "params", "bank_fraction_start"), "0.0"),
                            (("events", "reset_base", "params", "bank_path"), "E:/changed.npz"),
                            (("observations", "policy", "enable_corruption"), "false"),
                            (("seed",), "43")):
            with self.subTest(path=path):
                self.setUp()
                self.reject_env(path, value)

    def test_missing_and_unknown_schema_fields_rejected(self):
        del self.env["sim"]["dt"]
        with self.assertRaisesRegex(AssertionError, "Unexpected environment/reward change"):
            self.run_guard()
        self.setUp()
        self.reject_env(("unplanned_curriculum",), "true")

    def test_resume_checkpoint_and_iteration_budget_exact(self):
        for key, value in (("resume", "false"), ("load_run", ".*"),
                           ("load_checkpoint", "model_.*.pt"), ("max_iterations", "1000"),
                           ("run_name", "null")):
            with self.subTest(key=key):
                self.setUp()
                self.agent[key] = value
                with self.assertRaises(AssertionError):
                    self.run_guard()

    def test_ppo_noise_seed_and_unknown_agent_keys_rejected(self):
        for path, value in ((("algorithm", "gamma"), "0.99"),
                            (("algorithm", "learning_rate"), "0.001"),
                            (("policy", "init_noise_std"), "0.5"),
                            (("policy", "actor_hidden_dims"), ["256", "128", "128"]),
                            (("seed",), "43"), (("checkpoint",), "model_1999.pt")):
            with self.subTest(path=path):
                self.setUp()
                node = self.agent
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                with self.assertRaisesRegex(AssertionError, "Unexpected PPO/resume configuration change"):
                    self.run_guard()

    def test_log_paths_remain_on_e_and_null_sim_path_allowed(self):
        self.env["sim"]["log_dir"] = "null"
        self.run_guard()
        for value in ("C:/logs", "relative/logs", "E:/logs/../other", "null"):
            with self.subTest(value=value):
                self.setUp()
                self.reject_env(("log_dir",), value)

    def test_input_documents_are_not_mutated(self):
        before = copy.deepcopy((self.parent_env, self.parent_agent, self.env, self.agent))
        self.run_guard()
        self.assertEqual(before, (self.parent_env, self.parent_agent, self.env, self.agent))

    def test_python_yaml_tag_is_data_not_code(self):
        payload = '!!python/object/apply:os.system ["must-not-execute"]'
        with patch.object(Path, "read_text", return_value=payload), patch("os.system") as system:
            self.assertEqual(GUARD.read(CANDIDATE / "tagged.yaml"), ["must-not-execute"])
            system.assert_not_called()


if __name__ == "__main__":
    unittest.main()
