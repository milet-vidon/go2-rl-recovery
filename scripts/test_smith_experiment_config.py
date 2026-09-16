"""CPU regression tests for the reward-only experiment's full YAML guard.

Uses the tracked complete nominal-control schemas, never a reduced mock schema.
The archived local smoke configuration is also exercised when present. Mocked
reads always return deep copies because the guard normalizes its input objects.
No simulator is imported, no saved configuration is edited, and no output files
are written.
"""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smith_experiment_config_guard", ROOT / "scripts/check_smith_experiment_config.py")
assert SPEC and SPEC.loader
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
BASELINE = ROOT / "configs/20260916-target128x2000-nominal"
CANDIDATE = ROOT / "configs/_in_memory_smith_test_candidate"
LOCAL_SMOKE = (ROOT.parent / "repo/logs/rsl_rl/unitree_go2_recovery"
               / "2026-09-16_18-53-36_20260916_smithnominal_smoke16x2/params")


def candidate_from_control(control_env, control_agent, num_envs=16, iterations=2):
    """Independent expected delta, not derived from guard.REMOVED or its logic."""
    env, agent = copy.deepcopy(control_env), copy.deepcopy(control_agent)
    for name in ("upright_and_height", "orientation_progress", "recovery_success",
                 "stable_stand", "static_stance", "low_height_support", "low_height_lift",
                 "conditional_stand_posture", "uncrossed_stand", "crossed_limbs"):
        env["rewards"][name] = "null"
    reward_func = ("isaaclab_tasks.manager_based.locomotion.velocity.config.go2."
                   "recovery_smith_mdp:SmithRecoveryReward")
    env["rewards"]["smith_roll"] = {
        "func": reward_func, "params": {"mode": "roll", "target_height": "0.32"},
        "weight": "10.0"}
    env["rewards"]["smith_stand"] = {
        "func": reward_func, "params": {"mode": "stand", "target_height": "0.32"},
        "weight": "10.0"}
    env["scene"]["num_envs"] = str(num_envs)
    env["scene"]["terrain"]["num_envs"] = str(num_envs)
    env["sim"]["log_dir"] = "E:/IsaacLab/artifacts/in-memory-test/runtime"
    env["log_dir"] = "E:/IsaacLab/artifacts/in-memory-test/run"
    agent["run_name"] = "smith_in_memory_config_test"
    agent["max_iterations"] = str(iterations)
    return env, agent


class SmithExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control_env = GUARD.read(BASELINE / "env.yaml")
        cls.control_agent = GUARD.read(BASELINE / "agent.yaml")

    def setUp(self):
        self.env, self.agent = candidate_from_control(self.control_env, self.control_agent)

    def run_guard(self, num_envs=16, iterations=2):
        documents = {
            BASELINE / "env.yaml": self.control_env,
            BASELINE / "agent.yaml": self.control_agent,
            CANDIDATE / "env.yaml": self.env,
            CANDIDATE / "agent.yaml": self.agent,
        }

        def read_copy(path):
            path = Path(path)
            self.assertIn(path, documents, f"Unexpected config read: {path}")
            return copy.deepcopy(documents[path])

        with patch.object(GUARD, "read", side_effect=read_copy), contextlib.redirect_stdout(io.StringIO()):
            GUARD.check(CANDIDATE, num_envs, iterations)

    def test_complete_schema_smoke_16_envs_two_iterations_passes(self):
        self.run_guard()
        self.assertIn("robot", self.env["scene"])
        self.assertIn("observations", self.env)
        self.assertIn("rewards", self.control_env)

    def test_formal_128_envs_2000_iterations_passes(self):
        self.env, self.agent = candidate_from_control(
            self.control_env, self.control_agent, num_envs=128, iterations=2000)
        self.run_guard(num_envs=128, iterations=2000)

    @unittest.skipUnless((LOCAL_SMOKE / "env.yaml").is_file() and
                         (LOCAL_SMOKE / "agent.yaml").is_file(),
                         "Local archived smoke is optional; tracked full-schema fixtures remain mandatory")
    def test_real_saved_smoke_yaml_copies_pass(self):
        self.env = GUARD.read(LOCAL_SMOKE / "env.yaml")
        self.agent = GUARD.read(LOCAL_SMOKE / "agent.yaml")
        self.run_guard()

    def test_wrong_action_reference_rejected(self):
        self.env["actions"]["joint_pos"]["reference"] = "current"
        with self.assertRaisesRegex(AssertionError, "Non-reward environment changed"):
            self.run_guard()

    def test_wrong_action_scale_rejected(self):
        self.env["actions"]["joint_pos"]["scale"] = "0.50"
        with self.assertRaisesRegex(AssertionError, "Non-reward environment changed"):
            self.run_guard()

    def test_motor_torque_limit_changes_rejected(self):
        for field in ("effort_limit", "saturation_effort", "effort_limit_sim"):
            with self.subTest(field=field):
                self.setUp()
                self.env["scene"]["robot"]["actuators"]["base_legs"][field] = "180.0"
                with self.assertRaisesRegex(AssertionError, "Non-reward environment changed"):
                    self.run_guard()

    def test_bank_mixture_changes_rejected(self):
        for field in ("bank_fraction_start", "bank_fraction_end"):
            with self.subTest(field=field):
                self.setUp()
                self.env["events"]["reset_base"]["params"][field] = "0.2"
                with self.assertRaisesRegex(AssertionError, "Non-reward environment changed"):
                    self.run_guard()

    def test_wrong_bank_path_rejected(self):
        self.env["events"]["reset_base"]["params"]["bank_path"] = "E:/IsaacLab/another-bank"
        with self.assertRaisesRegex(AssertionError, "Non-reward environment changed"):
            self.run_guard()

    def test_ppo_gamma_change_rejected(self):
        self.agent["algorithm"]["gamma"] = "0.99"
        with self.assertRaisesRegex(AssertionError, "PPO/bootstrap changed"):
            self.run_guard()

    def test_bootstrap_change_rejected(self):
        self.agent["load_run"] = "failed_policy_warm_start"
        with self.assertRaisesRegex(AssertionError, "PPO/bootstrap changed"):
            self.run_guard()

    def test_soft_limit_regularizer_change_rejected(self):
        self.env["rewards"]["dof_pos_limits"]["weight"] = "-0.1"
        with self.assertRaisesRegex(AssertionError, "Unexpected reward configuration"):
            self.run_guard()

    def test_retained_crossed_limbs_penalty_rejected(self):
        self.env["rewards"]["crossed_limbs"] = copy.deepcopy(
            self.control_env["rewards"]["crossed_limbs"])
        with self.assertRaisesRegex(AssertionError, "Unexpected reward configuration"):
            self.run_guard()

    def test_wrongly_named_new_reward_rejected(self):
        self.env["rewards"]["smith_recovery"] = self.env["rewards"].pop("smith_roll")
        with self.assertRaisesRegex(AssertionError, "Unexpected reward configuration"):
            self.run_guard()

    def test_wrong_smith_mode_weight_or_target_rejected(self):
        for field, value in (("mode", "total"), ("target_height", "0.4"), ("weight", "20.0")):
            with self.subTest(field=field):
                self.setUp()
                reward = self.env["rewards"]["smith_roll"]
                if field == "weight":
                    reward[field] = value
                else:
                    reward["params"][field] = value
                with self.assertRaisesRegex(AssertionError, "Unexpected reward configuration"):
                    self.run_guard()

    def test_wrong_scene_environment_count_rejected(self):
        self.env["scene"]["num_envs"] = "128"
        with self.assertRaisesRegex(AssertionError, "Wrong environment count"):
            self.run_guard()

    def test_wrong_terrain_environment_count_rejected(self):
        self.env["scene"]["terrain"]["num_envs"] = "128"
        with self.assertRaises(AssertionError):
            self.run_guard()

    def test_wrong_training_iterations_rejected(self):
        self.agent["max_iterations"] = "2000"
        with self.assertRaisesRegex(AssertionError, "Wrong training budget"):
            self.run_guard()

    def test_unknown_environment_option_rejected(self):
        self.env["unplanned_reward_curriculum"] = "true"
        with self.assertRaisesRegex(AssertionError, "Non-reward environment changed"):
            self.run_guard()

    def test_mocked_reads_do_not_mutate_archived_documents(self):
        before = copy.deepcopy((self.control_env, self.control_agent, self.env, self.agent))
        self.run_guard()
        self.assertEqual(before, (self.control_env, self.control_agent, self.env, self.agent))


if __name__ == "__main__":
    unittest.main()
