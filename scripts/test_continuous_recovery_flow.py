"""CPU-only structure/receipt/clock tests; no simulation or physical PASS claim."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import evaluate_continuous_recovery_flow as entry
import continuous_flow_runtime as runtime


class ContinuousRuntimeContractTests(unittest.TestCase):
    def test_native_asset_path_spelling_matches_pinned_screen(self):
        trace = json.loads((entry.SCREEN / "retention_interface.json").read_text())
        self.assertEqual(str(Path("E:/IsaacLab/userdata/assets/Robots/Unitree/Go2/go2.usd")),
                         trace["config"]["after"]["robot"]["spawn"]["usd_path"])

    def test_only_actual_scene_constructor_namespace_expansion_is_expected(self):
        before = {"robot": {"prim_path": "{ENV_REGEX_NS}/Robot", "collision": True}, "dt": .005}
        original = copy.deepcopy(before)
        expected = runtime.expected_initialized_config(before, "/World/envs/env_.*")
        self.assertEqual(before, original)
        self.assertEqual(expected, {"robot": {"prim_path": "/World/envs/env_.*/Robot", "collision": True}, "dt": .005})
        bad = copy.deepcopy(expected)
        bad["robot"]["collision"] = False
        self.assertNotEqual(expected, bad)
        for namespace in ("/World/envs/env_0", "/Other/env_.*"):
            with self.assertRaises(RuntimeError):
                runtime.expected_initialized_config(before, namespace)

    def test_every_frozen_hash_has_full_sha256_width(self):
        for digest in (*entry.FROZEN.values(), *entry.MODEL_SHA.values(), *entry.EVIDENCE.values(), *entry.BALANCED_PINS.values()):
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_actual_screen_is_point5_complete_and_sha_bound(self):
        path = entry.SCREEN / "model_3947_stand_walk_stop.json"
        self.assertEqual(entry.sha(path), entry.EVIDENCE[path])
        result = entry.validate_screen(json.loads(path.read_text()))
        self.assertEqual(result["cmd_m_s"], .5)
        self.assertEqual(result["unique_physical_trajectories"], 1)

    def test_false_or_unchecked_screen_is_rejected(self):
        report = json.loads((entry.SCREEN / "model_3947_stand_walk_stop.json").read_text())
        for mutation in (lambda x: x.update(retention_passed=False),
                         lambda x: x["protocol"].update(walk_speed=.8),
                         lambda x: x["retention_acceptance"].update(straight_drift=False),
                         lambda x: x["acceptance"].pop("normal_stance_geometry_at_rest"),
                         lambda x: x["settled_phase_stats"]["walk"].update(vy_b_mean=.12),
                         lambda x: x.update(checkpoint_sha256="0"*64)):
            bad = copy.deepcopy(report)
            mutation(bad)
            with self.assertRaises(ValueError):
                entry.validate_screen(bad)

    def test_model_selection_is_explicit_default_control_and_separate_evidence(self):
        control = entry.locomotion_spec()
        balanced = entry.locomotion_spec("balanced4246")
        self.assertEqual(control["selected_model"], "control3947")
        self.assertEqual(control["path"], str(entry.CONTROL))
        self.assertEqual(balanced["sha"], entry.BALANCED_SHA)
        for key in ("path", "sha", "screen_report_path", "interface_path", "screen_protocol"):
            self.assertNotEqual(control[key], balanced[key])
        self.assertEqual(balanced["interface_path"], balanced["actual_passed_interface_path"])
        for name in ("balanced", "control4246", "", None):
            with self.assertRaises(ValueError):
                entry.locomotion_spec(name)
        self.assertEqual(entry.PROTOCOL, "continuous_recovery_flow_diagnostic_v3_sensor_identity")

    def test_sensor_topology_does_not_reuse_articulation_order(self):
        class Bodies:
            def __init__(self, names):
                self.body_names, self.num_bodies = names, len(names)
            def find_bodies(self, names, preserve_order):
                self.assert_preserve_order = preserve_order
                return [self.body_names.index(n) for n in names], names
        sensor = Bodies(["base", "RR_foot", "FR_foot", "RL_foot", "FL_foot"])
        asset = Bodies(["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        actual = runtime.contact_sensor_topology(sensor, asset)
        self.assertEqual(actual["foot_ids"], [4,2,3,1])
        self.assertEqual(actual["foot_articulation_ids"], [1,2,3,4])
        self.assertTrue(sensor.assert_preserve_order)
        sensor.num_bodies += 1
        with self.assertRaises(RuntimeError):
            runtime.contact_sensor_topology(sensor, asset)

    def test_balanced_pinned_evidence_exists_without_sim_or_tensor_load(self):
        for path, digest in entry.BALANCED_PINS.items():
            self.assertEqual(entry.sha(path), digest, str(path))

    def test_actual_balanced_point5_screen_does_not_inherit_control_result(self):
        control = entry.read_json(entry.SCREEN / "model_3947_stand_walk_stop.json")
        balanced = entry.read_json(entry.BALANCED_CASE / "model_4246_stand_walk_stop.json")
        result = entry.validate_screen(balanced, "balanced4246")
        self.assertEqual(result["report"], str(entry.BALANCED_CASE / "model_4246_stand_walk_stop.json"))
        self.assertEqual(result["cmd_m_s"], .5)
        self.assertIn("passed13/21", result["notice"])
        for report, selected in ((control,"balanced4246"), (balanced,"control3947")):
            with self.assertRaises(ValueError):
                entry.validate_screen(report, selected)

    def test_candidate_screen_identity_and_unchanged_acceptance_fail_closed(self):
        report = entry.read_json(entry.BALANCED_CASE / "model_4246_stand_walk_stop.json")
        for mutation in (lambda x: x["retention_experiment"].update(trace_path=str(entry.SCREEN / "retention_interface.json")),
                         lambda x: x["retention_experiment"].update(trace_sha256=entry.EVIDENCE[entry.SCREEN / "retention_interface.json"]),
                         lambda x: x["retention_experiment"].update(candidate_checkpoint_sha256=entry.MODEL_SHA["locomotion"]),
                         lambda x: x["retention_experiment"].update(no_added_reset_or_history_write=False),
                         lambda x: x["acceptance"].update(unrelated=x["acceptance"].pop("no_reset")),
                         lambda x: x["settled_phase_stats"]["walk"].update(contact_slip_mean=.12)):
            bad = copy.deepcopy(report)
            mutation(bad)
            with self.assertRaises(ValueError):
                entry.validate_screen(bad, "balanced4246")

    def test_actual_balanced_broader_failures_remain_explicit(self):
        summary = entry.read_json(entry.BALANCED_SCREEN / "summary.json")
        comparison = entry.read_json(entry.BALANCED_SCREEN / "matched_comparison.json")
        result = entry.validate_balanced_limitations(summary, comparison)
        self.assertEqual((result["screen_passed"], result["screen_total"]), (13, 21))
        self.assertIs(result["gait_selection_passed"], False)
        self.assertIs(result["promotion_performed"], False)
        for mutation in (lambda x: x.update(diagnostic_selection_passed=True),
                         lambda x: x.update(promotion_performed=True),
                         lambda x: x["checks"].update(balanced_all21_old_physical_criteria=True),
                         lambda x: x["checks"].update(mean_duty_gap_reduction_at_least25percent=True)):
            bad = copy.deepcopy(comparison)
            mutation(bad)
            with self.assertRaises(ValueError):
                entry.validate_balanced_limitations(summary, bad)
        bad = copy.deepcopy(summary)
        bad["rows"] = bad["rows"][:-1]
        with self.assertRaises(ValueError):
            entry.validate_balanced_limitations(bad, comparison)

    def test_selection_cli_and_full_verification_cannot_skip_receipt_or_audits(self):
        source = Path(entry.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        verifier = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_locomotion")
        calls = [n.func.id for n in ast.walk(verifier) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("candidate_receipt", calls)
        self.assertIn("audit", calls)
        self.assertIn("compare", calls)
        self.assertIn('choices=("control3947", "balanced4246"), default="control3947"', source)
        self.assertIn('require(len(preservation["files"]) == 16', source)

    def test_exact_frozen_helper_definitions_only(self):
        self.assertEqual(entry.sha(entry.MIRROR), entry.FROZEN["scripts/evaluate_handoff_mirror.py"])
        generated = entry.extract_definitions(entry.MIRROR.read_text(encoding="utf-8"))
        tree = ast.parse(generated)
        self.assertEqual({n.name for n in tree.body if isinstance(n, ast.FunctionDef)}, set(entry.HELPERS))
        self.assertFalse(any(isinstance(n, (ast.If, ast.Assign, ast.Expr)) for n in tree.body))
        self.assertNotIn("_reset_policy_history", generated)
        self.assertNotIn("def main", generated)

    def test_definition_extraction_never_executes_top_level(self):
        source = 'raise RuntimeError("must never run")\ndef safe(x):\n    return x+1\n'
        namespace = {}
        exec(compile(entry.extract_definitions(source, ("safe",)), "test", "exec"), namespace)
        self.assertEqual(namespace["safe"](2), 3)
        for bad in ('def other():\n    return 1\n', '@side_effect()\ndef safe():\n    pass\n',
                    'def safe():\n    pass\ndef safe():\n    pass\n'):
            with self.assertRaises(ValueError):
                entry.extract_definitions(bad, ("safe",))

    def test_physical_counter_records_pd_without_common_counter_rewrite(self):
        self.assertEqual(runtime.physical_index(1400, 1200), 50)
        before = {"sim_step": 1400, "episode_step": 50, "common_step": 0}
        after = {"sim_step": 1404, "episode_step": 51, "common_step": 1}
        runtime.assert_continuity(before, after, "recovery", 1200)
        runtime.assert_continuity({"sim_step": 1200,"episode_step":0,"common_step":0},
                                  {"sim_step":1204,"episode_step":1,"common_step":0},"preparation",1200)
        for wrong in (1401, 1196):
            with self.assertRaises(ValueError):
                runtime.physical_index(wrong, 1200)
        for key, value in (("sim_step",1408),("episode_step",1),("common_step",0)):
            with self.assertRaises(RuntimeError):
                runtime.assert_continuity(before, dict(after, **{key:value}), "recovery",1200)

    def test_live_guard_blocks_reset_before_mutation_and_records_attempt(self):
        env = SimpleNamespace(_sim_step_counter=200, calls=0)
        def reset(*args, **kwargs):
            env.calls += 1
        env.reset = env._reset_idx = reset
        guard = runtime.NoResetGuard(env)
        for method in (env.reset, env._reset_idx):
            with self.assertRaises(runtime.ResetForbidden):
                method([0])
        self.assertEqual(env.calls, 0)
        self.assertEqual(len(guard.attempts), 2)
        guard.restore_for_close()
        env.reset()
        self.assertEqual(env.calls, 1)

    def test_only_command_may_change_at_phase_edge(self):
        previous = {"root_pos_w":[[0.,0.,.3]], "actual_raw_action":[[.2]*12],
                    "sim_step":200, "velocity_command_b":[[0.,0.,0.]]}
        after = copy.deepcopy(previous)
        after["velocity_command_b"] = [[.5,0.,0.]]
        self.assertTrue(runtime.unchanged_between_intervals(previous, after))
        after["actual_raw_action"] = [[0.]*12]
        self.assertFalse(runtime.unchanged_between_intervals(previous, after))

    def test_nonfinite_evidence_not_forged_as_valid_scalar(self):
        value = {"actual": [1., float("nan"), float("inf")]}
        self.assertFalse(runtime.finite_evidence(value))
        encoded = runtime.safe_json(value)
        self.assertEqual(encoded["actual"][1], {"nonfinite_number":"nan"})
        json.dumps(encoded, allow_nan=False)

    def test_no_old_main_or_post_pd_reset_or_policy_reset_in_runtime(self):
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ("reset", "_reset_idx") for n in calls))
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == "_reset_policy_history" for n in calls))
        self.assertEqual(source.count("env.step(action)"), 1)
        self.assertNotIn("policies[\"roll\"](obs)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
