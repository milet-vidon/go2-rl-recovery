"""Pure stdlib source/logger tests: no simulator, GPU or torch is imported."""
import ast
from collections import deque
import json
import math
import sys
import unittest

import evaluate_handoff_combined as entry


class CombinedEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = entry.build_source()
        cls.tree = ast.parse(cls.source)

    def test_generated_compiles_without_simulator(self):
        compile(self.source, "<combined-generated>", "exec")
        self.assertNotIn("isaaclab.app", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_unique_anchor_failure_is_closed(self):
        for value in ("", "xx"):
            with self.assertRaises(ValueError):
                entry._replace_once(value, "x", "y")

    def test_explicit_two_flags_and_hard_only(self):
        self.assertIn('parser.add_argument("--startup_mode", required=True, choices=("off", "supported"))', self.source)
        self.assertIn('parser.add_argument("--mirror_mode", required=True, choices=("off", "initial_right"))', self.source)
        self.assertIn('parser.set_defaults(transition_mode="hard", ramp_seconds=0.0)', self.source)

    def test_wrapper_uses_only_exposed_transition_flags(self):
        wrapper = (entry.ROOT / "scripts/evaluate_handoff_combined.ps1").read_text(encoding="utf-8")
        self.assertIn("'--startup_mode',$Mode,'--mirror_mode',$MirrorMode", wrapper)
        self.assertNotIn("'--transition_mode'", wrapper)
        self.assertNotIn("'--ramp_seconds'", wrapper)

    def test_both_masks_decided_once_before_rollout(self):
        for function in ("supported_startup_mask", "initial_right_mask"):
            calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == function]
            self.assertEqual(len(calls), 1)
            self.assertLess(self.source.index(function + "("), self.source.index("for step in range(steps):"))

    def test_fixed_mirror_candidate_not_masked_after_handoff(self):
        assignments = [n for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "mirror_selected" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertIn('roll_obs = roll_input_copy(obs, mirror_selected)', self.source)
        self.assertIn('roll_action = physical_roll_action(roll_model_action, mirror_selected)', self.source)
        self.assertIn('stand_action = stand_policy(obs)  # ALWAYS real/native observations.', self.source)

    def test_actual_overlap_guard_has_hard_failure(self):
        nodes = [n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                 and any(isinstance(s, ast.Raise) and isinstance(s.exc, ast.Call)
                         and s.exc.args and isinstance(s.exc.args[0], ast.Constant)
                         and s.exc.args[0].value == "Supported startup and settled-fallen mirror masks overlap"
                         for s in n.body)]
        self.assertEqual(len(nodes), 1)
        self.assertIn('if bool((startup_selected & mirror_selected).any()):', self.source)

    def test_gate_and_execution_masks_are_distinct(self):
        for text in ('stand_active = startup_selected | switched',
                     'gate_steps = torch.where(startup_selected, gate_steps, next_gate)',
                     'switched = torch.where(startup_selected, switched, next_switched)',
                     'just_switched = torch.where(startup_selected, just_switched, next_edge)',
                     'previous_raw_action, stand_active, just_switched,',
                     '"triggered_trials": int(switched.sum())'):
            self.assertIn(text, self.source)

    def test_first_action_and_natural_history_evidence(self):
        for text in ('if step == 0 and len(transition_rows) != env.num_envs:',
                     'first_action_by_pose[pose_class].append(row)',
                     'torch.equal(env.unwrapped.action_manager.prev_action, previous_raw_action)',
                     'torch.equal(env.unwrapped.action_manager.action, action)',
                     'actual_previous_raw_action=actual_history[i]'):
            self.assertIn(text, self.source)

    def test_selected_logger_has_no_fake_switch(self):
        tree = ast.parse((entry.ROOT / "src/go2_recovery/handoff_transition_math.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransitionNeighborhood")
        namespace = {"deque": deque}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<logger-only>", "exec"), namespace)
        logger = namespace["TransitionNeighborhood"](2, 2, 2)
        logger.phases[0] = "startup_stand"
        for step in range(5):
            logger.record(0, step, {"phase": "startup_stand", "policy_time_s": step * .02, "alpha": 1.0}, False)
            logger.record(1, step, {"phase": "roll", "policy_time_s": step * .02, "alpha": 0.0}, False)
        for row in logger.report():
            self.assertFalse(row["triggered"])
            self.assertIsNone(row["switch_step"])
            self.assertEqual(row["events"], [])
        self.assertIn('neighborhood.phases[i] = "startup_stand"', self.source)

    def test_invalid_nonfinite_evidence_is_explicit(self):
        clean = entry.invalid_diagnostic_json({"valid": False, "actor": None, "v": [math.inf, -math.inf, math.nan]})
        json.dumps(clean, allow_nan=False)
        self.assertEqual(clean["v"], [{"nonfinite_number": "inf"}, {"nonfinite_number": "-inf"}, {"nonfinite_number": "nan"}])
        self.assertIsNone(clean["actor"])

    def test_honest_metadata_and_artifact_names(self):
        for text in ('"startup_selection_changed": args_cli.startup_mode == "supported"',
                     '"mirror_enabled": args_cli.mirror_mode == "initial_right"',
                     'experiment["baseline_hard_zero_semantics"] = experiment["hard_zero_semantics"]',
                     '"schema": "handoff_combined_neighborhood_v1"',
                     'f"{checkpoint_name}_handoff_combined_trace.json"',
                     '"generated_source_file": "combined_generated.py"'):
            self.assertIn(text, self.source)

    def test_audit_aliases(self):
        self.assertEqual(entry.MATH_PATH, entry.STARTUP_MATH)
        self.assertEqual(entry.MATH_SHA, entry.STARTUP_MATH_SHA)
        self.assertEqual(entry.MIRROR_MATH_PATH, entry.MIRROR_MATH)
        self.assertEqual(len(entry.ROLL_SHA), 64)
        self.assertEqual(len(entry.STAND_SHA), 64)


if __name__ == "__main__":
    unittest.main()
