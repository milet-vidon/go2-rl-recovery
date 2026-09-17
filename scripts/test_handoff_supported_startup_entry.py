"""Stdlib-only source/routing-log regression tests; no torch, Isaac or GPU import."""

import ast
from collections import deque
import copy
import json
import math
import sys
import unittest

import evaluate_handoff_supported_startup as entry
import compare_handoff_supported_startup as comparison

sys.path.insert(0, str(entry.ROOT / "src/go2_recovery"))
from supported_startup_math import supported_startup_mask, SupportedStartupInputError


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = entry.build_source()
        cls.tree = ast.parse(cls.source)
        # Execute just the frozen stdlib-only LOGGER class, not its torch module.
        logger_tree = ast.parse((entry.ROOT / "src/go2_recovery/handoff_transition_math.py").read_text())
        logger = next(node for node in logger_tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "TransitionNeighborhood")
        module = ast.fix_missing_locations(ast.Module(body=[logger], type_ignores=[]))
        namespace = {"deque": deque}
        exec(compile(module, "<frozen-logger-class-only>", "exec"), namespace)
        cls.logger = namespace["TransitionNeighborhood"]

    def test_compiles_without_simulator_import(self):
        compile(self.source, "<startup-generated>", "exec")
        self.assertNotIn("isaaclab.app", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_unique_anchor_rejects_missing_or_duplicate(self):
        for text in ("", "xx"):
            with self.assertRaises(ValueError):
                entry._replace_once(text, "x", "y")
        self.assertEqual(entry._replace_once("ax", "x", "y"), "ay")

    def test_selector_has_one_call_at_actual_start(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "supported_startup_mask"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, "policy_start_state")
        self.assertLess(self.source.index("predicate_mask, startup_records"),
                        self.source.index("for step in range(steps):"))

    def test_stand_active_does_not_relabel_genuine_switch(self):
        self.assertIn("stand_active = startup_selected | switched", self.source)
        self.assertIn("gate_steps = torch.where(startup_selected, gate_steps, next_gate)", self.source)
        self.assertIn("switched = torch.where(startup_selected, switched, next_switched)", self.source)
        self.assertIn("just_switched = torch.where(startup_selected, just_switched, next_edge)", self.source)
        self.assertIn('"triggered_trials": int(switched.sum())', self.source)
        self.assertIn('"selected_genuine_handoffs": int((switched & startup_selected).sum())', self.source)

    def test_selected_logger_has_no_fictitious_handoff_or_ramp(self):
        log = self.logger(2, 2, 2)
        log.phases[0] = "startup_stand"
        for step in range(5):
            log.record(0, step, {"phase": "startup_stand", "policy_time_s": step * .02, "alpha": 1.0}, False)
            log.record(1, step, {"phase": "roll", "policy_time_s": step * .02, "alpha": 0.0}, False)
        for row in log.report():
            self.assertFalse(row["triggered"])
            self.assertIsNone(row["switch_step"])
            self.assertEqual(row["events"], [])
        self.assertIn('neighborhood.phases[i] = "startup_stand"', self.source)

    def test_off_logger_matches_unchanged_logger(self):
        original, off = self.logger(2, 2, 2), self.logger(2, 2, 2)
        for step in range(6):
            for trial in range(2):
                event = trial == 0 and step == 2
                row = {"phase": "stand" if trial == 0 and step >= 2 else "roll",
                       "policy_time_s": step * .02, "alpha": float(trial == 0 and step >= 2)}
                original.record(trial, step, dict(row), event)
                off.record(trial, step, dict(row), event)
        self.assertEqual(original.report(), off.report())

    def test_first_actions_outlive_ring_and_keep_actual_history(self):
        self.assertIn('if step == 0 and len(transition_rows) != env.num_envs:', self.source)
        self.assertIn('first_action_by_pose[pose_class].append(row)', self.source)
        self.assertIn('actual_previous_raw_action=actual_history[i]', self.source)
        self.assertIn('"first_action_by_pose": first_action_by_pose', self.source)
        self.assertIn('if len(first_action_by_pose[pose_class]) != env.num_envs:', self.source)

    def test_invalid_nonfinite_evidence_stays_explicit_and_serializable(self):
        original = {"valid": False, "inputs": {"v": [math.inf, -math.inf, math.nan]}, "selected": None}
        clean = entry.invalid_diagnostic_json(original)
        json.dumps(clean, allow_nan=False)
        self.assertEqual(clean["inputs"]["v"][0], {"nonfinite_number": "inf"})
        self.assertEqual(clean["inputs"]["v"][1], {"nonfinite_number": "-inf"})
        self.assertEqual(clean["inputs"]["v"][2], {"nonfinite_number": "nan"})
        self.assertIsNone(clean["selected"])
        self.assertTrue(math.isinf(original["inputs"]["v"][0]))
        self.assertIn('"records": STARTUP_INVALID_JSON(exc.records)', self.source)

    def test_overflow_actual_invalid_predicate_record_can_be_written(self):
        path = entry.ROOT / "evaluations/20260917-handoff1999-to3547-upright-control/model_1999.pt_recovery_metrics.json"
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        start = copy.deepcopy(report["results"]["upright"]["policy_start_state"][0])
        start["root_linear_velocity_w_m_s"] = [1.7e308] * 3
        with self.assertRaises(SupportedStartupInputError) as raised:
            supported_startup_mask([start])
        clean = entry.invalid_diagnostic_json(raised.exception.records)
        json.dumps(clean, allow_nan=False)
        self.assertFalse(clean[0]["valid"])
        self.assertIsNone(clean[0]["selected_actor"])
        self.assertEqual(clean[0]["inputs"]["root_linear_speed_m_s"], {"nonfinite_number": "inf"})

    def test_no_runtime_source_or_model_rewrite_in_adapter(self):
        self.assertIn('with (args_cli.output_dir / "supported_startup_generated.py").open("x"', self.source)
        self.assertIn('"template_sha256": STARTUP_TEMPLATE_SHA', self.source)
        self.assertIn('"generated_source_sha256": STARTUP_GENERATED_SHA', self.source)
        self.assertIn('"adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()', self.source)

    def test_exact_comparator_does_not_coerce_bool_or_float(self):
        for candidate in (True, 1.0, 1.0000000000000002):
            match = comparison.ExactComparison()
            match.compare(1, candidate, "x")
            self.assertGreater(match.mismatch_count, 0)

    def test_exact_comparator_checks_all_old_fields_and_list_lengths(self):
        match = comparison.ExactComparison()
        match.compare({"x": [1, 2], "y": False}, {"x": [1]}, "row")
        self.assertEqual(match.mismatch_count, 2)
        match = comparison.ExactComparison()
        match.compare({"x": [1, 2]}, {"x": [1, 2], "diagnostic": True}, "row")
        self.assertEqual(match.mismatch_count, 0)


if __name__ == "__main__":
    unittest.main()
