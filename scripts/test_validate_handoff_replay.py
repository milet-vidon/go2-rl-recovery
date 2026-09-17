"""CPU-only input/AST checks. Never launch the validator's main or simulator."""

import ast
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import validate_handoff_replay as replay


class ReplayInputTests(unittest.TestCase):
    def test_explicit_all_samples_reads_all_79_train_rows_without_modifying_raw(self):
        arrays, manifest, selected, source = replay.load_raw_collection(replay.DEFAULT_DATASET, None, all_samples=True)
        self.assertEqual(len(selected), 79)
        np.testing.assert_array_equal(selected, np.arange(79))
        self.assertEqual(manifest["total_source_trials"], 80)
        self.assertEqual(manifest["untriggered_source_trials"], 1)
        self.assertFalse(manifest["training_ready"])
        self.assertEqual(source["archive_sha256"], replay.digest(replay.DEFAULT_DATASET / "handoff_observations.npz"))
        self.assertEqual(len(set(arrays["source_train_state_id"][selected].tolist())), 79)

    def test_all_samples_is_explicit_exclusive_and_capped_at_128(self):
        for ids, requested, all_samples in ((list(range(79)), None, False),
                                            (list(range(79)), list(range(79)), False),
                                            (list(range(79)), [0, 1], True),
                                            (list(range(79)), [], True),
                                            (list(range(129)), None, True),
                                            ([], None, True), ([0, 0], None, True),
                                            ([0, 1], None, 1)):
            with self.subTest(size=len(ids), requested=requested, all_samples=all_samples), self.assertRaises(ValueError):
                replay.select_sample_rows(ids, requested, all_samples)
        self.assertEqual(len(replay.select_sample_rows(list(range(128)), None, True)), 128)
        with self.assertRaises(ValueError):
            replay.load_raw_collection(replay.DEFAULT_DATASET, [0, 1], all_samples=True)

    def test_selection_cli_requires_exactly_one_mode(self):
        parser = argparse.ArgumentParser()
        replay.add_sample_selection_arguments(parser)
        selected = parser.parse_args(["--all-samples"])
        self.assertTrue(selected.all_samples)
        self.assertIsNone(selected.sample_ids)
        selected = parser.parse_args(["--sample-ids", "0", "1"])
        self.assertFalse(selected.all_samples)
        self.assertEqual(selected.sample_ids, [0, 1])
        for arguments in ([], ["--all-samples", "--sample-ids", "0", "1"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_probe_checks_each_physical_field_and_nan_serializes_safely(self):
        fields = {"root_link_pose_w": np.zeros((2, 7)), "root_com_velocity_w": np.zeros((2, 6)),
                  "joint_position_rad": np.zeros((2, 12)), "joint_velocity_rad_s": np.zeros((2, 12)),
                  "contact_forces_w": np.zeros((2, 17, 3))}
        self.assertTrue(replay.contact_probe_finiteness(fields)["finite"])
        for name in fields:
            for value in (float("nan"), float("inf"), -float("inf")):
                invalid = {key: array.copy() for key, array in fields.items()}
                invalid[name].flat[-1] = value
                with self.subTest(field=name, value=value):
                    result = replay.contact_probe_finiteness(invalid)
                    self.assertFalse(result["finite"])
                    self.assertFalse(result["finite_checks"][name])
        with self.assertRaises(ValueError):
            replay.contact_probe_finiteness({})
        safe = replay.json_safe_array(np.array([[0.0, np.nan, np.inf, -np.inf]]))
        self.assertEqual(safe, [[0.0, None, None, None]])
        json.dumps(safe, allow_nan=False)

    def test_nonfinite_probe_fails_without_relabeling_matched_input(self):
        with tempfile.TemporaryDirectory(prefix="handoff-probe-test-", dir=replay.WORKSPACE / "tmp") as folder:
            result = {"explicit_state_matches": True, "observation_comparison": {"all_elements_match": True},
                      "contact_probe": {"finite": False, "used_for_success_classification": False}}
            with patch.object(replay, "run_replay", return_value=result), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = replay.execute_and_record(None, None, None, None, Path(folder), {"training_ready": False})
            saved = json.loads((Path(folder) / "handoff_replay_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertTrue(saved["input_replay_pass"])
            self.assertFalse(saved["contact_probe_finite_pass"])
            self.assertFalse(saved["validation_pass"])
            self.assertFalse(saved["training_ready"])

    def test_lifecycle_result_and_exception_are_saved_before_hard_exit(self):
        for success, error in ((True, None), (False, None),
                               (False, ValueError("injected environment mismatch"))):
            with self.subTest(success=success, error=error), tempfile.TemporaryDirectory(
                    prefix="handoff-lifecycle-test-", dir=replay.WORKSPACE / "tmp") as folder:
                destination = Path(folder)
                events = []
                outer = self

                def read_report():
                    return json.loads((destination / "handoff_replay_validation.json").read_text(encoding="utf-8"))

                class FakeApp:
                    def close(self):
                        saved = read_report()
                        outer.assertTrue(saved["result_written_before_shutdown"])
                        outer.assertEqual(saved["input_replay_pass"], success)
                        outer.assertEqual(saved["intended_process_exit_code"], 0 if success else 1)
                        if error is not None:
                            outer.assertIn("injected environment mismatch", saved["error"])
                            outer.assertIn("ValueError", saved["traceback"])
                        events.append("app_close")
                        # Model a shutdown that never returns to its caller.
                        raise SystemExit(0)

                class FakeEnv:
                    def close(self):
                        read_report()
                        events.append("env_close")

                def fake_replay(*args):
                    args[-1].update(app=FakeApp(), wrapped=FakeEnv())
                    if error is not None:
                        raise error
                    return {"explicit_state_matches": success,
                            "observation_comparison": {"all_elements_match": True}}

                with patch.object(replay, "run_replay", side_effect=fake_replay), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    replay.execute_and_record(None, None, None, None, destination, {})
                self.assertEqual(events, ["env_close", "app_close"])

    def test_lifecycle_returns_nonzero_when_close_returns_on_failure(self):
        with tempfile.TemporaryDirectory(prefix="handoff-lifecycle-test-", dir=replay.WORKSPACE / "tmp") as folder:
            with patch.object(replay, "run_replay", side_effect=RuntimeError("before app acquired")), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = replay.execute_and_record(None, None, None, None, Path(folder), {})
            self.assertEqual(code, 1)
            saved = json.loads((Path(folder) / "handoff_replay_validation.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["input_replay_pass"])
            self.assertIn("before app acquired", saved["traceback"])

    def test_run_replay_never_closes_resources_before_caller_records(self):
        tree = ast.parse(Path(replay.__file__).read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_replay")
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr == "close" for node in ast.walk(run)))
        launches = [node for node in ast.walk(run) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "AppLauncher"]
        self.assertEqual(launches[0].keywords, [])
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr == "post_quit" for node in ast.walk(tree)))

    def test_import_and_cpu_checks_do_not_import_simulator(self):
        self.assertFalse(any(name == "isaaclab" or name.startswith("isaaclab.") for name in sys.modules))

    def test_actual_two_and_four_sample_ids_validate_train_and_native_order(self):
        for ids in ([0, 1], [0, 1, 40, 41]):
            with self.subTest(ids=ids):
                arrays, manifest, selected, source = replay.load_raw_collection(replay.DEFAULT_DATASET, ids)
                self.assertEqual(arrays["sample_id"][selected].tolist(), ids)
                self.assertEqual(manifest["source_split"], "train")
                self.assertFalse(manifest["training_ready"])
                self.assertEqual(len(source["archive_sha256"]), 64)
                actual = arrays["policy_observation"][selected].astype(np.float32)
                result = replay.compare_observations(actual, arrays["policy_observation"][selected])
                self.assertTrue(result["all_elements_match"])

    def test_wrong_sample_counts_duplicate_or_unknown_ids_rejected(self):
        for ids in ([0], [0, 1, 2], [0, 0], [0, 100000]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                replay.load_raw_collection(replay.DEFAULT_DATASET, ids)

    def test_compare_records_all_48_errors_without_hiding_single_failure(self):
        recorded = np.zeros((2, 48))
        actual = recorded.copy()
        actual[1, 47] = 0.01
        result = replay.compare_observations(actual, recorded)
        self.assertFalse(result["all_elements_match"])
        self.assertEqual(result["per_sample_maximum_absolute_error"], [0.0, 0.01])
        self.assertEqual(result["per_block_maximum_absolute_error"]["previous_raw_action"], 0.01)
        np.testing.assert_array_equal(recorded, np.zeros((2, 48)))

    def test_shape_and_nonfinite_observations_rejected(self):
        with self.assertRaises(ValueError):
            replay.compare_observations(np.zeros((2, 47)), np.zeros((2, 48)))
        for value in (float("nan"), float("inf")):
            actual = np.zeros((2, 48))
            actual[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                replay.compare_observations(actual, np.zeros((2, 48)))

    def test_no_policy_execution_or_observation_buffer_injection(self):
        tree = ast.parse(Path(replay.__file__).read_text(encoding="utf-8"))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("env.step", calls)
        self.assertFalse(any("OnPolicyRunner" in call or "get_inference_policy" in call for call in calls))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                self.assertFalse(any("obs_buf" in ast.unparse(target) or "observation_manager" in ast.unparse(target)
                                     for target in targets))
        computations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == "env.observation_manager.compute"]
        self.assertEqual(len(computations), 1)
        self.assertEqual({arg.arg: ast.literal_eval(arg.value) for arg in computations[0].keywords},
                         {"update_history": False})

    def test_only_physics_integration_is_inside_optional_contact_probe(self):
        tree = ast.parse(Path(replay.__file__).read_text(encoding="utf-8"))
        integrations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == "env.sim.step"]
        self.assertEqual(len(integrations), 1)
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "args.contact_probe"]
        self.assertEqual(len(guards), 1)
        self.assertIn(integrations[0], list(ast.walk(guards[0])))


if __name__ == "__main__":
    unittest.main()
