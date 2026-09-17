"""CPU/stdlib tests. Cloned positives are NOT independent simulation evidence."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import compare_handoff_transition_baseline as audit


ROOT = Path(__file__).resolve().parents[1]
BASELINES = [
    ROOT / f"evaluations/20260917-handoff1999-to3547-{pose}/model_1999.pt_recovery_metrics.json"
    for pose in ("side", "upside_down", "upright-control")
]


def experimental_clone(baseline):
    candidate = copy.deepcopy(baseline)
    candidate.update(protocol_version=audit.PROTOCOL,
                     baseline_metric_protocol_version=baseline["protocol_version"],
                     controller_type=audit.EXPERIMENT_CONTROLLER,
                     acceptance_eligible=False, single_policy_acceptance_eligible=False,
                     training_collection_eligible=False)
    candidate["transition_experiment"] = {"mode": "hard", "ramp_seconds": 0.0,
        "baseline_evaluator_sha256": audit.ORIGINAL_SHA256, "synthetic_fixture_only": True}
    return candidate


class ExactHandoffComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are repository-local real baseline reports, always read-only.
        cls.baselines = [audit.load_report(path)[0] for path in BASELINES]

    def setUp(self):
        self.old = copy.deepcopy(self.baselines[0])
        self.new = experimental_clone(self.old)

    def mutate(self, path, value):
        target = self.new
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    def assert_rejected(self):
        try:
            result = audit.compare_reports(self.old, self.new)
        except audit.ComparisonError:
            return
        self.assertFalse(result["exact_baseline_equivalence"])

    def test_all_three_real_baseline_clones_are_synthetic_only(self):
        self.assertEqual([x["results"][next(iter(x["results"]))]["final_valid_stands"]
                          for x in self.baselines], [9, 20, 12])
        for baseline in self.baselines:
            candidate = experimental_clone(baseline)
            before = copy.deepcopy((baseline, candidate))
            result = audit.compare_reports(baseline, candidate)
            self.assertTrue(result["exact_baseline_equivalence"])
            self.assertGreater(result["checked_scalar_values"]["result_values"], 1000)
            self.assertFalse(result["acceptance_eligible"])
            self.assertFalse(result["original_evaluator_file_verified"])
            self.assertIn("Synthetic", result["notice"])
            self.assertEqual((baseline, candidate), before)

    def test_new_dictionary_metadata_allowed(self):
        self.new["new_metadata"] = {"evidence": "additional only"}
        self.new["results"]["side"]["new_diagnostic"] = [1.0, 2.0]
        self.new["results"]["side"]["final_diagnostics"][0]["new_metric"] = 3.0
        self.assertTrue(audit.compare_reports(self.old, self.new)["exact_baseline_equivalence"])

    def test_tiny_state_and_action_changes_never_tolerated(self):
        for field in ("release_state_before_settling", "policy_start_state"):
            with self.subTest(field=field):
                self.new = experimental_clone(self.old)
                row = self.new["results"]["side"][field][0]
                row["height_m"] += 1e-12
                self.assert_rejected()
        for field in ("policy_observation", "previous_raw_action", "previous_previous_raw_action",
                      "previous_executed_joint_target_rad", "stand_actor_raw_action", "roll_actor_raw_action"):
            with self.subTest(field=field):
                self.new = experimental_clone(self.old)
                self.new["results"]["side"]["handoff_diagnostic"]["switch_records"][0][field][0] += 1e-12
                self.assert_rejected()

    def test_final_result_and_list_length_or_order_changes_rejected(self):
        for change in ("count", "final", "removed", "reordered", "appended"):
            with self.subTest(change=change):
                self.new = experimental_clone(self.old)
                data = self.new["results"]["side"]
                if change == "count": data["final_valid_stands"] += 1
                if change == "final": data["final_diagnostics"][3]["height_m"] += 1e-12
                if change == "removed": data["policy_start_state"].pop()
                if change == "reordered": data["policy_start_state"][0:2] = reversed(data["policy_start_state"][0:2])
                if change == "appended": data["final_diagnostics"].append(copy.deepcopy(data["final_diagnostics"][0]))
                self.assert_rejected()

    def test_missing_old_nested_field_and_changed_numeric_types_rejected(self):
        del self.new["results"]["side"]["handoff_diagnostic"]["switch_records"][0]["previous_raw_action"]
        self.assert_rejected()
        self.new = experimental_clone(self.old)
        self.new["results"]["side"]["trials"] = 20.0
        self.assert_rejected()
        self.new = experimental_clone(self.old)
        self.new["results"]["side"]["final_diagnostics"][0]["geometry_ok"] = 1
        self.assert_rejected()

    def test_physics_identity_protocol_and_timing_changes_rejected(self):
        mutations = [("task", "other"), ("seed", 1), ("angle_deg", 0.0),
                     ("criterion", "weakened"), ("settle_actual_s", 2.0),
                     ("time_definition", "extended horizon"), ("self_collisions_enabled", False),
                     ("baseline_metric_protocol_version", "other")]
        for key, value in mutations:
            with self.subTest(key=key):
                self.new = experimental_clone(self.old)
                self.new[key] = value
                self.assert_rejected()
        for path, value in [(["handoff_controller", "stand_sha256"], "a" * 64),
                            (["handoff_controller", "stand_checkpoint"], "E:/other.pt"),
                            (["state_bank", "physics", "dt"], .01),
                            (["state_bank", "selected_state_ids", 0], 1),
                            (["action_representation", "scale"], .5)]:
            self.new = experimental_clone(self.old)
            self.mutate(path, value)
            self.assert_rejected()

    def test_mode_sha_and_acceptance_fail_closed(self):
        for path, value in [(["protocol_version"], "wrong"),
                            (["controller_type"], audit.BASELINE_CONTROLLER),
                            (["transition_experiment", "mode"], "raw_ramp"),
                            (["transition_experiment", "ramp_seconds"], .2),
                            (["transition_experiment", "ramp_seconds"], False),
                            (["transition_experiment", "baseline_evaluator_sha256"], "b" * 64),
                            (["acceptance_eligible"], True),
                            (["single_policy_acceptance_eligible"], True),
                            (["training_collection_eligible"], 0)]:
            self.new = experimental_clone(self.old)
            self.mutate(path, value)
            self.assert_rejected()

    def test_new_nonfinite_metadata_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.new = experimental_clone(self.old)
            self.new["additional"] = [value]
            self.assert_rejected()

    def test_duplicate_keys_and_json_overflow_rejected(self):
        for raw in (b'{"a": 1, "a": 2}', b'{"extra": 1e999}'):
            with patch.object(Path, "read_bytes", return_value=raw):
                with self.assertRaises(audit.ComparisonError): audit.load_report("unused.json")

    def test_bank_presence_and_missing_baseline_schema_rejected(self):
        del self.new["state_bank"]
        self.assert_rejected()
        self.new = experimental_clone(self.old)
        del self.old["criterion"]
        self.assert_rejected()

    def test_output_rejects_non_e_existing_and_non_json(self):
        for value in ("C:/bad.json", "relative.json", str(BASELINES[0]), "E:/IsaacLab/no.txt"):
            with self.subTest(value=value), self.assertRaises(audit.ComparisonError):
                audit.output_path(value)

    def test_writer_uses_exclusive_create_and_fsync(self):
        target = MagicMock()
        stream = target.open.return_value.__enter__.return_value
        with patch.object(audit.os, "fsync") as fsync:
            audit.write_output(target, {"acceptance_eligible": False})
        target.open.assert_called_once_with("x", encoding="utf-8", newline="\n")
        stream.flush.assert_called_once()
        fsync.assert_called_once_with(stream.fileno.return_value)

    def test_cli_success_failure_exit_and_source_guard_no_output_write(self):
        for changed in (False, True):
            self.new = experimental_clone(self.old)
            if changed: self.new["results"]["side"]["successes"] += 1
            with patch.object(audit, "load_report", side_effect=[(self.old, "a" * 64), (self.new, "b" * 64)]), \
                    patch.object(audit, "write_output") as writer, redirect_stdout(io.StringIO()) as output:
                code = audit.main(["--baseline", "old.json", "--candidate", "new.json"])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, int(changed))
            self.assertFalse(payload["acceptance_eligible"])
            writer.assert_not_called()
        with patch.object(audit.ORIGINAL_ENTRY.__class__, "read_bytes", return_value=b"changed source"), \
                redirect_stdout(io.StringIO()) as output:
            code = audit.main(["--baseline", "old.json", "--candidate", "new.json"])
        self.assertEqual(code, 1)
        self.assertIn("original evaluator changed", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
