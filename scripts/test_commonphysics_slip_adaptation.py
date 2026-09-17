"""CPU-only tests. No test starts Isaac Sim, training or a subprocess.

These were written while the single simulator pipeline was active; execution
must be performed by the root reviewer after the E-workspace becomes idle.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import train_commonphysics_slip_adaptation as entry


class SlipAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = {arm: entry.adapter(arm) for arm in ("A", "B")}
        base = cls.adapters["A"]
        cls.parent_env = base.read(base.PARENT_CONFIG / "env.yaml")
        cls.parent_agent = base.read(base.PARENT_CONFIG / "agent.yaml")
        cls.reference = base.reference_document()
        cls.parent = base.parent_metadata()

    def documents(self, arm, n=128, updates=300):
        trainer = self.adapters[arm]
        tag = f"20260917-slip-{arm}-unit-test"
        env = trainer.expected_environment(self.parent_env, self.parent_env, self.reference, n)
        agent = copy.deepcopy(self.parent_agent)
        agent.update(run_name=tag, max_iterations=str(updates), load_run=entry.PARENT_RUN,
                     load_checkpoint="model_4246.pt")
        agent["algorithm"]["learning_rate"] = str(trainer.PARENT_LR)
        return env, agent, tag

    def check(self, arm, env, agent, tag, n=128, updates=300):
        return self.adapters[arm].check_documents(env, agent, self.parent_env, self.parent_agent,
                                                  self.reference, n, updates, tag)

    def test_source_program_compiles_with_declared_protocol(self):
        for arm, trainer in self.adapters.items():
            ast.parse(entry.program(arm))
            self.assertEqual(trainer.PROTOCOL, entry.PROTOCOL)
            self.assertEqual(trainer.PARENT, entry.PARENT)
            self.assertEqual(trainer.ARM, arm)

    def test_actual_parent_full_state_and_learning_rate(self):
        self.assertEqual(self.parent["iter"], 4246)
        self.assertEqual(self.parent["adam_steps"], [85120] * 17)
        self.assertEqual(self.parent["sha256"], entry.PARENT_SHA)
        for trainer in self.adapters.values():
            self.assertEqual(trainer.PARENT_LR, self.parent["optimizer_group"]["lr"])

    def test_arms_only_differ_by_existing_foot_slip_weight(self):
        a, aa, tag_a = self.documents("A")
        b, ab, tag_b = self.documents("B")
        self.check("A", a, aa, tag_a)
        guard = self.check("B", b, ab, tag_b)
        self.assertEqual(a["rewards"]["balanced_duration"], b["rewards"]["balanced_duration"])
        self.assertEqual(a["rewards"]["balanced_duration"]["weight"], "-10.0")
        self.assertEqual(a["rewards"]["foot_slip"]["weight"], "-0.5")
        self.assertEqual(b["rewards"]["foot_slip"]["weight"], "-1.0")
        b["rewards"]["foot_slip"]["weight"] = "-0.5"
        self.assertEqual(a, b)
        self.assertEqual(a["rewards"], self.parent_env["rewards"])
        self.assertEqual(guard["actual_physics_changed_paths"], [])

    def test_smoke_and_formal_finite_only(self):
        for arm, trainer in self.adapters.items():
            for n, updates in ((16, 2), (128, 300)):
                env, agent, tag = self.documents(arm, n, updates)
                self.check(arm, env, agent, tag, n, updates)
            for n, updates in ((16, 300), (128, 2), (128, 600), (512, 300)):
                with self.assertRaises(AssertionError):
                    trainer.arguments_ok(n, updates, f"20260917-slip-{arm}-bad")

    def test_final_is_predeclared_4545_with_all17_adam91120(self):
        trainer = self.adapters["B"]
        candidate = copy.deepcopy(self.parent)
        candidate.update(iter=4545, adam_steps=[91120] * 17, sha256="a" * 64)
        trainer.check_final(candidate, self.parent, 300)
        candidate["adam_steps"][4] = 85120
        with self.assertRaises(AssertionError):
            trainer.check_final(candidate, self.parent, 300)
        candidate.update(iter=4247, adam_steps=[85160] * 17)
        trainer.check_final(candidate, self.parent, 2)

    def test_wrong_parent_load_or_ppohyperparameter_rejected(self):
        for key, value in (("load_run", "smoke-weights"), ("load_checkpoint", "model_3947.pt"),
                           ("seed", "43")):
            env, agent, tag = self.documents("B")
            agent[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check("B", env, agent, tag)
        env, agent, tag = self.documents("B")
        agent["algorithm"]["learning_rate"] = "0.001"
        with self.assertRaises(AssertionError):
            self.check("B", env, agent, tag)

    def test_other_rewards_commands_push_or_physics_cannot_change(self):
        for path, value in ((["rewards", "balanced_duration", "weight"], "-20.0"),
                            (["rewards", "foot_slip", "params", "threshold"], "5.0"),
                            (["commands", "base_velocity", "rel_standing_envs"], "0.1"),
                            (["events", "push_robot"], "null"),
                            (["actions", "joint_pos", "scale"], "0.5")):
            env, agent, tag = self.documents("B")
            node = env
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(AssertionError):
                self.check("B", env, agent, tag)

    def test_cross_arm_environment_rejected(self):
        env, agent, _ = self.documents("B", 16, 2)
        tag = "20260917-slip-A-unit-test"
        agent["run_name"] = tag
        with self.assertRaises(AssertionError):
            self.check("A", env, agent, tag, 16, 2)

    def test_source_snapshot_matches_arms_and_preserves_exports(self):
        a = self.adapters["A"].all_sources()
        self.assertEqual(a, self.adapters["B"].all_sources())
        paths = {row["path"] for row in a}
        for path in (entry.PARENT, entry.PARENT_RECEIPT, entry.TEMPLATE, entry.VERIFIER,
                     entry.ROOT / "scripts/run_commonphysics_slip_adaptation.ps1"):
            self.assertIn(str(path.resolve()), paths)

    def test_formal_gate_rejects_absent_or_wrong_report(self):
        with self.assertRaises(ValueError):
            entry.formal_prerequisite(None)
        # Even an existing incorrectly named JSON receipt cannot substitute.
        with self.assertRaises(ValueError):
            entry.formal_prerequisite(entry.PARENT_RECEIPT)

    def prerequisite_fixture(self):
        """Only mocked unit-test evidence: no reports/files are created or run."""
        path = entry.ROOT / "evaluations/unit-test-only-do-not-create/continuous_flow_report.json"
        header = {"protocol_version": entry.CONTINUOUS_PROTOCOL, "passed": True,
                  "status": "diagnostic_passed_not_promoted", "smoke": False, "num_envs": 1,
                  "refinement": "off", "flow_state": {"phase": "complete"},
                  "actors": {"locomotion": entry.PARENT_SHA}, "command_schedule": {"move": [.5, 0., 0.]},
                  "live_interface_continuity_passed": True, "source_hashes_unchanged_at_end": True,
                  "reset_guard_attempts": []}
        proof = {"protocol": "continuous_flow_trace_audit_v1", "audit_passed": True,
                 "training_prerequisite_eligible": True, "metrics_recomputed_exact": True,
                 "explicit_sensor_identity_verified": True,
                 "checkpoint_sha256": entry.PARENT_SHA, "selected_model": "balanced4246",
                 "promotion_performed": False, "report_path": str(path.resolve()),
                 "report_sha256": "a" * 64, "source_sha256": "b" * 64,
                 "source_and_artifacts_sha256": {str(path.resolve()): "a" * 64,
                                                str(entry.CONTINUOUS_GATE.resolve()): "b" * 64,
                                                str((path.parent / "full_trace.jsonl").resolve()): "c" * 64}}
        return path, header, proof

    def run_mock_prerequisite(self, path, header, proof, digest_override=None):
        audit = Mock(return_value=proof)
        def digest(candidate):
            name = Path(candidate).name
            return {"continuous_flow_report.json": "a" * 64,
                    "audit_continuous_flow.py": digest_override or "b" * 64,
                    "full_trace.jsonl": "c" * 64}[name]
        module = SimpleNamespace(audit=audit)
        spec = SimpleNamespace(name="_slip_continuous_gate_mock", loader=SimpleNamespace(exec_module=lambda _: None))
        with patch.object(entry, "CONTINUOUS_GATE_SHA", "b" * 64), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(entry, "read_json", return_value=header), patch.object(entry, "sha", side_effect=digest), \
             patch.object(entry.importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(entry.importlib.util, "module_from_spec", return_value=module):
            result = entry.formal_prerequisite(path)
        return result, audit

    def test_formal_calls_pinned_audit_not_just_pass_label(self):
        path, header, proof = self.prerequisite_fixture()
        result, invoked = self.run_mock_prerequisite(path, header, proof)
        self.assertEqual(result, proof)
        invoked.assert_called_once_with(path.resolve())

    def test_old_smoke_or_failed_continuous_header_cannot_gate(self):
        for field, value in (("protocol_version", "continuous_recovery_flow_diagnostic_v1"),
                             ("protocol_version", "continuous_recovery_flow_diagnostic_v2_quiet_ready"),
                             ("smoke", True), ("passed", False), ("live_interface_continuity_passed", False),
                             ("source_hashes_unchanged_at_end", False), ("reset_guard_attempts", [{}])):
            path, header, proof = self.prerequisite_fixture()
            header[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_mock_prerequisite(path, header, proof)

    def test_wrong_auditor_sha_fails_before_loading(self):
        path, header, proof = self.prerequisite_fixture()
        with self.assertRaises(ValueError):
            self.run_mock_prerequisite(path, header, proof, digest_override="0" * 64)
        with patch.object(entry, "CONTINUOUS_GATE_SHA", None), patch.object(Path, "is_file", return_value=True), self.assertRaises(ValueError):
            entry.formal_prerequisite(path)

    def test_forged_audit_success_fields_do_not_qualify(self):
        for field, value in (("audit_passed", False), ("training_prerequisite_eligible", False),
                             ("explicit_sensor_identity_verified", False),
                             ("metrics_recomputed_exact", False), ("checkpoint_sha256", "0" * 64),
                             ("report_sha256", "0" * 64), ("source_sha256", "0" * 64),
                             ("selected_model", "control3947"), ("promotion_performed", True)):
            path, header, proof = self.prerequisite_fixture()
            proof[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_mock_prerequisite(path, header, proof)

    def test_nested_artifact_omission_or_drift_rejected(self):
        for mode in ("no_map", "no_report", "wrong_trace_hash", "C_drive"):
            path, header, proof = self.prerequisite_fixture()
            inventory = proof["source_and_artifacts_sha256"]
            if mode == "no_map":
                del proof["source_and_artifacts_sha256"]
            elif mode == "no_report":
                del inventory[str(path.resolve())]
            elif mode == "wrong_trace_hash":
                inventory[str((path.parent / "full_trace.jsonl").resolve())] = "0" * 64
            else:
                inventory["C:/forbidden/source.py"] = "a" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_mock_prerequisite(path, header, proof)

    def test_all_audited_artifacts_enter_start_and_end_source_snapshot(self):
        path, _, proof = self.prerequisite_fixture()
        audit_records = [{"path": name, "sha256": digest} for name, digest in proof["source_and_artifacts_sha256"].items()]
        with patch.object(entry, "formal_prerequisite", return_value=copy.deepcopy(proof)), \
             patch.object(entry, "continuous_source_records", return_value=audit_records):
            trainer = entry.adapter("A", proof)
            # Mutation of the caller's original dictionary must not change the
            # closure's selected evidence or its replayable training ledger.
            proof["selected_model"] = "caller-mutated"
            snapshot = trainer.all_sources()
            for record in audit_records:
                self.assertIn(record, snapshot)
            self.assertEqual(trainer.STATE["continuous_prerequisite"]["selected_model"], "balanced4246")

    def test_exact_load_and_real_step_guards_retained(self):
        text = entry.program("B")
        for required in ('equal_tree(runner.alg.policy.state_dict()',
                         'equal_tree(runner.alg.optimizer.state_dict()',
                         'runner.alg.optimizer.param_groups[0]["lr"] == PARENT_LR',
                         'verify_smoke(args.smoke_receipt, sources)',
                         'int(env.unwrapped.common_step_counter) - STATE["initial_common_steps"] == 24 * args.max_iterations'):
            self.assertIn(required, text)

    def test_template_hash_and_anchor_drift_fail_closed(self):
        with self.assertRaises(ValueError):
            entry.replace_once("twice twice", "twice", "one")
        with self.assertRaises(ValueError):
            entry.program("unknown")
        with patch.object(entry, "TEMPLATE_SHA", "0" * 64), self.assertRaises(ValueError):
            entry.program("A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
