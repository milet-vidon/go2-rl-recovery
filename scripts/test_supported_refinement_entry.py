"""CPU source/routing/stream tests; no Isaac, GPU execution or performance claim."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import torch

import evaluate_supported_refinement as entry
from supported_refinement_gate import advance_after_completed_interval


def code_fragment(source, start, stop):
    begin = source.index(start)
    end = source.index(stop, begin)
    return compile(textwrap.dedent(source[begin:end]), "<actual-generated-fragment>", "exec")


class RefinementSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = entry.base.build_source()
        cls.generated = entry.build_source()
        cls.tree = ast.parse(cls.generated)

    def test_exactly_predeclared_unique_anchors(self):
        tree = ast.parse(Path(entry.__file__).read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_source")
        assignment = next(n for n in function.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "edits" for t in n.targets))
        edits = ast.literal_eval(assignment.value)
        self.assertEqual(len(edits), 13)
        self.assertEqual([old.splitlines()[0].strip() for old, _ in edits], [
            'selected = [i for i in range(env.num_envs) if neighborhood.wants_row(i, step)]',
            'stand_policy = stand_runner.get_inference_policy(device=env.device)',
            '_validate_mirror_observation_contract(env.unwrapped, (policy_nn, stand_runner.alg.policy))',
            'bank = None',
            'legacy_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)',
            'if action.shape != a.joint_pos.shape:',
            'row["phase"] = "startup_stand"',
            'stable_steps = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))',
            '"stand" if bool(switched[0]) else "roll"))',
            'finally:',
            'report["startup_experiment"] = startup_experiment',
            '_write_json_new(report_path, report)',
            'heading = f"EXPERIMENTAL STARTUP {args_cli.startup_mode} / MIRROR {args_cli.mirror_mode}"',
        ])
        expected = self.original
        for old, new in edits:
            self.assertEqual(expected.count(old), 1)
            expected = expected.replace(old, new, 1)
        self.assertEqual(expected, self.generated)
        compile(self.generated, "<verified-generated>", "exec")

    def test_physics_reset_criterion_and_real48_helpers_untouched(self):
        original = ast.parse(self.original)
        old = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        new = {node.name: node for node in self.tree.body if isinstance(node, ast.FunctionDef)}
        changed = {name for name in old if ast.dump(old[name]) != ast.dump(new[name])}
        self.assertEqual(changed, {"main", "_transition_rows_before", "_annotate"})
        self.assertEqual(self.generated.count('_reset_policy_history('), self.original.count('_reset_policy_history('))
        self.assertEqual(self.generated.count('env.reset()'), self.original.count('env.reset()'))
        self.assertIn(entry.base.ROLL_SHA, self.generated)
        self.assertIn(entry.base.STAND_SHA, self.generated)

    def test_third_actor_inside_same_cpu_cuda_rng_fork(self):
        forks = [node for node in ast.walk(self.tree) if isinstance(node, ast.With)
                 and any('torch.random.fork_rng' in ast.unparse(item.context_expr) for item in node.items)]
        self.assertEqual(len(forks), 1)
        body = ast.unparse(forks[0])
        self.assertIn('stand_runner = OnPolicyRunner', body)
        self.assertIn('refinement_runner = OnPolicyRunner', body)
        self.assertIn('devices=rng_devices', body)
        self.assertIn('if policy_device.type == "cuda" else []', self.generated)
        class RandomRunner:
            def __init__(self, *args, **kwargs):
                torch.rand(23)
            def load(self, *args, **kwargs):
                torch.rand(11)
            def get_inference_policy(self, **kwargs):
                torch.rand(7)
                return lambda obs: obs
        namespace = dict(torch=torch, rng_devices=[], OnPolicyRunner=RandomRunner,
                         env=NS(device="cpu"), agent_cfg=NS(to_dict=lambda: {}),
                         args_cli=NS(device="cpu", stand_checkpoint="old-frozen"),
                         REFINEMENT_RECEIPT={"checkpoint": "new-frozen"})
        before = torch.random.get_rng_state().clone()
        module = ast.fix_missing_locations(ast.Module(body=forks, type_ignores=[]))
        exec(compile(module, "<actual-rng-block>", "exec"), namespace)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_after_step_full_strict_gate_only_affects_next_action(self):
        begin = self.generated.index('for step in range(steps):')
        action = self.generated.index('refinement_active = refinement_latched', begin)
        physical = self.generated.index('obs, _, dones, _ = env.step(action)', action)
        strict = self.generated.index('stable &= geometry_ok & ~base_contact & current_support', physical)
        gate = self.generated.index('refinement_count, refinement_latched, refinement_edge =', strict)
        stream = self.generated.index('refinement_recorder.record(', gate)
        self.assertLess(action, physical)
        self.assertLess(physical, strict)
        self.assertLess(strict, gate)
        self.assertLess(gate, stream)
        self.assertIn('stable & stand_active, refinement_count, refinement_latched', self.generated)
        self.assertIn('refinement_active & stable', self.generated)
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "advance_after_completed_interval"]
        self.assertEqual(len(calls), 1)

    def test_every_trial_before_after_is_real_and_history_unchanged(self):
        self.assertIn('selected = list(range(env.num_envs))', self.generated)
        self.assertIn('refinement_action = refinement_policy(obs)', self.generated)
        self.assertIn('torch.equal(obs["policy"], real_policy_observation_before)', self.generated)
        self.assertIn('torch.equal(env.unwrapped.action_manager.prev_action, previous_raw_action)', self.generated)
        self.assertIn('actual_previous_raw_action=actual_history[i]', self.generated)
        self.assertIn('refinement_recorder.close()\n        env.close()', self.generated)
        for bad in ('refinement_action.copy_', 'obs["policy"] = refinement'):
            self.assertNotIn(bad, self.generated)
        rollout = self.generated[self.generated.index('for step in range(steps):'):]
        self.assertNotIn('action_manager.reset()', rollout)
        self.assertEqual(self.generated.count('action_manager.reset()'), self.original.count('action_manager.reset()'))


class RefinementRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = entry.build_source()
        cls.actor = code_fragment(source, '                        recovery_action = action',
                                  '                        if action.shape != a.joint_pos.shape:')
        cls.gate = code_fragment(source,
            '                refinement_count, refinement_latched, refinement_edge = advance_after_completed_interval(',
            '                refinement_recorder.record(')

    def namespace(self, mode="supported", n=20):
        observation = torch.arange(n * 48, dtype=torch.float32).reshape(n, 48) / 100
        raw = torch.arange(n * 12, dtype=torch.float32).reshape(n, 12) / 100
        return dict(torch=torch, action=raw, original=raw.clone(), obs={"policy": observation},
                    real_policy_observation_before=observation.clone(), refinement_policy=lambda obs: -obs["policy"][:, :12],
                    refinement_latched=torch.zeros(n, dtype=torch.bool), stand_active=torch.ones(n, dtype=torch.bool),
                    REFINEMENT_MODE=mode, refinement_count=torch.zeros(n, dtype=torch.int64),
                    refinement_valid_hold=torch.zeros(n, dtype=torch.int64), stable=torch.ones(n, dtype=torch.bool),
                    env=NS(unwrapped=NS(step_dt=.02)), advance_after_completed_interval=advance_after_completed_interval)

    def test_off_exact_original_action_despite_latched_counter(self):
        ns = self.namespace("off")
        ns["refinement_latched"].fill_(True)
        observed = ns["obs"]["policy"].clone()
        exec(self.actor, ns)
        self.assertTrue(torch.equal(ns["action"], ns["original"]))
        self.assertFalse(bool(ns["refinement_active"].any()))
        self.assertTrue(torch.equal(ns["obs"]["policy"], observed))

    def test_149_intervals_are_insufficient_and_first_action_is150(self):
        ns = self.namespace()
        for step in range(151):
            ns["action"] = ns["original"].clone()
            exec(self.actor, ns)
            self.assertEqual(bool(ns["refinement_active"].all()), step >= 150)
            exec(self.gate, ns)
            if step == 148:
                self.assertEqual(ns["refinement_count"].tolist(), [149] * 20)
                self.assertFalse(bool(ns["refinement_latched"].any()))
            if step == 149:
                self.assertTrue(bool(ns["refinement_latched"].all()))
                self.assertEqual(ns["refinement_valid_hold"].tolist(), [0] * 20)
        self.assertEqual(ns["refinement_valid_hold"].tolist(), [1] * 20)

    def test_mixed_trials_require_original_stand_branch_and_strict_hold(self):
        ns = self.namespace(n=2)
        ns["stand_active"][1] = False
        for step in range(150):
            ns["action"] = ns["original"].clone()
            ns["stable"][0] = step != 149
            exec(self.actor, ns)
            exec(self.gate, ns)
        self.assertEqual(ns["refinement_latched"].tolist(), [False, False])
        self.assertEqual(ns["refinement_count"].tolist(), [0, 0])

    def test_regression_stays_visible_no_retry_and_refinement_hold_resets(self):
        ns = self.namespace()
        ns["refinement_latched"].fill_(True)
        ns["refinement_count"].fill_(150)
        ns["refinement_valid_hold"].fill_(120)
        ns["stable"].fill_(False)
        exec(self.actor, ns)
        exec(self.gate, ns)
        self.assertTrue(bool(ns["refinement_active"].all()))
        self.assertTrue(bool(ns["refinement_latched"].all()))
        self.assertEqual(ns["refinement_valid_hold"].tolist(), [0] * 20)

    def test_illegal_bypass_and_mutating_actor_fail(self):
        ns = self.namespace()
        ns["refinement_latched"].fill_(True)
        ns["stand_active"].fill_(False)
        with self.assertRaisesRegex(RuntimeError, "cannot bypass"):
            exec(self.actor, ns)
        ns = self.namespace()
        def mutating(obs):
            obs["policy"].add_(1.)
            return torch.zeros(20, 12)
        ns["refinement_policy"] = mutating
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            exec(self.actor, ns)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="refinement-entry-cpu-", dir=entry.ROOT.parent / "tmp")
        self.output = Path(self.temporary.name)
        self.recorder = None

    def tearDown(self):
        if self.recorder is not None and not self.recorder.stream.closed:
            self.recorder.close()
        self.temporary.cleanup()

    def create(self, mode="supported", n=20):
        self.recorder = entry.RefinementRecorder(self.output, n, mode, {"cpu_fixture_not_training_proof": True})
        return self.recorder

    def payload(self, step, n=20, active=False):
        rows = {i: {"trial": i, "step": step, "refinement_active": active,
                    "issued_action_target_history_assertions_passed": True} for i in range(n)}
        return ["upside_down", step, rows, torch.ones(n, dtype=torch.bool), torch.zeros(n, 4, 3),
                torch.zeros(n, 4, 3), torch.full((n,), min(step + 1, 150), dtype=torch.int64),
                torch.full((n,), step >= 149, dtype=torch.bool),
                torch.full((n,), max(0, step - 149) if active else 0, dtype=torch.int64),
                torch.tensor([[0., 0., -1.]]).repeat(n, 1)]

    def test_complete550_intervals_and20_trials_streamed(self):
        recorder = self.create()
        for step in range(550):
            recorder.record(*self.payload(step, active=step >= 150))
        recorder.close()
        report = recorder.report()
        self.assertEqual(report["full_measured_rows"], 20 * 550)
        self.assertEqual(report["poses"]["upside_down"]["first_refinement_action_step"], [150] * 20)
        self.assertEqual(report["poses"]["upside_down"]["final_refinement_valid_holds"], 20)
        self.assertEqual(report["trace_sha256"], hashlib.sha256(recorder.path.read_bytes()).hexdigest())
        self.assertEqual(report["gate_source_sha256"], entry.GATE_SHA)
        with recorder.path.open(encoding="utf-8") as stream:
            for step, line in enumerate(stream):
                value = json.loads(line)
                self.assertEqual(value["step"], step)
                self.assertEqual(len(value["rows"]), 20)
                self.assertEqual([row["trial"] for row in value["rows"]], list(range(20)))
                self.assertEqual([row["projected_gravity_b_after"] for row in value["rows"]], [[0., 0., -1.]] * 20)
        self.assertEqual(step, 549)

    def test_empty_incomplete_or_open_trace_never_reports_complete(self):
        recorder = self.create()
        with self.assertRaises(RuntimeError):
            recorder.report()
        recorder.close()
        with self.assertRaises(RuntimeError):
            recorder.report()

    def test_missing_reordered_false_identity_or_false_checks_fail(self):
        recorder = self.create()
        for case in ("missing", "reordered", "identity", "assertion", "shape"):
            values = self.payload(0)
            if case == "missing":
                del values[2][19]
            elif case == "reordered":
                values[1] = 1
            elif case == "identity":
                values[2][0]["trial"] = 1
            elif case == "assertion":
                values[2][0]["issued_action_target_history_assertions_passed"] = False
            else:
                values[4] = torch.zeros(19, 4, 3)
            with self.subTest(case=case), self.assertRaises(ValueError):
                recorder.record(*values)
        self.assertEqual(recorder.rows_written, 0)

    def test_partial_file_is_preserved_but_not_certified(self):
        recorder = self.create()
        recorder.record(*self.payload(0))
        recorder.close()
        self.assertTrue(recorder.path.exists())
        self.assertGreater(recorder.path.stat().st_size, 0)
        with self.assertRaises(RuntimeError):
            recorder.report()

    def test_nan_and_disk_write_failure_are_not_swallowed(self):
        recorder = self.create()
        values = self.payload(0)
        values[4][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            recorder.record(*values)
        with mock.patch.object(recorder.stream, "write", side_effect=OSError("disk fixture")):
            with self.assertRaisesRegex(OSError, "disk fixture"):
                recorder.record(*self.payload(0))
        self.assertEqual(recorder.rows_written, 0)

    def test_refusal_to_overwrite_or_activate_in_off_mode(self):
        recorder = self.create("off")
        with self.assertRaises(FileExistsError):
            entry.RefinementRecorder(self.output, 20, "off", {})
        with self.assertRaisesRegex(ValueError, "OFF"):
            recorder.record(*self.payload(0, active=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
