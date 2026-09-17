"""CPU-only transition math/source contracts. Never import the AppLauncher entry."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "go2_recovery"))
from handoff_transition_math import TransitionNeighborhood, ramp_step_count, transition_action

OLD = ROOT / "scripts" / "evaluate_go2_recovery.py"
NEW = ROOT / "scripts" / "evaluate_handoff_transition.py"
SHA = "4024ba713e3ed616225d64e1d59de97fd31c8369bba83ac3f4235d93552e9e2f"


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)}


def extract(names):
    nodes = functions(NEW)
    namespace = {"torch": torch, "Path": Path, "hashlib": hashlib, "json": json,
                 "os": os, "BASELINE_EVALUATOR_PATH": OLD, "BASELINE_EVALUATOR_SHA256": SHA}
    exec(compile(ast.Module(body=[nodes[name] for name in names], type_ignores=[]), str(NEW), "exec"), namespace)
    return namespace


def inputs():
    return [torch.arange(36, dtype=torch.float32).reshape(3, 12) / 10,
            torch.arange(36, dtype=torch.float32).reshape(3, 12) / -7,
            torch.full((3, 12), 2.0), torch.tensor([False, True, True]),
            torch.tensor([False, True, False]), torch.full((3, 12), -3.0),
            torch.tensor([0, 0, 4], dtype=torch.long)]


class MathTests(unittest.TestCase):
    def test_steps(self):
        self.assertEqual([ramp_step_count(t, .02) for t in (0, .2, .5)], [0, 10, 25])

    def test_invalid_steps(self):
        for seconds, dt in [(True, .02), (.3, .02), (float("nan"), .02), (.2, 0),
                            (.2, float("inf")), (.2, .03), (.2, True)]:
            with self.subTest(seconds=seconds, dt=dt), self.assertRaises(ValueError):
                ramp_step_count(seconds, dt)

    def test_hard_exact(self):
        args = inputs()
        args[-1].zero_()
        before_rng = torch.random.get_rng_state().clone()
        actual, anchor, count, alpha = transition_action(*args, 0)
        self.assertTrue(torch.equal(actual, torch.where(args[3][:, None], args[1], args[0])))
        self.assertTrue(torch.equal(anchor[1], args[2][1]))
        self.assertEqual(count.tolist(), [0, 0, 0])
        self.assertEqual(alpha.tolist(), [0, 1, 1])
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))

    def test_masks_edge_anchor_and_no_mutation(self):
        args = inputs()
        originals = [value.clone() for value in args]
        output, anchor, progress, alpha = transition_action(*args, 10)
        self.assertTrue(torch.equal(output[0], args[0][0]))
        self.assertTrue(torch.equal(anchor[1], args[2][1]))
        self.assertTrue(torch.equal(anchor[2], args[5][2]))
        self.assertEqual(progress.tolist(), [0, 1, 5])
        self.assertTrue(torch.allclose(output[1], .9 * args[2][1] + .1 * args[1][1]))
        self.assertTrue(torch.allclose(output[2], .5 * args[5][2] + .5 * args[1][2]))
        for old, value in zip(originals, args):
            self.assertTrue(torch.equal(old, value))

    def test_both_ramps_end_exactly_and_anchor_stays_fixed(self):
        for duration in (.2, .5):
            total = ramp_step_count(duration, .02)
            roll = torch.full((1, 12), -2.0)
            previous = torch.full((1, 12), 3.0)
            anchor = torch.zeros_like(previous)
            progress = torch.zeros(1, dtype=torch.long)
            switched = torch.ones(1, dtype=torch.bool)
            for step in range(1, total + 4):
                stand = torch.full((1, 12), step / 7)
                action, anchor, progress, alpha = transition_action(
                    roll, stand, previous, switched, torch.tensor([step == 1]), anchor, progress, total)
                self.assertTrue(torch.equal(anchor, torch.full_like(anchor, 3)))
                self.assertEqual(int(progress[0]), min(step, total))
                if step >= total:
                    self.assertTrue(torch.equal(action, stand))
                    self.assertEqual(float(alpha[0]), 1)
                previous = action.clone()

    def test_invalid_shapes_dtypes_and_nonfinite(self):
        changes = [(0, torch.zeros(3, 11)), (1, torch.zeros(3, 12, dtype=torch.float64)),
                   (2, torch.full((3, 12), float("nan"))), (3, torch.ones(3)),
                   (4, torch.zeros(2, dtype=torch.bool)), (6, torch.zeros(3, dtype=torch.int32))]
        for index, replacement in changes:
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                args = inputs()
                args[index] = replacement
                transition_action(*args, 10)

    def test_invalid_counters_and_edges(self):
        for counters in [[1, 0, 4], [0, 1, 4], [0, 0, 11], [0, 0, 0], [0, 0, -1]]:
            args = inputs()
            args[-1] = torch.tensor(counters)
            with self.subTest(counters=counters), self.assertRaises(ValueError):
                transition_action(*args, 10)
        args = inputs()
        args[4][0] = True
        with self.assertRaises(ValueError):
            transition_action(*args, 10)
        with self.assertRaises(ValueError):
            transition_action(*inputs(), True)

    def test_new_edge_does_not_change_previously_latched_neighbor(self):
        args = inputs()
        first = transition_action(*args, 10)
        args[4][1] = False
        args[6][1] = 3
        second = transition_action(*args, 10)
        self.assertTrue(torch.equal(first[0][2], second[0][2]))
        self.assertTrue(torch.equal(first[0][0], second[0][0]))

    def test_neighborhood_complete_denominator_and_bounds(self):
        trace = TransitionNeighborhood(2, 3, 3)
        for step in range(10):
            for trial in range(2):
                if trace.wants_row(trial, step):
                    phase = "roll" if trial == 1 or step < 4 else "raw_ramp" if step < 6 else "stand"
                    trace.record(trial, step, {"step": step, "phase": phase, "policy_time_s": .02 * step,
                                              "alpha": 0 if phase == "roll" else .5 if phase == "raw_ramp" else 1},
                                 just_switched=(trial == 0 and step == 4))
        report = trace.report()
        self.assertEqual(len(report), 2)
        self.assertEqual([row["step"] for row in report[0]["rows"]], list(range(1, 8)))
        self.assertEqual([row["step"] for row in report[1]["rows"]], [7, 8, 9])
        self.assertEqual([event["to"] for event in report[0]["events"]], ["raw_ramp", "stand"])
        self.assertFalse(report[1]["triggered"])
        self.assertTrue(report[1]["untriggered_tail_only"])
        with self.assertRaises(ValueError):
            trace.record(0, 10, {"phase": "stand"}, True)


class SourceAndBoundaryTests(unittest.TestCase):
    def test_original_collector_unchanged(self):
        self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(), SHA)

    def test_original_state_physics_acceptance_helpers_identical(self):
        old, new = functions(OLD), functions(NEW)
        for name in ("_pose_quaternions", "_set_pose_class", "_stable_stand", "_stance_geometry",
                     "_start_snapshot", "_settle_nominal_pose", "_reset_policy_history", "_select_bank_states",
                     "_set_bank_states", "_validate_bank_environment", "_trace_row", "_mark_action_mode",
                     "_mark_target_action_report"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]))

    def test_gate_call_is_identical_and_no_new_resets(self):
        old, new = functions(OLD)["main"], functions(NEW)["main"]
        calls = lambda node, name: [ast.dump(call) for call in ast.walk(node) if isinstance(call, ast.Call)
                                   and isinstance(call.func, ast.Name) and call.func.id == name]
        for name in ("update_handoff_gate", "_reset_policy_history", "_set_pose_class", "_set_bank_states",
                     "_settle_nominal_pose", "_stable_stand", "_stance_geometry"):
            self.assertEqual(calls(old, name), calls(new, name), name)
        for name in ("_transition_physical_rows", "_transition_rows_before"):
            for call in ast.walk(functions(NEW)[name]):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    self.assertNotIn(call.func.attr, ("step", "reset", "write_root_pose_to_sim",
                                                      "write_joint_state_to_sim", "set_joint_position_target"))

    def test_protocol_not_legacy_acceptance(self):
        text = NEW.read_text(encoding="utf-8")
        self.assertIn('report["protocol_version"] = "handoff_transition_experimental_v1"', text)
        self.assertIn('report["acceptance_eligible"] = False', text)
        self.assertIn('report["training_collection_eligible"] = False', text)
        self.assertIn('EXPERIMENTAL RAW RAMP', text)
        self.assertIn('"retry_enabled": False', text)
        self.assertIn('"actual_action_and_history_asserted_every_step": True', text)

    def test_cli_gates_and_wrapper_invocation_exception(self):
        check = extract(["_validate_transition_cli"])["_validate_transition_cli"]
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluations", prefix="cpu-transition-test-") as directory:
            args = argparse.Namespace(transition_mode="hard", ramp_seconds=0, state_bank_split="heldout",
                                      trials=20, horizon_s=8, hold_s=3, min_contacts=4, settle_s=1,
                                      output_dir=Path(directory))
            check(args)
            (Path(directory) / "invocation.json").write_text("{}", encoding="utf-8")
            (Path(directory) / "simulator.log").write_text("wrapper startup", encoding="utf-8")
            check(args)
            for field, value in [("transition_mode", "raw_ramp"), ("ramp_seconds", .2),
                                 ("state_bank_split", "train"), ("trials", 21), ("hold_s", 2),
                                 ("horizon_s", 9), ("min_contacts", 2), ("output_dir", Path("C:/not-allowed"))]:
                original = getattr(args, field)
                setattr(args, field, value)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check(args)
                setattr(args, field, original)
            args.transition_mode, args.ramp_seconds = "raw_ramp", .2
            check(args)
            (Path(directory) / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                check(args)

    def test_json_write_is_durable_new_and_nonfinite_rejected(self):
        write = extract(["_write_json_new"])["_write_json_new"]
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluations", prefix="cpu-transition-test-") as directory:
            output = Path(directory) / "report.json"
            write(output, {"real": True})
            self.assertEqual(json.loads(output.read_text()), {"real": True})
            with self.assertRaises(FileExistsError):
                write(output, {"overwrite": True})
            with self.assertRaises(ValueError):
                write(Path(directory) / "bad.json", {"nan": float("nan")})

    def test_actual_snapshot_fields_and_no_alias_to_live_tensors(self):
        ns = extract(["_transition_physical_rows", "_transition_rows_before"])
        class Scene(dict):
            pass
        a = types.SimpleNamespace(root_pos_w=torch.tensor([[1., 2., .3]]),
            root_quat_w=torch.tensor([[1., 0, 0, 0]]), root_lin_vel_w=torch.zeros(1, 3),
            root_ang_vel_w=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
            joint_pos=torch.ones(1, 12), joint_vel=torch.zeros(1, 12),
            projected_gravity_b=torch.tensor([[0., 0, -1]]), applied_torque=torch.arange(12.).reshape(1, 12),
            joint_pos_target=torch.ones(1, 12), default_joint_pos=torch.zeros(1, 12),
            soft_joint_pos_limits=torch.tensor([[[-2., 2.]]] * 12).reshape(1, 12, 2))
        sensor = types.SimpleNamespace(find_bodies=lambda name: ([4], ["base"]),
                                       data=types.SimpleNamespace(net_forces_w=torch.zeros(1, 5, 3)))
        scene = Scene(robot=types.SimpleNamespace(data=a))
        scene.env_origins = torch.tensor([[1., 2., 0]])
        scene.sensors = {"contact_forces": sensor}
        env = types.SimpleNamespace(scene=scene, num_envs=1, step_dt=.02,
            action_manager=types.SimpleNamespace(action=torch.ones(1, 12), prev_action=torch.zeros(1, 12)),
            command_manager=types.SimpleNamespace(get_command=lambda name: torch.zeros(1, 3)))
        zero, mask = torch.zeros(1, 12), torch.zeros(1, dtype=torch.bool)
        row = ns["_transition_rows_before"](env, [0, 1, 2, 3], {"policy": torch.zeros(1, 48)},
            0, torch.zeros(1, dtype=torch.long), mask, mask, zero, zero, zero, zero,
            torch.zeros(1), zero, TransitionNeighborhood(1))[0]
        a.joint_pos_target.fill_(99)
        self.assertEqual(row["previous_executed_joint_target_rad"], [1.] * 12)
        self.assertEqual(row["before"]["root_position_local_m"][:2], [0, 0])
        self.assertEqual(row["before"]["applied_torque_joint_abs_max_at_control_boundary_nm"], 11)
        self.assertEqual(row["phase"], "roll")
        a.applied_torque[0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            ns["_transition_physical_rows"](env, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
