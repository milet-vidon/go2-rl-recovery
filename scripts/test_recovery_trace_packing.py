"""CPU-only equivalence tests for recovery trace packing; no Isaac import."""

import ast
import csv
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "scripts/evaluate_go2_recovery.py"
ARCHIVED_TRACE = (ROOT / "evaluations/20260916-bank4899-heldout"
                  / "model_4899.pt_recovery_trace.csv")
SPEC = importlib.util.spec_from_file_location("recovery_math", ROOT / "src/go2_recovery/recovery_math.py")
math_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_module)
upright_error_squared = math_module.upright_error_squared


def _stance_geometry(env):
    return env.test_geometry


def original_trace_row(env, foot_ids, pose_class, step, stable_steps):
    """Frozen pre-packing implementation, independent of the current source."""
    asset = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    gravity = asset.data.projected_gravity_b[0]
    base_id = sensor.find_bodies("base")[0]
    geometry_ok, foot_b, knee_b, delta = _stance_geometry(env)
    row = {
        "pose": pose_class, "time_s": (step + 1) * env.step_dt, "trial": 0,
        "height": float(asset.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]),
        "roll_deg": float(torch.atan2(gravity[1], -gravity[2]) * 180 / torch.pi),
        "pitch_deg": float(torch.atan2(gravity[0], -gravity[2]) * 180 / torch.pi),
        "gravity_error": float(torch.sqrt(upright_error_squared(gravity))),
        "speed": float(asset.data.root_lin_vel_w[0].norm()),
        "angular_speed": float(asset.data.root_ang_vel_w[0].norm()),
        "feet_contact": int((sensor.data.net_forces_w[0, foot_ids, 2] > 5).sum()),
        "base_contact": bool((sensor.data.net_forces_w[0, base_id].norm(dim=-1) > 1).any()),
        "joint_rms": float((asset.data.joint_pos[0] - asset.data.default_joint_pos[0]).square().mean().sqrt()),
        "stable_hold_s": float(stable_steps[0]) * env.step_dt,
        "geometry_ok": bool(geometry_ok[0]),
    }
    for i, name in enumerate(("FL", "FR", "RL", "RR")):
        for j, axis in enumerate("xyz"):
            row[f"{name}_foot_{axis}_b"] = float(foot_b[0, i, j])
        row[f"{name}_knee_y_b"] = float(knee_b[0, i, 1])
    for i, name in enumerate(asset.joint_names):
        row[name] = float(asset.data.joint_pos[0, i])
        row[name + "_offset"] = float(delta[0, i])
    return row


def current_trace_row():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_trace_row")
    scope = {"torch": torch, "upright_error_squared": upright_error_squared,
             "_stance_geometry": _stance_geometry}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[node.name], node


class Scene(dict):
    pass


def fixture(dtype, case):
    # A private CPU generator keeps test setup and trace calls off the CUDA device.
    rng = torch.Generator(device="cpu").manual_seed(42 + case)

    def random(*shape):
        return torch.randn(shape, generator=rng, dtype=dtype)

    data = SimpleNamespace(
        projected_gravity_b=random(3, 3), root_pos_w=random(3, 3),
        root_lin_vel_w=random(3, 3), root_ang_vel_w=random(3, 3),
        joint_pos=random(3, 12), default_joint_pos=random(3, 12),
    )
    joint_names = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                   for leg in ("FL", "FR", "RL", "RR")]
    forces = random(3, 5, 3) * 10
    forces[0, :4, 2] = torch.tensor([5.0, 5.001, 4.999, -10.0], dtype=dtype)
    forces[0, 4] = torch.tensor([0.0, 0.0, (1.0, .999, 1.001)[case % 3]], dtype=dtype)
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces),
                             find_bodies=lambda pattern: ([4], ["base"]))
    scene = Scene(robot=SimpleNamespace(data=data, joint_names=joint_names))
    scene.sensors = {"contact_forces": sensor}
    scene.env_origins = random(3, 3)
    geometry = (torch.tensor([case % 2 == 0, False, True]), random(3, 4, 3),
                random(3, 4, 3), data.joint_pos - data.default_joint_pos)
    env = SimpleNamespace(scene=scene, step_dt=(.02, .005, .03)[case % 3], test_geometry=geometry)
    # Exceed both float32's exact integer range and, in two cases, float64's.
    large_counts = (2 ** 24 + 1, 2 ** 53 - 1, 2 ** 53 + 1, torch.iinfo(torch.int64).max)
    stable_steps = torch.tensor([large_counts[case] if case < 4 else case, 0, 17], dtype=torch.int64)
    return env, [2, 0, 3, 1], ("side", "upside_down")[case % 2], case, stable_steps


def csv_bytes(row):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode("utf-8")


class TracePackingTests(unittest.TestCase):
    def test_fifty_states_exact_values_types_order_and_csv(self):
        packed, _ = current_trace_row()
        cases = 0
        for dtype in (torch.float32, torch.float64):
            for case in range(25):
                with self.subTest(dtype=dtype, case=case):
                    args = fixture(dtype, case)
                    before_rng = torch.get_rng_state().clone()
                    expected = original_trace_row(*args)
                    actual = packed(*args)
                    self.assertEqual(list(actual), list(expected))
                    for key in expected:
                        self.assertIs(type(actual[key]), type(expected[key]), key)
                        self.assertEqual(actual[key], expected[key], key)
                    self.assertEqual(csv_bytes(actual), csv_bytes(expected))
                    self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
                    cases += 1
        self.assertEqual(cases, 50)

    def test_two_explicit_cpu_transfers(self):
        _, node = current_trace_row()
        calls = [n.func.attr for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(calls.count("cpu"), 2)
        self.assertEqual(calls.count("tolist"), 2)
        self.assertNotIn("item", calls)

    def test_archived_trial_zero_first_and_last_schema(self):
        if not ARCHIVED_TRACE.is_file():
            self.skipTest(f"Historical evaluation artifact not available: {ARCHIVED_TRACE}")
        with ARCHIVED_TRACE.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            first = next(reader)
            last = first
            for last in reader:
                pass
        packed, _ = current_trace_row()
        generated = packed(*fixture(torch.float32, 0))
        # The evaluator appends this bank-only field outside _trace_row.
        generated["bank_state_id"] = 132
        self.assertEqual(list(generated), fields)
        self.assertEqual(first["pose"], "side")
        self.assertEqual(last["pose"], "upside_down")
        for archived in (first, last):
            with self.subTest(pose=archived["pose"], time_s=archived["time_s"]):
                self.assertEqual(list(archived), fields)
                self.assertEqual(archived["trial"], "0")
                for key, example in generated.items():
                    value = archived[key]
                    self.assertIsInstance(value, str)
                    if type(example) is bool:
                        self.assertIn(value, ("True", "False"), key)
                    elif type(example) is int:
                        self.assertEqual(str(int(value)), value, key)
                    elif type(example) is float:
                        self.assertTrue(math.isfinite(float(value)), key)
                    else:
                        self.assertIs(type(example), str)
                        self.assertTrue(value, key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
