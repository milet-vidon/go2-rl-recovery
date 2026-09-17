"""Read-only AST/unit checks for handoff collection metadata; no simulator import."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


EVALUATOR = Path(__file__).with_name("evaluate_go2_recovery.py")


class RecordedArray:
    """Small immutable-source stand-in for the tensor reads used by metadata."""

    def __init__(self, values):
        self.values = copy.deepcopy(values)

    def __getitem__(self, index):
        return RecordedArray(self.values[index])

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


def metadata_fields(required_key):
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            fields = {key.value: value for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            if required_key in fields:
                found.append(fields)
    if len(found) != 1:
        raise AssertionError(f"Expected one metadata dictionary containing {required_key}")
    return found[0]


def evaluate_read(node, scope):
    expression = ast.Expression(body=node)
    return eval(compile(expression, str(EVALUATOR), "eval"), {"__builtins__": {}}, scope)


class HandoffCollectionFieldTests(unittest.TestCase):
    def setUp(self):
        self.records = metadata_fields("previous_raw_action")
        self.limits = metadata_fields("soft_joint_limits_rad")

    def test_previous_previous_is_real_second_history_and_correct_trial(self):
        current = RecordedArray([[10.0] * 12, [20.0] * 12])
        previous = RecordedArray([[-10.0] * 12, [-20.0] * 12])
        manager = SimpleNamespace(action=current, prev_action=previous)
        scope = {"env": SimpleNamespace(unwrapped=SimpleNamespace(action_manager=manager)), "i": 1}
        result = evaluate_read(self.records["previous_previous_raw_action"], scope)
        self.assertEqual(result, [-20.0] * 12)
        self.assertNotEqual(result, evaluate_read(self.records["previous_raw_action"], scope))
        self.assertEqual(previous.values, [[-10.0] * 12, [-20.0] * 12])

    def test_command_is_actual_manager_command_not_assumed_zero_or_obs(self):
        calls = []
        command = RecordedArray([[0.1, 0.2, 0.3], [0.4, -0.5, 0.6]])

        def get_command(name):
            calls.append(name)
            return command

        scope = {"env": SimpleNamespace(unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_command=get_command))), "i": 1}
        self.assertEqual(evaluate_read(self.records["velocity_command_b"], scope), [0.4, -0.5, 0.6])
        self.assertEqual(calls, ["base_velocity"])
        self.assertEqual(command.values, [[0.1, 0.2, 0.3], [0.4, -0.5, 0.6]])

    def test_hard_limits_use_actual_joint_pos_limits_not_soft_or_defaults(self):
        actual = [[-2.0, 2.0] for _ in range(12)]
        asset = SimpleNamespace(joint_pos_limits=RecordedArray([actual]),
                                soft_joint_pos_limits=RecordedArray([[[-1.8, 1.8]] * 12]),
                                default_joint_pos_limits=RecordedArray([[[-3.0, 3.0]] * 12]))
        self.assertEqual(evaluate_read(self.limits["hard_joint_limits_rad"], {"a": asset}), actual)
        self.assertEqual(asset.joint_pos_limits.values, [actual])

    def test_new_expressions_are_only_exact_read_operations(self):
        expressions = {
            "previous_previous_raw_action": "env.unwrapped.action_manager.prev_action[i].cpu().tolist()",
            "velocity_command_b": 'env.unwrapped.command_manager.get_command("base_velocity")[i].cpu().tolist()',
            "hard_joint_limits_rad": "a.joint_pos_limits[0].cpu().tolist()",
        }
        fields = {**self.records, **self.limits}
        for name, source in expressions.items():
            with self.subTest(field=name):
                expected = ast.parse(source, mode="eval").body
                self.assertEqual(ast.dump(fields[name]), ast.dump(expected))

    def test_earlier_policy_input_and_frame_fields_remain(self):
        self.assertTrue({"root_position_local_m", "root_quaternion_wxyz",
                         "root_linear_velocity_w_m_s", "root_angular_velocity_w_rad_s",
                         "joint_positions_rad", "joint_velocities_rad_s", "policy_observation",
                         "previous_raw_action", "previous_executed_joint_target_rad",
                         "stand_actor_raw_action", "roll_actor_raw_action"} <= self.records.keys())
        self.assertTrue({"joint_names", "default_joint_positions_rad", "soft_joint_limits_rad"}
                        <= self.limits.keys())


if __name__ == "__main__":
    unittest.main()
