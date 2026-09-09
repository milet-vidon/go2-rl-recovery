"""Simulator-free regressions for the evaluator's command schedule."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class CommandProtocolTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).with_name('evaluate_go2_stand_walk_stop.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_phase')
        self.config = SimpleNamespace(stand_s=4, walk_s=8, stop_s=6, walk_speed=.5, lateral_speed=0, yaw_rate=0)
        scope = {'args_cli': self.config}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<schedule>', 'exec'), scope)
        self.phase = scope['_phase']

    def test_boundaries(self):
        for t in (0, 3.98):
            self.assertEqual(self.phase(t), ('stand', (0, 0, 0)))
        for t in (4, 11.98):
            self.assertEqual(self.phase(t), ('walk', (.5, 0, 0)))
        self.assertEqual(self.phase(12), ('stop', (0, 0, 0)))

    def test_signed_commands(self):
        self.config.walk_speed = -.3
        self.config.lateral_speed = -.2
        self.config.yaw_rate = -.6
        self.assertEqual(self.phase(6), ('walk', (-.3, -.2, -.6)))
        self.assertEqual(self.phase(12), ('stop', (0, 0, 0)))


if __name__ == '__main__':
    unittest.main()
