"""Video labels must not confuse valid inverted leg shape with recovered stance."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class AnnotationTests(unittest.TestCase):
    def label(self, geometry, hold):
        path = Path(__file__).with_name('evaluate_go2_recovery.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_annotate')
        calls = []
        fake_cv = SimpleNamespace(COLOR_RGB2BGR=1, FONT_HERSHEY_SIMPLEX=0, LINE_AA=0,
                                  cvtColor=lambda frame, code: frame.copy(),
                                  putText=lambda *args: calls.append(args))
        scope = {'np': np, 'cv2': fake_cv, 'args_cli': SimpleNamespace(
            hold_s=3, checkpoint=Path('model_4899.pt'), settle_s=1, state_bank_path=None)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        row = {'geometry_ok': geometry, 'FL_foot_y_b': .16, 'FR_foot_y_b': -.16}
        scope['_annotate'](np.zeros((720, 1280, 3), dtype=np.uint8), 'upside_down', -1, .02, hold, row)
        return calls[2][1], calls[2][5]

    def test_inverted_leg_shape_alone_is_amber_not_success(self):
        text, color = self.label(True, 0)
        self.assertIn('LEG SHAPE ONLY: OK', text)
        self.assertIn('stand hold pending', text)
        self.assertEqual(color, (0, 190, 255))

    def test_incomplete_standing_hold_remains_amber(self):
        self.assertEqual(self.label(True, 2.98)[1], (0, 190, 255))

    def test_current_completed_valid_stand_is_green(self):
        text, color = self.label(True, 3)
        self.assertIn('valid stand held 3s', text)
        self.assertEqual(color, (80, 220, 80))

    def test_bad_geometry_is_red(self):
        text, color = self.label(False, 0)
        self.assertIn('INVALID LEG GEOMETRY', text)
        self.assertEqual(color, (60, 80, 255))


if __name__ == '__main__':
    unittest.main()
