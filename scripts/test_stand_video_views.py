"""Camera-only geometry tests; no simulator, rendering, or CUDA initialization."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np


class CameraTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path(__file__).with_name('evaluate_go2_stand_walk_stop.py').read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_camera_geometry', '_set_video_view')]
        self.args = NS(view='legacy')
        self.scope = {'np': np, 'args_cli': self.args}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<camera>', 'exec'), self.scope)
        self.geometry = self.scope['_camera_geometry']
        self.bodies = np.array([[-.4, -.2, .02], [.4, .2, .5]])

    def test_front_tracks_yaw_and_bounds(self):
        eye, target = self.geometry((1., 0., 0., 0.), self.bodies, 'front')
        np.testing.assert_allclose(target, [0., 0., .34])
        np.testing.assert_allclose(eye - target, [1.65, 0., .65])
        a = np.sqrt(.5)
        eye, target = self.geometry((a, 0., 0., a), self.bodies, 'front')
        np.testing.assert_allclose(eye - target, [0., 1.65, .65], atol=1e-14)

    def test_oblique_translation_and_inputs_unchanged(self):
        original = self.bodies.copy()
        eye, target = self.geometry((1., 0., 0., 0.), self.bodies, 'oblique')
        np.testing.assert_allclose(eye - target, [1.3, 1.3, .75])
        shift = np.array([8., -3., 0.])
        shifted_eye, shifted_target = self.geometry((1., 0., 0., 0.), self.bodies + shift, 'oblique')
        np.testing.assert_allclose(shifted_eye, eye + shift)
        np.testing.assert_allclose(shifted_target, target + shift)
        np.testing.assert_array_equal(self.bodies, original)

    def test_legacy_does_not_access_or_mutate_environment(self):
        self.scope['_set_video_view'](None)

    def test_unknown_view_rejected(self):
        with self.assertRaises(KeyError):
            self.geometry((1., 0., 0., 0.), self.bodies, 'sideways')


if __name__ == '__main__':
    unittest.main()
