"""Simulator-free guards separating sampled diagnostics from model acceptance."""
import argparse
import ast
import contextlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).with_name('evaluate_go2_recovery.py')
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
BACK = 'Isaac-Recovery-Bank-BackExplore-Flat-Unitree-Go2-Play-v0'
BANK = 'Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0'


def functions():
    names = {'_select_policy_action', '_mark_action_mode', '_new_motion_diagnostic',
             '_update_motion_diagnostic', '_motion_diagnostic_report', '_annotate'}
    scope = {'torch': torch, 'np': np}
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


def parse_flags(task, bank=True, stochastic=False):
    first = next(i for i,n in enumerate(TREE.body) if isinstance(n,ast.Assign) and
                 isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'POSE_CLASSES')
    last = next(i for i,n in enumerate(TREE.body) if isinstance(n,ast.Assign) and
                isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'app_launcher')
    argv = ['--task',task,'--checkpoint','model.pt','--output_dir','unused','--settle_s','1']
    if bank: argv += ['--state_bank_path','unused-bank']
    if stochastic: argv += ['--stochastic_diagnostic']
    class TestParser(argparse.ArgumentParser):
        def parse_args(self): return super().parse_args(argv)
    scope = {'argparse':SimpleNamespace(ArgumentParser=TestParser), 'Path':Path, 'math':math,
             'os':SimpleNamespace(environ={}), '__doc__':'test',
             'AppLauncher':SimpleNamespace(add_app_launcher_args=lambda parser:None)}
    with contextlib.redirect_stderr(io.StringIO()):
        exec(compile(ast.Module(body=TREE.body[first:last],type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope['args_cli']


class DiagnosticTests(unittest.TestCase):
    def test_default_and_sampled_policy_routes_are_distinct(self):
        scope = functions()
        mean = lambda obs: 'mean'
        sampled = lambda obs: 'sampled'
        runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(act=sampled)),
                                 get_inference_policy=lambda device:mean)
        self.assertIs(scope['_select_policy_action'](runner,'cpu'),mean)
        self.assertIs(scope['_select_policy_action'](runner,'cpu',True),sampled)

    def test_diagnostic_cannot_be_mistaken_for_acceptance(self):
        mark = functions()['_mark_action_mode']
        report = {'protocol_version':'fixed-v1','success_count_definition':'original', 'results':{'successes':1}}
        mark(report)
        self.assertTrue(report['acceptance_eligible'])
        self.assertEqual(report['protocol_version'],'fixed-v1')
        mark(report,True)
        self.assertFalse(report['acceptance_eligible'])
        self.assertEqual(report['policy_action_mode'],'stochastic_diagnostic')
        self.assertIn('NOT_ACCEPTANCE',report['protocol_version'])
        self.assertIn('cannot accept',report['success_count_definition'])
        self.assertEqual(report['results'],{'successes':1})

    def test_cli_requires_explicit_matching_task_and_bank(self):
        self.assertFalse(parse_flags(BANK).stochastic_diagnostic)
        self.assertTrue(parse_flags(BACK,stochastic=True).stochastic_diagnostic)
        for task,bank,stoch in ((BANK,True,True),(BACK,False,True),(BACK,True,False)):
            with self.assertRaises(SystemExit): parse_flags(task,bank,stoch)

    def test_motion_samples_measure_real_joint_span_speed_and_torque(self):
        scope = functions()
        data = SimpleNamespace(joint_pos=torch.zeros(2,12),joint_vel=torch.ones(2,12)*2,
             applied_torque=torch.ones(2,12)*3,root_pos_w=torch.tensor([[0.,0.,.057],[0.,0.,.15]]),
             projected_gravity_b=torch.tensor([[0.,0.,1.],[0.,1.,0.]]))
        class Scene(dict): pass
        scene = Scene(robot=SimpleNamespace(data=data)); scene.env_origins=torch.zeros(2,3)
        env=SimpleNamespace(scene=scene)
        state=scope['_new_motion_diagnostic'](env)
        data.joint_pos += .5
        before=data.joint_pos.clone()
        scope['_update_motion_diagnostic'](env,state,torch.ones(2,12)*.6)
        result=scope['_motion_diagnostic_report'](state,.01)
        self.assertTrue(torch.equal(data.joint_pos,before))
        self.assertEqual(result['joint_span_rad'],[[.5]*12]*2)
        self.assertEqual(result['joint_speed_peak'],[[2.]*12]*2)
        self.assertEqual(result['applied_torque_peak'],[[3.]*12]*2)
        self.assertEqual(result['min_tilt_deg'],[180.,90.])
        self.assertEqual(result['sample_rate_hz'],100.)
        self.assertFalse(result['acceptance_eligible'])

    def test_sampled_video_is_labeled_and_never_green(self):
        scope=functions(); calls=[]
        scope['cv2']=SimpleNamespace(COLOR_RGB2BGR=1,FONT_HERSHEY_SIMPLEX=0,LINE_AA=0,
            cvtColor=lambda frame,code:frame.copy(),putText=lambda *args:calls.append(args))
        scope['args_cli']=SimpleNamespace(hold_s=3,checkpoint=Path('model.pt'),settle_s=1,
            state_bank_path=None,stochastic_diagnostic=True)
        scope['_annotate'](np.zeros((540,960,3),np.uint8),'upside_down',1,.02,4,
            {'geometry_ok':True,'FL_foot_y_b':.16,'FR_foot_y_b':-.16})
        self.assertIn('NOT ACCEPTANCE',calls[0][1])
        self.assertNotEqual(calls[2][5],(80,220,80))


if __name__ == '__main__': unittest.main()
