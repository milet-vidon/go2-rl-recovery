"""CPU checks of the actual adapter class; runtime smoke remains mandatory."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('supported_math_test', ROOT / 'src/go2_recovery/recovery_supported_math.py')
math_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_module)


class MockBase:
    def __init__(self, cfg, env):
        pass


tree = ast.parse((ROOT / 'src/go2_recovery/recovery_supported_mdp.py').read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
scope = {'ManagerTermBase': MockBase, 'supported_pose_score': math_module.supported_pose_score,
         'validate_supported_joint_soft_range': math_module.validate_supported_joint_soft_range}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-adapter>', 'exec'), scope)
Adapter = scope['SupportedPostureReward']


def fixture():
    names = [f'{leg}_{kind}_joint' for kind in ('hip','thigh','calf') for leg in ('FL','FR','RL','RR')]
    default = torch.linspace(-1, 1, 12).repeat(3, 1)
    widths = torch.linspace(1, 4, 12).repeat(3, 1)
    data = NS(default_joint_pos=default.clone(), joint_pos=default.clone(),
              projected_gravity_b=torch.tensor([[0.,0.,-1.],[0.,0.,0.],[0.,0.,1.]]),
              soft_joint_pos_limits=torch.stack([default-widths/2, default+widths/2], -1))
    env = NS(scene={'robot': NS(joint_names=names, data=data)})
    return env, data


class AdapterTests(unittest.TestCase):
    def test_actual_default_and_gravity_sign(self):
        env, data = fixture()
        torch.testing.assert_close(Adapter(NS(), env)(env), torch.tensor([1.,0.,0.]))

    def test_actual_per_joint_ranges_not_assumed_order(self):
        env, data = fixture()
        data.projected_gravity_b[:,2] = -1
        data.joint_pos += (data.soft_joint_pos_limits[...,1]-data.soft_joint_pos_limits[...,0])/4
        term = Adapter(NS(),env)
        torch.testing.assert_close(term(env), torch.full((3,),.75))
        permutation = torch.randperm(12)
        robot = env.scene['robot']
        robot.joint_names = [robot.joint_names[i] for i in permutation]
        for name in ('joint_pos','default_joint_pos','soft_joint_pos_limits'):
            setattr(data, name, getattr(data,name)[:,permutation])
        torch.testing.assert_close(Adapter(NS(),env)(env), torch.full((3,),.75))

    def test_invalid_names_limits_and_default_fail(self):
        env, data = fixture()
        env.scene['robot'].joint_names[0] = 'unknown'
        with self.assertRaises(ValueError): Adapter(NS(),env)
        for case in ('zero','nan','outside'):
            env, data = fixture()
            if case=='zero': data.soft_joint_pos_limits[0,0,1] = data.soft_joint_pos_limits[0,0,0]
            elif case=='nan': data.soft_joint_pos_limits[0,0,1] = float('nan')
            else: data.default_joint_pos[0,0] = 100
            with self.subTest(case=case), self.assertRaises(ValueError): Adapter(NS(),env)

    def test_no_input_mutation(self):
        env, data = fixture()
        before = {k:v.clone() for k,v in vars(data).items()}
        Adapter(NS(),env)(env)
        for name, value in before.items(): torch.testing.assert_close(getattr(data,name),value)


if __name__ == '__main__': unittest.main(verbosity=2)
