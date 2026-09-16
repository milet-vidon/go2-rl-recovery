"""CPU action tests with a minimal affine parent; real integration needs simulator smoke."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).parents[1]


class AffineParent:
    def __init__(self, cfg, env):
        self.cfg, self._asset, self._joint_ids = cfg, env.asset, env.joint_ids
        shape = self._asset.data.joint_pos[:, self._joint_ids].shape
        self._raw_actions = torch.zeros(shape)
        self._processed_actions = torch.zeros(shape)

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self.cfg.scale + self.cfg.offset


def action_class():
    tree = ast.parse((ROOT / 'src/go2_recovery/recovery_control_targets.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = {'torch': torch, 'JointPositionAction': AffineParent}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<action>', 'exec'), scope)
    return scope[node.name]


def make(reference='current', joint_ids=slice(None), **overrides):
    q = torch.tensor([[.1,.2,.3,.4], [.5,.6,.7,.8]])
    data = NS(joint_pos=q, default_joint_pos=torch.zeros_like(q),
              soft_joint_pos_limits=torch.tensor([-1.,1.]).repeat(2,4,1))
    writes = []
    asset = NS(data=data, set_joint_position_target=lambda target, joint_ids: writes.append((target.clone(), joint_ids)))
    cfg = NS(reference=reference, scale=.25, offset=0., use_default_offset=False, clip=None)
    for key, value in overrides.items(): setattr(cfg, key, value)
    return action_class()(cfg, NS(asset=asset, joint_ids=joint_ids)), data, writes


class ActionTests(unittest.TestCase):
    def test_nominal_handover_uses_same_targets_not_relative_zero_actions(self):
        tree=ast.parse((ROOT/'scripts/evaluate_go2_recovery.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_settle_nominal_pose')
        class Legacy: pass
        class Sampled: pass
        scope={'torch':torch,'JointPositionAction':Legacy,'ControlStepJointPositionAction':Sampled}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<handover>','exec'),scope)
        results=[]
        for cls in (Legacy,Sampled):
            writes=[]; process_calls=[]; apply_calls=[]
            data=NS(default_joint_pos=torch.full((2,4),.25),joint_vel=torch.zeros(2,4),
                    root_lin_vel_w=torch.zeros(2,3),root_ang_vel_w=torch.zeros(2,3))
            asset=NS(data=data,num_joints=4,set_joint_position_target=lambda q:writes.append(q.clone()))
            class Scene(dict):
                def write_data_to_sim(self): pass
                def update(self,dt): pass
            scene=Scene(robot=asset)
            scene.sensors={'contact_forces':NS(data=NS(net_forces_w=torch.ones(2,4,3)))}
            term=cls(); term.cfg=NS(asset_name='robot',clip=None,use_default_offset=True); term.action_dim=4
            def apply():
                apply_calls.append(1); asset.set_joint_position_target(data.default_joint_pos)
            manager=NS(active_terms=['joint_pos'],get_term=lambda _:term,total_action_dim=4,
                       process_action=lambda q:process_calls.append(q.clone()),apply_action=apply)
            env=NS(action_manager=manager,scene=scene,num_envs=2,device='cpu',cfg=NS(decimation=4),
                   sim=NS(has_gui=lambda:False,has_rtx_sensors=lambda:False,step=lambda render:None),
                   _sim_step_counter=0,physics_dt=.005,episode_length_buf=torch.zeros(2),
                   termination_manager=NS(compute=lambda:torch.zeros(2,dtype=torch.bool)))
            quiet=scope['_settle_nominal_pose'](env,3)
            self.assertTrue(torch.equal(quiet,torch.full((2,),3,dtype=torch.int32)))
            self.assertEqual(env._sim_step_counter,12)
            self.assertEqual(len(process_calls),3 if cls is Legacy else 0)
            self.assertEqual(len(apply_calls),12 if cls is Legacy else 0)
            self.assertEqual(len(writes),12)
            results.append(torch.stack(writes))
        self.assertTrue(torch.equal(results[0],results[1]))

    def test_reference_is_only_target_difference(self):
        current, data, _ = make(); nominal, _, _ = make('nominal')
        actions = torch.ones(2,4)*.1  # Stay inside limits; saturation is tested separately.
        current.process_actions(actions); nominal.process_actions(actions)
        torch.testing.assert_close(current._target - nominal._target, data.joint_pos, atol=1e-7, rtol=0)

    def test_four_substeps_do_not_integrate_target(self):
        action, data, writes = make()
        action.process_actions(torch.ones(2,4)*.1)
        expected = action._target.clone()
        for _ in range(4):
            data.joint_pos += .01
            action.apply_actions()
        self.assertEqual(len(writes), 4)
        self.assertTrue(all(torch.equal(v, expected) for v, _ in writes))
        action.process_actions(torch.ones(2,4)*.1)
        torch.testing.assert_close(action._target, expected + .04)

    def test_subset_joint_order_and_identical_clamp(self):
        for ref in ('current','nominal'):
            action, data, writes = make(ref, [3,1])
            action.process_actions(torch.tensor([[100.,-100.],[-100.,100.]]))
            self.assertTrue(torch.equal(action._target, torch.tensor([[1.,-1.],[-1.,1.]])))
            action.apply_actions()
            self.assertEqual(writes[0][1], [3,1])
            action.process_actions(torch.zeros(2,2))
            expected = data.joint_pos[:,[3,1]] if ref == 'current' else torch.zeros(2,2)
            self.assertTrue(torch.equal(action._target, expected))

    def test_subset_reset_does_not_write_or_touch_other_environment(self):
        action, data, writes = make(joint_ids=[3,1])
        action.process_actions(torch.ones(2,2)*.2)
        saved = [v.clone() for v in (action._raw_actions, action._processed_actions, action._target)]
        data.joint_pos[1] = -.2
        action.reset(torch.tensor([1]))
        self.assertEqual(writes, [])
        for v, old in zip((action._raw_actions, action._processed_actions, action._target), saved):
            self.assertTrue(torch.equal(v[0], old[0]))
        self.assertEqual(action._raw_actions[1].abs().sum(),0)
        self.assertEqual(action._processed_actions[1].abs().sum(),0)
        self.assertTrue(torch.equal(action._target[1],data.joint_pos[1,[3,1]]))
        saved_target = action._target.clone(); action.reset([])
        self.assertTrue(torch.equal(saved_target,action._target))
        action.reset()
        self.assertTrue(torch.equal(action._target,data.joint_pos[:,[3,1]]))

    def test_invalid_modes_offsets_and_extra_clips_fail(self):
        for kw in ({'reference':'typo'}, {'offset':.1}, {'use_default_offset':True}, {'clip':{'.*':(-1,1)}}):
            with self.assertRaises(ValueError): make(**kw)

    def test_configuration_pair_only_differs_by_reference(self):
        class Base:
            def __init__(self): self.__post_init__()
            def __post_init__(self):
                self.actions=NS(joint_pos=None); self.events=NS(bank_fraction=.6)
                self.scene=NS(num_envs=512); self.observations=NS(policy=NS(enable_corruption=True))
        tree=ast.parse((ROOT/'src/go2_recovery/recovery_env_cfg.py').read_text())
        nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.ClassDef) and
               n.name.startswith(('UnitreeGo2RecoveryBankNominalTarget','UnitreeGo2RecoveryBankCurrentTarget'))]
        for node in nodes: node.decorator_list=[]
        scope={'UnitreeGo2RecoveryBankControlEnvCfg':Base,'ControlStepJointPositionActionCfg':NS}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<cfg>','exec'),scope)
        nominal=scope['UnitreeGo2RecoveryBankNominalTargetEnvCfg']()
        current=scope['UnitreeGo2RecoveryBankCurrentTargetEnvCfg']()
        self.assertEqual(current.actions.joint_pos.reference,'current')
        current.actions.joint_pos.reference='nominal'
        self.assertEqual(vars(nominal),vars(current))

    def test_fresh_bootstrap_is_reproducible_and_untrained(self):
        spec=importlib.util.spec_from_file_location('bootstrap',ROOT/'scripts/create_recovery_fresh_bootstrap.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        first=module.fresh_checkpoint(42); second=module.fresh_checkpoint(42)
        self.assertEqual(first['optimizer_state_dict']['state'],{})
        self.assertEqual(first['iter'],0)
        self.assertTrue(first['infos']['untrained'])
        for k,v in first['model_state_dict'].items(): self.assertTrue(torch.equal(v,second['model_state_dict'][k]))
        self.assertTrue(torch.equal(first['model_state_dict']['std'],torch.ones(12)))


if __name__=='__main__': unittest.main()
