"""CPU contract tests of the uninstalled adapter against a minimal Isaac stub.

This does not prove real articulation, contact, observation or rollout replay.
No simulator package is imported, registered, installed or started.
"""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/go2_recovery'
NAMES = tuple(f'{leg}_{joint}_joint' for joint in ('hip','thigh','calf') for leg in ('FL','FR','RL','RR'))


class ControlStepJointPositionAction:
    @property
    def raw_actions(self): return self._raw_actions
    @property
    def processed_actions(self): return self._processed_actions
    def process_actions(self, *args): raise AssertionError('Global processing forbidden')
    def apply_actions(self, *args): raise AssertionError('Global action application forbidden')


class Scene(dict):
    def update(self, *args): raise AssertionError('Scene update forbidden inside adapter')
    def write_data_to_sim(self): raise AssertionError('Global scene write forbidden inside adapter')


class StubManagerBasedRLEnv:
    constructions = 0
    def __init__(self, cfg, **kwargs):
        type(self).constructions += 1
        self.cfg, self.device, self.num_envs = cfg, torch.device('cpu'), cfg.scene.num_envs
        self.trace=[]; n=self.num_envs
        data=NS(default_joint_pos=torch.zeros(n,12),soft_joint_pos_limits=torch.tensor([-1.,1.]).repeat(n,12,1),
                joint_pos=torch.full((n,12),7.),joint_vel=torch.full((n,12),8.),
                joint_pos_target=torch.full((n,12),9.),root_link_pose_w=torch.full((n,7),10.),
                root_com_vel_w=torch.full((n,6),11.))
        def joints(q,qd,env_ids):
            self.trace.append('joints'); data.joint_pos[env_ids]=q; data.joint_vel[env_ids]=qd
        def pose(value,env_ids):
            self.trace.append('root_link_pose'); data.root_link_pose_w[env_ids]=value
        def velocity(value,env_ids):
            self.trace.append('root_com_velocity'); data.root_com_vel_w[env_ids]=value
        def target(value,joint_ids,env_ids):
            self.trace.append('target'); data.joint_pos_target[env_ids,joint_ids]=value
        asset=NS(data=data,joint_names=list(NAMES),num_joints=12,write_joint_state_to_sim=joints,
                 write_root_link_pose_to_sim=pose,write_root_com_velocity_to_sim=velocity,
                 set_joint_position_target=target)
        self.scene=Scene(robot=asset)
        self.scene.env_origins=torch.arange(n,dtype=torch.float32)[:,None].repeat(1,3)*10.
        term=ControlStepJointPositionAction()
        term.cfg=NS(class_type=type(term),reference='nominal',scale=.25,offset=0.,use_default_offset=False,clip=None)
        term._asset=asset; term._scale=.25; term._offset=0.; term._joint_ids=slice(None); term._joint_names=NAMES; term.action_dim=12
        term._raw_actions=torch.full((n,12),12.); term._processed_actions=torch.full((n,12),13.); term._target=torch.full((n,12),14.)
        self.action_manager=NS(active_terms=['joint_pos'],total_action_dim=12,get_term=lambda name:term,
                               action=torch.full((n,12),15.),prev_action=torch.full((n,12),16.))
        command=NS(vel_command_b=torch.full((n,3),17.),is_standing_env=torch.zeros(n,dtype=torch.bool),
                   is_heading_env=torch.ones(n,dtype=torch.bool))
        self.command_manager=NS(get_term=lambda name:command)
        self.episode_length_buf=torch.full((n,),18,dtype=torch.long)
        self._fake_observation=torch.full((n,48),19.)
        self.common_step_counter=1234; self._sim_step_counter=777
    def _reset_idx(self, env_ids):
        self.trace.append('super_reset')
        t=self.action_manager.get_term('joint_pos')
        for v in (self.action_manager.action,self.action_manager.prev_action,t._raw_actions,t._processed_actions): v[env_ids]=0.
        t._target[env_ids]=self.scene['robot'].data.joint_pos[env_ids]
        self.command_manager.get_term('base_velocity').vel_command_b[env_ids]=99.
        self.episode_length_buf[env_ids]=0
        self.trace.append('super_reset_complete')


# Isolate imports to this CPU test process and a new local package namespace.
isaac_stub=ModuleType('isaaclab'); env_stub=ModuleType('isaaclab.envs')
env_stub.ManagerBasedRLEnv=StubManagerBasedRLEnv
sys.modules['isaaclab']=isaac_stub; sys.modules['isaaclab.envs']=env_stub
package=ModuleType('handoff_reset_cpu_test'); package.__path__=[str(SRC)]
sys.modules[package.__name__]=package
spec=importlib.util.spec_from_file_location(package.__name__+'.handoff_reset_env',SRC/'handoff_reset_env.py')
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
Data=module.HandoffResetData
Env=module.HandoffResetEnv


def fixture_data(**overrides):
    q=torch.tensor([2.,3.,4.])[:,None].repeat(1,12)  # preserve measured out-of-soft-limit q
    raw=torch.arange(36,dtype=torch.float32).reshape(3,12)*.2-2.
    raw[0,0]=100.
    pose=torch.zeros(3,7); pose[:,2]=torch.tensor([.15,.18,.20]); pose[:,3]=1.
    pose[:,0]=torch.tensor([.1,-.2,.3])
    values=dict(archive_sha256='a'*64,receipt_sha256='b'*64,source_split='train',
        input_replay_verified=True,subset_reset_verified=True,contact_probe_finite=True,joint_names=NAMES,
        allowed_train_state_ids=torch.tensor([11,23,45,60]),source_train_state_id=torch.tensor([11,23,45]),
        root_link_pose_local=pose,root_com_velocity_w=torch.arange(18,dtype=torch.float32).reshape(3,6)*.01,
        joint_positions_rad=q,joint_velocities_rad_s=torch.arange(36,dtype=torch.float32).reshape(3,12)*.03,
        previous_raw_action=raw,previous_previous_raw_action=torch.full((3,12),-.35),
        previous_executed_joint_target_rad=(raw*.25).clamp(-1.,1.),velocity_command_b=torch.zeros(3,3),
        default_joint_positions_rad=torch.zeros(12),soft_joint_limits_rad=torch.tensor([-1.,1.]).repeat(12,1))
    values.update(overrides)
    return Data(**values)


def fixture_cfg():
    return NS(handoff_archive_sha256='a'*64,handoff_fraction=.2,scene=NS(num_envs=20),
              events=NS(reset_base=None,reset_robot_joints=None),commands=NS(base_velocity=NS(
                  heading_command=False,rel_standing_envs=1.,
                  ranges=NS(lin_vel_x=(0.,0.),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.)))))


def buffers(env):
    a=env.scene['robot'].data; t=env.action_manager.get_term('joint_pos'); c=env.command_manager.get_term('base_velocity')
    return [a.joint_pos,a.joint_vel,a.joint_pos_target,a.root_link_pose_w,a.root_com_vel_w,
            env.action_manager.action,env.action_manager.prev_action,t.raw_actions,t.processed_actions,t._target,
            c.vel_command_b,c.is_standing_env,c.is_heading_env,env.episode_length_buf,
            env.last_reset_was_handoff,env.last_handoff_sample_ids,env.last_handoff_source_ids,
            env._fake_observation]


class ResetContractTests(unittest.TestCase):
    def test_defaults_and_unverified_training_fail_before_base_construction(self):
        count=Env.constructions
        with self.assertRaises(ValueError): Env(fixture_cfg())
        for field in ('input_replay_verified','subset_reset_verified','contact_probe_finite'):
            with self.subTest(field=field):
                with self.assertRaises(ValueError): Env(fixture_cfg(),validated_handoff=fixture_data(**{field:False}))
        self.assertEqual(Env.constructions,count)

    def test_validation_only_breaks_receipt_bootstrap_cycle_without_training_permission(self):
        env=Env(fixture_cfg(),validated_handoff=fixture_data(input_replay_verified=False,
                subset_reset_verified=False,contact_probe_finite=False),validation_only=True)
        self.assertTrue(env.handoff_validation_only)
        self.assertFalse(env.handoff_training_permitted)
        self.assertEqual(env.trace,[])

    def test_cfg_old_reset_and_moving_commands_fail_closed(self):
        mutations=[lambda c:setattr(c.events,'reset_base',object()),
                   lambda c:setattr(c.events,'reset_robot_joints',object()),
                   lambda c:setattr(c,'handoff_fraction',.3),lambda c:setattr(c,'handoff_archive_sha256','c'*64),
                   lambda c:setattr(c.commands.base_velocity.ranges,'lin_vel_x',(0.,1.)),
                   lambda c:setattr(c.commands.base_velocity,'heading_command',True)]
        for mutate in mutations:
            cfg=fixture_cfg(); mutate(cfg)
            with self.assertRaises(ValueError): Env(cfg,validated_handoff=fixture_data())

    def test_training_ids_and_tensor_semantics_are_validated(self):
        invalid=[dict(source_split='heldout'),dict(receipt_sha256=''),
                 dict(source_train_state_id=torch.tensor([11,23,61])),
                 dict(source_train_state_id=torch.tensor([11,11,45])),
                 dict(previous_executed_joint_target_rad=torch.zeros(3,12)),
                 dict(previous_previous_raw_action=torch.full((3,12),float('nan'))),
                 dict(root_link_pose_local=torch.zeros(3,7)),dict(velocity_command_b=torch.ones(3,3)),
                 dict(joint_positions_rad=torch.zeros(3,11)),dict(joint_velocities_rad_s=torch.zeros(3,12,dtype=torch.float64))]
        for values in invalid:
            with self.subTest(fields=list(values)):
                with self.assertRaises(ValueError): Env(fixture_cfg(),validated_handoff=fixture_data(**values))

    def test_super_first_subset_only_restoration_and_recorded_history(self):
        data=fixture_data(); env=Env(fixture_cfg(),validated_handoff=data)
        old=[v.clone() for v in buffers(env)]
        ids=torch.tensor([17,3,9,1,8,5,12,6,15,10,2,14,4,13,7,11])
        untouched=torch.tensor([0,16,18,19])
        torch.manual_seed(42)
        env._reset_idx(ids)
        self.assertEqual(env.trace,['super_reset','super_reset_complete','joints','root_link_pose','root_com_velocity','target'])
        for v,prior in zip(buffers(env),old): self.assertTrue(torch.equal(v[untouched],prior[untouched]))
        self.assertTrue(torch.equal(env._fake_observation,old[-1]))
        self.assertEqual((env.common_step_counter,env._sim_step_counter),(1234,777))
        mask=env.last_reset_was_handoff[ids]
        self.assertGreater(int(mask.sum()),0); self.assertGreater(int((~mask).sum()),0)
        hand_ids=ids[mask]; ordinary_ids=ids[~mask]; picked=env.last_handoff_sample_ids[hand_ids]
        a=env.scene['robot'].data; t=env.action_manager.get_term('joint_pos'); c=env.command_manager.get_term('base_velocity')
        self.assertTrue(torch.equal(a.joint_pos[hand_ids],data.joint_positions_rad[picked]))
        self.assertTrue(torch.equal(a.joint_vel[hand_ids],data.joint_velocities_rad_s[picked]))
        expected_pose=data.root_link_pose_local[picked].clone(); expected_pose[:,:3]+=env.scene.env_origins[hand_ids]
        self.assertTrue(torch.equal(a.root_link_pose_w[hand_ids],expected_pose))
        self.assertTrue(torch.equal(a.root_com_vel_w[hand_ids],data.root_com_velocity_w[picked]))
        self.assertTrue(torch.equal(env.action_manager.action[hand_ids],data.previous_raw_action[picked]))
        self.assertTrue(torch.equal(env.action_manager.prev_action[hand_ids],data.previous_previous_raw_action[picked]))
        self.assertTrue(torch.equal(t.processed_actions[hand_ids],.25*data.previous_raw_action[picked]))
        self.assertTrue(torch.equal(a.joint_pos_target[hand_ids],data.previous_executed_joint_target_rad[picked]))
        self.assertTrue(torch.equal(t._target[hand_ids],a.joint_pos_target[hand_ids]))
        for v in (a.joint_vel,a.root_com_vel_w,env.action_manager.action,env.action_manager.prev_action,t.raw_actions,t.processed_actions):
            self.assertEqual(float(v[ordinary_ids].abs().sum()),0.)
        self.assertTrue(torch.equal(a.joint_pos_target[ordinary_ids],torch.zeros_like(a.joint_pos_target[ordinary_ids])))
        self.assertTrue(bool((a.joint_pos[ordinary_ids].abs()<=.1).all()))
        local_height=a.root_link_pose_w[ordinary_ids,2]-env.scene.env_origins[ordinary_ids,2]
        self.assertTrue(bool(((local_height>=.34-1e-5)&(local_height<=.38+1e-5)).all()))
        self.assertTrue(bool((c.vel_command_b[ids]==0).all()))
        self.assertTrue(bool(c.is_standing_env[ids].all())); self.assertFalse(bool(c.is_heading_env[ids].any()))
        self.assertTrue(bool((env.episode_length_buf[ids]==0).all()))
        self.assertTrue(bool((env.last_handoff_source_ids[ordinary_ids]==-1).all()))
        self.assertTrue(torch.equal(env.last_handoff_source_ids[hand_ids],data.source_train_state_id[picked]))

    def test_sampler_fraction_and_ordinary_roll_pitch_bounds(self):
        data=fixture_data(); generator=torch.Generator().manual_seed(55)
        batch=module.sample_handoff_reset_batch(data,torch.zeros(2000,3),generator=generator)
        fraction=float(batch['handoff'].float().mean())
        self.assertLess(abs(fraction-.2),.04)  # Statistical contract, not per-subset integer quotas.
        q=batch['root_link_pose_w'][~batch['handoff'],3:]
        w,x,y,z=q.unbind(-1)
        roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=torch.asin(torch.clamp(2*(w*y-z*x),-1,1))
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        self.assertLessEqual(float(roll.abs().max()),.15001)
        self.assertLessEqual(float(pitch.abs().max()),.15001)
        self.assertGreater(float(yaw.max()-yaw.min()),5.5)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(q,dim=-1),torch.ones(q.shape[0]),atol=1e-6,rtol=0))

    def test_invalid_reset_ids_and_changed_runtime_fail_before_super(self):
        for ids in (torch.tensor([1,1]),torch.tensor([-1]),torch.tensor([20]),torch.tensor([1.])):
            env=Env(fixture_cfg(),validated_handoff=fixture_data()); before=[v.clone() for v in buffers(env)]
            with self.assertRaises(ValueError): env._reset_idx(ids)
            self.assertEqual(env.trace,[])
            for v,old in zip(buffers(env),before): self.assertTrue(torch.equal(v,old))
        env=Env(fixture_cfg(),validated_handoff=fixture_data())
        env.scene['robot'].joint_names=list(reversed(NAMES))
        with self.assertRaises(ValueError): env._reset_idx(torch.tensor([1]))
        self.assertEqual(env.trace,[])

    def test_empty_subset_and_caller_mutation(self):
        data=fixture_data(); env=Env(fixture_cfg(),validated_handoff=data)
        old=[v.clone() for v in buffers(env)]
        env._reset_idx(torch.empty(0,dtype=torch.long))
        self.assertEqual(env.trace,[])
        for v,prior in zip(buffers(env),old): self.assertTrue(torch.equal(v,prior))
        expected=env._handoff_data.previous_raw_action.clone()
        data.previous_raw_action.fill_(999.)
        self.assertTrue(torch.equal(env._handoff_data.previous_raw_action,expected))

    def test_source_has_no_files_sim_steps_or_observation_injection(self):
        tree=ast.parse((SRC/'handoff_reset_env.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                self.assertNotIn(name,{'open','read_text','load','load_recovery_state_bank','step','forward',
                    'process_action','process_actions','apply_action','apply_actions','write_data_to_sim','get_observations'})
        self.assertNotIn('policy_observation',(SRC/'handoff_reset_env.py').read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
