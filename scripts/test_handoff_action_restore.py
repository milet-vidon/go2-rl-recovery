"""Small CPU-only mocks; no simulator imports, startup or GPU calls."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'src/go2_recovery/handoff_action_restore.py'
spec = importlib.util.spec_from_file_location('handoff_restore', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
restore = module.restore_handoff_action_history
NAMES = tuple(f'{leg}_{joint}_joint' for joint in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR'))


class ControlStepJointPositionAction:
    def __init__(self, asset):
        self.cfg = NS(class_type=type(self), reference='nominal', scale=.25,
                      offset=0., use_default_offset=False, clip=None)
        self._asset, self._scale, self._offset = asset, .25, 0.
        self._joint_ids, self._joint_names = slice(None), list(NAMES)
        self.action_dim = 12
        self._raw_actions = torch.full((5,12), 11.)
        self._processed_actions = torch.full((5,12), 22.)
        self._target = torch.full((5,12), 33.)

    @property
    def raw_actions(self): return self._raw_actions

    @property
    def processed_actions(self): return self._processed_actions

    def process_actions(self, *args): raise AssertionError('Must not process all environments')
    def apply_actions(self): raise AssertionError('Must not apply all environments')


def fixture():
    # Deliberately outside soft limits: measured state must NOT be clamped.
    q = torch.arange(60,dtype=torch.float32).reshape(5,12) + 10.
    nominal = torch.arange(60,dtype=torch.float32).reshape(5,12) * .002
    data = NS(joint_pos=q, default_joint_pos=nominal,
              soft_joint_pos_limits=torch.tensor([-1.,1.]).repeat(5,12,1),
              joint_pos_target=torch.full((5,12),44.))
    calls = []
    def set_target(target, joint_ids=None, env_ids=None):
        assert isinstance(joint_ids,slice) and joint_ids == slice(None)
        assert env_ids is not None
        calls.append((target.clone(),env_ids.clone()))
        data.joint_pos_target[env_ids,joint_ids] = target
    asset = NS(data=data,joint_names=list(NAMES),num_joints=12,set_joint_position_target=set_target)
    term = ControlStepJointPositionAction(asset)
    manager = NS(active_terms=['joint_pos'],total_action_dim=12,
                 action=torch.full((5,12),55.),prev_action=torch.full((5,12),66.),
                 get_term=lambda name: term,
                 process_action=lambda *args: (_ for _ in ()).throw(AssertionError('Must not process manager')))
    ids = torch.tensor([3,1],dtype=torch.long)
    raw = torch.arange(24,dtype=torch.float32).reshape(2,12) * .07 - .6
    raw[0,0], raw[1,11] = 100., -100.  # Raw actions are not action-clipped.
    previous = torch.full((2,12),-.37)
    targets = (nominal[ids] + .25 * raw).clamp(-1.,1.)
    args = dict(previous_raw_action=raw, previous_previous_raw_action=previous,
                previous_executed_joint_target_rad=targets, joint_names=NAMES)
    return manager,asset,term,ids,args,calls


def all_buffers(manager,asset,term):
    return [manager.action,manager.prev_action,term.raw_actions,term.processed_actions,
            term._target,asset.data.joint_pos_target,asset.data.joint_pos,asset.data.default_joint_pos,
            asset.data.soft_joint_pos_limits]


class RestoreTests(unittest.TestCase):
    def assert_rejected_without_writes(self, mutate):
        manager,asset,term,ids,args,calls = fixture()
        replacement = mutate(manager,asset,term,ids,args)
        if replacement is not None: ids = replacement
        before = [value.clone() for value in all_buffers(manager,asset,term)]
        with self.assertRaises((ValueError,AttributeError)):
            restore(manager,asset,ids,**args)
        self.assertEqual(calls,[])
        for value,old in zip(all_buffers(manager,asset,term),before):
            torch.testing.assert_close(value,old,rtol=0,atol=0,equal_nan=True)

    def test_subset_restores_exact_history_and_targets_not_measured_q(self):
        m,a,t,ids,args,calls = fixture()
        before = [value.clone() for value in all_buffers(m,a,t)]
        inputs = {key:value.clone() for key,value in args.items() if isinstance(value,torch.Tensor)}
        restore(m,a,ids,**args)
        expected = [args['previous_raw_action'],args['previous_previous_raw_action'],
                    args['previous_raw_action'],.25*args['previous_raw_action'],
                    args['previous_executed_joint_target_rad'],args['previous_executed_joint_target_rad']]
        for value,want in zip(all_buffers(m,a,t),expected): self.assertTrue(torch.equal(value[ids],want))
        untouched = torch.tensor([0,2,4])
        for value,old in zip(all_buffers(m,a,t),before): self.assertTrue(torch.equal(value[untouched],old[untouched]))
        for value,old in zip(all_buffers(m,a,t)[6:],before[6:]): self.assertTrue(torch.equal(value,old))
        self.assertEqual(len(calls),1)
        self.assertTrue(torch.equal(calls[0][1],ids))
        for key,value in inputs.items(): self.assertTrue(torch.equal(value,args[key]))
        self.assertEqual(float(t.raw_actions[3,0]),100.)
        self.assertEqual(float(t._target[3,0]),1.)

    def test_native_joint_index_representations(self):
        for indices in (list(range(12)),tuple(range(12)),torch.arange(12),slice(0,12,1)):
            with self.subTest(indices=indices):
                m,a,t,ids,args,_ = fixture(); t._joint_ids=indices
                restore(m,a,ids,**args)

    def test_empty_selection_is_noop(self):
        m,a,t,_,args,calls=fixture()
        for key in list(args):
            if isinstance(args[key],torch.Tensor): args[key]=args[key][:0]
        old=[value.clone() for value in all_buffers(m,a,t)]
        restore(m,a,torch.empty(0,dtype=torch.long),**args)
        self.assertEqual(calls,[])
        for value,before in zip(all_buffers(m,a,t),old): self.assertTrue(torch.equal(value,before))

    def test_wrong_ids_rejected(self):
        for ids in (torch.tensor([1,1]),torch.tensor([-1,1]),torch.tensor([5,1]),
                    torch.tensor([1.,3.]),torch.tensor([[1,3]]),[1,3],torch.tensor([1,3],dtype=torch.int32)):
            with self.subTest(ids=ids): self.assert_rejected_without_writes(lambda *unused: ids)

    def test_wrong_shapes_dtypes_grad_and_nonfinite_rejected(self):
        for key in ('previous_raw_action','previous_previous_raw_action','previous_executed_joint_target_rad'):
            for transform in (lambda x:x[:1],lambda x:x.double(),lambda x:x.requires_grad_(),
                              lambda x:torch.full_like(x,float('nan')),lambda x:torch.full_like(x,float('inf'))):
                with self.subTest(key=key,transform=transform):
                    def mutate(m,a,t,ids,args): args[key]=transform(args[key])
                    self.assert_rejected_without_writes(mutate)

    def test_inconsistent_executed_target_rejected(self):
        for delta in (.01,100.):
            def mutate(m,a,t,ids,args): args['previous_executed_joint_target_rad'][0,1]+=delta
            self.assert_rejected_without_writes(mutate)

    def test_wrong_config_semantics_rejected(self):
        for name,value in (('reference','current'),('scale',.5),('offset',.1),
                           ('use_default_offset',True),('clip',{}),('class_type',object)):
            with self.subTest(name=name):
                self.assert_rejected_without_writes(lambda m,a,t,ids,args:setattr(t.cfg,name,value))
        for name,value in (('_scale',.5),('_offset',.1),('_scale',torch.tensor(.25))):
            with self.subTest(name=name,value=value):
                self.assert_rejected_without_writes(lambda m,a,t,ids,args:setattr(t,name,value))

    def test_wrong_terms_assets_joint_orders_and_counts_rejected(self):
        mutations = [
            lambda m,a,t,ids,args:setattr(m,'active_terms',['joint_pos','extra']),
            lambda m,a,t,ids,args:setattr(t,'_asset',object()),
            lambda m,a,t,ids,args:args.update(joint_names=NAMES[::-1]),
            lambda m,a,t,ids,args:setattr(t,'_joint_names',list(reversed(NAMES))),
            lambda m,a,t,ids,args:setattr(t,'_joint_ids',list(reversed(range(12)))),
            lambda m,a,t,ids,args:setattr(t,'_joint_ids',list(range(11))),
            lambda m,a,t,ids,args:setattr(m,'total_action_dim',11),
            lambda m,a,t,ids,args:setattr(a,'num_joints',13),
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index): self.assert_rejected_without_writes(mutate)

    def test_bad_destination_and_selected_limits_rejected(self):
        mutations = [
            lambda m,a,t,ids,args:setattr(m,'prev_action',torch.zeros(4,12)),
            lambda m,a,t,ids,args:setattr(t,'_processed_actions',torch.zeros(5,12,dtype=torch.float64)),
            lambda m,a,t,ids,args:a.data.default_joint_pos.__setitem__((3,0),float('nan')),
            lambda m,a,t,ids,args:a.data.soft_joint_pos_limits.__setitem__((3,0,0),2.),
            lambda m,a,t,ids,args:a.data.soft_joint_pos_limits.__setitem__((3,0,1),float('inf')),
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index): self.assert_rejected_without_writes(mutate)

    def test_recorded_target_values_are_not_recomputed(self):
        m,a,t,ids,args,_=fixture()
        args['previous_executed_joint_target_rad'][0,1]+=2e-7
        restore(m,a,ids,**args)
        self.assertTrue(torch.equal(t._target[ids],args['previous_executed_joint_target_rad']))

    def test_snapshot_alias_is_copied_before_destination_write(self):
        m,a,t,_,args,_=fixture()
        ids=torch.tensor([1,0])
        m.action[:2]=args['previous_raw_action']
        args['previous_raw_action']=m.action[:2]  # Source rows overlap destination rows.
        args['previous_executed_joint_target_rad']=(a.data.default_joint_pos[ids]+.25*args['previous_raw_action']).clamp(-1.,1.)
        want=args['previous_raw_action'].clone()
        restore(m,a,ids,**args)
        self.assertTrue(torch.equal(m.action[ids],want))
        self.assertTrue(torch.equal(t.raw_actions[ids],want))

    def test_module_has_no_simulator_imports_or_forbidden_calls(self):
        tree=ast.parse(MODULE.read_text(encoding='utf-8'))
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
            if isinstance(node,ast.ImportFrom): imports.append(node.module)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,{'step','process_action','process_actions','apply_action','apply_actions','write_data_to_sim','get_observations'})
        self.assertEqual(set(imports),{'__future__','collections.abc','numbers','torch'})


if __name__ == '__main__': unittest.main()
