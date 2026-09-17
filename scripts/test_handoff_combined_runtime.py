"""CPU tensor execution of generated gate/actor snippets; no Isaac Sim."""
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
import evaluate_handoff_combined as entry

sys.path.insert(0, str(entry.ROOT / "src/go2_recovery"))
from recovery_handoff_math import update_handoff_gate
from handoff_transition_math import transition_action
from roll_mirror_math import roll_input_copy, physical_roll_action, reflect_policy_observation, reflect_joint_vector


class CombinedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = entry.build_source()
        start = source.index('                        just_switched = torch.zeros_like(switched)', source.index('for step in range(steps):'))
        split = source.index('                        _assert_real_observation_contract(env.unwrapped, obs)', start)
        end = source.index('                        if action.shape != a.joint_pos.shape:', split)
        cls.gate_code = compile(textwrap.dedent(source[start:split]), '<actual-combined-gate>', 'exec')
        cls.actor_code = compile(textwrap.dedent(source[split:end]), '<actual-combined-actor>', 'exec')

    def namespace(self, startup, mirrored):
        n = 4
        gravity = torch.tensor([[0., 0., -1.], [0., -1., 0.], [0., 1., 0.], [0., 0., -1.]])
        obs = TensorDict({'policy': torch.arange(n * 48, dtype=torch.float32).reshape(n, 48) / 100.}, batch_size=[n])
        return dict(torch=torch, startup_selected=torch.tensor(startup), mirror_selected=torch.tensor(mirrored),
            step=0, a=SimpleNamespace(projected_gravity_b=gravity, root_ang_vel_b=torch.zeros(n, 3)),
            env=SimpleNamespace(unwrapped=SimpleNamespace(step_dt=.02, action_manager=SimpleNamespace(action=torch.zeros(n, 12)))),
            gate_steps=torch.zeros(n, dtype=torch.int32), switched=torch.zeros(n, dtype=torch.bool),
            ramp_anchor=torch.zeros(n, 12), ramp_progress=torch.zeros(n, dtype=torch.long), ramp_steps=0,
            obs=obs, update_handoff_gate=update_handoff_gate, transition_action=transition_action,
            roll_input_copy=roll_input_copy, physical_roll_action=physical_roll_action,
            _assert_real_observation_contract=lambda env, observed: None,
            policy=lambda observed: observed['policy'][:, 12:24] * .1,
            stand_policy=lambda observed: observed['policy'][:, 12:24] * .2)

    def run_rows(self, namespace):
        before = namespace['obs']['policy'].clone()
        seen = []
        def standing(observed):
            self.assertIs(observed, namespace['obs'])
            seen.append(observed['policy'].clone())
            return observed['policy'][:, 12:24] * .2
        namespace['stand_policy'] = standing
        for step in range(12):
            namespace['step'] = step
            exec(self.gate_code, namespace)
            exec(self.actor_code, namespace)
        self.assertTrue(torch.equal(namespace['obs']['policy'], before))
        self.assertTrue(all(torch.equal(x, before) for x in seen))
        return namespace

    def test_mixed_startup_right_left_and_ordinary(self):
        ns = self.run_rows(self.namespace([True, False, False, False], [False, True, False, False]))
        self.assertEqual(ns['switched'].tolist(), [False, False, False, True])
        self.assertEqual(ns['stand_active'].tolist(), [True, False, False, True])
        self.assertEqual(int(ns['gate_steps'][0]), 0)
        self.assertTrue(torch.equal(ns['action'][0], ns['stand_action'][0]))
        self.assertTrue(torch.equal(ns['action'][3], ns['stand_action'][3]))
        mirrored = reflect_policy_observation(ns['obs']['policy'])
        expected = reflect_joint_vector(mirrored[:, 12:24] * .1)
        self.assertTrue(torch.equal(ns['action'][1], expected[1]))
        self.assertTrue(torch.equal(ns['action'][2], ns['policy'](ns['obs'])[2]))

    def test_off_matches_original_hard_gate_and_native_candidates(self):
        ns = self.run_rows(self.namespace([False] * 4, [False] * 4))
        self.assertEqual(ns['switched'].tolist(), [True, False, False, True])
        self.assertTrue(torch.equal(ns['roll_action'], ns['policy'](ns['obs'])))
        self.assertTrue(torch.equal(ns['action'], torch.where(ns['switched'][:, None], ns['stand_action'], ns['roll_action'])))

    def test_row_permutation_has_no_cross_trial_effect(self):
        first = self.namespace([True, False, False, False], [False, True, False, False])
        perm = torch.tensor([2, 0, 3, 1])
        second = self.namespace([True, False, False, False], [False, True, False, False])
        for key in ('startup_selected', 'mirror_selected'):
            second[key] = second[key][perm]
        second['obs'] = second['obs'][perm]
        second['a'].projected_gravity_b = second['a'].projected_gravity_b[perm]
        self.run_rows(first)
        self.run_rows(second)
        self.assertTrue(torch.equal(first['action'][perm], second['action']))
        self.assertTrue(torch.equal(first['switched'][perm], second['switched']))

    def test_mirror_candidate_stays_mirrored_after_genuine_switch(self):
        ns = self.namespace([False] * 4, [False, True, False, False])
        ns['a'].projected_gravity_b[1] = torch.tensor([0., 0., -1.])
        self.run_rows(ns)
        self.assertTrue(bool(ns['switched'][1]))
        self.assertTrue(bool(ns['mirror_selected'][1]))
        self.assertTrue(torch.equal(ns['roll_obs']['policy'][1], reflect_policy_observation(ns['obs']['policy'])[1]))
        self.assertTrue(torch.equal(ns['action'][1], ns['stand_action'][1]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
