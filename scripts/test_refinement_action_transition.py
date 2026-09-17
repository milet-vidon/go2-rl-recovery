"""CPU tensor tests only; NOT evidence of smoother physical recovery.

Run only when the parent has released the serial simulator boundary.
"""
import inspect
import unittest

import torch

import refinement_action_transition as transition
from supported_refinement_gate import advance_after_completed_interval


def inputs(n=1, *, dtype=torch.float32):
    return (torch.arange(n*12, dtype=dtype).reshape(n,12)/10,
            torch.full((n,12), 2., dtype=dtype),
            torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.int64))


class RefinementTransitionTests(unittest.TestCase):
    def test_before_gate_is_exact_old_output_and_zero_progress(self):
        old, new, enabled, edge, count = inputs(3)
        for mode in transition.MODES:
            action, proposed, alpha = transition.transition_action(old,new,enabled,edge,count,mode=mode)
            self.assertTrue(torch.equal(action,old))
            self.assertTrue(torch.equal(proposed,count))
            self.assertTrue(torch.equal(alpha,torch.zeros(3)))
            self.assertNotEqual(action.data_ptr(),old.data_ptr())

    def test_existing150_completed_gate_then_next_action_one_over25(self):
        old,new,enabled,edge,progress = inputs()
        strict_count = torch.zeros(1,dtype=torch.int64)
        valid = torch.ones(1,dtype=torch.bool)
        for completed in range(1,151):
            # The just-issued action is chosen BEFORE this new completed sample.
            action,proposed,alpha = transition.transition_action(old,new,enabled,edge,progress)
            self.assertTrue(torch.equal(action,old))
            self.assertEqual(float(alpha[0]),0.)
            progress = proposed
            strict_count,enabled,edge = advance_after_completed_interval(valid,strict_count,enabled)
            self.assertEqual(bool(enabled[0]),completed==150)
        action,proposed,alpha = transition.transition_action(old,new,enabled,edge,progress)
        expected = (1.-alpha[:,None])*old + alpha[:,None]*new
        self.assertTrue(torch.equal(action,expected))
        self.assertEqual(proposed.tolist(),[1])
        self.assertEqual(alpha.tolist(),torch.tensor([1/25]).tolist())

    def test_invalid_strict_sample_still_restarts_original_gate(self):
        count = torch.tensor([149],dtype=torch.int64)
        enabled = torch.tensor([False])
        count,enabled,edge = advance_after_completed_interval(torch.tensor([False]),count,enabled)
        old,new,_,_,progress = inputs()
        action,_,alpha = transition.transition_action(old,new,enabled,edge,progress)
        self.assertEqual(count.tolist(),[0])
        self.assertTrue(torch.equal(action,old))
        self.assertEqual(alpha.tolist(),[0.])

    def test_exact25_steps_last_endpoint_and_saturation(self):
        old,new,enabled,edge,count = inputs(dtype=torch.float64)
        enabled[:] = True
        for index in range(1,28):
            edge[:] = index==1
            current_new = new + index/7  # Current refiner changes each actual step.
            action,proposed,alpha = transition.transition_action(old,current_new,enabled,edge,count)
            self.assertEqual(proposed.tolist(),[min(index,25)])
            self.assertEqual(alpha.tolist(),[min(index,25)/25])
            if index>=25:
                self.assertTrue(torch.equal(action,current_new))
            count = proposed

    def test_live_old_endpoint_not_a_frozen_previous_action_anchor(self):
        old,new,enabled,edge,count = inputs()
        enabled[:] = True
        edge[:] = True
        first,count,_ = transition.transition_action(old,new,enabled,edge,count)
        next_old = old+5.
        next_new = new-1.
        second,_,alpha = transition.transition_action(next_old,next_new,enabled,torch.zeros_like(edge),count)
        expected = (1.-alpha[:,None])*next_old+alpha[:,None]*next_new
        frozen_first = (1.-alpha[:,None])*first+alpha[:,None]*next_new
        frozen_original = (1.-alpha[:,None])*old+alpha[:,None]*next_new
        self.assertTrue(torch.equal(second,expected))
        self.assertFalse(torch.equal(second,frozen_first))
        self.assertFalse(torch.equal(second,frozen_original))

    def test_hard_comparison_first_action_exact_refiner_no_hidden_ramp(self):
        old,new,enabled,edge,count = inputs()
        enabled[:],edge[:] = True,True
        action,count,alpha = transition.transition_action(old,new,enabled,edge,count,mode="hard")
        self.assertTrue(torch.equal(action,new))
        self.assertEqual((count.tolist(),alpha.tolist()),([1],[1.]))
        action,count,alpha = transition.transition_action(old,new+3.,enabled,torch.zeros_like(edge),count,mode="hard")
        self.assertTrue(torch.equal(action,new+3.))
        self.assertEqual(count.tolist(),[1])

    def test_batch_rows_and_permutation_are_independent(self):
        old,new,_,_,_ = inputs(4)
        enabled = torch.tensor([False,True,True,True])
        edge = torch.tensor([False,True,False,False])
        count = torch.tensor([0,0,12,25],dtype=torch.int64)
        result = transition.transition_action(old,new,enabled,edge,count)
        self.assertTrue(torch.equal(result[0][0],old[0]))
        self.assertTrue(torch.equal(result[0][3],new[3]))
        self.assertEqual(result[1].tolist(),[0,1,13,25])
        order = torch.tensor([3,0,2,1])
        permuted = transition.transition_action(*(x[order] for x in (old,new,enabled,edge,count)))
        for original,reordered in zip(result,permuted):
            self.assertTrue(torch.equal(original[order],reordered))

    def test_no_input_or_rng_mutation_and_no_output_alias(self):
        values = list(inputs(2))
        values[2][:],values[3][:] = True,True
        copies = [x.clone() for x in values]
        rng = torch.random.get_rng_state().clone()
        outputs = transition.transition_action(*values)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(values,copies)))
        for output in outputs:
            self.assertTrue(all(output.data_ptr()!=x.data_ptr() for x in values))

    def test_counter_proposal_does_not_commit_without_real_execution(self):
        old,new,enabled,edge,count = inputs()
        enabled[:],edge[:] = True,True
        first = transition.transition_action(old,new,enabled,edge,count)
        repeated = transition.transition_action(old,new,enabled,edge,count)
        self.assertEqual(count.tolist(),[0])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,repeated)))
        # Caller must commit only after its actual physical interval; this pure
        # function intentionally cannot fabricate/verify execution counters.

    def test_no_clip_scale_pd_or_target_transform(self):
        old,new,enabled,edge,count = inputs()
        old.fill_(12.)
        new.fill_(20.)
        enabled[:],edge[:] = True,True
        action,_,_ = transition.transition_action(old,new,enabled,edge,count)
        self.assertTrue(bool((action>12.).all()))
        self.assertTrue(bool((action<20.).all()))
        params = inspect.signature(transition.transition_action).parameters
        self.assertFalse(set(params)&{"env","reset","target","joint_pos","action_history","roll_action"})

    def test_coefficient_smoothing_does_not_guarantee_action_smoothing(self):
        old,new,enabled,edge,count = inputs()
        old.fill_(100.)
        new.fill_(100.)
        enabled[:],edge[:] = True,True
        first,count,_ = transition.transition_action(old,new,enabled,edge,count)
        second,_,_ = transition.transition_action(-old,-new,enabled,torch.zeros_like(edge),count)
        self.assertGreater(float((second-first).abs().max()),199.)

    def test_invalid_edge_unlatch_retry_and_count_rejected(self):
        old,new,_,_,_ = inputs()
        for enabled,edge,count in ((False,True,0),(False,False,1),(True,True,1),(True,False,0),
                                    (True,False,-1),(True,False,26)):
            with self.assertRaises(ValueError):
                transition.transition_action(old,new,torch.tensor([enabled]),torch.tensor([edge]),
                                             torch.tensor([count],dtype=torch.int64))
        with self.assertRaises(ValueError):
            transition.transition_action(old,new,torch.tensor([True]),torch.tensor([False]),torch.tensor([2]),mode="hard")

    def test_invalid_shapes_dtypes_nonfinite_and_device_rejected(self):
        original = inputs()
        variants = ((0,torch.zeros(1,11)),(0,torch.zeros(0,12)),(0,torch.zeros(1,12,dtype=torch.int64)),
                    (0,torch.zeros(1,12,dtype=torch.float16)),(1,torch.zeros(1,12,dtype=torch.float64)),
                    (2,torch.zeros(1,dtype=torch.int64)),(3,torch.zeros(1,1,dtype=torch.bool)),
                    (4,torch.zeros(1,dtype=torch.int32)),(1,torch.full((1,12),float("nan"))),
                    (0,torch.full((1,12),float("inf"))), (1,torch.empty((1,12),device="meta")))
        for index,value in variants:
            values = list(original)
            values[index] = value
            with self.assertRaises((ValueError,TypeError)):
                transition.transition_action(*values)
        with self.assertRaises(TypeError):
            transition.transition_action(None,*original[1:])

    def test_only_predeclared_duration_and_modes_allowed(self):
        values = inputs()
        for mode in ("off","smooth",25,None):
            with self.assertRaises(ValueError):
                transition.transition_action(*values,mode=mode)
        for dt in (.01,.021,0.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                transition.transition_action(*values,dt=dt)
        for dt in (True,"0.02",None):
            with self.assertRaises(TypeError):
                transition.transition_action(*values,dt=dt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
