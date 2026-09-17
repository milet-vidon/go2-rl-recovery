import ast
import unittest
import torch
from natural_stance_reward import supported_symmetry_score
from train_natural_handoff_stand import build_source


class NaturalStanceTests(unittest.TestCase):
    def setUp(self):
        self.feet = torch.tensor([[[.2,.16,-.3],[.2,-.16,-.3],[-.2,.16,-.3],[-.2,-.16,-.3]]])
        self.gravity = torch.tensor([[0.,0.,-1.]])
        self.height = torch.tensor([.32])
        self.forces = torch.full((1,4),40.)
        self.base = torch.zeros(1)
        self.command = torch.zeros(1,3)

    def cost(self):
        return supported_symmetry_score(self.feet,self.gravity,self.height,self.forces,self.base,self.command)

    def test_symmetric_best_and_asymmetric_positive_bounded(self):
        self.assertEqual(self.cost().item(),1.)
        self.feet[0,0,1] += .06
        self.assertGreater(self.cost().item(),0.)
        self.assertLess(self.cost().item(),1.)

    def test_inverted_disabled(self):
        self.feet[0,0,1] += .1
        self.gravity *= -1
        self.assertEqual(self.cost().item(),0.)

    def test_low_unloaded_or_base_contact_disabled(self):
        self.feet[0,0,1] += .1
        self.height[:] = .2
        self.assertEqual(self.cost().item(),0.)
        self.height[:] = .32
        self.forces[:,:2] = 0.
        self.assertEqual(self.cost().item(),0.)
        self.forces[:] = 40.
        self.base[:] = 2.
        self.assertEqual(self.cost().item(),0.)

    def test_moving_command_disabled(self):
        self.feet[0,0,1] += .1
        self.command[0,0] = .5
        self.assertEqual(self.cost().item(),0.)

    def test_reflection_invariance_and_no_mutation(self):
        self.feet[0,0,1] += .06
        old = self.feet.clone()
        value = self.cost()
        self.assertTrue(torch.equal(old,self.feet))
        self.feet = self.feet[:,[1,0,3,2]] * torch.tensor([1.,-1.,1.])
        self.assertTrue(torch.equal(value,self.cost()))

    def test_pinned_continuation_source(self):
        source=build_source()
        compile(ast.parse(source),'<natural-training>','exec')
        for text in ('3547, 71120','PARENT_LR = 1e-05','"formal": (128, 200)',
                     'load_optimizer=True','runner.alg.learning_rate = loaded_lr',
                     'Heldout source ID in training payload','weight=2.0'):
            self.assertIn(text,source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
