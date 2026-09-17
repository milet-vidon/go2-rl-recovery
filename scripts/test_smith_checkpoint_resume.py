"""Pure-Python checks: no torch, simulator or file writes."""
import unittest
from types import SimpleNamespace as NS
from resume_smith_checkpoint_entry import restore_resume_state

def runner():
    return NS(current_learning_iteration=2900,
              alg=NS(learning_rate=.0005,optimizer=NS(
                  param_groups=[{"lr":1e-5}],state={i:{"step":58040} for i in range(17)})))

class ResumeTests(unittest.TestCase):
    def test_exact_restore(self):
        r=runner(); state=r.alg.optimizer.state
        result=restore_resume_state(r)
        self.assertEqual(r.current_learning_iteration,2901)
        self.assertEqual(r.alg.learning_rate,1e-5)
        self.assertIs(r.alg.optimizer.state,state)
        self.assertEqual(result["remaining_updates"],98)
        self.assertEqual(2901+98-1,2998)
    def test_wrong_iteration(self):
        r=runner(); r.current_learning_iteration=2899
        with self.assertRaises(RuntimeError):restore_resume_state(r)
    def test_wrong_lr(self):
        r=runner(); r.alg.optimizer.param_groups[0]["lr"]=.0005
        with self.assertRaises(RuntimeError):restore_resume_state(r)
    def test_empty_optimizer(self):
        r=runner(); r.alg.optimizer.state={}
        with self.assertRaises(RuntimeError):restore_resume_state(r)
    def test_wrong_adam_budget(self):
        r=runner(); r.alg.optimizer.state[0]["step"]=58020
        with self.assertRaises(RuntimeError):restore_resume_state(r)

if __name__=="__main__":unittest.main()
