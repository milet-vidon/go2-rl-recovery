"""CPU-only matching/budget/config guards before either arm can start."""
import ast
import copy
import unittest

import train_balanced_gait_adaptation as entry


class MatchedGaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = {arm: entry.adapter(arm) for arm in ("control", "balanced")}
        base = cls.adapters["control"]
        cls.pe, cls.pa = base.read(base.PARENT_CONFIG / "env.yaml"), base.read(base.PARENT_CONFIG / "agent.yaml")
        cls.reference, cls.parent = base.reference_document(), base.parent_metadata()

    def docs(self, arm, n=128, updates=300):
        trainer=self.adapters[arm]
        tag=f"20260917-gait-{arm}-unit-test"
        env=trainer.expected_environment(self.pe,self.pe,self.reference,n)
        agent=copy.deepcopy(self.pa)
        agent.update(run_name=tag,max_iterations=str(updates),load_run=trainer.PARENT_RUN,load_checkpoint="model_3947.pt")
        return env,agent,tag

    def check(self, arm, env, agent, tag, n=128, updates=300):
        return self.adapters[arm].check_documents(env,agent,self.pe,self.pa,self.reference,n,updates,tag)

    def test_program_is_frozen_template_and_only_finite_budgets(self):
        for arm,trainer in self.adapters.items():
            ast.parse(entry.program(arm))
            self.assertEqual(trainer.PROTOCOL, entry.PROTOCOL)
            for n,updates in ((16,2),(128,300)):
                env,agent,tag=self.docs(arm,n,updates)
                self.assertTrue(self.check(arm,env,agent,tag,n,updates)["full_actual_yaml_checked"])
            for n,updates in ((128,100),(512,300),(16,300),(128,1000)):
                with self.assertRaises(AssertionError):
                    trainer.arguments_ok(n,updates,f"20260917-gait-{arm}-test")

    def test_arms_only_differ_by_new_weight(self):
        a,_,_=self.docs("control")
        b,_,_=self.docs("balanced")
        self.assertEqual(a["rewards"]["balanced_duration"]["weight"],"0.0")
        self.assertEqual(b["rewards"]["balanced_duration"]["weight"],"-10.0")
        b["rewards"]["balanced_duration"]["weight"]="0.0"
        self.assertEqual(a,b)
        for key,value in self.pe["rewards"].items():
            self.assertEqual(a["rewards"][key],value)

    def test_crossarm_or_old_smoke_config_rejected(self):
        env,agent,tag=self.docs("balanced",16,2)
        with self.assertRaises(AssertionError):
            self.check("control",env,agent,tag,16,2)
        tag="20260917-gait-control-unit-test"
        agent["run_name"]=tag
        with self.assertRaises(AssertionError):
            self.check("control",env,agent,tag,16,2)

    def test_changed_command_old_reward_or_physics_rejected(self):
        for path,value in ((["rewards","balanced_duration","weight"],"-20.0"),
                           (["rewards","track_lin_vel_xy_exp","weight"],"2.0"),
                           (["commands","base_velocity","ranges","lin_vel_x"],["-0.5","1.5"]),
                           (["actions","joint_pos","reference"],"current"),
                           (["events","push_robot"],"null")):
            env,agent,tag=self.docs("balanced")
            node=env
            for key in path[:-1]: node=node[key]
            node[path[-1]]=value
            with self.subTest(path=path),self.assertRaises(AssertionError):
                self.check("balanced",env,agent,tag)

    def test_formal_checkpoint_budget_real_adam(self):
        trainer=self.adapters["balanced"]
        candidate=copy.deepcopy(self.parent)
        candidate.update(iter=4246,adam_steps=[85120]*17,sha256="a"*64)
        trainer.check_final(candidate,self.parent,300)
        candidate["adam_steps"][3]=81120
        with self.assertRaises(AssertionError): trainer.check_final(candidate,self.parent,300)

    def test_sources_same_both_arms_and_frozen_models_unchanged(self):
        control=self.adapters["control"].all_sources()
        self.assertEqual(control,self.adapters["balanced"].all_sources())
        paths={row["path"] for row in control}
        for name in ("train_balanced_gait_adaptation.py","balanced_gait_reward.py","balanced_gait_exposure.py","test_balanced_gait_exposure.py"):
            self.assertIn(str((entry.ROOT/"scripts"/name).resolve()),paths)

    def test_source_anchors_fail_closed(self):
        with self.assertRaises(ValueError): entry.replace_once("twice twice","twice","one")
        with self.assertRaises(ValueError): entry.program("other")

    def test_program_retains_all_actual_resume_guards(self):
        for arm in self.adapters:
            text=entry.program(arm)
            for required in ("equal_tree(runner.alg.policy.state_dict()", "equal_tree(runner.alg.optimizer.state_dict()",
                             "verify_smoke(args.smoke_receipt, sources)", "STATE[\"exposure\"].finish()",
                             "int(env.unwrapped.common_step_counter) - STATE[\"initial_common_steps\"] == 24 * args.max_iterations"):
                self.assertIn(required,text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
