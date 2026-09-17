"""CPU-only auditor structure/mutation tests; NO fabricated passing full episode.

Historical v1 measured intervals are used solely as real schema substrates for
negative mutation tests. v1/full behavior failure is never upgraded to a gate.
Run the auditor separately on an actual complete passed v3 after these tests.
Synthetic sensor permutations below test only isolated index contracts and
cannot provide or substitute for an actual live sensor-topology receipt.
"""
import copy
import json
from pathlib import Path
import unittest

import audit_continuous_flow as audit


class ContinuousAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = audit.ROOT / "evaluations/20260917-continuous05-side-full-v1"
        cls.report = audit.read(cls.directory / "continuous_flow_report.json")
        with (cls.directory / "full_trace.jsonl").open(encoding="utf-8") as handle:
            cls.prefix = [audit.decode(next(handle)) for _ in range(51)]
        cls.interface = cls.report["physical_interface_before"]["roll"]

    def test_pins_full_width_and_real_current_source(self):
        for name, digest in audit.PINS.items():
            self.assertEqual(len(digest), 64)
            self.assertEqual(audit.sha(audit.ROOT / name), digest)

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                audit.decode(text)

    def test_boolean_number_coercion_is_not_exact_evidence(self):
        for a, b in ((True,1), (False,0), ([1],[1.]), ({"x":True},{"x":1})):
            with self.assertRaises(ValueError):
                audit.equal(a,b,"typed evidence")

    def test_actual_v1_failure_can_never_gate(self):
        with self.assertRaises(ValueError):
            audit.audit(self.directory / "continuous_flow_report.json")

    def test_smoke_headers_rejected_without_relabeling(self):
        for name in ("20260917-continuous05-control3947-side-smoke-v2", "20260917-continuous05-balanced4246-side-smoke-v2"):
            report = audit.read(audit.ROOT / "evaluations" / name / "continuous_flow_report.json")
            with self.assertRaises(ValueError):
                audit.validate_header(report)

    def test_old_v2_behavior_pass_without_sensor_identity_cannot_gate(self):
        for name in ("20260917-continuous05-control3947-side-full-v2", "20260917-continuous05-balanced4246-side-full-v2"):
            with self.assertRaises(ValueError):
                audit.audit(audit.ROOT / "evaluations" / name / "continuous_flow_report.json")

    def synthetic_topology(self):
        # Deliberately NOT a claimed reconstruction of any recorded episode.
        asset_names=list(self.interface["body_names"])
        names=list(reversed(asset_names))
        feet=[leg+"_foot" for leg in ("FL","FR","RL","RR")]
        return {"body_names":names,"num_bodies":len(names),"foot_names":feet,
                "foot_ids":[names.index(foot) for foot in feet],"base_names":["base"],
                "base_ids":[names.index("base")],"articulation_body_names":asset_names,
                "foot_articulation_ids":[asset_names.index(foot) for foot in feet]}

    def test_explicit_sensor_permutation_not_articulation_order(self):
        topology=self.synthetic_topology()
        audit.validate_sensor_topology(topology,self.interface)
        self.assertNotEqual(topology["foot_ids"],topology["foot_articulation_ids"])
        forces=[[0.,0.,float(i+1)] for i in range(topology["num_bodies"])]
        after={"current_forces_w":[forces],"force_history_w":[[copy.deepcopy(forces) for _ in range(3)]]}
        row={name+"_fz":forces[index][2] for name,index in zip(topology["foot_names"],topology["foot_ids"])}
        current,base=audit.check_contact_forces(row,after,topology)
        self.assertEqual(base,forces[topology["base_ids"][0]])
        wrong=copy.deepcopy(topology)
        wrong["foot_ids"]=wrong["foot_articulation_ids"]
        with self.assertRaises(ValueError):
            audit.validate_sensor_topology(wrong,self.interface)
        with self.assertRaises(ValueError):
            audit.check_contact_forces(row,after,wrong)
        self.assertEqual(current,forces)

    def test_missing_ambiguous_or_misresolved_sensor_identity_rejected(self):
        for mutate in (lambda x:x.pop("body_names"),lambda x:x.update(num_bodies=18),
                       lambda x:x["body_names"].__setitem__(0,x["body_names"][1]),
                       lambda x:x["foot_ids"].__setitem__(0,x["foot_ids"][1]),
                       lambda x:x.update(base_ids=x["foot_ids"][:1]),
                       lambda x:x["foot_articulation_ids"].reverse()):
            bad=self.synthetic_topology()
            mutate(bad)
            with self.assertRaises(ValueError):
                audit.validate_sensor_topology(bad,self.interface)

    def test_sensor_force_dimensions_and_exact_fz_cannot_be_skipped(self):
        topology=self.synthetic_topology()
        forces=[[0.,0.,float(i+1)] for i in range(topology["num_bodies"])]
        after={"current_forces_w":[forces],"force_history_w":[[copy.deepcopy(forces) for _ in range(3)]]}
        row={name+"_fz":forces[index][2] for name,index in zip(topology["foot_names"],topology["foot_ids"])}
        for mutate in (lambda x:x["current_forces_w"][0].pop(),lambda x:x["force_history_w"][0].pop(),
                       lambda x:x["current_forces_w"][0][topology["foot_ids"][0]].__setitem__(2,999.)):
            bad=copy.deepcopy(after)
            mutate(bad)
            with self.assertRaises(ValueError):
                audit.check_contact_forces(row,bad,topology)

    def test_policy_history_and_observation_mutations_rejected(self):
        mutations = (
            lambda x: x["before"]["actual_raw_action"][0].__setitem__(0,.25),
            lambda x: x["after"]["actual_previous_raw_action"][0].__setitem__(0,.25),
            lambda x: x["interface"]["real_policy48"].__setitem__(9,.5),
            lambda x: x["interface"]["retention_interface"]["previous_previous_raw_action"][0].__setitem__(0,.25),
            lambda x: x["interface"]["retention_interface"].update(done=[1]),
        )
        for mutate in mutations:
            row = copy.deepcopy(self.prefix[50])
            mutate(row)
            with self.assertRaises(ValueError):
                audit.check_policy_interface(row,self.interface,50)

    def test_matching_false_target_labels_do_not_evade_formula(self):
        row = copy.deepcopy(self.prefix[50])
        row["after"]["executed_target"][0][0] += .01
        row["interface"]["retention_interface"]["executed_target"] = copy.deepcopy(row["after"]["executed_target"])
        row["interface"]["retention_interface"]["expected_target"] = copy.deepcopy(row["after"]["executed_target"])
        with self.assertRaises(ValueError):
            audit.check_policy_interface(row,self.interface,50)

    def test_target_formula_float32_and_soft_bounds_not_action_clipping(self):
        result = audit.target_formula([1e6]*12,self.interface)
        self.assertEqual(result,[pair[1] for pair in self.interface["soft_joint_limits"]])
        result = audit.target_formula([-1e6]*12,self.interface)
        self.assertEqual(result,[pair[0] for pair in self.interface["soft_joint_limits"]])
        self.assertEqual(audit.target_formula([0.]*12,self.interface),self.interface["default_joint_positions"])

    def test_missing_substep_and_history_reset_between_intervals_rejected(self):
        import continuous_flow_runtime as runtime
        for mutate in (lambda x:x["before"].update(sim_step=x["before"]["sim_step"]+4),
                       lambda x:x["before"]["joint_pos"][0].__setitem__(0,123.),
                       lambda x:x["after"].update(common_step=123),
                       lambda x:x["row"].update(simstep_after=0)):
            row=copy.deepcopy(self.prefix[50])
            mutate(row)
            with self.assertRaises((ValueError,RuntimeError)):
                audit.check_snapshots(row,self.prefix[49]["after"],50,self.report["simstep_origin"],runtime)

    def test_measured_rows_cannot_disagree_with_real_force_joint_gravity(self):
        record=self.prefix[50]
        for key, value in (("FL_foot_fz",9999.),("FL_hip_joint_offset_rad",.6),
                           ("gravity_b",[0.,0.,-1.]),("linear_speed_3d",0.),("fallen_after",not record["row"]["fallen_after"])):
            row=copy.deepcopy(record["row"])
            row[key]=value
            with self.assertRaises(ValueError):
                audit.check_row_snapshot(row,record["after"],self.interface,self.synthetic_topology())

    def test_only_allowed_preinit_namespace_mapping_not_dynamics(self):
        import continuous_flow_runtime as runtime
        before={"robot":{"prim_path":"{ENV_REGEX_NS}/Robot","mass":7.},"dt":.005}
        result=runtime.expected_initialized_config(before,"/World/envs/env_.*")
        changed=copy.deepcopy(result)
        changed["robot"]["mass"]=8.
        with self.assertRaises(ValueError):
            audit.equal(changed,result,"physics unchanged")
        with self.assertRaises(RuntimeError):
            runtime.expected_initialized_config(before,"/World/envs/env_0")

    def test_no_simulator_or_external_mutation_in_audit_function(self):
        import ast
        source=Path(audit.__file__).read_text(encoding="utf-8")
        tree=ast.parse(source)
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="audit")
        calls=[node.func for node in ast.walk(function) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(func,ast.Attribute) and func.attr in ("step","reset","_reset_idx","write_text","write_bytes","mkdir") for func in calls))
        self.assertNotIn("AppLauncher",source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
