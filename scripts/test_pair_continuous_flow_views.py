"""Pure CPU structure/adversarial tests only; NO synthetic physical acceptance.

The in-memory header fixture checks schema rejection, not real-world behavior.
No test builds a paired video or writes a passed simulation report. Schedule
fixtures omit physical measurements deliberately and cannot pass load_case().
"""
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import pair_continuous_flow_views as pair


def schedule(recovery=150):
    rows = []
    for phase, count in (("preparation",50),("recovery",recovery),("move",400),("stop",300)):
        for interval in range(1, count+1):
            index = len(rows)
            rows.append({"step":index,"time_s":(index+1)*.02,"phase":phase,"phase_interval":interval,
                         "cmd_x":.5 if phase == "move" else 0.,"cmd_y":0.,"cmd_yaw":0.})
    return rows


def ledger(rows):
    return [{"frame":i,"time_s":0. if i == 0 else rows[i-1]["time_s"],"completed_interval_step":i-1,
             "rendered_preencoding_bgr_sha256":"d"*64} for i in range(len(rows)+1)]


def header(view="front"):
    """Only an in-memory metadata fixture; it contains no physical evidence."""
    actors = dict(pair.ACTOR_SHA,locomotion=pair.LOCOMOTION_SHA["control3947"])
    selection = {"path":"synthetic-model","sha":actors["locomotion"],"selected_model":"control3947",
                 "interface_path":"synthetic-interface","actual_passed_interface_path":"synthetic-interface",
                 "screen_report_path":"synthetic-screen","screen_protocol":"locomotion_recovery_physics_retention_v1",
                 "prerequisite":{},"verified_inputs":{"synthetic-model":actors["locomotion"]}}
    asset_names = ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    # Intentionally different PhysX view order, not an inferred articulation order.
    topology = {"body_names":["RR_foot", "base", "FR_foot", "FL_foot", "RL_foot"], "num_bodies":5,
                "foot_names":["FL_foot", "FR_foot", "RL_foot", "RR_foot"], "foot_ids":[3,2,4,0],
                "base_names":["base"], "base_ids":[1], "articulation_body_names":asset_names,
                "foot_articulation_ids":[1,2,3,4]}
    interfaces = {role:{"body_names":asset_names} for role in actors}
    return {"protocol_version":pair.PROTOCOL,"passed":True,"status":"diagnostic_passed_not_promoted",
            "smoke":False,"refinement":"off","video_view":view,"seed":20260918,"pose":"side","num_envs":1,
            "task":"Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0",
            "live_interface_continuity_passed":True,"source_hashes_unchanged_at_end":True,"mirror_classified_once":True,
            "post_initialization_reset_allowed":False,"training_performed":False,"promotion_performed":False,
            "reset_guard_attempts":[],"control_dt":.02,"physics_dt":.005,"decimation":4,
            "preparation_nominal_pd_intervals":50,"policy_action_mode":"deterministic_mean",
            "command_schedule":{"preparation":[0.,0.,0.],"recovery":[0.,0.,0.],"move":[.5,0.,0.],"stop":[0.,0.,0.]},
            "initialization":{"wrapper_initial_reset":True,"additional_env_reset":False,
                              "placement_before_recording":True,"no_policy_history_reset_after_pd":True},
            "actors":actors,"locomotion_selection":selection,"source_and_inputs":{role:digest for role,digest in actors.items()},
            "physical_interface_before":copy.deepcopy(interfaces),"physical_interface_after":copy.deepcopy(interfaces),
            "contact_sensor_topology":copy.deepcopy(topology),"contact_sensor_topology_after":copy.deepcopy(topology),
            "physical_config_after_initialization":{"synthetic":True},"physical_config_after_episode":{"synthetic":True},
            "startup_selection":[{"synthetic":True}],"release_state":[{"synthetic":True}],"mirror_joint_contract":{"synthetic":True},
            "bank_selection":{"selected_unique_state_count":1,"selected_state_ids":[132]},
            "policy_start_state":[{"bank_state_id":132,"eligible_settled_fallen_recovery":True,"fallen_at_policy_start":True}],
            "behavior":{"protocol":pair.METRICS_PROTOCOL,"motion_checks_passed":True,"checks":{k:True for k in pair.CHECKS}},
            "flow_state":{"phase":"complete","failure_reason":None,"initial_control_step":50,"stop_strict_count":150,
                          "recovery_ready_count":150,"recovery_strict_count":150,"config":{"move_speed":.5,"include_refinement":False}}}


def metadata_case(view):
    return {"directory":Path("synthetic-"+view),"report":header(view),"rows":schedule(),
            "invocation":{"args":{"view":view,"output_dir":"synthetic-"+view,"speed":.5},"evidence":{"synthetic":True}}}


def fake_decoder(*, rate=50, width=960, height=540, advertised=3, decoded=3):
    """Fake decoder API for accounting rejection only; no actual media generated."""
    class Capture:
        index = 0
        released = False
        def isOpened(self): return True
        def get(self,key): return {1:rate,2:advertised,3:width,4:height}[key]
        def read(self):
            self.index += 1
            return (True,SimpleNamespace(shape=(height,width,3))) if self.index <= decoded else (False,None)
        def release(self): self.released=True
    cap=Capture()
    return SimpleNamespace(CAP_PROP_FPS=1,CAP_PROP_FRAME_COUNT=2,CAP_PROP_FRAME_WIDTH=3,
                           CAP_PROP_FRAME_HEIGHT=4,VideoCapture=lambda _:cap),cap


class StructureTests(unittest.TestCase):
    def test_header_and_matching_structure_are_not_physical_acceptance(self):
        self.assertIsNone(pair.validate_report_header(header(),"front"))
        self.assertEqual(pair.match_cases(metadata_case("front"),metadata_case("oblique")),901)
        from continuous_flow_motion_metrics import analyze
        # Missing all real positions/forces: these schedules can never certify motion.
        with self.assertRaises((ValueError,KeyError)):
            analyze(schedule(),move_speed=.5)

    def test_schedule_bounds_structure_only(self):
        for count in (150,550):
            self.assertEqual(pair.validate_rows_schedule(schedule(count))["recovery"],count)

    def test_missing_row_or_phase_interruption(self):
        rows=schedule(); rows.pop(6)
        with self.assertRaises(ValueError): pair.validate_rows_schedule(rows)
        rows=schedule(); rows[49]["phase"]="recovery"
        with self.assertRaises(ValueError): pair.validate_rows_schedule(rows)

    def test_command_changed_or_bool_step(self):
        for field,value in (("cmd_x",.8),("step",True),("time_s",.02000000000001)):
            rows=schedule(); rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): pair.validate_rows_schedule(rows)

    def test_complete_frame_ledger_structure_only(self):
        rows=schedule(); self.assertIsNone(pair.validate_ledger(ledger(rows),rows))

    def test_frame_missing_duplicate_time_and_bad_hash(self):
        rows=schedule()
        frames=ledger(rows)[1:]
        with self.assertRaises(ValueError): pair.validate_ledger(frames,rows)
        for field,value in (("frame",5),("completed_interval_step",5),("time_s",.02),
                            ("rendered_preencoding_bgr_sha256","invalid")):
            frames=ledger(rows); frames[7][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): pair.validate_ledger(frames,rows)

    def test_old_protocol_smoke_failed_and_false_as_int(self):
        for key,value in (("protocol_version","continuous_recovery_flow_diagnostic_v1"),("smoke",True),
                          ("protocol_version","continuous_recovery_flow_diagnostic_v2_quiet_ready"),
                          ("passed",False),("passed",1),("status","smoke_complete_not_full_flow"),
                          ("live_interface_continuity_passed",False),("source_hashes_unchanged_at_end",False),
                          ("promotion_performed",True),("num_envs",True)):
            report=header(); report[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): pair.validate_report_header(report,"front")

    def test_missing_required_metadata(self):
        for key in ("locomotion_selection","actors","bank_selection","policy_start_state",
                    "physical_interface_before","behavior","flow_state","source_and_inputs",
                    "contact_sensor_topology","contact_sensor_topology_after"):
            report=header(); del report[key]
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError)): pair.validate_report_header(report,"front")

    def test_separate_actual_sensor_and_articulation_orders(self):
        report=header()
        self.assertNotEqual(report["contact_sensor_topology"]["body_names"],
                            report["physical_interface_before"]["stand"]["body_names"])
        self.assertEqual(pair.validate_sensor_topology(report)["foot_ids"],[3,2,4,0])
        report["contact_sensor_topology"]["foot_ids"]=[1,2,3,4]
        report["contact_sensor_topology_after"]=copy.deepcopy(report["contact_sensor_topology"])
        with self.assertRaises(ValueError):pair.validate_sensor_topology(report)

    def test_topology_bad_types_duplicate_indices_and_body_count(self):
        for key,value in (("num_bodies",True),("num_bodies",6),("foot_ids",[True,2,4,0]),
                          ("foot_ids",[3,2,4,-1]),("foot_ids",[3,2,4,5]),("foot_ids",[3,2,4,4]),
                          ("foot_ids",[3,2,4]),("base_ids",[0]),("foot_articulation_ids",[3,2,4,0]),
                          ("body_names",["RR_foot","base","FR_foot","FL_foot","FL_foot"]),
                          ("body_names",["RR_foot","base","FR_foot","FL_foot","unknown"]),
                          ("foot_names",["FR_foot","FL_foot","RL_foot","RR_foot"])):
            report=header();report["contact_sensor_topology"][key]=value
            report["contact_sensor_topology_after"]=copy.deepcopy(report["contact_sensor_topology"])
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):pair.validate_sensor_topology(report)

    def test_topology_changed_at_end_or_interface_mismatch(self):
        report=header();report["contact_sensor_topology_after"]["base_ids"]=[0]
        with self.assertRaises(ValueError):pair.validate_sensor_topology(report)
        for endpoint in ("physical_interface_before","physical_interface_after"):
            report=header();report[endpoint]["locomotion"]["body_names"].reverse()
            with self.subTest(endpoint=endpoint),self.assertRaises(ValueError):pair.validate_sensor_topology(report)

    def test_empty_or_failed_behavior_not_accepted(self):
        for checks in ({},{k:False for k in pair.CHECKS},{k:1 for k in pair.CHECKS}):
            report=header(); report["behavior"]["checks"]=checks
            with self.assertRaises(ValueError): pair.validate_report_header(report,"front")

    def test_wrong_bank_actor_physics_and_selection(self):
        modifications=(lambda r:r["policy_start_state"][0].update(bank_state_id=133),
                       lambda r:r["actors"].update(locomotion="e"*64),
                       lambda r:r["physical_interface_after"].update(changed=True),
                       lambda r:r["flow_state"].update(recovery_ready_count=149),
                       lambda r:r["locomotion_selection"].update(selected_model="balanced4246"))
        for change in modifications:
            report=header(); change(report)
            with self.assertRaises(ValueError): pair.validate_report_header(report,"front")

    def test_any_failure_key_or_reset_attempt_is_rejected(self):
        for key,value in (("failure",{}),("failure_record_error","bad"),("reset_guard_attempts",["attempt"])):
            report=header();report[key]=value
            with self.assertRaises(ValueError): pair.validate_report_header(report,"front")

    def test_exact_entire_rows_including_preparation(self):
        a,b=metadata_case("front"),metadata_case("oblique")
        b["rows"][3]["additional_measurement"]=1e-15
        with self.assertRaises(ValueError): pair.match_cases(a,b)

    def test_equal_endpoint_does_not_hide_middle_change(self):
        a,b=metadata_case("front"),metadata_case("oblique")
        b["rows"][432]["cmd_x"]+=1e-15
        with self.assertRaises(ValueError): pair.match_cases(a,b)

    def test_selection_and_unexpected_report_field_mismatch(self):
        for key,value in (("seed",20260919),("unexpected_new_metadata",True)):
            a,b=metadata_case("front"),metadata_case("oblique");b["report"][key]=value
            with self.assertRaises(ValueError):pair.match_cases(a,b)

    def test_exact_types_and_nonfinite(self):
        for a,b in ((1,True),(1,1.0),(0.,1e-16)):
            with self.assertRaises(ValueError):pair.exact(a,b,"different")
        with self.assertRaises(ValueError):pair.canonical({"x":float("nan")})

    def test_bad_artifact_names_and_missing_inventory(self):
        with self.assertRaises(ValueError):pair.verify_artifacts(Path("synthetic"),{})
        for name in ("../escape","x/y","x\\y","video:stream",pair.REPORT):
            inventory={k:"a"*64 for k in pair.REQUIRED_ARTIFACTS};inventory={name:"a"*64,**inventory}
            with self.assertRaises(ValueError):pair.verify_artifacts(Path("synthetic"),inventory)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pair-flow-test-",dir=pair.ROOT.parent/"tmp") as temp:
            file=Path(temp)/"invalid.json"
            for text in ('{"passed":false,"passed":true}','{"x":NaN}','{"x":Infinity}'):
                file.write_text(text,encoding="utf-8")
                with self.assertRaises(ValueError):pair.read_json(file)

    def test_artifact_modification_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pair-flow-test-",dir=pair.ROOT.parent/"tmp") as temp:
            directory=Path(temp).resolve();inventory={}
            for name in pair.REQUIRED_ARTIFACTS:
                file=directory/name;file.write_bytes(b"NON-PHYSICAL artifact hash fixture")
                inventory[name]=pair.digest(file)
            (directory/pair.VIDEO).write_bytes(b"changed")
            with self.assertRaises(ValueError):pair.verify_artifacts(directory,inventory)

    def test_ledger_media_endpoint_disclosure(self):
        rows=schedule();frames=ledger(rows)
        self.assertEqual(frames[-1]["time_s"],len(rows)/50)
        self.assertAlmostEqual(len(frames)/50-frames[-1]["time_s"],.02)
        self.assertTrue(any("NOT TROT OR FAST RUN" in s for s in pair.BANNER))
        self.assertTrue(any("NOT simultaneous" in s for s in pair.BANNER))

    def test_complete_decode_accounting_only(self):
        api,cap=fake_decoder()
        self.assertEqual(pair.probe_video("synthetic-not-a-file",3,960,540,api),3)
        self.assertTrue(cap.released)

    def test_bad_media_size_rate_short_or_extra_decode(self):
        for settings in ({"rate":25},{"width":480},{"height":270},{"advertised":2},
                         {"decoded":2},{"decoded":4}):
            api,cap=fake_decoder(**settings)
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                pair.probe_video("synthetic-not-a-file",3,960,540,api)
            self.assertTrue(cap.released)


if __name__ == "__main__":
    unittest.main()
