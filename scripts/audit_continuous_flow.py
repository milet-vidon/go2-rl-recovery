"""Read-only full nonvideo v3 trace audit; a failed audit is NEVER a training gate.

Replays recorded control bookkeeping, not physics or neural-network inference.
The camera, foot-body geometry and actor outputs remain source-bound runtime
measurements, not independently reconstructed meshes or network predictions.
No simulator, model promotion, deployment or file mutation occurs in audit().
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "continuous_flow_trace_audit_v1"
REPORT_PROTOCOL = "continuous_recovery_flow_diagnostic_v3_sensor_identity"
PINS = {
    "scripts/evaluate_continuous_recovery_flow.py": "82570a633b29a15b1b2f29e6f6d5437ec51c9846e963621efd4cf3b93548d9dc",
    "scripts/continuous_flow_runtime.py": "c4ca488783d791295221d1daef754ce4c0fc4352f3c6c076cfc75f21366e13b4",
    "scripts/continuous_flow_phase.py": "9cb68bb55540eeaf23fae6b21abee3f194ccabb48c9184627f364597eea2d373",
    "scripts/continuous_flow_motion_metrics.py": "ce12a20d1b6f72d3c656445529afdfcf77b0a969ae7f9fce2f8ad635214c14b8",
}
ARTIFACTS = {"actual_config.json", "continuous_flow.csv", "extracted_helpers.py", "full_trace.jsonl",
             "initial_state.json", "invocation.json", "measured_rows.json",
             "physical_configuration_comparison.json", "video_frames.json"}
SNAPSHOT_KEYS = {"root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel",
                 "projected_gravity_b", "actual_raw_action", "actual_previous_raw_action", "executed_target",
                 "velocity_command_b", "current_forces_w", "force_history_w", "applied_torque_at_control_boundary",
                 "sim_step", "common_step", "episode_step"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def read(path):
    return decode(Path(path).read_text(encoding="utf-8-sig"))


def equal(actual, expected, label):
    # Keep bool/int and list/shape distinctions visible in JSON evidence.
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), "Mismatch: " + label)


def finite(value):
    if type(value) in (bool, int, str) or value is None:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(finite(item) for item in value)
    if type(value) is dict:
        return not any(key in value for key in ("nonfinite_number", "configuration_nonfinite_number")) and all(finite(v) for v in value.values())
    return False


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def vector(value, width):
    require(type(value) is list and len(value) == width and
            all(type(x) in (int, float) and math.isfinite(x) for x in value), "Invalid measured vector")
    return value


def target_formula(action, interface):
    vector(action, 12)
    nominal = vector(interface["default_joint_positions"], 12)
    limits = interface["soft_joint_limits"]
    require(len(limits) == 12, "Missing soft limits")
    return [min(max(f32(q + f32(.25 * a)), vector(bounds, 2)[0]), bounds[1])
            for q, a, bounds in zip(nominal, action, limits)]


def validate_header(report):
    require(report["protocol_version"] == REPORT_PROTOCOL, "Only full v3 quiet-ready + actual sensor-identity evidence qualifies")
    require(report["passed"] is True and report["status"] == "diagnostic_passed_not_promoted" and
            report["smoke"] is False and report["refinement"] == "off", "Failed, incomplete or smoke report cannot gate training")
    require(report["video_view"] is None and report["frame_count_written"] == 0,
            "This audit is deliberately nonvideo only")
    for key in ("training_performed", "promotion_performed", "post_initialization_reset_allowed"):
        require(report[key] is False, "Unexpected mutation/deployment authority: " + key)
    require(report["source_hashes_unchanged_at_end"] is True and report["live_interface_continuity_passed"] is True,
            "Missing successful live guards")
    require(report["num_envs"] == 1 and report["control_dt"] == .02 and report["physics_dt"] == .005 and
            report["decimation"] == 4 and report["preparation_nominal_pd_intervals"] == 50,
            "Wrong physical dimensions or interval protocol")
    require(report["reset_guard_attempts"] == [] and report["termination_terms"] == ["time_out"] and
            not any(k in report for k in ("failure", "failure_record_error", "source_verification_error")), "Failure/reset evidence present")
    equal(report["initialization"], {"wrapper_initial_reset": True, "additional_env_reset": False,
          "placement_before_recording": True, "no_policy_history_reset_after_pd": True}, "initialization inventory")
    equal(report["command_schedule"], {"preparation": [0.,0.,0.], "recovery": [0.,0.,0.],
          "move": [.5,0.,0.], "stop": [0.,0.,0.]}, "real command schedule")
    require(report["policy_action_mode"] == "deterministic_mean", "Unknown actor inference semantics")


def check_snapshots(record, previous, index, origin, runtime):
    before, after, row = record["before"], record["after"], record["row"]
    require(set(before) == SNAPSHOT_KEYS and set(after) == SNAPSHOT_KEYS, "Incomplete physical/action/history snapshot")
    require(finite(record), "Invalid/nonfinite interval evidence")
    require(runtime.unchanged_between_intervals(previous, before), "Between-interval state/history changed")
    equal(before["velocity_command_b"], after["velocity_command_b"], "within-interval real command")
    require(before["sim_step"] == origin + 4 * index and after["sim_step"] == origin + 4 * (index + 1), "Wrong absolute physical interval")
    runtime.assert_continuity(before, after, row["phase"], origin)
    for field, key in (("simstep", "sim_step"), ("common_step", "common_step"), ("episode_step", "episode_step")):
        equal(row[field + "_before"], before[key], field + " before")
        equal(row[field + "_after"], after[key], field + " after")
    require(row["step"] == index and row["time_s"] == (index + 1) * .02 and row["done"] == 0, "Invalid measured index/time/done")
    command = [.5,0.,0.] if row["phase"] == "move" else [0.,0.,0.]
    equal(before["velocity_command_b"], [command], "phase command")
    equal([row[k] for k in ("cmd_x", "cmd_y", "cmd_yaw")], command, "row measured command")


def check_policy_interface(record, interface, index):
    before, after, detail = record["before"], record["after"], record["interface"]
    native = vector(detail["real_policy48"], 48)
    raw = vector(detail["issued_raw_action"], 12)
    check = detail["retention_interface"]
    require(check["step"] == index and check["physics_mode"] == "recovery" and check["done"] == [0], "Invalid retention step/done")
    equal(check["observation"], [native], "actual observation copy")
    pairs = {"previous_raw_action":"actual_raw_action", "previous_previous_raw_action":"actual_previous_raw_action",
             "previous_target":"executed_target", "joint_pos":"joint_pos", "joint_vel":"joint_vel"}
    for key, snapshot_key in pairs.items():
        equal(check[key], before[snapshot_key], "pre-action " + key)
    for key in ("raw_action", "actual_action"):
        equal(check[key], [raw], key)
    equal(after["actual_raw_action"], [raw], "issued/executed action")
    equal(after["actual_previous_raw_action"], before["actual_raw_action"], "native previous action update")
    equal(check["actual_prev_action"], before["actual_raw_action"], "retention previous action")
    expected = [target_formula(raw, interface)]
    for target in (after["executed_target"], check["executed_target"], check["expected_target"]):
        equal(target, expected, "independently recomputed float32 soft-clamp target")
    equal(native[6:9], before["projected_gravity_b"][0], "actual observation gravity")
    equal(native[9:12], before["velocity_command_b"][0], "actual observation command")
    equal(native[12:24], [f32(q-d) for q,d in zip(before["joint_pos"][0], interface["default_joint_positions"])], "actual relative joint positions")
    equal(native[24:36], [f32(q-d) for q,d in zip(before["joint_vel"][0], interface["default_joint_velocities"])], "actual relative joint velocities")
    equal(native[36:48], before["actual_raw_action"][0], "actual observation history")


def validate_sensor_topology(topology, interface):
    """Require names/indices actually recorded from BOTH different PhysX views.

    Never infer names by matching force time series. In particular the actual
    articulation body_names is NOT the contact sensor's rigid-body-view order.
    """
    keys = {"body_names", "num_bodies", "foot_names", "foot_ids", "base_names", "base_ids",
            "articulation_body_names", "foot_articulation_ids"}
    require(type(topology) is dict and set(topology) == keys, "Missing explicit live sensor/body identity")
    names = topology["body_names"]
    asset_names = topology["articulation_body_names"]
    require(type(names) is list and all(type(name) is str for name in names) and len(set(names)) == len(names),
            "Invalid/duplicate live sensor body names")
    require(type(topology["num_bodies"]) is int and topology["num_bodies"] == len(names) == len(interface["body_names"]),
            "Sensor body count differs from recorded interface")
    equal(asset_names, interface["body_names"], "actual articulation body order")
    require(set(names) == set(asset_names), "Sensor/robot body sets differ")
    equal(topology["foot_names"], [leg+"_foot" for leg in ("FL","FR","RL","RR")], "actual requested foot names")
    equal(topology["base_names"], ["base"], "actual requested base name")
    for labels, indices, ordered_names in ((topology["foot_names"],topology["foot_ids"],names),
            (topology["base_names"],topology["base_ids"],names),
            (topology["foot_names"],topology["foot_articulation_ids"],asset_names)):
        require(type(indices) is list and len(indices) == len(labels) and len(set(indices)) == len(indices) and
                all(type(i) is int and 0 <= i < len(ordered_names) for i in indices), "Invalid live named-body indices")
        equal([ordered_names[i] for i in indices], labels, "indices resolve exact live body names")
    return topology


def check_contact_forces(row, after, topology):
    """Exact current vertical force binding using the explicit SENSOR order."""
    count = topology["num_bodies"]
    current, history = after["current_forces_w"], after["force_history_w"]
    require(type(current) is list and len(current) == 1 and len(current[0]) == count,
            "Current force dimensions differ from explicit sensor topology")
    require(type(history) is list and len(history) == 1 and len(history[0]) == 3 and
            all(len(frame) == count for frame in history[0]), "History force dimensions differ from explicit sensor topology")
    for frame in [current[0], *history[0]]:
        for force in frame:
            vector(force, 3)
    for foot, index in zip(topology["foot_names"], topology["foot_ids"]):
        equal(row[foot+"_fz"], current[0][index][2], "current foot vertical force at actual sensor index")
    return current[0], current[0][topology["base_ids"][0]]


def check_row_snapshot(row, after, interface, topology):
    equal(row["gravity_b"], after["projected_gravity_b"][0], "measured gravity")
    equal([row["vx"], row["vy"], row["wz"]],
          [after["root_lin_vel_w"][0][0], after["root_lin_vel_w"][0][1], after["root_ang_vel_w"][0][2]], "measured world velocities")
    equal(row["yaw_speed"], abs(after["root_ang_vel_w"][0][2]), "measured yaw norm")
    # Binding a GPU norm to raw float32 vectors has reduction-order roundoff.
    # This narrowly bounded arithmetic check NEVER changes a behavior cutoff;
    # an ambiguous physical decision boundary is rejected below instead.
    def norm_binding(value, components, boundaries=()):
        expected = f32(math.hypot(*vector(components, len(components))))
        ulp = math.ldexp(1., math.frexp(expected)[1] - 24) if expected else 2. ** -149
        require(type(value) is float and abs(value - expected) <= 4 * ulp, "Recorded norm inconsistent with actual snapshot vector")
        require(all(abs(value - boundary) > 4 * ulp for boundary in boundaries), "Ambiguous norm decision boundary; fail closed")
    norm_binding(row["linear_speed_3d"], after["root_lin_vel_w"][0], (.06,.10,.5))
    norm_binding(row["angular_speed_3d"], after["root_ang_vel_w"][0], (.15,.20,1.))
    norm_binding(row["speed"], after["root_lin_vel_w"][0][:2], (.06,))
    current, base_force = check_contact_forces(row, after, topology)
    norm_binding(row["current_base_force_n"], base_force, (1.,))
    require(abs(after["projected_gravity_b"][0][2] + .5) > 4 * 2. ** -24,
            "Ambiguous60degree fall boundary; fail closed")
    grounded = any(math.hypot(*force) > 1. for force in current)
    fallen = grounded and (after["projected_gravity_b"][0][2] >= -.5 or
                           (row["current_base_force_n"] > 1. and row["height"] < f32(.26)))
    equal(row["fallen_after"], fallen, "independent later-fall flag")
    for name, q, nominal in zip(interface["native_joint_names"], after["joint_pos"][0], interface["default_joint_positions"]):
        equal(row[name + "_offset_rad"], f32(q - nominal), "measured joint offset")


def check_physics(report, comparison, actual_config, screen, runtime):
    reference, actual = comparison["screened_reference"], comparison["actual"]
    equal(reference, screen["config"]["after"], "selected original screened physics")
    for name in ("robot", "terrain_physics_material", "decimation", "action", "events"):
        equal(actual[name], reference[name], "common physics " + name)
    stripped = lambda value: {k:v for k,v in value.items() if k not in ("device", "render", "render_interval")}
    equal(stripped(actual["sim"]), stripped(reference["sim"]), "simulation dynamics")
    expected = runtime.expected_initialized_config(actual, "/World/envs/env_.*")
    equal(report["physical_config_after_initialization"], expected, "ONLY constructor namespace expansion")
    equal(report["physical_config_after_episode"], expected, "end physics unchanged")
    cfg = actual_config["env"]
    reconstructed = {"robot": cfg["scene"]["robot"], "sim":cfg["sim"],
        "terrain_physics_material":cfg["scene"]["terrain"]["physics_material"], "decimation":cfg["decimation"],
        "action":cfg["actions"]["joint_pos"], "events":{k:cfg["events"][k] for k in actual["events"]},
        "reset_base":cfg["events"]["reset_base"], "reset_robot_joints":cfg["events"]["reset_robot_joints"]}
    equal(actual, reconstructed, "saved actual configuration")
    require(cfg["scene"]["num_envs"] == 1 and cfg["episode_length_s"] == 28., "Unscreened episode topology")
    require(cfg["scene"]["contact_forces"]["prim_path"] == "{ENV_REGEX_NS}/Robot/.*" and
            cfg["scene"]["contact_forces"]["history_length"] == 3, "Unscreened sensor body/history topology")
    for role in ("roll", "stand", "locomotion"):
        equal(report["physical_interface_before"][role], screen["interface"], role + " actual initial interface")
    equal(report["physical_interface_after"], report["physical_interface_before"], "all actual final interfaces")
    require(set(report["physical_interface_before"]) == {"roll", "stand", "locomotion"}, "Wrong controller set")
    return screen["interface"]


def audit(report_path):
    report_path = Path(report_path).resolve()
    require(report_path.name == "continuous_flow_report.json" and report_path.is_relative_to(ROOT / "evaluations"), "Require actual E-drive diagnostic report")
    before_hashes = {str(report_path): sha(report_path), str(Path(__file__).resolve()): sha(__file__)}
    report = read(report_path)
    validate_header(report)
    def bind(path, digest):
        path = Path(path).resolve()
        require(type(digest) is str and len(digest) == 64 and sha(path) == digest, "Source/artifact changed: " + str(path))
        require(str(path) not in before_hashes or before_hashes[str(path)] == digest, "Conflicting evidence SHA")
        before_hashes[str(path)] = digest
    for name, digest in PINS.items():
        bind(ROOT / name, digest)
        require(report["source_and_inputs"].get(str((ROOT / name).resolve())) == digest, "Report did not use pinned v3 source")
    for path, digest in report["source_and_inputs"].items():
        bind(path, digest)
    require(set(report["artifact_sha256"]) == ARTIFACTS, "Require complete nonvideo artifact set")
    directory = report_path.parent
    for name, digest in report["artifact_sha256"].items():
        bind(directory / name, digest)
    # No shared code is executed until its actual bytes and report identity agree.
    import evaluate_continuous_recovery_flow as entry
    import continuous_flow_runtime as runtime
    import continuous_flow_phase as flow
    import continuous_flow_motion_metrics as metrics
    for name, digest in entry.FROZEN.items():
        bind(ROOT / name, digest)
    sys.path.insert(0, str(ROOT / "src/go2_recovery"))
    import torch
    from recovery_handoff_math import update_handoff_gate
    from supported_startup_math import supported_startup_mask
    from roll_mirror_math import initial_right_mask, reflect_policy_observation, reflect_joint_vector
    invocation = read(directory / "invocation.json")
    args = invocation["args"]
    selected = args["locomotion_model"]
    locomotion = entry.verify_locomotion(selected)
    equal(locomotion, report["locomotion_selection"], "selected actor's own full actual prerequisites")
    equal(invocation["evidence"]["locomotion"], locomotion, "invoked actor bundle")
    equal(invocation["evidence"]["source_and_inputs"], report["source_and_inputs"], "invocation/report input identity")
    actual_inventory = entry.source_inventory(SimpleNamespace(checkpoint=Path(args["checkpoint"]), stand_checkpoint=Path(args["stand_checkpoint"])), locomotion)
    equal(actual_inventory, report["source_and_inputs"], "complete actual source/16-export inventory")
    equal(report["actors"], dict(entry.MODEL_SHA, locomotion=locomotion["sha"]), "actual selected actor SHA")
    equal(report["prerequisite"], locomotion["prerequisite"], "selected prerequisite")
    for field in ("pose", "seed", "smoke"):
        equal(report[field], args[field], "invoked " + field)
    require(args["video"] is False and args["speed"] == .5 and args["refinement"] == "off" and
            Path(args["output_dir"]).resolve() == directory and report["task"] == entry.TASK, "Wrong invocation scope")
    equal(invocation["evidence"]["protocol"], REPORT_PROTOCOL, "invocation protocol")
    require((directory / "extracted_helpers.py").read_text(encoding="utf-8") ==
            entry.extract_definitions(entry.MIRROR.read_text(encoding="utf-8")), "Actual extracted helpers differ")
    interface = check_physics(report, read(directory / "physical_configuration_comparison.json"),
                              read(directory / "actual_config.json"), read(locomotion["interface_path"]), runtime)
    topology = validate_sensor_topology(report["contact_sensor_topology"], interface)
    equal(report["contact_sensor_topology_after"], topology, "actual sensor order unchanged at episode end")
    rows = read(directory / "measured_rows.json")
    trace = [decode(line) for line in (directory / "full_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows) == len(trace) == report["actual_completed_intervals"] and 900 <= len(rows) <= 1300, "Incomplete full trace")
    equal(read(directory / "video_frames.json"), [], "nonvideo frame ledger")
    with (directory / "continuous_flow.csv").open(encoding="utf-8", newline="") as handle:
        actual_csv = list(csv.DictReader(handle))
    expected_csv = [{k:json.dumps(v) if isinstance(v, (list,dict)) else str(v) for k,v in row.items()} for row in rows]
    equal(actual_csv, expected_csv, "ALL CSV rows vs original typed measurements")
    initial = read(directory / "initial_state.json")
    origin = report["simstep_origin"]
    require(type(origin) is int and initial["sim_step"] == origin and initial["common_step"] == 0 and initial["episode_step"] == 0, "Wrong initial physical clock")
    equal(initial["actual_raw_action"], [[0.]*12], "initial raw history")
    equal(initial["actual_previous_raw_action"], [[0.]*12], "initial previous history")
    require(type(report["release_state"]) is list and len(report["release_state"]) == 1, "One real initial placement required")
    for key, snapshot_key in (("projected_gravity_b","projected_gravity_b"), ("root_position_w_m","root_pos_w"),
            ("root_quaternion_wxyz","root_quat_w"), ("root_linear_velocity_w_m_s","root_lin_vel_w"),
            ("root_angular_velocity_w_rad_s","root_ang_vel_w"), ("joint_positions_rad","joint_pos")):
        equal(report["release_state"][0][key], initial[snapshot_key][0], "recorded release state " + key)
    start = report["policy_start_state"]
    require(type(start) is list and len(start) == 1, "Require one measured policy start")
    start = start[0]
    startup_mask, startup_records = supported_startup_mask([start])
    equal(report["startup_selection"], startup_records, "actual startup predicate")
    eligible = start["eligible_settled_fallen_recovery"]
    mirror = initial_right_mask(torch.tensor([start["projected_gravity_b"]], dtype=torch.float32),
                                torch.tensor([eligible], dtype=torch.bool), "initial_right")
    equal(report["mirror_selected"], mirror.tolist(), "once actual-start mirror")
    require(report["mirror_classified_once"] is True and not (startup_mask[0] and (eligible or bool(mirror[0]))), "Conflicting startup routing")
    require(report["pose"] == "upright" or eligible is True, "Fallen start not actually eligible")
    state = flow.initial_state(flow.FlowConfig(.5, False), actual_control_step=50)
    gate, switched = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.bool)
    previous, quiet = initial, 0
    for index, (row, record) in enumerate(zip(rows, trace)):
        require(set(record) == {"row", "before", "after", "interface"}, "Missing/extra interval fields")
        equal(record["row"], row, "ALL typed measurement rows vs trace")
        check_snapshots(record, previous, index, origin, runtime)
        before, after, detail = record["before"], record["after"], record["interface"]
        check_row_snapshot(row, after, interface, topology)
        metrics.measured_stance(row)
        if index < 50:
            require(row["phase"] == "preparation" and row["actor_role"] == "nominal_pd" and row["phase_interval"] == index+1, "PD stage lost or reordered")
            equal(detail, {"controller":"direct_nominal_PD", "issued_policy_action":None}, "explicit initialization PD")
            for name in ("actual_raw_action", "actual_previous_raw_action"):
                equal(after[name], before[name], "PD preserves native " + name)
            equal(after["executed_target"], [interface["default_joint_positions"]], "PD target actual nominal")
            grounded = any(math.hypot(*force) > 1. for force in after["current_forces_w"][0])
            quiet = quiet+1 if grounded and row["linear_speed_3d"] < .10 and row["angular_speed_3d"] < .20 and max(abs(v) for v in after["joint_vel"][0]) < .50 else 0
        else:
            check_policy_interface(record, interface, index)
            just = torch.zeros_like(switched)
            if state.phase == "recovery" and index > 50 and not startup_mask[0]:
                native = detail["real_policy48"]
                gate, switched, just = update_handoff_gate(torch.tensor([native[6:9]], dtype=torch.float32),
                    torch.tensor([native[3:6]], dtype=torch.float32), gate, switched, .02)
            equal([detail["old_gate_count"],detail["old_gate_switched"],detail["old_gate_just_switched"]],
                  [int(gate[0]),bool(switched[0]),bool(just[0])], "unchanged original old gate replay")
            require(detail["startup_selected"] is startup_mask[0] and detail["mirror_selected"] is bool(mirror[0]), "Latched routing changed")
            active = bool(startup_mask[0] or switched[0]) if state.phase == "recovery" else None
            plan = flow.next_action(state, recovery_stand_active=active)
            equal(detail["state_before"], asdict(state), "every exact state-machine boundary")
            equal(detail["plan"], asdict(plan), "NEXT action plan (no early phase switch)")
            require(row["phase"] == plan.phase and row["actor_role"] == plan.actor_role, "Actually executed actor role differs")
            if plan.phase == "recovery":
                native = torch.tensor([detail["real_policy48"]], dtype=torch.float32)
                virtual = reflect_policy_observation(native)[0].tolist() if bool(mirror[0]) else detail["real_policy48"]
                equal(detail["roll_virtual48"], virtual, "roll-only virtual observation")
                model_action = torch.tensor([detail["roll_model_action"]], dtype=torch.float32)
                physical = reflect_joint_vector(model_action)[0].tolist() if bool(mirror[0]) else detail["roll_model_action"]
                equal(detail["roll_physical_action"], physical, "inverse roll action reflection")
                equal(detail["issued_raw_action"], detail["stand_action"] if active else physical, "actual roll/stand actor selection")
            measured = flow.CompletedInterval(index, index+1, tuple(before["velocity_command_b"][0]),
                tuple(after["velocity_command_b"][0]), row["strict_valid_after"], row["fallen_after"], False, True, row["quiet_for_handoff"])
            state = flow.after_completed_interval(state, plan, measured)
        previous = after
        if index == 49:
            for key, snapshot_key in (("projected_gravity_b","projected_gravity_b"), ("root_position_w_m","root_pos_w"),
                    ("root_quaternion_wxyz","root_quat_w"), ("root_linear_velocity_w_m_s","root_lin_vel_w"),
                    ("root_angular_velocity_w_rad_s","root_ang_vel_w"), ("joint_positions_rad","joint_pos")):
                equal(start[key], after[snapshot_key][0], "actual post-PD policy start " + key)
            equal(start["quiet_supported_window_s"], quiet * .02, "complete PD settling duration")
            equal(start["geometry_ok"], bool(row["stance_geometry_ok"]), "policy-start measured geometry")
            equal(start["foot_vertical_forces_N"], [row[leg+"_foot_fz"] for leg in ("FL","FR","RL","RR")], "policy-start actual current forces")
            equal(start["fallen_at_policy_start"], row["fallen_after"], "actual-start fallen definition")
    require(state.phase == "complete", "Full phase replay did not complete")
    equal(report["flow_state"], asdict(state), "final phase counters and last150 holds")
    result = metrics.analyze(rows, move_speed=.5)
    equal(report["behavior"], result, "independently recomputed complete behavior metrics")
    require(result["motion_checks_passed"] is True, "Behavior did not pass independently")
    for path, digest in before_hashes.items():
        require(sha(path) == digest, "Evidence/source changed during audit: " + path)
    return {"protocol":PROTOCOL, "audit_passed":True, "training_prerequisite_eligible":True,
            "report_path":str(report_path), "report_sha256":before_hashes[str(report_path)],
            "selected_model":selected, "checkpoint_sha256":locomotion["sha"], "pose":report["pose"], "seed":report["seed"],
            "actual_completed_intervals":len(rows), "policy_intervals":len(rows)-50, "metrics_recomputed_exact":True,
            "source_sha256":before_hashes[str(Path(__file__).resolve())], "source_and_artifacts_sha256":before_hashes,
            "contact_sensor_topology":topology, "explicit_sensor_identity_verified":True,
            "promotion_performed":False,
            "scope":"Full nonvideo v3 .5-command DEVELOPMENT trace/identity/continuity/phase/metric audit only, with explicit live sensor indices. No independent neural inference or mesh/body-geometry replay; first6 body-frame observation values remain pinned runtime-validated measurements. Not natural gait, trot/run, unseen starts or hardware certification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.resolve().drive.upper() == "E:" and not args.output.exists(), "Use a new E-drive audit receipt")
    try:
        result = audit(args.report)
    except Exception as error:
        result = {"protocol":PROTOCOL, "audit_passed":False, "training_prerequisite_eligible":False,
                  "report_path":str(args.report.resolve()), "error":type(error).__name__ + ": " + str(error),
                  "source_sha256":sha(__file__), "promotion_performed":False}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k != "source_and_artifacts_sha256"}, indent=2))
    raise SystemExit(0 if result["audit_passed"] else 1)
