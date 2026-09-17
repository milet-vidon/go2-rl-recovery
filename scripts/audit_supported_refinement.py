"""Read-only full-trajectory consistency audit, never performance promotion.

OFF must exactly preserve the frozen combined baseline. ON additionally needs a
fully audited OFF report of the same pose. Inputs are never relabelled or edited.
Only the training-receipt verifier loads Torch, explicitly on CPU; no simulator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import audit_handoff_mirror_coordinates as mirror
import compare_handoff_combined as combined
import compare_natural_handoff_candidate as natural
import evaluate_supported_refinement as entry

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "supported_refinement_full_trajectory_audit_v1"
ENTRY_SHA = "bdb03cf09fafa9c0ed5aa14ee8818b4f95b6d145300e33b6e5cac8bd1b0ad0ff"
PINS = {
    "evaluate_supported_refinement.py": ENTRY_SHA,
    "audit_handoff_mirror_coordinates.py": "6337b44e1a9eae37b7697f4b9f4c6a7c984ab6577d1f08b6c1463c1228e3a281",
    "compare_handoff_combined.py": "07f0e3f3c9f2cea5cf56afb4987032328b240309c42336092add6896c2b9c7eb",
    "compare_natural_handoff_candidate.py": "c3c89878bc1a0c0b0c8fc71cf6df03a1144885252eaca2c7ddd302fed1bc847d",
}
ROLL_SHA = "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c"
STAND_SHA = "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"
REFINEMENT_SHA = "f40772c228996400a9813ad244b48509737dabf2c31af001144de34283433b9e"
RECEIPT_SHA = "531fde006c9f164f7535f0937f7a2828935eb868ed2f39cc00991fe49bdbb94f"
require, exact, vector, f32 = mirror.require, mirror.exact_subset, mirror.vector, mirror.f32
GROUP_EXCLUSIONS = {
    "transition_experiment": {"evaluator_source_sha256", "transition_trace_sha256", "hard_zero_semantics"},
    "startup_experiment": {"adapter_sha256", "generated_source_sha256", "transition_trace_sha256"},
    "mirror_experiment": {"evaluator_source_sha256", "mirror_trace_sha256"},
    "combined_experiment": {"adapter_sha256", "generated_source_sha256", "trace_sha256"},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sibling(report_path, name):
    require(type(name) is str and Path(name).name == name and ":" not in name and "\\" not in name,
            "Evidence must be a sibling basename")
    path = report_path.parent / name
    require(path.resolve().parent == report_path.resolve().parent, "Evidence escapes report directory")
    return path


def norm3(values):
    return math.sqrt(sum(x * x for x in vector(values, 3, "physical vector")))


def norm_comparison(values, threshold, less=True):
    """Fail closed near CPU/GPU reduction rounding, never soften acceptance."""
    value = norm3(values)
    require(abs(value - threshold) > 2e-7, "Ambiguous float32 norm threshold; cannot certify")
    return value < threshold if less else value > threshold


def strict_standing(row, defaults):
    """Recompute from AFTER-step measurements, not stored success flags.

    Four CURRENT vertical forces >5 imply the legacy history-max magnitude test
    because frozen sensor history contains its current sample. No contact-history
    union is substituted for simultaneous support.
    """
    after = row["after"]
    gravity = vector(row["projected_gravity_b_after"], 3, "actual gravity")
    feet, knees = row["feet_b_after_m"], row["knees_b_after_m"]
    require(len(feet) == len(knees) == 4, "Need FL/FR/RL/RR geometry")
    for item in feet + knees:
        vector(item, 3, "foot/knee xyz")
    q = vector(after["joint_positions_rad"], 12, "actual joint positions")
    delta = [f32(x - d) for x, d in zip(q, defaults)]
    geometry = (all(feet[i][1] * s > f32(.06) and abs(feet[i][1]) < f32(.30)
                    and knees[i][1] * s > f32(.04) for i, s in enumerate((1, -1, 1, -1)))
                and all(feet[i][0] * s > f32(.08) for i, s in enumerate((1, 1, -1, -1)))
                and max(abs(x) for x in delta) < f32(.65))
    foot_forces = after["current_foot_forces_w_n"]
    base_forces = after["current_base_forces_w_n"]
    require(len(foot_forces) == 4 and len(base_forces) == 1, "Wrong measured contact bodies")
    for force in foot_forces + base_forces:
        vector(force, 3, "current force")
    contacts = sum(force[2] > 5 for force in foot_forces)
    base = any(norm_comparison(force, 1, less=False) for force in base_forces)
    require(after["current_base_contact"] is base, "Base contact flag disagrees with actual force")
    gravity_delta = [gravity[0], gravity[1], f32(gravity[2] + 1)]
    valid = (norm_comparison(gravity_delta, f32(.35))
             and f32(.30) < after["height_m"] < f32(.55)
             and norm_comparison(after["root_linear_velocity_w_m_s"], .5)
             and norm_comparison(after["root_angular_velocity_w_rad_s"], 1)
             and contacts == 4 and not base and geometry)
    return {"valid": valid, "geometry": geometry, "contacts": contacts,
            "gravity_error": norm3(gravity_delta), "max_joint_offset_rad": max(abs(x) for x in delta)}


def gate_after(valid, stand_active, count, latched):
    require(type(valid) is bool and type(stand_active) is bool and type(latched) is bool
            and type(count) is int and 0 <= count <= 150, "Invalid gate input")
    require(count < 150 or latched, "Saturated but unlatched gate")
    next_count = min(count + 1, 150) if valid and stand_active else 0
    return next_count, latched or next_count == 150


def issued_and_history(row, previous, defaults, limits, selected_mirror):
    label = f"trial{row['trial']}.step{row['step']}"
    mirror.audit_coordinate_record(row, selected_mirror, label)
    require(row["issued_action_target_history_assertions_passed"] is True, "Missing live checks")
    require(type(row["stand_active"]) is bool and type(row["refinement_active"]) is bool, "Actor masks not bool")
    old = row["stand_actor_raw_action"] if row["stand_active"] else row["roll_actor_raw_action"]
    exact(old, row["original_recovery_raw_action"], label + ".old_actor_selection")
    refined = vector(row["refinement_actor_raw_action"], 12, "refinement action")
    action = refined if row["refinement_active"] else old
    exact(action, row["issued_raw_action"], label + ".issued")
    exact(action, row["actual_raw_action"], label + ".actual")
    exact(row["previous_raw_action"], row["actual_previous_raw_action"], label + ".advanced_history")
    raw = [f32(defaults[j] + f32(.25 * action[j])) for j in range(12)]
    target = [min(limits[j][1], max(limits[j][0], raw[j])) for j in range(12)]
    exact(target, row["expected_executed_joint_target_rad"], label + ".expected_target")
    exact(target, row["actual_executed_joint_target_rad"], label + ".actual_target")
    clamp = [raw[j] < limits[j][0] or raw[j] > limits[j][1] for j in range(12)]
    exact(clamp, row["clamp_mask"], label + ".clamp_mask")
    prev_target = vector(row["previous_executed_joint_target_rad"], 12, "previous target")
    delta = [f32(x - p) for x, p in zip(target, prev_target)]
    exact(delta, row["joint_target_delta_rad"], label + ".target_delta")
    real = row["real_policy_observation"]
    exact([f32(q - d) for q, d in zip(row["before"]["joint_positions_rad"], defaults)],
          real[12:24], label + ".real_joint_pos")
    exact(row["before"]["joint_velocities_rad_s"], real[24:36], label + ".real_joint_vel")
    exact([0.0, 0.0, 0.0], row["velocity_command_b"], label + ".zero_command")
    if previous is not None:
        exact(previous["after"], row["before"], label + ".physical_continuity")
        exact(previous["actual_raw_action"], row["previous_raw_action"], label + ".history1_continuity")
        exact(previous["previous_raw_action"], row["previous_previous_raw_action"], label + ".history2_continuity")
        exact(previous["actual_executed_joint_target_rad"], prev_target, label + ".target_continuity")
        exact(previous["projected_gravity_b_after"], real[6:9], label + ".real_gravity_continuity")
    return {"action_jump": max(abs(f32(a-p)) for a, p in zip(action, row["previous_raw_action"])),
            "target_jump": max(abs(x) for x in delta),
            "torque": row["after"]["applied_torque_joint_abs_max_at_control_boundary_nm"]}


def prefix_exact(on, off, first_refined_step):
    """Full row equality before the first new action; pre-action state at edge."""
    step = on["step"]
    if first_refined_step is None or step < first_refined_step:
        exact(off, on, "ON_OFF_before_refinement")
        exact(on, off, "ON_OFF_before_refinement_reverse")
        return "full_interval"
    if step == first_refined_step:
        for key in ("before", "policy_observation", "real_policy_observation", "roll_policy_input_observation",
                    "roll_actor_raw_action", "stand_actor_raw_action", "refinement_actor_raw_action",
                    "original_recovery_raw_action", "previous_raw_action", "previous_previous_raw_action",
                    "previous_executed_joint_target_rad", "velocity_command_b", "refinement_gate_count_before",
                    "startup_selected", "stand_active", "switched", "gate_steps", "roll_mirror_selected"):
            exact(off[key], on[key], "ON_OFF_edge_pre_action." + key)
        return "edge_pre_action"
    return None


def stream_intervals(path, pose, n=20, steps=550):
    """Read one interval at a time; reject incomplete/reordered/extra evidence."""
    with path.open(encoding="utf-8") as stream:
        count = 0
        for step, line in enumerate(stream):
            require(step < steps, "Extra full-trace interval")
            value = json.loads(line, object_pairs_hook=mirror.duplicate_reject)
            mirror.finite(value)
            require(value["pose"] == pose and type(value["step"]) is int and value["step"] == step,
                    "Wrong pose/missing/reordered interval")
            rows = value["rows"]
            require(type(rows) is list and len(rows) == n, "Missing all-trial interval")
            for trial, row in enumerate(rows):
                require(type(row["trial"]) is int and row["trial"] == trial
                        and type(row["step"]) is int and row["step"] == step, "Wrong trial/step identity")
            count += 1
            yield rows
        require(count == steps, "Truncated full-trace evidence")


def compact_distribution(values):
    require(values and all(math.isfinite(v) for v in values), "Missing finite measurements")
    ordered = sorted(values)
    pos = .9 * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": ordered[lo] + (ordered[hi] - ordered[lo]) * (pos-lo), "max": max(values)}


def foot_metrics(feet):
    out = {}
    for axle, i in (("front", 0), ("hind", 2)):
        left, right = feet[i], feet[i+1]
        for axis, delta in zip("xyz", (left[0]-right[0], left[1]+right[1], left[2]-right[2])):
            out[f"{axle}_paired_{axis}_abs_m"] = abs(delta)
        out[f"{axle}_lateral_bias_m"] = abs(left[1] + right[1])
        out[f"{axle}_width_m"] = left[1] - right[1]
    out["paired_lateral_bias_m"] = (out["front_lateral_bias_m"] + out["hind_lateral_bias_m"]) / 2
    return out


def load_baseline(pose):
    summary_path = natural.BASE / "summary.json"
    require(sha(summary_path) == natural.BASE_SUMMARY_SHA, "Frozen baseline summary changed")
    rows = [r for r in mirror.read_json(summary_path)["rows"] if r["startup"] == "supported"
            and r["mirror"] == "initial_right" and r["pose"] == pose]
    require(len(rows) == 1, "No unique original combined baseline")
    path = Path(rows[0]["report"])
    require(sha(path) == rows[0]["report_sha256"].lower(), "Original baseline report changed")
    report = mirror.read_json(path)
    trace, digest = mirror.load_trace(path, report)
    combined.audit_interfaces(report, trace, pose)
    return path, report, trace, digest


def load_verified(path):
    report = mirror.read_json(path)
    require(report["protocol_version"] == entry.PROTOCOL and report["controller_type"] == entry.CONTROLLER,
            "Wrong supported-refinement protocol/controller")
    require(len(report["results"]) == 1 and report["seed"] == 20260918, "One fixed development pose required")
    pose = next(iter(report["results"]))
    require(report["results"][pose]["trials"] == 20, "Full all20 audit only")
    require(all(report[k] is False for k in ("acceptance_eligible", "single_policy_acceptance_eligible", "training_collection_eligible")), "False acceptance flag")
    for name, digest in PINS.items():
        require(sha(ROOT / "scripts" / name) == digest, "Frozen audit dependency changed: " + name)
    mirror.verify_frozen_source()
    exp = report["refinement_experiment"]
    require(exp["mode"] in ("off", "supported"), "Unknown refinement mode")
    for key, expected in (("num_envs", 20), ("qualifying_completed_intervals", 150), ("step_dt", .02),
                          ("decision_applies_to_next_action", True), ("training_performed_in_this_run", False),
                          ("switch_time_physical_or_history_reset", False), ("extra_pd_ramp_or_retry", False),
                          ("promotion_performed", False), ("full_measured_rows", 11000),
                          ("entry_source_sha256", ENTRY_SHA), ("gate_source_sha256", entry.GATE_SHA)):
        exact(expected, exp[key], "refinement." + key)
    identity = exp["refinement_actor_training"]
    require(identity["checkpoint_sha256"] == REFINEMENT_SHA and identity["training_receipt_sha256"] == RECEIPT_SHA,
            "Not the predeclared final3746 trained actor")
    verified = entry.trained.candidate_receipt(entry.RUN / "training_result.json", entry.RUN / "model_3746.pt")
    exact(verified, identity, "actual_training_receipt")
    exact(identity, verified, "complete_training_receipt")
    ctl = report["handoff_controller"]
    for name, digest in (("roll", ROLL_SHA), ("stand", STAND_SHA)):
        require(ctl[name + "_sha256"] == sha(ctl[name + "_checkpoint"]) == digest, "Frozen old actor changed")
    require(report["checkpoint_sha256"] == sha(report["checkpoint"]) == ROLL_SHA, "Wrong primary checkpoint")
    generated = sibling(path, report["startup_experiment"]["generated_source_file"])
    expected_sha = hashlib.sha256(entry.build_source().encode()).hexdigest()
    require(sha(generated) == expected_sha == report["startup_experiment"]["generated_source_sha256"], "Generated source differs")
    require(report["startup_experiment"]["mode"] == "supported" and report["mirror_experiment"]["mode"] == "initial_right",
            "Predeclared original recovery routing changed")
    require(report["startup_experiment"]["selected_actor_retained"] is (exp["mode"] == "off"), "Wrong actor-retention label")
    require(report["combined_experiment"]["postrecovery_refinement_mode"] == exp["mode"]
            and report["combined_experiment"]["stand_active_is_original_recovery_branch_not_final_actor"] is True,
            "Wrong branch/final-actor semantics")
    require(report["transition_experiment"]["hard_zero_semantics"] == "where(refinement_active, refinement_action, original_recovery_action)"
            and report["transition_experiment"]["original_recovery_action_semantics"] ==
            "torch.where((startup_selected | switched)[:,None], stand_action, physical_roll_action)", "Action equation changed")
    trace, digest = mirror.load_trace(path, report)
    require(trace["schema"] == "handoff_combined_neighborhood_v1", "Wrong old-neighborhood schema")
    for group, hash_key in (("transition_experiment", "transition_trace_sha256"), ("startup_experiment", "transition_trace_sha256"),
                            ("mirror_experiment", "mirror_trace_sha256"), ("combined_experiment", "trace_sha256")):
        require(report[group][hash_key] == digest, "Conflicting neighborhood hashes")
        expected = {k:v for k,v in report[group].items() if k != hash_key}
        exact(expected, trace[group], "report_trace." + group)
        exact(trace[group], expected, "trace_report." + group)
    full_path = sibling(path, exp["trace_file"])
    require(full_path.name == "supported_refinement_full_trace.jsonl" and sha(full_path) == exp["trace_sha256"], "Full-trace SHA mismatch")
    invocation = mirror.read_json(path.parent / "invocation.json")
    for key, expected in (("schema", "supported_refinement_invocation_v1"), ("pose", pose), ("trials",20), ("seed",20260918),
                          ("mode", "supported"), ("mirror_mode", "initial_right"), ("refinement_mode", exp["mode"]), ("ramp_seconds",0)):
        exact(expected, invocation[key], "invocation." + key)
    seen = set()
    for item in invocation["source_and_inputs"]:
        source = Path(item["path"]).resolve()
        require(source not in seen and source.is_relative_to(ROOT.parent.resolve()), "Duplicate/escaped input snapshot")
        seen.add(source)
        require(sha(source) == item["sha256"], "Invocation source/input changed: " + str(source))
    for needed in (Path(entry.__file__), entry.GATE_PATH, Path(identity["checkpoint"]),
                   Path(identity["training_receipt"]), Path(ctl["roll_checkpoint"]), Path(ctl["stand_checkpoint"])):
        require(needed.resolve() in seen, "Missing critical source/model snapshot")
    return report, pose, trace, full_path, {"invocation_sha256": sha(path.parent / "invocation.json"),
        "verified_input_files": len(seen), "generated_source_sha256": expected_sha,
        "neighborhood_sha256": digest, "full_trace_sha256": sha(full_path), "training": verified}


def baseline_protocol(old, new, pose):
    for key in ("task", "seed", "criterion", "self_collisions_enabled", "settle_requested_s", "settle_actual_s",
                "settle_control_steps", "policy_action_mode", "checkpoint_sha256", "start_protocol_id", "joint_names",
                "handoff_controller"):
        exact(old[key], new[key], "baseline_protocol." + key)
    exact(new["handoff_controller"], old["handoff_controller"], "baseline_controller_reverse")
    require(("state_bank" in old) == ("state_bank" in new) == (pose != "upright"),
            "Upright must use ordinary reset; fallen cases require matching explicit state banks")
    if pose != "upright":
        exact(old["state_bank"], new["state_bank"], "baseline_protocol.state_bank")
    for key in ("mode", "ramp_seconds", "step_dt", "horizon_s", "hold_s", "min_contacts", "policy_control_steps"):
        exact(old["transition_experiment"][key], new["transition_experiment"][key], "transition_protocol." + key)
    for key in ("release_state_before_settling", "policy_start_state", "trials", "horizon_s", "stable_hold_s"):
        exact(old["results"][pose][key], new["results"][pose][key], "actual_start." + key)


def off_exact(old_path, old, old_trace, path, new, trace):
    checked = [0]
    for key, value in old.items():
        if key in ("protocol_version", "controller_type", "success_count_definition"):
            continue
        if key in GROUP_EXCLUSIONS:
            value = {k:v for k,v in value.items() if k not in GROUP_EXCLUSIONS[key]}
        exact(value, new[key], "OFF_original_common." + key, checked)
    old_csv, new_csv = sibling(old_path, old["trace_csv"]), sibling(path, new["trace_csv"])
    require(old_csv.read_bytes() == new_csv.read_bytes(), "OFF trial0 CSV not byte-for-byte original")
    exact(old_trace["poses"], trace["poses"], "OFF_original_old_neighborhood_subset", checked)
    exact(old_trace["first_action_by_pose"], trace["first_action_by_pose"], "OFF_original_first_actions_subset", checked)
    return {"exact_common_scalars": checked[0], "trial0_csv_byte_exact": True, "trial0_csv_sha256": sha(new_csv),
            "old_neighborhood_subset_exact": True, "results_all20_exact": True}


def audit_rows(report, pose, trace, full_path, off_path=None):
    result, exp = report["results"][pose], report["refinement_experiment"]
    info = exp["poses"][pose]
    require(info["intervals"] == 550 and set(exp["poses"]) == {pose}, "Wrong interval/pose budget")
    defaults = vector(report["handoff_controller"]["joint_limits"]["default_joint_positions_rad"],12,"defaults")
    limits = report["handoff_controller"]["joint_limits"]["soft_joint_limits_rad"]
    require(len(limits) == 12 and all(len(x) == 2 for x in limits), "Wrong target limits")
    startup, records = combined.supported_startup_mask(result["policy_start_state"])
    exact(result["startup_selection"]["selection_records"], trace["startup_selection_by_pose"][pose], "startup_trace")
    mirrored, stored = [], []
    require(len(trace["poses"][pose]) == len(trace["first_action_by_pose"][pose]) == 20, "Missing old neighborhood/first actions")
    for i, (chosen, record) in enumerate(zip(startup, records)):
        selection = result["startup_selection"]["selection_records"][i]
        require(selection["trial"] == i and selection["selected"] is chosen and selection["valid"] is True, "Wrong startup routing")
        exact(record["inputs"], selection["inputs"], "startup_inputs")
        exact(record["conditions"], selection["conditions"], "startup_conditions")
        mirrored.append(mirror.audit_selection(result["policy_start_state"][i], result["mirror_diagnostic"]["selection_records"][i], "initial_right", i)[0])
        require(not (chosen and mirrored[-1]), "Overlapping startup/mirror selection")
        trial = trace["poses"][pose][i]
        switch = trial["switch_step"]
        expected = list(range(500,550)) if switch is None else list(range(max(0,switch-50), min(550,switch+51)))
        require(trial["trial"] == i and [r["step"] for r in trial["rows"]] == expected, "Missing/reordered old neighborhood")
        stored.append({r["step"]:r for r in trial["rows"]})
        combined.audit_first_action(trace["first_action_by_pose"][pose][i], trial["rows"], result["policy_start_state"][i])
    previous, count, latch, hold, stable, first, recovered = ([None]*20, [0]*20, [False]*20, [0]*20, [0]*20, [None]*20, [None]*20)
    final_geom, final_measured = [False]*20, [None]*20
    samples = {"action_jump":[], "target_jump":[], "torque":[]}
    per_trial = [{"action_jump":[], "target_jump":[], "torque":[], "feet":[], "active_intervals":0,
                  "last150_strict_intervals":0, "last150_refinement_intervals":0} for _ in range(20)]
    prefix_counts = {"full_interval":0, "edge_pre_action":0}
    off_iter = stream_intervals(off_path, pose) if off_path is not None else None
    for rows in stream_intervals(full_path, pose):
        other = next(off_iter) if off_iter is not None else None
        for i, row in enumerate(rows):
            step = row["step"]
            require(row["policy_time_s"] == step*.02, "Wrong physical time")
            if step in stored[i]:
                exact(stored[i][step], row, "stored_neighborhood_vs_full")
            if step == 0:
                exact(trace["first_action_by_pose"][pose][i], row, "first_action_vs_full")
                exact(result["policy_start_state"][i]["projected_gravity_b"], row["real_policy_observation"][6:9], "first_real_gravity")
            switch = trace["poses"][pose][i]["switch_step"]
            old_switched = switch is not None and step >= switch
            old_active = startup[i] or old_switched
            require(row["startup_selected"] is startup[i] and row["switched"] is old_switched
                    and row["just_switched"] is (switch == step) and row["stand_active"] is old_active, "Original routing changed")
            if startup[i]:
                require(row["gate_steps"] == 0 and switch is None, "Startup fabricated handoff")
            active = latch[i] and exp["mode"] == "supported"
            require(row["refinement_active"] is active and (not active or old_active), "Refinement before qualifying completed hold")
            require(type(row["refinement_gate_count_before"]) is int and row["refinement_gate_count_before"] == count[i], "Gate before mismatch")
            phase = "postrecovery_refinement" if active else "startup_stand" if startup[i] else "stand" if old_switched else "roll"
            require(row["phase"] == phase and row["alpha"] == (1.0 if old_active else 0.0), "Wrong final actor phase/old hard switch")
            if active and first[i] is None:
                first[i] = step
            if other is not None:
                kind = prefix_exact(row, other[i], first[i])
                if kind:
                    prefix_counts[kind] += 1
            sample = issued_and_history(row, previous[i], defaults, limits, mirrored[i])
            measured = strict_standing(row, defaults)
            valid = measured["valid"]
            require(row["strict_valid_after"] is valid, "Strict-standing flag differs from measured geometry/dynamics")
            count[i], latch[i] = gate_after(valid, old_active, count[i], latch[i])
            hold[i] = hold[i]+1 if active and valid else 0
            stable[i] = stable[i]+1 if valid else 0
            if recovered[i] is None and stable[i] >= 150:
                recovered[i] = f32((step+1-150)*.02)
            require(type(row["refinement_gate_count_after"]) is int and row["refinement_gate_count_after"] == count[i]
                    and row["refinement_latched_for_next_action"] is latch[i]
                    and type(row["refinement_valid_hold_steps_after"]) is int and row["refinement_valid_hold_steps_after"] == hold[i],
                    "After-step gate/latch/active-only hold mismatch")
            final_geom[i], final_measured[i] = measured["geometry"], measured
            for key, value in sample.items():
                samples[key].append(value)
                per_trial[i][key].append(value)
            per_trial[i]["active_intervals"] += active
            if step >= 400:
                per_trial[i]["feet"].append(foot_metrics(row["feet_b_after_m"]))
                per_trial[i]["last150_strict_intervals"] += valid
                per_trial[i]["last150_refinement_intervals"] += active
            previous[i] = row
    if off_iter is not None:
        require(next(off_iter, None) is None, "OFF stream contains extra intervals")
    exact(first, info["first_refinement_action_step"], "first_refinement_actions")
    exact(hold, info["final_refinement_valid_hold_steps"], "ending_refinement_holds")
    require(info["final_refinement_valid_holds"] == sum(x >= 150 for x in hold), "Refinement hold total mismatch")
    require(result["successes"] == sum(x is not None for x in recovered)
            and result["final_valid_stands"] == sum(x >= 150 for x in stable)
            and result["final_geometry_passes"] == sum(final_geom), "Actual full-trace success/final counts differ")
    successes = [x for x in recovered if x is not None]
    if successes:
        require(result["median_recovery_s"] == statistics.median(successes), "Median recovery time differs")
    output_trials = []
    final3 = {}
    for i, data in enumerate(per_trial):
        diagnostic = result["final_diagnostics"][i]
        require(diagnostic["trial"] == i and diagnostic["geometry_ok"] is final_geom[i]
                and diagnostic["stable_hold_s"] == stable[i]*.02
                and diagnostic["vertical_foot_contacts"] == final_measured[i]["contacts"], "Final diagnostic mismatch")
        exact(diagnostic["feet_y_b"], [x[1] for x in previous[i]["feet_b_after_m"]], "final_feet")
        exact(diagnostic["knees_y_b"], [x[1] for x in previous[i]["knees_b_after_m"]], "final_knees")
        require(diagnostic["height_m"] == previous[i]["after"]["height_m"]
                and diagnostic["max_joint_offset_rad"] == final_measured[i]["max_joint_offset_rad"], "Final height/joints differ")
        require(abs(diagnostic["gravity_error"]-final_measured[i]["gravity_error"]) < 2e-7, "Final gravity diagnostic differs")
        terminal = {key:compact_distribution([r[key] for r in data["feet"]]) for key in data["feet"][0]}
        for key in terminal:
            final3.setdefault(key, []).extend(r[key] for r in data["feet"])
        output_trials.append({"trial":i, "bank_state_id":diagnostic.get("bank_state_id"),
            "first_refinement_action_step":first[i], "recovery_onset_s":recovered[i],
            "final_strict_hold_steps":stable[i], "final_refinement_controlled_valid_hold_steps":hold[i],
            "active_refinement_intervals":data["active_intervals"],
            "last150_strict_intervals":data["last150_strict_intervals"],
            "last150_refinement_intervals":data["last150_refinement_intervals"],
            "last3s_foot_geometry":terminal,
            "all550_action_and_control_boundary_torque":{k:compact_distribution(data[k]) for k in samples}})
    return {"physical_intervals":550, "measured_rows":11000, "all_trial_ids":list(range(20)),
        "independent_final_strict_holds":sum(x >= 150 for x in stable),
        "independent_final_refinement_controlled_strict_holds":sum(x >= 150 for x in hold),
        "independent_recovery_successes":len(successes), "prefix_comparison":prefix_counts if off_path else None,
        "last3s_all3000_foot_geometry":{k:compact_distribution(v) for k,v in final3.items()},
        "all11000_action_and_control_boundary_torque":{k:compact_distribution(v) for k,v in samples.items()},
        "trials":output_trials}


def audit(path, off=None):
    auditor_sha = sha(Path(__file__))
    report, pose, trace, full_path, provenance = load_verified(path)
    old_path, old, old_trace, old_trace_sha = load_baseline(pose)
    baseline_protocol(old, report, pose)
    mode = report["refinement_experiment"]["mode"]
    require((mode == "supported") == (off is not None), "ON requires an OFF report; OFF takes no OFF argument")
    reference, equivalence, off_full = None, None, None
    if mode == "off":
        equivalence = off_exact(old_path, old, old_trace, path, report, trace)
    else:
        require(off.resolve() != path.resolve(), "ON cannot use itself as OFF")
        reference = audit(off)
        require(reference["mode"] == "off" and reference["pose"] == pose, "Wrong OFF control")
        off_report = mirror.read_json(off)
        off_full = sibling(off, off_report["refinement_experiment"]["trace_file"])
        baseline_protocol(off_report, report, pose)
    measured = audit_rows(report, pose, trace, full_path, off_full)
    comparison = None
    if reference:
        keys = ("front_paired_x_abs_m", "front_lateral_bias_m", "front_paired_z_abs_m",
                "hind_paired_x_abs_m", "hind_lateral_bias_m", "hind_paired_z_abs_m", "paired_lateral_bias_m")
        comparison = {k:{"off_mean":reference["measured"]["last3s_all3000_foot_geometry"][k]["mean"],
                         "on_mean":measured["last3s_all3000_foot_geometry"][k]["mean"]} for k in keys}
        for row in comparison.values():
            row["relative_mean_reduction"] = 1-row["on_mean"]/row["off_mean"] if row["off_mean"] else None
    require(sha(Path(__file__)) == auditor_sha, "Auditor source changed while auditing; rerun frozen evidence")
    return {"protocol":PROTOCOL, "audit_passed":True, "mode":mode, "pose":pose,
        "report":str(path), "report_sha256":sha(path), "auditor_source_sha256":auditor_sha,
        "provenance":provenance, "original_baseline":{"report":str(old_path),"report_sha256":sha(old_path),
            "summary_sha256":natural.BASE_SUMMARY_SHA,"neighborhood_sha256":old_trace_sha},
        "off_original_exact_equivalence":equivalence,
        "off_control":None if reference is None else {"report":str(off),"report_sha256":sha(off),
            "audited_again_in_this_process":True,"full_trace_sha256":reference["provenance"]["full_trace_sha256"]},
        "measured":measured, "last3s_on_off_measured_geometry":comparison,
        "promotion_performed":False,"acceptance_eligible":False,
        "legacy_event_reason_notice":"Old neighborhood logger may label postrecovery refinement as 'raw ramp completed'. This is legacy descriptive text, not evidence of a ramp; actual full-trace masks, issued actions and gate chronology are audited.",
        "limits":"Repeated development states, zero command, three actors. Audit success means evidence consistency, not normality/promotion. Final3s metrics include every sample and failure, not just valid intervals. Torque is at50Hz control boundaries, not peak physics-substep torque. No walk/trot/run/full-flow or physical-asset symmetry claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--off",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    require(args.output.resolve().drive.upper() == "E:", "Keep new audit evidence on E:")
    require(not args.output.exists(), "Preserve existing audit evidence")
    try:
        result = audit(args.report, args.off)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, StopIteration) as error:
        result = {"protocol":PROTOCOL,"audit_passed":False,"error":str(error),"promotion_performed":False,"acceptance_eligible":False}
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ("measured","provenance")},indent=2,allow_nan=False))
    raise SystemExit(0 if result["audit_passed"] else 1)
