"""Isolated supported-startup diagnostic, assembled from one SHA-pinned hard entry.

The finite exact-anchor changes below are the entire intervention. No import of
the simulator occurs on module import or build_source(); CPU tests can inspect
the generated AST. The frozen evaluator is never edited. __file__ remains this
adapter, so reported evaluator SHA is honestly the adapter's SHA, not the template.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/evaluate_handoff_transition.py"
TEMPLATE_SHA = "c3a433568561fbe0893d61a570cf420304d44dcf986440e8d7efebd85ff7f10a"
MATH_PATH = ROOT / "src/go2_recovery/supported_startup_math.py"
MATH_SHA = "47991846f57272192b10171cf43eed5e772fdbc063853cc0601a82110af6fe93"
PROTOCOL = "handoff_supported_startup_experimental_v1"
CONTROLLER = "experimental_supported_startup_or_original_hard_handoff"
ROLL_SHA = "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c"
STAND_SHA = "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"


def _replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one frozen source anchor, found {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def invalid_diagnostic_json(value):
    """Make INVALID evidence serializable, never fabricate usable routing inputs."""
    import math
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_number": repr(value)}
    if type(value) is dict:
        return {key: invalid_diagnostic_json(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [invalid_diagnostic_json(item) for item in value]
    return value


def build_source():
    raw = TEMPLATE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Frozen hard-transition template SHA mismatch")
    if hashlib.sha256(MATH_PATH.read_bytes()).hexdigest() != MATH_SHA:
        raise ValueError("Reviewed startup predicate SHA mismatch")
    source = raw.decode("utf-8").replace("\r\n", "\n")
    edits = [
        ('    if args.transition_mode == "hard" and args.ramp_seconds != 0:',
         '''    if args.transition_mode != "hard" or args.ramp_seconds != 0:
        raise ValueError("Supported startup forbids ramp/mirror/retry/training")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != STARTUP_ROLL_SHA:
        raise ValueError("Frozen roll1999 checkpoint changed")
    if hashlib.sha256(args.stand_checkpoint.read_bytes()).hexdigest() != STARTUP_STAND_SHA:
        raise ValueError("Frozen stand3547 checkpoint changed")
    if args.transition_mode == "hard" and args.ramp_seconds != 0:'''),
        ('parser.add_argument("--transition_mode", required=True, choices=("hard", "raw_ramp"))',
         '''parser.add_argument("--startup_mode", required=True, choices=("off", "supported"))
parser.add_argument("--transition_mode", required=True, choices=("hard",))'''),
        ('        from recovery_handoff_math import update_handoff_gate',
         '''        from recovery_handoff_math import update_handoff_gate
        from supported_startup_math import supported_startup_mask, SupportedStartupInputError'''),
        ('    transition_trace_by_pose = {}',
         '''    transition_trace_by_pose = {}
    first_action_by_pose = {}
    startup_selection_by_pose = {}'''),
        ('                neighborhood = TransitionNeighborhood(env.num_envs, neighborhood_steps, neighborhood_steps)',
         '''                neighborhood = TransitionNeighborhood(env.num_envs, neighborhood_steps, neighborhood_steps)
                # Logger category only: no fabricated roll-to-stand/ramp event.
                # Never writes physical state, observations or actual action history.
                for i in startup_selected.nonzero(as_tuple=False).flatten().cpu().tolist():
                    neighborhood.phases[i] = "startup_stand"'''),
        ('            obs = env.get_observations()',
         '''            # Exactly one read-only decision at the actual policy-start boundary.
            # Invalid data aborts the WHOLE batch; it is never a false-mask fallback.
            try:
                predicate_mask, startup_records = supported_startup_mask(policy_start_state)
            except SupportedStartupInputError as exc:
                _write_json_new(args_cli.output_dir / "startup_invalid_inputs.json", {
                    "protocol": STARTUP_PROTOCOL, "pose": pose_class, "valid": False,
                    "invalid_indices": list(exc.invalid_indices), "records": STARTUP_INVALID_JSON(exc.records),
                    "acceptance_eligible": False, "no_action_was_issued": True})
                raise
            startup_selected = torch.tensor(
                [value and args_cli.startup_mode == "supported" for value in predicate_mask],
                device=env.device, dtype=torch.bool)
            for i, record in enumerate(startup_records):
                record.update(trial=i, predicate_selected_standing=record["selected_standing"],
                              enabled=args_cli.startup_mode == "supported",
                              selected=bool(startup_selected[i]),
                              actual_initial_actor="stand" if bool(startup_selected[i]) else "roll",
                              mask_latched_for_episode=True,
                              startup_is_not_handoff=True)
            startup_selection_by_pose[pose_class] = startup_records
            first_action_by_pose[pose_class] = []
            obs = env.get_observations()'''),
        ('''                            gate_steps, switched, just_switched = update_handoff_gate(
                                a.projected_gravity_b, a.root_ang_vel_b, gate_steps, switched, env.unwrapped.step_dt)''',
         '''                            next_gate, next_switched, next_edge = update_handoff_gate(
                                a.projected_gravity_b, a.root_ang_vel_b, gate_steps, switched, env.unwrapped.step_dt)
                            # A startup stand is NOT a gate event, even after 0.2 s.
                            # Unselected rows retain the original elementwise gate output.
                            gate_steps = torch.where(startup_selected, gate_steps, next_gate)
                            switched = torch.where(startup_selected, switched, next_switched)
                            just_switched = torch.where(startup_selected, just_switched, next_edge)
                        if bool((startup_selected & (switched | just_switched | (gate_steps != 0))).any()):
                            raise RuntimeError("Startup standing was falsely recorded as a handoff")
                        stand_active = startup_selected | switched'''),
        ('''                            roll_action, stand_action, previous_raw_action, switched, just_switched,
                            ramp_anchor, ramp_progress, ramp_steps)''',
         '''                            roll_action, stand_action, previous_raw_action, stand_active, just_switched,
                            ramp_anchor, ramp_progress, ramp_steps)'''),
        ('''                            roll_action, stand_action, action, ramp_anchor, ramp_alpha, target, neighborhood)
                        obs, _, dones, _ = env.step(action)''',
         '''                            roll_action, stand_action, action, ramp_anchor, ramp_alpha, target, neighborhood)
                        if step == 0 and len(transition_rows) != env.num_envs:
                            raise RuntimeError("First action must be recorded for every trial")
                        for i, row in transition_rows.items():
                            row.update(startup_selected=bool(startup_selected[i]),
                                       stand_active=bool(stand_active[i]))
                            if bool(startup_selected[i]):
                                row["phase"] = "startup_stand"
                        obs, _, dones, _ = env.step(action)'''),
        ('                                neighborhood.record(i, step, row, row["just_switched"])',
         '''                                neighborhood.record(i, step, row, row["just_switched"])
                                if step == 0:
                                    # Independent persistent first-action record, not ring retention.
                                    first_action_by_pose[pose_class].append(row)'''),
        ('''                                     policy_phase=("roll" if not bool(switched[0]) else
                                                   "raw_ramp" if float(ramp_alpha[0]) < 1 else "stand"))''',
         '''                                     policy_phase=("startup_stand" if bool(startup_selected[0]) else
                                                   "stand" if bool(switched[0]) else "roll"))'''),
        ('            if motion is not None:\n                results[pose_class]["stochastic_motion_diagnostic"]',
         '''            results[pose_class]["startup_selection"] = {
                "mode": args_cli.startup_mode, "selected_trials": int(startup_selected.sum()),
                "total_trials": env.num_envs, "all_trials_in_success_denominator": True,
                "selection_is_not_success_or_handoff": True, "selection_records": startup_records,
                "selected_final_valid_stands": int(((stable_steps >= hold_steps) & startup_selected).sum()),
                "unselected_final_valid_stands": int(((stable_steps >= hold_steps) & ~startup_selected).sum()),
                "selected_genuine_handoffs": int((switched & startup_selected).sum()),
            }
            if len(first_action_by_pose[pose_class]) != env.num_envs:
                raise RuntimeError("Incomplete all-trial first-action diagnostic")
            if motion is not None:
                results[pose_class]["stochastic_motion_diagnostic"]'''),
        ('    _write_json_new(transition_path, {',
         '''    startup_experiment = {
        "mode": args_cli.startup_mode,
        "predicate_protocol": "supported_nonfallen_startup_predicate_v1",
        "decision_boundary": "once at actual policy start after unchanged 1s nominal-position PD",
        "fixed_mask_per_episode": True, "selected_actor_retained": True,
        "selected_counts_as_handoff": False, "selected_counts_as_success": False,
        "unselected_original_hard_gate": True, "mirror_enabled": False,
        "ramp_enabled": False, "retry_enabled": False, "training_performed": False,
        "physical_state_or_history_mutated": False,
        "stand_actor_initialization_rng_isolated": True,
        "all_trial_first_actions_recorded": True,
        "template_sha256": STARTUP_TEMPLATE_SHA, "math_source_sha256": STARTUP_MATH_SHA,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_source_sha256": STARTUP_GENERATED_SHA,
        "generated_source_file": "supported_startup_generated.py",
        "transition_trace_file": transition_path.name,
        "first_action_trace_key": "first_action_by_pose",
        "ordinary_starts_are_not_fallen_recovery": True,
        "development_data_not_fresh_generalization": True,
    }
    report["baseline_transition_protocol_version"] = report["protocol_version"]
    report["protocol_version"] = STARTUP_PROTOCOL
    report["controller_type"] = STARTUP_CONTROLLER
    report["startup_experiment"] = startup_experiment
    with (args_cli.output_dir / "supported_startup_generated.py").open("x", encoding="utf-8", newline="\\n") as handle:
        handle.write(STARTUP_GENERATED_SOURCE)
        handle.flush()
        os.fsync(handle.fileno())
    _write_json_new(transition_path, {'''),
        ('        "schema": "handoff_transition_neighborhood_v1",',
         '''        "schema": "handoff_transition_neighborhood_v1",
        "startup_experiment": startup_experiment,
        "startup_selection_by_pose": startup_selection_by_pose,
        "first_action_by_pose": first_action_by_pose,'''),
        ('    experiment["transition_trace_sha256"] = hashlib.sha256(transition_path.read_bytes()).hexdigest()',
         '''    experiment["transition_trace_sha256"] = hashlib.sha256(transition_path.read_bytes()).hexdigest()
    startup_experiment["transition_trace_sha256"] = experiment["transition_trace_sha256"]'''),
        ('''        heading = (f"EXPERIMENTAL RAW RAMP T={args_cli.ramp_seconds:g}s"
                   if args_cli.transition_mode == "raw_ramp" else "EXPERIMENTAL HARD BASELINE")''',
         '''        heading = f"EXPERIMENTAL STARTUP {args_cli.startup_mode}"'''),
        ('            "protocol_version": "handoff_transition_experimental_v1",',
         '            "protocol_version": STARTUP_PROTOCOL,'),
    ]
    for old, new in edits:
        source = _replace_once(source, old, new)
    return source


def main():
    source = build_source()
    namespace = {
        "__name__": "__main__", "__file__": str(Path(__file__).resolve()),
        "STARTUP_TEMPLATE_SHA": TEMPLATE_SHA, "STARTUP_MATH_SHA": MATH_SHA,
        "STARTUP_ROLL_SHA": ROLL_SHA, "STARTUP_STAND_SHA": STAND_SHA,
        "STARTUP_PROTOCOL": PROTOCOL, "STARTUP_CONTROLLER": CONTROLLER,
        "STARTUP_INVALID_JSON": invalid_diagnostic_json,
        "STARTUP_GENERATED_SOURCE": source,
        "STARTUP_GENERATED_SHA": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    exec(compile(source, str(Path(__file__).resolve()) + "::<generated-hard-template>", "exec"), namespace)


if __name__ == "__main__":
    main()
