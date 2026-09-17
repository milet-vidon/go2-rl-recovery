"""Finite, isolated composition of supported startup and roll-only mirroring.

No simulator imports occur on import/build_source. A SHA-pinned mirror evaluator
is transformed in memory with unique anchors; no frozen file is changed. Both
flags are explicit. This combines two frozen inference mechanisms, not training,
single-policy acceptance, full locomotion integration or physical symmetry proof.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/evaluate_handoff_mirror.py"
TEMPLATE_SHA = "cf2fbbdceae0e460727a969bf080e1e230ed716f00f914f47381a167df50ba95"
STARTUP_REFERENCE = ROOT / "scripts/evaluate_handoff_supported_startup.py"
STARTUP_REFERENCE_SHA = "30e63d6f60261d3e163c212568613d0f8941da9ce2d8b50676ad6fc281800c60"
STARTUP_MATH = ROOT / "src/go2_recovery/supported_startup_math.py"
STARTUP_MATH_SHA = "47991846f57272192b10171cf43eed5e772fdbc063853cc0601a82110af6fe93"
MIRROR_MATH = ROOT / "src/go2_recovery/roll_mirror_math.py"
MIRROR_MATH_SHA = "9e3edf7cd402c745242fa41e8dcf3d4ccb3818cb474344cff52c104c00426d91"
PROTOCOL = "handoff_combined_startup_mirror_experimental_v1"
CONTROLLER = "experimental_supported_startup_and_roll_mirror_hard_handoff"
MATH_PATH, MATH_SHA = STARTUP_MATH, STARTUP_MATH_SHA
MIRROR_MATH_PATH = MIRROR_MATH
ROLL_SHA = "71e4c7a39ff81836d2ead1c4533988c696e5bd4a0fc939b8cef6b4f6daec1b1c"
STAND_SHA = "5527c4053cb6ef8e6616da65b7dc32a9464eb8d1366f4966a4eb4b65bc913fdb"


def _replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one frozen anchor, found {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def invalid_diagnostic_json(value):
    """Serialize invalid evidence explicitly; never create a usable fallback mask."""
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_number": repr(value)}
    if type(value) is dict:
        return {key: invalid_diagnostic_json(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [invalid_diagnostic_json(item) for item in value]
    return value


def build_source():
    for path, expected in ((TEMPLATE, TEMPLATE_SHA), (STARTUP_REFERENCE, STARTUP_REFERENCE_SHA),
                           (STARTUP_MATH, STARTUP_MATH_SHA), (MIRROR_MATH, MIRROR_MATH_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen composition source SHA mismatch: {path}")
    source = TEMPLATE.read_text(encoding="utf-8")
    edits = [
        ('''"""EXPERIMENTAL roll-only initial-right mirror diagnostic; never collection or acceptance.

Standalone snapshot of the frozen hard transition evaluator. Gate, resets,
physics, real observations, native action history and standing actor are unchanged.
Only a deep-cloned roll actor input and its output coordinates may be reflected.
There is no ramp, retry, startup selector, symmetry training or dynamics guarantee.
"""''',
         '''"""EXPERIMENTAL supported startup plus roll-only initial-right mirroring.

Two explicit inference factors, frozen actors, unchanged physics and native
history. Supported nonfallen starts may choose stand from the first action;
otherwise the original one-way hard gate applies. Only the roll input COPY
and its candidate output may be reflected. Never training or acceptance;
no ramp, retry, single-policy claim, symmetry training or dynamics guarantee.
"""'''),
        ('parser.add_argument("--mirror_mode", required=True, choices=("off", "initial_right"))',
         '''parser.add_argument("--startup_mode", required=True, choices=("off", "supported"))
parser.add_argument("--mirror_mode", required=True, choices=("off", "initial_right"))'''),
        ('        from recovery_handoff_math import update_handoff_gate',
         '''        from recovery_handoff_math import update_handoff_gate
        from supported_startup_math import supported_startup_mask, SupportedStartupInputError'''),
        ('    mirror_selection_by_pose = {}',
         '''    mirror_selection_by_pose = {}
    startup_selection_by_pose = {}
    first_action_by_pose = {}'''),
        ('            start_gravity = env.unwrapped.scene["robot"].data.projected_gravity_b.clone()',
         '''            # Real policy-start boundary only, after unchanged nominal PD.
            # Validate every row even in OFF mode; invalid is never a false-mask fallback.
            try:
                startup_predicate, startup_records = supported_startup_mask(policy_start_state)
            except SupportedStartupInputError as exc:
                _write_json_new(args_cli.output_dir / "combined_invalid_startup_inputs.json", {
                    "protocol_version": COMBINED_PROTOCOL, "pose": pose_class, "valid": False,
                    "invalid_indices": list(exc.invalid_indices),
                    "records": COMBINED_INVALID_JSON(exc.records),
                    "no_action_was_issued": True, "acceptance_eligible": False})
                raise
            startup_selected = torch.tensor(
                [value and args_cli.startup_mode == "supported" for value in startup_predicate],
                device=env.device, dtype=torch.bool)
            for i, record in enumerate(startup_records):
                record.update(trial=i, predicate_selected_standing=record["selected_standing"],
                              enabled=args_cli.startup_mode == "supported",
                              selected=bool(startup_selected[i]),
                              actual_initial_actor="stand" if bool(startup_selected[i]) else "roll",
                              mask_latched_for_episode=True, startup_is_not_handoff=True)
            startup_selection_by_pose[pose_class] = startup_records
            first_action_by_pose[pose_class] = []
            start_gravity = env.unwrapped.scene["robot"].data.projected_gravity_b.clone()'''),
        ('            mirror_selected = initial_right_mask(start_gravity, eligible_fallen, args_cli.mirror_mode)',
         '''            mirror_selected = initial_right_mask(start_gravity, eligible_fallen, args_cli.mirror_mode)
            # S requires eligible=False, M requires eligible=True. A collision is
            # invalid evidence, not permission to silently prioritize one branch.
            if bool((startup_selected & mirror_selected).any()):
                raise RuntimeError("Supported startup and settled-fallen mirror masks overlap")
            if bool((startup_selected & eligible_fallen).any()):
                raise RuntimeError("Startup standing cannot be a settled-fallen recovery start")'''),
        ('                neighborhood = TransitionNeighborhood(env.num_envs, neighborhood_steps, neighborhood_steps)',
         '''                neighborhood = TransitionNeighborhood(env.num_envs, neighborhood_steps, neighborhood_steps)
                # Initialize only the logger category, never real policy history.
                for i in startup_selected.nonzero(as_tuple=False).flatten().cpu().tolist():
                    neighborhood.phases[i] = "startup_stand"'''),
        ('''                            gate_steps, switched, just_switched = update_handoff_gate(
                                a.projected_gravity_b, a.root_ang_vel_b, gate_steps, switched, env.unwrapped.step_dt)''',
         '''                            next_gate, next_switched, next_edge = update_handoff_gate(
                                a.projected_gravity_b, a.root_ang_vel_b, gate_steps, switched, env.unwrapped.step_dt)
                            gate_steps = torch.where(startup_selected, gate_steps, next_gate)
                            switched = torch.where(startup_selected, switched, next_switched)
                            just_switched = torch.where(startup_selected, just_switched, next_edge)
                        if bool((startup_selected & (switched | just_switched | (gate_steps != 0))).any()):
                            raise RuntimeError("Startup standing was falsely counted as a gate event")
                        stand_active = startup_selected | switched'''),
        ('''                            roll_action, stand_action, previous_raw_action, switched, just_switched,
                            ramp_anchor, ramp_progress, ramp_steps)''',
         '''                            roll_action, stand_action, previous_raw_action, stand_active, just_switched,
                            ramp_anchor, ramp_progress, ramp_steps)'''),
        ('''                        obs, _, dones, _ = env.step(action)
                        # Read-only assertions every step''',
         '''                        if step == 0 and len(transition_rows) != env.num_envs:
                            raise RuntimeError("Every trial requires a real first-action record")
                        for i, row in transition_rows.items():
                            row.update(startup_selected=bool(startup_selected[i]),
                                       stand_active=bool(stand_active[i]))
                            if bool(startup_selected[i]):
                                row["phase"] = "startup_stand"
                        obs, _, dones, _ = env.step(action)
                        # Read-only assertions every step'''),
        ('                                neighborhood.record(i, step, row, row["just_switched"])',
         '''                                neighborhood.record(i, step, row, row["just_switched"])
                                if step == 0:
                                    # Persist outside the untriggered-tail ring buffer.
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
                raise RuntimeError("Incomplete all-trial first-action evidence")
            if motion is not None:
                results[pose_class]["stochastic_motion_diagnostic"]'''),
        ('f"{checkpoint_name}_handoff_mirror_trace.json"',
         'f"{checkpoint_name}_handoff_combined_trace.json"'),
        ('    report["protocol_version"] = "handoff_mirror_experimental_v1"',
         '''    report["baseline_mirror_protocol_version"] = "handoff_mirror_experimental_v1"
    report["baseline_startup_protocol_version"] = "handoff_supported_startup_experimental_v1"
    report["protocol_version"] = COMBINED_PROTOCOL'''),
        ('    report["controller_type"] = "experimental_roll_only_initial_side_mirror_hard_handoff"',
         '    report["controller_type"] = COMBINED_CONTROLLER'),
        ('"ramp_enabled": False, "retry_enabled": False, "startup_selection_changed": False,',
         '"ramp_enabled": False, "retry_enabled": False, "startup_selection_changed": args_cli.startup_mode == "supported",'),
        ('    report["mirror_experiment"] = mirror_experiment',
         '''    report["mirror_experiment"] = mirror_experiment
    # Do not misdescribe S|switched as the historical switched-only equation.
    experiment["baseline_hard_zero_semantics"] = experiment["hard_zero_semantics"]
    experiment["hard_zero_semantics"] = "torch.where((startup_selected | switched)[:,None], stand_action, physical_roll_action)"
    startup_experiment = {
        "mode": args_cli.startup_mode,
        "predicate_protocol": "supported_nonfallen_startup_predicate_v1",
        "decision_boundary": "once at actual policy start after unchanged 1s nominal-position PD",
        "fixed_mask_per_episode": True, "selected_actor_retained": True,
        "selected_counts_as_handoff": False, "selected_counts_as_success": False,
        "unselected_original_hard_gate": True,
        "mirror_enabled": args_cli.mirror_mode == "initial_right",
        "ramp_enabled": False, "retry_enabled": False, "training_performed": False,
        "physical_state_or_history_mutated": False,
        "stand_actor_initialization_rng_isolated": True,
        "all_trial_first_actions_recorded": True,
        "startup_reference_adapter_sha256": COMBINED_STARTUP_REFERENCE_SHA,
        "math_source_sha256": COMBINED_STARTUP_MATH_SHA,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_source_sha256": COMBINED_GENERATED_SHA,
        "generated_source_file": "combined_generated.py",
        "transition_trace_file": transition_path.name,
        "first_action_trace_key": "first_action_by_pose",
        "ordinary_starts_are_not_fallen_recovery": True,
        "development_data_not_fresh_generalization": True,
    }
    combined_experiment = {
        "startup_mode": args_cli.startup_mode, "mirror_mode": args_cli.mirror_mode,
        "startup_mirror_overlap_trials": 0,
        "startup_mirror_mutual_exclusion_asserted": True,
        "startup_selection_uses_real_start_snapshot": True,
        "mirror_selection_uses_real_start_gravity_and_eligibility": True,
        "both_masks_latched_once": True,
        "roll_candidate_keeps_fixed_mirror_mask_after_handoff": True,
        "standing_actor_always_real_observation": True,
        "stand_active_definition": "startup_selected | genuine_gate_switched",
        "genuine_gate_switched_excludes_startup_selected": True,
        "first_actions_all_trials": True,
        "template_sha256": COMBINED_TEMPLATE_SHA,
        "startup_reference_adapter_sha256": COMBINED_STARTUP_REFERENCE_SHA,
        "startup_math_sha256": COMBINED_STARTUP_MATH_SHA,
        "mirror_math_sha256": COMBINED_MIRROR_MATH_SHA,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_source_sha256": COMBINED_GENERATED_SHA,
        "generated_source_file": "combined_generated.py",
        "trace_file": transition_path.name,
        "no_model_training_or_promotion": True,
    }
    report["startup_experiment"] = startup_experiment
    report["combined_experiment"] = combined_experiment
    with (args_cli.output_dir / "combined_generated.py").open("x", encoding="utf-8", newline="\\n") as handle:
        handle.write(COMBINED_GENERATED_SOURCE)
        handle.flush()
        os.fsync(handle.fileno())'''),
        ('        "schema": "handoff_mirror_neighborhood_v1",',
         '''        "schema": "handoff_combined_neighborhood_v1",
        "startup_experiment": startup_experiment,
        "combined_experiment": combined_experiment,
        "startup_selection_by_pose": startup_selection_by_pose,
        "first_action_by_pose": first_action_by_pose,'''),
        ('    mirror_experiment["mirror_trace_sha256"] = experiment["transition_trace_sha256"]',
         '''    mirror_experiment["mirror_trace_sha256"] = experiment["transition_trace_sha256"]
    startup_experiment["transition_trace_sha256"] = experiment["transition_trace_sha256"]
    combined_experiment["trace_sha256"] = experiment["transition_trace_sha256"]'''),
        ('        heading = f"EXPERIMENTAL ROLL MIRROR {args_cli.mirror_mode}"',
         '        heading = f"EXPERIMENTAL STARTUP {args_cli.startup_mode} / MIRROR {args_cli.mirror_mode}"'),
        ('            "protocol_version": "handoff_mirror_experimental_v1",',
         '            "protocol_version": COMBINED_PROTOCOL,'),
    ]
    for old, new in edits:
        source = _replace_once(source, old, new)
    return source


def main():
    source = build_source()
    namespace = {
        "__name__": "__main__", "__file__": str(Path(__file__).resolve()),
        "COMBINED_PROTOCOL": PROTOCOL, "COMBINED_CONTROLLER": CONTROLLER,
        "COMBINED_TEMPLATE_SHA": TEMPLATE_SHA,
        "COMBINED_STARTUP_REFERENCE_SHA": STARTUP_REFERENCE_SHA,
        "COMBINED_STARTUP_MATH_SHA": STARTUP_MATH_SHA,
        "COMBINED_MIRROR_MATH_SHA": MIRROR_MATH_SHA,
        "COMBINED_INVALID_JSON": invalid_diagnostic_json,
        "COMBINED_GENERATED_SOURCE": source,
        "COMBINED_GENERATED_SHA": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    exec(compile(source, str(Path(__file__).resolve()) + "::<generated-mirror-template>", "exec"), namespace)


if __name__ == "__main__":
    main()
