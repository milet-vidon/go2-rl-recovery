"""Preserve old recovery, then experimentally refine after a real3s strict hold.

Three actors, zero command, no switch-time reset/PD/ramp. This is NOT locomotion.
All control intervals/trials are streamed as measured JSONL, including failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import evaluate_handoff_combined as base
import evaluate_natural_handoff_candidate as trained

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "0788beb2e8d4457f6c286d9aff80638b269e080a9b75167f9f7977456380bed0"
TRAINED_SHA = "a97f8592fc61dc2fe2e07fcd2a669eb040b64219e2348602d6d7aa015f0d53ee"
GATE_SHA = "f1c99fca606f4f52b34be040604e818d8d4a2ded48321847d00c9d5ac7c048df"
GATE_PATH = ROOT / "scripts/supported_refinement_gate.py"
RUN = ROOT.parent / "repo/logs/rsl_rl/unitree_go2_handoff_stand/20260917-natural-stand128x200-first"
PROTOCOL = "supported_postrecovery_refinement_v1"
CONTROLLER = "experimental_original_recovery_then_trained_posture_refinement"


class RefinementRecorder:
    """Streaming read-only evidence; never holds a complete multi-env rollout in RAM."""
    def __init__(self, output, num_envs, mode, receipt):
        if type(num_envs) is not int or num_envs not in (1, 2, 20):
            raise ValueError("Only bounded1/2/20-trial evidence is supported")
        if mode not in ("off", "supported"):
            raise ValueError("Explicit off/supported refinement mode required")
        self.path = output / "supported_refinement_full_trace.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.num_envs, self.mode, self.receipt = num_envs, mode, receipt
        self.poses = {}
        self.rows_written = 0

    def record(self, pose, step, rows, strict_valid, foot_b, knee_b, gate_count, latch, refined_hold, gravity_b_after):
        import torch
        if self.stream.closed:
            raise RuntimeError("Cannot record after closing evidence")
        if type(step) is not int or not 0 <= step < 550:
            raise ValueError("Physical interval must lie within the frozen550-step rollout")
        if len(rows) != self.num_envs or sorted(rows) != list(range(self.num_envs)):
            raise ValueError("Every trial needs an actual interval record")
        for i, row in rows.items():
            if row.get("trial") != i or row.get("step") != step or type(row.get("refinement_active")) is not bool:
                raise ValueError("Recorded interval/trial identity differs from its actual row")
            if row.get("issued_action_target_history_assertions_passed") is not True:
                raise ValueError("Every streamed row needs completed live action/target/history checks")
            if self.mode == "off" and row["refinement_active"]:
                raise ValueError("OFF mode cannot execute refinement")
        if pose not in self.poses:
            self.poses[pose] = {"first_refinement_action_step": [None] * self.num_envs,
                                "intervals": 0, "final_refinement_valid_hold_steps": None}
        info = self.poses[pose]
        if step != info["intervals"]:
            raise ValueError("Missing or reordered physical interval")
        fields = {"strict_valid_after": strict_valid, "feet_b_after_m": foot_b, "knees_b_after_m": knee_b,
                  "refinement_gate_count_after": gate_count, "refinement_latched_for_next_action": latch,
                  "refinement_valid_hold_steps_after": refined_hold, "projected_gravity_b_after": gravity_b_after}
        for name, tensor in fields.items():
            if name in ("feet_b_after_m", "knees_b_after_m"):
                expected_shape = (self.num_envs, 4, 3)
            elif name == "projected_gravity_b_after":
                expected_shape = (self.num_envs, 3)
            else:
                expected_shape = (self.num_envs,)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Missing/nonfinite full-trial evidence: " + name)
        if strict_valid.dtype != torch.bool or latch.dtype != torch.bool or gate_count.dtype != torch.int64 or refined_hold.dtype != torch.int64:
            raise ValueError("Invalid gate/strict/hold evidence types")
        if bool(((gate_count < 0) | (gate_count > 150) | (refined_hold < 0) | (refined_hold > step + 1)).any()):
            raise ValueError("Impossible completed-interval evidence counters")
        payload = {name: tensor.detach().cpu().tolist() for name, tensor in fields.items()}
        for i in range(self.num_envs):
            row = rows[i]
            row.update({name: value[i] for name, value in payload.items()})
            if row["refinement_active"] and info["first_refinement_action_step"][i] is None:
                info["first_refinement_action_step"][i] = step
        json.dump({"pose": pose, "step": step, "rows": [rows[i] for i in range(self.num_envs)]},
                  self.stream, separators=(",", ":"), allow_nan=False)
        self.stream.write("\n")
        info["intervals"] += 1
        info["final_refinement_valid_hold_steps"] = payload["refinement_valid_hold_steps_after"]
        self.rows_written += self.num_envs

    def close(self):
        self.stream.close()

    def report(self):
        if not self.stream.closed:
            raise RuntimeError("Close evidence before hashing")
        if not self.poses or self.rows_written != len(self.poses) * 550 * self.num_envs:
            raise RuntimeError("Missing/incomplete full measured experiment")
        if hashlib.sha256(GATE_PATH.read_bytes()).hexdigest() != GATE_SHA:
            raise RuntimeError("Frozen gate source changed during experiment")
        for value in self.poses.values():
            if value["intervals"] != 550:
                raise RuntimeError("Incomplete full11s experiment")
            value["final_refinement_valid_holds"] = sum(n >= 150 for n in value["final_refinement_valid_hold_steps"])
        return {"mode": self.mode, "num_envs": self.num_envs, "qualifying_completed_intervals": 150,
                "step_dt": .02, "decision_applies_to_next_action": True, "training_performed_in_this_run": False,
                "gate_source_sha256": GATE_SHA, "entry_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "refinement_actor_training": self.receipt, "poses": self.poses,
                "trace_file": self.path.name, "trace_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "full_measured_rows": self.rows_written, "switch_time_physical_or_history_reset": False,
                "extra_pd_ramp_or_retry": False, "promotion_performed": False,
                "scope": "Three-actor zero-command recovery/refinement only; original startup preparation retained, not a walk/run/full-flow demo"}


def build_source():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Frozen combined entry changed")
    if hashlib.sha256(Path(trained.__file__).read_bytes()).hexdigest() != TRAINED_SHA:
        raise ValueError("Frozen training validator changed")
    if hashlib.sha256(GATE_PATH.read_bytes()).hexdigest() != GATE_SHA:
        raise ValueError("Frozen supported-refinement gate changed")
    source = base.build_source()
    edits = [
        ('    selected = [i for i in range(env.num_envs) if neighborhood.wants_row(i, step)]',
         '    selected = list(range(env.num_envs))  # Full measured evidence, not inferred replay.'),
        ('            stand_policy = stand_runner.get_inference_policy(device=env.device)',
         '''            stand_policy = stand_runner.get_inference_policy(device=env.device)
            # Same RNG isolation block: third construction cannot change initial-state draws.
            refinement_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
            refinement_runner.load(REFINEMENT_RECEIPT["checkpoint"], load_optimizer=False)
            refinement_policy = refinement_runner.get_inference_policy(device=env.device)'''),
        ('        _validate_mirror_observation_contract(env.unwrapped, (policy_nn, stand_runner.alg.policy))',
         '        _validate_mirror_observation_contract(env.unwrapped, (policy_nn, stand_runner.alg.policy, refinement_runner.alg.policy))'),
        ('    bank = None\n\n    try:',
         '''    bank = None
    from supported_refinement_gate import advance_after_completed_interval
    refinement_recorder = REFINEMENT_RECORDER(args_cli.output_dir, env.num_envs, REFINEMENT_MODE, REFINEMENT_RECEIPT)

    try:'''),
        ('            legacy_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)',
         '''            legacy_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            refinement_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
            refinement_latched = torch.zeros_like(legacy_success)
            refinement_valid_hold = torch.zeros_like(refinement_count)'''),
        ('                        if action.shape != a.joint_pos.shape:',
         '''                        recovery_action = action
                        refinement_action = refinement_policy(obs)
                        refinement_active = refinement_latched & (REFINEMENT_MODE == "supported")
                        if bool((refinement_active & ~stand_active).any()):
                            raise RuntimeError("Refinement cannot bypass original standing branch")
                        action = torch.where(refinement_active[:, None], refinement_action, recovery_action)
                        if not torch.equal(obs["policy"], real_policy_observation_before):
                            raise RuntimeError("Refinement mutated real observations")
                        if action.shape != a.joint_pos.shape:'''),
        ('                                row["phase"] = "startup_stand"',
         '''                                row["phase"] = "startup_stand"
                            row.update(refinement_active=bool(refinement_active[i]),
                                       original_recovery_raw_action=recovery_action[i].cpu().tolist(),
                                       refinement_actor_raw_action=refinement_action[i].cpu().tolist(),
                                       refinement_gate_count_before=int(refinement_count[i]))
                            if bool(refinement_active[i]):
                                row["phase"] = "postrecovery_refinement"'''),
        ('                stable_steps = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))',
         '''                stable_steps = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))
                # Only completed physical intervals count. This state affects the NEXT action.
                refinement_count, refinement_latched, refinement_edge = advance_after_completed_interval(
                    stable & stand_active, refinement_count, refinement_latched, dt=env.unwrapped.step_dt)
                refinement_valid_hold = torch.where(refinement_active & stable,
                    refinement_valid_hold + 1, torch.zeros_like(refinement_valid_hold))
                refinement_recorder.record(pose_class, step, transition_rows, stable, foot_b, knee_b,
                                           refinement_count, refinement_latched, refinement_valid_hold,
                                           env.unwrapped.scene["robot"].data.projected_gravity_b)'''),
        ('                                                   "stand" if bool(switched[0]) else "roll"))',
         '''                                                   "stand" if bool(switched[0]) else "roll"))
                    if bool(refinement_active[0]):
                        trace[-1]["policy_phase"] = "postrecovery_refinement"'''),
        ('    finally:\n        env.close()', '    finally:\n        refinement_recorder.close()\n        env.close()'),
        ('    report["startup_experiment"] = startup_experiment',
         '''    experiment["original_recovery_action_semantics"] = experiment["hard_zero_semantics"]
    experiment["hard_zero_semantics"] = "where(refinement_active, refinement_action, original_recovery_action)"
    startup_experiment["selected_actor_retained"] = REFINEMENT_MODE == "off"
    combined_experiment["stand_active_is_original_recovery_branch_not_final_actor"] = True
    combined_experiment["postrecovery_refinement_mode"] = REFINEMENT_MODE
    report["startup_experiment"] = startup_experiment'''),
        ('    _write_json_new(report_path, report)',
         '''    report["refinement_experiment"] = refinement_recorder.report()
    report["success_count_definition"] = "Three-actor experiment: original recovery success and final strict hold are separate from final150intervals controlled by refinement actor. No walk/run claim."
    _write_json_new(report_path, report)'''),
        ('        heading = f"EXPERIMENTAL STARTUP {args_cli.startup_mode} / MIRROR {args_cli.mirror_mode}"',
         '        heading = f"POST-RECOVERY REFINEMENT {REFINEMENT_MODE} / 3 ACTORS"'),
    ]
    for old, new in edits:
        source = base._replace_once(source, old, new)
    compile(source, str(__file__) + "::<supported-refinement>", "exec")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--refinement_mode", required=True, choices=("off", "supported"))
    args, remaining = parser.parse_known_args()
    receipt = trained.candidate_receipt(RUN / "training_result.json", RUN / "model_3746.pt")
    source = build_source()
    sys.argv = [sys.argv[0], *remaining]
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
        "COMBINED_PROTOCOL": PROTOCOL, "COMBINED_CONTROLLER": CONTROLLER,
        "COMBINED_TEMPLATE_SHA": base.TEMPLATE_SHA, "COMBINED_STARTUP_REFERENCE_SHA": base.STARTUP_REFERENCE_SHA,
        "COMBINED_STARTUP_MATH_SHA": base.STARTUP_MATH_SHA, "COMBINED_MIRROR_MATH_SHA": base.MIRROR_MATH_SHA,
        "COMBINED_INVALID_JSON": base.invalid_diagnostic_json, "COMBINED_GENERATED_SOURCE": source,
        "COMBINED_GENERATED_SHA": hashlib.sha256(source.encode()).hexdigest(),
        "REFINEMENT_MODE": args.refinement_mode, "REFINEMENT_RECEIPT": receipt, "REFINEMENT_RECORDER": RefinementRecorder}
    exec(compile(source, str(Path(__file__)) + "::<supported-refinement>", "exec"), namespace)
