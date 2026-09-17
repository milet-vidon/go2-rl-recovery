"""CPU-only sensitivity audit of *recorded* stand-actor handoff observations.

This does not simulate, change an evaluated policy, or establish recovery success.
It refuses old reports lacking the real 48-element policy observation. Changing
observation slices below is a counterfactual network-input test, NOT a valid
robot state or a recommended runtime observation rewrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch


EXPECTED_JOINTS = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                   for leg in ("FL", "FR", "RL", "RR")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector(value, size: int, label: str) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float32)
    if tensor.shape != (size,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{label}: expected {size} finite numbers")
    return tensor


def load_actor(path: Path, expected_sha: str) -> torch.nn.Module:
    if sha256(path) != expected_sha:
        raise ValueError("stand checkpoint SHA256 differs from evaluation report")
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    # This audit deliberately supports only the verified unnormalized feed-forward
    # RSL actor architecture. A different architecture must be explicitly audited.
    actor = torch.nn.Sequential(
        torch.nn.Linear(48, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 12),
    )
    actor_state = {key.removeprefix("actor."): value for key, value in state.items()
                   if key.startswith("actor.")}
    actor.load_state_dict(actor_state, strict=True)
    if any("normaliz" in key for key in state):
        raise ValueError("normalizer state requires a separate explicit audit")
    if not all(torch.isfinite(value).all() for value in actor_state.values()):
        raise ValueError("non-finite actor weights")
    return actor.eval()


def summary(rows: list[dict]) -> dict:
    variants = {}
    for name in rows[0]["variants"] if rows else []:
        metrics = rows[0]["variants"][name]
        variants[name] = {
            metric: {"mean": statistics.mean(row["variants"][name][metric] for row in rows),
                     "max": max(row["variants"][name][metric] for row in rows)}
            for metric in metrics if isinstance(metrics[metric], (int, float))
        }
    return {"samples": len(rows), "variants": variants}


@torch.inference_mode()
def probe(report_path: Path, replay_tolerance: float = 1e-4) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("controller_type") != "two_policy_fixed_one_way_handoff_diagnostic":
        raise ValueError("requires an explicitly labelled dual-policy diagnostic report")
    if report.get("policy_action_mode") != "deterministic_mean":
        raise ValueError("recorded stochastic actions cannot validate deterministic replay")
    if report.get("joint_names") != EXPECTED_JOINTS:
        raise ValueError("unknown joint ordering")
    handoff = report["handoff_controller"]
    action = report["action_representation"]
    if action.get("reference") != "nominal" or action.get("scale") != 0.25:
        raise ValueError("expected nominal + 0.25 * raw action")
    if handoff["joint_limits"]["joint_names"] != EXPECTED_JOINTS:
        raise ValueError("joint-limit ordering mismatch")
    default = vector(handoff["joint_limits"]["default_joint_positions_rad"], 12, "default")
    limits = torch.tensor(handoff["joint_limits"]["soft_joint_limits_rad"], dtype=torch.float32)
    if limits.shape != (12, 2) or not torch.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
        raise ValueError("invalid soft limits")
    checkpoint = Path(handoff["stand_checkpoint"])
    actor = load_actor(checkpoint, handoff["stand_sha256"])
    rows = []
    untriggered = 0
    for pose, result in report["results"].items():
        diagnostics = {item["trial"]: item for item in result["final_diagnostics"]}
        switches = result["handoff_diagnostic"]["switch_records"]
        for record in switches:
            if record is None:
                untriggered += 1
                continue
            if "policy_observation" not in record:
                raise ValueError("missing recorded policy_observation; never reconstruct a substitute")
            obs = vector(record["policy_observation"], 48, "policy_observation")
            previous = vector(record["previous_raw_action"], 12, "previous_raw_action")
            executed = vector(record["previous_executed_joint_target_rad"], 12, "previous_executed_joint_target_rad")
            q = vector(record["joint_positions_rad"], 12, "joint_positions_rad")
            qd = vector(record["joint_velocities_rad_s"], 12, "joint_velocities_rad_s")
            saved_action = vector(record["stand_actor_raw_action"], 12, "stand_actor_raw_action")
            for actual, expected, label in ((obs[12:24], q-default, "q-default"),
                                             (obs[24:36], qd, "qd"),
                                             (obs[36:48], previous, "previous raw action")):
                if not torch.allclose(actual, expected, atol=2e-6, rtol=0):
                    raise ValueError(f"recorded observation inconsistent with {label}")
            expected_executed = (default + 0.25 * previous).clamp(limits[:, 0], limits[:, 1])
            if not torch.allclose(executed, expected_executed, atol=2e-6, rtol=0):
                raise ValueError("executed target inconsistent with recorded raw action and clamp")
            baseline = actor(obs)
            replay_error = (baseline-saved_action).abs().max().item()
            if replay_error > replay_tolerance:
                raise ValueError(f"CPU actor replay error {replay_error} exceeds {replay_tolerance}")
            histories = {"raw": previous, "executed_equivalent": (executed-default)/0.25,
                         "zero": torch.zeros_like(previous)}
            variants = {}
            for q_variant in ("actual_q", "nominal_q"):
                for history_name, history in histories.items():
                    changed = obs.clone()
                    changed[36:48] = history
                    if q_variant == "nominal_q":
                        changed[12:24] = 0  # algebraic input sensitivity, not a robot-state reset
                    output = actor(changed)
                    target_unclamped = default + 0.25 * output
                    target = target_unclamped.clamp(limits[:, 0], limits[:, 1])
                    delta = output-baseline
                    variants[f"{q_variant}__history_{history_name}"] = {
                        "raw_output_abs_max": output.abs().max().item(),
                        "raw_output_rms": output.square().mean().sqrt().item(),
                        "raw_output_change_linf": delta.abs().max().item(),
                        "raw_output_change_rms": delta.square().mean().sqrt().item(),
                        "clamped_joint_count": int(((target_unclamped < limits[:, 0]) | (target_unclamped > limits[:, 1])).sum()),
                        "target_jump_linf_rad": (target-executed).abs().max().item(),
                        "raw_output": output.tolist(),
                        "clamped_target_rad": target.tolist(),
                    }
            final = diagnostics[record["trial"]]
            rows.append({"pose": pose, "trial": record["trial"], "switch_time_s": record["policy_time_s"],
                         "recorded_final_stable_hold_s": final["stable_hold_s"],
                         "recorded_final_hold_ge_required": final["stable_hold_s"] >= result["stable_hold_s"],
                         "cpu_replay_max_abs_error": replay_error,
                         "recorded_previous_raw_abs_max": previous.abs().max().item(),
                         "recorded_q_offset_abs_max_rad": (q-default).abs().max().item(),
                         "variants": variants})
    if not rows:
        raise ValueError("no recorded switch observations to audit")
    return {"report_path": str(report_path.resolve()), "report_sha256": sha256(report_path),
            "stand_checkpoint_path": str(checkpoint.resolve()), "stand_checkpoint_sha256": handoff["stand_sha256"],
            "roll_checkpoint_sha256_from_report": handoff["roll_sha256"],
            "probe_source_sha256": sha256(Path(__file__)),
            "architecture": "48->128 ELU->128 ELU->128 ELU->12; no actor normalization; CPU float32",
            "obs_contract": "linvel_body[3],angvel_body[3],gravity_body[3],command[3],q-default[12],qd[12],previous_raw_action[12]; unit scales",
            "simulation_performed": False, "acceptance_eligible": False,
            "interpretation_limit": "Counterfactual input sensitivity only. Nominal q with other channels unchanged need not describe any physically reachable state. No success rate for altered inputs is established. Sensitivity alone cannot prove training-distribution OOD or causation of a fall.",
            "maximum_cpu_replay_abs_error": max(row["cpu_replay_max_abs_error"] for row in rows),
            "untriggered_trials_not_probed": untriggered,
            "all_triggered": summary(rows),
            "recorded_final_hold_pass": summary([row for row in rows if row["recorded_final_hold_ge_required"]]),
            "recorded_final_hold_fail": summary([row for row in rows if not row["recorded_final_hold_ge_required"]]),
            "trials": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--compact", action="store_true", help="Omit per-trial vectors from stdout")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = [probe(path) for path in args.reports]
    if args.compact:
        for result in output:
            result.pop("trials")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Handoff input probe refused: {exc}")
