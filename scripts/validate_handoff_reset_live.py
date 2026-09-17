"""Eight-env, 24-round real subset-reset contract harness; never trains.

Run explicitly only while no other simulator is active. Unlike the earlier
input smoke, subset comparison does NOT forward/step physics between _reset_idx
and observation_manager.compute(update_history=False), matching auto-reset flow.
Saved observations are comparison targets only. This cannot certify full PhysX
trajectory replay, recovery, or training readiness. Importing does not launch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType

import numpy as np

from validate_handoff_replay import (
    DEFAULT_DATASET, ROOT, close_resources, compare_observations, configure_runtime,
    contact_probe_finiteness, digest, json_safe_array, load_raw_collection, load_report,
    output_path, require,
)


SOURCE = ROOT / "src/go2_recovery"
DEFAULT_RECEIPT = ROOT / "evaluations/20260917-handoff-replay-input-contact4/handoff_replay_validation.json"
NUM_ENVS, ROUNDS = 8, 24
RESET_IDS, UNTOUCHED_IDS = (6, 2, 4), (0, 1, 3, 5, 7)
TOLERANCE = 2e-5
REPORT_NAME = "handoff_reset_subset_validation.json"


def verify_train_ids(manifest):
    """Read original bank bytes and derive TRAIN IDs from split==0, never row IDs."""
    source = manifest["provenance"]["source_bank"]
    path = Path(source["path"]).resolve()
    require(path.drive.upper() == "E:", "Original TRAIN bank must be on E:")
    require(digest(path) == source["sha256"], "Original bank SHA mismatch")
    require(digest(source["manifest_path"]) == source["manifest_sha256"], "Original bank manifest SHA mismatch")
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        split, ids = archive["split"].copy(), archive["state_id"].copy()
    require(split.ndim == ids.ndim == 1 and split.shape == ids.shape and ids.dtype == np.int64,
            "Original bank ID/split layout differs")
    require(np.isin(split, [0, 1]).all() and len(np.unique(ids)) == len(ids), "Invalid original split/IDs")
    train_ids = ids[split == 0]
    require(len(train_ids) == source["train_state_count"], "Original TRAIN count differs")
    return train_ids


def partial_receipt(path, archive_sha):
    path = Path(path).resolve()
    require(path.drive.upper() == "E:", "Reference receipt must be on E:")
    receipt = load_report(path)
    require(receipt.get("protocol") == "handoff_explicit_state_input_replay_smoke_v1", "Wrong partial receipt protocol")
    require(receipt.get("dataset", {}).get("archive_sha256") == archive_sha, "Partial receipt archive mismatch")
    require(receipt.get("input_replay_pass") is True and receipt.get("training_ready") is False,
            "Reference receipt must be a passed input-only diagnostic, not training authority")
    samples = receipt.get("result", {}).get("sample_ids", [])
    require(len(samples) == 4 and len(set(samples)) == 4, "Expected the existing four-sample reference receipt")
    return {"path": str(path), "sha256": digest(path), "sample_ids": samples,
            "scope": "Partial four-sample historical input evidence only; all adapter receipt flags deliberately remain false."}


def load_local_adapter():
    """New namespace; no install, registry mutation, or replacement of old tasks."""
    package_name = "isolated_handoff_subset_validation"
    require(package_name not in sys.modules, "Local validation package already loaded")
    package = ModuleType(package_name)
    package.__path__ = [str(SOURCE)]
    sys.modules[package_name] = package
    loaded = {}
    for stem in ("handoff_reset_env", "handoff_env_cfg"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{stem}", SOURCE / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[stem] = module
    return loaded["handoff_reset_env"], loaded["handoff_env_cfg"]


def assert_environment(env, cfg, manifest):
    import torch

    p = manifest["provenance"]
    a = env.scene["robot"]
    require(env.num_envs == NUM_ENVS and not env.handoff_training_permitted and env.handoff_validation_only,
            "Harness requires eight validation-only environments")
    require(cfg.events.reset_base is None and cfg.events.reset_robot_joints is None, "Old reset event remains enabled")
    require(cfg.observations.policy.enable_corruption is False and cfg.observations.policy.history_length in (None, 0),
            "Require uncorrupted unstacked policy observations")
    require(list(a.joint_names) == p["joint_names"], "Native joint order mismatch")
    require(abs(env.physics_dt - p["physics"]["dt"]) < 1e-10 and cfg.decimation == p["physics"]["decimation"],
            "Physics dt/decimation differs")
    require(cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True, "Self collision is disabled")
    require(cfg.events.add_base_mass is None and cfg.events.base_com is None, "Mass/CoM randomization differs")
    for key, expected in p["fixed_material"].items():
        require(tuple(cfg.events.physics_material.params[key + "_range"]) == (expected, expected), "Material mismatch")
    reference = p["joint_limits"]
    for key, actual in (("default_joint_positions_rad", a.data.default_joint_pos),
                        ("soft_joint_limits_rad", a.data.soft_joint_pos_limits),
                        ("hard_joint_limits_rad", a.data.joint_pos_limits)):
        value = torch.as_tensor(reference[key], device=actual.device, dtype=actual.dtype).expand_as(actual)
        require(torch.allclose(actual, value, rtol=0, atol=1e-6), f"Runtime {key} mismatch")
    require(bool((a.data.default_joint_vel == 0).all() and (a.data.joint_vel_target == 0).all()
                 and (a.data.joint_effort_target == 0).all()), "Unexpected velocity/effort target or default velocity")


def state_snapshot(env):
    a, m = env.scene["robot"], env.action_manager
    t, c = m.get_term("joint_pos"), env.command_manager.get_term("base_velocity")
    values = {"root_link_pose_w": a.data.root_link_pose_w, "root_com_velocity_w": a.data.root_com_vel_w,
              "q": a.data.joint_pos, "qd": a.data.joint_vel,
              "action_tm1": m.action, "action_tm2": m.prev_action, "term_raw": t.raw_actions,
              "term_processed": t.processed_actions, "term_target": t._target,
              "executed_target": a.data.joint_pos_target, "command": c.vel_command_b,
              "is_standing": c.is_standing_env, "is_heading": c.is_heading_env,
              "episode_length": env.episode_length_buf, "sample_ids": env.last_handoff_sample_ids,
              "source_train_ids": env.last_handoff_source_ids, "was_handoff": env.last_reset_was_handoff}
    return {name: value.detach().clone() for name, value in values.items()}


def verify_untouched(before, after, untouched):
    import torch

    physical = {"root_link_pose_w", "root_com_velocity_w", "q", "qd"}
    errors, exact = {}, {}
    for name in before:
        old, new = before[name][untouched], after[name][untouched]
        if name in physical:
            error = float((old - new).abs().max())
            require(np.isfinite(error), f"Nonfinite untouched {name}")
            errors[name] = error
        else:
            exact[name] = bool(torch.equal(old, new))
    return {"physical_max_errors": errors, "history_command_episode_and_labels_exact": exact,
            "passed": all(x <= TOLERANCE for x in errors.values()) and all(exact.values())}


def run_subset(args, arrays, manifest, train_ids, receipt, destination, resources, report):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    resources["app"] = launcher.app
    import torch
    import isaaclab_tasks  # noqa: F401 - standard definitions after Kit initialization
    from isaaclab.utils.math import quat_apply_inverse

    adapter, configs = load_local_adapter()
    dtype, device = torch.float32, torch.device(args.device)
    limits = manifest["provenance"]["joint_limits"]
    tensor_names = ("root_link_pose_local", "root_com_velocity_w", "joint_positions_rad", "joint_velocities_rad_s",
                    "previous_raw_action", "previous_previous_raw_action", "previous_executed_joint_target_rad", "velocity_command_b")
    kwargs = {key: torch.as_tensor(arrays[key], dtype=dtype, device=device) for key in tensor_names}
    data = adapter.HandoffResetData(
        archive_sha256=manifest["archive_sha256"], receipt_sha256=receipt["sha256"], source_split="train",
        input_replay_verified=False, subset_reset_verified=False, contact_probe_finite=False,
        joint_names=tuple(manifest["provenance"]["joint_names"]),
        allowed_train_state_ids=torch.as_tensor(train_ids, dtype=torch.long, device=device),
        source_train_state_id=torch.as_tensor(arrays["source_train_state_id"], dtype=torch.long, device=device),
        default_joint_positions_rad=torch.as_tensor(limits["default_joint_positions_rad"], dtype=dtype, device=device),
        soft_joint_limits_rad=torch.as_tensor(limits["soft_joint_limits_rad"], dtype=dtype, device=device), **kwargs)
    cfg = configs.HandoffStandEnvCfg_PLAY()
    cfg.scene.num_envs, cfg.scene.env_spacing = NUM_ENVS, 2.
    cfg.seed, cfg.sim.device, cfg.log_dir = args.seed, args.device, str(destination)
    if hasattr(cfg.sim, "log_dir"):
        cfg.sim.log_dir = str(destination)
    env = adapter.HandoffResetEnv(cfg, validated_handoff=data, validation_only=True)
    resources["wrapped"] = env
    env.reset(seed=args.seed)  # One ordinary environment reset/forward, BEFORE the test interval.
    assert_environment(env, cfg, manifest)
    ids = torch.tensor(RESET_IDS, device=device, dtype=torch.long)
    untouched = torch.tensor(UNTOUCHED_IDS, device=device, dtype=torch.long)
    a = env.scene["robot"]
    initial_counter, initial_common = env._sim_step_counter, env.common_step_counter
    rows = report["rounds"]
    seen_samples, handoff_count, ordinary_count = set(), 0, 0
    with torch.no_grad():
        for index in range(ROUNDS):
            before = state_snapshot(env)
            counter, common = env._sim_step_counter, env.common_step_counter
            env._reset_idx(ids)
            # Deliberately NO sim.forward(), scene.update(), scene.write_data_to_sim()
            # or physics integration here. Saved input is NEVER injected.
            computed = env.observation_manager.compute(update_history=False)
            observation = computed["policy"]
            require(isinstance(observation, torch.Tensor) and tuple(observation.shape) == (NUM_ENVS, 48)
                    and bool(torch.isfinite(observation).all()), "Invalid recomputed 48-value observation")
            after = state_snapshot(env)
            unchanged = verify_untouched(before, after, untouched)
            mask = env.last_reset_was_handoff[ids]
            hand_ids, ordinary_ids = ids[mask], ids[~mask]
            picked = env.last_handoff_sample_ids[hand_ids]
            handoff_count += int(hand_ids.numel())
            ordinary_count += int(ordinary_ids.numel())
            seen_samples.update(picked.cpu().tolist())
            row = {"round": index, "reset_env_ids": list(RESET_IDS), "untouched_env_ids": list(UNTOUCHED_IDS),
                   "handoff_env_ids": hand_ids.cpu().tolist(), "ordinary_env_ids": ordinary_ids.cpu().tolist(),
                   "handoff_sample_indices": picked.cpu().tolist(),
                   "source_train_state_ids": env.last_handoff_source_ids[hand_ids].cpu().tolist(),
                   "untouched_comparison": unchanged,
                   "simulation_counter_unchanged": env._sim_step_counter == counter,
                   "common_step_counter_unchanged": env.common_step_counter == common,
                   "handoff_observation_comparison": None, "ordinary_observation_max_errors": {},
                   "selected_history_exact": True, "selected_physical_max_errors": {}}
            if hand_ids.numel():
                picked_np = picked.cpu().numpy()
                row["handoff_observation_comparison"] = compare_observations(observation[hand_ids].cpu().numpy(), arrays["policy_observation"][picked_np])
                expected_pose = data.root_link_pose_local[picked].clone()
                expected_pose[:, :3] += env.scene.env_origins[hand_ids]
                for name, expected in (("root_link_pose_w", expected_pose), ("root_com_velocity_w", data.root_com_velocity_w[picked]),
                                       ("q", data.joint_positions_rad[picked]), ("qd", data.joint_velocities_rad_s[picked])):
                    row["selected_physical_max_errors"][name] = float((after[name][hand_ids] - expected).abs().max())
                for name, expected in (("action_tm1", data.previous_raw_action[picked]),
                                       ("action_tm2", data.previous_previous_raw_action[picked]),
                                       ("term_raw", data.previous_raw_action[picked]),
                                       ("term_processed", .25 * data.previous_raw_action[picked]),
                                       ("term_target", data.previous_executed_joint_target_rad[picked]),
                                       ("executed_target", data.previous_executed_joint_target_rad[picked])):
                    row["selected_history_exact"] &= bool(torch.equal(after[name][hand_ids], expected))
            if ordinary_ids.numel():
                ordinary_obs = observation[ordinary_ids]
                gravity = torch.zeros((ordinary_ids.numel(), 3), device=device, dtype=dtype)
                gravity[:, 2] = -1.
                expected_blocks = {"velocities": (slice(0, 6), torch.zeros_like(ordinary_obs[:, :6])),
                                   "gravity": (slice(6, 9), quat_apply_inverse(a.data.root_link_quat_w[ordinary_ids], gravity)),
                                   "commands": (slice(9, 12), torch.zeros_like(ordinary_obs[:, 9:12])),
                                   "joint_offsets": (slice(12, 24), a.data.joint_pos[ordinary_ids] - a.data.default_joint_pos[ordinary_ids]),
                                   "joint_velocities": (slice(24, 36), torch.zeros_like(ordinary_obs[:, 24:36])),
                                   "previous_action": (slice(36, 48), torch.zeros_like(ordinary_obs[:, 36:48]))}
                row["ordinary_observation_max_errors"] = {key: float((ordinary_obs[:, block] - expected).abs().max())
                                                         for key, (block, expected) in expected_blocks.items()}
                q_error = a.data.joint_pos[ordinary_ids] - a.data.default_joint_pos[ordinary_ids]
                height = a.data.root_link_pos_w[ordinary_ids, 2] - env.scene.env_origins[ordinary_ids, 2]
                row["ordinary_drop_bounds_pass"] = bool((q_error.abs() <= .1 + 1e-6).all() and
                    (height >= .34 - TOLERANCE).all() and (height <= .38 + TOLERANCE).all())
                for name in ("action_tm1", "action_tm2", "term_raw", "term_processed"):
                    row["selected_history_exact"] &= bool((after[name][ordinary_ids] == 0).all())
                row["selected_history_exact"] &= bool(torch.equal(after["term_target"][ordinary_ids], a.data.default_joint_pos[ordinary_ids]))
                row["selected_history_exact"] &= bool(torch.equal(after["executed_target"][ordinary_ids], a.data.default_joint_pos[ordinary_ids]))
            else:
                row["ordinary_drop_bounds_pass"] = True
            row["selected_commands_and_episode_zero"] = bool((after["command"][ids] == 0).all() and (after["episode_length"][ids] == 0).all())
            observation_match = row["handoff_observation_comparison"] is None or row["handoff_observation_comparison"]["all_elements_match"]
            numeric_errors = list(row["selected_physical_max_errors"].values()) + list(row["ordinary_observation_max_errors"].values())
            row["passed"] = bool(unchanged["passed"] and row["simulation_counter_unchanged"] and row["common_step_counter_unchanged"]
                and row["selected_history_exact"] and row["selected_commands_and_episode_zero"] and row["ordinary_drop_bounds_pass"]
                and observation_match and all(np.isfinite(error) and error <= TOLERANCE for error in numeric_errors))
            rows.append(row)
            require(row["passed"], f"Subset reset round {index} failed; partial round evidence retained")
        report["coverage"] = {"rounds": len(rows), "handoff_rows": handoff_count, "ordinary_rows": ordinary_count,
                              "unique_handoff_sample_indices": sorted(seen_samples), "total_raw_samples": len(arrays["sample_id"]),
                              "all_raw_samples_tested": len(seen_samples) == len(arrays["sample_id"])}
        require(handoff_count > 0 and ordinary_count > 0, "Both reset branches must be exercised")
        require(env._sim_step_counter == initial_counter and env.common_step_counter == initial_common, "Reset comparison advanced counters")
        report["subset_pass"] = True
        # Separate one-substep contact probe after ALL zero-step comparisons.
        old_target = a.data.joint_pos_target.clone()
        env.action_manager.apply_action()
        require(torch.equal(a.data.joint_pos_target, old_target), "Contact probe changed the previously cached target")
        env.scene.write_data_to_sim()
        env._sim_step_counter += 1
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        forces = env.scene.sensors["contact_forces"].data.net_forces_w
        finite = contact_probe_finiteness({
            "root_link_pose_w": a.data.root_link_pose_w.cpu().numpy(), "root_com_velocity_w": a.data.root_com_vel_w.cpu().numpy(),
            "joint_position_rad": a.data.joint_pos.cpu().numpy(), "joint_velocity_rad_s": a.data.joint_vel.cpu().numpy(),
            "contact_forces_w": forces.cpu().numpy()})
        report["contact_probe"] = {**finite, "physics_substeps": 1, "elapsed_s": env.physics_dt,
            "control": "hold existing cached targets, no policy or env.step",
            "target_unchanged": bool(torch.equal(a.data.joint_pos_target, old_target)),
            "root_heights_m": json_safe_array((a.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()),
            "used_for_success_classification": False}
        report["contact_probe_finite_pass"] = bool(finite["finite"] and report["contact_probe"]["target_unchanged"])


def write_report_before_close(destination, report):
    """Kit close may terminate Python: durably persist success OR traceback first."""
    path = destination / REPORT_NAME
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"report": str(path), "subset_pass": report["subset_pass"],
                      "contact_probe_finite_pass": report["contact_probe_finite_pass"],
                      "validation_pass": report["validation_pass"], "training_ready": False,
                      "intended_process_exit_code": report["intended_process_exit_code"]}, indent=2), flush=True)
    if "traceback" in report:
        print(report["traceback"], file=sys.stderr, flush=True)


def main():
    configure_runtime()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference-receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--preflight-only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device="cuda:0")
    args = parser.parse_args()
    require(args.device == "cuda:0", "This bounded real harness uses cuda:0, eight environments")
    destination = output_path(args.output_dir)
    if args.preflight_only:
        arrays, manifest, _, _ = load_raw_collection(args.dataset, [0, 1])
        train_ids = verify_train_ids(manifest)
        receipt = partial_receipt(args.reference_receipt, manifest["archive_sha256"])
        require(len(arrays["sample_id"]) == 79 and np.isin(arrays["source_train_state_id"], train_ids).all(), "Unexpected RAW79 TRAIN membership")
        print(json.dumps({"preflight_pass": True, "samples": 79, "train_ids": len(train_ids),
                          "reference_receipt": receipt, "simulator_started": False, "output_created": False}, indent=2))
        return 0
    destination.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "handoff_reset_subset_live_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "num_envs": NUM_ENVS, "planned_rounds": ROUNDS, "absolute_tolerance": TOLERANCE,
              "subset_pass": False, "contact_probe_finite_pass": False, "validation_pass": False,
              "training_ready": False, "full_physx_replay_validated": False, "recovery_acceptance_eligible": False,
              "input_all_samples_validated": False, "saved_observations_injected": False,
              "policy_executed": False, "adapter_validation_only": True, "adapter_receipt_flags_all_false": True,
              "rounds": [], "seed": args.seed,
              "boundary": "Subset/zero-step-input consistency plus one finite contact substep only. The initial env.reset may forward once before the comparison interval. No intermediate forward/step in 24 reset rounds. Randomly selected TRAIN handoffs do not cover every RAW sample. No recovery, full future trajectory, or training-readiness claim."}
    resources = {}
    try:
        arrays, manifest, _, report["dataset"] = load_raw_collection(args.dataset, [0, 1])
        require(len(arrays["sample_id"]) == 79, "Expected the full RAW79 collection")
        train_ids = verify_train_ids(manifest)
        require(np.isin(arrays["source_train_state_id"], train_ids).all(), "RAW contains non-TRAIN IDs")
        receipt = partial_receipt(args.reference_receipt, manifest["archive_sha256"])
        report["partial_reference_receipt"] = receipt
        paths = [Path(__file__), SOURCE / "handoff_reset_env.py", SOURCE / "handoff_action_restore.py",
                 SOURCE / "handoff_env_cfg.py", Path(__file__).with_name("validate_handoff_replay.py")]
        report["source_snapshot"] = [{"path": str(path), "sha256": digest(path)} for path in paths]
        run_subset(args, arrays, manifest, train_ids, receipt, destination, resources, report)
        for source in report["source_snapshot"]:
            require(digest(source["path"]) == source["sha256"], "Harness or local source changed during validation")
        report["validation_pass"] = bool(report["subset_pass"] and report["contact_probe_finite_pass"])
    except BaseException as error:
        report["validation_pass"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["intended_process_exit_code"] = 0 if report["validation_pass"] else 1
    report["result_written_before_shutdown"] = True
    write_report_before_close(destination, report)
    close_resources(resources)
    return report["intended_process_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
