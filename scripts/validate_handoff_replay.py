"""Small explicit-state/input replay smoke; NOT full PhysX or recovery validation.

Only this script creates an environment. Importing it performs no simulator work.
The main agent must launch it explicitly while no other simulator is active.
Saved observations are comparison targets only and are never fed to a policy or
written into the observation manager. No learned policy is loaded or run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from audit_recovery_report import load_report
from prepare_handoff_train_collection import prepare_collection, require, output_path, SCHEMA


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_DATASET = ROOT / "datasets/handoff_states/train1999_to3448_20260917"
HELPER = ROOT / "src/go2_recovery/handoff_action_restore.py"
TASK = "Isaac-Recovery-Bank-SmithNominal-Flat-Unitree-Go2-Play-v0"
OBS_TOLERANCE = 2e-5
STATE_TOLERANCE = 2e-5
MAX_REPLAY_ENVS = 128
OBS_BLOCKS = {"body_linear_velocity": (0, 3), "body_angular_velocity": (3, 6),
              "projected_gravity": (6, 9), "velocity_command": (9, 12),
              "joint_offset": (12, 24), "joint_velocity": (24, 36), "previous_raw_action": (36, 48)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_sample_rows(ids, sample_ids, all_samples=False):
    """Explicit all-sample opt-in; keep small default smokes and a hard env cap."""
    require(isinstance(all_samples, bool), "all_samples must be an explicit boolean")
    require(len(set(ids)) == len(ids), "Duplicate collection sample IDs")
    if all_samples:
        require(sample_ids is None, "all_samples cannot be combined with sample_ids")
        require(0 < len(ids) <= MAX_REPLAY_ENVS, "All-sample replay requires 1..128 environments")
        return np.arange(len(ids), dtype=np.int64)
    require(sample_ids is not None and len(sample_ids) in (2, 4)
            and len(set(sample_ids)) == len(sample_ids),
            "Exactly 2 or 4 distinct explicit sample IDs are required")
    lookup = {value: index for index, value in enumerate(ids)}
    require(all(value in lookup for value in sample_ids), "Requested sample ID is absent")
    return np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)


def load_raw_collection(path, sample_ids, all_samples=False):
    """Verify the raw archive against original TRAIN reports, without Isaac imports."""
    path = Path(path).resolve()
    require(path.drive.upper() == "E:", "Dataset must resolve to E:")
    manifest_path = path / "manifest.json"
    manifest = load_report(manifest_path)
    require(manifest.get("schema_version") == SCHEMA, "Not a RAW handoff observation collection")
    require(manifest.get("source_split") == "train", "Only TRAIN-origin collection is allowed")
    for key in ("training_ready", "replay_validated", "legacy_bank_loader_compatible", "acceptance_eligible"):
        require(manifest.get(key) is False, f"RAW manifest must declare {key}=false")
    require(manifest.get("archive_filename") == "handoff_observations.npz", "Unexpected raw archive filename")
    archive = path / manifest["archive_filename"]
    payload = archive.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == manifest["archive_sha256"], "Raw archive SHA mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    # Recheck source TRAIN membership, recorded obs/state consistency, all trial
    # denominators, checkpoint hashes and the externally pinned collector source.
    source = manifest["collector_source"]
    verified, regenerated = prepare_collection(
        [report["path"] for report in manifest["source_reports"]], source["path"], source["sha256"])
    require(set(arrays) == set(verified), "Raw array names differ from verified source reports")
    for key in arrays:
        require(arrays[key].dtype == verified[key].dtype and np.array_equal(arrays[key], verified[key]),
                f"Raw array {key} differs from source-report conversion")
    require(manifest["provenance"] == regenerated["provenance"], "RAW provenance differs from verified sources")
    require(manifest["source_reports"] == regenerated["source_reports"], "RAW report ledger differs from sources")
    for key in ("num_samples", "total_source_trials", "untriggered_source_trials", "unique_triggered_source_ids"):
        require(manifest[key] == regenerated[key], f"RAW {key} differs from verified sources")
    ids = arrays["sample_id"].tolist()
    selected = select_sample_rows(ids, sample_ids, all_samples)
    return arrays, manifest, selected, {"path": str(path), "manifest_sha256": digest(manifest_path),
                                       "archive_sha256": manifest["archive_sha256"]}


def compare_observations(actual, recorded):
    actual, recorded = np.asarray(actual, dtype=np.float64), np.asarray(recorded, dtype=np.float64)
    require(actual.shape == recorded.shape and actual.ndim == 2 and actual.shape[1] == 48,
            "Expected matching [N,48] actual and recorded observations")
    require(np.isfinite(actual).all() and np.isfinite(recorded).all(), "Nonfinite replay observation")
    error = np.abs(actual-recorded)
    return {"absolute_tolerance": OBS_TOLERANCE, "relative_tolerance": 0.0,
            "all_elements_match": bool(np.all(error <= OBS_TOLERANCE)),
            "maximum_absolute_error": float(error.max()),
            "per_sample_maximum_absolute_error": error.max(axis=1).tolist(),
            "per_block_maximum_absolute_error": {name: float(error[:, lo:hi].max())
                                                 for name, (lo, hi) in OBS_BLOCKS.items()},
            "absolute_error_per_element": error.tolist(),
            "recorded_observation": recorded.tolist(), "recomputed_observation": actual.tolist()}


def contact_probe_finiteness(state_arrays):
    """Numerical validity only; finite is not stable, standing, or recovery."""
    required = {"root_link_pose_w", "root_com_velocity_w", "joint_position_rad",
                "joint_velocity_rad_s", "contact_forces_w"}
    require(required.issubset(state_arrays), "Contact probe lacks required physical state fields")
    checks = {name: bool(np.asarray(value).size > 0 and np.isfinite(value).all())
              for name, value in state_arrays.items()}
    return {"finite": all(checks.values()), "finite_checks": checks}


def json_safe_array(value):
    """Keep a failing probe report serializable; flag nonfinite values separately."""
    value = np.asarray(value)
    safe = value.astype(object)
    safe[~np.isfinite(value)] = None
    return safe.tolist()


def configure_runtime():
    require(WORKSPACE.drive.upper() == "E:", "Workspace must be on E:")
    values = {"OMNI_KIT_ACCEPT_EULA": "YES", "OMNI_USER_HOME": str(WORKSPACE / "userdata"),
              "OV_USER_HOME": str(WORKSPACE / "userdata"), "TEMP": str(WORKSPACE / "tmp"),
              "TMP": str(WORKSPACE / "tmp"), "PIP_CACHE_DIR": str(WORKSPACE / "cache/pip"),
              "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
              "CONDA_PREFIX": str(WORKSPACE / "env"), "ISAACLAB_RECOVERY_BANK_COLLECTION": "1",
              "ISAACLAB_GO2_USD": str(WORKSPACE / "userdata/assets/Robots/Unitree/Go2/go2.usd"),
              "ISAACLAB_GROUND_USD": str(WORKSPACE / "userdata/assets/Environments/Grid/default_environment.usd")}
    os.environ.update(values)
    os.environ["PATH"] = str(WORKSPACE / "env") + os.pathsep + str(WORKSPACE / "env/Scripts") + os.pathsep + os.environ.get("PATH", "")


def verify_environment(env, cfg, manifest):
    import torch

    provenance = manifest["provenance"]
    asset = env.scene["robot"]
    require(list(asset.joint_names) == provenance["joint_names"], "Runtime joint order differs")
    require(cfg.observations.policy.enable_corruption is False, "Input replay requires noise disabled")
    require(cfg.observations.policy.history_length in (None, 0), "Observation stacks are unsupported")
    require(abs(env.physics_dt-provenance["physics"]["dt"]) < 1e-10 and cfg.decimation == provenance["physics"]["decimation"],
            "Runtime physics timestep/decimation differs")
    require(cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True, "Self collisions must remain enabled")
    require(cfg.events.add_base_mass is None and cfg.events.base_com is None, "Mass/CoM randomization differs")
    require(cfg.events.reset_base.params["collection_mode"] is True and cfg.events.reset_robot_joints is None,
            "Runtime must not sample the old reset bank")
    for name, target in provenance["fixed_material"].items():
        require(tuple(cfg.events.physics_material.params[name+"_range"]) == (target, target), "Fixed material differs")
    recorded = provenance["joint_limits"]
    for field, runtime in (("default_joint_positions_rad", asset.data.default_joint_pos),
                           ("soft_joint_limits_rad", asset.data.soft_joint_pos_limits),
                           ("hard_joint_limits_rad", asset.data.joint_pos_limits)):
        reference = torch.as_tensor(recorded[field], dtype=runtime.dtype, device=runtime.device).expand_as(runtime)
        require(torch.allclose(runtime, reference, atol=1e-6, rtol=0), f"Runtime {field} differs")
    require(bool((asset.data.default_joint_vel == 0).all()), "Nonzero default joint velocities are unsupported")


def run_replay(args, arrays, manifest, selected, destination, resources):
    """Launch only when explicitly called by main; all application imports are local."""
    from isaaclab.app import AppLauncher

    app = None
    wrapped = None
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401 - official task registration after Kit initialization
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        spec = importlib.util.spec_from_file_location("isolated_handoff_action_restore", HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        cfg.scene.num_envs = len(selected)
        cfg.scene.env_spacing = 2.0
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.log_dir = str(destination)
        if hasattr(cfg.sim, "log_dir"):
            cfg.sim.log_dir = str(destination)
        wrapped = gym.make(TASK, cfg=cfg)
        env = wrapped.unwrapped
        env.reset()
        verify_environment(env, cfg, manifest)
        asset = env.scene["robot"]
        dtype, device = asset.data.joint_pos.dtype, env.device
        ids = torch.arange(len(selected), dtype=torch.long, device=device)

        def tensor(key):
            return torch.as_tensor(arrays[key][selected], dtype=dtype, device=device)

        pose = tensor("root_link_pose_local").clone()
        pose[:, :3] += env.scene.env_origins[ids]
        velocity = tensor("root_com_velocity_w")
        q, qd = tensor("joint_positions_rad"), tensor("joint_velocities_rad_s")
        require(bool((tensor("velocity_command_b") == 0).all()), "This fixed zero-command smoke rejects nonzero commands")
        command = env.command_manager.get_command("base_velocity")
        require(torch.equal(command[ids], tensor("velocity_command_b")), "Runtime command differs from recorded command")
        # The collected fixed position-action protocol has zero velocity and
        # feed-forward effort targets. Verify those buffers rather than inventing
        # missing target history or adding a manual PD settling phase.
        require(bool((asset.data.joint_vel_target == 0).all() and (asset.data.joint_effort_target == 0).all()),
                "Unexpected velocity/effort target buffers")
        with torch.no_grad():
            asset.write_joint_state_to_sim(q, qd, env_ids=ids)
            asset.write_root_pose_to_sim(pose, env_ids=ids)  # root LINK pose
            asset.write_root_velocity_to_sim(velocity, env_ids=ids)  # root CoM world velocity
            module.restore_handoff_action_history(
                env.action_manager, asset, ids,
                previous_raw_action=tensor("previous_raw_action"),
                previous_previous_raw_action=tensor("previous_previous_raw_action"),
                previous_executed_joint_target_rad=tensor("previous_executed_joint_target_rad"),
                joint_names=manifest["provenance"]["joint_names"])
            env.scene.sensors["contact_forces"].reset(ids)
            env.scene.write_data_to_sim()
            env.sim.forward()  # kinematics/update only, no sim.step integration
            env.scene.update(env.physics_dt)
            computed = env.observation_manager.compute(update_history=False)
            observation = computed["policy"].detach().cpu().numpy()
            comparison = compare_observations(observation, arrays["policy_observation"][selected])

            def max_error(actual, expected):
                return float((actual-expected).abs().max())

            quat_error = torch.minimum((asset.data.root_quat_w-pose[:, 3:]).abs().amax(dim=1),
                                       (asset.data.root_quat_w+pose[:, 3:]).abs().amax(dim=1))
            state_errors = {
                "root_link_position_world_m": max_error(asset.data.root_pos_w, pose[:, :3]),
                "root_quaternion_sign_invariant": float(quat_error.max()),
                "root_com_linear_velocity_world_m_s": max_error(asset.data.root_lin_vel_w, velocity[:, :3]),
                "root_com_angular_velocity_world_rad_s": max_error(asset.data.root_ang_vel_w, velocity[:, 3:]),
                "joint_position_rad": max_error(asset.data.joint_pos, q),
                "joint_velocity_rad_s": max_error(asset.data.joint_vel, qd),
                "action_tm1": max_error(env.action_manager.action, tensor("previous_raw_action")),
                "action_tm2": max_error(env.action_manager.prev_action, tensor("previous_previous_raw_action")),
                "executed_target_rad": max_error(asset.data.joint_pos_target, tensor("previous_executed_joint_target_rad")),
            }
            require(all(np.isfinite(value) for value in state_errors.values()), "Nonfinite replay state errors")
            result = {"task": TASK, "sample_ids": arrays["sample_id"][selected].tolist(),
                      "selection_mode": "all_samples" if args.all_samples else "explicit_sample_ids",
                      "selected_count": len(selected), "collection_sample_count": len(arrays["sample_id"]),
                      "source_train_state_ids": arrays["source_train_state_id"][selected].tolist(),
                      "source_pose_classes": arrays["source_pose_class"][selected].tolist(),
                      "physics_steps_after_restore_at_input_comparison": 0,
                      "initial_contact_values_valid_for_success": False,
                      "observation_comparison": comparison, "explicit_state_maximum_errors": state_errors,
                      "explicit_state_absolute_tolerance": STATE_TOLERANCE,
                      "explicit_state_matches": all(value <= STATE_TOLERANCE for value in state_errors.values()),
                      "sample_soft_limit_excess_rad": arrays["soft_limit_excess_rad"][selected].tolist(),
                      "sample_hard_limit_excess_rad": arrays["hard_limit_excess_rad"][selected].tolist(),
                      "joint_positions_were_clipped": False, "saved_observations_injected": False,
                      "policy_executed": False, "manual_pd_settling_performed": False,
                      "contact_probe": None}
            if args.contact_probe:
                # Optional and separate from the zero-step input match. This is
                # one physics substep holding the old recorded target, not the
                # first stand-policy action or a success/trajectory comparison.
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env._sim_step_counter += 1
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                sensor = env.scene.sensors["contact_forces"]
                feet = sensor.find_bodies(("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)[0]
                base = sensor.find_bodies("base")[0]
                forces = sensor.data.net_forces_w
                base_force_norm = forces[:, base].norm(dim=-1)
                root_height = asset.data.root_pos_w[:, 2]-env.scene.env_origins[:, 2]
                finite_status = contact_probe_finiteness({
                    "root_link_pose_w": torch.cat((asset.data.root_pos_w, asset.data.root_quat_w), dim=1).cpu().numpy(),
                    "root_com_velocity_w": torch.cat((asset.data.root_lin_vel_w, asset.data.root_ang_vel_w), dim=1).cpu().numpy(),
                    "joint_position_rad": asset.data.joint_pos.cpu().numpy(),
                    "joint_velocity_rad_s": asset.data.joint_vel.cpu().numpy(),
                    "contact_forces_w": forces.cpu().numpy(),
                    "base_contact_force_norm_N": base_force_norm.cpu().numpy(),
                    "root_height_after_step_m": root_height.cpu().numpy()})
                result["contact_probe"] = {
                    **finite_status,
                    "physics_substeps": 1, "elapsed_s": env.physics_dt,
                    "control": "hold previous recorded target; no learned policy; no env.step or auto-reset",
                    "foot_vertical_forces_N": json_safe_array(forces[:, feet, 2].cpu().numpy()),
                    "base_contact_force_norm_N": json_safe_array(base_force_norm.cpu().numpy()),
                    "root_height_after_step_m": json_safe_array(root_height.cpu().numpy()),
                    "nonfinite_values_serialized_as_null": not finite_status["finite"],
                    "used_for_success_classification": False}
        return result
    finally:
        # Transfer even partially initialized resources to the caller. Never
        # close Kit here: close() can terminate Python before an exception or
        # return value reaches main, silently losing the validation report.
        resources.update(app=app, wrapped=wrapped)


def write_report_before_shutdown(destination, report):
    """Durably save the authoritative pass/fail result before either close()."""
    report_path = destination / "handoff_replay_validation.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"report": str(report_path), "input_replay_pass": report["input_replay_pass"],
                      "contact_probe_finite_pass": report["contact_probe_finite_pass"],
                      "validation_pass": report["validation_pass"],
                      "intended_process_exit_code": report["intended_process_exit_code"],
                      "training_ready": False, "full_physx_replay_validated": False}, indent=2), flush=True)
    if "traceback" in report:
        print(report["traceback"], file=sys.stderr, flush=True)


def close_resources(resources):
    """Use the established default shutdown only after the report is durable."""
    app, wrapped = resources.get("app"), resources.get("wrapped")
    try:
        if wrapped is not None:
            wrapped.close()
    finally:
        if app is not None:
            app.close()


def execute_and_record(args, arrays, manifest, selected, destination, report):
    """CPU-testable lifecycle boundary: report and traceback precede shutdown."""
    resources = {}
    try:
        report["result"] = run_replay(args, arrays, manifest, selected, destination, resources)
        result = report["result"]
        report["input_replay_pass"] = result["explicit_state_matches"] and result["observation_comparison"]["all_elements_match"]
        probe = result.get("contact_probe")
        report["contact_probe_finite_pass"] = None if probe is None else probe["finite"]
        report["validation_pass"] = report["input_replay_pass"] and report["contact_probe_finite_pass"] is not False
    except BaseException as error:
        report["input_replay_pass"] = False
        report["contact_probe_finite_pass"] = False if getattr(args, "contact_probe", False) else None
        report["validation_pass"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    exit_code = 0 if report["validation_pass"] else 1
    report["intended_process_exit_code"] = exit_code
    report["result_written_before_shutdown"] = True
    # This file, not a bare process exit code, is the acceptance authority.
    # No successful shutdown is needed for it to contain the real exception.
    write_report_before_shutdown(destination, report)
    close_resources(resources)
    return exit_code


def add_sample_selection_arguments(parser):
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-ids", type=int, nargs="+")
    selection.add_argument("--all-samples", action="store_true",
                           help="Explicitly check all verified TRAIN samples, at most 128 environments")


def main():
    configure_runtime()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    add_sample_selection_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--contact-probe", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    destination = output_path(args.output_dir)
    arrays, manifest, selected, source = load_raw_collection(args.dataset, args.sample_ids, args.all_samples)
    destination.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "handoff_explicit_state_input_replay_smoke_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": source, "script_sha256": digest(__file__), "restore_helper_sha256": digest(HELPER),
              "training_ready": False, "recovery_acceptance_eligible": False, "full_physx_replay_validated": False,
              "boundary": "Only explicit root LINK pose, CoM world velocity, q/qd, action history and recomputed policy input are compared. PhysX contact/friction/solver caches are not restored. Matching input is not standing, recovery, or identical future trajectory evidence. Raw dataset remains unchanged."}
    return execute_and_record(args, arrays, manifest, selected, destination, report)


if __name__ == "__main__":
    raise SystemExit(main())
