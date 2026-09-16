"""Real-simulator subset-reset check for RecoveryBankReset; no policy or physics stepping.

Unlike the pure-data tests, this exercises the full env._reset_idx event/manager
ordering and reads both Isaac Lab buffers and the live PhysX articulation view.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank_path", type=Path, required=True)
parser.add_argument("--output_file", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Recovery-Bank-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--seed", type=int, default=20260916)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.bank_path.exists():
    parser.error(f"Missing state bank: {args_cli.bank_path}")
if args_cli.output_file.resolve().drive.upper() != "E:" or args_cli.output_file.suffix.lower() != ".json":
    parser.error("output_file must be an E: drive JSON path")
if args_cli.output_file.exists():
    parser.error("Refusing to overwrite an earlier diagnostic report; choose a new output_file")
if os.environ.get("ISAACLAB_RECOVERY_BANK_COLLECTION") == "1":
    parser.error("Live reset validation requires ordinary bank mode, not collection mode")
os.environ["ISAACLAB_RECOVERY_BANK_COLLECTION"] = "0"
os.environ["ISAACLAB_RECOVERY_BANK_PATH"] = str(args_cli.bank_path.resolve())

report = {
    "protocol": "live_recovery_bank_subset_reset_v1",
    "status": "started", "success": False,
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "bank_path": str(args_cli.bank_path.resolve()), "task": args_cli.task, "seed": args_cli.seed,
    "num_envs": 16, "selected_env_ids": [1, 7, 11], "collection_mode": False,
    "checks": {},
    "scope": "full reset event ordering and per-env state isolation; not a cold-replay or learned-recovery test",
}
args_cli.output_file.parent.mkdir(parents=True, exist_ok=True)


def _write_report():
    args_cli.output_file.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


# A startup/import/native-shutdown failure must never leave a success-looking file.
_write_report()
try:
    app_launcher = AppLauncher(args_cli)
except BaseException:
    report["status"] = "failed"
    report["error"] = traceback.format_exc()
    _write_report()
    raise
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils import math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_bank_mdp import (  # noqa: E402
    RecoveryBankReset,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_state_bank import (  # noqa: E402
    load_recovery_state_bank,
    transform_bank_root,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def _require(name, condition, message):
    passed = bool(condition)
    report["checks"][name] = passed
    if not passed:
        raise AssertionError(message)


def _snapshot(env, reset_term):
    robot = env.scene["robot"]
    view = robot.root_physx_view
    return {
        "root_state_w": robot.data.root_state_w.clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "physx_root_transforms_xyzw": view.get_root_transforms().clone(),
        "physx_root_com_velocities": view.get_root_velocities().clone(),
        "physx_joint_pos": view.get_dof_positions().clone(),
        "physx_joint_vel": view.get_dof_velocities().clone(),
        "action": env.action_manager.action.clone(),
        "prev_action": env.action_manager.prev_action.clone(),
        "episode_length": env.episode_length_buf.clone(),
        "last_state_ids": reset_term.last_state_ids.clone(),
        "sim_step_counter": int(env._sim_step_counter),
        "common_step_counter": int(env.common_step_counter),
        "sim_time_s": float(env.sim.current_time),
        "sim_time_step_index": int(env.sim.current_time_step_index),
    }


def _check_selected_root(env, bank, rows, selected, after):
    """Infer only world yaw, then independently reconstruct the root replay."""
    source_pose = bank["root_pose_local"][rows]
    actual_pose = after["root_state_w"][selected, :7]
    relative_q = math_utils.quat_mul(actual_pose[:, 3:], math_utils.quat_conjugate(source_pose[:, 3:]))
    _require("selected/root_transform_is_yaw_only", torch.all(relative_q[:, 1:3].abs() < 2e-5),
             "Reset added roll/pitch to a settled bank state")
    yaw = 2.0 * torch.atan2(relative_q[:, 3], relative_q[:, 0])
    expected_pose, expected_velocity = transform_bank_root(
        source_pose, bank["root_velocity_w"][rows], env.scene.env_origins[selected], yaw)
    _require("selected/root_position_uses_target_origin",
             torch.allclose(actual_pose[:, :3], expected_pose[:, :3], atol=3e-5, rtol=0),
             "Selected root positions do not match yaw-transformed local poses plus target origins")
    _require("selected/root_orientation_matches_bank",
             torch.all((actual_pose[:, 3:] * expected_pose[:, 3:]).sum(1).abs() > 1 - 2e-5),
             "Selected root orientations differ from the yaw-transformed bank poses")
    _require("selected/root_com_velocity_matches_bank",
             torch.allclose(after["root_state_w"][selected, 7:], expected_velocity, atol=3e-5, rtol=0),
             "Selected root CoM velocities do not match yaw-transformed recorded velocities")
    live_pose = after["physx_root_transforms_xyzw"][selected]
    live_q_wxyz = live_pose[:, [6, 3, 4, 5]]
    _require("selected/physx_root_position_matches_buffer",
             torch.allclose(live_pose[:, :3], actual_pose[:, :3], atol=3e-5, rtol=0),
             "PhysX selected root positions disagree with Isaac Lab buffers")
    _require("selected/physx_root_quaternion_matches_buffer",
             torch.all((live_q_wxyz * actual_pose[:, 3:]).sum(1).abs() > 1 - 2e-5),
             "PhysX selected root orientations disagree with Isaac Lab buffers")
    _require("selected/physx_root_velocity_matches_bank",
             torch.allclose(after["physx_root_com_velocities"][selected], expected_velocity, atol=3e-5, rtol=0),
             "PhysX selected root CoM velocities disagree with the bank replay")


def main():
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 16
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    _require("configuration/joint_reset_disabled", cfg.events.reset_robot_joints is None,
             "A later reset_robot_joints event would overwrite bank joint states")
    params = cfg.events.reset_base.params
    _require("configuration/normal_bank_mode", params.get("collection_mode") is False,
             "Task construction unexpectedly enabled collection mode")
    params.update({"bank_path": str(args_cli.bank_path.resolve()), "collection_mode": False,
                   "bank_fraction_start": 1.0, "bank_fraction_end": 1.0})
    env = None
    try:
        env = gym.make(args_cli.task, cfg=cfg).unwrapped
        env.reset()
        # EventManager resolves class terms into their live ManagerTermBase instance.
        event_cfg = env.event_manager.get_term_cfg("reset_base")
        reset_term = event_cfg.func
        _require("configuration/live_bank_reset_instance", isinstance(reset_term, RecoveryBankReset),
                 "reset_base did not resolve to the installed RecoveryBankReset instance")
        _require("configuration/bank_loaded", reset_term.bank is not None,
                 "The live reset term did not load a training bank")
        bank = load_recovery_state_bank(args_cli.bank_path, env.scene["robot"].joint_names,
                                        env.device, split="all")
        report["bank_sha256"] = bank["archive_sha256"]
        report["bank_train_states"] = int((bank["split"] == 0).sum())
        report["bank_heldout_states"] = int((bank["split"] == 1).sum())
        report["reset_event_names"] = list(env.event_manager.active_terms.get("reset", []))
        selected = torch.tensor([1, 7, 11], device=env.device, dtype=torch.long)
        unselected_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        unselected_mask[selected] = False
        unselected = unselected_mask.nonzero().flatten()
        with torch.no_grad():
            # Populate current AND previous action history without executing actions
            # or stepping physics. Unique rows make broad/global resets detectable.
            action_shape = (env.num_envs, env.action_manager.total_action_dim)
            first_action = (torch.arange(action_shape[0] * action_shape[1], device=env.device,
                                         dtype=torch.float32).reshape(action_shape) + 1) / 1000.0
            env.action_manager.process_action(first_action)
            env.action_manager.process_action(first_action + 0.25)
            env.observation_manager.compute(update_history=True)
            env.episode_length_buf[:] = torch.arange(env.num_envs, device=env.device) + 50
            before = _snapshot(env, reset_term)
            _require("precondition/nonzero_action_history",
                     torch.all(before["action"] != 0) & torch.all(before["prev_action"] != 0),
                     "Action-history fixture was not populated")
            _require("precondition/nonzero_episode_lengths", torch.all(before["episode_length"] > 0),
                     "Episode-length fixture was not populated")
            # Exercise the REAL full reset ordering, not just the bank term itself.
            env._reset_idx(selected)
            after = _snapshot(env, reset_term)

            for name, old in before.items():
                if isinstance(old, torch.Tensor):
                    _require(f"unselected/{name}", torch.equal(old[unselected], after[name][unselected]),
                             f"Subset reset modified unselected environments' {name}")
                else:
                    _require(f"global/{name}", old == after[name],
                             f"Subset reset advanced or modified global {name}")
            chosen_ids = after["last_state_ids"][selected]
            matches = chosen_ids[:, None] == bank["state_id"][None, :]
            _require("selected/state_ids_exist_once", torch.all(matches.sum(1) == 1),
                     "Selected last_state_ids are absent or duplicated in the independent full-bank load")
            rows = matches.to(torch.int64).argmax(1)
            _require("selected/state_ids_are_train", torch.all(bank["split"][rows] == 0),
                     "Reset used a heldout state or an invalid state ID")
            for name, bank_key in (("joint_pos", "joint_pos"), ("joint_vel", "joint_vel"),
                                   ("physx_joint_pos", "joint_pos"), ("physx_joint_vel", "joint_vel")):
                _require(f"selected/{name}_exact_bank_match",
                         torch.equal(after[name][selected], bank[bank_key][rows]),
                         f"Selected {name} was overwritten after the bank reset event")
            _require("selected/action_reset", torch.equal(after["action"][selected], torch.zeros_like(after["action"][selected])),
                     "Selected action histories were not reset")
            _require("selected/prev_action_reset", torch.equal(after["prev_action"][selected], torch.zeros_like(after["prev_action"][selected])),
                     "Selected previous-action histories were not reset")
            _require("selected/episode_reset", torch.all(after["episode_length"][selected] == 0),
                     "Selected episode lengths were not reset")
            _check_selected_root(env, bank, rows, selected, after)
        scalar_names = ("sim_step_counter", "common_step_counter", "sim_time_s", "sim_time_step_index")
        report["counters_before"] = {name: before[name] for name in scalar_names}
        report["counters_after"] = {name: after[name] for name in scalar_names}
        report["selected_state_ids"] = chosen_ids.cpu().tolist()
        report["selected_pose_classes"] = [bank["pose_class"][i] for i in rows.cpu().tolist()]
        report["selected_joint_positions"] = after["joint_pos"][selected].cpu().tolist()
        report["selected_joint_velocities"] = after["joint_vel"][selected].cpu().tolist()
        report["selected_target_origins"] = env.scene.env_origins[selected].cpu().tolist()
        report["unselected_env_ids"] = unselected.cpu().tolist()
        report["all_checks_passed"] = all(report["checks"].values())
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
        report["status"] = "passed"
        report["success"] = True
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report()
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                          "selected_state_ids": report["selected_state_ids"],
                          "report": str(args_cli.output_file)}, indent=2), flush=True)
    except BaseException:
        exit_code = 1
        report["status"] = "failed"
        report["success"] = False
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["error"] = traceback.format_exc()
        _write_report()
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
        simulation_app.close()
    # Some native Kit shutdown paths terminate before returning; callers must
    # additionally require a fresh report with success=true and status=passed.
    raise SystemExit(exit_code)
