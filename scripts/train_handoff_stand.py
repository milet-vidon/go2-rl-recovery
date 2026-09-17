"""Isolated, receipt-gated 3448 standing continuation; no promotion or hardware.

Only smoke=16 environments x 2 updates and formal=128 x 100 are permitted.
Both start from the frozen full 3448 checkpoint. This is not a 300-update
automatic chain: later blocks require a separately reviewed upright guard.
--preflight-only imports no simulator and creates no run directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from types import ModuleType

import numpy as np

from validate_handoff_replay import (
    ROOT, WORKSPACE, DEFAULT_DATASET, digest, load_raw_collection, load_report,
    require, configure_runtime, close_resources,
)


SOURCE = ROOT / "src/go2_recovery"
INSTALLED = WORKSPACE / "repo/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2"
PARENT = WORKSPACE / "repo/logs/rsl_rl/unitree_go2_recovery/2026-09-10_17-15-48_aligned_from_uncrossed2849/model_3448.pt"
PARENT_SHA = "1f546523baa57c7997ad2883689667d6e738eac098f14fb5fa595aae778a2de2"
PARENT_ITER, PARENT_ADAM_STEP = 3448, 69120
PARENT_LR = 2.2500000000000008e-05
BUDGETS = {"smoke": (16, 2), "formal": (128, 100)}
RUN_ROOT = WORKSPACE / "repo/logs/rsl_rl/unitree_go2_handoff_stand"
DEFAULT_INPUT_RECEIPT = ROOT / "evaluations/20260917-handoff-replay-all79/handoff_replay_validation.json"
DEFAULT_SUBSET_RECEIPT = ROOT / "evaluations/20260917-handoff-reset-subset8/handoff_reset_subset_validation.json"
INSTALLED_REFERENCE = WORKSPACE / "artifacts/recovery-20260917/20260917-speedretention-smoke/source-snapshot.json"
FROZEN_MODELS = {
    "locomotion/natural_model950.pt": "8f24061945e885ae435ce56ffc183d55037533fe150503f87d0f63e3bc71a040",
    "locomotion/natural_robust_push_model3648.pt": "475f07bf0b05101dd4157212b4e3fd48a04731d452d7a349fe467c508dbaaac8",
    "locomotion/natural_stop_diagonal_model2250.pt": "33be609b3daab4a2f773d70aeeb41bc42c5cbc843780efc963ed04d5dd368417",
    "locomotion/natural_stop_model1349.pt": "226e9a8df7e619a0430a17dbeef17f93ebbe7cc35fcc1f3be58305d790444403",
    "locomotion/robust_model799.pt": "a237ba7b5edfd854f42b57f96409cd97d6ef599c8dbb97da2f10261402229648",
    "locomotion/standard_stance_model950.pt": "2b4f5be56f0cec18db125931619590ddec2ee1f35233c03c69f7ebd6fbc5df9c",
    "recovery/recovery_aligned_model3448.pt": PARENT_SHA,
    "recovery/recovery_bank_rejected_model4899.pt": "a021ba7a25d5a9803f7f905b4a242df8582a4d2730de5843303e76beaa26b4e2",
    "recovery/recovery_condposture_model3498.pt": "b823eefb610a43b4740839d04451ba1d4632ed2b04bb3e9a28e600ddb6ef95bb",
    "recovery/recovery_fourfeet_hard45_from2200_model3199.pt": "a5299bde3d530c97e2543d5ebc3a3164139b4038740d302578560b3585d420e1",
    "recovery/recovery_hard45_side_cur1_model3699.pt": "f31aa7ad513dc768728fb6ae97304935d059d0164c269fd00cd7d101c6ebaa04",
    "recovery/recovery_hard45_side_from2200_model2699.pt": "6261776c6ae571a31069115b680bd7635d4faccce23f053fec024d3a5d0124f1",
    "recovery/recovery_lift_model2300.pt": "ecb1c328e34ebc44ed82275e2769536b4a9d1dd20880dd695c5288e575148d4e",
    "recovery/recovery_rehearsal_model3900.pt": "70f1429040ebe3db1f8bdcbe246bff0c791f50c9739a46568932332e1ffe81aa",
    "recovery/recovery_stage30_model2200.pt": "bc9c74566fc3aae72d906198cd82686563b35b5fd3082f835f8451d702773d5d",
    "recovery/recovery_uncrossed_intermediate_model2849.pt": "849abe1b0a6810d42433a86c284bc015c884a5f76497f2cfab87f691860703fe",
}
SUBSET_REQUIRED_SOURCES = [SOURCE / name for name in (
    "handoff_reset_env.py", "handoff_action_restore.py", "handoff_env_cfg.py")]
SUBSET_REQUIRED_SOURCES += [ROOT / "scripts/validate_handoff_replay.py", ROOT / "scripts/validate_handoff_reset_live.py"]


def e_file(path, label):
    path = Path(path).resolve()
    require(path.drive.upper() == "E:" and path.is_file(), f"Missing E-drive {label}: {path}")
    return path


def run_destination(tag):
    require(isinstance(tag, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", tag), "Invalid new run tag")
    path = (RUN_ROOT / tag).resolve()
    require(path.drive.upper() == "E:" and path.parent == RUN_ROOT.resolve(), "Run must stay inside E-drive run root")
    require(not path.exists(), f"Run already exists; refusing overwrite: {path}")
    return path


def verify_frozen_models():
    actual = {p.relative_to(ROOT / "models").as_posix(): digest(p)
              for p in sorted((ROOT / "models").rglob("*.pt"))}
    require(actual == FROZEN_MODELS, "The 16 frozen portfolio models changed or model inventory differs")
    require(digest(PARENT) == PARENT_SHA, "Original full parent checkpoint SHA mismatch")
    return actual


def validate_input_receipt(receipt, arrays, dataset):
    require(receipt.get("protocol") == "handoff_explicit_state_input_replay_smoke_v1", "Wrong input receipt protocol")
    for field in ("input_replay_pass", "contact_probe_finite_pass", "validation_pass", "result_written_before_shutdown"):
        require(receipt.get(field) is True, f"Input receipt missing/pending/false: {field}")
    require(receipt.get("training_ready") is False, "Input receipt must remain diagnostic, not a training-ready RAW claim")
    require(receipt.get("dataset") == dataset, "Input receipt dataset path/manifest/archive SHA differs")
    require(receipt.get("script_sha256") == digest(ROOT / "scripts/validate_handoff_replay.py"), "Input validator source changed")
    require(receipt.get("restore_helper_sha256") == digest(SOURCE / "handoff_action_restore.py"), "Input helper source changed")
    result = receipt["result"]
    require(result.get("selection_mode") == "all_samples" and result.get("selected_count") == len(arrays["sample_id"]),
            "Input receipt is not an explicit all-sample run")
    require(result.get("sample_ids") == arrays["sample_id"].tolist() and
            result.get("source_train_state_ids") == arrays["source_train_state_id"].tolist(), "All TRAIN rows are not covered exactly")
    require(result.get("physics_steps_after_restore_at_input_comparison") == 0 and result.get("explicit_state_matches") is True,
            "Input receipt lacks zero-step explicit state match")
    require(result.get("observation_comparison", {}).get("all_elements_match") is True, "Recomputed observation mismatch")
    require(result.get("saved_observations_injected") is False and result.get("policy_executed") is False, "Unexpected replay protocol")
    probe = result.get("contact_probe", {})
    require(probe.get("finite") is True and probe.get("physics_substeps") == 1 and
            probe.get("used_for_success_classification") is False, "Missing finite one-substep diagnostic")
    checks = probe.get("finite_checks", {})
    required = {"root_link_pose_w", "root_com_velocity_w", "joint_position_rad", "joint_velocity_rad_s", "contact_forces_w"}
    require(required.issubset(checks) and all(value is True for value in checks.values()), "Incomplete/nonfinite contact probe checks")


def validate_subset_receipt(receipt, dataset):
    require(receipt.get("protocol") == "handoff_reset_subset_live_v1", "Wrong subset receipt protocol")
    for field in ("subset_pass", "contact_probe_finite_pass", "validation_pass"):
        require(receipt.get(field) is True, f"Subset receipt missing/pending/false: {field}")
    require(receipt.get("training_ready") is False, "Subset receipt must remain diagnostic")
    require(receipt.get("dataset") == dataset, "Subset dataset path/manifest/archive SHA differs")
    require(receipt.get("num_envs") == 8 and receipt.get("planned_rounds") == 24, "Unexpected live subset protocol size")
    coverage = receipt.get("coverage", {})
    require(coverage.get("rounds") == 24 and coverage.get("handoff_rows", 0) > 0 and
            coverage.get("ordinary_rows", 0) > 0 and coverage.get("total_raw_samples") == 79, "Subset did not exercise both branches")
    rounds = receipt.get("rounds", [])
    require(isinstance(rounds, list) and len(rounds) == 24 and all(row.get("passed") is True for row in rounds),
            "Subset live rounds incomplete/failed")
    for row in rounds:
        untouched = row.get("untouched_comparison", {})
        require(untouched.get("passed") is True, "Subset changed untouched environments")
        exact = untouched.get("history_command_episode_and_labels_exact", {})
        require(exact and all(value is True for value in exact.values()), "Untouched manager history was not exact")
        for field in ("simulation_counter_unchanged", "common_step_counter_unchanged", "selected_history_exact", "selected_commands_and_episode_zero"):
            require(row.get(field) is True, f"Subset row failed {field}")
    snapshot = receipt.get("source_snapshot", [])
    require(isinstance(snapshot, list) and snapshot, "Subset source SHA ledger missing")
    pins = {}
    for entry in snapshot:
        path = e_file(entry["path"], "subset source")
        require(str(path) not in pins and digest(path) == entry["sha256"], f"Subset source changed: {path}")
        pins[str(path)] = entry["sha256"]
    for path in SUBSET_REQUIRED_SOURCES:
        require(str(path.resolve()) in pins, f"Subset receipt did not pin required source: {path}")


def train_ids_from_bank(manifest):
    source = manifest["provenance"]["source_bank"]
    path = e_file(source["path"], "original TRAIN bank")
    require(digest(path) == source["sha256"], "Original TRAIN bank SHA changed")
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as bank:
        ids, split = bank["state_id"].copy(), bank["split"].copy()
    require(ids.ndim == split.ndim == 1 and ids.shape == split.shape and len(set(ids.tolist())) == len(ids), "Invalid source bank IDs")
    require(np.isin(split, [0, 1]).all(), "Unknown source split")
    return ids[split == 0]


def verify_installed_reference():
    # This is separate historical/current dependency evidence. The subset
    # receipt only pinned its five local files; do not retrofit that claim.
    reference = e_file(INSTALLED_REFERENCE, "independent installed-source snapshot")
    entries = json.loads(reference.read_text(encoding="utf-8-sig"))
    pins = {str(Path(row["path"]).resolve()): row["sha256"].lower() for row in entries}
    verified = []
    for path in sorted(INSTALLED.rglob("*.py")):
        require(str(path.resolve()) in pins and digest(path) == pins[str(path.resolve())],
                f"Installed dependency changed from independent snapshot: {path}")
        local = SOURCE / path.name
        if local.is_file():
            require(digest(path) == digest(local), f"Installed/portfolio source mismatch: {path}")
        verified.append(str(path))
    require(digest(INSTALLED / "agents/rsl_rl_ppo_cfg.py") == digest(SOURCE / "go2_ppo_snapshot.py"), "Installed PPO differs")
    return {"path": str(reference), "sha256": digest(reference), "verified_installed_files": verified,
            "scope": "Independent pre-existing installed-source snapshot plus current equality; not embedded in the subset receipt."}


def checkpoint_summary(checkpoint, expected_iteration, expected_adam_step):
    import torch

    require(checkpoint.get("iter") == expected_iteration, "Checkpoint iteration mismatch")
    model, optimizer = checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"]
    require(len(model) == 17 and "std" in model and tuple(model["std"].shape) == (12,), "Unexpected actor/critic/std checkpoint layout")
    for name, value in model.items():
        require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), f"Nonfinite model tensor: {name}")
    require(bool((model["std"] > 0).all()), "Policy std must be positive")
    states, groups = optimizer["state"], optimizer["param_groups"]
    require(len(states) == 17 and len(groups) == 1 and len(groups[0]["params"]) == 17, "Incomplete Adam optimizer state")
    require(set(states) == set(groups[0]["params"]), "Adam state/parameter IDs differ")
    steps = []
    for key, state in states.items():
        require(set(state) >= {"step", "exp_avg", "exp_avg_sq"}, f"Missing Adam moment for {key}")
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), f"Nonfinite Adam {key}/{name}")
        step = float(state["step"])
        require(step == expected_adam_step, f"Adam step {step} != {expected_adam_step}")
        steps.append(step)
    lr = groups[0]["lr"]
    require(math.isfinite(lr) and lr > 0, "Invalid optimizer LR")
    return {"iteration": expected_iteration, "model_tensor_count": len(model), "adam_state_count": len(states),
            "adam_steps": steps, "optimizer_learning_rate": lr, "all_model_and_adam_tensors_finite": True,
            "std_min": float(model["std"].min()), "std_max": float(model["std"].max())}


def snapshot_sources():
    files = set(SOURCE.glob("*.py")) | set(INSTALLED.rglob("*.py"))
    files.update([Path(__file__), ROOT / "scripts/validate_handoff_replay.py",
                  ROOT / "scripts/validate_handoff_reset_live.py", ROOT / "scripts/prepare_handoff_train_collection.py",
                  WORKSPACE / "env/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
                  WORKSPACE / "env/Lib/site-packages/rsl_rl/algorithms/ppo.py"])
    return [{"path": str(path.resolve()), "sha256": digest(path)} for path in sorted(files)]


def preflight(args):
    import torch

    require(args.mode in BUDGETS and args.device == "cuda:0", "Only fixed smoke/formal budgets on cuda:0 are supported")
    destination = run_destination(args.tag)
    arrays, manifest, selected, dataset = load_raw_collection(args.dataset, None, all_samples=True)
    require(len(selected) == 79 and manifest["total_source_trials"] == 80, "Expected frozen 79/80 TRAIN collection")
    input_path = e_file(args.input_receipt, "all79 input/finite receipt")
    subset_path = e_file(args.subset_receipt, "live subset receipt")
    validate_input_receipt(load_report(input_path), arrays, dataset)
    validate_subset_receipt(load_report(subset_path), dataset)
    installed_reference = verify_installed_reference()
    train_ids = train_ids_from_bank(manifest)
    require(np.isin(arrays["source_train_state_id"], train_ids).all(), "Heldout source ID in training payload")
    frozen = verify_frozen_models()
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    summary = checkpoint_summary(parent, PARENT_ITER, PARENT_ADAM_STEP)
    require(summary["optimizer_learning_rate"] == PARENT_LR, "Parent optimizer LR changed")
    n, updates = BUDGETS[args.mode]
    report = {"protocol": "handoff_stand_training_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "preflight_pass": True, "mode": args.mode, "num_envs": n, "updates": updates, "seed": args.seed,
              "run_dir": str(destination), "dataset": dataset,
              "input_receipt": {"path": str(input_path), "sha256": digest(input_path)},
              "subset_receipt": {"path": str(subset_path), "sha256": digest(subset_path)},
              "parent": {"path": str(PARENT), "sha256": PARENT_SHA, **summary},
              "source_snapshot": snapshot_sources(), "frozen_models": frozen,
              "installed_dependency_evidence": installed_reference,
              "expected_final_iteration": PARENT_ITER + updates - 1,
              "expected_final_adam_step": PARENT_ADAM_STEP + 20 * updates,
              "expected_environment_steps": n * 24 * updates,
              "promotion_performed": False, "hardware_execution": False, "recovery_acceptance_eligible": False,
              "training_ready": False, "raw_dataset_manifest_unchanged": True,
              "boundary": "Receipt gates certify recorded inputs, subset reset isolation and finite microsteps only. The ordinary 80% branch is a shallow training drop, not pre-settled standing. Training completion is not recovery or no-regression acceptance. No automatic continuation beyond one bounded block."}
    return arrays, manifest, train_ids, parent, report


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def equal_tree(actual, expected, path="root"):
    import torch

    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()), f"Loaded tensor differs: {path}")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Loaded keys differ: {path}")
        for key in expected:
            equal_tree(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), f"Loaded sequence differs: {path}")
        for i, (one, two) in enumerate(zip(actual, expected)):
            equal_tree(one, two, f"{path}/{i}")
    else:
        require(actual == expected, f"Loaded value differs: {path}")


def load_local_modules():
    package_name = "isolated_handoff_stand_training"
    require(package_name not in sys.modules, "Isolated training namespace already loaded")
    package = ModuleType(package_name)
    package.__path__ = [str(SOURCE)]
    sys.modules[package_name] = package
    loaded = []
    for name in ("handoff_reset_env", "handoff_env_cfg"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", SOURCE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def config_guard(cfg, agent):
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_env_cfg import (
        UnitreeGo2RecoveryAlignedEnvCfg, UnitreeGo2RecoveryBankSmithNominalEnvCfg)

    aligned, smith = UnitreeGo2RecoveryAlignedEnvCfg(), UnitreeGo2RecoveryBankSmithNominalEnvCfg()
    for field in ("rewards", "observations", "terminations", "commands", "curriculum"):
        require(getattr(cfg, field).to_dict() == getattr(aligned, field).to_dict(), f"Aligned {field} changed")
    require(cfg.observations.policy.enable_corruption is True, "Training noise must remain enabled")
    require(cfg.actions.to_dict() == smith.actions.to_dict(), "Smith nominal control-step action semantics changed")
    require(cfg.scene.robot.to_dict() == smith.scene.robot.to_dict(), "Robot/actuation/self-collision physics changed")
    require(cfg.sim.dt == smith.sim.dt and cfg.decimation == smith.decimation, "Smith timestep/decimation changed")
    require(cfg.sim.physx.to_dict() == smith.sim.physx.to_dict(), "Smith PhysX configuration changed")
    require(cfg.events.physics_material.to_dict() == smith.events.physics_material.to_dict(), "Smith fixed material changed")
    require(cfg.events.add_base_mass is None and cfg.events.base_com is None and
            cfg.events.reset_base is None and cfg.events.reset_robot_joints is None, "Unexpected mass or legacy reset events")
    require(agent.num_steps_per_env == 24 and agent.algorithm.num_learning_epochs == 5 and
            agent.algorithm.num_mini_batches == 4, "PPO update counting assumptions changed")
    require(agent.policy.actor_hidden_dims == agent.policy.critic_hidden_dims == [128, 128, 128] and
            agent.policy.activation == "elu" and not agent.policy.actor_obs_normalization and
            not agent.policy.critic_obs_normalization, "Parent policy architecture/normalization changed")
    return {"aligned_rewards_observations_terminations_commands_curriculum_equal": True,
            "smith_robot_physx_dt_decimation_material_actions_equal": True,
            "changes_from_original_aligned": ["80% shallow upright training drops / 20% verified TRAIN handoffs",
                "fixed Smith material; no mass/CoM randomization", "Smith nominal control-step soft-clamped targets",
                "12-second episode; post-manager-reset history restoration", "explicit finite budget/seed/LR resume"],
            "ordinary_branch_is_already_settled_standing": False,
            "compatibility_note": "Old 3448 actions did not use this clamp. Compatible Smith upright evaluation is prior evidence, not proof this new reset mixture preserves standing."}


def run_training(args, arrays, manifest, train_ids, parent, report, resources):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    resources["app"] = launcher.app
    import torch
    import isaaclab_tasks  # noqa: F401 - only after AppLauncher
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RecoveryPPORunnerCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    adapter, configs = load_local_modules()
    device, dtype = torch.device(args.device), torch.float32
    limits = manifest["provenance"]["joint_limits"]
    fields = ("root_link_pose_local", "root_com_velocity_w", "joint_positions_rad", "joint_velocities_rad_s",
              "previous_raw_action", "previous_previous_raw_action", "previous_executed_joint_target_rad", "velocity_command_b")
    tensors = {key: torch.as_tensor(arrays[key], dtype=dtype, device=device) for key in fields}
    data = adapter.HandoffResetData(
        archive_sha256=manifest["archive_sha256"], receipt_sha256=report["subset_receipt"]["sha256"], source_split="train",
        input_replay_verified=True, subset_reset_verified=True, contact_probe_finite=True,
        joint_names=tuple(manifest["provenance"]["joint_names"]),
        allowed_train_state_ids=torch.as_tensor(train_ids, dtype=torch.long, device=device),
        source_train_state_id=torch.as_tensor(arrays["source_train_state_id"], dtype=torch.long, device=device),
        default_joint_positions_rad=torch.as_tensor(limits["default_joint_positions_rad"], dtype=dtype, device=device),
        soft_joint_limits_rad=torch.as_tensor(limits["soft_joint_limits_rad"], dtype=dtype, device=device), **tensors)
    cfg, agent = configs.HandoffStandEnvCfg(), UnitreeGo2RecoveryPPORunnerCfg()
    agent_base = agent.to_dict()
    n, updates = BUDGETS[args.mode]
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = n, args.seed, args.device
    cfg.log_dir = report["run_dir"]
    if hasattr(cfg.sim, "log_dir"):
        cfg.sim.log_dir = report["run_dir"]
    agent.seed, agent.device, agent.max_iterations, agent.run_name = args.seed, args.device, updates, args.tag
    agent.experiment_name = "unitree_go2_handoff_stand"
    agent.resume, agent.load_run, agent.load_checkpoint = True, PARENT.parent.name, PARENT.name
    agent.algorithm.learning_rate = PARENT_LR
    require(agent.logger == "tensorboard", "External logging is forbidden")
    report["config_guard"] = config_guard(cfg, agent)
    expected_agent = dict(agent_base)
    expected_agent.update(seed=args.seed, device=args.device, max_iterations=updates, run_name=args.tag)
    expected_agent.update(experiment_name="unitree_go2_handoff_stand", resume=True,
                          load_run=PARENT.parent.name, load_checkpoint=PARENT.name)
    expected_agent["algorithm"] = dict(agent_base["algorithm"], learning_rate=PARENT_LR)
    require(agent.to_dict() == expected_agent, "Unapproved PPO configuration change")
    env = adapter.HandoffResetEnv(cfg, validated_handoff=data, validation_only=False)
    resources["wrapped"] = env
    require(env.handoff_training_permitted and not env.handoff_validation_only, "Adapter denied training")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    resources["wrapped"] = wrapped
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=report["run_dir"], device=agent.device)
    runner.load(str(PARENT), load_optimizer=True, map_location=args.device)
    equal_tree(runner.alg.policy.state_dict(), parent["model_state_dict"], "model")
    equal_tree(runner.alg.optimizer.state_dict(), parent["optimizer_state_dict"], "optimizer")
    require(runner.current_learning_iteration == PARENT_ITER, "Runner resumed wrong iteration")
    loaded_lr = runner.alg.optimizer.param_groups[0]["lr"]
    require(loaded_lr == PARENT_LR, "Loaded optimizer LR mismatch")
    runner.alg.learning_rate = loaded_lr
    require(runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"], "Scalar and optimizer LR diverge")
    report["loaded_full_parent_exact"] = True
    report["initial_algorithm_scalar_learning_rate"] = runner.alg.learning_rate
    destination = Path(report["run_dir"])
    dump_yaml(str(destination / "params/env.yaml"), cfg)
    dump_yaml(str(destination / "params/agent.yaml"), agent)
    write_json_exclusive(destination / "before_learning.json", report)
    runner.learn(num_learning_iterations=updates, init_at_random_ep_len=True)
    expected_iteration = PARENT_ITER + updates - 1
    require(runner.current_learning_iteration == expected_iteration, "Runner completed unexpected update count")
    final_path = destination / f"model_{expected_iteration}.pt"
    require(final_path.is_file(), "Runner did not write expected final checkpoint")
    checkpoints = []
    for path in sorted(destination.glob("model_*.pt")):
        iteration = int(path.stem.split("_")[-1])
        require(PARENT_ITER <= iteration <= expected_iteration, "Unexpected checkpoint iteration")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        summary = checkpoint_summary(saved, iteration, PARENT_ADAM_STEP + 20 * (iteration-PARENT_ITER+1))
        checkpoints.append({"path": str(path), "sha256": digest(path), **summary})
    require(checkpoints, "No actual checkpoint was verified")
    require(report["frozen_models"] == verify_frozen_models(), "Frozen model changed during training")
    require(snapshot_sources() == report["source_snapshot"], "Training source changed during this run")
    require(digest(args.input_receipt) == report["input_receipt"]["sha256"] and
            digest(args.subset_receipt) == report["subset_receipt"]["sha256"], "Receipt changed during training")
    require(digest(Path(args.dataset) / "manifest.json") == report["dataset"]["manifest_sha256"] and
            digest(Path(args.dataset) / "handoff_observations.npz") == report["dataset"]["archive_sha256"], "RAW dataset changed")
    report["checkpoints"] = checkpoints
    report["final_checkpoint"] = str(final_path)
    report["verified_new_updates"] = updates
    report["verified_environment_steps"] = runner.tot_timesteps
    require(runner.tot_timesteps == report["expected_environment_steps"], "Environment step count mismatch")
    report["completion_verified"] = True
    report["actual_yaml"] = [{"path": str(path), "sha256": digest(path)} for path in
                              (destination / "params/env.yaml", destination / "params/agent.yaml")]
    if runner.writer is not None:
        runner.writer.flush()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--input-receipt", type=Path, default=DEFAULT_INPUT_RECEIPT)
    parser.add_argument("--subset-receipt", type=Path, default=DEFAULT_SUBSET_RECEIPT)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--kit_args", default="--/app/vulkan=false")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main():
    configure_runtime()
    args = build_parser().parse_args()
    try:
        arrays, manifest, train_ids, parent, report = preflight(args)
    except Exception as error:
        print(json.dumps({"preflight_pass": False, "simulation_started": False,
                          "error": f"{type(error).__name__}: {error}"}, indent=2), flush=True)
        return 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return 0
    destination = Path(report["run_dir"])
    destination.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(destination / "preflight.json", report)
    resources = {}
    report["completion_verified"] = False
    try:
        run_training(args, arrays, manifest, train_ids, parent, report, resources)
    except BaseException as error:
        report["completion_verified"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    code = 0 if report["completion_verified"] else 1
    report["intended_process_exit_code"] = code
    report["report_written_before_shutdown"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_exclusive(destination / "training_result.json", report)
    print(json.dumps({"report": str(destination / "training_result.json"), "completion_verified": report["completion_verified"],
                      "intended_process_exit_code": code, "promotion_performed": False}, indent=2), flush=True)
    if "traceback" in report:
        print(report["traceback"], file=sys.stderr, flush=True)
    close_resources(resources)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
