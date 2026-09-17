"""Convert TRAIN handoff reports into a RAW observation collection, NOT a reset bank.

Default is read-only validation. --export creates a new E:-drive directory only.
No simulator, checkpoint deserialization, policy training, clipping of measured
joint states, or legacy recovery-bank loader is involved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

from audit_handoff_report import audit_handoff_report
from audit_recovery_report import AuditError, load_report


SCHEMA = "handoff_observation_collection_v1"
DEFAULT_COLLECTOR = Path(__file__).with_name("evaluate_go2_recovery.py")
VECTOR_FIELDS = {
    "joint_positions_rad": 12, "joint_velocities_rad_s": 12,
    "previous_raw_action": 12, "previous_previous_raw_action": 12,
    "previous_executed_joint_target_rad": 12, "policy_observation": 48,
    "velocity_command_b": 3, "stand_actor_raw_action": 12, "roll_actor_raw_action": 12,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value, shape, name):
    array = np.asarray(value)
    require(array.dtype.kind in "fi" and array.shape == shape, f"{name}: expected numeric shape {shape}")
    array = array.astype(np.float64)
    require(np.isfinite(array).all(), f"{name}: nonfinite values")
    return array


def same(actual, expected, name, tolerance=2e-5):
    require(np.allclose(actual, expected, atol=tolerance, rtol=0), f"{name}: inconsistent recorded values")


def output_path(path):
    path = Path(path).resolve()
    require(path.drive.upper() == "E:", "Output must resolve to the E: drive")
    require(not path.exists(), "Output already exists; refusing overwrite")
    return path


def read_source_bank(bank):
    require(bank.get("split") == "train" and bank.get("diagnostic_train_split_only") is True,
            "Report must explicitly use the TRAIN-only diagnostic split")
    path = Path(bank["path"]).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == bank["sha256"], "Original source bank SHA mismatch")
    manifest_path = Path(bank["manifest_path"]).resolve()
    source_manifest = load_report(manifest_path)
    require(source_manifest.get("sha256") == digest, "Source bank manifest SHA mismatch")
    require(source_manifest.get("schema_version") == "nominal_pd_fallen_v1", "Unexpected source bank schema")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        ids, splits, poses = (archive[key] for key in ("state_id", "split", "pose_class"))
        require(ids.ndim == 1 and ids.dtype.kind in "iu", "Invalid source state_id array")
        require(splits.shape == ids.shape and splits.dtype.kind in "iu", "Invalid source split array")
        require(poses.shape == ids.shape and poses.dtype.kind in "US", "Invalid source pose_class array")
        require(len(set(ids.tolist())) == len(ids), "Duplicate source state IDs")
        require(set(splits.tolist()) == {0, 1}, "Source split encoding must be 0=TRAIN, 1=heldout")
        require(set(poses.tolist()) <= {"left", "right", "back"}, "Unknown source pose classes")
        index = {int(i): (int(split), str(pose)) for i, split, pose in zip(ids, splits, poses)}
    return index, {"path": str(path), "sha256": digest, "manifest_path": str(manifest_path),
                   "manifest_sha256": sha256(manifest_path), "joint_names": source_manifest["joint_names"],
                   "train_state_count": sum(split == 0 for split, _ in index.values())}


def inverse_rotate(quaternion, vector):
    w, xyz = quaternion[0], quaternion[1:]
    return vector - 2 * w * np.cross(xyz, vector) + 2 * np.cross(xyz, np.cross(xyz, vector))


def extract_report(report, source_index):
    """Validate all trials, including untriggered IDs; never alter report values."""
    audit_handoff_report(report)
    bank = report["state_bank"]
    require(bank.get("split") == "train" and bank.get("diagnostic_train_split_only") is True,
            "Heldout/non-TRAIN report cannot become a training collection")
    limits = report["handoff_controller"]["joint_limits"]
    default = numeric(limits["default_joint_positions_rad"], (12,), "default joints")
    soft = numeric(limits["soft_joint_limits_rad"], (12, 2), "soft limits")
    hard = numeric(limits["hard_joint_limits_rad"], (12, 2), "hard limits")
    require(np.all(hard[:, 0] < hard[:, 1]), "Hard limits not ordered")
    require(np.all(soft[:, 0] >= hard[:, 0]) and np.all(soft[:, 1] <= hard[:, 1]),
            "Soft limits must lie inside actual hard limits")
    representation = report["action_representation"]
    require(representation["reference"] == "nominal" and representation["scale"] == .25,
            "Collection requires the recorded nominal + .25 action semantics")
    samples, ledgers = [], []
    for pose, result in report["results"].items():
        require(pose in ("side", "upside_down"), "Unexpected source rollout pose")
        selection = result["state_bank_selection"]
        require(selection["yaw_augmentation_rad"] == 0, "Expected unaugmented source collection")
        records = result["handoff_diagnostic"]["switch_records"]
        trials = []
        for trial, record in enumerate(records):
            source_id = selection["selected_state_ids"][trial]
            require(source_id in source_index, f"Unknown original source state ID {source_id}")
            split, source_pose = source_index[source_id]
            require(split == 0, f"Source state {source_id} is heldout, not TRAIN")
            require(selection["selected_actual_pose_classes"][trial] == source_pose,
                    "Report source pose differs from original bank")
            require(source_pose in (("left", "right") if pose == "side" else ("back",)),
                    "Source pose incompatible with requested rollout")
            require(result["policy_start_state"][trial]["bank_state_id"] == source_id,
                    "Policy-start source ID mismatch")
            ledger = {"trial": trial, "source_train_state_id": source_id,
                      "source_pose_class": source_pose, "triggered": record is not None,
                      "switch_time_s": None if record is None else record["policy_time_s"]}
            trials.append(ledger)
            if record is None:
                continue
            values = {key: numeric(record[key], (size,), key) for key, size in VECTOR_FIELDS.items()}
            position = numeric(record["root_position_local_m"], (3,), "root position")
            quaternion = numeric(record["root_quaternion_wxyz"], (4,), "root quaternion")
            linear = numeric(record["root_linear_velocity_w_m_s"], (3,), "root CoM linear velocity")
            angular = numeric(record["root_angular_velocity_w_rad_s"], (3,), "root CoM angular velocity")
            obs = values["policy_observation"]
            same(obs[:3], inverse_rotate(quaternion, linear), "body linear observation")
            same(obs[3:6], inverse_rotate(quaternion, angular), "body angular observation")
            same(obs[6:9], inverse_rotate(quaternion, np.array([0., 0., -1.])), "gravity observation")
            same(obs[9:12], values["velocity_command_b"], "command observation")
            same(obs[12:24], values["joint_positions_rad"]-default, "joint offset observation")
            same(obs[24:36], values["joint_velocities_rad_s"], "joint velocity observation")
            same(obs[36:48], values["previous_raw_action"], "last raw action observation")
            expected_target = np.clip(default + .25 * values["previous_raw_action"], soft[:, 0], soft[:, 1])
            same(values["previous_executed_joint_target_rad"], expected_target, "actual previous target")
            q = values["joint_positions_rad"]
            # Measure, NEVER clamp the recorded physical joint positions.
            values["soft_limit_excess_rad"] = np.maximum(np.maximum(soft[:, 0]-q, q-soft[:, 1]), 0)
            values["hard_limit_excess_rad"] = np.maximum(np.maximum(hard[:, 0]-q, q-hard[:, 1]), 0)
            samples.append({**values, "root_link_pose_local": np.concatenate((position, quaternion)),
                            "root_com_velocity_w": np.concatenate((linear, angular)),
                            "source_train_state_id": source_id, "source_pose_class": source_pose,
                            "rollout_pose": pose, "source_trial": trial, "source_seed": report["seed"],
                            "switch_time_s": record["policy_time_s"]})
        ledgers.append({"pose": pose, "total_trials": result["trials"],
                        "triggered_trials": sum(row["triggered"] for row in trials),
                        "untriggered_trials": sum(not row["triggered"] for row in trials), "trials": trials})
    return samples, ledgers


def prepare_collection(report_paths, collector_source, collector_sha256):
    collector_source = Path(collector_source).resolve()
    require(re.fullmatch(r"[0-9a-fA-F]{64}", collector_sha256) is not None, "Invalid declared collector SHA")
    require(sha256(collector_source) == collector_sha256.lower(), "Collector source differs from declared pinned SHA")
    all_samples, reports, shared = [], [], None
    seen_paths = set()
    for report_path in report_paths:
        report_path = Path(report_path).resolve()
        require(str(report_path).casefold() not in seen_paths, "Duplicate input report path")
        seen_paths.add(str(report_path).casefold())
        report = load_report(report_path)
        report_digest = sha256(report_path)
        audit_handoff_report(report, verify_checkpoint_files=True)
        source_index, source = read_source_bank(report["state_bank"])
        require(report["joint_names"] == source["joint_names"], "Original source bank joint ordering differs")
        controller = report["handoff_controller"]
        identity = {"joint_names": report["joint_names"], "joint_limits": controller["joint_limits"],
                    "roll_checkpoint": controller["roll_checkpoint"], "roll_sha256": controller["roll_sha256"],
                    "stand_checkpoint": controller["stand_checkpoint"], "stand_sha256": controller["stand_sha256"],
                    "source_bank": source, "action_representation": report["action_representation"],
                    "physics": report["state_bank"]["physics"], "fixed_material": report["state_bank"]["fixed_material"],
                    "self_collisions_enabled": report["self_collisions_enabled"], "gate": controller["gate"]}
        require(shared is None or identity == shared, "Reports disagree on source/model/physics/action/limit identity")
        shared = identity
        samples, ledgers = extract_report(report, source_index)
        for sample in samples:
            sample["source_report_index"] = len(reports)
        all_samples.extend(samples)
        reports.append({"path": str(report_path), "sha256": report_digest, "seed": report["seed"],
                        "recorded_acceptance_eligible": report["acceptance_eligible"],
                        "protocol_version": report["protocol_version"], "pose_ledgers": ledgers})
        require(sha256(report_path) == report_digest, "Input report changed during validation")
    require(all_samples, "No triggered TRAIN observations to export")
    keys = all_samples[0].keys()
    arrays = {key: np.asarray([sample[key] for sample in all_samples]) for key in keys}
    arrays["sample_id"] = np.arange(len(all_samples), dtype=np.int64)
    for name in ("source_train_state_id", "source_trial", "source_seed", "source_report_index"):
        arrays[name] = arrays[name].astype(np.int64)
    for key, value in arrays.items():
        require(value.dtype.kind in "fiUS", f"Unsupported/object array {key}")
        if value.dtype.kind in "fi":
            require(np.isfinite(value).all(), f"Nonfinite export array {key}")
    ledgers = [ledger for report in reports for ledger in report["pose_ledgers"]]
    manifest = {"schema_version": SCHEMA, "status": "raw_observation_collection_not_replay_validated",
                "training_ready": False, "replay_validated": False, "acceptance_eligible": False,
                "legacy_bank_loader_compatible": False,
                "created_utc": datetime.now(timezone.utc).isoformat(), "source_split": "train",
                "num_samples": len(all_samples), "total_source_trials": sum(x["total_trials"] for x in ledgers),
                "untriggered_source_trials": sum(x["untriggered_trials"] for x in ledgers),
                "unique_triggered_source_ids": len(set(arrays["source_train_state_id"].tolist())),
                "source_reports": reports, "provenance": shared,
                "collector_source": {"path": str(collector_source), "sha256": collector_sha256.lower(),
                                     "attestation": "Caller declares capture-time SHA; current source file verified against it. Reports do not embed collector SHA, so source-at-capture identity is not independently proven by report bytes."},
                "exporter_source_sha256": sha256(__file__),
                "numeric_storage": "Recorded JSON numbers preserved as float64; no measured-state clipping or joint reordering.",
                "frames": {"root_link_pose_local": "xyz in world axes minus source environment origin; root LINK world quaternion wxyz",
                           "root_com_velocity_w": "root CoM linear xyz then angular xyz in world axes, m/s and rad/s",
                           "velocity_command_b": "body-frame vx, vy in m/s and yaw rate in rad/s",
                           "joint_positions_rad": "asset joint_names native order; rad",
                           "joint_velocities_rad_s": "asset joint_names native order; rad/s",
                           "previous_raw_action": "a[t-1] at pre-stand-action switch boundary",
                           "previous_previous_raw_action": "a[t-2] at the same boundary"},
                "observation_contract": "48 unnormalized, uncorrupted values: body linear3, angular3, gravity3, command3, q-default12, qd12, previous raw action12; scales=1, default joint velocities=0",
                "replay_limits": ["NOT a training/reset bank; legacy loader must reject this schema.",
                                  "Recompute observations from restored physical state and history before validating input replay; do not substitute stored obs to hide a mismatch.",
                                  "PhysX contact manifolds, friction and solver caches are NOT restored. Matching explicit state/input does not guarantee identical future trajectories.",
                                  "Sensor contacts at replay start are not fresh evidence; no success classification until new physics observations.",
                                  "Soft/hard joint-limit exceedances are measured, not repaired or certified safe; cold replay needs separate validation."],
                "joint_limit_exceedance": {name: {"samples_with_any_excess": int((arrays[name] > 0).any(axis=1).sum()),
                                                   "maximum_rad": float(arrays[name].max())}
                                           for name in ("soft_limit_excess_rad", "hard_limit_excess_rad")},
                "array_shapes": {key: list(value.shape) for key, value in arrays.items()}}
    return arrays, manifest


def export_collection(arrays, manifest, destination):
    destination = output_path(destination)
    # All validation precedes creation. A failed write leaves visible raw artifacts
    # in the new directory; never overwrite or silently delete evidence on retry.
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "handoff_observations.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(archive, allow_pickle=False) as check:
        require(set(check.files) == set(arrays), "Exported NPZ field mismatch")
        for key, value in arrays.items():
            require(np.array_equal(check[key], value), f"Exported NPZ array mismatch: {key}")
    result = dict(manifest, archive_filename=archive.name, archive_sha256=sha256(archive))
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--collector-source", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--collector-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export", action="store_true", help="Create a new RAW NPZ + manifest; default is read-only dry run")
    args = parser.parse_args()
    destination = output_path(args.output_dir)
    arrays, manifest = prepare_collection(args.reports, args.collector_source, args.collector_sha256)
    if args.export:
        manifest = export_collection(arrays, manifest, destination)
    print(json.dumps({"mode": "exported_raw_only" if args.export else "validated_dry_run_no_files_written",
                      "output_dir": str(destination),
                      **{key: manifest[key] for key in ("schema_version", "status", "training_ready", "replay_validated",
                                                       "num_samples", "total_source_trials", "untriggered_source_trials",
                                                       "unique_triggered_source_ids", "joint_limit_exceedance")}}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (AuditError, ValueError, KeyError, TypeError, OSError) as error:
        sys.exit(f"RAW handoff collection refused: {error}")
