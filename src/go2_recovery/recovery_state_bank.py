"""Validated nominal-PD fallen-state archives and simulator-free root replay math.

Root poses are root-link poses relative to the source environment origin. Root
velocities are CoM linear/angular velocities in world axes, not link velocities.
Loading and validation never modify the archive, simulator, or joint state.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "nominal_pd_fallen_v1"
_MATRIX_KEYS = ("root_pose_local", "root_velocity_w", "joint_pos", "joint_vel")
_LABEL_KEYS = ("pose_class", "requested_pose_class")
_REQUIRED_KEYS = (*_MATRIX_KEYS, *_LABEL_KEYS, "state_id", "split")
_REQUESTED_CLASSES = {"left", "right", "back", "random", "side", "fore_aft", "upside_down", "upright"}


def _names(value, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a nonempty sequence of names")
    if any(not isinstance(name, str) or not name.strip() for name in value) or len(set(value)) != len(value):
        raise ValueError(f"{field} must contain unique nonempty strings")
    return list(value)


def _finite_json(value, field: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_json(item, f"{field}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a nonfinite number")


def validate_recovery_state_bank(
    arrays: Mapping[str, np.ndarray], metadata: dict, joint_names: Sequence[str] | None = None,
    *, archive_sha256: str | None = None,
) -> None:
    """Validate the full archive before selecting either train or heldout data."""
    missing = set(_REQUIRED_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"State bank is missing arrays: {sorted(missing)}")
    if not isinstance(metadata, dict):
        raise ValueError("manifest must be a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if metadata.get("controller") != "nominal_pose_PD" or metadata.get("self_collisions_enabled") is not True:
        raise ValueError("Bank must declare nominal_pose_PD controller and enabled self collisions")
    names = _names(metadata.get("joint_names"), "joint_names")
    _names(metadata.get("body_names"), "body_names")
    if joint_names is not None and names != _names(list(joint_names), "requested joint_names"):
        raise ValueError("Joint names/order do not match the robot; automatic reordering is prohibited")
    replay = metadata.get("replay_validation")
    if not isinstance(replay, dict):
        raise ValueError("replay_validation must record thresholds and statistics")
    for field in ("thresholds", "statistics"):
        if not isinstance(replay.get(field), dict) or not replay[field]:
            raise ValueError(f"replay_validation.{field} must be a nonempty object")
    _finite_json(metadata, "manifest")

    pose = arrays["root_pose_local"]
    if not isinstance(pose, np.ndarray) or pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError("root_pose_local must have shape [N,7]")
    count, joints = pose.shape[0], len(names)
    if count < 2:
        raise ValueError("Bank needs nonempty train and heldout splits")
    shapes = {"root_pose_local": (count, 7), "root_velocity_w": (count, 6),
              "joint_pos": (count, joints), "joint_vel": (count, joints)}
    for key, shape in shapes.items():
        array = arrays[key]
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype.kind != "f":
            raise ValueError(f"{key} must be a floating-point array with shape {shape}")
        if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
            raise ValueError(f"{key} must contain finite float32-representable values")
    norms = np.linalg.norm(pose[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
        raise ValueError("root quaternion must be unit norm within 1e-3 (wxyz convention)")

    labels = {}
    for key in _LABEL_KEYS:
        array = arrays[key]
        if not isinstance(array, np.ndarray) or array.shape != (count,) or array.dtype.kind not in "US":
            raise ValueError(f"{key} must be a string array with shape [{count}] (no object dtype)")
        labels[key] = array.astype(str).tolist()
        if any(not value.strip() for value in labels[key]):
            raise ValueError(f"{key} contains empty labels")
    state_ids = arrays["state_id"]
    if not isinstance(state_ids, np.ndarray) or state_ids.shape != (count,) or state_ids.dtype.kind not in "iu":
        raise ValueError("state_id must be an integer array with shape [N]")
    if np.any(state_ids > np.iinfo(np.int64).max):
        raise ValueError("state_id values must be representable as int64")
    id_values = state_ids.astype(np.int64).tolist()
    if len(set(id_values)) != count:
        raise ValueError("state_id values must be globally unique across both splits")
    if not set(labels["pose_class"]) <= {"left", "right", "back"}:
        raise ValueError("Actual pose_class must be left, right, or back")
    if not set(labels["requested_pose_class"]) <= _REQUESTED_CLASSES:
        raise ValueError("Unrecognized requested_pose_class")
    # Project world gravity (0,0,-1) into the root frame. Yaw does not change it.
    w, x, y, z = pose[:, 3:7].T
    gravity = np.stack((2 * (w * y - x * z), -2 * (y * z + w * x),
                        2 * (x * x + y * y) - 1), axis=1)
    dominant = np.argmax(np.abs(gravity), axis=1)
    actual = np.where((dominant == 1) & (gravity[:, 1] > 0), "left",
             np.where((dominant == 1) & (gravity[:, 1] < 0), "right",
             np.where((dominant == 2) & (gravity[:, 2] > 0), "back", "not_fallen_class")))
    if not np.array_equal(actual, np.asarray(labels["pose_class"])):
        raise ValueError("Actual pose_class disagrees with quaternion/projected-gravity dominant axis")

    split = arrays["split"]
    if not isinstance(split, np.ndarray) or split.shape != (count,) or split.dtype.kind not in "iu":
        raise ValueError("split must be an integer array with shape [N]")
    if set(split.tolist()) != {0, 1}:
        raise ValueError("split must contain both 0=train and 1=heldout, and no other values")
    for key, expected in (("num_states", count), ("num_joints", joints)):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"manifest {key} disagrees with archive")
    if "split_counts" in metadata:
        expected = {"train": int((split == 0).sum()), "heldout": int((split == 1).sum())}
        if metadata["split_counts"] != expected:
            raise ValueError("manifest split_counts disagrees with archive")
    if "states" in replay:
        states = replay["states"]
        if not isinstance(states, list) or len(states) != count:
            raise ValueError("replay_validation.states must contain one record per state")
        if any(not isinstance(state, dict) or "state_id" not in state for state in states):
            raise ValueError("Every replay state record must identify its state_id")
        ids = [state["state_id"] for state in states]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
            raise ValueError("Replay state IDs must be JSON integers")
        if len(set(ids)) != count or set(ids) != set(id_values):
            raise ValueError("Replay state IDs disagree with the archive")
    if "sha256" in metadata:
        expected = metadata["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
            raise ValueError("manifest sha256 must be 64 hexadecimal characters")
        if archive_sha256 is None or expected.lower() != archive_sha256.lower():
            raise ValueError("states.npz SHA-256 does not match manifest")


def load_recovery_state_bank(path, joint_names: Sequence[str], device, split: str = "train") -> dict:
    """Load states.npz + manifest.json, check the whole bank, then select a split.

    Physical arrays become float32 tensors (split/state_id/source_indices int64).
    Pose labels remain lists of strings. Both train and heldout must be present in
    the source even when loading only train. Pickled NumPy arrays are prohibited.
    """
    if split not in ("train", "heldout", "all"):
        raise ValueError("split must be 'train', 'heldout', or 'all'")
    archive = Path(path).expanduser().resolve()
    if archive.is_dir():
        archive = archive / "states.npz"
    if archive.suffix.lower() != ".npz":
        raise ValueError("State bank path must be a directory or .npz archive")
    manifest = archive.parent / "manifest.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    # Hash and decode the SAME bytes to avoid a manifest/archive race.
    import io
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_recovery_state_bank(arrays, metadata, joint_names, archive_sha256=digest)
    selected = np.arange(len(arrays["split"])) if split == "all" else np.flatnonzero(arrays["split"] == (split == "heldout"))
    result = {key: torch.as_tensor(np.ascontiguousarray(arrays[key][selected]), dtype=torch.float32, device=device)
              for key in _MATRIX_KEYS}
    result.update({key: arrays[key][selected].astype(str).tolist() for key in _LABEL_KEYS})
    result.update({"split": torch.as_tensor(arrays["split"][selected].astype(np.int64), device=device),
                   "state_id": torch.as_tensor(arrays["state_id"][selected].astype(np.int64), device=device),
                   "source_indices": torch.as_tensor(selected.astype(np.int64), device=device),
                   "metadata": metadata, "archive_sha256": digest,
                   "archive_path": str(archive), "manifest_path": str(manifest)})
    return result


def validate_bank_joint_limits(bank: Mapping, lower, upper, *, tolerance: float = 1e-5) -> None:
    """Reject out-of-limit states without clipping or modifying them.

    Bounds may be [J], [1,J], or [N,J] and must describe the same joint order.
    This is separate from loading because the target robot owns its soft limits.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("joint limit tolerance must be finite and nonnegative")
    positions = torch.as_tensor(bank["joint_pos"])
    if positions.ndim != 2 or not torch.is_floating_point(positions) or not torch.isfinite(positions).all():
        raise ValueError("joint_pos must be a finite floating [N,J] matrix")
    lo = torch.as_tensor(lower, device=positions.device, dtype=positions.dtype)
    hi = torch.as_tensor(upper, device=positions.device, dtype=positions.dtype)
    allowed_shapes = ((positions.shape[1],), (1, positions.shape[1]), tuple(positions.shape))
    if tuple(lo.shape) not in allowed_shapes or tuple(hi.shape) not in allowed_shapes:
        raise ValueError("Joint bounds must have shape [J], [1,J], or [N,J]")
    if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (lo <= hi).all():
        raise ValueError("Joint bounds must be finite and ordered lower <= upper")
    invalid = (positions < lo - tolerance) | (positions > hi + tolerance)
    if invalid.any():
        state, joint = invalid.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(f"Bank joint position outside robot limits: selected state {state}, joint {joint}")


def transform_bank_root(root_pose_local: torch.Tensor, root_velocity_w: torch.Tensor, origins, yaw):
    """Return new (root_pose_world, CoM_velocity_world) under yaw + translation.

    Origins accept [3] or [N,3]; yaw accepts scalar or [N] radians. Pose quaternions
    are wxyz and left-multiplied by world yaw. Z is translated by origin_z only;
    velocities are rotated, never translated. Inputs and joint states are untouched.
    """
    pose, velocity = root_pose_local, root_velocity_w
    if pose.ndim != 2 or pose.shape[1] != 7 or pose.shape[0] == 0 or not torch.is_floating_point(pose):
        raise ValueError("root_pose_local must be a nonempty floating [N,7] tensor")
    if velocity.shape != (pose.shape[0], 6) or not torch.is_floating_point(velocity):
        raise ValueError("root_velocity_w must be a floating [N,6] tensor")
    if velocity.device != pose.device or velocity.dtype != pose.dtype:
        raise ValueError("Root pose and velocity must share dtype and device")
    if not torch.isfinite(pose).all() or not torch.isfinite(velocity).all():
        raise ValueError("Root poses and velocities must be finite")
    if not torch.allclose(pose[:, 3:].norm(dim=1), torch.ones_like(pose[:, 0]), atol=1e-3, rtol=0):
        raise ValueError("Root quaternion must be unit norm")
    origin = torch.as_tensor(origins, device=pose.device, dtype=pose.dtype)
    angle = torch.as_tensor(yaw, device=pose.device, dtype=pose.dtype)
    if tuple(origin.shape) not in ((3,), (pose.shape[0], 3)) or tuple(angle.shape) not in ((), (pose.shape[0],)):
        raise ValueError("origins must be [3]/[N,3], yaw scalar/[N]")
    if not torch.isfinite(origin).all() or not torch.isfinite(angle).all():
        raise ValueError("Origins and yaw must be finite")
    angle = angle.expand(pose.shape[0])
    cosine, sine = angle.cos(), angle.sin()
    result_pose, result_velocity = pose.clone(), velocity.clone()
    result_pose[:, 0] = cosine * pose[:, 0] - sine * pose[:, 1]
    result_pose[:, 1] = sine * pose[:, 0] + cosine * pose[:, 1]
    result_pose[:, :3] += origin
    c, s = (angle / 2).cos(), (angle / 2).sin()
    w, x, y, z = pose[:, 3:].unbind(1)
    result_pose[:, 3:] = torch.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), dim=1)
    for offset in (0, 3):
        result_velocity[:, offset] = cosine * velocity[:, offset] - sine * velocity[:, offset + 1]
        result_velocity[:, offset + 1] = sine * velocity[:, offset] + cosine * velocity[:, offset + 1]
    return result_pose, result_velocity
