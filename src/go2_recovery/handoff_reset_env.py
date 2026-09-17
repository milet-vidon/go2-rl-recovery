"""Unregistered, opt-in post-manager-reset handoff adapter; NOT a ready bank.

The future launcher must verify files and real replay receipts before supplying
the tensors/evidence below. This module reads no files and trusts that explicit
attestation. It never upgrades RAW data or proves future trajectory equivalence.
Use validation_only=True for isolated reset tests until all receipt checks exist;
such instances expose handoff_training_permitted=False and MUST NOT be trained.

The fixed mixture is 80% ordinary shallow upright *training drops* and 20% TRAIN
handoffs in expectation per reset draw, not integer quotas for tiny subsets.
The ordinary branch is NOT already-settled standing. Neither branch advances
physics or writes observations here; normal outer environment flow owns that.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import re

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .handoff_action_restore import restore_handoff_action_history


@dataclass(frozen=True)
class HandoffResetData:
    """Launcher-attested tensors, already on the environment dtype/device.

    Frozen fields do not make tensors immutable; the environment clones them.
    Receipt booleans certify inputs/subset isolation/finiteness only, NEVER full
    trajectory equality or recovery ability. No tensor belongs in the env cfg.
    """

    archive_sha256: str
    receipt_sha256: str
    source_split: str
    input_replay_verified: bool
    subset_reset_verified: bool
    contact_probe_finite: bool
    joint_names: tuple[str, ...]
    allowed_train_state_ids: torch.Tensor
    source_train_state_id: torch.Tensor
    root_link_pose_local: torch.Tensor
    root_com_velocity_w: torch.Tensor
    joint_positions_rad: torch.Tensor
    joint_velocities_rad_s: torch.Tensor
    previous_raw_action: torch.Tensor
    previous_previous_raw_action: torch.Tensor
    previous_executed_joint_target_rad: torch.Tensor
    velocity_command_b: torch.Tensor
    default_joint_positions_rad: torch.Tensor
    soft_joint_limits_rad: torch.Tensor


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_handoff_reset_data(data, *, validation_only=False):
    """Pure tensor/metadata checks; does NOT authenticate an external receipt."""
    _require(isinstance(data, HandoffResetData), "Explicit HandoffResetData is required; RAW data is not ready")
    _require(type(validation_only) is bool, "validation_only must be boolean")
    for name in ("archive_sha256", "receipt_sha256"):
        value = getattr(data, name)
        _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value), f"Invalid {name}")
    _require(data.source_split == "train", "Only TRAIN handoff sources are supported")
    flags = (data.input_replay_verified, data.subset_reset_verified, data.contact_probe_finite)
    _require(all(type(value) is bool for value in flags), "Receipt flags must be booleans")
    _require(validation_only or all(flags), "Training requires verified input replay, subset reset and finite contact probe")
    _require(len(data.joint_names) == 12 and len(set(data.joint_names)) == 12 and
             all(isinstance(name, str) and name for name in data.joint_names), "Require 12 unique native joint names")
    q = data.joint_positions_rad
    _require(isinstance(q, torch.Tensor) and q.ndim == 2 and q.shape[1] == 12 and q.shape[0] > 0, "Invalid joint-position tensor")
    _require(q.dtype in (torch.float32, torch.float64), "Unsupported floating-point dtype")
    count = q.shape[0]
    shapes = {
        "root_link_pose_local": (count, 7), "root_com_velocity_w": (count, 6),
        "joint_positions_rad": (count, 12), "joint_velocities_rad_s": (count, 12),
        "previous_raw_action": (count, 12), "previous_previous_raw_action": (count, 12),
        "previous_executed_joint_target_rad": (count, 12), "velocity_command_b": (count, 3),
        "default_joint_positions_rad": (12,), "soft_joint_limits_rad": (12, 2),
    }
    for name, shape in shapes.items():
        value = getattr(data, name)
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and
                 value.device == q.device and value.dtype == q.dtype and not value.requires_grad and
                 bool(torch.isfinite(value).all()), f"Invalid tensor {name}")
    for name in ("allowed_train_state_ids", "source_train_state_id"):
        value = getattr(data, name)
        _require(isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() > 0 and
                 value.dtype == torch.long and value.device == q.device and bool((value >= 0).all()) and
                 value.unique().numel() == value.numel(), f"Invalid TRAIN ID list {name}")
    _require(data.source_train_state_id.numel() == count and
             bool(torch.isin(data.source_train_state_id, data.allowed_train_state_ids).all()), "Source IDs are not the attested TRAIN subset")
    _require(bool((data.velocity_command_b == 0).all()), "Only zero-command handoff states are supported")
    quat_norm = torch.linalg.vector_norm(data.root_link_pose_local[:, 3:], dim=-1)
    _require(torch.allclose(quat_norm, torch.ones_like(quat_norm), rtol=0, atol=1e-5), "Source quaternion is not unit length")
    limits = data.soft_joint_limits_rad
    nominal = data.default_joint_positions_rad
    _require(bool((limits[:, 0] < limits[:, 1]).all()), "Invalid soft limits")
    # Ordinary q +/- .1 must fit unchanged; reject rather than silently clipping.
    _require(bool(((nominal - .1 >= limits[:, 0]) & (nominal + .1 <= limits[:, 1])).all()), "Nominal upright jitter does not fit soft limits")
    expected = torch.clamp(nominal + .25 * data.previous_raw_action, min=limits[:, 0], max=limits[:, 1])
    executed = data.previous_executed_joint_target_rad
    _require(bool(((executed >= limits[:, 0]) & (executed <= limits[:, 1])).all()) and
             torch.allclose(executed, expected, rtol=0, atol=1e-6), "Executed source target contradicts nominal+.25 action semantics")
    # Measured joint_positions_rad can exceed limits slightly: preserve them.


def _validate_cfg(cfg, data):
    _require(getattr(cfg, "handoff_fraction", None) == .2, "cfg.handoff_fraction must explicitly be 0.2")
    archive = getattr(cfg, "handoff_archive_sha256", None)
    _require(isinstance(archive, str) and archive.lower() == data.archive_sha256.lower(), "cfg/archive SHA mismatch")
    events = cfg.events
    _require(hasattr(events, "reset_base") and events.reset_base is None and
             hasattr(events, "reset_robot_joints") and events.reset_robot_joints is None,
             "Disable both legacy reset_base and reset_robot_joints before constructing this adapter")
    for value in vars(events).values():
        func = getattr(value, "func", None)
        _require(not getattr(func, "__module__", "").endswith(".recovery_bank_mdp"), "Legacy bank events are forbidden")
    command = cfg.commands.base_velocity
    _require(command.heading_command is False and command.rel_standing_envs == 1., "Require zero-command standing task")
    _require(all(tuple(getattr(command.ranges, name)) == (0., 0.) for name in
                 ("lin_vel_x", "lin_vel_y", "ang_vel_z")), "Moving command ranges are unsupported")


def _quaternion_xyz(roll, pitch, yaw):
    """Same wxyz Euler convention as Isaac Lab quat_from_euler_xyz."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                        cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy), dim=-1)


def sample_handoff_reset_batch(data, env_origins, *, generator=None):
    """Pure subset-sized sampler. No file, simulator, manager or state writes."""
    n = env_origins.shape[0]
    dtype, device = data.joint_positions_rad.dtype, data.joint_positions_rad.device
    _require(env_origins.shape == (n, 3) and env_origins.dtype == dtype and env_origins.device == device and
             bool(torch.isfinite(env_origins).all()), "Invalid selected environment origins")

    def uniform(low, high, shape):
        return low + (high-low) * torch.rand(shape, dtype=dtype, device=device, generator=generator)

    handoff = uniform(0, 1, (n,)) < .2
    q = data.default_joint_positions_rad.expand(n, -1).clone() + uniform(-.1, .1, (n, 12))
    qd = torch.zeros_like(q)
    xyz = torch.zeros((n, 3), dtype=dtype, device=device)
    xyz[:, 2] = uniform(.34, .38, (n,))
    quat = _quaternion_xyz(uniform(-.15, .15, (n,)), uniform(-.15, .15, (n,)), uniform(-math.pi, math.pi, (n,)))
    pose = torch.cat((xyz, quat), dim=-1)
    velocity = torch.zeros((n, 6), dtype=dtype, device=device)
    action, previous = torch.zeros_like(q), torch.zeros_like(q)
    target = data.default_joint_positions_rad.expand(n, -1).clone()
    sample_ids = torch.full((n,), -1, device=device, dtype=torch.long)
    source_ids = torch.full_like(sample_ids, -1)
    k = int(handoff.sum())
    if k:
        chosen = torch.randint(data.joint_positions_rad.shape[0], (k,), device=device, generator=generator)
        sample_ids[handoff] = chosen
        source_ids[handoff] = data.source_train_state_id[chosen]
        for destination, source in (
            (pose, data.root_link_pose_local), (velocity, data.root_com_velocity_w),
            (q, data.joint_positions_rad), (qd, data.joint_velocities_rad_s),
            (action, data.previous_raw_action), (previous, data.previous_previous_raw_action),
            (target, data.previous_executed_joint_target_rad),
        ):
            destination[handoff] = source[chosen]
    pose[:, :3] += env_origins  # Translation only; preserve source world-axis velocity and quaternion.
    return dict(root_link_pose_w=pose, root_com_velocity_w=velocity, joint_pos=q, joint_vel=qd,
                previous_raw_action=action, previous_previous_raw_action=previous,
                previous_executed_joint_target_rad=target, handoff=handoff,
                sample_ids=sample_ids, source_train_state_ids=source_ids)


class HandoffResetEnv(ManagerBasedRLEnv):
    """Opt-in adapter: super reset FIRST, then selected physical/history writes.

    Instantiate directly, never through a historical registry entry:
        HandoffResetEnv(cfg, validated_handoff=data, validation_only=True)
    cfg contains only handoff_archive_sha256/handoff_fraction metadata, not data.
    A future training launcher MUST check handoff_training_permitted and verify
    the actual receipt files itself. This class cannot authenticate them.
    """

    def __init__(self, cfg, validated_handoff=None, *, validation_only=False, **kwargs):
        validate_handoff_reset_data(validated_handoff, validation_only=validation_only)
        _validate_cfg(cfg, validated_handoff)
        copied = {field.name: getattr(validated_handoff, field.name).clone() for field in fields(validated_handoff)
                  if isinstance(getattr(validated_handoff, field.name), torch.Tensor)}
        self._handoff_data = replace(validated_handoff, **copied)
        self.handoff_training_permitted = not validation_only
        self.handoff_validation_only = validation_only
        super().__init__(cfg, **kwargs)
        self._validate_runtime()
        self.last_reset_was_handoff = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_handoff_sample_ids = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.last_handoff_source_ids = torch.full_like(self.last_handoff_sample_ids, -1)

    def _validate_runtime(self):
        data, asset = self._handoff_data, self.scene["robot"]
        q = asset.data.default_joint_pos
        _require(tuple(asset.joint_names) == tuple(data.joint_names), "Runtime native joint order mismatch")
        _require(q.shape == (self.num_envs, 12) and q.device == data.joint_positions_rad.device and
                 q.dtype == data.joint_positions_rad.dtype, "Runtime data dtype/device/shape mismatch")
        _require(torch.equal(q, data.default_joint_positions_rad.expand_as(q)), "Runtime nominal q differs from source")
        _require(torch.equal(asset.data.soft_joint_pos_limits, data.soft_joint_limits_rad.expand(self.num_envs, -1, -1)), "Runtime soft limits differ from source")
        empty = torch.empty((0, 12), device=q.device, dtype=q.dtype)
        restore_handoff_action_history(self.action_manager, asset, torch.empty(0, device=q.device, dtype=torch.long),
            previous_raw_action=empty, previous_previous_raw_action=empty,
            previous_executed_joint_target_rad=empty, joint_names=data.joint_names)
        command = self.command_manager.get_term("base_velocity")
        _require(command.vel_command_b.shape == (self.num_envs, 3) and command.vel_command_b.device == q.device and
                 command.vel_command_b.dtype == q.dtype, "Invalid command buffer")
        for name in ("is_standing_env", "is_heading_env"):
            value = getattr(command, name)
            _require(value.shape == (self.num_envs,) and value.dtype == torch.bool and value.device == q.device, "Invalid command flags")

    def _reset_idx(self, env_ids):
        _validate_cfg(self.cfg, self._handoff_data)
        self._validate_runtime()
        ids = torch.as_tensor(env_ids, device=self.scene["robot"].data.default_joint_pos.device)
        _require(ids.ndim == 1 and ids.dtype == torch.long and ids.unique().numel() == ids.numel() and
                 bool(((ids >= 0) & (ids < self.num_envs)).all()), "Reset IDs must be unique in-range int64")
        if not ids.numel():
            return
        batch = sample_handoff_reset_batch(self._handoff_data, self.scene.env_origins[ids])
        super()._reset_idx(ids)  # Resets action/command/observation managers BEFORE restoration.
        asset = self.scene["robot"]
        asset.write_joint_state_to_sim(batch["joint_pos"], batch["joint_vel"], env_ids=ids)
        asset.write_root_link_pose_to_sim(batch["root_link_pose_w"], env_ids=ids)
        asset.write_root_com_velocity_to_sim(batch["root_com_velocity_w"], env_ids=ids)
        restore_handoff_action_history(self.action_manager, asset, ids,
            previous_raw_action=batch["previous_raw_action"],
            previous_previous_raw_action=batch["previous_previous_raw_action"],
            previous_executed_joint_target_rad=batch["previous_executed_joint_target_rad"],
            joint_names=self._handoff_data.joint_names)
        command = self.command_manager.get_term("base_velocity")
        command.vel_command_b[ids] = 0.
        command.is_standing_env[ids] = True
        command.is_heading_env[ids] = False
        self.last_reset_was_handoff[ids] = batch["handoff"]
        self.last_handoff_sample_ids[ids] = batch["sample_ids"]
        self.last_handoff_source_ids[ids] = batch["source_train_state_ids"]
