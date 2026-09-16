"""Subset-safe reset from a physically collected, cold-replay-validated fallen bank."""

from __future__ import annotations

import math
import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import math as math_utils

from .recovery_math import reset_clearance_height
from .recovery_state_bank import load_recovery_state_bank, transform_bank_root, validate_bank_joint_limits


class RecoveryBankReset(ManagerTermBase):
    """Own root AND joint reset; never advance physics or modify other environments.

    The task must disable its later reset_robot_joints term. Collection mode is
    explicit and does not load a bank. Ordinary training fails if no bank exists.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        if env.cfg.events.reset_robot_joints is not None:
            raise ValueError("RecoveryBankReset requires reset_robot_joints=None; later writes corrupt bank geometry")
        self.bank = None
        self.class_indices = []
        self.last_state_ids = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        if not cfg.params["collection_mode"]:
            self.bank = load_recovery_state_bank(cfg.params["bank_path"], self.robot.joint_names, env.device, split="train")
            limits = self.robot.data.soft_joint_pos_limits[0]
            validate_bank_joint_limits(self.bank, limits[:, 0], limits[:, 1])
            physics = self.bank["metadata"].get("physics", {})
            if "dt" in physics and not math.isclose(float(physics["dt"]), env.physics_dt, abs_tol=1e-9):
                raise ValueError("Bank physics dt differs from target environment")
            if "decimation" in physics and int(physics["decimation"]) != env.cfg.decimation:
                raise ValueError("Bank control decimation differs from target environment")
            for label in ("left", "right", "back"):
                indices = [i for i, value in enumerate(self.bank["pose_class"]) if value == label]
                if not indices:
                    raise ValueError(f"Training bank has no validated {label} states")
                self.class_indices.append(torch.tensor(indices, dtype=torch.long, device=env.device))

    def __call__(self, env, env_ids, bank_path: str, collection_mode: bool,
                 bank_fraction_start: float, bank_fraction_end: float, curriculum_steps: int):
        if not 0 <= bank_fraction_start <= bank_fraction_end <= 1:
            raise ValueError("Bank fractions must satisfy 0 <= start <= end <= 1")
        ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
               else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
        if ids.numel() == 0:
            return
        n = ids.numel()
        dtype = self.robot.data.default_joint_pos.dtype

        def uniform(low, high, shape):
            return low + (high - low) * torch.rand(shape, device=env.device, dtype=dtype)

        # Familiar upright / shallow-drop rehearsal; this new task owns the
        # joint writes too. It does not change the historical reset RNG path.
        joint_pos = self.robot.data.default_joint_pos[ids].clone()
        joint_pos += uniform(-0.1, 0.1, joint_pos.shape)
        limits = self.robot.data.soft_joint_pos_limits[ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        joint_vel = torch.zeros_like(joint_pos)
        classes = uniform(0, 1, (n,))
        upright, side = classes < 0.6, (classes >= 0.6) & (classes < 0.8)
        fore = classes >= 0.8
        roll, pitch = uniform(-0.15, 0.15, (n,)), uniform(-0.15, 0.15, (n,))
        signs = torch.where(uniform(0, 1, (n,)) < 0.5, -1., 1.)
        angle = torch.deg2rad(uniform(20, 30, (n,)))
        roll[side] = (signs * angle)[side]
        pitch[fore] = (signs * angle)[fore]
        quat = math_utils.quat_from_euler_xyz(roll, pitch, uniform(-math.pi, math.pi, (n,)))
        height = reset_clearance_height(math_utils.matrix_from_quat(quat), uniform(0.015, 0.035, (n,)))
        height = torch.where(upright, uniform(0.34, 0.38, (n,)), height)
        local_position = torch.stack((uniform(-0.4, 0.4, (n,)), uniform(-0.4, 0.4, (n,)), height), dim=-1)
        root_pose = torch.cat((local_position + env.scene.env_origins[ids], quat), dim=-1)
        root_velocity = torch.zeros((n, 6), device=env.device, dtype=dtype)
        self.last_state_ids[ids] = -1

        if not collection_mode:
            if self.bank is None:
                raise RuntimeError("Cannot enable bank sampling after collection-only construction")
            progress = min(1., max(0., float(env.common_step_counter) / max(1, curriculum_steps)))
            fraction = bank_fraction_start + progress * (bank_fraction_end - bank_fraction_start)
            mask = uniform(0, 1, (n,)) < fraction
            k = int(mask.sum())
            if k:
                # Balance actual side-left/side-right/back poses, not collection labels.
                chosen_class = torch.randint(3, (k,), device=env.device)
                selected = torch.empty(k, device=env.device, dtype=torch.long)
                for c, candidates in enumerate(self.class_indices):
                    subset = chosen_class == c
                    selected[subset] = candidates[torch.randint(candidates.numel(), (int(subset.sum()),), device=env.device)]
                bank_pose, bank_velocity = transform_bank_root(
                    self.bank["root_pose_local"][selected], self.bank["root_velocity_w"][selected],
                    env.scene.env_origins[ids[mask]], uniform(-math.pi, math.pi, (k,)),
                )
                root_pose[mask], root_velocity[mask] = bank_pose, bank_velocity
                joint_pos[mask], joint_vel[mask] = self.bank["joint_pos"][selected], self.bank["joint_vel"][selected]
                self.last_state_ids[ids[mask]] = self.bank["state_id"][selected]

        # No sim.step, sim.forward, scene.update or global history reset here.
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=ids)
