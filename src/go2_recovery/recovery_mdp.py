# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task terms for Go2 self-recovery and recovery-aware locomotion."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from .recovery_math import reset_clearance_height, upright_error_squared

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def reset_root_state_mixed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    pose_probabilities: tuple[float, float, float, float, float],
    upright_height_range: tuple[float, float],
    fallen_height_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset Go2 from a controlled mix of upright and fallen poses.

    The five probabilities correspond to upright/tilted, side, fore-aft, upside-down,
    and random SO(3) poses. A rotated conservative body envelope sets the height
    above the plane. These are controlled drop starts, not settled fallen poses.
    """
    if len(pose_probabilities) != 5 or not math.isclose(sum(pose_probabilities), 1.0, abs_tol=1.0e-6):
        raise ValueError("pose_probabilities must contain five values that sum to 1.0.")

    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = asset.device
    dtype = asset.data.default_root_state.dtype

    samples = torch.rand(num_envs, device=device)
    # Curriculum: learn stable standing and small disturbances first, then expose
    # progressively more severe falls. The final distribution is exactly the one
    # supplied by the task config; this only changes early exploration.
    progress = min(float(getattr(env, "common_step_counter", 0)) / 1_000_000.0, 1.0)
    easy = torch.tensor((0.70, 0.15, 0.10, 0.03, 0.02), device=device, dtype=dtype)
    hard_values = pose_probabilities
    if os.getenv("ISAACLAB_RECOVERY_FOCUS_SIDE"):
        hard_values = (0.08, 0.68, 0.12, 0.08, 0.04)
    hard = torch.tensor(hard_values, device=device, dtype=dtype)
    active_probabilities = easy * (1.0 - progress) + hard * progress
    active_probabilities = active_probabilities / active_probabilities.sum()
    boundaries = active_probabilities.cumsum(dim=0)
    pose_class = torch.bucketize(samples, boundaries, right=False)

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(num_envs, device=device, dtype=dtype)

    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    x = uniform(*x_range)
    y = uniform(*y_range)
    roll = torch.zeros(num_envs, device=device, dtype=dtype)
    pitch = torch.zeros_like(roll)
    yaw = uniform(*pose_range.get("yaw", (-math.pi, math.pi)))
    root_height = uniform(*fallen_height_range)

    upright = pose_class == 0
    side = pose_class == 1
    fore_aft = pose_class == 2
    upside_down = pose_class == 3
    random_orientation = pose_class == 4

    # Near-upright starts include light roll/pitch disturbances.
    roll[upright] = uniform(-0.25, 0.25)[upright]
    pitch[upright] = uniform(-0.25, 0.25)[upright]
    root_height[upright] = uniform(*upright_height_range)[upright]

    # Side and fore-aft falls cover both directions with a modest perturbation.
    side_sign = torch.where(torch.rand(num_envs, device=device) < 0.5, -1.0, 1.0)
    fore_aft_sign = torch.where(torch.rand(num_envs, device=device) < 0.5, -1.0, 1.0)
    # Optional staged training cap.  Keeping this in the reset term allows a
    # checkpoint to learn support from shallow tilts before full side falls.
    angle_cap = os.getenv("ISAACLAB_RECOVERY_FALL_ANGLE_DEG")
    if angle_cap:
        try:
            fall_angle = math.radians(float(angle_cap))
        except ValueError:
            fall_angle = math.pi / 2.0
    else:
        fall_angle = 0.35 + progress * (math.pi / 2.0 - 0.35)
    fall_angle = max(0.35, min(fall_angle, math.pi / 2.0))
    roll[side] = (side_sign * fall_angle + uniform(-0.20, 0.20))[side]
    pitch[side] = uniform(-0.20, 0.20)[side]
    roll[fore_aft] = uniform(-0.20, 0.20)[fore_aft]
    pitch[fore_aft] = (fore_aft_sign * fall_angle + uniform(-0.20, 0.20))[fore_aft]

    # A roll near pi produces a back-down orientation without privileging one side.
    upside_angle = min(math.pi, 2.0 * fall_angle)
    roll[upside_down] = (side_sign * upside_angle + uniform(-0.20, 0.20))[upside_down]
    pitch[upside_down] = uniform(-0.20, 0.20)[upside_down]

    orientations = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    if torch.any(random_orientation):
        orientations[random_orientation] = math_utils.random_orientation(
            int(random_orientation.sum().item()), device=device
        ).to(dtype=dtype)

    # A fixed low root height can put the legs or nose below the plane. Use the
    # support function of the rotated default-pose envelope for every orientation.
    clearance_height = reset_clearance_height(
        math_utils.matrix_from_quat(orientations), uniform(*fallen_height_range)
    )
    root_height = torch.where(upright, root_height, clearance_height)
    positions = torch.stack((x, y, root_height), dim=-1) + env.scene.env_origins[env_ids]
    velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys], device=device, dtype=dtype
    )
    velocities = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (num_envs, 6), device=device
    )

    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def upright_and_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    orientation_std: float,
    height_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a stable, standing base with dense errors around the target pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    orientation_error = upright_error_squared(asset.data.projected_gravity_b)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - target_height)
    return torch.exp(-orientation_error / orientation_std - height_error / height_std)


def recovery_success_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    orientation_std: float,
    height_std: float,
    max_linear_speed: float,
    max_angular_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense stable-posture shaping, NOT a measured recovery success rate.

    Repeated rewards and discounting favor earlier recovery. No episode-time gate
    is used, because training can randomize episode counters on the first rollout.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    orientation_error = upright_error_squared(asset.data.projected_gravity_b)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - target_height)
    planar_speed = torch.linalg.norm(asset.data.root_lin_vel_b, dim=1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)
    posture = torch.exp(-orientation_error / orientation_std - height_error / height_std)
    stability = torch.exp(
        -torch.square(planar_speed / max_linear_speed) - torch.square(angular_speed / max_angular_speed)
    )
    return posture * stability


def stable_stand_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    orientation_std: float,
    height_std: float,
    max_linear_speed: float,
    max_angular_speed: float,
    contact_force_threshold: float,
    min_contacts: int,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense terminal-like reward for a supported, quiet upright stand."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    orientation_error = upright_error_squared(asset.data.projected_gravity_b)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - target_height)
    posture = torch.exp(-orientation_error / orientation_std - height_error / height_std)
    linear_speed = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_w, dim=1)
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(dim=1)
    supported = (forces > contact_force_threshold).sum(dim=1) >= min_contacts
    quiet = torch.exp(
        -torch.square(linear_speed / max_linear_speed) - torch.square(angular_speed / max_angular_speed)
    )
    return posture * quiet * supported


def static_stance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    orientation_std: float,
    height_std: float,
    joint_std: float,
    max_linear_speed: float,
    max_angular_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep zero-command behavior close to the symmetric nominal Go2 stance."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
    orientation_error = upright_error_squared(asset.data.projected_gravity_b)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - target_height)
    joint_error = torch.square(asset.data.joint_pos - asset.data.default_joint_pos).mean(dim=1)
    linear_speed = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_w, dim=1)
    posture = torch.exp(-orientation_error / orientation_std - height_error / height_std)
    nominal = torch.exp(-joint_error / joint_std)
    quiet = torch.exp(-torch.square(linear_speed / max_linear_speed) - torch.square(angular_speed / max_angular_speed))
    return (command < 0.05) * posture * nominal * quiet


def orientation_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Broad shaping across side and back orientations, with no inversion alias."""
    return (0.5 - 0.5 * env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)


def locomotion_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset = env.scene["robot"]
    return (asset.data.projected_gravity_b[:, 2] < -0.90) & (
        asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] > 0.25
    )


def track_linear_upright(env, std, command_name, asset_cfg=SceneEntityCfg("robot")):
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import track_lin_vel_xy_yaw_frame_exp
    return locomotion_gate(env) * track_lin_vel_xy_yaw_frame_exp(env, std, command_name, asset_cfg)


def track_angular_upright(env, std, command_name, asset_cfg=SceneEntityCfg("robot")):
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import track_ang_vel_z_world_exp
    return locomotion_gate(env) * track_ang_vel_z_world_exp(env, command_name, std, asset_cfg)


def air_time_upright(env, command_name, sensor_cfg, threshold):
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_air_time
    return locomotion_gate(env) * feet_air_time(env, command_name, sensor_cfg, threshold)


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize planar foot velocity while a foot carries contact force."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    in_contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold
    planar_foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    return locomotion_gate(env) * torch.sum(in_contact * planar_foot_speed, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    speed_scale: float,
) -> torch.Tensor:
    """Reward moving feet at swing height only during upright commanded motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3] - target_height
    )
    planar_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    moving = torch.tanh(speed_scale * planar_speed)
    command = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    return locomotion_gate(env) * (command > 0.1) * (moving * torch.exp(-foot_height_error / std)).mean(dim=1)


class TrotRewardWhenUpright(ManagerTermBase):
    """Reward diagonal trot timing only after recovery and while moving.

    A diagonal pair should have matching contact/air timers, while legs across the
    pairs should alternate.  The upright gate keeps this locomotion prior out of the
    self-righting manoeuvre.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or any(len(pair) != 2 for pair in pair_names):
            raise ValueError("TrotRewardWhenUpright requires exactly two diagonal foot pairs.")
        self.synced_feet_pairs = [self.contact_sensor.find_bodies(pair)[0] for pair in pair_names]
        if not self.contact_sensor.cfg.track_air_time or any(len(pair) != 2 for pair in self.synced_feet_pairs):
            raise ValueError("Trot reward requires four resolved feet and track_air_time=True.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        upright_threshold: float,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del synced_feet_pair_names, sensor_cfg
        first_pair, second_pair = self.synced_feet_pairs
        sync = self._sync(first_pair[0], first_pair[1], std, max_err) * self._sync(
            second_pair[0], second_pair[1], std, max_err
        )
        alternating = (
            self._alternating(first_pair[0], second_pair[0], std, max_err)
            * self._alternating(first_pair[1], second_pair[1], std, max_err)
            * self._alternating(first_pair[0], second_pair[1], std, max_err)
            * self._alternating(first_pair[1], second_pair[0], std, max_err)
        )
        command = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
        asset: Articulation = env.scene["robot"]
        upright_error = torch.sqrt(upright_error_squared(asset.data.projected_gravity_b))
        active = (command > velocity_threshold) & (upright_error < upright_threshold) & locomotion_gate(env)
        return torch.where(active, sync * alternating, 0.0)

    def _sync(self, foot_a: int, foot_b: int, std: float, max_err: float) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        error = torch.square(air_time[:, foot_a] - air_time[:, foot_b]) + torch.square(
            contact_time[:, foot_a] - contact_time[:, foot_b]
        )
        return torch.exp(-torch.clamp(error, max=max_err**2) / std)

    def _alternating(self, foot_a: int, foot_b: int, std: float, max_err: float) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        error = torch.square(air_time[:, foot_a] - contact_time[:, foot_b]) + torch.square(
            contact_time[:, foot_a] - air_time[:, foot_b]
        )
        return torch.exp(-torch.clamp(error, max=max_err**2) / std)
