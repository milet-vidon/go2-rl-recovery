# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 configurations for self-righting and recovery-aware flat-ground locomotion."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from . import recovery_mdp
from .flat_env_cfg import UnitreeGo2FlatEnvCfg


_ZERO_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


@configclass
class UnitreeGo2RecoveryEnvCfg(UnitreeGo2FlatEnvCfg):
    """Learn to self-right on a plane before training commanded locomotion."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.0
        self.scene.sky_light.spawn.texture_file = None
        # Self-righting needs enough joint travel to push against the plane.
        self.actions.joint_pos.scale = 0.25
        self.commands.base_velocity.heading_command = False
        # The command arrow USD is a remote UI asset. Keep the task fully offline
        # and avoid spawning it during training; policy observations are unchanged.
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None

        self.events.reset_base = EventTerm(
            func=recovery_mdp.reset_root_state_mixed,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": _ZERO_VELOCITY_RANGE,
                # upright/side/fore-aft/upside-down/random-SO(3). Recovery starts
                # are deliberately dominated by fallen classes; upright samples
                # keep the policy from forgetting ordinary standing control.
                "pose_probabilities": (0.10, 0.32, 0.32, 0.16, 0.10),
                "upright_height_range": (0.36, 0.42),
                # Clearance added above the rotated conservative robot envelope.
                "fallen_height_range": (0.015, 0.035),
            },
        )
        # Keep the nominal leg shape plus small offsets: random joint poses make
        # fallen starts harder to reproduce and can start outside soft limits.
        self.events.reset_robot_joints.params["position_range"] = (-0.10, 0.10)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Contact with the base is expected during a recovery manoeuvre.
        self.terminations.base_contact = None

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.upright_and_height = RewTerm(
            func=recovery_mdp.upright_and_height_exp,
            weight=3.0,
            # Broad shaping while falling; stable_stand below supplies the strict
            # terminal criterion once the robot is upright and supported.
            params={"target_height": 0.40, "orientation_std": 1.0, "height_std": 0.08},
        )
        self.rewards.orientation_progress = RewTerm(func=recovery_mdp.orientation_progress, weight=1.0)
        self.rewards.recovery_success = RewTerm(
            func=recovery_mdp.recovery_success_exp,
            weight=4.0,
            params={
                "target_height": 0.40,
                "orientation_std": 0.8,
                "height_std": 0.08,
                "max_linear_speed": 0.25,
                "max_angular_speed": 0.80,
            },
        )
        self.rewards.lin_vel_z_l2.weight = -0.10
        self.rewards.ang_vel_xy_l2.weight = -0.005
        self.rewards.dof_torques_l2.weight = -0.00005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_pos_limits.weight = -0.10
        self.rewards.stable_stand = RewTerm(
            func=recovery_mdp.stable_stand_reward,
            weight=5.0,
            params={
                "target_height": 0.40,
                "orientation_std": 0.35,
                "height_std": 0.02,
                "max_linear_speed": 0.35,
                "max_angular_speed": 1.0,
                "contact_force_threshold": 5.0,
                "min_contacts": 2,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.static_stance = RewTerm(
            func=recovery_mdp.static_stance_reward,
            weight=2.0,
            params={
                "target_height": 0.40,
                "orientation_std": 0.25,
                "height_std": 0.018,
                "joint_std": 0.025,
                "max_linear_speed": 0.20,
                "max_angular_speed": 0.60,
            },
        )


@configclass
class UnitreeGo2RecoveryEnvCfg_PLAY(UnitreeGo2RecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RecoveryStableEnvCfg(UnitreeGo2RecoveryEnvCfg):
    """Recovery curriculum calibrated to the Go2's settled physical stance.

    The historical task used a 0.40 m reward target, which is an initialization
    height rather than the learned flat-ground base height (about 0.30 m). This
    variant keeps the old task intact and trains side/fore-aft recovery against
    the measured support height before exposing rare inverted starts.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.num_envs = 1024
        self.events.reset_base.params["pose_probabilities"] = (0.16, 0.38, 0.38, 0.03, 0.05)
        self.events.reset_base.params["upright_height_range"] = (0.34, 0.38)

        self.rewards.upright_and_height.params.update({
            "target_height": 0.31,
            "orientation_std": 0.75,
            "height_std": 0.045,
        })
        self.rewards.recovery_success.params.update({
            "target_height": 0.31,
            "orientation_std": 0.60,
            "height_std": 0.045,
            "max_linear_speed": 0.30,
            "max_angular_speed": 0.90,
        })
        self.rewards.stable_stand.params.update({
            "target_height": 0.31,
            "orientation_std": 0.28,
            "height_std": 0.018,
            "max_linear_speed": 0.30,
            "max_angular_speed": 0.90,
        })
        self.rewards.static_stance.params.update({
            "target_height": 0.31,
            "orientation_std": 0.22,
            "height_std": 0.016,
        })
        self.rewards.upright_and_height.weight = 3.5
        self.rewards.recovery_success.weight = 5.0
        # A valid terminal recovery must be a normal four-foot stand. The base
        # task keeps its historical two-foot shaping; this stable variant uses
        # the stricter support requirement to avoid rewarding low three-foot poses.
        self.rewards.stable_stand.weight = 11.0
        self.rewards.stable_stand.params["min_contacts"] = 4
        self.rewards.static_stance.weight = 4.0


@configclass
class UnitreeGo2RecoveryStableEnvCfg_PLAY(UnitreeGo2RecoveryStableEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RecoveryPhasedEnvCfg(UnitreeGo2RecoveryStableEnvCfg):
    """State-selected recovery phases with a hard normal-height landing target.

    This is an experimental continuation task.  The earlier Stable task remains
    unchanged so its negative controls stay reproducible.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.events.reset_base.params["pose_probabilities"] = (0.20, 0.34, 0.34, 0.06, 0.06)
        self.rewards.recovery_phase = RewTerm(
            func=recovery_mdp.RecoveryPhaseReward,
            weight=4.0,
            params={
                "target_height": 0.31,
                "min_height": 0.30,
                "upright_threshold": 0.25,
                "height_threshold": 0.30,
                "contact_force_threshold": 5.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        # A four-foot, normal-height stand dominates the lower-height local
        # optimum seen in the previous Stable Recovery continuations.
        self.rewards.stable_stand.weight = 15.0
        self.rewards.stable_stand.params["min_height"] = 0.30
        self.rewards.static_stance.weight = 5.0
        self.rewards.static_stance.params["min_height"] = 0.30
        self.rewards.upright_and_height.weight = 2.5
        self.rewards.orientation_progress.weight = 0.5


@configclass
class UnitreeGo2RecoveryPhasedEnvCfg_PLAY(UnitreeGo2RecoveryPhasedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RecoveryLiftEnvCfg(UnitreeGo2RecoveryStableEnvCfg):
    """Gentle continuation from the validated Stable checkpoint.

    Unlike the experimental phased task, this keeps the Stable reward scale and
    adds a narrow lift cue for the low-height, already-supported posture that
    caused recent continuations to stall.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.events.reset_base.params["pose_probabilities"] = (0.18, 0.36, 0.36, 0.05, 0.05)
        self.rewards.low_height_support = RewTerm(
            func=recovery_mdp.low_height_support_penalty,
            weight=-2.0,
            params={
                "min_height": 0.30,
                "orientation_threshold": 0.35,
                "contact_force_threshold": 5.0,
                "min_contacts": 3,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.low_height_lift = RewTerm(
            func=recovery_mdp.low_height_lift_velocity,
            weight=1.5,
            params={
                "min_height": 0.30,
                "orientation_threshold": 0.35,
                "contact_force_threshold": 5.0,
                "min_contacts": 3,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        # Do not constrain curled/rolling recovery.  Once upright and above
        # the floor, bias the joints back toward the normal symmetric Go2
        # stance so the policy cannot finish in a low crouch.
        self.rewards.conditional_stand_posture = RewTerm(
            func=recovery_mdp.conditional_stand_posture,
            weight=2.5,
            params={
                "orientation_threshold": 0.25,
                "min_height": 0.28,
                "target_height": 0.31,
                "joint_std": 0.10,
            },
        )


@configclass
class UnitreeGo2RecoveryLiftEnvCfg_PLAY(UnitreeGo2RecoveryLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


@configclass
class UnitreeGo2RecoveryLocomotionEnvCfg(UnitreeGo2RecoveryEnvCfg):
    """Fine-tune the recovered policy into a smooth, diagonal-trot velocity policy."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.scale = 0.25
        self.episode_length_s = 12.0
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.5
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.events.reset_base.params["pose_probabilities"] = (0.60, 0.16, 0.16, 0.04, 0.04)

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.func = recovery_mdp.track_linear_upright
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.func = recovery_mdp.track_angular_upright
        self.rewards.feet_air_time.weight = 0.20
        self.rewards.feet_air_time.func = recovery_mdp.air_time_upright
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*_foot")
        self.rewards.upright_and_height.weight = 1.5
        self.rewards.recovery_success.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.static_stance.weight = 3.0
        self.rewards.foot_slip = RewTerm(
            func=recovery_mdp.foot_slip_penalty,
            weight=-0.15,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=recovery_mdp.foot_clearance_reward,
            weight=0.12,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": 0.075,
                "std": 0.0025,
                "speed_scale": 8.0,
            },
        )
        self.rewards.trot_when_upright = RewTerm(
            func=recovery_mdp.TrotRewardWhenUpright,
            weight=0.30,
            params={
                "std": 0.10,
                "max_err": 0.20,
                "velocity_threshold": 0.10,
                "upright_threshold": 0.35,
                "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
            },
        )


@configclass
class UnitreeGo2RecoveryLocomotionEnvCfg_PLAY(UnitreeGo2RecoveryLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
