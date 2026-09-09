# SPDX-License-Identifier: BSD-3-Clause
"""Go2 standing and trot, adapting Isaac Lab's upstream Spot gait rewards.

Separate from the historical standard/robust tasks to preserve reproducibility.
This is PPO with explicit rewards, not adversarial motion imitation (AMP).
"""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp
from .standard_env_cfg import UnitreeGo2StandardEnvCfg


@configclass
class UnitreeGo2NaturalEnvCfg(UnitreeGo2StandardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.sky_light.spawn.texture_file = None
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.30
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.rewards.static_stance.params['target_height'] = 0.30
        self.rewards.static_stance.weight = 1.0
        self.rewards.feet_air_time = None
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.dof_torques_l2.weight = -0.00015
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.base_height = RewTerm(func=mdp.base_height_l2, weight=-30.0, params={'target_height': 0.30})
        self.rewards.air_time = RewTerm(func=spot_mdp.air_time_reward, weight=2.0, params={
            'mode_time': 0.35, 'velocity_threshold': 0.1, 'asset_cfg': SceneEntityCfg('robot'),
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot')})
        self.rewards.gait = RewTerm(func=spot_mdp.GaitReward, weight=2.0, params={
            'std': 0.1, 'max_err': 0.2, 'velocity_threshold': 0.1,
            'synced_feet_pair_names': (('FL_foot', 'RR_foot'), ('FR_foot', 'RL_foot')),
            'asset_cfg': SceneEntityCfg('robot'), 'sensor_cfg': SceneEntityCfg('contact_forces')})
        self.rewards.foot_slip = RewTerm(func=spot_mdp.foot_slip_penalty, weight=-0.5, params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*_foot'),
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot'), 'threshold': 1.0})
        self.rewards.foot_clearance = RewTerm(func=spot_mdp.foot_clearance_reward, weight=0.25, params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*_foot'),
            'target_height': 0.075, 'std': 0.01, 'tanh_mult': 2.0})
        self.rewards.joint_posture = RewTerm(func=spot_mdp.joint_position_penalty, weight=-0.2, params={
            'asset_cfg': SceneEntityCfg('robot'), 'stand_still_scale': 5.0, 'velocity_threshold': 0.1})


@configclass
class UnitreeGo2NaturalEnvCfg_PLAY(UnitreeGo2NaturalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
