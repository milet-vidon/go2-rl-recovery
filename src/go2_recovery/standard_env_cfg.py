"""Go2 standard locomotion with an explicit quiet standing objective."""

from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeGo2FlatEnvCfg, UnitreeGo2FlatEnvCfg_PLAY
from . import recovery_mdp


@configclass
class UnitreeGo2StandardEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        self.actions.joint_pos.scale = 0.25
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.rel_heading_envs = 0.5
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.feet_air_time.weight = 0.12
        self.rewards.action_rate_l2.weight = -0.04
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.static_stance = RewTermCfg(
            func=recovery_mdp.static_stance_reward,
            weight=3.0,
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
class UnitreeGo2StandardEnvCfg_PLAY(UnitreeGo2StandardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
