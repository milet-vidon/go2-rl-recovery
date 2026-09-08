"""Robust Go2 locomotion with perturbation training and quiet standing."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from . import recovery_mdp
from .standard_env_cfg import UnitreeGo2StandardEnvCfg, UnitreeGo2StandardEnvCfg_PLAY


@configclass
class UnitreeGo2RobustEnvCfg(UnitreeGo2StandardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Frequent bounded pushes train recovery without making the policy
        # chase impossible impulses. This follows the robust locomotion setup
        # used in legged RL and AMP-style sim-to-real work.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
        )
        self.events.base_external_force_torque = None
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.rewards.static_stance.weight = 3.5
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -0.00025


@configclass
class UnitreeGo2RobustEnvCfg_PLAY(UnitreeGo2RobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
