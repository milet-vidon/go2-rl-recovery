"""Isolated stand-preserving TRAIN-handoff experiment, not a registered task.

Keeps Aligned3448's reward design with the fixed Smith diagnostic physics and
nominal, soft-clamped control-step actions. The reset adapter is mandatory;
using this configuration with the ordinary environment would not reset robots.
Nothing here changes a historical task or certifies a dataset for training.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_env_cfg import (
    UnitreeGo2RecoveryAlignedEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.recovery_control_targets import (
    ControlStepJointPositionActionCfg,
)


@configclass
class HandoffStandEnvCfg(UnitreeGo2RecoveryAlignedEnvCfg):
    handoff_fraction: float = 0.20
    handoff_archive_sha256: str = "4546db065dad9127b1591440f1248ae4ebaf50f556af9a5533c6a9b2abc3c1b5"

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        self.episode_length_s = 12.0
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.physics_material.params.update({
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
        })
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.actions.joint_pos = ControlStepJointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], reference="nominal")


@configclass
class HandoffStandEnvCfg_PLAY(HandoffStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
