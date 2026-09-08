# SPDX-License-Identifier: BSD-3-Clause
"""Command-gated gait rewards to prevent continued trotting at zero command."""
from isaaclab.utils import configclass
from .natural_env_cfg import UnitreeGo2NaturalEnvCfg


@configclass
class UnitreeGo2NaturalStopEnvCfg(UnitreeGo2NaturalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Spot also enables gait rewards above a measured-speed threshold. During
        # long training the policy can exploit this by moving without a command.
        # Infinity disables only that branch: the requested command selects gait
        # versus stance throughout stopping, regardless of residual body velocity.
        for reward in (self.rewards.air_time, self.rewards.gait, self.rewards.joint_posture):
            reward.params['velocity_threshold'] = float('inf')
        self.rewards.static_stance.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -5.0


@configclass
class UnitreeGo2NaturalStopEnvCfg_PLAY(UnitreeGo2NaturalStopEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
