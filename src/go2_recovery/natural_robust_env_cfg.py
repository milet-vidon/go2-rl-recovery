# SPDX-License-Identifier: BSD-3-Clause
"""Bounded push training on the validated command-gated trot task.

Uses Isaac Lab's velocity-reset event. This is not a force-calibrated push,
AMP, or arbitrary-fall recovery. Observation/action layout stays unchanged.
"""
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from .natural_stop_env_cfg import UnitreeGo2NaturalStopEnvCfg


@configclass
class UnitreeGo2NaturalRobustEnvCfg(UnitreeGo2NaturalStopEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
        )


@configclass
class UnitreeGo2NaturalRobustEnvCfg_PLAY(UnitreeGo2NaturalRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class UnitreeGo2NaturalRobustPushEnvCfg(UnitreeGo2NaturalRobustEnvCfg):
    """Push-focused fine-tuning variant kept separate from the baseline task."""

    def __post_init__(self):
        super().__post_init__()
        # The evaluation protocol applies a bounded lateral impulse while the
        # robot is standing, walking, and stopping.  Match that regime without
        # changing observations or the policy architecture.
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.9, 0.9),
            "y": (-0.9, 0.9),
        }
        self.rewards.static_stance.weight = 3.0
        self.rewards.flat_orientation_l2.weight = -7.0


@configclass
class UnitreeGo2NaturalRobustPushEnvCfg_PLAY(UnitreeGo2NaturalRobustPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
