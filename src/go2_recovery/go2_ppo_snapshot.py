# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class UnitreeGo2NaturalStopPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.desired_kl = 0.005
        self.algorithm.entropy_coef = 0.005


@configclass
class UnitreeGo2RecoveryPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Longer horizon and slower discounting for self-righting trajectories."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_go2_recovery"
        self.algorithm.gamma = 0.995
        self.algorithm.learning_rate = 5.0e-4


@configclass
class UnitreeGo2RecoveryLocomotionPPORunnerCfg(UnitreeGo2RecoveryPPORunnerCfg):
    """Uses the recovery experiment root so RSL-RL can resume its checkpoint directly."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 4000
