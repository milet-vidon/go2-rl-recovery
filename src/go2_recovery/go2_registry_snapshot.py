# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

for task_id, config_name in (
    ('Isaac-Natural-Robust-Flat-Unitree-Go2-v0', 'UnitreeGo2NaturalRobustEnvCfg'),
    ('Isaac-Natural-Robust-Flat-Unitree-Go2-Play-v0', 'UnitreeGo2NaturalRobustEnvCfg_PLAY'),
    ('Isaac-Natural-Robust-Push-Flat-Unitree-Go2-v0', 'UnitreeGo2NaturalRobustPushEnvCfg'),
    ('Isaac-Natural-Robust-Push-Flat-Unitree-Go2-Play-v0', 'UnitreeGo2NaturalRobustPushEnvCfg_PLAY'),
):
    gym.register(
        id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}.natural_robust_env_cfg:{config_name}',
                'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NaturalStopPPORunnerCfg'},
    )

for task_id, config_name in (
    ('Isaac-Natural-Stop-Flat-Unitree-Go2-v0', 'UnitreeGo2NaturalStopEnvCfg'),
    ('Isaac-Natural-Stop-Flat-Unitree-Go2-Play-v0', 'UnitreeGo2NaturalStopEnvCfg_PLAY'),
):
    gym.register(
        id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}.natural_stop_env_cfg:{config_name}',
                'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NaturalStopPPORunnerCfg'},
    )

for task_id, config_name in (
    ('Isaac-Natural-Flat-Unitree-Go2-v0', 'UnitreeGo2NaturalEnvCfg'),
    ('Isaac-Natural-Flat-Unitree-Go2-Play-v0', 'UnitreeGo2NaturalEnvCfg_PLAY'),
):
    gym.register(
        id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}.natural_env_cfg:{config_name}',
                'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg'},
    )

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Robust-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_env_cfg:UnitreeGo2RobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Robust-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_env_cfg:UnitreeGo2RobustEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Standard-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standard_env_cfg:UnitreeGo2StandardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Standard-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standard_env_cfg:UnitreeGo2StandardEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Recovery-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Stable-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryStableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Stable-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryStableEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Phased-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryPhasedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Phased-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryPhasedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Lift-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Lift-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Locomotion-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryLocomotionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Locomotion-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeGo2RecoveryLocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RecoveryLocomotionPPORunnerCfg",
    },
)
