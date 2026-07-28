import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_flat_env_cfg:ChitrakFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_flat_env_cfg:ChitrakFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_stand_env_cfg:ChitrakStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakStandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-Stand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_stand_env_cfg:ChitrakStandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakStandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_walk_env_cfg:ChitrakWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakWalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-Walk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_walk_env_cfg:ChitrakWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakWalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Chitrak-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_rough_env_cfg:ChitrakRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Chitrak-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_rough_env_cfg:ChitrakRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakRoughPPORunnerCfg",
    },
)
