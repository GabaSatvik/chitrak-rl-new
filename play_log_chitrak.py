"""Roll out a Chitrak checkpoint headless (no rendering -- Isaac Sim's RTX
renderer hangs on this T4) and log the trajectory to .npz for offline MuJoCo
rendering. See mujoco/replay_to_mp4.py for the consumer."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--out", type=str, default="/tmp/chitrak_traj.npz")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import importlib.metadata as metadata

try:
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        handle_deprecated_rsl_rl_checkpoint,
        handle_deprecated_rsl_rl_cfg,
    )
except ImportError:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    handle_deprecated_rsl_rl_checkpoint = lambda ckpt, ver: ckpt
    handle_deprecated_rsl_rl_cfg = lambda cfg, ver: cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chitrak_isaac  # noqa: F401,E402  -- runs gym.register() via the __init__ chain


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(args_cli.checkpoint, metadata.version("rsl-rl-lib"))
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names

    joint_pos_log, root_pos_log, root_quat_log = [], [], []

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.num_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            # no policy.reset(dones) -- ActorCritic here is a plain MLP (no
            # recurrent state), so there's nothing to reset between episodes.
            joint_pos_log.append(robot.data.joint_pos[0].cpu().numpy().copy())
            root_pos_log.append(robot.data.root_pos_w[0].cpu().numpy().copy())
            root_quat_log.append(robot.data.root_quat_w[0].cpu().numpy().copy())

    np.savez(
        args_cli.out,
        joint_names=np.array(joint_names),
        joint_pos=np.array(joint_pos_log),
        root_pos=np.array(root_pos_log),
        root_quat=np.array(root_quat_log),
    )
    print(f"[INFO]: Saved {len(joint_pos_log)} steps to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
