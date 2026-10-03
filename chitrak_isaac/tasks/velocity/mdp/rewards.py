from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the asset for approaching a target base height, exponential kernel.

    Same shape as the stock velocity-tracking rewards (exp(-error^2/std^2)):
    positive, bounded in (0, 1], peaks at `target_height`, falls off
    symmetrically on either side -- rewards standing taller as height
    approaches the target without unboundedly rewarding jumping past it
    (unlike a plain `reward = height` term, which has no ceiling).

    Isaac Lab's stock `base_height_l2` is the penalty version of this same
    quantity (squared error, negative weight); this is a positive-reward
    variant of the same underlying error term.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / std**2)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_clearance: float = 0.04,
    std: float = 0.02,
) -> torch.Tensor:
    """Reward foot height during swing phase (when foot is NOT in contact).

    Addresses the core gait-quality problem: the policy learns a shuffle
    (duty factor 91-95%) because feet_air_time with threshold=0.05 is too
    weak to enforce real swing phase.

    For each foot that is currently in swing (not in contact), reward it for
    being above `target_clearance`. Uses an exponential kernel so the reward
    gradient is smooth even far from the target.

    During stance (foot in contact), the term contributes zero -- it only
    pays for clearance when the foot is actually supposed to be in the air.

    Args:
        sensor_cfg:       contact sensor covering the calf links.
        asset_cfg:        robot articulation for reading body positions.
        target_clearance: desired foot height above ground during swing (m).
        std:              kernel width -- how quickly reward falls off below
                          target_clearance. 0.02 m gives ~37% reward at
                          target-std, ~14% at target-2*std.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    # (N, num_feet) bool -- True if foot is currently in contact
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(
        dim=-1
    ).max(dim=1).values > 1.0  # 1 N threshold, same as feet_air_time

    # foot z-positions in world frame: (N, num_feet)
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # height error below target -- negative means foot is too low
    height_error = torch.square(foot_pos_z - target_clearance)
    # reward shape: exp(-err/std^2), peaks at target_clearance
    clearance_reward = torch.exp(-height_error / std**2)

    # Only pay reward when foot is in SWING (not in contact)
    swing_mask = ~in_contact  # (N, num_feet)
    per_foot_reward = clearance_reward * swing_mask.float()

    # Mean over feet so the total doesn't scale with num_feet
    return per_foot_reward.mean(dim=-1)


def hip_roll_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise hip_roll joints deviating from zero.

    L1 (absolute value) instead of L2 so small splay is proportionally
    priced -- L2 lets the first few degrees of splay be nearly free.

    Keeps the stance symmetric and suppresses the back-right hip-roll flare
    that locked in during the baseline walk run between checkpoints 100-200.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # default_joint_pos for hip_roll is 0.0, so deviation = joint_pos directly
    return torch.sum(torch.abs(joint_pos), dim=-1)
