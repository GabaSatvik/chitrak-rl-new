from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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
