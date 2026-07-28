# Custom curriculum terms for Chitrak, mirroring soloRL's `kc` reward curriculum
# (see CHITRAK_VS_SOLO12_COMPARISON.md, secs 8/10/11).
#
# soloRL ramps a single scalar kc from 0.1 -> 1.0 over training and multiplies
# every negative reward term's weight by it, starting "easy" and increasing
# difficulty as the policy improves (Environment.hpp: curriculumUpdate(),
# cfg.yaml: reward_curriculum_increment=0.0003). Isaac Lab's RewardManager has
# no built-in shared-multiplier concept (each RewTerm has an independent static
# weight), but the generic isaaclab.envs.mdp.curriculums.modify_term_cfg
# mechanism is well-suited to building this: register one CurrTerm per
# negative reward term, all sharing this one modify_fn, each carrying its own
# pre-curriculum "full-strength" base_weight as a param.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def kc_scaled_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: float,
    base_weight: float,
    increment: float = 0.0014,
    num_steps_per_env: int = 24,
) -> float:
    """Scale a reward term's weight by a soloRL-style curriculum factor kc.

    kc ramps 0.1 -> 1.0 linearly in units of PPO "iterations"
    (env.common_step_counter // num_steps_per_env matches RSL-RL's own
    iteration unit, and raisimGymTorch's curriculumUpdate() semantics -- one
    increment per policy-update cycle, not per rollout step).

    Default increment (0.0014) is tuned for a ~1000-iteration run: reaches
    kc=1.0 around iteration ~640, leaving the back third of training at full
    penalty strength. soloRL's own literal increment (0.0003) targets a much
    longer ~3000-iteration horizon and would leave kc at only ~0.4 by
    iteration 1000 -- rescaled here to fit the shorter run, not copied as-is.
    """
    iteration = env.common_step_counter // num_steps_per_env
    kc = min(1.0, 0.1 + increment * iteration)
    return base_weight * kc
