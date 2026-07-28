from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .chitrak_rough_env_cfg import ChitrakRoughEnvCfg
from .mdp.curriculums import kc_scaled_weight


def _kc_curriculum_term(term_name: str, base_weight: float) -> CurrTerm:
    """CurrTerm that ramps `rewards.<term_name>.weight` by soloRL's kc factor.

    See tasks/velocity/mdp/curriculums.py and CHITRAK_VS_SOLO12_COMPARISON.md
    sec 8/10/11 -- soloRL scales every negative reward term by one shared,
    training-progress-dependent scalar (0.1 -> 1.0); this reproduces that using
    Isaac Lab's generic modify_term_cfg runtime-parameter curriculum mechanism.
    """
    return CurrTerm(
        func=base_mdp.modify_term_cfg,
        params={
            "address": f"rewards.{term_name}.weight",
            "modify_fn": kc_scaled_weight,
            "modify_params": {"base_weight": base_weight},
        },
    )


# Direct port of isaaclab_tasks/.../config/go2/flat_env_cfg.py's UnitreeGo2FlatEnvCfg.
@configclass
class ChitrakFlatEnvCfg(ChitrakRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # this T4 has no Nucleus-cached actuator/material assets and the GPU
        # tensor view isn't ready before the first sim step -- these three
        # startup/reset events deadlock headless here. not present in Go2's
        # own stock flat cfg; required on this hardware regardless of robot.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None

        # soloRL-style kc reward curriculum (0.1 -> 1.0, see mdp/curriculums.py):
        # ramps every currently-active negative reward term together, starting
        # "easy" and increasing to full penalty strength as training progresses.
        # base_weight per term = this file's own final configured weight for it
        # (i.e. the value it would have with no curriculum at all).
        self.curriculum.lin_vel_z_l2_kc = _kc_curriculum_term("lin_vel_z_l2", -2.0)
        self.curriculum.ang_vel_xy_l2_kc = _kc_curriculum_term("ang_vel_xy_l2", -0.05)
        self.curriculum.dof_torques_l2_kc = _kc_curriculum_term("dof_torques_l2", -0.0002)
        self.curriculum.dof_acc_l2_kc = _kc_curriculum_term("dof_acc_l2", -2.5e-7)
        self.curriculum.action_rate_l2_kc = _kc_curriculum_term("action_rate_l2", -0.01)
        self.curriculum.flat_orientation_l2_kc = _kc_curriculum_term("flat_orientation_l2", -2.5)


@configclass
class ChitrakFlatEnvCfg_PLAY(ChitrakFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
