from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from chitrak_isaac.robots.chitrak import CHITRAK_CFG
from chitrak_isaac.tasks.velocity.mdp.rewards import base_height_exp


# Direct port of isaaclab_tasks/.../config/go2/rough_env_cfg.py's UnitreeGo2RoughEnvCfg.
# Only body-name regexes are renamed (Chitrak has no separate "base"/"foot" bodies --
# "base_link" and "*_calf_link" instead); every weight/value is Go2's, unchanged.
@configclass
class ChitrakRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CHITRAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.actions.joint_pos.scale = 0.25

        # reverted from the standing-orders-only experiment (rel_standing_envs
        # was temporarily set to 1.0) back to the stock default: only 2% of
        # envs are zero-velocity "standing" envs, the rest sample real
        # locomotion commands -- back to a normal walk/turn task.
        self.commands.base_velocity.rel_standing_envs = 0.02

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_calf_link"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # positive reward for approaching the 0.17m standing height (exp kernel,
        # same shape as the velocity-tracking rewards -- bounded, peaks at the
        # target, doesn't unboundedly reward jumping past it). std=0.15 chosen
        # wide enough that a crouched/fallen robot (height near 0) still gets a
        # meaningful, non-vanishing gradient signal toward standing up, not just
        # very close to the target -- see CHITRAK_VS_SOLO12_COMPARISON.md's
        # torque-margin findings for why "manage to reach 0.17m at all" is
        # itself the hard part for this robot.
        #
        # weight=5.0 (bumped 5x from 1.0): the checkpoint-300 rollout of the
        # weight=1.0 run (video_logs/run7/model_300_traj.npz) showed the robot
        # converging to a genuinely static equilibrium at z=0.1386m -- ~3.1cm
        # short of the 0.17m target (RMS error 0.0311m), not actually at the
        # target. Raising the weight so height-holding matters more relative
        # to the other reward terms, to see if the policy closes that gap.
        self.rewards.base_height_exp = RewTerm(
            func=base_height_exp,
            weight=5.0,
            params={"target_height": 0.17, "std": 0.15},
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class ChitrakRoughEnvCfg_PLAY(ChitrakRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
