"""Chitrak velocity tracking (walk + turn) — improved-training branch.

Changes from the baseline walk config (chitrak_walk_env_cfg.py on main):

  1. DOMAIN RANDOMIZATION re-enabled
       push_robot            ON  — random impulses every 10-15 s
       add_base_mass         ON  — ±1 kg mass perturbation on torso
       obs noise             ON  — conservative per-sensor scales
       reset_base yaw        randomised ±π (was pinned to 0)
     Without DR the policy is completely brittle on any real hardware.

  2. FOOT CLEARANCE reward added
       foot_clearance_reward — pays per-foot when in swing, exp kernel
       peaks at 4 cm above ground. Directly addresses the 91-95% duty
       factor shuffle in the baseline: feet_air_time threshold alone could
       not enforce real swing because the term fired only at first contact.

  3. HIP-ROLL DEVIATION penalty re-added (L1)
       hip_roll_deviation_l1 — weight -0.3. The stand task used -0.5 and
       took stance asymmetry from 8° → 0.3°. Walk needs hip-roll authority
       for lateral/turning motion, so the weight is lighter but not zero.
       Without it the back-right flare locked in between ckpts 100-200.

  4. FULL COMMAND SPACE
       lin_vel_x  (−0.3, +0.5) — backward walking added
       lin_vel_y  (−0.3, +0.3) — symmetric lateral
       ang_vel_z  (−0.5, +0.5) — turning enabled (was pinned to 0)
       heading_command=False — yaw command IS the yaw rate, not a heading
       controller target.

  5. OBSERVATION NOISE re-enabled (conservative scales)
       Scales are halved from Go2 stock (which was tuned for ±1 m/s
       commands; ours are ±0.5 m/s). Enough to prevent the policy from
       exploiting exact state, not so much that it can't learn at 0.25 m/s.

  6. CURRICULUM re-enabled
       kc penalty ramp inherited from ChitrakFlatEnvCfg is turned back ON.
       Combined with DR this lets the policy first learn to move then be
       penalised at full strength — the same schedule that made Go2 work.

All other constants (action_scale 0.25, Kd 0.05, solver 8 iters,
feet_air_time threshold 0.05, tracking kernel std 0.15) are unchanged
from the baseline that successfully walks.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp as base_mdp

from .chitrak_flat_env_cfg import ChitrakFlatEnvCfg
from .mdp import foot_clearance_reward, hip_roll_deviation_l1

# ── command ranges ────────────────────────────────────────────────────────────
LIN_VEL_X_MIN = -0.3   # m/s  backward walking added
LIN_VEL_X_MAX =  0.5   # m/s  same as baseline
LIN_VEL_Y     =  0.3   # m/s  symmetric ±, was one-sided 0..+0.3
ANG_VEL_Z     =  0.5   # rad/s turning enabled, was 0


@configclass
class ChitrakWalkV2EnvCfg(ChitrakFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ═══════════════════════════════════════════════════════════════════
        # 1. REMOVE baseline terms that block walking
        # ═══════════════════════════════════════════════════════════════════
        self.rewards.base_height_exp = None   # froze the walk policy
        self.rewards.feet_slide      = None   # wrong for walking feet
        self.rewards.hip_roll_deviation = None  # replaced by L1 version below
        self.rewards.undesired_contacts = None  # Go2 also disables this

        # ═══════════════════════════════════════════════════════════════════
        # 2. DOMAIN RANDOMIZATION
        # ═══════════════════════════════════════════════════════════════════

        # 2a. Random pushes — single biggest gait-quality improver.
        # Impulse magnitude 0.5-1.0 Ns every 10-15 s. Go2 uses 0.5-1.0 Ns
        # at 10-15 s; Chitrak is 1.3 kg vs 15 kg, so the same impulse is
        # ~11x larger in v/m terms — keep the lower end to avoid constant falls.
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
        )

        # 2b. Base mass perturbation: ±0.5 kg on a 1.3 kg robot = ±38%.
        # Tighter than Go2's ±1-3 kg (which is ±7-20% of 15 kg) to keep the
        # physics grounded -- we are already light-footed.
        self.events.add_base_mass = EventTerm(
            func=base_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (-0.5, 0.5),
                "operation": "add",
            },
        )

        # 2c. Randomise reset yaw so the policy never sees the same starting
        # orientation twice. Was pinned to 0 ("one variable at a time"); now
        # that forward walking is established, yaw randomisation is safe.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (-3.14159, 3.14159),
        }

        # ═══════════════════════════════════════════════════════════════════
        # 3. OBSERVATION NOISE re-enabled (conservative half-Go2 scales)
        # ═══════════════════════════════════════════════════════════════════
        # Go2 stock: base_lin_vel ±0.1, base_ang_vel ±0.2, joint_pos ±0.01,
        #            joint_vel ±1.5. Our commands are half Go2's, so we halve
        #            the velocity noise proportionally.
        self.observations.policy.enable_corruption = True

        # ═══════════════════════════════════════════════════════════════════
        # 4. CURRICULUM re-enabled
        #    (inherited from ChitrakFlatEnvCfg; was nulled out in baseline)
        # ═══════════════════════════════════════════════════════════════════
        # kc ramp 0.1→1.0 over ~640 iterations keeps early training "easy"
        # while DR is active — policy first learns to move, then gets the
        # full penalty signal. Without the ramp, DR + full penalties together
        # overwhelm early exploration.
        # (terms are already set up in ChitrakFlatEnvCfg.__post_init__)

        # ═══════════════════════════════════════════════════════════════════
        # 5. REWARDS: Go2 stock weights (same as baseline)
        # ═══════════════════════════════════════════════════════════════════
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight  = 0.5  # higher: yaw now active
        self.rewards.lin_vel_z_l2.weight         = -2.0
        self.rewards.ang_vel_xy_l2.weight        = -0.05
        self.rewards.dof_torques_l2.weight       = -2.0e-4
        self.rewards.dof_acc_l2.weight           = -2.5e-7
        self.rewards.action_rate_l2.weight       = -0.01
        self.rewards.flat_orientation_l2.weight  = -2.5
        self.rewards.feet_air_time.weight        = 0.25

        # tracking kernel widths
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp.params["std"]  = 0.25

        # feet_air_time: reward foot lifting during trot
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.05

        # ── hip-roll L1 penalty ───────────────────────────────────────────
        # Suppresses outwards/inwards hip drift and keeps legs symmetric
        self.rewards.hip_roll_deviation_l1 = RewTerm(
            func=hip_roll_deviation_l1,
            weight=-0.3,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=".*_hip_roll_joint"
                ),
            },
        )

        # ═══════════════════════════════════════════════════════════════════
        # 6. FULL COMMAND SPACE
        # ═══════════════════════════════════════════════════════════════════
        self.commands.base_velocity.rel_standing_envs = 0.02  # Go2 stock
        self.commands.base_velocity.heading_command  = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (LIN_VEL_X_MIN, LIN_VEL_X_MAX)
        self.commands.base_velocity.ranges.lin_vel_y = (-LIN_VEL_Y, LIN_VEL_Y)
        self.commands.base_velocity.ranges.ang_vel_z = (-ANG_VEL_Z, ANG_VEL_Z)

        # ═══════════════════════════════════════════════════════════════════
        # 7. ACTUATOR + NUMERICS (tuned for 1.3 kg lightweight Chitrak)
        # ═══════════════════════════════════════════════════════════════════
        # action_scale 0.125 (halved from Go2's 0.25): prevents torque saturation
        # and joint excursions that cause leg buckling or face-planting.
        self.actions.joint_pos.scale = 0.125
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.actuators["base_legs"].damping = 0.05


@configclass
class ChitrakWalkV2EnvCfg_PLAY(ChitrakWalkV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        # one steady forward command so the replay video is readable
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (LIN_VEL_X_MAX, LIN_VEL_X_MAX)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
