"""Chitrak velocity tracking (walk + turn) -- Go2's STOCK reward structure,
rescaled to Chitrak. No curriculum, no extra reward terms, plain commands.

WHY THIS IS A REWRITE. The previous version of this file inherited the standing
task's reward set, and the result did not walk -- it learned to stand perfectly
still and ignore the velocity command entirely. Measured from
video_logs/run18/model_100_traj.npz under a steady 0.2 m/s forward command:

    net displacement 0.0070 m over 6.0 s  ->  0.0012 m/s  (commanded 0.2)
    mean joint range 0.0412 rad           ->  limbs essentially frozen
    height std 0.52 mm                    ->  a statue

The reward arithmetic made that optimal. base_height_exp paid 3.75 out of a
possible 4.0 for doing nothing at all, while full velocity tracking was worth
only 2.0 and was being realised at 0.77 -- and any actual movement additionally
paid dof_acc_l2, action_rate_l2 and feet_slide, plus risked a fall (32%
termination). Standing still strictly dominated walking.

Go2's flat config, which does produce walking, has NO base-height term at all.
So this file goes back to that structure. Every term below is Go2's stock value
unless there is a physical or geometric reason to rescale it, and each such
rescale is stated.
"""

from isaaclab.utils import configclass

from .chitrak_flat_env_cfg import ChitrakFlatEnvCfg

# Commands. Go2's stock range is +-1.0 m/s. The range below is instead
# derived from the JOINT VELOCITY LIMIT (8 rad/s, enforced as a hard cap in PhysX
# via articulation.py's write_joint_velocity_limit_to_sim), not picked by feel.
#
# Full stride-cycle analysis -- 2-link IK around the loop, Jacobian pseudo-inverse
# at each point, including the 0.03 m swing clearance -- gives the peak required
# joint speed as a fraction of the 8 rad/s cap:
#
#     v (m/s)   stride   step freq   max|w| stance   max|w| swing   % of cap
#      0.15     0.10 m     1.5 Hz         0.95           1.69          21%
#      0.25     0.10 m     2.5 Hz         1.58           2.82          35%
#      0.50     0.10 m     5.0 Hz         3.17           5.64          70%
#      1.00     0.10 m    10.0 Hz         6.33          11.28         141%  SATURATED
#
# Swing is always the binding phase. Note the naive w = (v/r)*D/(1-D) estimate
# UNDERSTATES the real requirement by ~2.5x, because it ignores the vertical lift
# component and the Jacobian degrading at the ends of the sweep -- so 1.0 m/s is
# not merely tight, it is impossible, and 0.50 m/s sits at 70% of the cap with
# little margin left once a real motor's torque-speed derating is considered.
# Commanding a velocity the joints cannot reach is just another way to build a
# flat, uninformative reward -- the same failure mode as target_height=0.17.
#
# 0.15-0.25 m/s runs at 21-35% of the cap, with real headroom.
#
# FORWARD-ONLY, and deliberately NOT through zero. Sampling uniformly through
# zero was half the reason the policy froze: at std=0.25 a robot standing still
# under a 0.1 m/s command still collected 85% of the tracking ceiling, and
# uniform sampling kept generating those small commands. With the range bounded
# away from zero, standing still is never near-optimal.
LIN_VEL_X_MIN = 0.0  # m/s  (2026-07-28: widened 0.15 -> 0.0 by request; everything
# else identical to the best run 2026-07-27_19-58-09, the Go2-originals config.)
LIN_VEL_X_MAX = 0.5  # m/s  (2026-07-28: 0.25 -> 0.5 -> 0.3 -> 0.5; span 0.0-0.5)
LIN_VEL_Y = 0.3  # m/s  (2026-07-28: lateral axis on, one-sided 0.0-0.3)
# Foot turning radius is sqrt(0.1034^2 + 0.076^2) = 0.128 m, so 0.5 rad/s is only
# ~0.064 m/s of foot speed -- the same conservative band as the linear range.
# Go2's stock +-1.0 would be ~0.128 m/s, borderline defensible, but there is no
# reason to spend the margin.
ANG_VEL_Z = 0.5  # rad/s


@configclass
class ChitrakWalkEnvCfg(ChitrakFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ================= REMOVE the terms that froze it =================
        # base_height_exp is NOT part of Go2's reward set. It was added here long
        # ago when a spawn bug meant the robot could not get off the floor; that
        # bug is fixed (the init pose now lands the feet +0.5 mm above ground), so
        # the term has no remaining justification -- and it is the single reason
        # the previous run stood still. Height is maintained implicitly by
        # flat_orientation_l2 plus the gait itself, exactly as on Go2.
        self.rewards.base_height_exp = None

        # feet_slide penalizes foot velocity WHILE IN CONTACT. That is right for
        # holding a static stance and wrong for walking: during stance a walking
        # foot is necessarily moving backwards relative to the body, so this term
        # taxes the stride directly. Not in Go2's set. Removed.
        self.rewards.feet_slide = None

        # hip_roll_deviation pins hip_roll to 0. It was an anti-splay term for
        # standing; turning and lateral motion both need hip_roll authority.
        # Not in Go2's set. Removed.
        self.rewards.hip_roll_deviation = None

        # ================= NO CURRICULUM, per request =================
        # The soloRL-style kc ramp (every negative weight scaled 0.1 -> 1.0 over
        # training) is inherited from the flat cfg. Off: it would also silently
        # overwrite the weights set below on every step.
        self.curriculum.lin_vel_z_l2_kc = None
        self.curriculum.ang_vel_xy_l2_kc = None
        self.curriculum.dof_torques_l2_kc = None
        self.curriculum.dof_acc_l2_kc = None
        self.curriculum.action_rate_l2_kc = None
        self.curriculum.flat_orientation_l2_kc = None
        self.curriculum.terrain_levels = None  # already None for flat; explicit

        # ================= REWARDS: Go2 stock values =================
        # weights straight from Go2's rough+flat config
        # track_lin_vel 1.5 -> 2.5 and track_ang_vel 0.75 -> 0.25, measured from
        # run 2026-07-26_15-55-29 at iteration 245, where the policy idled:
        #
        #   track_ang_vel_z_exp   +0.3209  = 43% of its 0.75 ceiling
        #   track_lin_vel_xy_exp  +0.0914  =  6% of its 1.5 ceiling
        #   all penalties          -0.05   = 12% of positives
        #
        # 78% of every unit of reward the policy earned came from
        # track_ang_vel_z_exp paying it for NOT ROTATING, while the actual task
        # contributed 6%. Penalties were NOT the problem (12%, not the 142% seen
        # earlier) -- a robot that does nothing incurs nothing. The free lunch was
        # the angular term. Shrinking it and enlarging the linear term makes
        # forward motion unambiguously the dominant payoff.
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        # ---- ANTI-ROCKING: the only change from BASE (git tag walk-base) ----
        # BASE stands upright and ROCKS BACK AND FORTH. Rocking makes base velocity
        # oscillate through the commanded 0.15-0.25 m/s, and track_lin_vel_xy_exp is
        # evaluated INSTANTANEOUSLY each step, so it collects 50.2% of the tracking
        # ceiling with ZERO net displacement (error_vel_xy 0.3946, worse than a
        # statue's 0.20). Mean reward 29.24 vs 11.9 for standing still.
        #
        # A full-path term-by-term comparison against a Go2 flat reference run on
        # this machine shows nothing is HOGGING -- the opposite. Each penalty as a
        # share of that run's own positive reward:
        #
        #                    Chitrak(BASE)   Go2    gap
        #   ang_vel_xy_l2        0.6%        3.1%   5.2x   <- taxes the rocking itself
        #   lin_vel_z_l2         0.2%        1.1%   5.5x   <- taxes vertical bobbing
        #   action_rate_l2       0.2%        3.1%  15.5x
        #   dof_acc_l2           1.2%        2.9%   2.4x
        #   TOTAL                2.7%       16.3%
        #
        # Go2 STARTS penalty-dominated (203% of positives at iteration 25) and grows
        # out of it monotonically. Chitrak is never penalized at all, so rocking is
        # free. Raise the two terms that directly price it, to Go2's realized share.
        #
        # action_rate_l2 and dof_acc_l2 deliberately NOT raised despite larger gaps:
        # both tax FAST MOTION, which is what we are trying to elicit, and Go2's high
        # shares there come from a robot that is genuinely walking.
        # REVERTED to BASE (Go2 stock). Tried lin_vel_z_l2=-11.0 and
        # ang_vel_xy_l2=-0.26 (5.5x / 5.2x, to match Go2's realized penalty share):
        # the robot STILL rocked, just more slowly. Pricing the oscillation does not
        # remove the exploit -- it only makes the policy buy less of it -- because
        # track_lin_vel_xy_exp is still evaluated INSTANTANEOUSLY and a slow rock
        # still sweeps the commanded velocity band. The exploit is structural to an
        # instantaneous velocity kernel, not a question of penalty magnitude.
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01  # was -0.05 (standing tweak)
        self.rewards.flat_orientation_l2.weight = -2.5  # was -10.0 (standing tweak)
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.undesired_contacts = None  # Go2 disables this too

        # --- the two reward params that MUST be rescaled, with reasons ---
        # (1) tracking kernel width. Go2 pairs std=sqrt(0.25)=0.5 with a +-1.0 m/s
        #     range, i.e. std/range = 0.5. Carrying std=0.5 over to this much
        #     smaller range would make the kernel nearly FLAT, which is precisely
        #     how the policy learned to freeze: commands here are 0.15-0.25 m/s, so
        #     a robot standing perfectly still has error ~0.2, and at std=0.5 that
        #     still pays exp(-0.04/0.25) = 85% of the tracking ceiling. Doing
        #     nothing was worth 85% of doing the task.
        #     std=0.1 was tried and OVER-corrected: it stopped the statue being
        #     paid, but it also flattened the reward for partial progress, turning
        #     the term into a step function with no slope out of standing still.
        #     Measured at a 0.2 m/s command:
        #
        #        tracking error   std=0.10   std=0.15
        #             0.20 (statue)   1.8%      17%
        #             0.15           10.5%      37%
        #             0.10           36.8%      64%
        #             0.05           77.9%      90%
        #
        #     A robot that begins to shuffle at 0.05 m/s collected 10% of the term
        #     at std=0.1 -- effectively nothing, so there was no gradient to climb.
        #     std=0.15 restores a real slope in the "starting to move" region while
        #     still paying a statue only 17%.
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        #     ang: the yaw command is now pinned to 0 (one axis at a time), so this
        #     term acts purely as a mild anti-spin penalty in reward clothing. std
        #     kept at 0.25; its weight is what was reduced, above.
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # (2) feet_air_time threshold. The reward is
        #         (last_air_time - threshold) * first_contact
        #     so any step with less flight time than the threshold scores
        #     NEGATIVE. Go2's 0.5 s suits its long slow stride. Chitrak is a
        #     0.234 m-tall robot whose natural stride frequency is much higher
        #     (period ~ sqrt(L/g)); at ~0.4 m/s with a ~0.1 m stride it steps at
        #     ~3 Hz, giving ~0.15 s of flight per step at 50% duty. Go2's 0.5 s
        #     would actively PUNISH Chitrak's natural stride. Confirmed
        #     empirically: at threshold=0.2 this term was still reading NEGATIVE
        #     (-0.019) in the previous run.
        #     0.15 was still too high: measured at -0.0054 in the idling run, i.e.
        #     the term was PENALIZING the few short contacts that happened rather
        #     than paying for them. Since it is the one term that directly rewards
        #     lifting a foot -- the specific behaviour that is missing -- it must be
        #     positive for a real stride. 0.05 s of flight is ~1/4 of the swing
        #     phase at a 2.5 Hz gait, so anything resembling a step now pays.
        self.rewards.feet_air_time.params["threshold"] = 0.05

        # ================= COMMANDS: plain, no heading controller =================
        self.commands.base_velocity.rel_standing_envs = 0.02  # Go2 stock
        # heading_command=True (Go2 stock) OVERWRITES ang_vel_z with an internal
        # heading controller driven by ranges.heading (+-pi), so ranges.ang_vel_z
        # is silently ignored. "Basic commands" means the yaw command IS the yaw
        # command, so this is off.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (LIN_VEL_X_MIN, LIN_VEL_X_MAX)
        # one-sided 0.0..+LIN_VEL_Y, not +-: same reasoning as lin_vel_x's lower
        # bound -- a symmetric range makes the mean command zero, which a frozen
        # robot collects reward for under the instantaneous tracking kernel.
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, LIN_VEL_Y)
        # yaw pinned to zero for now -- one new axis at a time, same reasoning as
        # lin_vel_y. A uniform +-0.5 range meant the average |command| was 0.25, at
        # which a frozen robot collected 37% of the angular term for free; that was
        # 78% of all reward it earned. Turning comes back once forward walking works.
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ================= ALL NOISE OFF (deliberate, temporary) =================
        # Observation corruption: this disables every Unoise term in the policy
        # observation group at once. Those scales were Go2's and were never
        # rescaled -- base_lin_vel +-0.1 m/s and base_ang_vel +-0.2 rad/s against
        # commands of +-0.5 m/s / +-1.0 rad/s, and joint_vel +-1.5 rad/s against
        # an 8 rad/s limit (19% of full scale). While standing still the true base
        # velocity is ~0, so that input was PURE noise. The validated PID reads
        # exact state; giving the policy the same clean feedback removes a variable
        # rather than adding one.
        self.observations.policy.enable_corruption = False

        # Reset randomization: spawn deterministically. Domain randomization is
        # already fully off upstream (physics_material, add_base_mass,
        # base_external_force_torque, base_com and push_robot are all None), so
        # this pose jitter was the last remaining source of per-episode
        # variability. reset_robot_joints is already (1.0, 1.0), i.e. the exact
        # default pose.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # ================= kept from the working stand setup =================
        # action_scale back to Go2's 0.25 (the stand task halved it to 0.125 to
        # gentle the foot accelerations; walking needs the joint authority).
        # action_scale 0.25 -> 0.125. THE ONLY CHANGE from the previous run.
        #
        # Derived from a like-for-like trend comparison against a Go2 flat run on
        # this same machine (300 iterations, 4096 envs). The single large, robust
        # difference between the two robots is the FALL RATE during exploration:
        #
        #     Go2     peak base_contact = 0.109  at iteration 43
        #     Chitrak peak base_contact = 0.915  at iteration 44
        #
        # Both robots explore. Go2 explores and mostly stays upright, accumulates
        # tracking reward, and climbs out of the penalty-dominated early phase
        # (penalties 394% -> 16% of positives over 300 iterations). Chitrak
        # explores and face-plants, so every attempt truncates the episode and it
        # correctly learns that moving is catastrophic -- track_lin plateaus at
        # ~20% of ceiling from iteration 100 through 500 and never moves again.
        #
        # The reward weights are NOT the cause. Three independent checks:
        #   - realized penalty share: Chitrak 16.4% of positives vs Go2's 20.9%,
        #     i.e. Chitrak is already LESS penalized than the robot that walks
        #   - dof_acc_l2 compared at MATCHED fall rates is 0.87x / 1.06x / 2.55x
        #     Go2's, not the 60x seen at iteration 50 -- that spike was caused by
        #     falling (impact accelerations), not by commanding motion
        #   - dof_acc per unit action_rate is 15x Go2's, matching the physics
        #     prediction of 16x (alpha = Kp*da/J, J=0.0015 vs 0.05), but the
        #     absolute penalty is unaffected because Chitrak's action_rate is 20x
        #     smaller -- the two effects cancel
        #
        # So the fix has to target STABILITY DURING EARLY MOTION, not the reward.
        # Halving action_scale halves the joint excursion per step, which is
        # exactly what made the STANDING task work. Prediction to falsify: peak
        # base_contact should drop well below 0.9.
        #
        # Side effect to keep in mind: this also halves exploration in joint space
        # (init_noise_std 0.2 * 0.125 = +-0.025 rad), so if the fall rate drops but
        # tracking still plateaus, init_noise_std needs raising to compensate.
        # GO2 ORIGINAL for this run (was 0.125). See rsl_rl_ppo_cfg.py.
        self.actions.joint_pos.scale = 0.25
        # 8 solver position iterations is a light-foot CONTACT-NUMERICS fix, not a
        # reward shaping choice, so it stays. effort_limit stays the real 2.5 Nm.
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8

        # ================= Kd 0.2 -> 0.05: the legs were overdamped =================
        # Measured effective joint inertia at the stance pose, and the closed-loop
        # PD it produces with Kp=3.0 / Kd=0.2:
        #
        #     joint        J_eff (kg m2)    f_n (Hz)    zeta
        #     hip_roll        0.001334        7.55      1.581
        #     hip_pitch       0.001511        7.09      1.485
        #     knee            0.000306       15.75      3.299
        #
        # Go2, for comparison (Kp=25, Kd=0.5, J_eff ~0.03-0.08): zeta = 0.18-0.29.
        # Go2's legs are UNDERDAMPED -- they swing. Chitrak's were overdamped by
        # 5-15x -- they creep. The knee at zeta=3.3 is barely a spring at all.
        #
        # Why this blocks walking specifically: JointPositionAction commands
        # position with desired velocity ZERO, so actuator torque is
        # Kp*(q_des - q) - Kd*qdot. Any joint that is MOVING is being actively
        # braked. At the 2.82 rad/s that a 0.25 m/s swing needs:
        #     Chitrak  0.2 * 2.82 = 0.564 Nm = 22.6% of its 2.5 Nm budget
        #     Go2      0.5 * 2.82 = 1.41  Nm =  6.0% of its 23.5 Nm budget
        # i.e. Chitrak spent 3.8x more of its torque budget fighting its own damper.
        # Standing costs nothing (qdot=0, the damping term vanishes) while swinging
        # a leg costs a quarter of the available torque before any useful work --
        # so the ACTUATOR, not just the reward, made standing cheaper than walking.
        #
        # Root cause: Kp=3.0/Kd=0.2 were borrowed wholesale from Solo12 (2.3-2.5 kg,
        # far heavier legs). zeta scales as Kd/sqrt(Kp*J), and Chitrak's leg inertia
        # is ~20x smaller, so the same numbers overdamp badly. Matching Go2's
        # zeta~0.25 needs Kd~0.034 or Kp~106 -- the RATIO is what is wrong.
        # 0.05 puts zeta at ~0.37, between Go2's 0.25 and where we were.
        self.scene.robot.actuators["base_legs"].damping = 0.05


@configclass
class ChitrakWalkEnvCfg_PLAY(ChitrakWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # one steady forward command so the replay video is readable instead of
        # 50 robots each chasing a different random velocity
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (LIN_VEL_X_MAX, LIN_VEL_X_MAX)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
