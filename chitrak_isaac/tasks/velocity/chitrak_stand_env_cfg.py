"""Chitrak dedicated STANDING task.

Physics-corrected diagnosis (supersedes the old "2.5 Nm can't hold a stand"
conclusion in ABLATIONS.md #1-24, which was a MISDIAGNOSIS):

    Empirically, adding mass to the *feet* let the robot get up and even walk,
    and *increasing* mass helped rather than hurt. If a static torque ceiling
    were the real wall, added mass would make standing strictly harder (more
    weight to hold). It made it easier -> torque saturation is NOT the
    bottleneck. The bottleneck is CONTACT/DYNAMICS INSTABILITY from feet that
    are far too light.

Why light feet break standing (the actual physics):
    Each calf/foot link is ~0.049 kg (torso is 0.531 kg; total ~1.30 kg). With
    a joint PD that can output up to 2.5 Nm, F = m*a means a ~49 g foot driven
    by that torque sees ENORMOUS accelerations -> it overshoots, chatters, and
    bounces off the ground. Contact impulses on a near-massless body cause
    huge instantaneous velocity changes, so the contact sensor flickers and the
    foot never settles into a stable planted stance. The old runs read the
    policy's constant fight against this jitter as "torque saturation" -- it was
    really the symptom of unstable, skating/bouncing light feet.

    Adding foot mass fixed it by (a) adding inertia so the same torque produces
    smaller, smoother foot accelerations (no bounce), and (b) raising the local
    normal force -> more friction budget -> less slip. We do NOT fake foot mass
    here. Instead we address the same physics honestly:
      - REWARD (per user's request): penalize feet sliding, stop rewarding feet
        leaving the ground, and reward smooth control -> the policy learns to
        keep feet planted and move gently, which is what stabilizes light feet.
      - NUMERICS (not fake torque): more solver position iterations + a bit more
        joint mechanical damping -> resolves the light-foot contact without
        inventing torque the real motors don't have. effort_limit stays at the
        REAL 2.5 Nm.

Task framing: all envs are zero-velocity "standing" envs (rel_standing_envs=1.0),
so track_lin/ang_vel rewards now reward staying still, and base_height_exp +
flat_orientation drive it up to the 0.17 m standing height, upright.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .chitrak_flat_env_cfg import ChitrakFlatEnvCfg


@configclass
class ChitrakStandEnvCfg(ChitrakFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- task: pure standing (every env zero-velocity) ----
        self.commands.base_velocity.rel_standing_envs = 1.0
        # keep the command ranges tiny so the (rare) non-standing sampling and
        # the standing-still target both stay near zero -- a stand, not a walk.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # NOTE: the PID-validated init pose (spawn z=0.242081, hip_pitch
        # +-0.722868, knee -+1.586182, hip_roll 0) used to be overridden here.
        # It now lives in the BASE robot cfg (chitrak_isaac/robots/chitrak.py)
        # instead, because the old pose was burying the feet 65.3 mm below the
        # floor at spawn in EVERY task -- flat and rough included, not just this
        # one. Fixing it only here left the walk tasks broken. Inherited now.

        # ---- numerics: stabilize light-foot contact (NOT fake torque) ----
        # effort_limit stays REAL (2.5 Nm) -- torque is not the bottleneck.
        # More solver position iterations resolve the stiff contact of a ~49 g
        # foot without bouncing; a bit more joint damping mechanically absorbs
        # the residual foot oscillation (velocity-proportional, physically
        # honest -- real geartrains have damping/friction too).
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        # damping back to 0.2 (was 0.5): the MuJoCo PID stand that provably
        # works used kv=0.2 with kp=3.0. Since we are now reproducing that exact
        # setup, match it rather than keeping the 0.5 that was guessed to
        # compensate for what turned out to be a spawn-penetration problem.
        self.scene.robot.actuators["base_legs"].damping = 0.2

        # smaller position-target residuals -> smaller commanded foot
        # accelerations -> far less of the F=m*a slamming that light feet can't
        # absorb. (Old stand-only work found 0.125 already helped stability.)
        self.actions.joint_pos.scale = 0.125

        # ---- kill the walking-only curriculum (it would overwrite the weights
        #      we set below with stale walking base_weights each step) ----
        self.curriculum.lin_vel_z_l2_kc = None
        self.curriculum.ang_vel_xy_l2_kc = None
        self.curriculum.dof_torques_l2_kc = None
        self.curriculum.dof_acc_l2_kc = None
        self.curriculum.action_rate_l2_kc = None
        self.curriculum.flat_orientation_l2_kc = None

        # ---- rewards: physics-grounded standing set ----
        # feet_air_time REWARDS lifting feet off the ground (for stepping) --
        # exactly the wrong incentive for a light-footed robot trying to hold a
        # planted stance; it directly encourages the bouncing that destabilizes
        # it. Off for standing.
        self.rewards.feet_air_time.weight = 0.0

        # NEW: penalize feet sliding while in contact -- the single most
        # physics-aligned term for this failure mode. Discourages the skating
        # that light feet are prone to; rewards planted, still feet.
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf_link"),
            },
        )

        # smoother control -> lighter, gentler foot motions (less slamming of
        # low-inertia feet). Bumped 5x from the stock -0.01.
        self.rewards.action_rate_l2.weight = -0.05

        # stay upright. -10.0 (was -5.0): run 1 settled at a persistent ~6 deg
        # pitch, where the term only contributed ~-0.07/step -- far too weak to
        # compete with the smoothness penalties.
        self.rewards.flat_orientation_l2.weight = -10.0

        # reach + hold the standing height -- the core stand objective.
        #
        # target_height 0.17 -> 0.2339. THIS WAS THE MAIN REASON RL FAILED WHERE
        # THE PLAIN PID SUCCEEDS. The two were being asked to do different tasks:
        # the validated PID stand holds the torso at 0.2339 m, while this reward
        # demanded 0.17 m -- a 7 cm deeper crouch, with worse moment arms and much
        # higher static torque demand. Measured in run16/model_150_traj.npz, the
        # robot spawned correctly at 0.2398 and the reward then dragged it DOWN to
        # 0.1645, i.e. the reward was actively destroying the working pose. At
        # std=0.03 the old target paid only 1.1% of max at the PID's real height,
        # so holding the good pose was worth almost nothing.
        #
        # Where 0.2339 comes from (NOT the same as the 0.242081 spawn):
        #   commanded hip-to-foot depth     0.200
        #   actual depth achieved (PD sag)  0.2157
        #   resulting settled torso z       0.2339   <-- this target
        #   kinematic spawn z               0.242081 <-- init_state.pos
        # 0.242081 is the zero-load height where the feet exactly touch; under
        # gravity this PD (kp=3.0) sags ~8 mm to 0.2339. Targeting the spawn
        # height instead would pay 92.8% and ask the policy to fight gravity
        # forever for reward it cannot collect; targeting the settled height puts
        # the peak exactly on the reachable equilibrium, which makes the
        # zero-action policy -- i.e. the PID itself -- reward-maximizing.
        # Do NOT "simplify" this to 0.20: that is the hip-to-foot DEPTH, not a
        # torso height. The torso sits 0.0421 m above it (verified by mesh FK).
        self.rewards.base_height_exp.params["target_height"] = 0.2339
        #
        # std 0.15 -> 0.03 (run-1 fix). The inherited std=0.15 made the exp
        # kernel nearly FLAT near the target: at a 14 mm miss it still paid
        # 99.1% of max, so closing that last 14 mm was worth only +0.07 reward
        # -- less than the dof_acc/action_rate/feet_slide savings the policy got
        # by crouching lower. Run 1 therefore *drifted down* over training
        # (0.1698 m @iter150 -> 0.1560 m @iter499) while base_height_exp still
        # read 98.8% of ceiling, i.e. the reward could not see the error.
        # At std=0.03 that same 14 mm is worth +1.57 -- a real gradient.
        # Safe to tighten: the robot now spawns at 0.242081 m, only ~8 mm above
        # the 0.2339 target, so no wide "get up off the floor" basin is needed.
        self.rewards.base_height_exp.weight = 8.0
        self.rewards.base_height_exp.params["std"] = 0.03

        # anti-splay: keep hip_roll joints near neutral so the stance stays
        # symmetric instead of leaning on one splayed leg.
        self.rewards.hip_roll_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll_joint")},
        )


@configclass
class ChitrakStandEnvCfg_PLAY(ChitrakStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
